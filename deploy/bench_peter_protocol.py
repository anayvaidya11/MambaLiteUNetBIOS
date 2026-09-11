#!/usr/bin/env python3
"""FPS and accuracy of MambaLiteUNet on the Jetson, timed the way Peter's MCBNet hybrid
pipeline is timed (Eso-seg/MCBNet/JetsonTesting/TensorRT_Eso/hybridTRTinf_plot.py), so both
models can sit on one table measured under one protocol.

Peter's timed window per slice (his `infer_slice`, between two torch.cuda.synchronize()):
    normalise -> /255 -> H2D -> GPU bilinear resize to the model size -> model
    -> GPU bilinear resize back to (h, w) -> D2H -> threshold
No disk read and no metric computation inside the window; he times every slice with no
warm-up. This script applies exactly that window to MambaLiteUNet (TensorRT engines and
eager PyTorch), prints his text layout per case and overall, and also reports the average
without the first --warmup slices (first-call engine/cuDNN cost) so either convention can be
quoted. Every result file carries its provenance: board, power mode, versions, engine, git
revision, protocol.

Preprocessing: MambaLiteUNet was trained on uint8 / 255 with an antialiased bilinear resize
to 256x256 (dataprepare/prepare_eso_oct.py), NOT on per-slice min-max normalised inputs the
way MCBNet is fed, so --normalize defaults to "raw" (the model's own input contract).
"minmax" reproduces Peter's normalize_to_u8 for a like-for-like op count but changes what
the model sees. Metrics use medpy when it is installed (bit-identical to Peter's helper),
otherwise the same definitions in scipy: Dice/Jaccard in %, HD95 pooled over both
directions, ASD one-directional (pred -> gt), NaN when either mask is empty.

Ground truth: --gt is a folder of uint8 mask frames named like the prediction files
(`pat08_sq2_0003.tif`); export it on the Mac from the val NRRDs (see deploy/README.md).
Without --gt only timing is reported and masks are still written for scoring elsewhere.

    python deploy/bench_peter_protocol.py --ckpt checkpoints/best-epoch163-loss0.1703.pth \
        --engines deploy/engines/mambaliteunet_e163_plugin_fp16.engine \
        --frames data/Input515_val --gt data/val_gt --out deploy/results
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from predict_eso import list_frames, output_name  # noqa: E402

_CASE_RE = re.compile(r"^(.*)_(\d+)$")


def case_of(path):
    """`pat08_sq2_0003_0000.tif` -> `pat08_sq2` (drop the nnU-Net channel and frame suffixes)."""
    stem = output_name(path)
    m = _CASE_RE.match(stem)
    return m.group(1) if m else stem


def group_by_case(frames):
    """{case: [frame paths sorted]} with cases in sorted order."""
    groups = {}
    for f in frames:
        groups.setdefault(case_of(f), []).append(f)
    return OrderedDict((c, sorted(groups[c])) for c in sorted(groups))


def normalize_to_u8(img):
    """Peter's per-slice min-max normalisation (hybridTRTinf_helper.normalize_to_u8)."""
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def load_frame(path):
    """Frame as a 2-D uint8 array (same conversion as the training loader for RGB / 16-bit)."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    arr = arr.astype(np.float64)
    if arr.max() > 255:
        arr = arr / arr.max() * 255.0
    return arr.astype(np.uint8)


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_slice(fn, img, size=(256, 256), threshold=0.5, device="cuda", normalize="raw"):
    """One slice through Peter's timed window. fn: (1,3,H,W) float tensor -> (1,1,h,w) prob.
    Returns (prob at frame size float32, uint8 mask, milliseconds)."""
    h, w = img.shape
    _sync(device)
    t0 = time.perf_counter()
    if normalize == "minmax":
        img = normalize_to_u8(img)
    x = torch.from_numpy(np.ascontiguousarray(img)).to(device).float().div_(255.0)[None, None]
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    x = x.expand(-1, 3, -1, -1).contiguous()
    prob = fn(x)
    prob = F.interpolate(prob.float(), size=(h, w), mode="bilinear", align_corners=False)
    prob_np = prob[0, 0].cpu().numpy().astype(np.float32, copy=False)
    pred = (prob_np >= threshold).astype(np.uint8)
    _sync(device)
    ms = (time.perf_counter() - t0) * 1000.0
    return prob_np, pred, ms


# ---------------------------------------------------------------- metrics (Peter's helper)

try:
    from medpy import metric as _medpy_metric
except ImportError:
    _medpy_metric = None
try:
    from scipy import ndimage as _ndimage       # fallback with medpy's definitions
except ImportError:
    _ndimage = None                             # overlap metrics only, distances NaN


def _boundary(mask):
    fp = _ndimage.generate_binary_structure(mask.ndim, 1)
    return mask ^ _ndimage.binary_erosion(mask, structure=fp, iterations=1)


def _surface_distances(a, b):
    dt = _ndimage.distance_transform_edt(~_boundary(b))
    return dt[_boundary(a)]


def metric_2d(pred, gt):
    """(dice %, jaccard %, hd95 px, asd px) with hybridTRTinf_helper.metric_2d's conventions:
    both empty -> 100/100/nan/nan; exactly one empty -> 0/0/nan/nan."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.any() and gt.any():
        if _medpy_metric is not None:
            return (float(_medpy_metric.binary.dc(pred, gt) * 100), float(_medpy_metric.binary.jc(pred, gt) * 100),
                    float(_medpy_metric.binary.hd95(pred, gt)), float(_medpy_metric.binary.asd(pred, gt)))
        inter = np.count_nonzero(pred & gt)
        ps, gs = np.count_nonzero(pred), np.count_nonzero(gt)
        dice, jac = float(2.0 * inter / (ps + gs) * 100), float(inter / (ps + gs - inter) * 100)
        if _ndimage is None:
            return dice, jac, math.nan, math.nan
        d_pg, d_gp = _surface_distances(pred, gt), _surface_distances(gt, pred)
        return dice, jac, float(np.percentile(np.hstack((d_pg, d_gp)), 95)), float(d_pg.mean())
    if not pred.any() and not gt.any():
        return 100.0, 100.0, math.nan, math.nan
    return 0.0, 0.0, math.nan, math.nan


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return math.nan if x.size == 0 or np.all(np.isnan(x)) else float(np.nanmean(x))


def speed_summary(times_ms, warmup=0):
    n = len(times_ms)
    avg_all = float(np.mean(times_ms)) if n else math.nan
    after = times_ms[warmup:] if n > warmup else times_ms
    avg_after = float(np.mean(after)) if after else math.nan
    return {"n_slices": n, "warmup": warmup,
            "avg_ms_all": avg_all, "fps_all": 1000.0 / avg_all if avg_all else math.nan,
            "avg_ms_after_warmup": avg_after,
            "fps_after_warmup": 1000.0 / avg_after if avg_after else math.nan}


def format_report(per_case, overall, speed):
    """Peter's stdout layout (per-case line, final metrics block, speed block), plus the
    post-warm-up numbers."""
    lines = []
    for case, m in per_case.items():
        lines.append(f"{case} | Dice: {m['dice']:.2f}% | Jaccard: {m['jaccard']:.2f}% | "
                     f"HD95: {m['hd95']:.2f} | ASD: {m['asd']:.2f}")
    if overall:
        lines += ["", "Final 2D Slice-Level Metrics", "-" * 40,
                  f"Dice:    {overall['dice']:.2f}%", f"Jaccard: {overall['jaccard']:.2f}%",
                  f"HD95:    {overall['hd95']:.2f} pixels", f"ASD:     {overall['asd']:.2f} pixels"]
    lines += ["", "Speed", "-" * 40,
              f"Total slices: {speed['n_slices']}",
              f"Average time per slice: {speed['avg_ms_all']:.2f} ms",
              f"FPS: {speed['fps_all']:.2f}",
              f"Average time per slice (excluding warm-up, first {speed.get('warmup', 0)}): "
              f"{speed['avg_ms_after_warmup']:.2f} ms",
              f"FPS (excluding warm-up): {speed['fps_after_warmup']:.2f}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- provenance / driver

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def gpu_mhz():
    """Current GPU clock on a Jetson (devfreq), or 'n/a' elsewhere. Orin's governor idles
    the GPU at 306 MHz and ramps under load, so a loop with CPU work between slices can run
    the engine far below its tight-loop speed; `sudo jetson_clocks` pins the clocks."""
    for path in ("/sys/class/devfreq/17000000.gpu/cur_freq", "/sys/class/devfreq/17000000.ga10b/cur_freq"):
        try:
            return int(open(path).read().strip()) // 1_000_000
        except (OSError, ValueError):
            continue
    return "n/a"


def provenance(args):
    info = {
        "script": "deploy/bench_peter_protocol.py",
        "protocol": "per-slice, between two CUDA syncs: normalise, /255, H2D, GPU bilinear resize to "
                    f"{args.size}x{args.size} (antialias), model, GPU bilinear resize back, D2H, threshold "
                    f"{args.threshold}; no disk read; mirrors hybridTRTinf_plot.py infer_slice",
        "normalize": args.normalize,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": sh("hostname"),
        "board": sh("cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'"),
        "power_mode": sh("nvpmodel -q 2>/dev/null | head -1"),
        "gpu_mhz_at_start": gpu_mhz(),
        "jetson_clocks_note": "GPU MHz is sampled at the start of the run and before/after every variant; "
                              "at max (918 for the Orin Nano Super in 25W) clocks were locked or busy",
        "jetson_release": sh("head -1 /etc/nv_tegra_release 2>/dev/null"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "git": sh(f"cd {REPO} && git rev-parse --short HEAD 2>/dev/null")
               + (" (dirty)" if sh(f"cd {REPO} && git status --porcelain 2>/dev/null") else ""),
        "metrics_impl": ("medpy" if _medpy_metric is not None else
                         "scipy (medpy definitions)" if _ndimage is not None else
                         "overlap only (no medpy/scipy: HD95/ASD NaN)"),
    }
    try:
        from deploy.trt_runner import trt
        info["tensorrt"] = trt.__version__
    except Exception:  # noqa: BLE001
        info["tensorrt"] = "n/a"
    return info


def run_variant(name, fn, groups, gt_dir, mask_dir, args, device="cuda"):
    print(f"\n== {name}  (GPU {gpu_mhz()} MHz at start)")
    per_case, times, all_m = OrderedDict(), [], {"dice": [], "jaccard": [], "hd95": [], "asd": []}
    mhz_start = gpu_mhz()
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)
    for case, paths in groups.items():
        cm = {"dice": [], "jaccard": [], "hd95": [], "asd": []}
        for p in paths:
            img = load_frame(p)
            _, pred, ms = infer_slice(fn, img, (args.size, args.size), args.threshold, device, args.normalize)
            times.append(ms)
            if mask_dir:
                Image.fromarray(pred).save(os.path.join(mask_dir, output_name(p) + ".tif"))
            if gt_dir:
                gt = np.asarray(Image.open(os.path.join(gt_dir, output_name(p) + ".tif"))) > 0
                for k, v in zip(("dice", "jaccard", "hd95", "asd"), metric_2d(pred, gt)):
                    cm[k].append(v)
                    all_m[k].append(v)
        if gt_dir:
            per_case[case] = {k: nanmean(v) for k, v in cm.items()}
            print(f"{case} | Dice: {per_case[case]['dice']:.2f}% | Jaccard: {per_case[case]['jaccard']:.2f}% | "
                  f"HD95: {per_case[case]['hd95']:.2f} | ASD: {per_case[case]['asd']:.2f}  "
                  f"({len(paths)} slices, {np.mean(times[-len(paths):]):.2f} ms/slice)")
        else:
            print(f"{case}: {len(paths)} slices, {np.mean(times[-len(paths):]):.2f} ms/slice")
    overall = {k: nanmean(v) for k, v in all_m.items()} if gt_dir else None
    speed = speed_summary(times, args.warmup)
    mhz_end = gpu_mhz()
    print(f"   GPU {mhz_start} MHz at start, {mhz_end} MHz at end of variant")
    return {"per_case": per_case, "overall": overall, "speed": speed, "times_ms": times,
            "gpu_mhz_start": mhz_start, "gpu_mhz_end": mhz_end}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="PyTorch checkpoint (adds the eager fp32 mamba_ssm variant)")
    ap.add_argument("--engines", nargs="*", default=[], help="TensorRT engines (deploy/trt_runner.TRTModule)")
    ap.add_argument("--frames", required=True, help="folder of 2-D frames, e.g. data/Input515_val")
    ap.add_argument("--gt", default=None, help="folder of ground-truth mask frames named like the predictions")
    ap.add_argument("--out", default="deploy/results")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5, help="slices excluded from the second average")
    ap.add_argument("--normalize", choices=["raw", "minmax"], default="raw")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--no-masks", action="store_true", help="do not write mask folders")
    args = ap.parse_args()

    frames = list_frames(args.frames)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    groups = group_by_case(frames)
    if args.limit_cases:
        groups = OrderedDict(list(groups.items())[:args.limit_cases])
    n = sum(len(v) for v in groups.values())
    print(f"{len(groups)} cases, {n} slices from {args.frames}; gt: {args.gt or 'none'}")

    variants = OrderedDict()
    for path in args.engines:
        from deploy.trt_runner import TRTModule
        eng = TRTModule(path)
        variants[f"TensorRT {Path(path).stem}"] = (lambda x, e=eng: e.run(x), path)
    if args.ckpt:
        from predict_eso import load_model
        model, _ = load_model(args.ckpt)
        variants["PyTorch fp32 eager (mamba_ssm kernels)"] = (lambda x, m=model: m(x), args.ckpt)
    if not variants:
        raise SystemExit("give --engines and/or --ckpt")

    info = provenance(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = OrderedDict()
    with torch.no_grad():
        for name, (fn, src) in variants.items():
            tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
            mask_dir = None if args.no_masks else str(out / f"val_pred_protocol_{tag}")
            r = run_variant(name, fn, groups, args.gt, mask_dir, args)
            r["source"] = src
            r["mask_dir"] = mask_dir
            results[name] = r
            print(format_report(r["per_case"], r["overall"], r["speed"]))

    (out / f"peter_protocol_{stamp}.json").write_text(json.dumps({"info": info, "results": results}, indent=2))
    lines = [f"# MambaLiteUNet under Peter's per-slice protocol ({info['date']})", "",
             f"{info['board']} | {info['power_mode']} | GPU {info['gpu_mhz_at_start']} MHz at start | "
             f"{info['jetson_release']} | torch {info['torch']} | TensorRT {info['tensorrt']} | git {info['git']} | "
             f"metrics: {info['metrics_impl']}", "",
             f"Protocol: {info['protocol']}. Input normalisation: {info['normalize']}.", "",
             "| Variant | Slices | Avg ms/slice (all) | FPS (all) | Avg ms (excl. warm-up) | FPS (excl. warm-up) "
             "| Dice % | Jaccard % | HD95 px | ASD px | GPU MHz start/end | Source |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s, o = r["speed"], r["overall"] or {}
        fmt = lambda k: f"{o[k]:.2f}" if o else "-"
        lines.append(f"| {name} | {s['n_slices']} | {s['avg_ms_all']:.2f} | **{s['fps_all']:.2f}** | "
                     f"{s['avg_ms_after_warmup']:.2f} | {s['fps_after_warmup']:.2f} | {fmt('dice')} | "
                     f"{fmt('jaccard')} | {fmt('hd95')} | {fmt('asd')} | {r['gpu_mhz_start']}/{r['gpu_mhz_end']} | "
                     f"`{r['source']}` |")
    (out / f"peter_protocol_{stamp}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / f'peter_protocol_{stamp}.md'}")


if __name__ == "__main__":
    main()
