#!/usr/bin/env python3
"""FPS benchmark of MambaLiteUNet on the Jetson: PyTorch (mamba_ssm kernels) vs TensorRT.

Model-only latency: 1x3x256x256 input already on the GPU, CUDA-event timed, after warm-up.
End-to-end (with --frames): file read + preprocess + H2D + model + D2H + resize back +
threshold, wall-clock per frame over the given folder.

    python deploy/bench_jetson.py --ckpt best.pth --engines deploy/engines/*.engine \
        --frames ~/Input515_val --out deploy/results
"""

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from predict_eso import list_frames, load_model, original_size, postprocess, preprocess  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def cuda_timed(fn, iters, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
    return times


def summarize(ms):
    mean = statistics.fmean(ms)
    return {"mean_ms": mean, "median_ms": statistics.median(ms), "p95_ms": float(np.percentile(ms, 95)),
            "min_ms": min(ms), "fps": 1000.0 / mean, "n": len(ms)}


def end_to_end(fn, frames, threshold, warmup=5):
    with torch.no_grad():
        for path in frames[:warmup]:
            x = torch.from_numpy(preprocess(path))[None].cuda()
            fn(x)
        torch.cuda.synchronize()
        times = []
        for path in frames:
            t0 = time.perf_counter()
            x = torch.from_numpy(preprocess(path))[None].cuda()
            prob = fn(x)[0, 0].float().cpu().numpy()
            postprocess(prob, original_size(path), threshold)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engines", nargs="*", default=[])
    ap.add_argument("--frames", default=None, help="folder of frames for the end-to-end timing")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="deploy/results")
    ap.add_argument("--with-export-model", action="store_true",
                    help="also time the export-friendly model in eager PyTorch")
    args = ap.parse_args()

    info = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "power_mode": sh("nvpmodel -q 2>/dev/null | head -1"),
        "jetson": sh("cat /etc/nv_tegra_release | head -1"),
        "free_mb_before": sh("free -m | awk '/Mem:/{print $7}'"),
        "iters": args.iters, "warmup": args.warmup,
    }
    try:
        from deploy.trt_runner import trt
        info["tensorrt"] = trt.__version__
    except Exception:  # noqa: BLE001
        info["tensorrt"] = "n/a"

    x = torch.rand(1, 3, 256, 256, device="cuda")
    variants = {}
    model, _ = load_model(args.ckpt)
    variants["PyTorch fp32 (mamba_ssm kernels)"] = lambda inp: model(inp)

    def fp16_forward(inp):
        with torch.autocast("cuda", dtype=torch.float16):
            return model(inp)
    variants["PyTorch fp16 autocast (mamba_ssm kernels)"] = fp16_forward

    if args.with_export_model:
        from deploy.mamba_export import build_export_model
        exp = build_export_model(args.ckpt).cuda()
        variants["export model, eager PyTorch fp32"] = lambda inp, m=exp: m(inp)
    if args.engines:
        from deploy.trt_runner import TRTModule
        for path in args.engines:
            eng = TRTModule(path)
            variants[f"TensorRT {Path(path).stem}"] = lambda inp, e=eng: e.run(inp)

    frames = list_frames(args.frames) if args.frames else []
    results = {}
    for name, fn in variants.items():
        print(f"== {name}")
        r = {"model_only": summarize(cuda_timed(lambda: fn(x), args.iters, args.warmup))}
        print(f"   model-only: {r['model_only']['mean_ms']:.2f} ms mean, "
              f"{r['model_only']['median_ms']:.2f} median, {r['model_only']['fps']:.1f} FPS")
        if frames:
            r["end_to_end"] = summarize(end_to_end(fn, frames, args.threshold))
            print(f"   end-to-end: {r['end_to_end']['mean_ms']:.2f} ms/frame, {r['end_to_end']['fps']:.1f} FPS "
                  f"({len(frames)} frames)")
        r["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        results[name] = r
    info["free_mb_after"] = sh("free -m | awk '/Mem:/{print $7}'")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"bench_{stamp}.json").write_text(json.dumps({"info": info, "results": results}, indent=2))
    lines = [f"# MambaLiteUNet FPS on the Jetson ({info['date']})", "",
             f"GPU: {info['gpu']} | {info['jetson']} | power mode: {info['power_mode']} | "
             f"torch {info['torch']} | TensorRT {info['tensorrt']}", "",
             f"Input 1x3x256x256, batch 1, {args.warmup} warm-up + {args.iters} timed iterations "
             f"(CUDA events, synchronized). End-to-end = file read + preprocess + H2D + model + D2H + "
             f"resize back + threshold, per frame over {len(frames)} frames.", "",
             "| Variant | Model-only mean (ms) | Median (ms) | p95 (ms) | Model-only FPS | End-to-end ms/frame | End-to-end FPS |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["model_only"]
        e = r.get("end_to_end")
        e_cols = f"{e['mean_ms']:.2f} | {e['fps']:.1f}" if e else "- | -"
        lines.append(f"| {name} | {m['mean_ms']:.2f} | {m['median_ms']:.2f} | {m['p95_ms']:.2f} | "
                     f"**{m['fps']:.1f}** | {e_cols} |")
    (out / f"bench_{stamp}.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / f'bench_{stamp}.md'}")


if __name__ == "__main__":
    main()
