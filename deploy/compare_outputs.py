#!/usr/bin/env python3
"""Numerical parity of the deployed variants against the PyTorch model (mamba_ssm kernels).

For every frame: max and mean |prob - prob_ref|, Dice between the thresholded masks, and
the fraction of pixels whose label flips. Variants: the export-friendly model run in
PyTorch (isolates the closed-form scan from TensorRT) and each engine given.

    python deploy/compare_outputs.py --ckpt best.pth --input ~/Input515_val \
        --engines deploy/engines/*_fp32.engine deploy/engines/*_fp16.engine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from predict_eso import list_frames, load_model, preprocess  # noqa: E402


def mask_dice(a, b):
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * inter / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--engines", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (0 = all)")
    ap.add_argument("--no-export-model", action="store_true")
    args = ap.parse_args()

    frames = list_frames(args.input)
    if args.limit:
        frames = frames[:args.limit]
    ref_model, _ = load_model(args.ckpt)

    variants = {}
    if not args.no_export_model:
        from deploy.mamba_export import build_export_model
        exp = build_export_model(args.ckpt).cuda()
        variants["export model (torch fp32)"] = lambda x, m=exp: m(x)
    if args.engines:
        from deploy.trt_runner import TRTModule
        for path in args.engines:
            eng = TRTModule(path)
            variants[Path(path).name] = lambda x, e=eng: e.run(x)

    stats = {k: {"max": [], "mean": [], "dice": [], "flip": []} for k in variants}
    with torch.no_grad():
        for i, path in enumerate(frames):
            x = torch.from_numpy(preprocess(path))[None].cuda()
            ref = ref_model(x)[0, 0].float().cpu().numpy()
            ref_mask = ref >= args.threshold
            for name, fn in variants.items():
                out = fn(x)[0, 0].float().cpu().numpy()
                d = np.abs(out - ref)
                m = out >= args.threshold
                s = stats[name]
                s["max"].append(d.max()); s["mean"].append(d.mean())
                s["dice"].append(mask_dice(m, ref_mask)); s["flip"].append((m != ref_mask).mean())
            if (i + 1) % 25 == 0 or i + 1 == len(frames):
                print(f"  {i + 1}/{len(frames)}")

    print(f"\nReference: PyTorch fp32 with mamba_ssm kernels, {len(frames)} frames, threshold {args.threshold}\n")
    print("| Variant | max abs diff (worst frame) | mean abs diff | mask Dice vs ref (min / mean) | pixels flipped (mean) |")
    print("|---|---|---|---|---|")
    for name, s in stats.items():
        print(f"| {name} | {max(s['max']):.2e} | {np.mean(s['mean']):.2e} "
              f"| {min(s['dice']):.4f} / {np.mean(s['dice']):.4f} | {np.mean(s['flip']) * 100:.3f}% |")


if __name__ == "__main__":
    main()
