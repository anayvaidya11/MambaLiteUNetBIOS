#!/usr/bin/env python3
"""Predict epithelium masks with a TensorRT engine on a folder of 2D OCT frames.

Same preprocessing, postprocessing and output layout as predict_eso.py, so the output
folder drops straight into the val_eval pipeline (evaluate_val.py).

    python deploy/predict_trt.py --engine deploy/engines/<name>_fp16.engine \
        --input ~/Input515_val --output deploy/results/val_pred_trt_fp16
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.trt_runner import TRTModule  # noqa: E402
from predict_eso import list_frames, original_size, output_name, postprocess, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input", required=True, help="folder of 2D frames (.tif/.png)")
    ap.add_argument("--output", required=True, help="folder for the predicted masks")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    frames = list_frames(args.input)
    if not frames:
        raise SystemExit(f"no frames found in {args.input}")
    engine = TRTModule(args.engine)
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(frames)} frames, engine {args.engine}, threshold {args.threshold}")

    coverage = []
    for i, path in enumerate(frames):
        x = torch.from_numpy(preprocess(path))[None].cuda()
        prob = engine.run(x)[0, 0].float().cpu().numpy()
        mask = postprocess(prob, original_size(path), args.threshold)
        Image.fromarray(mask).save(os.path.join(args.output, output_name(path) + ".tif"))
        coverage.append(mask.mean())
        if (i + 1) % 25 == 0 or i + 1 == len(frames):
            print(f"  {i + 1}/{len(frames)}")
    print(f"done: {len(frames)} masks in {args.output}, mean foreground fraction {np.mean(coverage):.4f}")


if __name__ == "__main__":
    main()
