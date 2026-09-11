"""bench_peter_protocol.py: MambaLiteUNet timed the way Peter's hybridTRTinf_plot.py times
MCBNet (per slice: normalise, H2D, resize to model size, model, resize back, D2H, threshold,
between two CUDA syncs), reported in his exact text layout. The pieces are testable on the
Mac with a fake model on CPU."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploy.bench_peter_protocol import (  # noqa: E402
    case_of, format_report, group_by_case, infer_slice, metric_2d, normalize_to_u8,
    speed_summary)


def test_case_of_strips_frame_and_channel_suffix():
    assert case_of("/x/pat08_sq2_0003_0000.tif") == "pat08_sq2"
    assert case_of("pat41_0007_0000.tif") == "pat41"
    assert case_of("pat37_roi3sq_0013_0000.tif") == "pat37_roi3sq"


def test_group_by_case_keeps_cases_and_frames_sorted():
    frames = ["d/pat41_0001_0000.tif", "d/pat08_sq2_0001_0000.tif", "d/pat41_0000_0000.tif",
              "d/pat08_sq2_0000_0000.tif"]
    groups = group_by_case(frames)
    assert list(groups) == ["pat08_sq2", "pat41"]
    assert groups["pat41"] == ["d/pat41_0000_0000.tif", "d/pat41_0001_0000.tif"]


def test_normalize_to_u8_is_min_max_per_slice():
    ramp = np.arange(0, 100, dtype=np.uint16).reshape(10, 10) * 3   # 0..297, not 0..255
    out = normalize_to_u8(ramp)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    flat = normalize_to_u8(np.full((4, 4), 7, dtype=np.uint8))
    assert flat.dtype == np.uint8 and not flat.any()


def test_infer_slice_returns_full_size_mask_and_time():
    img = np.zeros((40, 60), dtype=np.uint8)
    img[:, :30] = 255                                          # left half bright
    fake_model = lambda x: x[:, :1]                            # prob = channel 0 in [0, 1]
    prob, pred, ms = infer_slice(fake_model, img, size=(16, 16), threshold=0.5, device="cpu")
    assert prob.shape == (40, 60) and prob.dtype == np.float32
    assert pred.shape == (40, 60) and pred.dtype == np.uint8
    assert pred[:, :28].all() and not pred[:, 32:].any()      # edge blurred by the resizes
    assert ms > 0


def test_infer_slice_raw_mode_feeds_uint8_over_255_not_min_max():
    img = np.full((8, 8), 64, dtype=np.uint8)                  # constant: min-max would give 0
    seen = {}
    def probe(x):
        seen["mean"] = float(x.mean())
        return x[:, :1]
    infer_slice(probe, img, size=(8, 8), threshold=0.5, device="cpu", normalize="raw")
    assert seen["mean"] == pytest.approx(64 / 255, abs=1e-6)
    infer_slice(probe, img, size=(8, 8), threshold=0.5, device="cpu", normalize="minmax")
    assert seen["mean"] == 0.0


def test_metric_2d_follows_peters_conventions():
    gt = np.zeros((30, 40), dtype=np.uint8)
    gt[10:20, 10:30] = 1                                       # 10 x 20 rectangle
    same = metric_2d(gt, gt)
    assert same[0] == 100.0 and same[1] == 100.0 and same[2] == 0.0 and same[3] == 0.0
    shifted = np.roll(gt, 2, axis=1)                           # overlap 10 x 18
    dice, jac, hd95, asd = metric_2d(shifted, gt)
    assert dice == pytest.approx(2 * 180 / 400 * 100)
    assert jac == pytest.approx(180 / 220 * 100)
    assert 0.0 < asd <= 2.0 and 0.0 < hd95 <= 2.0
    empty = np.zeros_like(gt)
    assert metric_2d(empty, empty)[:2] == (100.0, 100.0) and math.isnan(metric_2d(empty, empty)[2])
    assert metric_2d(empty, gt)[:2] == (0.0, 0.0) and math.isnan(metric_2d(empty, gt)[3])


def test_speed_summary_reports_all_and_post_warmup():
    times_ms = [100.0, 100.0, 10.0, 10.0, 10.0, 10.0]
    s = speed_summary(times_ms, warmup=2)
    assert s["n_slices"] == 6 and s["avg_ms_all"] == pytest.approx(40.0)
    assert s["fps_all"] == pytest.approx(25.0)
    assert s["avg_ms_after_warmup"] == pytest.approx(10.0) and s["fps_after_warmup"] == pytest.approx(100.0)


def test_format_report_matches_peters_text_layout():
    per_case = {"pat08_sq2": {"dice": 90.0, "jaccard": 81.818, "hd95": 1.0, "asd": 0.5}}
    speed = {"n_slices": 14, "avg_ms_all": 169.74, "fps_all": 5.89,
             "avg_ms_after_warmup": 160.0, "fps_after_warmup": 6.25}
    text = format_report(per_case, {"dice": 90.0, "jaccard": 81.818, "hd95": 1.0, "asd": 0.5}, speed)
    assert "pat08_sq2 | Dice: 90.00% | Jaccard: 81.82% | HD95: 1.00 | ASD: 0.50" in text
    assert "Total slices: 14" in text
    assert "Average time per slice: 169.74 ms" in text
    assert "FPS: 5.89" in text
    assert "excluding warm-up" in text and "6.25" in text


def test_metric_2d_without_medpy_or_scipy_still_scores_overlap(monkeypatch):
    import deploy.bench_peter_protocol as bp
    monkeypatch.setattr(bp, "_medpy_metric", None)
    monkeypatch.setattr(bp, "_ndimage", None)
    gt = np.zeros((30, 40), dtype=np.uint8)
    gt[10:20, 10:30] = 1
    dice, jac, hd95, asd = bp.metric_2d(np.roll(gt, 2, axis=1), gt)
    assert dice == pytest.approx(90.0) and jac == pytest.approx(180 / 220 * 100)
    assert math.isnan(hd95) and math.isnan(asd)
