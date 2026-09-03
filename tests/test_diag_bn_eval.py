"""Tests for the pure metric helper in diag_bn_eval (no CUDA needed)."""
import torch

from diag_bn_eval import pooled_dice


def test_pooled_dice_thresholds_predictions_and_pools_pixels():
    pred = torch.tensor([[[[0.9, 0.2], [0.6, 0.1]]]])
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    # TP=1 (0.9), FP=1 (0.6), FN=1 (0.1 at a gt pixel) -> 2*1 / (2*1 + 1 + 1)
    assert pooled_dice(pred, gt, threshold=0.5) == 0.5


def test_pooled_dice_is_one_for_perfect_prediction():
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert pooled_dice(gt.clone(), gt, threshold=0.5) == 1.0
