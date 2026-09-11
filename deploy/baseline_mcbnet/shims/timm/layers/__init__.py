import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_  # noqa: F401  (same signature as timm's)


class DropPath(nn.Module):
    """Stochastic depth per sample; identity in eval mode or with drop_prob 0 (timm semantics)."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        if keep > 0.0 and self.scale_by_keep:
            mask.div_(keep)
        return x * mask

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
