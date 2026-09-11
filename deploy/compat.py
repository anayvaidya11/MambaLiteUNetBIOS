"""Stand-ins for the two timm helpers models/MambaLiteUNet.py needs (DropPath,
trunc_normal_), used when timm is not installed (e.g. the Jetson venv) so nothing
has to be pip-installed there. Same semantics as timm.layers."""

import torch
import torch.nn as nn

trunc_normal_ = torch.nn.init.trunc_normal_


class DropPath(nn.Module):
    """Stochastic depth; identity in eval mode."""

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
