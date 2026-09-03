"""Re-estimate BatchNorm running statistics from training data ("precise BN").

MambaLiteUNet's only train/eval-dependent layers are the BatchNorm2d layers inside the
cross-gated attention gates (the encoder/decoder use GroupNorm, the Mamba blocks
LayerNorm, and there is no Dropout). In eval mode those layers use running averages
that lag behind the weights whenever the learning rate is high, so validation loss
(eval mode) spikes while training loss (batch statistics) looks fine. Refreshing the
running statistics with a few gradient-free forward passes over training batches right
before validation removes that artefact and makes the saved checkpoints self-contained.
"""
import torch
from torch.nn.modules.batchnorm import _BatchNorm


def recalibrate_bn(model, loader, n_batches):
    """Reset every BatchNorm layer's running stats and re-estimate them as the exact
    (cumulative) average over the first `n_batches` batches of `loader`.

    Only the BatchNorm layers run in train mode; every other layer stays in eval mode.
    No gradients are recorded, BatchNorm momenta are restored afterwards, and the model
    is left in eval mode. `n_batches <= 0` only switches the model to eval mode.
    """
    model.eval()
    bn_layers = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if n_batches <= 0 or not bn_layers:
        return
    device = next(model.parameters()).device
    momenta = {}
    for m in bn_layers:
        m.reset_running_stats()
        momenta[m] = m.momentum
        m.momentum = None  # cumulative moving average
        m.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(images.to(device, non_blocking=True).float())
    for m in bn_layers:
        m.momentum = momenta[m]
        m.eval()
