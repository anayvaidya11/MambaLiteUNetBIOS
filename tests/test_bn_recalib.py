"""Tests for bn_recalib.recalibrate_bn (CPU-only, toy model)."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bn_recalib import recalibrate_bn


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(4, 1, 1),
        nn.BatchNorm2d(1),
        nn.Sigmoid(),
    )


def make_loader(n_batches=5, batch=4):
    torch.manual_seed(1)
    x = torch.randn(n_batches * batch, 3, 8, 8) * 3 + 2
    y = torch.zeros(n_batches * batch, 1, 8, 8)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def corrupt(bn):
    bn.running_mean.fill_(100.0)
    bn.running_var.fill_(100.0)


def test_running_stats_are_cumulative_average_over_first_n_batches():
    model, loader = make_model(), make_loader(n_batches=5, batch=4)
    bn = model[1]
    corrupt(bn)

    recalibrate_bn(model, loader, n_batches=3)

    batches = [x for x, _ in loader][:3]
    with torch.no_grad():
        acts = [model[0](x) for x in batches]
    expected_mean = torch.stack([a.mean(dim=(0, 2, 3)) for a in acts]).mean(0)
    expected_var = torch.stack([a.var(dim=(0, 2, 3), unbiased=True) for a in acts]).mean(0)
    assert bn.num_batches_tracked.item() == 3
    assert torch.allclose(bn.running_mean, expected_mean, atol=1e-5)
    assert torch.allclose(bn.running_var, expected_var, atol=1e-5)


def test_all_batchnorm_layers_are_recalibrated_not_just_the_first():
    model, loader = make_model(), make_loader()
    corrupt(model[5])

    recalibrate_bn(model, loader, n_batches=2)

    assert model[5].num_batches_tracked.item() == 2
    assert not torch.allclose(model[5].running_mean, torch.full_like(model[5].running_mean, 100.0))


def test_model_is_left_in_eval_mode_with_momentum_restored():
    model, loader = make_model(), make_loader()
    model[1].momentum = 0.3
    model.train()

    recalibrate_bn(model, loader, n_batches=2)

    assert model.training is False
    assert all(m.training is False for m in model.modules())
    assert model[1].momentum == 0.3


def test_only_batchnorm_runs_in_train_mode_during_recalibration():
    model, loader = make_model(), make_loader()
    seen = {}

    def record(key):
        def hook(module, _inputs, _output):
            seen.setdefault(key, module.training)  # a hook must return None
        return hook

    model[3].register_forward_hook(record('dropout'))
    model[1].register_forward_hook(record('bn'))

    recalibrate_bn(model, loader, n_batches=1)

    assert seen == {'dropout': False, 'bn': True}


def test_zero_batches_leaves_stats_untouched_and_sets_eval_mode():
    model, loader = make_model(), make_loader()
    corrupt(model[1])
    model.train()

    recalibrate_bn(model, loader, n_batches=0)

    assert torch.all(model[1].running_mean == 100.0)
    assert model.training is False


def test_no_gradients_are_recorded():
    model, loader = make_model(), make_loader()
    recalibrate_bn(model, loader, n_batches=2)
    assert all(p.grad is None for p in model.parameters())
