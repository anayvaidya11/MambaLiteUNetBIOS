import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploy.mamba_export import MambaExport, build_export_model, selective_scan_chunked  # noqa: E402
from deploy.tests.reference_scan import mamba_slow_forward, selective_scan_ref  # noqa: E402

ISIC_CKPT = REPO / "pre_trained_models" / "ISIC2017" / "best-epoch51-loss0.1905.pth"


@pytest.mark.parametrize("l,chunk", [(64, 16), (256, 16), (1024, 16), (4096, 16), (100, 16), (7, 16), (64, 64)])
def test_chunked_scan_matches_reference(l, chunk):
    torch.manual_seed(0)
    b, d, n = 2, 24, 16
    u, delta, z = (torch.randn(b, d, l) for _ in range(3))
    A = -torch.exp(torch.randn(d, n))
    B, C = torch.randn(b, n, l), torch.randn(b, n, l)
    D, bias = torch.randn(d), torch.randn(d)
    ref = selective_scan_ref(u, delta, A, B, C, D, z, bias, delta_softplus=True)
    out = selective_scan_chunked(u, delta, A, B.transpose(1, 2), C.transpose(1, 2), D, z, bias,
                                 delta_softplus=True, chunk=chunk)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.parametrize("d_model,l", [(8, 64), (12, 1024), (24, 256), (32, 64)])
def test_block_matches_upstream_slow_path(d_model, l):
    torch.manual_seed(1)
    m = MambaExport(d_model=d_model, d_state=16, d_conv=4, expand=2)
    with torch.no_grad():  # non-trivial weights
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(2, l, d_model)
    with torch.no_grad():
        out, ref = m(x), mamba_slow_forward(m, x)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not ISIC_CKPT.exists(), reason="ISIC checkpoint not present")
def test_full_model_loads_isic_weights_and_chunking_is_exact():
    torch.manual_seed(2)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        y64 = build_export_model(ISIC_CKPT, chunk=64)(x)
        y_one = build_export_model(ISIC_CKPT, chunk=64)(x)   # different chunking, same result
    assert y64.shape == (1, 1, 256, 256)
    assert 0.0 <= y64.min() and y64.max() <= 1.0
    assert torch.allclose(y64, y_one, atol=1e-4), (y64 - y_one).abs().max()
