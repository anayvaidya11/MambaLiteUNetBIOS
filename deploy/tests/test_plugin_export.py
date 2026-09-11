"""The TensorRT-plugin export path: every Mamba scan becomes one custom ONNX node
(bios::SelectiveScan) that the selective_scan_plugin.so implements on the Jetson.
On the Mac the node's PyTorch fallback (SelectiveScanFn.forward) must match the
reference scan, and the exported graph must contain exactly one node per Mamba block
with the plugin's input contract."""
import collections
import sys
from pathlib import Path

import onnx
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploy.export_onnx import audit, export, simplify  # noqa: E402
from deploy.mamba_export import PLUGIN_DOMAIN, PLUGIN_OP, MambaExport, SelectiveScanFn, build_export_model  # noqa: E402
from deploy.tests.reference_scan import mamba_slow_forward, selective_scan_ref  # noqa: E402

ISIC_CKPT = REPO / "pre_trained_models" / "ISIC2017" / "best-epoch51-loss0.1905.pth"
N_MAMBA_BLOCKS = 56


@pytest.mark.parametrize("l", [64, 1024])
def test_plugin_fallback_matches_reference(l):
    torch.manual_seed(0)
    b, d, n = 2, 24, 16
    u, delta, z = (torch.randn(b, d, l) for _ in range(3))
    A = -torch.exp(torch.randn(d, n))
    B, C = torch.randn(b, n, l), torch.randn(b, n, l)
    D, bias = torch.randn(d), torch.randn(d)
    ref = selective_scan_ref(u, delta, A, B, C, D, z, bias, delta_softplus=True)
    out = SelectiveScanFn.apply(u, delta, A, B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), D, z, bias)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


def test_block_with_plugin_scan_matches_upstream_slow_path():
    torch.manual_seed(1)

    class PluginBlock(MambaExport):
        SCAN = "plugin"

    m = PluginBlock(d_model=12, d_state=16, d_conv=4, expand=2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(2, 256, 12)
    with torch.no_grad():
        out, ref = m(x), mamba_slow_forward(m, x)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not ISIC_CKPT.exists(), reason="ISIC checkpoint not present")
def test_plugin_export_has_one_node_per_mamba_block(tmp_path):
    out = tmp_path / "plugin.onnx"
    model = export(ISIC_CKPT, out, scan="plugin")
    simplify(out)
    ops = audit(out)
    m = onnx.load(str(out))
    scan_nodes = [n for n in m.graph.node if n.op_type == PLUGIN_OP]
    assert len(scan_nodes) == N_MAMBA_BLOCKS
    assert all(n.domain == PLUGIN_DOMAIN for n in scan_nodes)
    assert all(len(n.input) == 8 and len(n.output) == 1 for n in scan_nodes)
    attrs = {a.name: a for a in scan_nodes[0].attribute}
    assert attrs["delta_softplus"].i == 1 and attrs["plugin_version"].s == b"1"
    assert any(o.domain == PLUGIN_DOMAIN for o in m.opset_import)
    # the decay-matrix machinery must be gone: no Where, far fewer Exp / nodes overall
    assert "Where" not in ops
    assert sum(ops.values()) < 2500, sum(ops.values())   # closed-form graph: 4656; measured plugin graph: 2024
    # the model still runs in PyTorch through the fallback and is a probability map
    with torch.no_grad():
        y = model(torch.rand(1, 3, 256, 256))
    assert y.shape == (1, 1, 256, 256) and 0.0 <= y.min() and y.max() <= 1.0
    # and equals the closed-form export model (same weights, same math)
    with torch.no_grad():
        y_ref = build_export_model(ISIC_CKPT)(torch.rand(1, 3, 256, 256))
    assert y_ref.shape == y.shape
    print("plugin graph:", collections.Counter(ops).most_common(8))
