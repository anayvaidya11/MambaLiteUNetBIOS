import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploy.export_onnx import audit, check, export, simplify  # noqa: E402

ISIC_CKPT = REPO / "pre_trained_models" / "ISIC2017" / "best-epoch51-loss0.1905.pth"
# ops TensorRT has no native layer for; none may appear in the exported graph
FORBIDDEN = {"Loop", "Scan", "If", "Einsum", "CumSum", "NonZero", "TopK", "ScatterND"}


@pytest.mark.skipif(not ISIC_CKPT.exists(), reason="ISIC checkpoint not present")
def test_onnx_export_matches_torch(tmp_path):
    out = tmp_path / "m.onnx"
    model = export(ISIC_CKPT, out)
    simplify(out)
    ops = audit(out)
    assert "If" not in ops and "Shape" not in ops, "graph is not static"
    assert not (set(ops) & FORBIDDEN), set(ops) & FORBIDDEN
    assert check(model, out, n_inputs=2) < 1e-4
