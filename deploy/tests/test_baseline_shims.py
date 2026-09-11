"""Stand-ins that let Peter's hybridTRTinf_plot.py run in the umamba venv without SimpleITK
or medpy (neither is installed on the Jetson): a minimal NRRD reader with SimpleITK's
(Z, Y, X) array convention, and medpy.metric.binary's four functions with medpy's
definitions. Only I/O and scoring go through them; the timed window is untouched."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SHIMS = REPO / "deploy" / "baseline_mcbnet" / "shims"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SHIMS))

VAL = Path.home() / "Downloads" / "val" / "pat08_sq2"


@pytest.mark.skipif(not VAL.exists(), reason="val NRRDs not on this machine")
@pytest.mark.parametrize("name", ["pat08_sq2_image.nrrd", "pat08_sq2_mask1.seg.nrrd"])
def test_sitk_shim_reads_nrrd_like_simpleitk(name):
    import nrrd                      # pynrrd, the reference reader
    import SimpleITK as sitk         # the shim
    ref, _ = nrrd.read(str(VAL / name))            # (X, Y, Z) index order
    got = sitk.GetArrayFromImage(sitk.ReadImage(str(VAL / name)))
    assert got.shape == ref.T.shape                # SimpleITK convention: (Z, Y, X)
    assert got.dtype == ref.dtype
    assert np.array_equal(got, ref.T)


def test_medpy_shim_matches_medpy_definitions():
    from medpy import metric
    from deploy.bench_peter_protocol import metric_2d
    gt = np.zeros((30, 40), dtype=np.uint8)
    gt[10:20, 10:30] = 1
    pred = np.roll(gt, 2, axis=1)
    assert metric.binary.dc(pred, gt) == pytest.approx(0.9)
    assert metric.binary.jc(pred, gt) == pytest.approx(180 / 220)
    d, j, hd95, asd = metric_2d(pred, gt)          # same definitions, scipy path
    assert metric.binary.hd95(pred, gt) == pytest.approx(hd95)
    assert metric.binary.asd(pred, gt) == pytest.approx(asd)
    assert metric.binary.dc(np.zeros_like(gt), np.zeros_like(gt)) == 0.0   # medpy: 0 on empty/empty


def _load_shim(name, relpath):
    """Import a stand-in from its file, so the test exercises the shim even when the real
    package is installed on this machine (the Mac venv has timm)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, SHIMS / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_timm_shim_droppath_is_identity_in_eval_and_exports_trunc_normal():
    import torch
    layers = _load_shim("timm_shim_layers", "timm/layers/__init__.py")
    x = torch.randn(2, 3, 4, 4)
    assert torch.equal(layers.DropPath(0.3).eval()(x), x)
    assert torch.equal(layers.DropPath(0.0).train()(x), x)
    assert not torch.equal(layers.DropPath(0.5).train()(torch.ones(64, 2)), torch.ones(64, 2))
    w = torch.empty(256, 256)
    layers.trunc_normal_(w, std=0.02)
    assert abs(w.std().item() - 0.02) < 0.003 and w.abs().max() <= 2.0
    alias = (SHIMS / "timm/models/layers/__init__.py").read_text()
    assert "from timm.layers import DropPath, trunc_normal_" in alias
