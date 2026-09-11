#!/usr/bin/env python3
"""Export MambaLiteUNet to ONNX with every Mamba block replaced by the export-friendly
MambaExport (plain tensor ops, no mamba_ssm kernels), then optionally verify the ONNX
graph against PyTorch with onnxruntime.

Runs anywhere with torch + onnx (mamba_ssm is not needed):
    python deploy/export_onnx.py --ckpt <best-*.pth> --out deploy/onnx/mambaliteunet.onnx --check
    python deploy/export_onnx.py --ckpt <best-*.pth> --out deploy/onnx/mambaliteunet_plugin.onnx --scan plugin
The graph is static: input 'input' (1, 3, 256, 256) float32 in [0, 1], output 'prob'
(1, 1, 256, 256) sigmoid probabilities -- the same contract as predict_eso.py.

--scan closed_form (default): stock ONNX ops only; builds with plain trtexec but is
    memory-bound (T x T decay matrices), 15 FPS on the Orin Nano.
--scan plugin: one bios::SelectiveScan node per Mamba block; needs the TensorRT plugin
    from deploy/plugin/ at build and run time (trtexec --staticPlugins=...). onnxruntime
    cannot run that node, so --check is skipped for this variant; parity is measured on the
    Jetson with deploy/compare_outputs.py instead.
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.mamba_export import PLUGIN_DOMAIN, build_export_model  # noqa: E402


def export(ckpt, out, opset=17, chunk=16, height=256, width=256, scan="closed_form"):
    model = build_export_model(ckpt, chunk=chunk, scan=scan)
    x = torch.rand(1, 3, height, width)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # nn.MultiheadAttention's fused eval-mode fast path has no ONNX symbolic; the plain
    # path traces to MatMul / Softmax, which is what TensorRT wants anyway.
    torch.backends.mha.set_fastpath_enabled(False)
    extra = {"custom_opsets": {PLUGIN_DOMAIN: 1}} if scan == "plugin" else {}
    with torch.no_grad():
        torch.onnx.export(model, (x,), str(out), input_names=["input"], output_names=["prob"],
                          opset_version=opset, dynamo=False, do_constant_folding=True, **extra)
    return model


def simplify(onnx_path):
    """Fold shape arithmetic / constant Ifs with onnx-simplifier (optional dependency)."""
    import onnx
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not installed; skipping simplification")
        return False
    try:
        model, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
    except Exception as e:  # noqa: BLE001  (e.g. onnxruntime cannot load a custom-op graph)
        print(f"onnxsim failed ({type(e).__name__}: {str(e)[:120]}); keeping the unsimplified graph")
        return False
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    onnx.save(model, str(onnx_path))
    return True


def audit(onnx_path):
    """Op-type histogram of the graph (to eyeball TensorRT support before building)."""
    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    return collections.Counter(node.op_type for node in m.graph.node)


def check(model, onnx_path, n_inputs=3, seed=0):
    """Max |onnxruntime - torch| over a few random inputs (fp32, CPU)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n_inputs):
        x = torch.rand(1, 3, 256, 256, generator=g)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        worst = max(worst, float(np.abs(got - ref).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--chunk", type=int, default=16, help="scan chunk length (tokens, closed_form only)")
    ap.add_argument("--scan", choices=["closed_form", "plugin"], default="closed_form",
                    help="closed_form: stock ops; plugin: bios::SelectiveScan nodes for the TensorRT plugin")
    ap.add_argument("--check", action="store_true", help="verify with onnxruntime after export")
    ap.add_argument("--no-simplify", action="store_true", help="skip the onnx-simplifier pass")
    args = ap.parse_args()
    model = export(args.ckpt, args.out, args.opset, args.chunk, scan=args.scan)
    if not args.no_simplify:
        simplify(args.out)
    ops = audit(args.out)
    print(f"exported {args.out}: {sum(ops.values())} nodes")
    for op, cnt in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {op:24s} {cnt}")
    if args.check and args.scan == "plugin":
        print("--check skipped: onnxruntime has no kernel for bios::SelectiveScan; "
              "verify on the Jetson with deploy/compare_outputs.py")
    elif args.check:
        print(f"max |onnxruntime - torch| = {check(model, args.out):.2e}")


if __name__ == "__main__":
    main()
