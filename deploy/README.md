# TensorRT deployment of MambaLiteUNet (Jetson Orin)

Runs the trained esophagus-OCT MambaLiteUNet as a **single TensorRT engine**, no
mamba_ssm at inference time, and measures FPS and accuracy against the PyTorch model.

## Why a rewrite is needed

`mamba_ssm.Mamba` computes its selective scan and causal conv1d with custom CUDA kernels.
`torch.onnx.export` has no symbolic for them and TensorRT has no state-space-scan layer,
so the network cannot be exported as-is. The lab's earlier MCBNet deployment worked
around this by cutting the network at its two Mamba blocks (TensorRT engines for the CNN
pieces, Mamba in PyTorch or a custom `.cu`). MambaLiteUNet has 56 Mamba blocks inside its
fusion modules and attention gates, so that split does not scale.

`mamba_export.py` keeps the exact parameters of `mamba_ssm.Mamba` (checkpoints load
unchanged) and offers two ways to express the scan (`MambaExport.SCAN`):

* **`plugin` (the deployed path):** each scan is one custom ONNX node `bios::SelectiveScan`
  that `plugin/selective_scan_plugin.cu` implements as a single CUDA kernel doing the real
  recurrence (one warp per channel, fp32 state, same semantics as mamba_ssm). O(L) memory
  traffic, 56 plugin layers inside one engine. Needs the `.so` at build and run time.
* **`closed_form` (first attempt, kept as a stock-ops fallback):** inside every chunk of
  16 tokens the recurrence is a masked matrix of decays `exp(L_t - L_s)`, batched over
  chunks with a second level for the carries. Builds with plain trtexec, but it
  materialises ~1.6 G elements per frame (3.2 GB fp16): 95% of the engine time was those
  tensors, 66 ms / 15 FPS on the Orin Nano (per-layer profile, 2026-09-09). Not fast.

`nn.MultiheadAttention` is exported on its plain path (MatMul/Softmax).

## Files

| File | Purpose |
|---|---|
| `mamba_export.py` | `MambaExport` block, `selective_scan_chunked`, `SelectiveScanFn` (custom op), `build_export_model(ckpt, scan=...)` |
| `plugin/selective_scan_plugin.cu` | TensorRT plugin (IPluginV2DynamicExt) for `bios::SelectiveScan`; `plugin/build_plugin.sh` builds the `.so` on the Jetson |
| `compat.py` | `DropPath` / `trunc_normal_` stand-ins so the model imports without timm |
| `export_onnx.py` | checkpoint -> ONNX (opset 17, onnx-simplifier), `--scan plugin|closed_form`, onnxruntime parity check (closed_form only) |
| `build_engines.sh` | ONNX -> `<name>_fp32.engine` (TF32 off) and `<name>_fp16.engine` via trtexec; passes the plugin `.so` when present |
| `trt_runner.py` | `TRTModule`: torch CUDA tensors in/out (`execute_async_v3`); loads the plugin `.so` before deserializing |
| `compare_outputs.py` | per-frame parity of export model / engines vs PyTorch + mamba_ssm |
| `bench_jetson.py` | FPS: PyTorch fp32 / fp16 vs TensorRT fp32 / fp16, model-only (GPU tensor in/out) and end-to-end incl. disk read |
| `bench_peter_protocol.py` | FPS + Dice/Jaccard/HD95/ASD per case under **Peter's per-slice protocol** (the one his MCBNet hybrid pipeline is timed with), so both models fit on one table; writes provenance with every result |
| `predict_trt.py` | engine -> masks for the val_eval pipeline (same layout as `predict_eso.py`) |
| `tests/` | scan vs upstream reference loop, block wiring, full model, ONNX export |

## Workflow

Mac (or any box with torch + onnx; mamba_ssm not needed):
```
python -m pytest deploy/tests -q
python deploy/export_onnx.py --ckpt checkpoints/best-epoch163-loss0.1703.pth \
    --out deploy/onnx/mambaliteunet_e163_plugin.onnx --scan plugin
```
Jetson (venv `~/Eso-seg/umamba`, TensorRT from JetPack):
```
deploy/plugin/build_plugin.sh                      # once: -> deploy/plugin/libselective_scan_plugin.so
deploy/build_engines.sh deploy/onnx/mambaliteunet_e163_plugin.onnx deploy/engines
python deploy/compare_outputs.py --ckpt checkpoints/best-epoch163-loss0.1703.pth \
    --input data/Input515_val --engines deploy/engines/*.engine
python deploy/bench_jetson.py --ckpt checkpoints/best-epoch163-loss0.1703.pth \
    --engines deploy/engines/*.engine --frames data/Input515_val --out deploy/results
python deploy/predict_trt.py --engine deploy/engines/mambaliteunet_e163_fp16.engine \
    --input data/Input515_val --output deploy/results/val_pred_trt_fp16
```
Then score the prediction folders with `evaluate_val.py` (val_eval pipeline) against the
11-patient ground truth.

## Comparing against MCBNet: Peter's protocol

Peter's 25 FPS figure for MCBNet comes from
`Eso-seg/MCBNet/JetsonTesting/TensorRT_Eso/hybridTRTinf_plot.py` (3 TensorRT engines + 2
mamba_ssm blocks in PyTorch). It times **every slice** between two `torch.cuda.synchronize()`
calls: min-max normalise, /255, H2D, GPU bilinear resize to the model size, model, GPU
bilinear resize back to the frame size, D2H, threshold. Disk reads and metrics are outside
the window; there is no warm-up. `bench_jetson.py`'s "model-only" number excludes the
resizes and copies, its "end-to-end" number includes a disk read and a CPU (PIL) resize, so
neither is comparable to his. `bench_peter_protocol.py` applies his window to MambaLiteUNet:

```
python deploy/bench_peter_protocol.py --ckpt checkpoints/best-epoch163-loss0.1703.pth \
    --engines deploy/engines/mambaliteunet_e163_plugin_fp16.engine deploy/engines/mambaliteunet_e163_plugin_fp32.engine \
    --frames data/Input515_val --gt data/val_gt --out deploy/results
```

`--gt` is a folder of uint8 mask frames named like the predictions (`pat08_sq2_0003.tif`),
exported on the Mac from the val NRRDs with the z-th slice assigned to the z-th frame stem
in sorted order (the mapping `evaluate_val.py` uses; two patients have non-contiguous frame
numbers). Input normalisation defaults to the model's own contract (uint8 / 255, antialiased
bilinear resize, `--normalize raw`); `--normalize minmax` reproduces Peter's per-slice
min-max for op parity but changes the model input. Metrics use medpy when installed
(Peter's helper), otherwise the same definitions in scipy. Run the MCBNet pipeline
back-to-back in the same terminal session and power mode for the comparison row.

## Results and provenance

Result tables land in `deploy/results/` (`bench_*.md`, `peter_protocol_*.md/json`). Every
table carries the script, date, board, power mode (`nvpmodel -q`), torch / TensorRT
versions, git revision and engine path it came from. **No FPS or accuracy number is
reported without that provenance line**: on 2026-09-08 a saved output of Peter's own
MCBNet notebook (June 2026, 5.89 FPS on one case) was mistaken for a MambaLiteUNet result
and sent as such.

Results as of 2026-09-11 (Orin Nano Super, 25 W, TensorRT 10.3, `best-epoch163` checkpoint),
full write-up with per-number provenance in
`~/Anay-BIOS/mambalite-replication/jetson-tensorrt-results-2026-09.md`:

| Under Peter's per-slice protocol, same 152 slices | default clocks | `jetson_clocks` locked |
|---|---|---|
| MCBNet hybrid (Peter's script, 3 engines + mamba_ssm) | 85.5 ms, 11.7 FPS | 42.9 ms, 23.3 FPS |
| MambaLiteUNet plugin engine fp16 | 56.0 ms, 17.8 FPS | 19.7 ms, 50.8 FPS |
| MambaLiteUNet plugin engine fp32 | 63.2 ms, 15.8 FPS | 21.4 ms, 46.8 FPS |
| MambaLiteUNet eager PyTorch fp32 | 227 ms, 4.4 FPS | 197 ms, 5.1 FPS |

trtexec engine-only with CUDA graph: 13.1 ms / 76 qps (plugin fp16). Accuracy on all 152 val
frames (val_eval): fp32 engine identical to the PyTorch checkpoint (mean Dice 0.8463, max
per-frame ΔDice 0.0001); fp16 engine 0.8462 (max ΔDice 0.0016).
