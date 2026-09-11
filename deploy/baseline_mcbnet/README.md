# MCBNet baseline under Peter's own protocol

Peter's hybrid pipeline (`~/Eso-seg/MCBNet/JetsonTesting/TensorRT_Eso/hybridTRTinf_plot.py`,
three TensorRT engines + two mamba_ssm blocks) is copied here on the Jetson, unchanged
except for four lines: `ROOT_DIR` -> our 11-patient NRRD copy (`data/val_nrrd`),
`DEPLOY_DIR` -> his engines under `JetsonTesting/TensorRT_Eso/deploy_trt`,
`PLOT_ENABLE = False`, and the `TRTModule` import -> the local copy of his `trtrunner.py`.
Run it back-to-back with `deploy/bench_peter_protocol.py` in the same terminal, power mode
and clock state; that gives MCBNet and MambaLiteUNet on the same 152 slices under the same
timing window (his: per slice between two CUDA syncs, no disk read, no warm-up).

`shims/` holds stand-ins for the packages his scripts import that the umamba venv lacks
(he evidently ran them in another environment): a minimal NRRD reader named `SimpleITK`,
`medpy.metric.binary` (dc, jc, hd95, asd with medpy's definitions, in scipy) and
`timm.layers.DropPath` / `trunc_normal_` (identity in eval mode). His `trtrunner.py` copy
gets one more edit: fall back to the JetPack TensorRT bindings in
`/usr/lib/python3.10/dist-packages` when `import tensorrt` fails, as our own runner does.
Only file reading, scoring and module construction go through the stand-ins; the timed
window is his code. Tests: `deploy/tests/test_baseline_shims.py`.

The copied scripts, logs and any `pylib/` are git-ignored (Peter's code stays in his repo).
