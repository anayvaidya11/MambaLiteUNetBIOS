"""Thin TensorRT runtime wrapper: torch CUDA tensors in, torch CUDA tensors out.

Same pattern as the lab's TensorRT_Eso/trtrunner.py (execute_async_v3 on the current
torch stream). Works from the umamba venv on the Jetson: if `tensorrt` is not installed
in the venv, the JetPack system bindings in /usr/lib/python3.10/dist-packages are used.

Output buffers are allocated once and reused, so copy a result out (e.g. .cpu()) before
the next call if you need to keep it.

Engines built from the `--scan plugin` ONNX contain bios::SelectiveScan layers; the plugin
library (deploy/plugin/libselective_scan_plugin.so, built by build_plugin.sh) must be
loaded into the process before such an engine is deserialized. TRTModule does that
automatically when the .so exists next to this file (or pass plugin_libs=[...]).
"""

import ctypes
import sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:  # venv without system site-packages
    sys.path.append("/usr/lib/python3.10/dist-packages")
    import tensorrt as trt

import torch

_DTYPES = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
           trt.int8: torch.int8, trt.bool: torch.bool}

PLUGIN_LIB = Path(__file__).resolve().parent / "plugin" / "libselective_scan_plugin.so"
_LOGGER = trt.Logger(trt.Logger.WARNING)
_loaded_plugins = set()


def load_plugins(paths=None):
    """dlopen plugin libraries (their static initializers register the creators with
    TensorRT's registry) and initialise the built-in plugins. Idempotent."""
    paths = [Path(p) for p in (paths if paths is not None else [PLUGIN_LIB])]
    for path in paths:
        if path.exists() and str(path) not in _loaded_plugins:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            _loaded_plugins.add(str(path))
    trt.init_libnvinfer_plugins(_LOGGER, "")
    return sorted(_loaded_plugins)


class TRTModule:
    def __init__(self, engine_path, verbose=False, plugin_libs=None):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.plugins = load_plugins(plugin_libs)
        with open(engine_path, "rb") as f:
            blob = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to load TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_in else self.outputs).append(name)
        self._held = {}
        self._out = {}

    def dtype(self, name):
        return _DTYPES[self.engine.get_tensor_dtype(name)]

    def __call__(self, inputs):
        """inputs: {name: cuda tensor} -> {name: cuda tensor}"""
        stream = torch.cuda.current_stream().cuda_stream
        for name in self.inputs:
            x = inputs[name]
            if not x.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            x = x.contiguous().to(self.dtype(name))
            self._held[name] = x                       # keep alive until execution
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, x.data_ptr())
        outs = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = self._out.get(name)
            if buf is None or tuple(buf.shape) != shape:
                buf = torch.empty(shape, device="cuda", dtype=self.dtype(name))
                self._out[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            outs[name] = buf
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")
        return outs

    def run(self, x):
        """Single-input, single-output convenience: x (cuda) -> output tensor (cuda)."""
        return self({self.inputs[0]: x})[self.outputs[0]]
