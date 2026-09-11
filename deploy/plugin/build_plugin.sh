#!/bin/bash
# Build the SelectiveScan TensorRT plugin on the Jetson (JetPack 6.x: CUDA 12.6, TensorRT 10.3).
#   deploy/plugin/build_plugin.sh            -> deploy/plugin/libselective_scan_plugin.so
# Same include/link setup as the lab's MambainCpp Makefile (sm_87 = Orin).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCH="${ARCH:-sm_87}"
OUT="$HERE/libselective_scan_plugin.so"
[ -x "$NVCC" ] || { echo "nvcc not found at $NVCC (set CUDA_HOME or NVCC)"; exit 1; }
"$NVCC" -std=c++17 -O3 -arch="$ARCH" -shared -Xcompiler -fPIC \
    -Xcompiler -Wno-deprecated-declarations -diag-suppress 1215,1216 \
    -I"$CUDA_HOME/include" -I/usr/include/aarch64-linux-gnu \
    -L"$CUDA_HOME/lib64" -L/usr/lib/aarch64-linux-gnu \
    -o "$OUT" "$HERE/selective_scan_plugin.cu" -lnvinfer -lcudart
echo "built $OUT"
nm -D "$OUT" | grep -c "PluginRegistrar" >/dev/null && echo "plugin registrar symbol present" || echo "WARNING: registrar symbol not found"
