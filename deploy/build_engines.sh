#!/bin/bash
# Build TensorRT engines from the exported ONNX on the Jetson (JetPack 6.2, TensorRT 10.3).
#   deploy/build_engines.sh deploy/onnx/mambaliteunet_e163.onnx [deploy/engines]
# Produces <name>_fp32.engine (TF32 disabled: a true fp32 reference) and <name>_fp16.engine,
# each with a trtexec build+timing log next to it. trtexec's own throughput numbers are a
# cross-check for bench_jetson.py.
# If deploy/plugin/libselective_scan_plugin.so exists (build_plugin.sh) it is passed to
# trtexec, which is required for ONNX files exported with `--scan plugin`; override the
# path with PLUGIN=..., or PLUGIN=none to build without it.
# PRECS="fp16" builds only that precision (default "fp32 fp16"). A trtexec timing cache in
# the output dir makes rebuilds of the same graph much faster than the first build.
set -e
ONNX="$1"
OUT="${2:-deploy/engines}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGIN="${PLUGIN:-$HERE/plugin/libselective_scan_plugin.so}"
[ -f "$ONNX" ] || { echo "onnx not found: $ONNX"; exit 1; }
[ -x "$TRTEXEC" ] || { echo "trtexec not found: $TRTEXEC"; exit 1; }
pflags=""
if [ "$PLUGIN" != none ] && [ -f "$PLUGIN" ]; then
    if "$TRTEXEC" --help 2>&1 | grep -q -- "--staticPlugins"; then pflags="--staticPlugins=$PLUGIN"; else pflags="--plugins=$PLUGIN"; fi
    echo "using plugin: $pflags"
fi
mkdir -p "$OUT"
base=$(basename "$ONNX" .onnx)
cache="--timingCacheFile=$OUT/${base}_timing.cache"
for prec in ${PRECS:-fp32 fp16}; do
    if [ "$prec" = fp16 ]; then flags="--fp16 $pflags $cache"; else flags="--noTF32 $pflags $cache"; fi
    log="$OUT/${base}_${prec}.trtexec.log"
    echo "== building $prec -> $OUT/${base}_${prec}.engine (log: $log)"
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$OUT/${base}_${prec}.engine" $flags \
        --memPoolSize=workspace:1024 --useCudaGraph --separateProfileRun \
        --warmUp=500 --iterations=200 --avgRuns=50 > "$log" 2>&1 \
        || { tail -40 "$log"; exit 1; }
    grep -E "Throughput|Latency: min|GPU Compute Time: min|Engine built|engine size" "$log" | sed 's/^/   /'
done
echo "done"
