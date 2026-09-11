// TensorRT plugin: Mamba selective scan as one CUDA kernel per block.
//
// Implements the custom ONNX op  bios::SelectiveScan  emitted by deploy/mamba_export.py
// (SelectiveScanFn.symbolic). Contract, all float32, linear layout:
//   inputs   0 u       (b, d, l)     conv1d+SiLU output
//            1 delta   (b, d, l)     dt_proj output, BEFORE bias and softplus
//            2 A       (d, n)        = -exp(A_log), folded at export time
//            3 B       (b, l, n)
//            4 C       (b, l, n)
//            5 D       (d)           skip gain
//            6 z       (b, d, l)     gate
//            7 dt_bias (d)
//   output   0 y       (b, d, l)  = (sum_n C_t[n] * h_t[n] + D * u_t) * silu(z_t)
//            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t,  delta_t = softplus(delta + dt_bias)
//   attribute delta_softplus (int, default 1)
//
// Kernel: one thread block per (batch, channel). The block is 16 lanes (one per state n)
// x up to 64 chunks of the sequence. Pass 1 scans every chunk in parallel from a zero
// state and records its total decay and end state; 16 threads then chain the chunk
// carries in shared memory (nchunks tiny sequential steps); pass 2 rescans each chunk from
// its true entering state and emits y, with the sum over n as a 16-lane shuffle
// reduction. So a 4096-token block costs ~2 x 64 dependent steps per thread instead of
// 4096, with ~1000 threads in flight per block to hide memory latency. (The first version,
// one warp per channel walking all L tokens, measured 1.5 us per step: 87 ms per frame.)
// Same semantics as mamba_ssm's selective_scan_fn (fp32 state, softplus threshold 20),
// so the numerics match the PyTorch checkpoint. The plugin only advertises fp32 IO: in an
// --fp16 engine TensorRT inserts the casts around it and the recurrence stays in fp32.
//
// Build (Jetson, JetPack 6.x, TensorRT 10): deploy/plugin/build_plugin.sh
// Use:  trtexec --onnx=... --staticPlugins=deploy/plugin/libselective_scan_plugin.so ...
//       python: deploy/trt_runner.py loads the .so before deserializing the engine.
//
// Written against the IPluginV2DynamicExt interface (deprecated in TensorRT 10 in favour of
// IPluginV3, still fully supported); the deprecation warnings are silenced in the build.

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Named (not anonymous) namespace: nvcc's generated stub cannot disambiguate an anonymous
// namespace here from the one inside nvinfer1 (NvInferRuntime.h) and fails to compile.
namespace ssplugin {

using namespace nvinfer1;

constexpr char const* kPLUGIN_NAME = "SelectiveScan";
constexpr char const* kPLUGIN_VERSION = "1";
constexpr int kLANES = 16;       // threads per chunk = states per channel; d_state <= 16
constexpr int kMAX_CHUNKS = 64;  // 16 x 64 = 1024 threads, the CUDA block maximum
constexpr int kMIN_CHUNK = 16;   // tokens per chunk for short sequences

__device__ __forceinline__ float softplus_f(float x) { return x <= 20.f ? log1pf(expf(x)) : x; }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// Chunk geometry for a sequence length: (tokens per chunk, number of chunks). nchunks is
// even so every warp holds two whole chunks (shuffles never cross a chunk boundary and
// all lanes run the same trip count).
__host__ __device__ inline void chunk_geometry(int seqlen, int& T, int& nchunks)
{
    T = kMIN_CHUNK;
    nchunks = (seqlen + T - 1) / T;
    if (nchunks > kMAX_CHUNKS) {
        nchunks = kMAX_CHUNKS;
        T = (seqlen + nchunks - 1) / nchunks;
    }
    if (nchunks & 1) ++nchunks;  // the extra chunk is entirely past the end: inert
}

__global__ void __launch_bounds__(kLANES * kMAX_CHUNKS)
selective_scan_chunked_kernel(float const* __restrict__ u, float const* __restrict__ delta,
                              float const* __restrict__ A, float const* __restrict__ Bm,
                              float const* __restrict__ Cm, float const* __restrict__ Dskip,
                              float const* __restrict__ z, float const* __restrict__ dt_bias,
                              float* __restrict__ y, int d_inner, int seqlen, int n_state,
                              int do_softplus, int T)
{
    __shared__ float s_dec[kMAX_CHUNKS * kLANES];  // total decay of each chunk
    __shared__ float s_end[kMAX_CHUNKS * kLANES];  // end state of each chunk from a zero start
    __shared__ float s_in[kMAX_CHUNKS * kLANES];   // true state entering each chunk

    int const bd = blockIdx.x;
    int const b = bd / d_inner;
    int const d = bd - b * d_inner;
    int const lane = threadIdx.x & (kLANES - 1);   // state index n
    int const c = threadIdx.x >> 4;                // chunk index
    int const nchunks = blockDim.x >> 4;
    bool const active = lane < n_state;

    float const a = active ? A[d * n_state + lane] : 0.f;
    float const dsk = Dskip[d];
    float const bias = dt_bias[d];

    size_t const row = static_cast<size_t>(bd) * seqlen;
    float const* ub = u + row;
    float const* db = delta + row;
    float const* zb = z + row;
    float* yb = y + row;
    float const* Bb = Bm + static_cast<size_t>(b) * seqlen * n_state;
    float const* Cb = Cm + static_cast<size_t>(b) * seqlen * n_state;
    int const l0 = c * T;

    // pass 1: every chunk from a zero state; record its decay product and end state
    float dec = 1.f, h = 0.f;
#pragma unroll 4
    for (int t = 0; t < T; ++t) {
        int const l = l0 + t;
        bool const valid = l < seqlen;
        float const uv = valid ? ub[l] : 0.f;
        float dv = valid ? db[l] + bias : 0.f;
        if (do_softplus) dv = softplus_f(dv);
        if (!valid) dv = 0.f;                       // decay 1, input 0: padding is inert
        float const bv = (valid && active) ? Bb[static_cast<size_t>(l) * n_state + lane] : 0.f;
        float const da = expf(dv * a);
        h = h * da + dv * bv * uv;
        dec *= da;
    }
    s_dec[c * kLANES + lane] = dec;
    s_end[c * kLANES + lane] = h;
    __syncthreads();

    // carry chain across chunks: 16 threads, one per state, nchunks sequential steps
    if (c == 0) {
        float hin = 0.f;
        for (int cc = 0; cc < nchunks; ++cc) {
            s_in[cc * kLANES + lane] = hin;
            hin = s_dec[cc * kLANES + lane] * hin + s_end[cc * kLANES + lane];
        }
    }
    __syncthreads();

    // pass 2: rescan from the true entering state and emit y
    h = s_in[c * kLANES + lane];
#pragma unroll 4
    for (int t = 0; t < T; ++t) {
        int const l = l0 + t;
        bool const valid = l < seqlen;
        float const uv = valid ? ub[l] : 0.f;
        float dv = valid ? db[l] + bias : 0.f;
        if (do_softplus) dv = softplus_f(dv);
        if (!valid) dv = 0.f;
        size_t const idx = static_cast<size_t>(l) * n_state + lane;
        float const bv = (valid && active) ? Bb[idx] : 0.f;
        float const cv = (valid && active) ? Cb[idx] : 0.f;
        h = h * expf(dv * a) + dv * bv * uv;
        float part = h * cv;
        part += __shfl_xor_sync(0xffffffffu, part, 8, kLANES);
        part += __shfl_xor_sync(0xffffffffu, part, 4, kLANES);
        part += __shfl_xor_sync(0xffffffffu, part, 2, kLANES);
        part += __shfl_xor_sync(0xffffffffu, part, 1, kLANES);
        if (lane == 0 && valid) yb[l] = (part + dsk * uv) * silu_f(zb[l]);
    }
}

class SelectiveScanPlugin final : public IPluginV2DynamicExt {
public:
    explicit SelectiveScanPlugin(int32_t softplus) : mSoftplus(softplus) {}
    SelectiveScanPlugin(void const* data, size_t length)
    {
        if (length >= sizeof(int32_t)) std::memcpy(&mSoftplus, data, sizeof(int32_t));
    }

    // IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* p = new SelectiveScanPlugin(mSoftplus);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t, IExprBuilder&) noexcept override
    {
        return inputs[0];  // y has u's shape (b, d, l)
    }
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t, int32_t) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        Dims const& ud = inputDesc[0].dims;   // (b, d, l)
        Dims const& ad = inputDesc[2].dims;   // (d, n)
        if (ud.nbDims != 3 || ad.nbDims != 2) return 1;
        int const b = ud.d[0], d = ud.d[1], l = ud.d[2], n = ad.d[1];
        if (n > kLANES || b * d <= 0 || l <= 0) return 1;
        int T, nchunks;
        chunk_geometry(l, T, nchunks);
        selective_scan_chunked_kernel<<<b * d, kLANES * nchunks, 0, stream>>>(
            static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
            static_cast<float const*>(inputs[2]), static_cast<float const*>(inputs[3]),
            static_cast<float const*>(inputs[4]), static_cast<float const*>(inputs[5]),
            static_cast<float const*>(inputs[6]), static_cast<float const*>(inputs[7]),
            static_cast<float*>(outputs[0]), d, l, n, mSoftplus, T);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

    // IPluginV2Ext
    DataType getOutputDataType(int32_t, DataType const*, int32_t) const noexcept override { return DataType::kFLOAT; }

    // IPluginV2
    char const* getPluginType() const noexcept override { return kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return sizeof(int32_t); }
    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &mSoftplus, sizeof(int32_t)); }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    int32_t mSoftplus{1};
    std::string mNamespace;
};

class SelectiveScanPluginCreator final : public IPluginCreator {
public:
    SelectiveScanPluginCreator()
    {
        mFields.emplace_back(PluginField{"delta_softplus", nullptr, PluginFieldType::kINT32, 1});
        mFC.nbFields = static_cast<int32_t>(mFields.size());
        mFC.fields = mFields.data();
    }
    char const* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override
    {
        int32_t softplus = 1;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            if (std::strcmp(fc->fields[i].name, "delta_softplus") == 0 && fc->fields[i].data != nullptr)
                softplus = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        auto* p = new SelectiveScanPlugin(softplus);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    IPluginV2* deserializePlugin(char const*, void const* data, size_t length) noexcept override
    {
        auto* p = new SelectiveScanPlugin(data, length);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    PluginFieldCollection mFC{};
    std::vector<PluginField> mFields;
    std::string mNamespace;
};

}  // namespace ssplugin

using ssplugin::SelectiveScanPluginCreator;
REGISTER_TENSORRT_PLUGIN(SelectiveScanPluginCreator);
