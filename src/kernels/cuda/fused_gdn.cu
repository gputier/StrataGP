// src/kernels/cuda/fused_gdn.cu - see include/strata/kernels/fused_gdn.hpp.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/launch_grid.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per thread

std::atomic<bool> g_state_bf16{false};

// St = float (the default) or uint16_t (the BF16 state of issue #53): only the state's load and store differ.
template <typename St>
__global__ void __launch_bounds__(S * RG) gdn_step_norm_kernel(St* __restrict__ state, const float* __restrict__ q,
                                                               const float* __restrict__ k, const float* __restrict__ v,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               const float* __restrict__ z,
                                                               const float* __restrict__ gamma, float eps,
                                                               float* __restrict__ y, int h_k, int h_v) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;          // 0..127
    const int rg = threadIdx.y;           // 0..3
    const int tid = rg * S + col;
    const int qh = head % h_k;
    if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
    float s[RPG];
    St* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = gdn_state_load(base + r * row_stride);
    __syncthreads();
    const float g = __expf(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
    red[rg][col] = kv;
    __syncthreads();
    const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
        o = fmaf(s[r], sq[rg * RPG + r], o);
        gdn_state_store(base + r * row_stride, s[r]);
    }
    __syncthreads();                      // every thread has read red[] for kv_col
    red[rg][col] = o;
    __syncthreads();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127.
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    __syncthreads();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[head * S + col];
        y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
    }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ qkv,
                                                        const float* __restrict__ w, float* __restrict__ h,
                                                        int qk_heads, float eps) {
    __shared__ float part[S / 32];
    const int c = blockIdx.x * S + threadIdx.x;
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[c] = y;
}

__global__ void __launch_bounds__(256) gdn_ab_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                     const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                                     const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                     float* __restrict__ beta, int n, int h_v) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float4 xa = *reinterpret_cast<const float4*>(x + j * 8);
        const float4 xb = *reinterpret_cast<const float4*>(x + j * 8 + 4);
        acc = fmaf(__uint_as_float(wv.x << 16), xa.x, acc); acc = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(wv.y << 16), xa.z, acc); acc = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(wv.z << 16), xb.x, acc); acc = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(wv.w << 16), xb.z, acc); acc = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + __expf(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : log1pf(__expf(v));
        gate[r] = sp * ssm_a[r];
    }
}

// Issue #26 (O7): the same rows, one per block of four warps.  The kernel above puts its 96 rows on 12 blocks (12
// SMs of a 170-SM RTX 5090) and each lane walks its 10 chunks of 8 with a DRAM round trip every few of them.  Here
// all 128 threads load the row and x into shared memory at once (4 chunks each at most, every load in flight
// together), then warp 0 runs the chain above unchanged - same lane per chunk, same chunk order, same fmaf
// sequence, same butterfly, same epilogue - so gate and beta are bitwise the warp-per-row kernel's.
constexpr int AB_THREADS = 128;
constexpr int AB_LOADS = 4;                       // chunks of 8 per thread: n_embd <= 4 * 128 * 8 = 4096

__global__ void __launch_bounds__(AB_THREADS) gdn_ab_row_kernel(const float* __restrict__ x,
                                                                const uint16_t* __restrict__ wa,
                                                                const uint16_t* __restrict__ wb,
                                                                const float* __restrict__ dt,
                                                                const float* __restrict__ ssm_a,
                                                                float* __restrict__ gate, float* __restrict__ beta,
                                                                int n, int h_v) {
    extern __shared__ uint4 ab_smem[];            // [n/8 weight chunks][n/8 x lows][n/8 x highs], 48 B per chunk
    const int n8 = n / 8;
    uint4* ws = ab_smem;
    float4* xls = reinterpret_cast<float4*>(ab_smem + n8);
    float4* xhs = xls + n8;
    const int row = blockIdx.x;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    uint4 wv[AB_LOADS];
    float4 xl[AB_LOADS], xh[AB_LOADS];
#pragma unroll
    for (int u = 0; u < AB_LOADS; ++u) {
        const int j = (int) threadIdx.x + u * AB_THREADS;
        if (j < n8) { wv[u] = __ldg(w4 + j); xl[u] = __ldg(x4 + 2 * j); xh[u] = __ldg(x4 + 2 * j + 1); }
    }
#pragma unroll
    for (int u = 0; u < AB_LOADS; ++u) {
        const int j = (int) threadIdx.x + u * AB_THREADS;
        if (j < n8) { ws[j] = wv[u]; xls[j] = xl[u]; xhs[j] = xh[u]; }
    }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    float acc = 0.0f;
    for (int j = lane; j < n8; j += 32) {
        const uint4 wv = ws[j];
        const float4 xa = xls[j];
        const float4 xb = xhs[j];
        acc = fmaf(__uint_as_float(wv.x << 16), xa.x, acc); acc = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(wv.y << 16), xa.z, acc); acc = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(wv.z << 16), xb.x, acc); acc = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(wv.w << 16), xb.z, acc); acc = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + __expf(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : log1pf(__expf(v));
        gate[r] = sp * ssm_a[r];
    }
}

// Issue #26 (O7): the step above with a head's 128 columns split over a cluster of GC blocks (32 columns x 4 row
// groups each), so the 6.3 MB of state traffic runs on 4x the SMs.  A column's arithmetic never leaves its thread
// and is the kernel's above line for line (its 32 rows per row group, the four row-group partials summed in the
// same order).  The RMS norm spans the head: warp 0 of each block holds 32 consecutive columns with the same lane
// per column as warp `part` of the kernel above, so its butterfly is the same, and its sum is pushed into every
// block of the cluster (distributed shared memory) - `ss` adds the same four warp sums in the same order.  The
// result is bitwise the one-block-per-head kernel's.  sm_90 and newer; the launcher checks before using it.
constexpr int GC = 4;                             // blocks per head
constexpr int GCW = S / GC;                       // 32 columns per block

// St as in gdn_step_norm_kernel (#53): only the state's load and store differ.
template <typename St>
__global__ void __launch_bounds__(GCW * RG) gdn_step_norm_cluster_kernel(St* __restrict__ state,
                                                                         const float* __restrict__ q,
                                                                         const float* __restrict__ k,
                                                                         const float* __restrict__ v,
                                                                         const float* __restrict__ gate,
                                                                         const float* __restrict__ beta,
                                                                         const float* __restrict__ z,
                                                                         const float* __restrict__ gamma, float eps,
                                                                         float* __restrict__ y, int h_k, int h_v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][GCW];
    __shared__ float wsum[S / 32];                // the head's four warp sums, one pushed by each block
    __cluster_barrier_arrive_relaxed();           // waited below, before the first store into another block
    const int part = (int) __clusterRelativeBlockRank();
    const int head = blockIdx.x / GC;
    const int c = threadIdx.x;                    // 0..31
    const int col = part * GCW + c;
    const int rg = threadIdx.y;                   // 0..3
    const int tid = rg * GCW + c;                 // 0..127: loads sk/sq like the first 128 threads above
    const int qh = head % h_k;
    if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
    float s[RPG];
    St* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = gdn_state_load(base + r * row_stride);
    __syncthreads();
    const float g = __expf(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
    red[rg][c] = kv;
    __syncthreads();
    const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
        o = fmaf(s[r], sq[rg * RPG + r], o);
        gdn_state_store(base + r * row_stride, s[r]);
    }
    __syncthreads();                              // every thread has read red[] for kv_col
    red[rg][c] = o;
    __syncthreads();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
        sq_part = oc * oc;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    __cluster_barrier_wait();                     // every block of the cluster is running: DSMEM is live
    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < GC; ++b) static_cast<float*>(__cluster_map_shared_rank(wsum, (unsigned) b))[part] = sq_part;
    }
    __cluster_barrier_arrive();                   // release: the pushes above ...
    __cluster_barrier_wait();                     // ... are visible here, and no block reads another after this
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[head * S + col];
        y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
    }
#endif
}

template <typename St>
bool step_cluster_ready() {
    static const bool ready = cluster_launchable(gdn_step_norm_cluster_kernel<St>, dim3(GCW, RG), GC);
    return ready;
}
__global__ void gdn_state_widen_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = gdn_state_load(src + i);
}

__global__ void gdn_state_narrow_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        gdn_state_store(dst + i, src[i]);
}

template <typename St>
void launch_step_norm(St* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                      const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    if (!old_gdn_step().on() && step_cluster_ready<St>()) {   // issue #26 (O7)
        cudaLaunchAttribute at{};
        at.id = cudaLaunchAttributeClusterDimension;
        at.val.clusterDim.x = GC;
        at.val.clusterDim.y = 1;
        at.val.clusterDim.z = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((unsigned) (h_v * GC));
        cfg.blockDim = dim3(GCW, RG);
        cfg.stream = (cudaStream_t) stream;
        cfg.attrs = &at;
        cfg.numAttrs = 1;
        const cudaError_t le = cudaLaunchKernelEx(&cfg, gdn_step_norm_cluster_kernel<St>, state, q, k, v, gate, beta, z,
                                                  gamma, eps, y, h_k, h_v);
        if (le != cudaSuccess) {
            std::fprintf(stderr, "fused_gdn_step_norm (cluster; STRATA_OLD_GDN_STEP=1 avoids it): %s\n",
                         cudaGetErrorString(le));
            std::exit(1);
        }
    } else {
        gdn_step_norm_kernel<St><<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, q, k, v, gate, beta,
                                                                                           z, gamma, eps, y, h_k, h_v);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gdn_step_norm: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

unsigned blocks_1d(int64_t n) { return (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096); }

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_kernel<<<(unsigned) (channels / S), S, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, qk_heads, eps);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_conv_l2: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    if (old_gdn_ab().on() || n_embd > AB_LOADS * AB_THREADS * 8)
        gdn_ab_kernel<<<(unsigned) ((2 * h_v + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a,
                                                                                         gate, beta, n_embd, h_v);
    else
        gdn_ab_row_kernel<<<(unsigned) (2 * h_v), AB_THREADS, (size_t) (n_embd / 8) * 48, (cudaStream_t) stream>>>(
            x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_ab: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (gdn_state_bf16_enabled())   // #53: the slice's first half holds the BF16 state
        launch_step_norm(reinterpret_cast<uint16_t*>(state), q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
    else
        launch_step_norm(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}

void gdn_state_bf16_set_enabled(bool enabled) { g_state_bf16.store(enabled, std::memory_order_relaxed); }
bool gdn_state_bf16_enabled() { return g_state_bf16.load(std::memory_order_relaxed); }

void fused_gdn_step_norm_bf16(uint16_t* state, const float* q, const float* k, const float* v, const float* gate,
                              const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                              int h_v, void* stream) {
    launch_step_norm(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}

void gdn_state_widen(const uint16_t* src, float* dst, int64_t n, void* stream) {
    if (!src || !dst || n < 0) { std::fprintf(stderr, "gdn_state_widen: invalid arguments\n"); std::exit(1); }
    if (n == 0) return;
    gdn_state_widen_kernel<<<blocks_1d(n), 256, 0, (cudaStream_t) stream>>>(src, dst, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "gdn_state_widen: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void gdn_state_narrow(const float* src, uint16_t* dst, int64_t n, void* stream) {
    if (!src || !dst || n < 0) { std::fprintf(stderr, "gdn_state_narrow: invalid arguments\n"); std::exit(1); }
    if (n == 0) return;
    gdn_state_narrow_kernel<<<blocks_1d(n), 256, 0, (cudaStream_t) stream>>>(src, dst, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "gdn_state_narrow: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool fused_gdn_step_cluster_path() {
    return !old_gdn_step().on() && step_cluster_ready<float>();
}

}  // namespace strata::kernels
