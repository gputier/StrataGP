// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
// B7 (STRATA_PREFILL_F16_SAT=1, off by default): the FP16 images saturate finite values beyond the FP16 range to
// +-65504 instead of rounding them to +-inf; NaN and +-inf pass through, so an FP32 overflow upstream stays visible.
// Off, hf is __float2half_rn exactly.
__constant__ int c_f16_sat = 0;
__device__ __forceinline__ uint16_t hf(float f) {
    if (c_f16_sat != 0 && fabsf(f) > 65504.0f && fabsf(f) != INFINITY) f = copysignf(65504.0f, f);
    return __half_as_ushort(__float2half_rn(f));
}
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::atoi(v) != 0;
}
// KernelPath::Env -> the rewrite, unless the function's STRATA_OLD_PREFILL_* variable (`env_old`, read once) is 1
bool use_old(KernelPath p, bool env_old) { return p == KernelPath::Old || (p == KernelPath::Env && env_old); }
// B7's flag: STRATA_PREFILL_F16_SAT=1 sets it once per device (__constant__ memory is per device, and a layer split
// runs a prompt path on each of two or three cards), before the first launch there of a kernel that calls hf, with
// a synchronous copy and a device synchronization, so every later launch sees it whatever its stream.
// set_f16_saturate (the tests, one GPU) sets it the same way and wins over the variable.
constexpr int kF16Devices = 64;
std::once_flag g_f16_once[kF16Devices];
std::atomic<bool> g_f16_forced{false};
void f16_mode(void* /*stream*/) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kF16Devices) dev = 0;
    std::call_once(g_f16_once[dev], [] {
        const int on = 1;
        if (!g_f16_forced.load() && env_on("STRATA_PREFILL_F16_SAT")) {
            if (cudaMemcpyToSymbol(c_f16_sat, &on, sizeof on) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
                check("f16_mode");
        }
    });
}

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}
// P8: the chain without the FP32 image of xn.  gr_norm_kernel's arithmetic, keeping the row's scale instead of
// xn; gr_mix_scale_kernel recomputes xn = (R * rs) * w, the value gr_norm_kernel stores.  256 threads a row.
__global__ void gr_norm_scale_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                     float* __restrict__ rsc, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rsc[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
__global__ void gr_mix_scale_kernel(const float* __restrict__ R, const float* __restrict__ rsc,
                                    const float* __restrict__ w, const float* __restrict__ g, float* __restrict__ mixed,
                                    uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(R[j] * rsc[t * HC + c] * w[c * N + d], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
// gr_write_kernel's update of a row, kept in registers, then gr_norm_scale_kernel of it (same thread-to-d map, so
// the same sum of squares): the next half's norm without reading R again.  256 threads a row.
constexpr int GR_THREADS = 256;
__global__ void __launch_bounds__(GR_THREADS) gr_write_norm_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                                   const float* __restrict__ inj, int64_t inj_ld,
                                                                   const float* __restrict__ w, float eps,
                                                                   float* __restrict__ rsc,
                                                                   uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x, t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float f = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[N / GR_THREADS];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < N / GR_THREADS; ++k) {
        const int d = threadIdx.x + k * GR_THREADS;
        v[k] = fmaf(bo[t * N + d], f, r[d]);
        r[d] = v[k];
        ss += v[k] * v[k];
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rsc[row] = rs;
#pragma unroll
    for (int k = 0; k < N / GR_THREADS; ++k) {
        const int d = threadIdx.x + k * GR_THREADS;
        xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
    }
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    // block (t, head) over the 32 q/k heads, 128 threads
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
// P2: the conv is a 4-tap FIR, so output t needs only inputs t-3..t (the history below t = 0): block per (head of
// 128 channels, CONV_TB tokens), a thread per channel, the tile's inputs loaded once, then the L2 norm of the 32 q/k
// heads in the same block.  The arithmetic is gdn_conv_kernel's and gdn_l2_kernel's, in their order (same bits);
// the history is written by gdn_conv_hist_kernel after every block has read it.
constexpr int CONV_TB = 16;
__global__ void __launch_bounds__(S) gdn_conv_tile_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                                          const float* __restrict__ w, float* __restrict__ h, int64_t T,
                                                          float eps) {
    __shared__ float part[CONV_TB][4];
    const int head = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c = head * S + threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TB;
    float x[CONV_TB + 3];   // inputs t0-3 .. t0+CONV_TB-1
#pragma unroll
    for (int j = 0; j < CONV_TB + 3; ++j) {
        const int64_t t = t0 + j - 3;
        x[j] = t < 0 ? hist[c * 3 + 3 + t] : (t < T ? qkv[t * C + c] : 0.0f);
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float o[CONV_TB];
#pragma unroll
    for (int j = 0; j < CONV_TB; ++j) {
        const float v0 = x[j], v1 = x[j + 1], v2 = x[j + 2], xt = x[j + 3];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + xt * w3;
        o[j] = s / (1.0f + __expf(-s));
    }
    if (head < 2 * HK) {   // block-uniform: the q and k heads
#pragma unroll
        for (int j = 0; j < CONV_TB; ++j) {
            const float v = o[j];
            const float sq = warp_sum(v * v);
            if (lane == 0) part[j][warp] = sq;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < CONV_TB; ++j) {
            const float ss = part[j][0] + part[j][1] + part[j][2] + part[j][3];
            o[j] = o[j] * rsqrtf(ss + eps);
        }
    }
#pragma unroll
    for (int j = 0; j < CONV_TB; ++j)
        if (t0 + j < T) h[(t0 + j) * C + c] = o[j];
}
// the chunk's last three inputs (the history's own older entries when T < 3), after gdn_conv_tile_kernel
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int64_t t = T - 3 + j;
        v[j] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + 3 + t];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) hist[c * 3 + j] = v[j];
}
constexpr int RG = 4, RPG = S / RG;
__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][S], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);
            y[t * HV * S + head * S + col] = v;
            y16[t * HV * S + head * S + col] = hf(v);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// P3: the columns of a value head are independent (only the output norm couples them): grid (48 heads, 4 quarters of
// 32 columns).  A warp holds 8 columns x the 4 row groups of 32 rows (lane = 4 * column + group), so kv and o are
// gdn_rec_kernel's four partial sums, added in its order through shuffles instead of shared memory and barriers.
// The q|k of a token go through shared memory, three buffers deep: token t+1's are written before token t's barrier,
// so the chain of kv(t+1) (the updated state against k(t+1)) runs alongside the chain of o(t) - the same operations
// in the same order as computing it at the next token - and the next inputs are loaded while a token runs.  One
// barrier a token.  o * rsqrt(128) goes to `o`; gdn_out_norm_kernel applies the norm.  Same bits.
constexpr int SPLIT = 4, SC = S / SPLIT, QK_LD = RPG + 4;   // column blocks per head, their columns, a group's stride
// h is far larger than L2 at the chunk sizes the prompt path runs, so a token's inputs come from DRAM: they are loaded
// two tokens ahead, and their cache lines requested REC_PF tokens ahead (a prefetch does not touch the arithmetic)
constexpr int REC_PF = 8;
__device__ __forceinline__ float col_sum(float x, int src) {   // the column's four row-group partials, in order
    return __shfl_sync(0xffffffffu, x, src) + __shfl_sync(0xffffffffu, x, src + 1) +
           __shfl_sync(0xffffffffu, x, src + 2) + __shfl_sync(0xffffffffu, x, src + 3);
}
__global__ void __launch_bounds__(SC * RG) gdn_rec_split_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta, float* __restrict__ o,
                                                               int64_t T) {
    // q and k by row group, 32 rows at stride QK_LD: the four groups of a warp read four different sets of banks.
    // Buffer (t % 3) holds token t: written before barrier t-1, last read before barrier t+1.
    __shared__ __align__(16) float sqk[3][2][RG][QK_LD];
    const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, rg = lane & 3, src = lane & ~3;
    const int col = blockIdx.y * SC + (tid >> 5) * 8 + (lane >> 2), qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    // token 0 (T >= 1): its q|k to buffer 0 and its kv.  In registers: token 1's q|k (thread tid carries q[tid] and
    // k[tid]), the gate, beta and v of tokens 0 (c*) and 1 (n*).
    sqk[0][0][tid >> 5][tid & 31] = h[qh * S + tid];
    sqk[0][1][tid >> 5][tid & 31] = h[HK * S + qh * S + tid];
    float nq = 0.0f, nk = 0.0f, ng = 0.0f, nb = 0.0f, nv = 0.0f;
    if (T > 1) {
        nq = h[C + qh * S + tid];
        nk = h[C + HK * S + qh * S + tid];
        ng = gate[HV + head];
        nb = beta[HV + head];
        nv = h[C + 2 * HK * S + head * S + col];
    }
    float cg = gate[head], cb = beta[head], cv = h[2 * HK * S + head * S + col];
    __syncthreads();
    float kv_col;
    {
        const float4* k4 = reinterpret_cast<const float4*>(sqk[0][1][rg]);
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < RPG / 4; ++i) {
            const float4 k = k4[i];
            kv = fmaf(s[4 * i], k.x, kv);
            kv = fmaf(s[4 * i + 1], k.y, kv);
            kv = fmaf(s[4 * i + 2], k.z, kv);
            kv = fmaf(s[4 * i + 3], k.w, kv);
        }
        kv_col = col_sum(kv, src);
    }
    int b = 0, bn = 1;   // the buffers of tokens t and t+1
    for (int64_t t = 0; t < T; ++t) {
        if (t + 1 < T) {
            sqk[bn][0][tid >> 5][tid & 31] = nq;
            sqk[bn][1][tid >> 5][tid & 31] = nk;
        }
        const float g = __expf(cg), bt = cb, vt = cv;
        cg = ng;
        cb = nb;
        cv = nv;
        if (t + 2 < T) {   // token t+2's inputs, two tokens ahead
            const float* h2 = h + (t + 2) * C;
            nq = h2[qh * S + tid];
            nk = h2[HK * S + qh * S + tid];
            ng = gate[(t + 2) * HV + head];
            nb = beta[(t + 2) * HV + head];
            nv = h2[2 * HK * S + head * S + col];
        }
        if (t + REC_PF < T && tid < 11) {   // and token t+REC_PF's lines into L2: q, k (4 lines each), v, gate, beta
            const float* hp = h + (t + REC_PF) * C;
            const float* a = tid < 4   ? hp + qh * S + tid * 32
                             : tid < 8 ? hp + HK * S + qh * S + (tid - 4) * 32
                             : tid == 8 ? hp + 2 * HK * S + head * S + blockIdx.y * SC
                             : tid == 9 ? gate + (t + REC_PF) * HV + head
                                        : beta + (t + REC_PF) * HV + head;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
        __syncthreads();
        const float delta = (vt - g * kv_col) * bt;
        const float4* q4 = reinterpret_cast<const float4*>(sqk[b][0][rg]);
        const float4* k4 = reinterpret_cast<const float4*>(sqk[b][1][rg]);
        const float4* kn4 = reinterpret_cast<const float4*>(sqk[bn][1][rg]);   // stale at the last token: unused
        float oo = 0.0f, kv = 0.0f;
#pragma unroll
        for (int i = 0; i < RPG / 4; ++i) {
            const float4 k = k4[i], q = q4[i], kn = kn4[i];
            s[4 * i] = fmaf(g, s[4 * i], k.x * delta);
            oo = fmaf(s[4 * i], q.x, oo);
            kv = fmaf(s[4 * i], kn.x, kv);
            s[4 * i + 1] = fmaf(g, s[4 * i + 1], k.y * delta);
            oo = fmaf(s[4 * i + 1], q.y, oo);
            kv = fmaf(s[4 * i + 1], kn.y, kv);
            s[4 * i + 2] = fmaf(g, s[4 * i + 2], k.z * delta);
            oo = fmaf(s[4 * i + 2], q.z, oo);
            kv = fmaf(s[4 * i + 2], kn.z, kv);
            s[4 * i + 3] = fmaf(g, s[4 * i + 3], k.w * delta);
            oo = fmaf(s[4 * i + 3], q.w, oo);
            kv = fmaf(s[4 * i + 3], kn.w, kv);
        }
        const float oc = col_sum(oo, src) * rsqrtf((float) S);
        if (rg == 0) o[t * HV * S + head * S + col] = oc;
        kv_col = col_sum(kv, src);
        b = bn;
        bn = bn == 2 ? 0 : bn + 1;
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// P3's second pass, in place (o is y): block per (token, head), a thread per column.  The sum of o^2 is
// gdn_rec_kernel's (the same butterfly per warp of 32 columns, the four warp sums in order), so is y.
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(float* y, const float* __restrict__ z,
                                                        const float* __restrict__ gamma, float eps,
                                                        uint16_t* __restrict__ y16) {
    __shared__ float wsum[S / 32];
    const int col = threadIdx.x;
    const int64_t i = (int64_t) blockIdx.x * S + col;   // blockIdx.x = t * HV + head
    const float oc = y[i];
    const float sp = warp_sum(__fmul_rn(oc, oc));   // unfused: gdn_rec_kernel's square reaches the sum through a select
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[i]);
    y[i] = v;
    y16[i] = hf(v);
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
    bo[i] = s + shared[i] * sigm(sg[t]);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    f16_mode(stream);
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{});
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_mode(stream);
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    if (mixed_h) f16_mode(stream);
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gr_norm_scale(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    gr_norm_scale_kernel<<<(unsigned) (T * HC), GR_THREADS, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_scale");
}
void gr_mix_scale(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed,
                  uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h) {
    if (T <= 0) return;
    if (mixed_h) f16_mode(stream);
    gr_mix_scale_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                             mixed_h);
    check("gr_mix_scale");
}
void gr_write_norm(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                   float* rs, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    gr_write_norm_kernel<<<(unsigned) (T * HC), GR_THREADS, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next,
                                                                                      eps, rs, xn16);
    check("gr_write_norm");
}
bool gr_old_path() {
    static const bool v = env_on("STRATA_OLD_PREFILL_GR");
    return v;
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              KernelPath path) {
    static const bool env_old = env_on("STRATA_OLD_PREFILL_GDN_CONV");
    if (use_old(path, env_old)) {
        gdn_conv_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps);
        check("gdn_conv");
        return;
    }
    if (T <= 0) return;
    gdn_conv_tile_kernel<<<dim3(C / S, (unsigned) ((T + CONV_TB - 1) / CONV_TB)), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, T, eps);
    gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
    check("gdn_conv");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, KernelPath path) {
    static const bool env_old = env_on("STRATA_OLD_PREFILL_GDN_REC");
    f16_mode(stream);
    if (use_old(path, env_old)) {
        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T);
        check("gdn_recurrence");
        return;
    }
    if (T <= 0) return;
    gdn_rec_split_kernel<<<dim3(HV, SPLIT), SC * RG, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
    gdn_out_norm_kernel<<<(unsigned) (T * HV), S, 0, (cudaStream_t) stream>>>(y, z, gamma, eps, y16);
    check("gdn_recurrence");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    static const bool decode = env_on("STRATA_PREFILL_ROUTE_DECODE");
    if (decode) {
        route_decode(logits, ids, weights, T, n_expert, stream);
        return;
    }
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
void route_decode(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (T <= 0) return;
    if (strata::kernels::native_router_enabled() && n_expert == 512) route_native(logits, ids, weights, T, stream);
    else strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route_decode");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    f16_mode(stream);
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_mode(stream);
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    f16_mode(stream);
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    const float theta_scale = powf(freq_base, -2.0f / 64.0f);
    rope_kernel<<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(x, heads, dim, ld, pos0, theta_scale, strata::kernels::mrope_table());
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    f16_mode(stream);
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}
void set_f16_saturate(bool on) {
    g_f16_forced.store(true);
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(c_f16_sat, &v, sizeof v) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        check("set_f16_saturate");
}

}  // namespace strata::prefill
