// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
// Options: --kv fp32|q4 (the K / V cache format), --rope-scaling none|linear|yarn with --rope-scale F and
// --rope-orig-ctx N (the GGUF's rope.scaling.* keys when absent).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/pcie_share.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q4_group.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "q36_doorbell.hpp"
#include "q36_kernels.cuh"
#include "q36_pcie.hpp"
#include "q36_rebalance.hpp"
#include "q36_sampling.hpp"
#include "q36_stats.hpp"
#include "q36_threads.hpp"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)
// the PCIe copies' DMA mode unless STRATA_DMA_BATCH is set (dma_batch.hpp: 1 = one batch per group)
constexpr int kPcieDmaBatch = 1;

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    void* const* kc = nullptr;       // device [n_sessions]: this layer's K / V base of each session (fp32 or q4_0 cells)
    void* const* vc = nullptr;
};

// ---- the 4-bit KV cache (--kv q4): kv_q4.hpp's layout (a cell of one KV head = 8 q4_0 blocks of 32 values, 144 bytes,
// row (kv head, cell) at kvh * ctx + cell) and its Hadamard rotation of K, V and q.
using strata::kernels::block_q4_0;
constexpr int kQ4Blocks = 256 / strata::kernels::QK4_0;   // q4_0 blocks per head_dim 256

/// Orthonormal 256-point Walsh-Hadamard transform of the block's 256 values (one per thread, blockDim 256) through
/// `buf`: kv_q4's fwht256_cuda for one row; its own inverse.
__device__ float fwht256_block(float v, float* buf) {
    const int i = threadIdx.x;
    __syncthreads();   // (buf's earlier readers)
    buf[i] = v;
    __syncthreads();
    for (int h = 1; h < 256; h <<= 1) {
        const float a = buf[i], b = buf[i ^ h];
        __syncthreads();
        buf[i] = (i & h) ? b - a : a + b;
        __syncthreads();
    }
    const float r = buf[i] * (1.0f / 16.0f);
    __syncthreads();
    return r;
}
/// The q4_0 block of the 32 values held one per lane of a warp, stored at `blk` (kv_q4_group.cuh's quantizer, Flash-Next's).
__device__ __forceinline__ void q4_store_group(block_q4_0* blk, float x, int lane) {
    uint8_t byte;
    const uint16_t d = strata::kernels::q4_group(x, byte);
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}
/// Value e (0 .. 31) of a q4_0 block.
__device__ __forceinline__ float q4_value(const block_q4_0* blk, int e) {
    const uint8_t b = blk->qs[e & 15];
    return (float) ((e < 16 ? (b & 0x0F) : (b >> 4)) - 8) * strata::kernels::f32_from_f16(blk->d);
}

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Q4: q, k and v are rotated by the Hadamard transform (q.k is unchanged) and k, v stored as q4_0 blocks (HD == 256).
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).  `ra`: the
/// rope scaling's constants (rope_scaling.hpp; the identity when off).
/// grid (H + HKV, T), block 256 (HD <= 256).
template <bool Q4>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, void* kc, void* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs, strata::kernels::RopeKernelArgs ra) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        if (ra.freq_scale == 1.0f && ra.ext_factor == 0.0f && ra.attn_factor == 1.0f)
            sincosf(theta, &sn, &cs);   // no scaling: the trained angles, as before
        else
            strata::kernels::rope_scaled_angle(theta, ra.freq_scale, ra.corr_low, ra.corr_high, ra.ext_factor, ra.attn_factor, p,
                                               cs, sn);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        if constexpr (Q4) v = fwht256_block(v, buf);
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const float vv = vx[((size_t) t * HKV + kh) * HD + i];
        if constexpr (Q4) {
            const size_t row = (size_t) kh * ctx + pos;
            const float kr = fwht256_block(v, buf), vr = fwht256_block(vv, buf);
            q4_store_group(static_cast<block_q4_0*>(kc) + row * kQ4Blocks + i / 32, kr, i % 32);
            q4_store_group(static_cast<block_q4_0*>(vc) + row * kQ4Blocks + i / 32, vr, i % 32);
        } else {
            static_cast<float*>(kc)[((size_t) kh * ctx + pos) * HD + i] = v;
            static_cast<float*>(vc)[((size_t) kh * ctx + pos) * HD + i] = vv;
        }
    }
}

/// Split-K attention, one block per (chunk of `tiles` x 256 cells, query row): query row qi = t * H + h (t from t0) sees
/// the cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum (the tiles of a chunk merged by the
/// online softmax); chunks past the query's cells write nothing (the merge stops before them).  Q4: the cache holds
/// q4_0 cells and q is rotated (k_attn_prep).  grid (n_chunk, n_rows), block 256, HD == 256.
template <bool Q4>
__global__ void k_attn_part(const float* q, const void* kc, const void* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, int tiles, float* pacc, float* pm, float* pl,
                            RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    if (c * tiles * kChunkCells >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    float mrun = -INFINITY, lrun = 0.f, acc = 0.f;
    for (int tile = 0; tile < tiles; ++tile) {
        const int p0 = (c * tiles + tile) * kChunkCells;
        if (p0 >= npos) break;
        __syncthreads();   // (qs, and the previous tile's sc readers)
        const int p = p0 + j;
        float s = -INFINITY;
        if (p < npos) {
            float a = 0.f;
            if constexpr (Q4) {
                const block_q4_0* kb = static_cast<const block_q4_0*>(kc) + ((size_t) kvh * ctx + p) * kQ4Blocks;
                for (int b = 0; b < kQ4Blocks; ++b) {
                    float ab = 0.f;
                    for (int e = 0; e < 16; ++e) {
                        const uint8_t byte = kb[b].qs[e];
                        ab += qs[b * 32 + e] * (float) ((byte & 0x0F) - 8) + qs[b * 32 + 16 + e] * (float) ((byte >> 4) - 8);
                    }
                    a += ab * strata::kernels::f32_from_f16(kb[b].d);
                }
            } else {
                const float* kp = static_cast<const float*>(kc) + ((size_t) kvh * ctx + p) * HD;
                for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
            }
            s = a * scale;
        }
        const float mt = block_max(s, sh);
        const float mnew = fmaxf(mrun, mt);
        const float alpha = expf(mrun - mnew);   // 0 for the first tile
        const float e = p < npos ? expf(s - mnew) : 0.f;
        sc[j] = e;
        lrun = lrun * alpha + block_sum(e, sh);
        const int n = min(kChunkCells, npos - p0);
        acc *= alpha;
        if constexpr (Q4) {
            const block_q4_0* vb = static_cast<const block_q4_0*>(vc) + ((size_t) kvh * ctx + p0) * kQ4Blocks + j / 32;
            for (int k = 0; k < n; ++k) acc += sc[k] * q4_value(vb + (size_t) k * kQ4Blocks, j % 32);
        } else {
            const float* vp = static_cast<const float*>(vc) + ((size_t) kvh * ctx + p0) * HD + j;
            for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
        }
        mrun = mnew;
    }
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = mrun;
        pl[o] = lrun;
    }
}
/// The chunks (`cells` each) merged, times sigmoid(gate): grid (n_rows), block HD.  Q4: the merged values are the
/// rotated V's mix, rotated back before the gate.
template <bool Q4>
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, int cells, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + cells - 1) / cells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    if constexpr (Q4) {
        __shared__ float buf[HD];
        out[gi] = fwht256_block(num / den, buf) * sigm(gate[gi]);
    } else {
        out[gi] = num / den * sigm(gate[gi]);
    }
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The hybrid mode's entry split inside a graph, from the device's own slot table (a rebalance rewrites the table, not
/// the graph): miss[e] = the entry's expert has no VRAM slot; hit_rows[0, *n_hit) = the entries that have one.  One block.
inline constexpr int kHySplitMax = 256;   // entries (tokens x top_k) one k_hy_split block serves
__global__ void k_hy_split(const int32_t* ids, int n_ent, const int32_t* slot_of, int32_t* miss, int32_t* hit_rows,
                           int32_t* n_hit) {
    __shared__ unsigned char hit[kHySplitMax];
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        hit[e] = slot_of[ids[e]] >= 0;
        miss[e] = !hit[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int e = 0; e < n_ent; ++e)
            if (hit[e]) hit_rows[n++] = e;
        *n_hit = n;
    }
}
/// The prompt attention's FP16 operands: cells [c0, c0 + P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves (Q4: the q4_0 cells dequantized, still rotated).  grid1(HKV * P * HD).
template <bool Q4>
__global__ void k_kv_f16(const void* kc, const void* vc, __half* k16, __half* v16, int ctx, int c0, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    if constexpr (Q4) {
        const size_t cell = (size_t) kvh * ctx + c0 + rest / HD;   // (HD == 256)
        const int e = (int) (rest % HD);
        k16[i] = __ushort_as_half(strata::kernels::f16_from_f32(
            q4_value(static_cast<const block_q4_0*>(kc) + cell * kQ4Blocks + e / 32, e % 32)));
        v16[i] = __ushort_as_half(strata::kernels::f16_from_f32(
            q4_value(static_cast<const block_q4_0*>(vc) + cell * kQ4Blocks + e / 32, e % 32)));
    } else {
        const size_t src = ((size_t) kvh * ctx + c0) * HD + rest;
        k16[i] = __float2half(static_cast<const float*>(kc)[src]);
        v16[i] = __float2half(static_cast<const float*>(vc)[src]);
    }
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// One key window [c0, c0 + Pw) of the prompt attention with the online softmax (Q4: the keys are gathered a window at a
/// time, kKvWindow cells, so the FP16 operands stay small): row (h, t) of the slice (nt queries from query t0 of the chunk,
/// whose first sits at cell pos0) takes the window's scores S (ld floats a row), raises its running max / sum (mrun, lrun,
/// one per query row t * H + h), writes the window's probabilities relative to the new max as FP16 (unnormalized: they
/// are at most 1) and scales the rows already accumulated in O by the change of the max.  A window with no cell the row
/// sees writes zeros and leaves the row alone.  grid (nt, H), block 256.
__global__ void k_attn_win(const float* S, __half* S16, int ld, int nt, int Pw, int c0, int pos0, int t0, int H, float scale,
                           float* O, float* mrun, float* lrun) {
    constexpr int HD = 256;
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * nt + t) * ld;
    __half* out = S16 + ((size_t) h * nt + t) * ld;
    const size_t r = (size_t) (t0 + t) * H + h;
    const int n = min(Pw, pos0 + t + 1 - c0);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    const float mold = c0 == 0 ? -INFINITY : mrun[r];
    const float mnew = fmaxf(mold, mx);
    float sum = 0.f;
    for (int p = threadIdx.x; p < Pw; p += blockDim.x) {
        const float e = p < n ? expf(row[p] * scale - mnew) : 0.f;
        out[p] = __float2half(e);
        sum += e;
    }
    sum = block_sum(sum, sh);
    const float alpha = expf(mold - mnew);   // 0 on the first window, 1 when the row sees nothing here
    if (threadIdx.x == 0) {
        mrun[r] = mnew;
        lrun[r] = (c0 == 0 ? 0.f : lrun[r]) * alpha + sum;
    }
    if (c0 > 0)
        for (int d = threadIdx.x; d < HD; d += blockDim.x) O[r * HD + d] *= alpha;
}
/// O[row] /= lrun[row] over the windows' accumulated rows: n = T * H rows of 256.  grid1(n * 256).
__global__ void k_attn_win_norm(float* O, const float* lrun, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) O[i] /= lrun[i / 256];
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prefetch's predicted ids to the host's mapped memory; the fence orders them before the doorbell the next launch rings.
__global__ void k_publish_ids(const int32_t* __restrict__ ids, volatile int32_t* out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = ids[i];
    __threadfence_system();
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties), -1 when the row holds a NaN or no finite logit.  Stage 1:
/// `gridDim.x` blocks each write the best of their stride into (pv, pi), pv = NaN when their stride holds one; stage 2 (one block,
/// n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    __shared__ int bn[kThreads];   // a NaN seen by the thread (then by the block)
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    int nan = 0;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (isnan(pv[i])) nan = 1;
            else if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            if (isnan(x[i])) nan = 1;
            else if (x[i] > v) { v = x[i]; id = i; }
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    bn[threadIdx.x] = nan;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
            bn[threadIdx.x] |= bn[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // a NaN anywhere in the row, or no finite logit (all -inf): -1, the callers' ERR
        if (stage2) *out = (bn[0] || !isfinite(bv[0])) ? -1 : bi[0];
        else { pv[blockIdx.x] = bn[0] ? NAN : bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)
// the ERR (or stderr) text of a row with no finite logit: serve/server.py raises it, the tests read it
constexpr const char* kNoLogitMsg = "the logits hold no finite maximum or a NaN: no token drawn";

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    strata::kernels::RopeKernelArgs rope;   // the rope scaling's constants (identity = off), set by run()
    std::string rope_type;                  // the GGUF's rope.scaling.type / .factor / .original_context_length, if any
    double rope_factor = 0, rope_orig_ctx = 0;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15)";
        return false;
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.scaling.type")) m.rope_type = v->s;
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.scaling.factor")) m.rope_factor = v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.scaling.original_context_length")) m.rope_orig_ctx = v->num();
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

/// Bytes of one KV head's K (or V) cell: a q4_0 row of the head, or head_dim fp32 floats.
static size_t kv_cell_bytes_for(bool q4, int64_t head_dim) {
    return q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) head_dim) : (size_t) head_dim * 4;
}

// VERIFY's checks on the device (Engine::verify_cmp, Engine::group_verify). One block compares the words as bit patterns, as
// memcmp did; thread 0 counts the call in its slot and, for a mismatching call, keeps the first differing word while the slot
// has room. The calls of a slot run in stream order, so a slot needs no atomics.
constexpr int kVerifyBlock = 1024;   // threads of a check
constexpr int kVerifyRecs = 20;      // mismatching calls kept per slot (a group slot can print kVerifyGroupLines of them)
constexpr int kVerifyGroupLines = 20;   // group mismatch lines printed in all, as HEAD printed them
constexpr int kVerifyGroups = 64;    // groups and entries the grouping check takes (route_group and group_entries bound them)
constexpr long long kNoWord = 0x7fffffffffffffffLL;
struct VRec { long long idx; unsigned a, b; };   // the first differing word of a mismatching call, both bit patterns
struct VSlot { unsigned long long calls, bad; VRec rec[kVerifyRecs]; };
struct VSite { std::string what, tag; long long n; bool group; int shown; };   // host: a slot's check site and lines shown
struct VGroup {   // the grouping of the routed entries (n: the group count, ptr: two words per group, start, dst, tok: words)
    const int32_t *n_a, *n_b;
    const void *ptr_a, *ptr_b;
    const int32_t *start_a, *start_b, *dst_a, *dst_b, *tok_a, *tok_b;
};
__device__ inline void vcmp_words(const uint32_t* a, const uint32_t* b, long long len, long long base, unsigned long long& cnt,
                                  long long& first) {
    for (long long i = threadIdx.x; i < len; i += blockDim.x)
        if (a[i] != b[i]) {
            ++cnt;
            first = min(first, base + i);
        }
}
__device__ inline void vcmp_store(VSlot* s, unsigned long long cnt, long long first, unsigned a, unsigned b) {
    ++s->calls;
    if (cnt == 0) return;
    if (s->bad < kVerifyRecs) s->rec[s->bad] = VRec{first, a, b};
    ++s->bad;
}
// The block's reduction of a check's counts, then thread 0 stores the call. words(i) gives the record of the word at i, and is
// read only for a mismatching call.
template <class Words>
__device__ inline void vcmp_block(VSlot* s, unsigned long long c, long long f, Words words) {
    __shared__ unsigned long long cnt[kVerifyBlock];
    __shared__ long long first[kVerifyBlock];
    cnt[threadIdx.x] = c;
    first[threadIdx.x] = f;
    __syncthreads();
    for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            cnt[threadIdx.x] += cnt[threadIdx.x + k];
            first[threadIdx.x] = min(first[threadIdx.x], first[threadIdx.x + k]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        VRec w{0, 0u, 0u};
        if (cnt[0]) w = words(first[0]);
        vcmp_store(s, cnt[0], first[0], w.a, w.b);
    }
}
__global__ void k_vcmp(const uint32_t* a, const uint32_t* b, long long n, VSlot* s) {
    unsigned long long c = 0;
    long long f = kNoWord;
    vcmp_words(a, b, n, 0, c, f);
    vcmp_block(s, c, f, [a, b](long long i) { return VRec{i, a[i], b[i]}; });
}
__global__ void k_vcmp_group(VGroup g, VSlot* s) {
    const int nn = g.n_a[0], no = g.n_b[0];
    bool ok = nn == no && nn >= 0 && nn <= kVerifyGroups;   // (every thread reads the same words: the branch is uniform)
    const int tot = ok ? g.start_a[nn] : 0;
    ok = ok && tot >= 0 && tot <= kVerifyGroups;
    if (!ok) {   // the counts differ or a count is past the bound: a mismatch with no word to show
        if (threadIdx.x == 0) vcmp_store(s, 1, -1, nn, no);
        return;
    }
    unsigned long long c = 0;
    long long f = kNoWord;
    const uint32_t* pa = (const uint32_t*) g.ptr_a;
    const uint32_t* pb = (const uint32_t*) g.ptr_b;
    vcmp_words(pa, pb, 2LL * nn, 0, c, f);
    vcmp_words((const uint32_t*) g.start_a, (const uint32_t*) g.start_b, nn + 1, 2LL * nn, c, f);
    vcmp_words((const uint32_t*) g.dst_a, (const uint32_t*) g.dst_b, tot, 3LL * nn + 1, c, f);
    vcmp_words((const uint32_t*) g.tok_a, (const uint32_t*) g.tok_b, tot, 3LL * nn + 1 + tot, c, f);
    vcmp_block(s, c, f, [nn, no](long long) { return VRec{0, (unsigned) nn, (unsigned) no}; });
}

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    // attention partials: att_rows query rows x n_chunk chunks of chunk_tiles x kChunkCells cells.  With --kv q4 the chunk
    // grows with the context so the partials stay under kMaxChunks per row: n_head x kGemmT x kMaxChunks x head_dim x 4 B
    // at most for a decode window of kGemmT tokens (128 MiB at 16 heads of 256, 256 MiB at 32)
    int n_chunk = 0, att_rows = 0, chunk_tiles = 1;
    static constexpr int kMaxChunks = 1024;
    bool kv_q4 = false;   // --kv q4: the K / V caches are q4_0 cells (kv_q4.hpp's layout) under a Hadamard rotation, not fp32
    std::vector<float*> conv, S;
    std::vector<void*> kc, vc;   // per layer; the MTP block's K / V at index n_layers: the CURRENT session's (select)
    /// Bytes of one KV head's K (or V) cell.
    size_t kv_cell_bytes() const { return kv_cell_bytes_for(kv_q4, m->g.head_dim); }
    static constexpr int kKvWindow = 16384;   // q4: cells of the prompt attention's key window (k16 / v16 hold one)
    float *win_m = nullptr, *win_l = nullptr;   // q4: the windows' running max / sum per query row [tmax][H]
    /// Cells k16 / v16 hold: the whole context (fp32: one window), or one key window (q4).
    int kv_window() const { return kv_q4 ? std::min(ctx, kKvWindow) : ctx; }
    /// Bytes of the K (or V) cache of one attention layer of one session.
    size_t kv_plane_bytes() const { return (size_t) m->g.n_head_kv * (size_t) ctx * kv_cell_bytes(); }
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S;
        std::vector<void*> kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr;
    void **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of;
        std::vector<double> score;   // routing counts, decayed at each rebalance
        std::vector<double> heat;    // the routing counted since the start (--expert-profile-save), never decayed
        int32_t* slot_of_d = nullptr;
        void count(size_t x, double n) { score[x] += n; heat[x] += n; }
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
    std::vector<strata::kernels::cpu::ActQ> actq;   // per token: its ActQ, the Q2_0 gate/up kernels' activation (the pool reads it)
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;   // the eager path's handshake (a record inside a capture is dropped: see the doorbell)
    // inside a captured graph the host cannot be called between nodes: the layer's routing goes to the host with a
    // doorbell kernel and the pool's rows come back behind a device-side wait (q36_doorbell.hpp).  The mapped pinned
    // regions are h_h / h_ids / cpu_out seen from the device; each captured hybrid layer is a HyStep, served in order
    // by the host after the graph's launch.
    struct HyStep { size_t li; int T; };
    strata::q36::HyDoorbell db;
    float *d_h_h = nullptr, *d_cpu_out = nullptr;
    int32_t *d_h_ids = nullptr, *d_hit_rows = nullptr, *d_n_hit = nullptr;
    std::vector<HyStep>* cap_steps = nullptr;       // set while a graph is being captured
    uint32_t cap_ring = 0;                          // the ring of the layer being captured (its place in cap_steps)
    std::map<cudaGraphExec_t, std::vector<HyStep>> hy_steps;   // per graph with hybrid layers: what the host serves
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    // the PCIe share of the missed experts (q36_pcie.hpp): hy_pcie = entries computed on the GPU from the staging region
    // (not counted in hy_cpu), hy_cpu_jobs = the pool's experts (one per expert and call), what PcieAuto reads
    long long hy_pcie = 0, hy_cpu_jobs = 0;
    long long pc_want = 0, pc_sent = 0;   // PCIe groups the share asked for / the staging region let through, since the last log line
    bool pcie_on = false;          // the graph carries the PCIe call of each hybrid layer; fixed before the first capture
    int pcie_num = 0;              // the share of the missed experts, in 256ths (0 with pcie_on: nothing goes that way)
    strata::core::PcieAuto pcie_auto;
    uint8_t* pc_stage = nullptr;   // kPcieSlots blobs of pc_blob bytes
    size_t pc_blob = 0;
    cudaStream_t pcs = nullptr;    // the staging copies (the host issues them, the graph never waits for them)
    // the prefetch (q36_pcie.hpp): while layer L - 1 is served, the blobs layer L is predicted to send over PCIe are copied on
    // pfs into the staging area's region of L's parity (pf_ids: its experts in slot order, pf_layer: the layer they are for),
    // ev_pf[par] marks the region's last copy; the predicted ids come from the router of L applied to L - 1's input
    bool pf_on = false;
    cudaStream_t pfs = nullptr;
    cudaEvent_t ev_pf[2] = {};
    int32_t pf_ids[2][strata::q36::kPrefetchSlots] = {};
    int pf_n[2] = {0, 0}, pf_layer[2] = {-1, -1};
    float *logits_pf = nullptr, *pred_w = nullptr;
    int32_t *d_pred = nullptr, *h_pred = nullptr, *d_h_pred = nullptr;
    int32_t *pc_plan = nullptr, *d_pc_plan = nullptr, *d_pc = nullptr;   // the plan: mapped (host, device) and its device copy
    std::vector<char> pc_mark;     // per expert: sent over PCIe by this call
    /// --expert-profile: the profile the slots started from (main layers only; empty: none), kept for --expert-profile-save
    std::vector<std::pair<int32_t, int32_t>> profile_prior;
    double hy_cpu_ms = 0;
    // what PcieAuto reads (the clocks behind them run only while it is on): pa_cpu_ms = the CPU path's own time, the pool's row
    // phases plus the jobs' build (not the PCIe launches the same host loop makes); the host time spent picking, launching and
    // planning the PCIe copies is WireStep::launch_ms
    double pa_cpu_ms = 0;
    // the wire clock (it runs with the PcieAuto clocks): each layer step stamps its demand copies (stream pcs) and its prefetch
    // copies (stream pfs) with a begin and an end event; a step is read, without a sync, once its events have completed, and
    // the busy time is the union of the intervals read (wire_ms), against the blobs the steps chose for PCIe (wire_chosen);
    // the launch time and the chosen blobs are read from the same steps, so PcieAuto divides both by the same count
    struct WireStep {
        cudaEvent_t ev[4] = {};   // demand begin, demand end, prefetch begin, prefetch end (null: that stream copied nothing)
        int chosen = 0, demand = 0, prefetch = 0;
        size_t bytes = 0;
        double launch_ms = 0.0;   // the host time spent picking, launching and planning the step's PCIe copies
    };
    struct WireCount {
        double wire_ms = 0, bytes = 0, cpu_ms = 0, launch_ms = 0;
        long long chosen = 0, demand = 0, prefetch = 0, jobs = 0;
    };
    cudaEvent_t ev_wire0 = nullptr;   // the common origin of the stamps
    std::vector<cudaEvent_t> wire_free;
    std::deque<WireStep> wire_pending;
    WireCount wire, wire_mark;
    int wire_steps = 0, wire_logs = 0;
    // STRATA_Q36_STATS=1 (q36_stats.hpp): the bytes each path reads, the host's wait on the doorbell, the experts' union over
    // consecutive tokens; every use below is behind stats_on, so a run without the switch reads no clock and adds no sync
    const bool stats_on = strata::q36::stats_enabled();
    strata::q36::DecodeStats stat_acc;
    strata::q36::ExpertUnion stat_union;
    long long stat_mixed = 0;          // decode windows of two or more sessions, and requests started beside a decoding one
    std::vector<char> stat_hit_mark;   // per expert: counted as a resident expert read by this layer step
    /// The accumulated counters now, the pool's included (the request's figures are the difference of two snapshots).
    strata::q36::DecodeStats stat_snapshot() const {
        strata::q36::DecodeStats d = stat_acc;
#if defined(STRATA_Q36_CPU_POOL)
        if (pool) {
            d.cpu_bytes = pool->multi_bytes;
            d.pool_gu_ms = pool->ms_multi_gu;
            d.pool_q_ms = pool->ms_multi_q;
            d.pool_down_ms = pool->ms_multi_down;
        }
#endif
        return d;
    }
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; a single session's slots follow the counts at every kRebalanceEvery-th position
    // exactly (a window never steps over one), so plain and speculative decoding see the same slot tables at every
    // position; the batch windows (run_rows) rebalance every kRebalanceEvery-th window instead, positions being per session
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty()) {
                for (int e = 0; e < n * Kp; ++e)
                    if (strata::q36::first_of_row(row_ids[li].data(), e, Kp)) hy[li].count((size_t) row_ids[li][(size_t) e], 1.0);
                if (stats_on)
                    for (int r = 0; r < n; ++r) stat_union.add(li, row_ids[li].data() + (size_t) r * Kp);
            }
    }
    void wire_stamp(cudaEvent_t& ev, cudaStream_t s) {
        if (wire_free.empty()) {
            CK(cudaEventCreate(&ev));
        } else {
            ev = wire_free.back();
            wire_free.pop_back();
        }
        CK(cudaEventRecord(ev, s));
    }
    /// Reads the layer steps whose copies have completed, in order (the first one still copying ends the pass).
    void wire_resolve() {
        std::vector<std::pair<double, double>> iv;
        while (!wire_pending.empty()) {
            WireStep& r = wire_pending.front();
            bool done = true;
            for (cudaEvent_t ev : r.ev) done = done && (ev == nullptr || cudaEventQuery(ev) == cudaSuccess);
            if (!done) break;
            bool ok = true;
            std::vector<std::pair<double, double>> mine;
            for (int i = 0; i < 4; i += 2) {
                if (r.ev[i] == nullptr) continue;
                float a = 0.f, b = 0.f;
                ok = ok && cudaEventElapsedTime(&a, ev_wire0, r.ev[i]) == cudaSuccess &&
                     cudaEventElapsedTime(&b, ev_wire0, r.ev[i + 1]) == cudaSuccess;
                mine.emplace_back((double) a, (double) b);
            }
            if (ok) {
                iv.insert(iv.end(), mine.begin(), mine.end());
                wire.launch_ms += r.launch_ms;
                wire.chosen += r.chosen;
                wire.demand += r.demand;
                wire.prefetch += r.prefetch;
                wire.bytes += (double) r.bytes;
            }
            for (cudaEvent_t ev : r.ev)
                if (ev != nullptr) wire_free.push_back(ev);
            wire_pending.pop_front();
        }
        wire.wire_ms += strata::q36::interval_union_ms(iv);
    }
    /// STRATA_Q36_STATS=1: one line per window (48 steps, then 96 each, as PcieAuto's updates): the wire time measured against
    /// the blobs chosen for PCIe and the blobs that crossed the link, and the share c / (c + w - l) gives with it.
    void wire_stats() {
        if (!pcie_on || ++wire_steps < (wire_logs == 0 ? 48 : 96)) return;
        wire_steps = 0;
        ++wire_logs;
        WireCount now = wire;
        now.cpu_ms = pa_cpu_ms;
        now.jobs = hy_cpu_jobs;
        const long long chosen = now.chosen - wire_mark.chosen, on_wire = now.demand - wire_mark.demand + now.prefetch - wire_mark.prefetch;
        const long long window_jobs = now.jobs - wire_mark.jobs;
        const double busy_us = (now.wire_ms - wire_mark.wire_ms) * 1000.0, bytes = now.bytes - wire_mark.bytes;
        if (chosen > 0 && on_wire > 0 && window_jobs > 0 && busy_us > 0.0) {
            const double w = busy_us / (double) chosen;
            const double c = (now.cpu_ms - wire_mark.cpu_ms) * 1000.0 / (double) window_jobs;
            const double l = (now.launch_ms - wire_mark.launch_ms) * 1000.0 / (double) chosen;
            std::fprintf(stderr, "strata-q36: stats: wire window %d: share %.2f, %lld blobs chosen for PCIe (%lld on demand, %lld "
                         "prefetched), busy %.0f us on the copy streams: %.1f us per chosen blob, %.1f us per blob on the wire, %.1f "
                         "GB/s; CPU path %.0f us per job, launch %.1f us per chosen blob -> c / (c + w - l) = %.3f\n", wire_logs,
                         pcie_num / 256.0, chosen, now.demand - wire_mark.demand, now.prefetch - wire_mark.prefetch, busy_us, w,
                         busy_us / (double) on_wire, bytes / (busy_us * 1e-6) / 1e9, c, l, c / (c + w - l));
        }
        wire_mark = now;
    }
    /// After a decode step / verify window: the PCIe share refined from the pool's speed (PcieAuto, Flash-Next's rule: three
    /// updates at most, after 48 then 96 and 96 more windows).
    void pcie_window() {
        if ((!pcie_auto.on && !stats_on) || warming) return;
        wire_resolve();
        if (stats_on) wire_stats();
        if (!pcie_auto.on) return;
        double tc_us = 0.0, rate = 0.0;
        const double nf = pcie_auto.window(pa_cpu_ms, hy_cpu_jobs, pcie_num / 256.0, &tc_us, &rate, wire.launch_ms, wire.wire_ms,
                                           wire.chosen);
        if (nf < 0.0) return;
        std::fprintf(stderr, "strata-q36: PCIe share: the CPU path takes %.0f us per missed expert (%.1f GB/s of expert bytes; the "
                     "pool's rows and the jobs' build), the copy streams %.0f us per chosen blob (%s; the link probe said %.1f GB/s) "
                     "with %.0f us of host launch per chosen blob -> %.2f (was %.2f; --pcie-frac N fixes it); since the last update "
                     "%lld of %lld announced groups went over PCIe (the staging region holds %d per layer step)\n", tc_us, rate,
                     pcie_auto.wire_us, pcie_auto.wire_measured ? "measured" : "not measured: no blob went over PCIe in the window",
                     pcie_auto.link_gbps, pcie_auto.launch_us, nf, pcie_num / 256.0, pc_sent, pc_want,
                     strata::q36::kPcieSlots);
        pc_want = pc_sent = 0;
        pcie_num = strata::core::pcie_num_of(nf);
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        pcie_window();
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(dalloc_t<uint8_t>(kv_plane_bytes()));
                ss.vc.push_back(dalloc_t<uint8_t>(kv_plane_bytes()));
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(dalloc_t<uint8_t>(kv_plane_bytes()));
            ss.vc.push_back(dalloc_t<uint8_t>(kv_plane_bytes()));
        }
        ss.hlast = dalloc((size_t) g.n_embd);
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    void init(const Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        k16 = dalloc_t<__half>((size_t) g.n_head_kv * kv_window() * g.head_dim);
        v16 = dalloc_t<__half>((size_t) g.n_head_kv * kv_window() * g.head_dim);
        if (kv_q4) {
            win_m = dalloc((size_t) T * g.n_head);
            win_l = dalloc((size_t) T * g.n_head);
        }
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        if (kv_q4 && (g.head_dim != 256 || kThreads != 256)) {
            std::fprintf(stderr, "strata-q36: --kv q4 needs a head_dim of 256 (the Hadamard transform's size), got %d\n",
                         (int) g.head_dim);
            std::exit(2);
        }
        const int tiles_total = (ctx + kChunkCells - 1) / kChunkCells;
        chunk_tiles = kv_q4 ? (tiles_total + kMaxChunks - 1) / kMaxChunks : 1;   // fp32 keeps one tile per chunk at any context
        n_chunk = (tiles_total + chunk_tiles - 1) / chunk_tiles;
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, kGemmT));   // the split-K attention only runs for T <= kGemmT
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (const Layer& L : m->layers) {   // the verify windows' GDN inputs (one set, whichever session runs)
            if (L.attn) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32 or q4_0, for the attention
            // layers and the MTP block, and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t kv_bytes = 2 * kv_plane_bytes() * n_attn;
            const size_t per = kv_bytes + (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
            size_t fr0 = 0, tot0 = 0;
            CK(cudaMemGetInfo(&fr0, &tot0));
            std::fprintf(stderr, "strata-q36: KV cache %s: %.2f GiB per session at a context of %d (%zu attention layers); attention "
                         "partials %.0f MiB (%d chunks of %d cells); prompt K / V halves %.0f MiB\n", kv_q4 ? "q4 (Hadamard q4_0)" : "fp32",
                         (double) kv_bytes / (1 << 30), ctx, n_attn, (double) att_rows * n_chunk * (g.head_dim + 2) * 4 / (1 << 20),
                         n_chunk, chunk_tiles * kChunkCells, (double) 2 * g.n_head_kv * kv_window() * g.head_dim * 2 / (1 << 20));
            if (n_sess > 1)
                std::fprintf(stderr, "strata-q36: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                             "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                             (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
            if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                std::fprintf(stderr, "strata-q36: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                             "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                             (double) per / (1 << 30), (double) fr0 / (1 << 30));
                std::exit(1);
            }
        }
        for (Sess& ss : sess) alloc_session(ss);
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hc(ns * (size_t) n_sess), hs(hc);
            std::vector<void*> hk(hc.size()), hv(hc.size());
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                }
            const std::pair<void**, const void*> tabs[4] = {{(void**) &d_conv_tab, hc.data()}, {(void**) &d_S_tab, hs.data()},
                                                            {(void**) &d_kc_tab, hk.data()}, {(void**) &d_vc_tab, hv.data()}};
            for (const auto& tb : tabs) {   // (each table: ns * n_sess pointers)
                CK(cudaMalloc(tb.first, hc.size() * sizeof(void*)));
                CK(cudaMemcpy(*tb.first, tb.second, hc.size() * sizeof(void*), cudaMemcpyHostToDevice));
            }
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
        }
        d_delta = dalloc_t<int32_t>(1);
        cur = 0;
        load_cur();
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        if (cache_prior > 0.0f) {
            if (!delta_mean) {
                delta_mean = dalloc_t<float>(m->layers.size() + 1);
                delta_count = dalloc_t<int32_t>(m->layers.size() + 1);
            }
            CK(cudaMemsetAsync(delta_mean, 0, (m->layers.size() + 1) * sizeof(float), st));
            CK(cudaMemsetAsync(delta_count, 0, (m->layers.size() + 1) * sizeof(int32_t), st));
        }
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    /// The routing kernels that take the routing options: the new kernels on a 256-expert, 8-pick model (the start refuses
    /// --cache-prior and --ser when this is false).
    bool fast_route() const { return fast_k && m->g.n_expert == 256 && m->top_k == 8; }
    // routing options (--cache-prior, --ser), both off: they act in route_group's decode / verify windows (T <= 8) only
    float cache_prior = 0.0f;   // the bias on resident experts' logits, in units of the layer's Delta_avg (q36_delta.hpp)
    // Delta_avg's state, one pair per layer (the MTP block last), engine-wide: it is one session's state because the start
    // refuses --cache-prior with two or more sessions (--batch) and with MTP drafts (their rejected rows would stay folded in).
    // Only with --cache-prior; zeroed by reset().
    float* delta_mean = nullptr;
    int32_t* delta_count = nullptr;
    int ser_kmin = 8;           // the picks of rank >= ser_kmin are kept only above ser_t * the top probability
    float ser_t = 0.0f;
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    // the verify scratch (vbuf), in floats: the start refuses a model whose matrices would not fit it
    static constexpr size_t kVerifyFloats = (size_t) 8 * 16384;
    static constexpr int kVerifySlots = 512;   // check sites of VERIFY: the start refuses a model that would need more
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    // The counters are on the device, so a check runs inside a captured graph: a check site (what, tag, n) binds a slot when
    // the host code runs (at capture time for a graph), and every launch of its kernel counts, a graph replay too. The host
    // reads the slots back outside any capture (verify_flush), which keeps v_calls, v_bad and v_by as they were.
    VSlot* vslots = nullptr;                 // device: kVerifySlots slots
    std::vector<VSite> v_sites;              // host: the check sites, in slot order
    std::map<std::string, int> v_slot_of;    // "what tag n" -> slot
    std::map<std::string, int> v_lines;      // mismatch lines printed, per check name (3 at most)
    int v_group_lines = 0;                   // group mismatch lines printed (kVerifyGroupLines at most)
    VSlot* verify_slot(const char* what, const std::string& tag, long long n, bool group) {
        const std::string key = std::string(what) + " " + tag + " " + std::to_string(n);
        auto it = v_slot_of.find(key);
        if (it == v_slot_of.end()) {
            if ((int) v_sites.size() == kVerifySlots) {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_VERIFY has more than %d check sites (%s%s): the slot table is full\n",
                             kVerifySlots, what, tag.c_str());
                // the bound slots' lines go out first: a capture being recorded ends (its graph is dropped), since the slots
                // cannot be read while it runs
                cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
                CK(cudaStreamIsCapturing(st, &cs));
                if (cs != cudaStreamCaptureStatusNone) {
                    cudaGraph_t dropped;
                    CK(cudaStreamEndCapture(st, &dropped));
                    CK(cudaGraphDestroy(dropped));
                }
                verify_flush();
                std::exit(2);
            }
            it = v_slot_of.emplace(key, (int) v_sites.size()).first;
            v_sites.push_back(VSite{what, tag, n, group, 0});
        }
        return vslots + it->second;
    }
    /// VERIFY: the bit patterns of a and b compared on the device, the call counted in its check site's slot.
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        k_vcmp<<<1, kVerifyBlock, 0, st>>>((const uint32_t*) a, (const uint32_t*) b, (long long) n,
                                           verify_slot(what, tag, (long long) n, false));
    }
    /// VERIFY: the slots read back, outside any capture (the engine's own syncs come first). v_calls, v_bad and v_by take the
    /// slots' totals; each check's mismatch lines print from the records not yet shown (3 per check, group lines 20 in all).
    void verify_flush() {
        if (!verify_k || v_sites.empty()) return;
        CK(cudaStreamSynchronize(st));
        std::vector<VSlot> h(v_sites.size());
        CK(cudaMemcpy(h.data(), vslots, h.size() * sizeof(VSlot), cudaMemcpyDeviceToHost));
        v_calls = 0;
        v_bad = 0;
        v_by.clear();
        for (size_t i = 0; i < v_sites.size(); ++i) {
            VSite& s = v_sites[i];
            v_calls += (long long) h[i].calls;
            v_bad += (long long) h[i].bad;
            if (!s.group && h[i].bad) v_by[s.what] += (long long) h[i].bad;
            const int seen = (int) std::min<unsigned long long>(h[i].bad, kVerifyRecs);
            for (; s.shown < seen; ++s.shown) {
                const VRec& r = h[i].rec[s.shown];
                if (s.group) {
                    if (v_group_lines < kVerifyGroupLines) {
                        ++v_group_lines;
                        std::fprintf(stderr, "strata-q36 VERIFY MISMATCH group (%s, groups %u vs %u)\n", s.tag.c_str(), r.a, r.b);
                    }
                    continue;
                }
                if (v_lines[s.what] >= 3) continue;
                ++v_lines[s.what];
                float fa, fb;
                std::memcpy(&fa, &r.a, 4);
                std::memcpy(&fb, &r.b, 4);
                std::fprintf(stderr, "strata-q36 VERIFY MISMATCH %s%s: first at %lld of %lld: new %g old %g (bits %08x %08x)\n",
                             s.what.c_str(), s.tag.c_str(), r.idx, s.n, (double) fa, (double) fb, r.a, r.b);
            }
        }
    }
    /// The Q8_0 matrices q8_small takes, whatever T: fast_k, a shape the kernel reads, not on the packed decode kernel. Only
    /// these write vbuf (T x n_out floats), so the VERIFY start guard tests exactly them.
    bool q8_small_shape(const Mat& w) const {
        if (!fast_k || w.type != 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        return w.n_in / 32 <= q36k::kQ8MaxBlocks;
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (T < 1 || T > 8 || !q8_small_shape(w)) return false;
        const int nb = w.n_in / 32;
        const void* wp = w.d;
        const half* dpl = nullptr;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        q36k::launch_q8_rows_nb<NCV, false>(wp, dpl, q, out, nb, w.n_out, st);                               \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, vbuf + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vbuf, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The layer's shared expert takes shared_gu, whatever T: fast_k, Q8_0 gate and up of FS outputs, not on the packed decode
    /// kernel. Only these write vbuf (gate at 0, up at 8 x FS, T <= 4 rows of each), so the VERIFY start guard tests exactly them.
    bool shared_gu_shape(const Layer& L) const {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        return fast_k && a.type == 8 && b.type == 8 && a.n_in == b.n_in && a.n_out == FS && b.n_out == FS && a.n_in % 32 == 0 &&
               a.n_in / 32 <= q36k::kQ8MaxBlocks && a.n_in / 32 > q36k::kQ8RowBlocks && FS % 32 == 0 &&
               !(K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled());
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (T < 1 || T > 4 || !shared_gu_shape(L)) return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db_half = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            q36k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db_half, (const q36k::QB*) xq,        \
                                                               (q36k::QB*) q8, nb, FS);                      \
        else if (nb <= 2 * q36k::kQ8RowBlocks)                                                               \
            q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        else                                                                                                 \
            q36k::k_sh_gu<NCV, 3, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            float* vg = vbuf;
            float* vu = vbuf + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        const VGroup g{n_groups, o_ng, grp_ptr, o_ptr, grp_start, o_start, ent_dst, o_dst, ent_tok, o_tok};
        k_vcmp_group<<<1, kVerifyBlock, 0, st>>>(g, verify_slot("group", "T " + std::to_string(T), 0, true));
    }
    /// A prompt chunk route_group does not serve (T > 8) still feeds the hybrid layer's Delta_avg (--cache-prior only).
    void delta_rows(size_t li, int T, int ld) {
        if (delta_mean && fast_route() && T > 8 && lhy(li))
            q36k::k_delta_rows<<<1, dim3(32, 8), 0, st>>>(logits_r, T, ld, delta_mean + li, delta_count + li);
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, size_t li, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_route() || T > 8) return false;
        const bool opt = cache_prior > 0.0f || ser_kmin < Kp;
        if (slot_of) {
            const auto kern = opt ? q36k::k_route_group<true, true> : q36k::k_route_group<true, false>;
            kern<<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                            ent_dst, ent_tok, cache_prior, ser_kmin, ser_t, delta_mean ? delta_mean + li : nullptr,
                                            delta_count ? delta_count + li : nullptr);
        } else {
            const auto kern = opt ? q36k::k_route_group<false, true> : q36k::k_route_group<false, false>;
            kern<<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                            ent_dst, ent_tok, cache_prior, ser_kmin, ser_t, nullptr, nullptr);
        }
        grouped_by_route = true;
        if (verify_k && !opt) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            int32_t* vid = (int32_t*) (vbuf + 8 * 8192);
            float* vw = vbuf + 8 * 8192 + 64;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// VERIFY's buffers, allocated once at the start (main, before any graph capture): a cudaMalloc inside a captured path
    /// fails, so the verify paths allocate nothing.
    void verify_alloc() {
        const int N = (int) m->g.n_embd, FS = (int) m->g.n_ff_shexp;
        CK(cudaMalloc((void**) &vbuf, kVerifyFloats * 4));
        // (vq: verify_q8's rows of n_embd, T <= 8, and the shared expert's gate/up quantised at FS, T <= 4)
        CK(cudaMalloc(&vq, std::max(K::native_q8_1_bytes(N, 1) * 8, K::native_q8_1_bytes(FS, 1) * 4)));
        CK(cudaMalloc((void**) &vgrp, 2048));
        CK(cudaMalloc((void**) &vx_, (size_t) 8 * N * 4));
        CK(cudaMalloc((void**) &vh_, (size_t) 8 * N * 4));
        CK(cudaMalloc((void**) &vslots, kVerifySlots * sizeof(VSlot)));
        CK(cudaMemsetAsync(vslots, 0, kVerifySlots * sizeof(VSlot), st));
    }
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= q36k::kMoeOutSmallN && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            const auto attn_prep = kv_q4 ? k_attn_prep<true> : k_attn_prep<false>;
            attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, rows_mode ? nullptr : kc[li],
                                                            rows_mode ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                                                            m->rope_base, m->eps, d_delta, mtab_now, rs, m->rope);
            if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit.  The keys come in windows of kv_window()
                // cells gathered into k16 / v16: fp32 is one window (the whole prefix, the softmax of a row in one pass);
                // q4 walks windows of kKvWindow cells and merges them by the online softmax (k_attn_win), so k16 / v16
                // and the scores stay small at any context
                const int P = (int) p0 + T;
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                const int Wn = kv_window();
                const auto kv_f16 = kv_q4 ? k_kv_f16<true> : k_kv_f16<false>;
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int c0 = 0; c0 < P; c0 += Wn) {
                    const int Pw = std::min(Wn, P - c0);
                    const int ldp = (Pw + 7) / 8 * 8;
                    const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                    kv_f16<<<grid1((int64_t) HKV * Pw * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, ctx, c0, Pw, HD, HKV);
                    for (int t0 = 0; t0 < T; t0 += tq) {
                        const int nt = std::min(tq, T - t0);
                        if (c0 >= (int) p0 + t0 + nt) continue;   // no row of the slice sees this window
                        for (int kvh = 0; kvh < HKV; ++kvh) {
                            const __half* Kh = k16 + (size_t) kvh * Pw * HD;
                            const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                            float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                            if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, Pw, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                           0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                           (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                                CUBLAS_STATUS_SUCCESS) {
                                std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                                std::exit(1);
                            }
                        }
                        const float scale = 1.0f / sqrtf((float) HD);
                        if (kv_q4)
                            k_attn_win<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, Pw, c0, (int) p0 + t0, t0, H, scale,
                                                                         att, win_m, win_l);
                        else
                            k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, Pw, (int) p0 + t0, scale);
                        for (int kvh = 0; kvh < HKV; ++kvh) {
                            const __half* Vh = v16 + (size_t) kvh * Pw * HD;
                            const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                            float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                            if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, Pw, &one, Vh, CUDA_R_16F, HD,
                                                           0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, c0 == 0 ? &zero : &one,
                                                           Oh, CUDA_R_32F, H * HD, HD, G, CUBLAS_COMPUTE_32F,
                                                           CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                                std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                                std::exit(1);
                            }
                        }
                    }
                }
                if (kv_q4) {
                    k_attn_win_norm<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, win_l, (int64_t) T * H * HD);
                    strata::kernels::fwht256_inplace_cuda(att, (int64_t) T * H, st);   // V was rotated: back before the gate
                }
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int cells = chunk_tiles * kChunkCells;
                    const int nc = graph_mode ? n_chunk : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + cells - 1) / cells);
                    const auto attn_part = kv_q4 ? k_attn_part<true> : k_attn_part<false>;
                    attn_part<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, rows_mode ? nullptr : kc[li],
                                                                       rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                       1.0f / sqrtf((float) HD), n_chunk, chunk_tiles, pacc, pm, pl, rs);
                    const auto attn_merge = kv_q4 ? k_attn_merge<true> : k_attn_merge<false>;
                    attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, cells, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        const bool ext = one || (fast_k && std::getenv("STRATA_Q36_NO_EXT") == nullptr && !verify_k && T <= kGemmT && g.n_expert == 256 && Kp == 8);
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        const int ld_r = ext ? (int) g.n_expert + 1 : (int) g.n_expert;
        if (!(!mmq_rows && route_group(L, li, hy_rows ? hy[li].slot_of_d : nullptr, T, ld_r))) {
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
            delta_rows(li, T, ld_r);
        }
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (hy_rows) {
            if (pf_predicts(li, T)) pf_predict(li, T);
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].count((size_t) x, h_cnt[li * g.n_expert + x]);
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            group_entries(L, nullptr, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        if (T <= kGemmT) {
            if (!shared_gu(L, T)) {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
            }
            mmq(L.sh_down, q8, sy, T);
        } else {
            apply(L.sh_gate, xq, sg, T);
            apply(L.sh_up, xq, su, T);
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && T > hybrid_t && use_mmq) {
            std::vector<int> seq;
            for (size_t l = 0; l < m->layers.size(); ++l)
                if (lhy(l)) seq.push_back((int) l);
            stream_begin(seq);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const bool last = l + 1 == m->layers.size();
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, last ? m->out_norm : m->layers[l + 1].attn_norm,
                  last ? hmain : h);
        }
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 <= q36k::kQ8RowBlocks || N / 32 > q36k::kQ8MaxBlocks) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const half* dpl = reinterpret_cast<const half*>(pk + (size_t) hd.n_in * hd.n_out);
            q36k::launch_q8_rows_nb<1, true>(pk, dpl, q8, logits_m, N / 32, draft_n, st);
        } else {
            q36k::launch_q8_rows_nb<1, false>(hd.d, nullptr, q8, logits_m, N / 32, draft_n, st);
        }
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    uint8_t* pf_region(int par) const { return pc_stage + (size_t) (strata::q36::kPcieSlots + par * strata::q36::kPrefetchSlots) * pc_blob; }
    /// Layer li + 1's picks are predicted at layer li: the layer ahead is a hybrid main layer and a step this wide has a share.
    bool pf_predicts(size_t li, int T) const {
        return pf_on && li + 1 < m->layers.size() && lhy(li + 1) && (int64_t) T * m->top_k <= strata::q36::kPcieMaxEntries;
    }
    /// The graph's part of the prefetch: layer li + 1's router on layer li's input, the top-k of it to the host's mapped memory
    /// (before the routing of li rings the doorbell, so the host finds both when it serves li).
    void pf_predict(size_t li, int T) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        f32mm(m->layers[li + 1].router, h, logits_pf, N, (int) m->g.n_expert, T);
        K::native_router_topk(logits_pf, d_pred, pred_w, (int) m->g.n_expert, Kp, T, st);
        k_publish_ids<<<1, 256, 0, st>>>(d_pred, d_h_pred, T * Kp);
    }
    /// The host's part, while it serves layer li: the predicted picks of li + 1 (the real rule on the predicted routing, at
    /// most kPrefetchSlots) start copying into li + 1's region.  That region was last read by the PCIe call of li - 1,
    /// which ran before the routing of li was published.  Returns the blobs issued.
    int pf_issue(size_t li, int n_ent, WireStep* ws) {
        const size_t nx = li + 1;
        const int par = (int) (nx & 1);
        const Layer& Ln = m->layers[nx];
        const int n = strata::q36::pcie_pick(h_pred, n_ent, hy[nx].slot_of, pcie_num, pf_ids[par], nullptr, strata::q36::kPrefetchSlots);
        pf_n[par] = n;
        pf_layer[par] = (int) nx;
        if (n == 0) return 0;
        void* dst[strata::q36::kPrefetchSlots];
        const void* src[strata::q36::kPrefetchSlots];
        size_t bytes[strata::q36::kPrefetchSlots];
        for (int j = 0; j < n; ++j) {
            dst[j] = pf_region(par) + (size_t) j * pc_blob;
            src[j] = Ln.hblobs + (size_t) pf_ids[par][j] * Ln.stride;
            bytes[j] = Ln.stride;
        }
        if (ws != nullptr) wire_stamp(ws->ev[2], pfs);
        // the PCIe copies as one cudaMemcpyBatchAsync per group, on unless STRATA_DMA_BATCH=0 (q36's default is kPcieDmaBatch;
        // Flash-Next's default is its own).
        CK(strata::core::copy_blobs(dst, src, bytes, (size_t) n, pfs, strata::core::dma_batch_mode(kPcieDmaBatch)));
        if (ws != nullptr) {
            wire_stamp(ws->ev[3], pfs);
            ws->prefetch = n;
            ws->bytes += (size_t) n * Ln.stride;
        }
        CK(cudaEventRecord(ev_pf[par], pfs));
        (void) cudaStreamQuery(pfs);   // submits the copies to the GPU (on WDDM the driver batches until asked)
        return n;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        if (cap_steps) {
            cap_steps->push_back({li, T});
            cap_ring = (uint32_t) cap_steps->size();
            K::doorbell_publish(h, ids, nullptr, (int64_t) T * N, n_ent, d_h_h, d_h_ids, nullptr, db.d_seq, st);
        } else {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
        }
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The CPU half's arithmetic: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's
    /// formats), from the routing and the hidden rows the GPU handed over (h_ids, h_h); their rows land in cpu_out.
    /// Returns how many entries it computed.
    int hy_cpu_compute(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        const auto t0 = Clock::now();
        const bool pa_on = (pcie_auto.on || stats_on) && !warming;   // PcieAuto's clocks (and the stats'): none once it has settled
        WireStep ws;
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        // the PCIe share: its experts leave the pool's jobs; their copies start now and run beside the pool, and the
        // plan the graph's PCIe call reads is written before the layer is answered (no share: an empty plan)
        const auto l0 = pa_on ? Clock::now() : Clock::time_point();
        int32_t picked[strata::q36::kPcieSlots];
        int want = 0;
        const int n_pcie = pcie_on ? strata::q36::pcie_pick(h_ids, n_ent, H.slot_of, pcie_num, picked, &want) : 0;
        pc_want += want;
        pc_sent += n_pcie;
        std::fill(pc_mark.begin(), pc_mark.end(), 0);
        // a pick whose blob the prefetch already put in this layer's region is not copied again: the plan's group points at it
        const int par = (int) (li & 1);
        int pf_slot[strata::q36::kPcieSlots];
        const int hits = strata::q36::pcie_prefetch_match(picked, n_pcie, pf_ids[par], pf_layer[par] == (int) li ? pf_n[par] : 0, pf_slot);
        void* cdst[strata::q36::kPcieSlots];
        const void* csrc[strata::q36::kPcieSlots];
        size_t cbytes[strata::q36::kPcieSlots];
        unsigned long long blob_ptr[strata::q36::kPcieSlots];
        int n_copy = 0;
        for (int k = 0; k < n_pcie; ++k) {
            pc_mark[(size_t) picked[k]] = 1;
            const uint8_t* at = pf_slot[k] >= 0 ? pf_region(par) + (size_t) pf_slot[k] * pc_blob : pc_stage + (size_t) k * pc_blob;
            blob_ptr[k] = (unsigned long long) (uintptr_t) at;
            if (pf_slot[k] >= 0) continue;
            cdst[n_copy] = pc_stage + (size_t) k * pc_blob;
            csrc[n_copy] = L.hblobs + (size_t) picked[k] * L.stride;
            cbytes[n_copy] = L.stride;
            ++n_copy;
        }
        if (n_copy > 0) {
            if (pa_on) wire_stamp(ws.ev[0], pcs);
            CK(strata::core::copy_blobs(cdst, csrc, cbytes, (size_t) n_copy, pcs, strata::core::dma_batch_mode(kPcieDmaBatch)));
            if (pa_on) {
                wire_stamp(ws.ev[1], pcs);
                ws.demand = n_copy;
                ws.bytes = (size_t) n_copy * L.stride;
            }
            (void) cudaStreamQuery(pcs);   // submits the copies to the GPU (on WDDM the driver batches until asked)
        }
        const int n_pf = pf_predicts(li, T) ? pf_issue(li, n_ent, pa_on ? &ws : nullptr) : 0;
        if (pcie_on) strata::q36::pcie_plan(h_ids, n_ent, Kp, picked, n_pcie, blob_ptr, pc_plan);
        double launch_ms = 0.0;
        if (pa_on) {
            // every step is read, one with no copy too (no events, so it resolves at once): its host time is launch time
            launch_ms = ms_since(l0);
            ws.chosen = n_pcie;
            ws.launch_ms = launch_ms;
            wire_pending.push_back(ws);
        }
        int n_pcie_ent = 0;
        const bool q2 = C::q2_native_kernels(H.fmt.gu_type);   // the Q2_0 gate/up kernels read an ActQ, not nact
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer && strata::q36::first_of_row(h_ids, e, Kp)) H.count((size_t) x, 1.0);
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            if (pcie_on && pc_mark[(size_t) x]) {
                ++n_pcie_ent;
                continue;
            }
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                if (q2) C::act_quant_any(h_h + (size_t) t * N, N, actq[(size_t) t]);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            if (q2) j.act[j.nt] = &actq[(size_t) t];
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss - n_pcie_ent;
        hy_pcie += n_pcie_ent;
        hy_gpu += n_ent - n_miss;
        if (stats_on) {   // the resident experts read once per layer step (the grouped kernel), the staged blobs; the pool counts its own
            stat_hit_mark.assign((size_t) g.n_expert, 0);
            for (int e = 0; e < n_ent; ++e)
                if (!h_miss[e] && !stat_hit_mark[(size_t) h_ids[e]]) {
                    stat_hit_mark[(size_t) h_ids[e]] = 1;
                    stat_acc.gpu_bytes += (long long) L.stride;
                }
            stat_acc.pcie_bytes += (long long) n_pcie * (long long) L.stride;
            stat_acc.pcie_demand += n_copy;
            stat_acc.pf_issued += n_pf;
            stat_acc.pf_hits += hits;
        }
        if (n_miss == 0) return 0;
        if (!jobs.empty()) {
            const double built_ms = pa_on ? ms_since(t0) - launch_ms : 0.0;
            const double rows_ms = pa_on ? pool->ms_multi_gu + pool->ms_multi_q + pool->ms_multi_down : 0.0;
            pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
            hy_cpu_ms += ms_since(t0);
            hy_cpu_jobs += (long long) jobs.size();
            if (pa_on) pa_cpu_ms += built_ms + pool->ms_multi_gu + pool->ms_multi_q + pool->ms_multi_down - rows_ms;
        }
        if (n_copy > 0) CK(cudaStreamSynchronize(pcs));   // the share is in staging before the layer is answered
        if (hits > 0) CK(cudaEventSynchronize(ev_pf[par]));
        return n_miss;
#else
        (void) li;
        (void) T;
        return 0;
#endif
    }
    /// The CPU half: the entries without a slot on the pool computed, their rows uploaded and merged into the entries'
    /// rows.  In a capture nothing runs on the host: the rows are copied from the mapped memory once the host has
    /// answered the layer's ring (hy_serve), and the split into hit and miss entries is made on the device.
    void moe_hybrid_cpu(size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        if (cap_steps) {
            if (n_ent > kHySplitMax) {
                std::fprintf(stderr, "strata-q36: a hybrid layer of %d entries does not fit the graph's split kernel\n", n_ent);
                std::exit(1);
            }
            k_hy_split<<<1, 128, 0, st>>>(ids, n_ent, hy[li].slot_of_d, d_miss, d_hit_rows, d_n_hit);
            K::copy_rows_from_mapped_after(db.d_flag, cap_ring, ecpu, d_cpu_out, n_ent, N, d_hit_rows, d_n_hit, st);
            k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
            if (pcie_on) pcie_call(layer_at((int) li), n_ent);
            return;
        }
        const auto w0 = stats_on ? Clock::now() : Clock::time_point();
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        if (stats_on) stat_wait(li, ms_since(w0));
        const auto s0 = stats_on ? Clock::now() : Clock::time_point();
        const int served = hy_cpu_compute(li, T);
        if (stats_on) stat_acc.serve_ms += ms_since(s0);
        if (served == 0) return;
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
        if (pcie_on) pcie_call(layer_at((int) li), n_ent);
    }
    /// The PCIe share's expert call, after the CPU rows are merged: the plan the host wrote (q36_pcie.hpp) to the device,
    /// then the grouped expert kernel (the one of the resident experts, same bits) over its groups, whose blobs the host
    /// copied to staging.  It writes the rows of those entries over what the merge put there (the host left the pool's rows
    /// of those entries unwritten).  With no share the plan has no group and the call does nothing.
    void pcie_call(const Layer& L, int n_ent) {
        using namespace strata::q36;
        if (n_ent > kPcieMaxEntries) return;   // pcie_pick sends nothing for a step this wide
        K::copy_i32_from_mapped(d_pc, d_pc_plan, kPlanWords, st);
        K::native_expert_grouped(L.L, reinterpret_cast<const unsigned long long*>(d_pc + kPlanPtr), d_pc + kPlanStart,
                                 d_pc + kPlanGroups, d_pc + kPlanDst, d_pc + kPlanTok, kPcieSlots, n_ent, xq, escr, eparts,
                                 st, kPcieGroupRows);
    }
    /// The host waited `ms` for the GPU to publish layer `li`'s routing (stats only).
    void stat_wait(size_t li, double ms) {
        if (stat_acc.wait_layer_ms.size() <= li) stat_acc.wait_layer_ms.resize(li + 1, 0.0);
        stat_acc.wait_layer_ms[li] += ms;
        stat_acc.wait_ms += ms;
        ++stat_acc.steps;
    }
    /// After the launch: answers the graph's hybrid layers in capture order.  Ring k is the k-th layer's: the GPU has
    /// published its routing when the sequence word reaches k, the pool's rows are in place when the flag is set to k.
    void hy_serve(const std::vector<HyStep>& steps) {
        (void) cudaStreamQuery(st);   // submits the launch to the GPU (on WDDM the driver batches until it is asked)
        uint32_t ring = 0;
        for (const HyStep& step : steps) {
            ++ring;
            const auto w0 = stats_on ? Clock::now() : Clock::time_point();
            const auto w = db.wait_ring(ring, [&] { return cudaStreamQuery(st) == cudaErrorNotReady; }, std::chrono::seconds(20));
            if (stats_on) stat_wait(step.li, ms_since(w0));
            if (w != strata::q36::RingWait::Rang) {
                db.release();   // no wait kernel may outlive the engine
                const cudaError_t qe = cudaStreamQuery(st);   // a stream in error ends the wait too: say the CUDA error
                std::fprintf(stderr, "strata-q36: the hybrid layer %zu %s\n", step.li,
                             w == strata::q36::RingWait::Ended
                                 ? (qe != cudaSuccess ? cudaGetErrorString(qe) : "never rang (the graph ended)")
                                 : "did not ring in 20 s");
                std::exit(1);
            }
            const auto s0 = stats_on ? Clock::now() : Clock::time_point();
            hy_cpu_compute(step.li, step.T);
            if (stats_on) stat_acc.serve_ms += ms_since(s0);
            db.answer(ring);
        }
        CK(cudaStreamSynchronize(st));
    }

    /// The slots to the experts the routing used most: the replacements with the best gain over all layers (q36_rebalance.hpp),
    /// at most max_swaps of them, between forwards (the copies are on the engine's stream).  Every layer's counts decay
    /// afterwards, those of the layers it did not touch too.  Returns the replacements made.
    int rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return 0;
        std::vector<strata::q36::LayerCounts> views(hy.size());
        for (size_t li = 0; li < hy.size(); ++li) {
            const int S = layer_at((int) li).n_slots;
            if (S > 0 && S < m->g.n_expert) views[li] = {&hy[li].score, &hy[li].slot_of};
        }
        const std::vector<strata::q36::SlotSwap> swaps = strata::q36::choose_swaps(views, max_swaps);
        std::vector<char> changed(hy.size(), 0);
        for (const strata::q36::SlotSwap& w : swaps) {
            Hy& H = hy[(size_t) w.layer];
            const Layer& L = layer_at(w.layer);
            const int s = H.slot_of[(size_t) w.out];
            H.slot_of[(size_t) w.out] = -1;
            H.slot_of[(size_t) w.in] = s;
            CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) w.in * L.stride, L.L.bytes,
                               cudaMemcpyHostToDevice, st));
            changed[(size_t) w.layer] = 1;
            ++hy_swaps;
        }
        for (size_t li = 0; li < hy.size(); ++li) {
            if (changed[li])
                CK(cudaMemcpyAsync(hy[li].slot_of_d, hy[li].slot_of.data(), (size_t) m->g.n_expert * 4,
                                   cudaMemcpyHostToDevice, st));
            for (double& v : hy[li].score) v *= strata::q36::kRebalanceDecay;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
        return (int) swaps.size();
    }

    /// --expert-profile-save: what the run learned, as a profile of the main layers - the experts in the slots first, then
    /// the rest by the routing counted since the start, then by the profile the run started from.  Nothing is written
    /// when no routing was counted (the file on disk stays the best knowledge there is).
    void save_profile(const std::string& path) {
        if (!m->hybrid) {
            std::fprintf(stderr, "strata-q36: --expert-profile-save %s: every expert is in VRAM, nothing to learn\n", path.c_str());
            return;
        }
        const int64_t nl = (int64_t) m->layers.size(), ne = m->g.n_expert;
        std::vector<uint8_t> resident((size_t) (nl * ne), 1);
        std::vector<double> heat((size_t) (nl * ne), 0.0);
        double total = 0;
        for (size_t li = 0; li < (size_t) nl && li < hy.size(); ++li) {
            const Hy& H = hy[li];
            if (H.slot_of.empty()) continue;   // a layer with every expert in VRAM
            for (int64_t e = 0; e < ne; ++e) {
                resident[li * (size_t) ne + (size_t) e] = H.slot_of[(size_t) e] >= 0;
                heat[li * (size_t) ne + (size_t) e] = H.heat[(size_t) e];
                total += H.heat[(size_t) e];
            }
        }
        if (total <= 0) {
            std::fprintf(stderr, "strata-q36: --expert-profile-save %s: no routing was counted, not written\n", path.c_str());
            return;
        }
        std::string err;
        if (!strata::core::write_expert_profile(path, nl, ne, strata::core::rank_learned_profile(nl, ne, resident, heat, profile_prior), err))
            std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        else
            std::fprintf(stderr, "strata-q36: expert profile saved to %s (%lld layers x %lld experts, %lld slot replacements this run)\n",
                         path.c_str(), (long long) nl, (long long) ne, hy_swaps);
    }

    void shrink_att(size_t floats) {
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
        hy_steps.clear();
    }

    /// The hybrid layers need no re-capture after a rebalance: the graph holds the slot tables' device pointers
    /// (Hy::slot_of_d) and the slot blobs' addresses, and a rebalance only rewrites what they point to, between launches.
    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            std::vector<HyStep> steps;
            cudaGraph_t graph = nullptr;
            cap_steps = &steps;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            cap_steps = nullptr;
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
            if (!steps.empty()) hy_steps[gx] = std::move(steps);
        }
        const auto served = hy_steps.find(gx);
        if (served == hy_steps.end()) {
            CK(cudaGraphLaunch(gx, st));
            return;
        }
        db.arm();   // the previous launch has drained (hy_serve ends with a sync): no wait kernel still spins on the old flag
        CK(cudaGraphLaunch(gx, st));
        hy_serve(served->second);
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(cudaFree(d_img));
            CK(cudaMalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        const int N = (int) m->g.n_embd;
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            verify_flush();
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) rebalance_after_prompt();
                return i + T;
            }
        }
        if (m->hybrid) rebalance_after_prompt();
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key], body);
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        if (stats_on && G > 1) ++stat_mixed;
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        verify_flush();   // (the window's graph is out of its capture: the host may read the VERIFY slots)
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            pcie_window();
            if (++win_count % kRebalanceEvery == 0) rebalance(8);
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    /// A draft whose row had no finite logit (-1) is proposed as token 0: a token id is never -1 on its way into a window,
    /// and the window keeps the draft only when its own pick is 0 too, so the text is the plain one.
    static int32_t draft_token(int32_t t) { return t < 0 ? 0 : t; }
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = draft_token(hrow[32 + r]);
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t row = (size_t) ctx * kv_cell_bytes();
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) a.pos * kv_cell_bytes(), (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], row, a.vc[l], row, (size_t) a.pos * kv_cell_bytes(), (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        if (dst == cur) load_cur();
    }
    int hybrid_swaps_after_prompt = 2048;
    /// The prompt's routing moves the slots at once; the decode windows' rebalances add up in hy_swaps (the request's log line).
    void rebalance_after_prompt() {
        const int n = rebalance(hybrid_swaps_after_prompt);
        if (!warming)
            std::fprintf(stderr, "strata-q36: rebalance after the prompt: %d slot replacements (%lld since the start)\n", n,
                         hy_swaps);
    }
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    /// `sample_row` (a sampled request): the token plain decoding would pick after the window's row t, from that row's
    /// logits (host_logits_row) and the drafts before it; a draft is kept when it equals the pick of the row before it.
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr,
                  const std::function<int32_t(int, const int32_t*)>& sample_row = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = draft_token(hpin[2]);
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = draft_token(hpin[2]);
        }
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        verify_flush();
        int a;
        if (sample_row) {   // sampled: row t's pick is the draw of its position over its own history (q36_sampling.hpp)
            a = strata::q36::spec_accept(drafts.data(), D, picks, [&](int t) { return sample_row(t, drafts.data()); });
        } else {
            a = 0;
            while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
            for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        }
        if (cancel != nullptr && cancel->load()) a = 0;   // (picks[0] is row 0's pick either way)
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        if (tmax > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) {
            std::fill(H.score.begin(), H.score.end(), 0.0);
            std::fill(H.heat.begin(), H.heat.end(), 0.0);
        }
        hy_cpu = hy_gpu = hy_swaps = hy_pcie = hy_cpu_jobs = pc_want = pc_sent = 0;
        hy_cpu_ms = pa_cpu_ms = 0;
        wire = wire_mark = WireCount();
        wire_steps = wire_logs = 0;
        pcie_auto.mark(0.0, 0);
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, double pcie_frac, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0, max_main = 0, pc_blob = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
        if (L != &m.mtp) max_main = std::max(max_main, L->stride * (size_t) ne);
        pc_blob = std::max(pc_blob, L->stride);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 768 << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    const bool fits_all = total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack);
    // the hybrid mode, where giving the copies back does not make every expert fit: their room becomes slots, and each
    // slot moves one expert of its layer from the CPU pool to the GPU.  The copies speed the dense decode up by at most
    // kPackedGain (the whole token counted as dense MMVQ, which a hybrid token is not: the pool is most of it); the slots
    // cut the pool's missed experts by (slots gained) / (experts not in a slot), if the routing were uniform (a ranked
    // layer misses less, but the marginal slot is the coldest expert).  Both bounds lean toward keeping the copies; the
    // copies go when even so the slots win.  Ornith (IQ3_XXS / IQ4_XS, 13.55 GiB free): 3.80 GiB of copies = 55 -> 76 slots
    // per layer of 256, 10.4% fewer missed experts against 2.3%.  No profile or timing: a fixed rule from the two numbers.
    constexpr double kPackedGain = 0.023;   // arch_defaults.cpp: STRATA_Q8_PACKED, "+2.3% (S26)"
    // the PCIe share's staging area (q36_pcie.hpp): kPcieStageBlobs blobs of the widest layer's stride (the layer's own slots and the
    // prefetch's two regions); --pcie-frac 0 is off.  The
    // estimate counts it before main_hybrid is known; the real size (pcie_bytes) drops it when only the MTP block is hybrid
    const size_t pcie_stage = pcie_frac != 0.0 ? (size_t) strata::q36::kPcieStageBlobs * pc_blob + kMmqTail : 0;
    const size_t fixed_est = (E.use_mmq ? max_layer + max_main + 2 * kMmqTail : 0) + pcie_stage;
    const auto slots_in = [&](size_t room) {
        return room > reserve + fixed_est ? std::min<size_t>((size_t) ne, (room - reserve - fixed_est) / per_slot) : (size_t) 0;
    };
    const size_t s_keep = slots_in(fr), s_free = slots_in(fr + packed_bytes);
    const bool slots_win = s_free > s_keep && s_keep < (size_t) ne &&
                           (double) (s_free - s_keep) / (double) ((size_t) ne - s_keep) > kPackedGain;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 && (fits_all || slots_win) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        if (fits_all)
            std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                         "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                         (double) packed_bytes / (1 << 30));
        else
            std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back for expert slots: %zu "
                         "instead of %zu per layer of %lld (the copies speed the dense decode up by 2.3%% at most, the "
                         "slots cut the missed experts by %.1f%%; STRATA_Q36_KEEP_PACKED=1 keeps the copies)\n",
                         (double) packed_bytes / (1 << 30), s_free, s_keep, (long long) ne,
                         100.0 * (double) (s_free - s_keep) / (double) ((size_t) ne - s_keep));
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-q36: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-q36: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    // never below kGemmT: a batch of 2 .. kGemmT rows past hybrid_t would take the resident-experts path, which reads the
    // blobs as if every expert were in VRAM
    if (hybrid_t < Engine::kGemmT)
        std::fprintf(stderr, "strata-q36: --hybrid-rows %d raised to %d (a batch of up to %d rows always runs on the slots and the pool)\n",
                     hybrid_t, Engine::kGemmT, Engine::kGemmT);
    E.hybrid_t = std::min(std::max(hybrid_t, Engine::kGemmT), E.tmax);
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    // the graphs (a decode step, a verify window) run the hybrid layers through the doorbell; a window over hybrid_t
    // rows would stream the layer's experts with host syncs in the middle, which a capture cannot hold
    if (E.main_hybrid && E.hybrid_t < Engine::kVerify) E.use_graph = false;
    // the two staging buffers: a streamed forward over the main layers alternates between them, and the MTP block (the
    // widest layer where its gate / up are Q4_K: 1.24 GiB against 0.97 GiB for a main layer on Ornith) is streamed alone,
    // so it only ever takes the first (stream_begin with one layer): the second holds a main layer
    const size_t staging = E.use_mmq && E.main_hybrid ? max_layer + max_main + 2 * kMmqTail : 0;
    // the main layers' hybrid path only (the MTP block alone runs few rows)
    const size_t pcie_bytes = E.main_hybrid ? pcie_stage : 0;
    // no --pcie-frac: the share the link's probed bandwidth gives (as Flash-Next's native packs: 0.55 on a 20 GB/s link or
    // faster), then refined from the pool's speed (PcieAuto); a probe that fails keeps the 0.55
    double link_gbps = -1.0;
    std::string bursts;
    if (pcie_bytes > 0 && pcie_frac < 0.0) link_gbps = strata::core::probe_pcie_h2d_gbps(&bursts);
    size_t slot_bytes = budget2 > staging + pcie_bytes ? budget2 - staging - pcie_bytes : 0;
    if (slot_bytes + staging + pcie_bytes + ((size_t) 256 << 20) > fr)
        slot_bytes = fr > staging + pcie_bytes + ((size_t) 512 << 20) ? fr - staging - pcie_bytes - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = staging + pcie_bytes + Tn * Kn * Nn * 4 + Tn * Kn * 4;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies)
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    if (E.stats_on)
        std::fprintf(stderr, "strata-q36: stats: the expert arena (%.2f GiB) is %s\n", (double) host / (1024.0 * 1024.0 * 1024.0),
                     pinned ? "cudaHostAlloc memory (pinned, default flags, no large pages)"
                            : "malloc memory (cudaHostAlloc failed, not pinned)");
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        // the slots hold the layer's best-ranked experts of the profile, else experts 0 .. S - 1 (the profile ranks the
        // main layers only: the MTP block, past them, always takes the default order)
        const std::vector<int> placed = strata::q36::profile_slot_experts(E.profile_prior, (int) li, S, (int) ne);
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            for (int s = 0; s < S; ++s)
                CK(cudaMemcpy(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) placed[(size_t) s] * L.stride, L.stride,
                              cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        for (int s = 0; s < S; ++s) H.slot_of[(size_t) placed[(size_t) s]] = s;
        H.score.assign((size_t) ne, 0.0);
        H.heat.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        // IQ3_XXS gate/up for a lone token and IQ4_XS down on AVX-512, and an IQ4_NL down for a lone token on AVX-2: the same
        // rows alone and in a window.  Opt-in: on the real AVX-512 CPU of the Ornith box they run slower than ggml-cpu's dots
        // (native_expert_ff1024_test --bench), so the default is the Flash-Next selection
        static const bool one_token = std::getenv("STRATA_Q36_ONE_TOKEN_CPU") != nullptr;
        H.fmt.one_token_512 = one_token;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    // mapped pinned: the doorbell's payload and the pool's rows have one address for the host and another for the device
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**) &E.d_h_h, E.h_h, 0));
    CK(cudaHostGetDevicePointer((void**) &E.d_cpu_out, E.cpu_out, 0));
    CK(cudaHostGetDevicePointer((void**) &E.d_h_ids, E.h_ids, 0));
    uint32_t *db_words = nullptr, *d_db_words = nullptr;
    CK(cudaHostAlloc((void**) &db_words, strata::q36::HyDoorbell::kWords * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**) &d_db_words, db_words, 0));
    std::memset(db_words, 0, strata::q36::HyDoorbell::kWords * 4);
    E.db.bind(db_words, d_db_words);
    CK(cudaMalloc(&E.d_hit_rows, T * Kp * 4));
    CK(cudaMalloc(&E.d_n_hit, 4));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, T * Kp * N * 4));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
#if defined(STRATA_Q36_CPU_POOL)
    E.actq.resize((size_t) T);
#endif
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            const size_t bytes = (b == 0 ? max_layer : max_main) + kMmqTail;
            CK(cudaMalloc(&E.stage[b], bytes));
            CK(cudaMemset(E.stage[b], 0, bytes));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(strata::q36::pool_workers(threads, (int) C::detect_cpu_topology(true).worker_cores.size()));
    if (pcie_frac == 0.0) {
        std::fprintf(stderr, "strata-q36: PCIe share off (--pcie-frac 0)\n");
    } else if (pcie_bytes > 0 && !pinned) {
        std::fprintf(stderr, "strata-q36: the PCIe share is off: the experts are in pageable host memory\n");
    } else if (pcie_bytes > 0) {
        E.pcie_on = true;
        E.pc_blob = pc_blob;
        CK(cudaMalloc(&E.pc_stage, pcie_bytes));
        CK(cudaMemset(E.pc_stage, 0, pcie_bytes));
        CK(cudaStreamCreateWithFlags(&E.pcs, cudaStreamNonBlocking));
        CK(cudaEventCreate(&E.ev_wire0));
        CK(cudaEventRecord(E.ev_wire0, E.pcs));
        CK(cudaEventSynchronize(E.ev_wire0));
        E.pf_on = std::getenv("STRATA_Q36_NO_PREFETCH") == nullptr && m.layers.size() > 1;   // the staging area is the same size without it
        if (E.pf_on) {
            CK(cudaStreamCreateWithFlags(&E.pfs, cudaStreamNonBlocking));
            for (cudaEvent_t& ev : E.ev_pf) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            CK(cudaMalloc(&E.logits_pf, T * (size_t) ne * 4));
            CK(cudaMalloc(&E.pred_w, T * Kp * 4));
            CK(cudaMalloc(&E.d_pred, T * Kp * 4));
            CK(cudaHostAlloc((void**) &E.h_pred, T * Kp * 4, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void**) &E.d_h_pred, E.h_pred, 0));
        }
        CK(cudaHostAlloc((void**) &E.pc_plan, (size_t) strata::q36::kPlanWords * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.d_pc_plan, E.pc_plan, 0));
        std::memset(E.pc_plan, 0, (size_t) strata::q36::kPlanWords * 4);
        CK(cudaMalloc(&E.d_pc, (size_t) strata::q36::kPlanWords * 4));
        E.pc_mark.assign((size_t) ne, 0);
        constexpr double kBase = 0.55;   // the share measured on x16 links (Flash-Next's native packs)
        double frac = kBase;
        if (pcie_frac >= 0.0) {
            frac = std::min(1.0, pcie_frac);
        } else if (link_gbps > 0.0) {
            frac = strata::core::pcie_frac_for_gbps(link_gbps, kBase);
            double sum = 0.0;
            for (const Layer& L : m.layers) sum += (double) L.L.bytes;
            E.pcie_auto.on = true;
            E.pcie_auto.min_share = 0.0;   // (Flash-Next's 0.10 floor would keep sending a share a slow link cannot carry)
            E.pcie_auto.max_share = 0.95;  // (Flash-Next's 0.60 cap would leave a large part of Ornith's experts to the CPU pool)
            E.pcie_auto.pool_factor = 1.0; // (Flash-Next's 1.85 is fitted to a pool time that holds the launches and the job build; here they are counted apart)
            E.pcie_auto.link_gbps = link_gbps;
            E.pcie_auto.blob_bytes = sum / (double) m.layers.size();
        }
        E.pcie_num = strata::core::pcie_num_of(frac);
        if (pcie_frac >= 0.0)
            std::fprintf(stderr, "strata-q36: PCIe share %.2f of the missed experts (--pcie-frac)\n", frac);
        else if (link_gbps > 0.0)
            std::fprintf(stderr, "strata-q36: PCIe probe: %.1f GB/s host->device (best of %s) -> PCIe share %.2f of the missed "
                         "experts, refined from the CPU pool's speed (--pcie-frac N fixes it, 0 turns it off)\n", link_gbps,
                         bursts.c_str(), frac);
        else
            std::fprintf(stderr, "strata-q36: PCIe probe failed -> PCIe share %.2f of the missed experts (--pcie-frac N fixes it, "
                         "0 turns it off)\n", frac);
    }
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
using strata::q36::Sampling;
using strata::q36::draw_rng;
using strata::q36::sample;

// the ids of a comma or space separated list; false when a token is not an int32 integer: the list is refused, not cut short
bool parse_ids(const char* s, std::vector<int32_t>& v) {
    v.clear();
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s || (*e != '\0' && *e != ',' && *e != ' ') || x < INT32_MIN || x > INT32_MAX) return false;
        v.push_back((int32_t) x);
        s = e;
    }
    return true;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into that slot's
/// session - the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one
/// window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the
/// control lines.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }

    /// A slot whose row has no finite logit ends with no token drawn: BDONE ... error, and it leaves the batch (the others go on).
    void end_no_logit(int b, Slot& sl, Engine::Sess& ss) {
        std::printf("BDONE %d %lld error %.1f\n", b, (long long) sl.produced, ms_since(sl.t0));
        sl.active = false;
        ss.active = false;
        --n_active;
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                std::mt19937_64 rng = draw_rng(sl.smp.seed, (uint64_t) ss.pos);   // (the window did not move ss.pos: the fed token's cell)
                y0 = sample(lg, sl.smp, sl.hist, rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                if (y < 0) {   // no finite logit: the slot ends here with no token drawn, the other slots go on
                    end_no_logit(b, sl, ss);
                    done = true;
                    break;
                }
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                std::mt19937_64 rng = draw_rng(sl.smp.seed, (uint64_t) (ss.pos - 1));   // (the step moved ss.pos: the fed token's cell is the one before)
                y = sample(lg, sl.smp, sl.hist, rng);
            } else {
                y = E.rows_pick[r];
            }
            if (y < 0) {   // no finite logit: the slot ends with no token drawn, the other slots go on
                end_no_logit(b, sl, ss);
                continue;
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids, imgs);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        bool bad_repeat = false;   // a repeat penalty of 0 or below (or not finite) inverts the logits: refused, not run
        std::vector<uint64_t> img_keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") {
                smp.repeat = std::strtof(v, nullptr);
                bad_repeat = !std::isfinite(smp.repeat) || smp.repeat <= 0.0f;
            }
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        if (bad_repeat) {
            std::printf("ERR bad request: penalty_repeat must be a finite number above 0\n");
            std::fflush(stdout);
            return;
        }
        std::vector<int32_t> ids;
        if (!parse_ids(p, ids)) {
            std::printf("ERR bad request: the ids must be integers separated by commas or spaces\n");
            std::fflush(stdout);
            return;
        }
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back)
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        const long long hc0 = E.hy_cpu, hg0 = E.hy_gpu, hs0 = E.hy_swaps, hp0 = E.hy_pcie;   // the hybrid counters at the start of the decode
        // the expert union is the engine's ring: it is reset only when no other session decodes (another live session's ring is
        // kept), and a request started beside one is counted in stat_mixed, which leaves its stats line out (single_decoder)
        bool alone = true;
        for (int i = 0; i < E.n_sess; ++i) alone = alone && (i == s || !E.sess[(size_t) i].active);
        if (E.stats_on && !alone) ++E.stat_mixed;
        const long long mixed0 = E.stat_mixed;
        const strata::q36::DecodeStats stat0 = E.stats_on ? E.stat_snapshot() : strata::q36::DecodeStats();
        if (E.stats_on && alone) E.stat_union.reset(E.m->layers.size(), (int) E.m->top_k);
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // requests alone with the MTP block: speculative steps of mtp_d drafts, verified (greedy: the greedy text; sampled:
        // each position's own draw, so the text is the one plain sampling gives for the seed); requests beside others,
        // one token at a time, drawn the same way (a seed gives one text alone or in a batch)
        const bool spec = E.mtp_d > 0 && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                std::mt19937_64 r = draw_rng(smp.seed, (uint64_t) (E.pos - 1));
                nx = sample(lg, smp, hist, r);
            }
            if (nx < 0) {   // no finite logit: the request ends in ERR, no slot taken
                std::printf("ERR %s\n", kNoLogitMsg);
                std::fflush(stdout);
                return;
            }
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        // the draw after the token at position p is draw_rng(seed, p): a sampled step's pick, and a verify window's row t
        // (the token fed at E.pos + t, the penalty history running on through the drafts before it)
        const std::function<int32_t(int, const int32_t*)> sample_row = [&](int t, const int32_t* dr) {
            E.host_logits_row(t, lg);
            std::mt19937_64 r = draw_rng(smp.seed, (uint64_t) (E.pos + t));
            return (int32_t) sample(lg, smp, hist, r, dr, (size_t) t);
        };
        bool known = false;   // nx came out of a verify window already
        bool no_logit = false;   // a pick had no finite logit (-1): the request ends in ERR, that pick is not emitted
        for (;;) {
            if (smp.needs_logits() && !known) {
                E.host_logits(lg);
                std::mt19937_64 r = draw_rng(smp.seed, (uint64_t) (E.pos - 1));
                nx = sample(lg, smp, hist, r);
            }
            known = false;
            auto emit = [&](int t) {   // false: the answer ends here
                if (t < 0) {
                    no_logit = true;
                    return false;
                }
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop,
                                          smp.needs_logits() ? sample_row : std::function<int32_t(int, const int32_t*)>());
                offered += E.mtp_d;
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                known = true;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        if (no_logit) {
            std::printf("ERR %s\n", kNoLogitMsg);
            std::fflush(stdout);
            return;
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
        if (E.m->hybrid) {   // the hit rate of the slots over this request's decode
            const long long hg = E.hy_gpu - hg0, hp = E.hy_pcie - hp0, hcpu = E.hy_cpu - hc0, hn = hg + hp + hcpu;
            const auto pct = [hn](long long v) { return hn ? 100.0 * (double) v / (double) hn : 0.0; };
            std::fprintf(stderr, "strata serve: session %d: experts on the GPU %.1f%% (%lld of %lld rows), over PCIe %.1f%% "
                         "(%lld), on the CPU %.1f%% (%lld), %lld slot replacements, PCIe share %.2f\n", s, pct(hg), hg, hn,
                         pct(hp), hp, pct(hcpu), hcpu, E.hy_swaps - hs0, E.pcie_num / 256.0);
            if (E.stats_on && strata::q36::single_decoder(alone, mixed0, E.stat_mixed)) {
                const strata::q36::DecodeStats d = E.stat_snapshot() - stat0;
                std::fprintf(stderr, "%s\n%s\n%s\n", strata::q36::format_decode_stats(s, d, produced).c_str(),
                             strata::q36::format_wait_layers(s, d).c_str(), strata::q36::format_union(s, E.stat_union).c_str());
            }
        }
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                if (l == "QUIT") break;   // the server ends the engine (an unload too): the run returns and saves the profile
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    Serve sv(E);
    return sv.run();
}

}  // namespace

/// The rope scaling of the run into m.rope, with strata's resolution (generate.cpp): an explicit option over the GGUF's
/// rope.scaling.* keys over none; YaRN with ext factor 1, attention factor 1, betas 32 / 1.  Before any graph capture:
/// the kernels take the constants as arguments.
static bool resolve_rope(Model& m, const std::string& type_opt, double scale_opt, double orig_opt, int ctx) {
    using RST = strata::kernels::RopeScalingType;
    strata::kernels::RopeScaling cfg;
    cfg.freq_base = m.rope_base;
    if (type_opt != "" && type_opt != "none" && type_opt != "linear" && type_opt != "yarn") {
        std::fprintf(stderr, "strata-q36: --rope-scaling must be none, linear or yarn (got '%s')\n", type_opt.c_str());
        return false;
    }
    if (!std::isfinite(scale_opt) || !std::isfinite(orig_opt) || scale_opt < 0 || orig_opt < 0 ||
        (scale_opt != 0 && scale_opt < 1.0)) {
        std::fprintf(stderr, "strata-q36: --rope-scale must be >= 1 and --rope-orig-ctx positive (0 = the model's)\n");
        return false;
    }
    const std::string& type = type_opt.empty() ? m.rope_type : type_opt;
    if (type == "linear") cfg.type = RST::Linear;
    else if (type == "yarn") cfg.type = RST::YaRN;
    else if (!type.empty() && type != "none") {
        std::fprintf(stderr, "strata-q36: the model carries rope.scaling.type '%s' - none, linear or yarn only\n", type.c_str());
        return false;
    }
    if (cfg.type == RST::None && (scale_opt > 0 || orig_opt > 0)) {
        std::fprintf(stderr, "strata-q36: --rope-scale and --rope-orig-ctx need --rope-scaling linear or yarn (the run has "
                     "no rope scaling: %s)\n", type_opt == "none" ? "--rope-scaling none" : "no --rope-scaling option and "
                     "no rope.scaling.type in the GGUF, or none");
        return false;
    }
    if (scale_opt > 0) cfg.factor = scale_opt;           // an explicit factor, 1 included (generate.cpp's rule)
    else if (m.rope_factor > 1.0) cfg.factor = m.rope_factor;
    if (orig_opt > 0) cfg.orig_ctx = orig_opt;
    else if (m.rope_orig_ctx >= 1) cfg.orig_ctx = m.rope_orig_ctx;
    cfg.ext_factor = cfg.type == RST::YaRN ? 1.0 : 0.0;
    if (const char* why = strata::kernels::rope_scaling_invalid(cfg)) {
        std::fprintf(stderr, "strata-q36: invalid rope scaling configuration: %s (factor %g, original context %g)\n", why,
                     cfg.factor, cfg.orig_ctx);
        return false;
    }
    if (cfg.type != RST::None && cfg.factor == 1.0) {
        // the trained angles: m.rope stays the identity, so the angles are bit-identical to the run without the option
        std::fprintf(stderr, "strata-q36: rope scaling %s with factor 1: the trained angles, no scaling is applied\n",
                     cfg.type == RST::YaRN ? "yarn" : "linear");
        return true;
    }
    m.rope = cfg.kernel_args(m.n_rot);
    if (cfg.type != RST::None)
        std::fprintf(stderr, "strata-q36: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), --max-context %d "
                     "against a trained context of %.0f\n", cfg.type == RST::YaRN ? "yarn" : "linear", cfg.factor,
                     cfg.freq_scale(), cfg.freq_base, cfg.mscale(), ctx, cfg.orig_ctx);
    return true;
}

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = -1, hybrid_rows = 128;   // cpu_threads < 0: q36_threads.hpp's default
    double expert_vram = -1.0;
    float cache_prior = 0.0f, ser_t = 0.0f;   // routing options, off
    int ser_kmin = 8;
    double pcie_frac = -1.0;   // the PCIe share of the missed experts: < 0 the probed one, 0 off
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    std::string rope_scaling;
    std::string expert_profile, expert_profile_save;   // the STRP profile the slots start from / is written to at the end
    double rope_scale = 0, rope_orig_ctx = 0;   // 0 = the option is absent (the GGUF's keys, else off)
    bool kv_q4 = false;                          // --kv q4: the 4-bit (Hadamard q4_0) K / V cache
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--expert-profile") expert_profile = next();
        else if (a == "--expert-profile-save") expert_profile_save = next();
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--pcie-frac") pcie_frac = std::atof(next().c_str());
        else if (a == "--cache-prior") cache_prior = (float) std::atof(next().c_str());   // 0 off, else in (0, 1]
        else if (a == "--ser") {                                                          // Kmin,t
            const std::string v = next();
            char* end = nullptr;
            const long k = std::strtol(v.c_str(), &end, 10);
            const bool comma = end != nullptr && *end == ',';
            ser_t = comma ? (float) std::atof(end + 1) : -1.0f;
            if (!comma || k < 1 || k > 8 || !(ser_t >= 0.0f && ser_t <= 1.0f)) {
                std::fprintf(stderr, "strata-q36: --ser must be Kmin,t with 1 <= Kmin <= 8 and 0 <= t <= 1 (got '%s')\n", v.c_str());
                return 2;
            }
            ser_kmin = (int) k;
        }
        else if (a == "--kv") {                                                    // fp32 (default) | q4, as setup writes it
            const std::string v = next();
            if (v != "fp32" && v != "q4") { std::fprintf(stderr, "strata-q36: --kv must be fp32 or q4 (got '%s')\n", v.c_str()); return 2; }
            kv_q4 = v == "q4";
        }
        else if (a == "--rope-scaling") rope_scaling = next();                     // none | linear | yarn, as strata's
        else if (a == "--rope-scale") rope_scale = std::atof(next().c_str());
        else if (a == "--rope-orig-ctx" || a == "--yarn-orig-ctx") rope_orig_ctx = std::atof(next().c_str());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph] [--kv fp32|q4]\n"
                         "       [--rope-scaling none|linear|yarn [--rope-scale F] [--rope-orig-ctx N]]\n"
                         "       [--expert-profile FILE] [--expert-profile-save FILE] [--pcie-frac F]\n"
                         "       [--cache-prior L] [--ser Kmin,t]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
    if (cache_prior < 0.0f || cache_prior > 1.0f) {
        std::fprintf(stderr, "strata-q36: --cache-prior must be in [0, 1] (got %g)\n", (double) cache_prior);
        return 2;
    }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_Q8_PACKED", "1", 0);   // the packed Q8_0 decode copy unless STRATA_Q8_PACKED is set otherwise
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("STRATA_Q8_PACKED") == nullptr) _putenv_s("STRATA_Q8_PACKED", "1");   // as on Linux (it was missing here)
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    if (!resolve_rope(m, rope_scaling, rope_scale, rope_orig_ctx, ctx)) return 2;
    Engine E;
    E.use_graph = graph;
    E.kv_q4 = kv_q4;
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        const size_t fr_raw = fr;   // before the packed Q8_0 copies count as given back
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        const double GiB = (double) (1ull << 30);
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts.  Its buffers grow by about 0.5 MiB per
            // token (measured 0.47: 1.66 GiB more at 4096 than at 512), the sessions' K / V caches by 50 KB per cell and session.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (measured on a 32 GB R9700 at 8192 cells, 512 chunk: 1.77 GiB of buffers + K / V; the free room after init must still hold
            // every expert and the 0.77 GiB reserve, or place_experts gives the packed decode copies back - slower decode - or goes hybrid)
            const int ns = std::max(1, batch);
            size_t n_attn = m.has_mtp ? 1 : 0;
            for (const Layer& L : m.layers) n_attn += L.attn ? 1 : 0;
            const size_t kv_cell = kv_cell_bytes_for(kv_q4, m.g.head_dim);
            const double kv_one = (double) (n_attn * 2 * (size_t) m.g.n_head_kv * kv_cell);   // 45056 on Qwen3.6-35B (11 planes)
            // per cell: the fp16 copies the prompt attention keeps.  5120 B is what was measured on Qwen3.6-35B-A3B; the formula
            // 2 (K, V) x n_head_kv x head_dim x 2 B gives 2048 there, so the constant stays: in fp32 k16 / v16 grow with the
            // context and it counts them; in q4 they hold one kKvWindow window, so that part drops
            const double kv_f16_cell = 4.0 * (double) m.g.n_head_kv * (double) m.g.head_dim;
            const double kv_shared = kv_q4 ? 5120.0 - kv_f16_cell : 5120.0;
            const double sess_extra = std::max(0.0, (double) ctx * (kv_one * ns + kv_shared) - 8192.0 * (kv_one + kv_shared));
            const int cap = ns > 1 ? 1024 : 4096;
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + 2.75 * GiB + (double) (c - 512) * 0.5 * 1048576.0 + sess_extra <= (double) fr_raw) {   // (no give-back of the packed decode copies for a bigger chunk)
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("STRATA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (raw %.2f, with %.2f packed), main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, fr_raw / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0, chunk);
    }
    E.init(m, ctx, chunk, batch, vision);
    if (!expert_profile.empty()) {
        // a profile of another geometry, or one that cannot be read, is said once and the slots keep experts 0 .. S - 1
        int64_t pslots = 0;
        std::string perr;
        if (strata::core::read_expert_profile(expert_profile, (int64_t) m.layers.size(), m.g.n_expert, E.profile_prior, pslots, perr))
            std::fprintf(stderr, "strata-q36: expert profile %s: %zu ranked pairs, built for %lld slots, the slots start from it\n",
                         expert_profile.c_str(), E.profile_prior.size(), (long long) pslots);
        else {
            E.profile_prior.clear();
            std::fprintf(stderr, "strata-q36: %s - the slots start with experts 0 .. S - 1\n", perr.c_str());
        }
    }
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, pcie_frac, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    // the model's geometry must fit the buffers it is written to, checked here with place_experts' final sizes (it shrinks the
    // score buffer when the experts fit is close): the prompt attention writes H x ldp scores per query row, at least one row
    if ((size_t) E.m->g.n_head * (size_t) ((E.kv_window() + 7) / 8 * 8) > E.kAttScoreFloats) {
        std::fprintf(stderr, "strata-q36: the prompt attention needs %d heads x %d key cells of scores, more than the %zu floats "
                     "the engine holds: lower --ctx or use --kv q4\n", (int) E.m->g.n_head, E.kv_window(), E.kAttScoreFloats);
        return 2;
    }
    if (E.verify_k) {   // VERIFY's scratch (vbuf) must hold what the small Q8_0 path and the shared expert's gate and up write
        // the kernels' own predicates decide which matrices write vbuf: the others are not tested
        bool fits = true;
        std::set<std::pair<int, int>> shapes;   // the small path's Q8_0 shapes: one check site per shape and T
        const auto q8_fits = [&E, &shapes](const Mat& w) {
            if (!E.q8_small_shape(w)) return true;
            shapes.insert({(int) w.n_in, (int) w.n_out});
            return 8 * (size_t) w.n_out <= Engine::kVerifyFloats;
        };
        std::vector<const Layer*> layers;   // the MTP block's Mats are empty (type -1) unless it is loaded
        for (const Layer& L : E.m->layers) layers.push_back(&L);
        layers.push_back(&E.m->mtp);
        for (const Layer* L : layers) {
            for (const Mat* w : {&L->wq, &L->wk, &L->wv, &L->wo, &L->qkv, &L->zg, &L->ssm_out, &L->sh_gate, &L->sh_up,
                                 &L->sh_down})
                fits = fits && q8_fits(*w);
            if (E.shared_gu_shape(*L)) fits = fits && 12 * (size_t) E.m->g.n_ff_shexp <= Engine::kVerifyFloats;   // up at 8 x FS, T <= 4
        }
        fits = fits && q8_fits(E.m->head) && q8_fits(E.m->eh_proj);
        if (!fits) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_VERIFY needs every Q8_0 matrix on the small path to have at most %zu "
                         "outputs and the shared expert at most %zu; this model has more: unset STRATA_Q36_VERIFY to run it\n",
                         Engine::kVerifyFloats / 8, Engine::kVerifyFloats / 12);
            return 2;
        }
        // the check sites: one per shape and T for the small Q8_0 path, and ten fixed checks per T, with T <= kGemmT rows
        const size_t sites = (shapes.size() + 10) * Engine::kGemmT;
        if (sites > (size_t) Engine::kVerifySlots) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_VERIFY needs %zu check sites (%zu Q8_0 shapes on the small path), the "
                         "table holds %d: unset STRATA_Q36_VERIFY to run it\n", sites, shapes.size(), Engine::kVerifySlots);
            return 2;
        }
        E.verify_alloc();
    }
    // the routing options that cannot hold their state or cannot apply are refused here, before anything runs
    if (cache_prior > 0.0f && E.n_sess > 1) {
        std::fprintf(stderr, "strata-q36: --cache-prior needs one session (--batch %d decodes %d at once): its Delta_avg state "
                     "is kept for the engine, not per session. Use --batch 1 or drop --cache-prior\n", batch, E.n_sess);
        return 2;
    }
    if (cache_prior > 0.0f && m.has_mtp && mtp_d > 0) {
        std::fprintf(stderr, "strata-q36: --cache-prior and --mtp %d cannot be combined: the rejected MTP draft rows would stay "
                     "folded into Delta_avg. Use --mtp 0 or drop --cache-prior\n", mtp_d);
        return 2;
    }
    if (cache_prior > 0.0f && !m.hybrid) {
        std::fprintf(stderr, "strata-q36: --cache-prior needs the hybrid expert mode (not every expert fits in VRAM): with every "
                     "expert resident there is no slot miss to bias\n");
        return 2;
    }
    if ((cache_prior > 0.0f || ser_kmin < (int) m.top_k) && !E.fast_route()) {
        std::fprintf(stderr, "strata-q36: --cache-prior and --ser need the fast routing path (STRATA_Q36_OLD_KERNELS unset, 256 "
                     "experts, 8 picks); this model or setting is not on it\n");
        return 2;
    }
    struct Report {
        Engine& E;
        const std::string& profile_save;
        ~Report() {
            if (!profile_save.empty()) E.save_profile(profile_save);
            E.verify_flush();
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu + E.hy_pcie;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld over PCIe, "
                         "%lld on the CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_pcie, E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E, expert_profile_save};
    E.cache_prior = cache_prior;
    E.ser_kmin = ser_kmin;
    E.ser_t = ser_t;
    if (cache_prior > 0.0f || ser_kmin < (int) m.top_k)
        std::fprintf(stderr, "strata-q36: routing options on: cache prior %g, SER Kmin %d t %g (decode and verify windows of up to 8 rows)\n",
                     (double) cache_prior, ser_kmin, (double) ser_t);
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                if (one < 0) {
                    std::fprintf(stderr, "strata-q36: %s\n", kNoLogitMsg);
                    return 2;
                }
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    std::vector<int32_t> prompt, forced;
    if (!parse_ids(tokens.c_str(), prompt) || !parse_ids(force.c_str(), forced)) {
        std::fprintf(stderr, "strata-q36: --tokens and --force take integers separated by commas or spaces\n");
        return 2;
    }
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-q36: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-q36: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            if (nx < 0) {
                std::fprintf(stderr, "strata-q36: %s\n", kNoLogitMsg);
                return 2;
            }
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        if (a < 0) {
            std::fprintf(stderr, "strata-q36: %s\n", kNoLogitMsg);
            return 2;
        }
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
