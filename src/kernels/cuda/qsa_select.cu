// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Four dimensions of a key row for one lane: the fp32 row, or (O6, #21) its fp16 shadow widened exactly.
__device__ __forceinline__ float4 key4(const float* key, int lane) {
    return *reinterpret_cast<const float4*>(key + lane * 4);
}
__device__ __forceinline__ float4 key4(const uint16_t* key, int lane) {
    const uint2 raw = *reinterpret_cast<const uint2*>(key + lane * 4);
    return make_float4(__half2float(__ushort_as_half((unsigned short) (raw.x & 0xffffu))),
                       __half2float(__ushort_as_half((unsigned short) (raw.x >> 16))),
                       __half2float(__ushort_as_half((unsigned short) (raw.y & 0xffffu))),
                       __half2float(__ushort_as_half((unsigned short) (raw.y >> 16))));
}

// `KEY` float: the fp32 keys.  `KEY` uint16_t: the fp16 shadow (256 B per block instead of 512), the same arithmetic
// on the widened values - so its scores are exactly the fp32 path's on keys rounded to fp16.
template <typename KEY>
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const KEY* __restrict__ pooled,
                                                                        const KEY* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const KEY* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = key4(key, lane);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

// ---- O6b (#22): the same selection, read the way the memory wants it.
//
// `block_topk_kernel` gives each thread a contiguous run of ~nb/256 blocks, so every warp-wide load touches 32
// sectors; its histogram is one shared array whose few hot bins (the ReLU piles the scores into a handful of
// exponents) serialise the atomics; and thread 0 walks the 256 bins and the 256 per-thread counts one by one.  Here:
//   * the histogram passes read 32 consecutive blocks per warp, TOPK2_U loads ahead;
//   * the lanes of a warp that fall in the same bin add ONCE (`__match_any_sync`), into that warp's own histogram;
//   * the digit comes from a block-wide scan of the 256 bins (thread t holds digit 255 - t, so an exclusive prefix
//     is "cells in the digits above"), which is the serial walk's rule evaluated at every digit at once;
//   * counting and emission give each WARP a contiguous range of blocks and walk it 32 blocks at a time, placing
//     each lane's cells with a warp scan: ascending output without a per-thread contiguous walk.
// Every count is an integer sum, whose order cannot change it, so the threshold, the budget of equal cells and the
// emitted cells are exactly those of `block_topk_kernel` (`qsa_select_parity` asserts it on the device).
constexpr int TOPK2_T = 512;
constexpr int TOPK2_W = TOPK2_T / 32;
constexpr int TOPK2_U = 4;

__device__ __forceinline__ int warp_isum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Warp-wide exclusive prefix sum; `total` is the warp's sum.
__device__ __forceinline__ int warp_excl_scan(int v, int& total) {
    const int lane = threadIdx.x & 31;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    total = __shfl_sync(0xffffffffu, x, 31);
    return x - v;
}

// Block-wide exclusive prefix sum of one int per thread; `total` is the block's sum.  Every thread must call it.
__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int wt;
    const int x = warp_excl_scan(v, wt);
    if (lane == 0) s_warp[w] = wt;
    __syncthreads();
    int before = 0, all = 0;
#pragma unroll
    for (int i = 0; i < TOPK2_W; ++i) {
        const int c = s_warp[i];
        before += i < w ? c : 0;
        all += c;
    }
    __syncthreads();   // s_warp is reused by the next call
    total = all;
    return before + x;
}

__global__ void __launch_bounds__(TOPK2_T) block_topk2_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TOPK2_W][256];   // one histogram per warp: 16 KB
    __shared__ int s_warp[TOPK2_W];
    __shared__ int s_gt[TOPK2_W], s_eq[TOPK2_W];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int j = t; j < n_kv; j += TOPK2_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int nb = n_bid + 1;                          // blocks 0..n_bid, the last possibly empty
    const int tail_w = n_kv - n_bid * R;               // cells of block n_bid: 0..R-1
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < TOPK2_W * 256; i += TOPK2_T) (&hist[0][0])[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int base = warp * 32; base < nb; base += TOPK2_T * TOPK2_U) {
            float v[TOPK2_U];
#pragma unroll
            for (int u = 0; u < TOPK2_U; ++u) {
                const int b = base + u * TOPK2_T + lane;
                v[u] = b < nb ? sc[b] : 0.0f;
            }
#pragma unroll
            for (int u = 0; u < TOPK2_U; ++u) {
                const int b = base + u * TOPK2_T + lane;
                const int w = b < n_bid ? R : (b == n_bid ? tail_w : 0);
                const uint32_t k = order_key(v[u]);
                const bool hit = w > 0 && (k & hi_mask) == (prefix & hi_mask);
                const uint32_t digit = hit ? (k >> shift) & 255u : 256u;   // 256: no bin
                const unsigned peers = __match_any_sync(0xffffffffu, digit);
                const unsigned tails = __ballot_sync(0xffffffffu, hit && b == n_bid);
                if (hit && lane == __ffs(peers) - 1)
                    atomicAdd(&hist[warp][digit], R * __popc(peers & ~tails) + ((peers & tails) ? tail_w : 0));
            }
        }
        __syncthreads();
        // the digit: the largest d with above + (cells in digits >= d) >= width, else 0 - the serial walk's rule
        const int d = 255 - t;
        int cnt = 0;
        if (t < 256) {
#pragma unroll
            for (int w = 0; w < TOPK2_W; ++w) cnt += hist[w][d];
        }
        int total;
        const int excl = block_excl_scan(cnt, s_warp, total);   // cells in the digits above d
        if (t < 256 && (d == 0 || above + excl + cnt >= width) && above + excl < width) {
            s_digit = d;
            s_above = above + excl;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int eq_budget = width - above;               // cells equal to thr that fit, lowest index first
    // ---- each warp owns a contiguous range of whole 32-block steps: count, then place by warp scans
    const int per_w = (((nb + TOPK2_W - 1) / TOPK2_W) + 31) & ~31;
    const int w0 = min(warp * per_w, nb), w1 = min(w0 + per_w, nb);
    int gt = 0, eq = 0;
    for (int base = w0; base < w1; base += 32) {
        const int b = base + lane;
        const int w = b >= w1 ? 0 : (b < n_bid ? R : tail_w);
        if (w > 0) {
            const uint32_t k = order_key(sc[b]);
            if (k > thr) gt += w;
            else if (k == thr) eq += w;
        }
    }
    gt = warp_isum(gt);
    eq = warp_isum(eq);
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    int wpos = 0, eq_seen = 0;                         // cells the warps before emit, and their equal cells
    for (int i = 0; i < warp; ++i) {
        const int e = s_eq[i];
        wpos += s_gt[i] + min(max(eq_budget - eq_seen, 0), e);
        eq_seen += e;
    }
    for (int base = w0; base < w1; base += 32) {
        const int b = base + lane;
        const int w = b >= w1 ? 0 : (b < n_bid ? R : tail_w);
        int wg = 0, we = 0;
        if (w > 0) {
            const uint32_t k = order_key(sc[b]);
            if (k > thr) wg = w;
            else if (k == thr) we = w;
        }
        int e_tot, n_tot;
        const int e_before = warp_excl_scan(we, e_tot);                       // equal cells before b in this step
        const int ne = min(max(eq_budget - (eq_seen + e_before), 0), we);   // this block's equal cells that fit
        const int n = wg + ne;
        const int at = wpos + warp_excl_scan(n, n_tot);
        for (int c = 0; c < n; ++c) out[at + c] = (int32_t) (b * R + c);
        wpos += n_tot;
        eq_seen += e_tot;
    }
}

std::atomic<int> g_topk_old{-1};   // STRATA_OLD_TOPK, read once unless a test sets it

void launch_scores_check(const QsaShapes& s, int64_t nq, const char* who) {
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "%s: unsupported indexer geometry\n", who);
        std::exit(1);
    }
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    launch_scores_check(s, nq, "qsa_block_scores");
    const dim3 grid((unsigned) ((max_blocks + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<float><<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps,
                                                                                     max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_scores_f16(const uint16_t* pooled16, const uint16_t* dead16, const float* q_idx, const int32_t* steps,
                          int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    launch_scores_check(s, nq, "qsa_block_scores_f16");
    if (pooled16 == nullptr || dead16 == nullptr || reinterpret_cast<uintptr_t>(pooled16) % 8 != 0 ||
        reinterpret_cast<uintptr_t>(dead16) % 8 != 0) {
        std::fprintf(stderr, "qsa_block_scores_f16: the fp16 shadow is not allocated or not 8-byte aligned\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) ((max_blocks + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<uint16_t><<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled16, dead16, q_idx,
                                                                                        steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_f16: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_topk_old() {
    int v = g_topk_old.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("STRATA_OLD_TOPK");
        v = (e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0) ? 1 : 0;
        g_topk_old.store(v, std::memory_order_relaxed);
    }
    return v != 0;
}

void qsa_block_topk_set_old(bool old) { g_topk_old.store(old ? 1 : 0, std::memory_order_relaxed); }

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    if (qsa_block_topk_old())
        block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk2_kernel<<<(unsigned) nq, TOPK2_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
