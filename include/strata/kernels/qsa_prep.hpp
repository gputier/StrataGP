// include/strata/kernels/qsa_prep.hpp - O6e: a QSA layer's K and Q preparation, one launch each.
//
// The decode layer prepared K in three to five launches - a norm, a rotation, (Q4) the Hadamard transforms of K and
// V, the append - and Q in five or six: a 2-D copy of the query half out of the q/gate projection, a norm, a
// rotation, (Q4) a transform, then the indexer query's norm and rotation. The kernels here do the same work in one
// launch for K and V and one for both queries, one 256-thread block per row: the row is reduced, weighted, rotated
// in shared memory, transformed by one warp, and quantized and written from registers.
//
// **THEY ARE BITWISE THE SEPARATE LAUNCHES, NOT AN APPROXIMATION OF THEM.**  Every floating-point operation is one
// the separate kernels execute, on the same values, in the same order, fused or not exactly as ptxas fuses it there
// (CUDA 13.0, sm_120):
//   - `rms_norm_weighted` (precise): one warp's lane-strided FMA chain, a shfl_down tree, lane 0's IEEE divide and
//     rsqrtf, then `(x * w) * inv`;
//   - `native_qsa_rms_norm_weighted` (a --use_fast_math file): the square fused into +0, two XOR butterflies over
//     256 threads, `sum * rcp(n) + eps` fused into one FTZ FMA (that is how ptxas lowers its approximate divide),
//     the approximate rsqrt, then `(scale * x) * gamma`, all flushing sub-normals;
//   - `rope_neox_apply` over the float64-built table: `a*c` fused with `-(b*s)`, `b*c` fused with `a*s`;
//   - `native_rope_apply`: `theta = pos * ex2(lg2(theta_scale) * pair)`, the approximate cos/sin, `c*a` fused with
//     `-(s*b)`, `s*a` fused with `c*b`;
//   - `fwht256_cuda`'s butterfly, `kv_append*_step`'s quantizers (the Q4 one is the same function).
// The fast-math arithmetic is written here as the PTX those files compile to, so it holds in a precise translation
// unit; the multiply-adds are pinned with `.rn` operations so ptxas cannot fuse anything differently. What makes it
// bitwise is therefore a property of the compiled SEPARATE kernels too: `qsa_prep_parity` checks it on every format
// and variant, including rows whose squares are sub-normal, zero rows and rows that overflow. Run it after any
// toolkit change.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// The arithmetic the separate launches would use, as the layer chooses it for them.
struct QsaNormRope {
    float eps = 0.0f;
    bool native_norm = false;         ///< `native_qsa_rms_norm_weighted` (else `rms_norm_weighted`)
    bool native_rope = false;         ///< `native_rope_apply` (else `rope_neox_apply` over `cos_tab`/`sin_tab`)
    float freq_base = 0.0f;           ///< the native rotation's base (`qsa_freq_base()`)
    const float* cos_tab = nullptr;   ///< the table rotation's (max_pos, n_rot/2) tables
    const float* sin_tab = nullptr;
};

/// Whether the fused preparation covers this geometry and arithmetic: head_dim 256 (one row per 256-thread block,
/// and the Hadamard size), idx_dim at most 256, an even n_rot inside both widths, and each variant's own contract
/// (the native rotation: n_rot 64, widths 128/256 and a finite base above 1, as `native_rope_apply` requires; the
/// table rotation: its tables), and idx_n_head <= n_head (the indexer query's rows read their positions from the
/// same n_head entries per token as q's). A caller keeps the separate launches otherwise.
bool qsa_prep_supported(const QsaShapes& s, const QsaNormRope& nr);

/// K and V of `n_tok` cells into the cache, bitwise the `norm(K) ; rope(K) ; kv_append*_step` sequence (for Q4 with
/// `fwht256` on K and V before the append). Token t reads `kcur`/`vcur` + t * n_head_kv * head_dim, its step at
/// `step` + t * kStepCount and its positions at `pos` + t * `pos_stride` (row h rotates at pos[h], as the separate
/// rotation reads them). The decode layer passes one token; a verify window may pass its T, whose cells are distinct.
/// `kcur` (and for Q4 `vcur`) are left holding what the separate launches leave there: the prepared rows.
void kv_append_prep_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                         float* kcur, float* vcur, const float* k_gamma, const int32_t* pos, int64_t pos_stride,
                         int64_t n_tok, const QsaNormRope& nr, const QsaShapes& s, void* stream,
                         const KvHostPools* host = nullptr);
void kv_append_q8_prep_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                            const int32_t* page_table, const int32_t* step, float* kcur, float* vcur,
                            const float* k_gamma, const int32_t* pos, int64_t pos_stride, int64_t n_tok,
                            const QsaNormRope& nr, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr);
void kv_append_q4_prep_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                            float* kcur, float* vcur, const float* k_gamma, const int32_t* pos, int64_t pos_stride,
                            int64_t n_tok, const QsaNormRope& nr, const QsaShapes& s, void* stream,
                            const KvHostPools* host = nullptr);

/// Both queries of `n_tok` tokens in one launch. The query rows are read STRAIGHT from the q/gate projection `q_full`
/// (the first head_dim of each head's 2 * head_dim block) and written normalized and rotated to `qcur` - bitwise the
/// `cudaMemcpy2DAsync` split + norm + rope, with `fwht` (Q4) also `fwht256` on qcur. The indexer query `q_idx` is
/// normalized and rotated in place (null: none). Token t: q_full + t * n_head * 2 * head_dim, qcur + t * n_head *
/// head_dim, q_idx + t * idx_n_head * idx_dim, positions `pos` + t * `pos_stride`.
void qsa_q_prep(const float* q_full, float* qcur, const float* q_gamma, float* q_idx, const float* idx_gamma,
                const int32_t* pos, int64_t pos_stride, int64_t n_tok, bool fwht, const QsaNormRope& nr,
                const QsaShapes& s, void* stream);

#if defined(__CUDACC__)
namespace qsa_prep {

constexpr int kThreads = 256;       ///< one row per block, one element per thread
constexpr int kShared = 256 + 16;   ///< floats: the row, then the reduction slots

/// What a block needs to prepare one row besides its data (kernel arguments, by value).
struct RowArgs {
    float eps;
    float theta_scale;              ///< native rotation: `native_rope_theta_scale(freq_base, n_rot)`
    int n_rot;
    const float* cos_tab;
    const float* sin_tab;
    const int32_t* mtab;            ///< `mrope_table()`, as the separate rotations read it
};

/// `nr`'s row arguments for this geometry (host side, qsa_prep.cu).
RowArgs row_args(const QsaNormRope& nr, const QsaShapes& s);

/// Launches `kernel<NATIVE_NORM, NATIVE_ROPE>` for `nr`'s two choices, one 256-thread block per row.
#define STRATA_QSA_PREP_LAUNCH(nr, kernel, grid, stream, ...)                                                    \
    do {                                                                                                         \
        const cudaStream_t qp_cs_ = (cudaStream_t) (stream);                                                     \
        if ((nr).native_norm && (nr).native_rope)                                                                \
            kernel<true, true><<<(grid), ::strata::kernels::qsa_prep::kThreads, 0, qp_cs_>>>(__VA_ARGS__);       \
        else if ((nr).native_norm)                                                                               \
            kernel<true, false><<<(grid), ::strata::kernels::qsa_prep::kThreads, 0, qp_cs_>>>(__VA_ARGS__);      \
        else if ((nr).native_rope)                                                                               \
            kernel<false, true><<<(grid), ::strata::kernels::qsa_prep::kThreads, 0, qp_cs_>>>(__VA_ARGS__);      \
        else                                                                                                     \
            kernel<false, false><<<(grid), ::strata::kernels::qsa_prep::kThreads, 0, qp_cs_>>>(__VA_ARGS__);     \
    } while (0)

// The fast-math files' arithmetic, as the PTX they compile to: FTZ, and `.rn` so nothing here is fused further.
__device__ __forceinline__ float ftz_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
__device__ __forceinline__ float ftz_add(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float ftz_mul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
#define STRATA_QSA_PREP_UNARY(name, op)                                  \
    __device__ __forceinline__ float name(float a) {                     \
        float r;                                                         \
        asm(op " %0, %1;" : "=f"(r) : "f"(a));                           \
        return r;                                                        \
    }
STRATA_QSA_PREP_UNARY(ftz_rcp, "rcp.approx.ftz.f32")
STRATA_QSA_PREP_UNARY(ftz_rsqrt, "rsqrt.approx.ftz.f32")
STRATA_QSA_PREP_UNARY(ftz_lg2, "lg2.approx.ftz.f32")
STRATA_QSA_PREP_UNARY(ftz_ex2, "ex2.approx.ftz.f32")
STRATA_QSA_PREP_UNARY(ftz_sin, "sin.approx.ftz.f32")
STRATA_QSA_PREP_UNARY(ftz_cos, "cos.approx.ftz.f32")
#undef STRATA_QSA_PREP_UNARY

/// ONE ROW of `cols` (<= 256) floats, element t held by thread t of a 256-thread block (t >= cols: anything).
/// Returns element t normalized, weighted and rotated at `pos` (t >= cols: 0). `sh` is kShared floats of shared
/// memory. It synchronises the block, so every thread must reach it, and it leaves the result in sh[0..cols).
template <bool NATIVE_NORM, bool NATIVE_ROPE>
__device__ __forceinline__ float norm_rope_row(float x, int cols, const float* __restrict__ gamma, int pos,
                                               const RowArgs& a, float* sh) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float* row = sh;
    float* red = sh + 256;
    float y = 0.0f;
    if (NATIVE_NORM) {
        // native_qsa.cu `norm<256>`: each thread's square (its loop runs once: fused into +0), a warp butterfly,
        // the 8 warp sums through shared memory, the butterfly again (every lane ends with the same sum)
        float partial = t < cols ? ftz_fma(x, x, 0.0f) : 0.0f;
        for (int o = 16; o > 0; o >>= 1) partial = ftz_add(partial, __shfl_xor_sync(0xffffffffu, partial, o, 32));
        if (lane == 0) red[warp] = partial;
        __syncthreads();
        partial = lane < kThreads / 32 ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) partial = ftz_add(partial, __shfl_xor_sync(0xffffffffu, partial, o, 32));
        // `rsqrtf(partial / n_cols + epsilon)`: ptxas lowers the approximate divide to rcp and a multiply, and fuses
        // that multiply with the epsilon's add
        const float scale = ftz_rsqrt(ftz_fma(partial, ftz_rcp((float) cols), a.eps));
        if (t < cols) y = ftz_mul(ftz_mul(scale, x), gamma[t]);
    } else {
        // elementwise.cu `rms_norm_weighted_kernel`: ONE warp per row, lane-strided FMA chain from 0, shfl_down tree,
        // lane 0's `rsqrtf(acc / cols + eps)` broadcast
        if (t < cols) row[t] = x;
        __syncthreads();
        if (warp == 0) {
            float acc = 0.0f;
            for (int c = lane; c < cols; c += 32) acc = __fmaf_rn(row[c], row[c], acc);
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) red[8] = rsqrtf(acc / (float) cols + a.eps);
        }
        __syncthreads();
        if (t < cols) y = __fmul_rn(__fmul_rn(x, gamma[t]), red[8]);
    }
    // the rotation: NEOX pairs (i, i + n_rot/2), one thread per pair, in shared memory; the tail passes through
    if (t < cols) row[t] = y;
    __syncthreads();
    const int half = a.n_rot / 2;
    if (t < half) {
        const float ea = row[t], eb = row[t + half];
        float oa, ob;
        if (NATIVE_ROPE) {
            const float theta = ftz_mul(ftz_ex2(ftz_mul(ftz_lg2(a.theta_scale), (float) t)),
                                        (float) mrope_pos(a.mtab, pos, t));
            const float c = ftz_cos(theta), s = ftz_sin(theta);
            oa = ftz_fma(c, ea, -ftz_mul(s, eb));
            ob = ftz_fma(s, ea, ftz_mul(c, eb));
        } else {
            const size_t toff = (size_t) mrope_pos(a.mtab, pos, t) * half;
            const float c = a.cos_tab[toff + t], s = a.sin_tab[toff + t];
            oa = __fmaf_rn(ea, c, -__fmul_rn(eb, s));
            ob = __fmaf_rn(eb, c, __fmul_rn(ea, s));
        }
        row[t] = oa;
        row[t + half] = ob;
    }
    __syncthreads();
    return t < cols ? row[t] : 0.0f;
}

/// kv_q4.cu `fwht256_kernel` on one row of 256 held one element per thread: warp 0 transforms it in `sh` with the
/// same butterfly (8 values per lane, lanes then registers), and every thread returns its element.
__device__ __forceinline__ float fwht256_row(float y, float* sh) {
    const int t = threadIdx.x;
    sh[t] = y;
    __syncthreads();
    if (t < 32) {
        constexpr int warp_size = 32, N = 256, el_w = N / warp_size;
        const int lane = t;
        float reg[el_w];
#pragma unroll
        for (int i = 0; i < el_w; ++i) reg[i] = __fmul_rn(sh[i * warp_size + lane], 1.0f / 16.0f);
#pragma unroll
        for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
            for (int j = 0; j < el_w; ++j) {
                const float val = reg[j];
                const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
#pragma unroll
        for (int h = warp_size; h < N; h *= 2) {
            const int step = h / warp_size;
#pragma unroll
            for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
                for (int k = 0; k < step; ++k) {
                    const float x = reg[j + k];
                    const float y2 = reg[j + k + step];
                    reg[j + k] = x + y2;
                    reg[j + k + step] = x - y2;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < el_w; ++i) sh[i * warp_size + lane] = reg[i];
    }
    __syncthreads();
    return sh[t];
}

}  // namespace qsa_prep
#endif

}  // namespace strata::kernels
