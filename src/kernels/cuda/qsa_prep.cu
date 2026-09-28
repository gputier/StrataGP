// src/kernels/cuda/qsa_prep.cu - O6e, see include/strata/kernels/qsa_prep.hpp: the fused preparation's host side,
// the FP16 append and the two queries. The INT8 and Q4 appends live beside their quantizers (kv_q8.cu, kv_q4.cu).
//
// A PRECISE translation unit on purpose (no --use_fast_math): the canonical arithmetic and the quantizers need IEEE
// division and sub-normals, and the fast-math arithmetic of the native variants is spelled out in PTX by the header.
#include "strata/kernels/qsa_prep.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void fail(const char* what) {
    std::fprintf(stderr, "qsa_prep: %s\n", what);
    std::exit(1);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prep: %s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void require(const QsaShapes& s, const QsaNormRope& nr, const float* gamma, const int32_t* pos, int64_t n_tok,
             const char* what) {
    if (!qsa_prep_supported(s, nr) || gamma == nullptr || pos == nullptr || n_tok < 1 || n_tok > 65535) {
        std::fprintf(stderr, "qsa_prep: %s: geometry, arithmetic or arguments outside the fused path\n", what);
        std::exit(1);
    }
}

// K (blockIdx.y == 0) or V (1) of KV head blockIdx.x of token blockIdx.z; 256 threads, one element each.
// kv_append_kernel's store, after the norm and the rotation of K.
template <bool NATIVE_NORM, bool NATIVE_ROPE>
__global__ void __launch_bounds__(qsa_prep::kThreads)
kv_append_prep_kernel(uint16_t* __restrict__ k_pool, uint16_t* __restrict__ v_pool, const int32_t* __restrict__ table,
                      const int32_t* __restrict__ step, float* __restrict__ kcur, float* __restrict__ vcur,
                      const float* __restrict__ k_gamma, const int32_t* __restrict__ rope_pos, int pos_stride,
                      qsa_prep::RowArgs a, int kv_heads, int head_dim, int page_size, KvHostPools host) {
    __shared__ float sh[qsa_prep::kShared];
    const int h = blockIdx.x, t = threadIdx.x, tok = blockIdx.z;
    const bool is_v = blockIdx.y == 1;
    const long long pos = (long long) __ldg(step + (size_t) tok * kStepCount + kStepPos);
    float* row_p = (is_v ? vcur : kcur) + ((size_t) tok * kv_heads + h) * head_dim;
    float y = row_p[t];
    if (!is_v) {   // block-uniform: the barriers inside are reached by every thread of the block
        y = qsa_prep::norm_rope_row<NATIVE_NORM, NATIVE_ROPE>(y, head_dim, k_gamma,
                                                              rope_pos[(size_t) tok * pos_stride + h], a, sh);
        row_p[t] = y;
    }
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_pool : k_pool)[row * head_dim + t] = f16_from_f32(y);
    }
    if (host.k_pool != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_pool : host.k_pool)[row * head_dim + t] = f16_from_f32(y);
    }
}

// Row blockIdx.x of token blockIdx.y: the query rows first (n_head of them, read from the q/gate projection), then
// the indexer query rows (idx_n_head, in place).
struct QPrepArgs {
    const float* q_full;
    float* qcur;
    const float* q_gamma;
    float* q_idx;
    const float* idx_gamma;
    const int32_t* pos;
    int pos_stride;
    int n_head, head_dim, idx_n_head, idx_dim;
    int fwht;
};

template <bool NATIVE_NORM, bool NATIVE_ROPE>
__global__ void __launch_bounds__(qsa_prep::kThreads) qsa_q_prep_kernel(QPrepArgs q, qsa_prep::RowArgs a) {
    __shared__ float sh[qsa_prep::kShared];
    const int r = blockIdx.x, tok = blockIdx.y, t = threadIdx.x;
    const bool is_q = r < q.n_head;   // block-uniform
    const int row = is_q ? r : r - q.n_head;
    const int cols = is_q ? q.head_dim : q.idx_dim;
    // `per_head[:, :head_dim]`: the FIRST half of the head's 2 * head_dim block, read where the projection left it
    const float* src = is_q ? q.q_full + ((size_t) tok * q.n_head + row) * 2 * q.head_dim
                            : q.q_idx + ((size_t) tok * q.idx_n_head + row) * q.idx_dim;
    float* dst = is_q ? q.qcur + ((size_t) tok * q.n_head + row) * q.head_dim
                      : q.q_idx + ((size_t) tok * q.idx_n_head + row) * q.idx_dim;
    const float x = t < cols ? src[t] : 0.0f;
    float y = qsa_prep::norm_rope_row<NATIVE_NORM, NATIVE_ROPE>(x, cols, is_q ? q.q_gamma : q.idx_gamma,
                                                                q.pos[(size_t) tok * q.pos_stride + row], a, sh);
    if (is_q && q.fwht) y = qsa_prep::fwht256_row(y, sh);   // Q4: <Hq, Hk> = <q, k>
    if (t < cols) dst[t] = y;
}

}  // namespace

bool qsa_prep_supported(const QsaShapes& s, const QsaNormRope& nr) {
    if (s.head_dim != qsa_prep::kThreads || s.n_head_kv < 1 || s.n_head < 1 || s.page_size < 1) return false;
    if (s.idx_dim < 1 || s.idx_dim > qsa_prep::kThreads || s.idx_n_head < 0) return false;
    if (s.n_rot < 2 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) return false;
    if (s.n_head + s.idx_n_head > 65535) return false;
    if (nr.native_rope) {
        // native_rope_apply's own contract; anything else would throw there
        if (s.n_rot != 64 || (s.idx_dim != 128 && s.idx_dim != 256) || !std::isfinite(nr.freq_base) ||
            nr.freq_base <= 1.0f)
            return false;
    } else if (nr.cos_tab == nullptr || nr.sin_tab == nullptr) {
        return false;
    }
    return std::isfinite(nr.eps) && nr.eps >= 0.0f;
}

namespace qsa_prep {
RowArgs row_args(const QsaNormRope& nr, const QsaShapes& s) {
    RowArgs a;
    a.eps = nr.eps;
    a.theta_scale = nr.native_rope ? native_rope_theta_scale(nr.freq_base, (int) s.n_rot) : 0.0f;
    a.n_rot = (int) s.n_rot;
    a.cos_tab = nr.cos_tab;
    a.sin_tab = nr.sin_tab;
    a.mtab = mrope_table();
    return a;
}
}  // namespace qsa_prep

void kv_append_prep_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                         float* kcur, float* vcur, const float* k_gamma, const int32_t* pos, int64_t pos_stride,
                         int64_t n_tok, const QsaNormRope& nr, const QsaShapes& s, void* stream,
                         const KvHostPools* host) {
    require(s, nr, k_gamma, pos, n_tok, "kv_append_prep");
    if (step == nullptr) fail("kv_append_prep: step is null");
    const dim3 grid((unsigned) s.n_head_kv, 2, (unsigned) n_tok);
    STRATA_QSA_PREP_LAUNCH(nr, kv_append_prep_kernel, grid, stream, k_pool, v_pool, page_table, step, kcur, vcur,
                           k_gamma, pos, (int) pos_stride, qsa_prep::row_args(nr, s), (int) s.n_head_kv,
                           (int) s.head_dim, (int) s.page_size, host ? *host : KvHostPools{});
    check_launch("kv_append_prep");
}

void qsa_q_prep(const float* q_full, float* qcur, const float* q_gamma, float* q_idx, const float* idx_gamma,
                const int32_t* pos, int64_t pos_stride, int64_t n_tok, bool fwht, const QsaNormRope& nr,
                const QsaShapes& s, void* stream) {
    require(s, nr, q_gamma, pos, n_tok, "qsa_q_prep");
    if (q_full == nullptr || qcur == nullptr || (q_idx != nullptr && idx_gamma == nullptr))
        fail("qsa_q_prep: a buffer is null");
    QPrepArgs q;
    q.q_full = q_full;
    q.qcur = qcur;
    q.q_gamma = q_gamma;
    q.q_idx = q_idx;
    q.idx_gamma = idx_gamma;
    q.pos = pos;
    q.pos_stride = (int) pos_stride;
    q.n_head = (int) s.n_head;
    q.head_dim = (int) s.head_dim;
    q.idx_n_head = q_idx != nullptr ? (int) s.idx_n_head : 0;
    q.idx_dim = (int) s.idx_dim;
    q.fwht = fwht ? 1 : 0;
    const dim3 grid((unsigned) (q.n_head + q.idx_n_head), (unsigned) n_tok);
    STRATA_QSA_PREP_LAUNCH(nr, qsa_q_prep_kernel, grid, stream, q, qsa_prep::row_args(nr, s));
    check_launch("qsa_q_prep");
}

}  // namespace strata::kernels
