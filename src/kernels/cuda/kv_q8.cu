// src/kernels/cuda/kv_q8.cu - see include/strata/kernels/kv_q8.hpp.
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa_prep.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
__global__ void kv_append_q8_kernel(int8_t* __restrict__ k_q, int8_t* __restrict__ v_q,
                                    uint16_t* __restrict__ k_scale, uint16_t* __restrict__ v_scale,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur, int kv_heads,
                                    int head_dim, int page_size, KvHostPools host) {
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const bool is_v = blockIdx.z == 1;
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(x / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
}

// O6e (qsa_prep.hpp): K (blockIdx.y == 0) normalized and rotated, or V (1), of KV head blockIdx.x of token
// blockIdx.z, then kv_append_q8_kernel's quantizer on it. 256 threads: element t is value t % 64 of group t / 64,
// held by warps 2g and 2g + 1 exactly as that kernel's two warps hold them.
template <bool NATIVE_NORM, bool NATIVE_ROPE>
__global__ void __launch_bounds__(qsa_prep::kThreads)
kv_append_q8_prep_kernel(int8_t* __restrict__ k_q, int8_t* __restrict__ v_q, uint16_t* __restrict__ k_scale,
                         uint16_t* __restrict__ v_scale, const int32_t* __restrict__ table,
                         const int32_t* __restrict__ step, float* __restrict__ kcur, float* __restrict__ vcur,
                         const float* __restrict__ k_gamma, const int32_t* __restrict__ rope_pos, int pos_stride,
                         qsa_prep::RowArgs ra, int kv_heads, int head_dim, int page_size, KvHostPools host) {
    __shared__ float sh[qsa_prep::kShared];
    __shared__ float warp_max[qsa_prep::kThreads / 32];
    const int h = blockIdx.x, t = threadIdx.x, tok = blockIdx.z;
    const bool is_v = blockIdx.y == 1;
    const long long pos = (long long) __ldg(step + (size_t) tok * kStepCount + kStepPos);
    float* row_p = (is_v ? vcur : kcur) + ((size_t) tok * kv_heads + h) * head_dim;
    float x = row_p[t];
    if (!is_v) {   // block-uniform
        x = qsa_prep::norm_rope_row<NATIVE_NORM, NATIVE_ROPE>(x, head_dim, k_gamma,
                                                              rope_pos[(size_t) tok * pos_stride + h], ra, sh);
        row_p[t] = x;
    }
    const int groups = head_dim / KV_Q8_GROUP, g = t / KV_Q8_GROUP, e = t % KV_Q8_GROUP;
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[2 * g], warp_max[2 * g + 1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(x / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + e] = (int8_t) q;
        if (e == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + e] = (int8_t) q;
        if (e == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
}

// One thread = 4 consecutive values of one cell and head (as the FP16 gather does with uint2).
__global__ void kv_gather_q8_kernel(const int8_t* __restrict__ k_q, const int8_t* __restrict__ v_q,
                                    const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const char4 kc = reinterpret_cast<const char4*>(k_q + row * head_dim)[q4];
    const char4 vc = reinterpret_cast<const char4*>(v_q + row * head_dim)[q4];
    ushort4 ko, vo;
    ko.x = f16_from_f32((float) kc.x * ks); ko.y = f16_from_f32((float) kc.y * ks);
    ko.z = f16_from_f32((float) kc.z * ks); ko.w = f16_from_f32((float) kc.w * ks);
    vo.x = f16_from_f32((float) vc.x * vs); vo.y = f16_from_f32((float) vc.y * vs);
    vo.z = f16_from_f32((float) vc.z * vs); vo.w = f16_from_f32((float) vc.w * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<ushort4*>(k_scratch)[dst] = ko;
    reinterpret_cast<ushort4*>(v_scratch)[dst] = vo;
}

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_Q8_GROUP), 2);
    kv_append_q8_kernel<<<grid, KV_Q8_GROUP, 0, (cudaStream_t) stream>>>(
        k_q, v_q, k_scale, v_scale, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_q8 launch");
}

void kv_append_q8_prep_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                            const int32_t* page_table, const int32_t* step, float* kcur, float* vcur,
                            const float* k_gamma, const int32_t* pos, int64_t pos_stride, int64_t n_tok,
                            const QsaNormRope& nr, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append_q8_prep");
    if (!qsa_prep_supported(s, nr) || step == nullptr || k_gamma == nullptr || pos == nullptr || n_tok < 1 ||
        n_tok > 65535) {
        std::fprintf(stderr, "kv_q8: kv_append_q8_prep: geometry, arithmetic or arguments outside the fused path\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) s.n_head_kv, 2, (unsigned) n_tok);
    STRATA_QSA_PREP_LAUNCH(nr, kv_append_q8_prep_kernel, grid, stream, k_q, v_q, k_scale, v_scale, page_table, step,
                           kcur, vcur, k_gamma, pos, (int) pos_stride, qsa_prep::row_args(nr, s), (int) s.n_head_kv,
                           (int) s.head_dim, (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_q8_prep launch");
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256);
    kv_gather_q8_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
        k_q, v_q, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
        k_scratch, v_scratch);
    check("kv_gather_q8 launch");
}

}  // namespace strata::kernels
