// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

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

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        const uint4 raw = *reinterpret_cast<const uint4*>(base);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
        const float sc = __half2float(__ushort_as_half(sbits));
        const uint2 raw = *reinterpret_cast<const uint2*>(codes);
        const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = __half2float(__ushort_as_half(blk->d));
        const int j = (rem == 0 || rem == 16) ? 0 : 8;
        const uint8_t* bytes = blk->qs + j;
        if (rem < 16) {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
        }
    }
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // B14: a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // B14: masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// ---- O6c (#23): the same chunk with its reads in flight together.
//
// The kernel above has one K load in flight per warp in its score loop, and its value loop walks the chunk's V rows
// one 2-byte read after another: per chunk it waits for memory 8 + 64 times.  This one starts the chunk's V rows as
// `cp.async` copies into shared memory as soon as their pool rows are known, loads each warp's K rows into registers
// before its first dot product, and only then waits: the V copies run under QK and the softmax.
//
// THE ARITHMETIC IS THE SAME EXPRESSIONS IN THE SAME ORDER as `attn_chunk_kernel` - the same `load8`, the same dot
// and warp trees, the same `fmaf` chain over ascending cells, V decoded from the same bits (now read from shared
// memory) - so with CH = 64 its partials are bitwise those of the kernel above (`qsa_decode_attn_parity`).  CH = 32
// gives twice the blocks but a different split of the softmax, so it is opt-in (STRATA_QSA_CHUNK=32).

// V bytes of one cell and KV head: the fp16 row, the int8 codes (their 4 fp16 scales are copied apart), the q4_0 blocks.
template <int KV_MODE>
__host__ __device__ constexpr int v_bytes() {
    return KV_MODE == 0 ? HD * 2 : (KV_MODE == 1 ? HD : (HD / QK4_0) * (int) sizeof(block_q4_0));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<uint2*>(smem) = *reinterpret_cast<const uint2*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

template <int KV_MODE, int CH>
__global__ void __launch_bounds__(THREADS) attn_chunk_pf_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                const int32_t* __restrict__ ids,
                                                                const int32_t* __restrict__ step, int n_kv_heads,
                                                                int page_size, float scale,
                                                                float* __restrict__ part_acc,
                                                                float* __restrict__ part_m,
                                                                float* __restrict__ part_l, int n_chunks, int cap,
                                                                long long scratch_stride) {
    static_assert(CH == 32 || CH == 64, "the softmax maps one or two cells to a lane");
    constexpr int VB = v_bytes<KV_MODE>();
    constexpr int V16 = VB / 16;                  // 16-byte copies per cell: 32 fp16, 16 int8, 9 q4_0
    constexpr int NS = HD / KV_Q8_GROUP;          // int8: fp16 scales per cell (8 bytes)
    constexpr int PER = CH / WARPS;               // cells per warp in the score loop
    static_assert(VB % 16 == 0, "a V row must be whole 16-byte copies");
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];                          // 12 KB: this KV head's query heads
    __shared__ float sp[G][CH];                                        // scores, then probabilities
    __shared__ long long srow[CH];                                     // pool row of each cell, -1 when masked
    __shared__ __align__(16) uint8_t sv[CH * VB];                      // the chunk's V rows (32 KB in fp16)
    __shared__ __align__(16) uint16_t svs[KV_MODE == 1 ? CH * NS : 4]; // int8: their scales
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CH;
    const int n_here = min(CH, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CH) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);   // B14, as above
        }
        srow[t] = r;
    }
    __syncthreads();
    // the V rows start now and land in shared memory while QK and the softmax run
    {
        const uint8_t* vbase = KV_MODE == 0 ? reinterpret_cast<const uint8_t*>(p.v_pool)
                             : (KV_MODE == 1 ? reinterpret_cast<const uint8_t*>(p.v_q) : p.v_q4);
        for (int i = t; i < n_here * V16; i += THREADS) {
            const int c = i / V16, j = i - c * V16;
            const long long r = srow[c];
            if (r >= 0) cp_async16(sv + c * VB + j * 16, vbase + r * VB + j * 16);
        }
        if constexpr (KV_MODE == 1) {
            for (int c = t; c < n_here; c += THREADS)
                if (srow[c] >= 0) cp_async8(svs + c * NS, p.v_scale + srow[c] * NS);
        }
        cp_async_commit();
    }
    // scores: warp w takes cells w, w+8, ...; all its K rows are loaded before the first dot product.
    float kk[PER][8];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = warp + i * WARPS;
        if (c < n_here && srow[c] >= 0) load8<KV_MODE>(p, false, srow[c], lane * 8, kk[i]);
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = warp + i * WARPS;
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = kk[i][0] * qa.x + kk[i][1] * qa.y + kk[i][2] * qa.z + kk[i][3] * qa.w +
                      kk[i][4] * qb.x + kk[i][5] * qb.y + kk[i][6] * qb.z + kk[i][7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        if constexpr (CH == 64) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float m = warp_max(fmaxf(a, b));
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float l = warp_sum(ea + eb);
            if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
        } else {
            const float a = sp[h][lane];
            const float m = warp_max(a);
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            sp[h][lane] = ea;
            const float l = warp_sum(ea);
            if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
        }
    }
    cp_async_wait_all();   // this thread's copies; the barrier makes every thread's visible
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads, reading the rows from shared memory.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked: weight 0, and its row was never copied
        const uint8_t* vr = sv + c * VB;
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(vr)[t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(svs[c * NS + t / KV_Q8_GROUP]));
            v = (float) reinterpret_cast<const int8_t*>(vr)[t] * sc;
        } else {
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(vr) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// Query head h, dimension d: the chunks combined with the usual log-sum-exp rescale.
__device__ __forceinline__ float merge_one(const float* __restrict__ part_acc, const float* __restrict__ part_m,
                                           const float* __restrict__ part_l, int n_chunks, int h, int d) {
    const int kvh = h / G, hl = h % G;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    return L > 0.0f ? acc / L : 0.0f;
}

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int d = threadIdx.x;
    attn[(size_t) h * HD + d] = merge_one(part_acc, part_m, part_l, n_chunks, h, d);
}

// ---- O6c: the output gate folded into the merge (opt-in, STRATA_QSA_MERGE_GATE=1): one launch less per QSA layer.
//
// The gate is the layer's own formula on the same float the merge would have stored, so the gated output is the one
// the separate gate kernel writes - IF the formula compiles to the same instructions.  kQsaGateF64 is
// `qsa_gate_apply_f32_kernel`'s source in a TU built with the same flags (neither is fast-math).  kQsaGateNativeF32
// is `native_qsa_gate_apply`, whose TU IS fast-math: its `1 / (1 + expf(-x))` is written out as the PTX that TU
// emits for it (mul by -log2(e), ex2.approx, add, rcp.approx, mul, all .ftz), with explicit `.rn` so that nothing
// is fused.  `qsa_decode_attn_parity` compares both with the separate kernels, bitwise.
__device__ __forceinline__ float gate_f64(float a, float raw) {
    const double g = (double) raw;
    const double sig = 1.0 / (1.0 + exp(-g));
    return (float) ((double) a * sig);
}
__device__ __forceinline__ float gate_native_f32(float a, float raw) {
    float out;
    asm("{\n\t.reg .f32 t0, t1, t2, t3;\n\t"
        "mul.rn.ftz.f32 t0, %1, 0fBFB8AA3B;\n\t"
        "ex2.approx.ftz.f32 t1, t0;\n\t"
        "add.rn.ftz.f32 t2, t1, 0f3F800000;\n\t"
        "rcp.approx.ftz.f32 t3, t2;\n\t"
        "mul.rn.ftz.f32 %0, %2, t3;\n\t}"
        : "=f"(out)
        : "f"(raw), "f"(a));
    return out;
}

template <int GATE>
__global__ void __launch_bounds__(HD) attn_merge_gate_kernel(const float* __restrict__ part_acc,
                                                             const float* __restrict__ part_m,
                                                             const float* __restrict__ part_l, int n_chunks,
                                                             float* __restrict__ attn, long long scratch_stride,
                                                             const float* __restrict__ q_full,
                                                             float* __restrict__ gated) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    gated += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    q_full += (size_t) blockIdx.y * (size_t) gridDim.x * 2 * HD;
    const int h = blockIdx.x;                 // global query head
    const int d = threadIdx.x;
    const float a = merge_one(part_acc, part_m, part_l, n_chunks, h, d);
    attn[(size_t) h * HD + d] = a;            // still written: a reader of the ungated output sees the same thing
    const float raw = q_full[(size_t) h * 2 * HD + HD + d];   // the gate: the SECOND half of the head's block
    gated[(size_t) h * HD + d] = GATE == kQsaGateF64 ? gate_f64(a, raw) : gate_native_f32(a, raw);
}

}  // namespace

namespace {

// The kernel variant, read once from the environment unless a test sets it (qsa_decode_attn_set_variant).
std::atomic<int> g_variant{-1};
// The smallest chunk any scratch was sized for (qsa_decode_attn_scratch_floats), 0 before the first sizing: a
// smaller chunk has more partials per query than such a scratch holds.
std::atomic<int> g_sized_chunk{0};

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

int chunk_of(int variant) { return variant == kQsaAttnChunk32 ? 32 : CHUNK; }

uint64_t scratch_floats_for(int ch, int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + ch - 1) / ch;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

// STRATA_QSA_ATTN_BATCH_OLD=1: the previous kernel for n_q > 1 only (batched prefill, MTP and verify windows), the
// prefetching one for decode.  The prefetching kernel's 48 KB of static shared memory leave 2 blocks per SM against
// ~5 for the previous one's 16 KB; both chunk by 64 with the same stride and partials, so the choice is bitwise
// neutral and only an A/B of STRATA_PREFILL_TIMING can say which the batch should take.
bool batch_old() {
    static const bool on = env_on("STRATA_QSA_ATTN_BATCH_OLD");
    return on;
}

// cp.async copies whole 16-byte pieces of a V row (8 bytes for the int8 scales): every row length is a multiple of
// that, so the pool base decides.  A pool that is not aligned keeps the previous kernel rather than faulting.
bool prefetch_aligned(const QsaAttnPools& p, int kv_mode) {
    const auto aligned = [](const void* x, uintptr_t n) { return reinterpret_cast<uintptr_t>(x) % n == 0; };
    if (kv_mode == 2) return aligned(p.v_q4, 16);
    if (kv_mode == 1) return aligned(p.v_q, 16) && aligned(p.v_scale, 8);
    return aligned(p.v_pool, 16);
}

// Both entry points: `n_q` queries (the single-query form is n_q = 1, whose offsets are all zero).
void launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* scratch, float* attn, int64_t n_q, cudaStream_t st, const QsaAttnGate& gate) {
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr ? 1 : 0);
    int variant = qsa_decode_attn_variant();
    const int sized = g_sized_chunk.load(std::memory_order_relaxed);
    if (sized > 0 && chunk_of(variant) < sized) {
        // set after a scratch was sized for bigger chunks (qsa_decode_attn_set_variant): it would not hold these
        static std::atomic<bool> told{false};
        if (!told.exchange(true))
            std::fprintf(stderr, "qsa_decode_attn: variant %d needs more scratch than was sized; using 64-cell chunks\n",
                         variant);
        variant = kQsaAttnPrefetch;
    }
    if (variant == kQsaAttnPrefetch && n_q > 1 && batch_old()) variant = kQsaAttnOld;
    if (variant != kQsaAttnOld && !prefetch_aligned(pools, kv_mode)) variant = kQsaAttnOld;
    const int ch = chunk_of(variant);
    const int n_chunks = (int) ((cap + ch - 1) / ch);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) scratch_floats_for(ch, cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size, c = (int) cap;
#define STRATA_QSA_CHUNK_ARGS q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, c, stride
    if (variant == kQsaAttnOld) {
        if (kv_mode == 2) attn_chunk_kernel<2><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else if (kv_mode == 1) attn_chunk_kernel<1><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else attn_chunk_kernel<0><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
    } else if (ch == 64) {
        if (kv_mode == 2) attn_chunk_pf_kernel<2, 64><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else if (kv_mode == 1) attn_chunk_pf_kernel<1, 64><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else attn_chunk_pf_kernel<0, 64><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
    } else {
        if (kv_mode == 2) attn_chunk_pf_kernel<2, 32><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else if (kv_mode == 1) attn_chunk_pf_kernel<1, 32><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
        else attn_chunk_pf_kernel<0, 32><<<grid, THREADS, 0, st>>>(STRATA_QSA_CHUNK_ARGS);
    }
#undef STRATA_QSA_CHUNK_ARGS
    const dim3 mgrid((unsigned) s.n_head, (unsigned) n_q);
    if (gate.kind == kQsaGateF64)
        attn_merge_gate_kernel<kQsaGateF64><<<mgrid, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn, stride,
                                                                  gate.q_full, gate.out);
    else if (gate.kind == kQsaGateNativeF32)
        attn_merge_gate_kernel<kQsaGateNativeF32><<<mgrid, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn,
                                                                        stride, gate.q_full, gate.out);
    else
        attn_merge_kernel<<<mgrid, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn, stride);
}

void check_gate(const QsaAttnGate& gate, const char* who) {
    if (gate.kind == kQsaGateNone) return;
    if ((gate.kind != kQsaGateF64 && gate.kind != kQsaGateNativeF32) || gate.q_full == nullptr || gate.out == nullptr) {
        std::fprintf(stderr, "%s: a folded gate needs its kind, q_full and out\n", who);
        std::exit(1);
    }
}

}  // namespace

bool qsa_decode_attn_gate_fold() {
    static const bool on = env_on("STRATA_QSA_MERGE_GATE");
    return on;
}

int qsa_decode_attn_variant() {
    int v = g_variant.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* chunk = std::getenv("STRATA_QSA_CHUNK");
        v = env_on("STRATA_OLD_QSA_ATTN") ? kQsaAttnOld
          : (chunk != nullptr && std::atoi(chunk) == 32) ? kQsaAttnChunk32 : kQsaAttnPrefetch;
        g_variant.store(v, std::memory_order_relaxed);
    }
    return v;
}

void qsa_decode_attn_set_variant(int variant) {
    if (variant < kQsaAttnPrefetch || variant > kQsaAttnChunk32) {
        std::fprintf(stderr, "qsa_decode_attn_set_variant: unknown variant %d\n", variant);
        std::exit(1);
    }
    g_variant.store(variant, std::memory_order_relaxed);
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                           const QsaAttnGate& gate) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    check_gate(gate, "qsa_decode_attn_batch");
    launch(q, pools, ids, steps, cap, s, scratch, attn, n_q, (cudaStream_t) stream, gate);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int ch = chunk_of(qsa_decode_attn_variant());
    int prev = g_sized_chunk.load(std::memory_order_relaxed);
    while ((prev == 0 || ch < prev) && !g_sized_chunk.compare_exchange_weak(prev, ch, std::memory_order_relaxed)) {}
    return scratch_floats_for(ch, cap, s);
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream,
                          const QsaAttnGate& gate) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr ? 1 : 0);
    if (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    check_gate(gate, "qsa_decode_attn");
    launch(q, pools, ids, step, cap, s, scratch, attn, 1, (cudaStream_t) stream, gate);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
