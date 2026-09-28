// src/kernels/cuda/shared_expert.cu - P2.S2: the shared expert.
//
// `ref/moe.py::shared_expert`, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and `docs/semantics.md` records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The legacy canonical projections use their explicit Q8_0/Q8_K activation images. The optional native
// BF16 path affects only the scalar gate: it reads the original F32 input and uses pinned CUDA MMVF plus
// an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1 MMVQ and FP32 SwiGLU.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

// `f32_to_f16` used to live here as a private copy.  Round 198 found it - and the identical one in
// `quantize_act.cu` - returning a NaN for every FINITE value that overflows fp16 (1e30, 65536, 1e45) instead
// of saturating to inf, because `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent.
// This file's fixture is all O(1), so it could not see it.  The conversion now lives in one place, with the
// regimes spelled out.

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float) (x / (1.0 + exp(-x))) * up[i];
}

__global__ void native_swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the default legacy arithmetic in this file.
    out[i] = __fdividef(gate[i], 1.0f + __expf(-gate[i])) * up[i];
}

__global__ void to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f16_from_f32(in[i]);
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).
//
// **ONE THREAD WAS THE LARGEST SINGLE KERNEL IN THE MODEL, AND THE COMMENT DEFENDING IT SAID WHY IT SHOULD
// NOT BE.**  It read: "n_embd is 2560, so this is a short serial loop and the parallelism is elsewhere."
// 2560 iterations is not short when each one is a DOUBLE multiply-add.  This card's double rate is **1/64** of
// its float rate, so the loop is a 2560-long dependency chain through a 1/64-rate unit - tens of microseconds
// from the chain alone, before counting 2560 sequential global loads issued by one thread with no coalescing.
//
// MEASURED, and this is what the number looks like from outside: `shared_expert` is **0.2374 ms of a
// 0.8883 ms block - 26.7% of the whole block, 11.4 ms per token** - for three projections that read 5.5 MB and
// whose memory floor is 0.009 ms.  It is seven kernels, and six of them are GEMVs or elementwise passes over
// 640-2560 elements.  This is the one that cannot be explained by bandwidth.
//
// THE DOUBLE ACCUMULATOR IS KEPT.  `ref/gr.py` reduces in float64 and `block_sum` in `gr.cu` exists for the
// same reason - the summation order of a tree is not the reference's, so the defence is enough precision that
// the order stops mattering.  What changes is the CHAIN: ten terms per thread instead of 2560, then a
// tree-reduce.  Through a sigmoid, a 1e-16 relative difference in the dot is not a difference.
//
// **BOTH OPERANDS ARE BF16, AND THAT IS THE REFERENCE'S CONTRACT, NOT A STORAGE DETAIL.**  `gate_inp_shexp` is a
// BF16 weight, so ggml converts the activation to its `vec_dot_type` - BF16 - exactly as it does for the router
// (`docs/activation-contract.md`).  This kernel used to take `const float* w` while the LOADER had already
// re-rounded the tensor to 2 B/elem, so it read 2560 floats out of a 5120-byte buffer: 5120 B past the end of
// the tensor, inside the 4.5 GiB arena, where nothing faults.  The dot came out astronomically large, sigmoid
// saturated to 1.0, and `scale_kernel` multiplied the shared expert's output by it - which is why the FIRST
// symptom was 2560 non-finite outputs rather than a wrong scalar.
//
// A wrong scalar here is the quiet failure mode: sigmoid bounds the damage to [0,1], so a gate that should be
// 0.5 and reads 1.0 scales the shared expert by 2x and produces perfectly finite, perfectly plausible logits.
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

// One token's gate by one block; `scalar_gate_kernel` runs it once, `scalar_gate_rows_kernel` once per block.
__device__ __forceinline__ void scalar_gate_row(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                                float* __restrict__ out, int n_embd) {
    __shared__ double scratch[8];   // 8 warps: the launch is <<<1, 256>>>
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    __syncthreads();
    const int nw = ((int) blockDim.x + 31) >> 5;
    if (warp == 0) {
        acc = (threadIdx.x < nw) ? scratch[threadIdx.x] : 0.0;
        acc = warp_sum_d(acc);
        if (threadIdx.x == 0) out[0] = (float) (1.0 / (1.0 + exp(-acc)));
    }
}

__global__ void scalar_gate_kernel(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                   float* __restrict__ out, int n_embd) {
    scalar_gate_row(x_bf16, w_bf16, out, n_embd);
}

// #19: the verify window's per-token gate launches as one, a block per token (blocks share nothing).
__global__ void scalar_gate_rows_kernel(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                        float* __restrict__ out, int n_embd) {
    scalar_gate_row(x_bf16 + (size_t) blockIdx.x * n_embd, w_bf16, out + blockIdx.x, n_embd);
}

__global__ void scale_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= g[0];
}

__global__ void native_scalar_sigmoid_kernel(float* gate) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations without changing legacy kernels'
    // compilation flags. The dot product was already reduced by the pinned native MMVF implementation.
    gate[0] = __fdividef(1.0f, 1.0f + __expf(-gate[0]));
}

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared, float* __restrict__ y, int n_embd,
                                   int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
    for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }

bool shared_expert_native_bf16() { return native_bf16; }

namespace {
__global__ void scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t) t * n + i] *= g[t];
}

// #19: `native_scalar_sigmoid_kernel` per token and then `scale_rows_kernel`, as one launch.  Every thread
// evaluates the sigmoid's own expression on the dot (this file, these flags: the same float) instead of one
// thread writing it back first.  `dot` is left as the dot.
__global__ void sigmoid_scale_rows_kernel(float* __restrict__ out, const float* __restrict__ dot, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = __fdividef(1.0f, 1.0f + __expf(-dot[t]));
    out[(size_t) t * n + i] *= g;
}

// #44 (E1) and #19: `moe_combine_kernel`'s window form, each routed row read where it was left (see
// native_moe_combine_window, which this mirrors for the double-accumulating combination).  This file is built
// without --use_fast_math, like the kernels it replaces, so the explicit single-rounding intrinsics below are
// their instructions: `+0 + hit` (moe_hit_add), `out * g` (the scale), the FMAs in expert order, the shared add.
constexpr int kWindowMaxK = 16;
constexpr int kWindowMaxRows = 128;
constexpr int kWindowThreads = 128;

template <int GATE>
__launch_bounds__(kWindowThreads)
__global__ void moe_combine_window_kernel(const float* y_miss, const float* __restrict__ hit_out,
                                          const int32_t* __restrict__ hit_dst, const int32_t* __restrict__ hit_count,
                                          const float* __restrict__ weights, const float* __restrict__ shared,
                                          const float* __restrict__ gate, float* __restrict__ y, int n4, int k) {
    __shared__ unsigned char s_gpu[kWindowMaxRows];
    const int t = blockIdx.y, rows = (int) gridDim.y * k;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) s_gpu[i] = 0;
    __syncthreads();
    if (hit_count != nullptr) {
        const int count = min(*hit_count, rows);
        for (int h = threadIdx.x; h < count; h += blockDim.x) {
            const int row = hit_dst[h];
            if (row >= 0 && row < rows) s_gpu[row] = 1;
        }
    }
    __syncthreads();
    const int c = (int) blockIdx.x * kWindowThreads + (int) threadIdx.x;
    if (c >= n4) return;
    float4 p[kWindowMaxK];
#pragma unroll
    for (int e = 0; e < kWindowMaxK; ++e) {
        if (e < k) {
            const int row = t * k + e;
            const size_t at = (size_t) row * (size_t) n4 + (size_t) c;
            if (s_gpu[row]) {
                const float4 h = reinterpret_cast<const float4*>(hit_out)[at];
                p[e] = make_float4(__fadd_rn(0.0f, h.x), __fadd_rn(0.0f, h.y), __fadd_rn(0.0f, h.z),
                                   __fadd_rn(0.0f, h.w));
            } else {
                p[e] = __ldcv(reinterpret_cast<const float4*>(y_miss) + at);
            }
        }
    }
    float4 sh = reinterpret_cast<const float4*>(shared)[(size_t) t * (size_t) n4 + (size_t) c];
    if constexpr (GATE != 0) {
        const float g = GATE == 2 ? __fdividef(1.0f, 1.0f + __expf(-gate[t])) : gate[t];
        sh = make_float4(__fmul_rn(sh.x, g), __fmul_rn(sh.y, g), __fmul_rn(sh.z, g), __fmul_rn(sh.w, g));
    }
    const float* w = weights + (size_t) t * (size_t) k;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
#pragma unroll
    for (int e = 0; e < kWindowMaxK; ++e) {
        if (e < k) {
            const double we = (double) w[e];
            a0 = __fma_rn(we, (double) p[e].x, a0);
            a1 = __fma_rn(we, (double) p[e].y, a1);
            a2 = __fma_rn(we, (double) p[e].z, a2);
            a3 = __fma_rn(we, (double) p[e].w, a3);
        }
    }
    reinterpret_cast<float4*>(y)[(size_t) t * (size_t) n4 + (size_t) c] =
        make_float4(__double2float_rn(__dadd_rn(a0, (double) sh.x)), __double2float_rn(__dadd_rn(a1, (double) sh.y)),
                    __double2float_rn(__dadd_rn(a2, (double) sh.z)), __double2float_rn(__dadd_rn(a3, (double) sh.w)));
}
}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    cudaStream_t cs = (cudaStream_t) stream;
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    const int n = (int) (n_ff * n_tok);
    native_swiglu_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate, up, gate, n);
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    for (int t = 0; t < n_tok; ++t) {
        if (native_bf16) {
            bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
            native_scalar_sigmoid_kernel<<<1, 1, 0, cs>>>(g + t);
        } else {
            scalar_gate_kernel<<<1, 256, 0, cs>>>(x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd);
        }
    }
    scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok), THREADS, 0, cs>>>(
        out, g, (int) n_embd);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi: ") + cudaGetErrorString(e));
}

SharedGate shared_expert_multi_batched(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                                       const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out,
                                       int64_t n_embd, int64_t n_ff, void* stream, bool defer_gate) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream ||
        (!native_bf16 && !x_bf16))
        throw std::invalid_argument("shared_expert_multi_batched: needs 1..8 tokens, native weights, scratch, a stream "
                                    "and, for the BF16 gate, the rounded input");
    cudaStream_t cs = (cudaStream_t) stream;
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    native_swiglu_quantize_q8_1(gate, up, nw.q8_1, (int) n_ff, n_tok, stream);   // `gate` keeps the projection
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    // the scalar gate for all the tokens in one launch: g[t] is the dot (native) or already the sigmoid (BF16)
    if (native_bf16) bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
    else scalar_gate_rows_kernel<<<(unsigned) n_tok, 256, 0, cs>>>(x_bf16, gate_inp_bf16, g, (int) n_embd);
    SharedGate mode = native_bf16 ? SharedGate::Logit : SharedGate::Value;
    if (!defer_gate) {
        const dim3 grid((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok);
        if (native_bf16) sigmoid_scale_rows_kernel<<<grid, THREADS, 0, cs>>>(out, g, (int) n_embd);
        else scale_rows_kernel<<<grid, THREADS, 0, cs>>>(out, g, (int) n_embd);
        mode = SharedGate::Applied;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi_batched: ") + cudaGetErrorString(e));
    return mode;
}

void moe_combine_window(const float* y_miss, const float* hit_out, const int32_t* hit_dst, const int32_t* hit_count,
                        const float* weights, const float* shared, const float* shared_gate, SharedGate gate_mode,
                        float* y, int64_t n_embd, int64_t k, int n_tok, void* stream) {
    const int mode = (int) gate_mode;
    if (!stream || !y_miss || !weights || !shared || !y || n_embd <= 0 || n_embd % 4 != 0 || n_embd > INT_MAX ||
        k < 1 || k > kWindowMaxK || n_tok < 1 || (int64_t) n_tok * k > kWindowMaxRows || mode < 0 || mode > 2 ||
        (mode != 0 && !shared_gate) || (hit_count != nullptr) != (hit_dst != nullptr) || (hit_count && !hit_out))
        throw std::invalid_argument("moe_combine_window: needs a stream, a width that is a multiple of 4, 1..16 "
                                    "experts, at most 128 rows, the shared output and its gate");
    auto misaligned = [](const void* p) { return p != nullptr && ((uintptr_t) p & 15u) != 0; };
    if (misaligned(y_miss) || misaligned(hit_out) || misaligned(shared) || misaligned(y))
        throw std::invalid_argument("moe_combine_window: rows must be 16-byte aligned");
    const int n4 = (int) (n_embd / 4);
    const dim3 grid((unsigned) ((n4 + kWindowThreads - 1) / kWindowThreads), (unsigned) n_tok);
    const cudaStream_t cs = (cudaStream_t) stream;
    if (mode == 0)
        moe_combine_window_kernel<0><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights,
                                                                     shared, shared_gate, y, n4, (int) k);
    else if (mode == 1)
        moe_combine_window_kernel<1><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights,
                                                                     shared, shared_gate, y, n4, (int) k);
    else
        moe_combine_window_kernel<2><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights,
                                                                     shared, shared_gate, y, n4, (int) k);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("moe_combine_window: ") + cudaGetErrorString(e));
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH.  This used to be four `cudaMalloc`s and four `cudaFree`s per call, which
    // is illegal during stream capture AND a token-path allocation - P2.T10 forbids both.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    const unsigned g_ff = (unsigned) ((n_ff + THREADS - 1) / THREADS);
    const unsigned g_embd = (unsigned) ((n_embd + THREADS - 1) / THREADS);

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection)
        native_swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);
    else
        swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);

    // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
    // what `ggml_mul_mat` does for every matmul in the model.  It used to be rounded to fp16 with no
    // justification beyond "the kernel takes fp16".
    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else if (down_form.act_kind == 1) {
        if (n_ff % 256 != 0) {
            std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                 "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
            std::exit(1);
        }
        quantize_q8_K(gate, h_q8k, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else if (down_form.code_bits == 2) {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
    // state, not from anything the expert produced. The historical branch uses BF16-rounded inputs; the
    // native branch uses the pinned CUDA FP32 activation contract.
    // `<<<1, 256>>>`: one block, because the output is ONE scalar and a second block would only add a global
    // round trip.  256 threads is the reduction's width, not the problem's size.
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        native_scalar_sigmoid_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(g);
    } else {
        scalar_gate_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(x_bf16, gate_inp_bf16, g, (int) n_embd);
    }
    scale_kernel<<<g_embd, THREADS, 0, (cudaStream_t) stream>>>(out, g, (int) n_embd);

    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "shared_expert: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_embd + THREADS - 1) / THREADS);
    moe_combine_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(parts, weights, shared, y, (int) n_embd,
                                                                   (int) k, shared != nullptr);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "moe_combine: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
