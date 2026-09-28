// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k) {
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float sum = parts[col] * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
// ---- #44 (E1) and #19: the verify window's combination in one launch.
//
// It replaces three steps: `copy_from_mapped` (every routed row, the GPU's zero rows included, over PCIe),
// `moe_hit_add` (parts[row] += hit_out[row] on the GPU's rows) and one `combine` per token.  Each routed row now
// comes from where it was left - a CPU row straight from the mapped `y_miss`, a GPU row from `hit_out` - and the
// arithmetic is the three steps' own, value for value:
//
//   * a GPU row is `+0 + hit` (the pool zeroed that row of y_miss and moe_hit_add added to it);
//   * the shared term, when its gate is folded in here, is `out * g` rounded on its own, and g the sigmoid
//     `__fdividef(1, 1 + __expf(-dot))` - both from shared_expert.cu, which is NOT built with --use_fast_math;
//   * then `combine` above: the first product, the FMAs in expert order, the shared term added once.
//
// The first two are not this file's flush-to-zero arithmetic, so their PTX is spelled out; the combination is
// this file's, written with explicit single-rounding intrinsics so nothing can be contracted differently.
constexpr int kWindowMaxK = 16;
constexpr int kWindowMaxRows = 128;
constexpr int kWindowThreads = 128;

__device__ __forceinline__ float add_no_ftz(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float mul_no_ftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float sigmoid_no_ftz(float x) {
    float r;
    asm("{\n\t"
        ".reg .f32 e, s, one;\n\t"
        "mul.rn.f32 e, %1, 0fBFB8AA3B;\n\t"   // -log2(e) * x, as __expf(-x) scales it
        "ex2.approx.f32 e, e;\n\t"
        "add.rn.f32 s, e, 0f3F800000;\n\t"
        "mov.f32 one, 0f3F800000;\n\t"
        "div.approx.f32 %0, one, s;\n\t"
        "}"
        : "=f"(r) : "f"(x));
    return r;
}

// GATE: 0 = `shared` is final, 1 = multiply it by gate[t], 2 = gate[t] is the dot, sigmoid first.
template <int GATE>
__launch_bounds__(kWindowThreads)
__global__ void combine_window(const float* y_miss, const float* __restrict__ hit_out,
                               const int32_t* __restrict__ hit_dst, const int32_t* __restrict__ hit_count,
                               const float* __restrict__ weights, const float* __restrict__ shared,
                               const float* __restrict__ gate, float* __restrict__ output, int n4, int k) {
    __shared__ unsigned char s_gpu[kWindowMaxRows];
    const int t = blockIdx.y, rows = int(gridDim.y) * k;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) s_gpu[i] = 0;
    __syncthreads();
    if (hit_count != nullptr) {
        const int count = min(*hit_count, rows);   // moe_hit_add's grid was `rows` tall
        for (int h = threadIdx.x; h < count; h += blockDim.x) {
            const int row = hit_dst[h];
            if (row >= 0 && row < rows) s_gpu[row] = 1;
        }
    }
    __syncthreads();
    const int c = int(blockIdx.x) * kWindowThreads + int(threadIdx.x);
    if (c >= n4) return;
    // Every row's load first, so the PCIe reads of the CPU rows are in flight together.
    float4 p[kWindowMaxK];
#pragma unroll
    for (int e = 0; e < kWindowMaxK; ++e) {
        if (e < k) {
            const int row = t * k + e;
            const size_t at = size_t(row) * size_t(n4) + size_t(c);
            if (s_gpu[row]) {
                const float4 h = reinterpret_cast<const float4*>(hit_out)[at];
                p[e] = make_float4(add_no_ftz(0.0f, h.x), add_no_ftz(0.0f, h.y), add_no_ftz(0.0f, h.z),
                                   add_no_ftz(0.0f, h.w));
            } else {
                p[e] = __ldcv(reinterpret_cast<const float4*>(y_miss) + at);
            }
        }
    }
    float4 sh = reinterpret_cast<const float4*>(shared)[size_t(t) * size_t(n4) + size_t(c)];
    if constexpr (GATE != 0) {
        const float g = GATE == 2 ? sigmoid_no_ftz(gate[t]) : gate[t];
        sh = make_float4(mul_no_ftz(sh.x, g), mul_no_ftz(sh.y, g), mul_no_ftz(sh.z, g), mul_no_ftz(sh.w, g));
    }
    const float* w = weights + size_t(t) * size_t(k);
    const float w0 = w[0];
    float4 sum = make_float4(__fmul_rn(p[0].x, w0), __fmul_rn(p[0].y, w0), __fmul_rn(p[0].z, w0),
                             __fmul_rn(p[0].w, w0));
#pragma unroll
    for (int e = 1; e < kWindowMaxK; ++e) {
        if (e < k) {
            const float we = w[e];
            sum.x = __fmaf_rn(p[e].x, we, sum.x);
            sum.y = __fmaf_rn(p[e].y, we, sum.y);
            sum.z = __fmaf_rn(p[e].z, we, sum.z);
            sum.w = __fmaf_rn(p[e].w, we, sum.w);
        }
    }
    sum = make_float4(__fadd_rn(sum.x, sh.x), __fadd_rn(sum.y, sh.y), __fadd_rn(sum.z, sh.z), __fadd_rn(sum.w, sh.w));
    reinterpret_cast<float4*>(output)[size_t(t) * size_t(n4) + size_t(c)] = sum;
}

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_window(const float* y_miss, const float* hit_out, const int32_t* hit_dst,
                               const int32_t* hit_count, const float* weights, const float* shared,
                               const float* shared_gate, int gate_mode, float* output, int64_t n_embd, int64_t k,
                               int n_tok, void* stream) {
    // k >= 2: with one expert the single-token kernel's product and shared add may be contracted into one FMA.
    if (!stream || n_embd <= 0 || n_embd % 4 != 0 || n_embd > std::numeric_limits<int>::max() || k < 2 || k > 15 ||
        n_tok < 1 || int64_t(n_tok) * k > kWindowMaxRows || gate_mode < 0 || gate_mode > 2 ||
        (gate_mode != 0 && !shared_gate) || !shared || (hit_count != nullptr) != (hit_dst != nullptr))
        throw std::invalid_argument("native MoE window combine requires a stream, a width that is a multiple of 4, "
                                    "2..15 experts, at most 128 rows, the shared output and its gate");
    const size_t row_bytes = size_t(n_embd) * sizeof(float), rows = size_t(n_tok) * size_t(k);
    const size_t out_bytes = row_bytes * size_t(n_tok), part_bytes = row_bytes * rows;
    auto aligned16 = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; };
    const bool gpu_rows = hit_count != nullptr;
    if (!valid_span(y_miss, part_bytes) || !aligned16(y_miss) ||
        (gpu_rows && (!valid_span(hit_out, part_bytes) || !aligned16(hit_out))) ||
        !valid_span(shared, out_bytes) || !aligned16(shared) ||
        !valid_span(output, out_bytes) || !aligned16(output) || !valid_span(weights, rows * sizeof(float)) ||
        (gpu_rows && overlap(output, out_bytes, hit_out, part_bytes)) || overlap(output, out_bytes, shared, out_bytes) ||
        overlap(output, out_bytes, weights, rows * sizeof(float)) ||
        (shared_gate && overlap(output, out_bytes, shared_gate, size_t(n_tok) * sizeof(float))))
        throw std::invalid_argument("native MoE window combine requires 16-byte aligned rows and a disjoint output");
    const int n4 = int(n_embd / 4);
    const dim3 grid(unsigned((n4 + kWindowThreads - 1) / kWindowThreads), unsigned(n_tok));
    const cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (gate_mode == 0)
        combine_window<0><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights, shared,
                                                           shared_gate, output, n4, int(k));
    else if (gate_mode == 1)
        combine_window<1><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights, shared,
                                                           shared_gate, output, n4, int(k));
    else
        combine_window<2><<<grid, kWindowThreads, 0, cs>>>(y_miss, hit_out, hit_dst, hit_count, weights, shared,
                                                           shared_gate, output, n4, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
