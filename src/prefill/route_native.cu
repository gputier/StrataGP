// src/prefill/route_native.cu - the token path's native router (src/kernels/cuda/native_router.cu) over a prompt
// chunk, for STRATA_PREFILL_ROUTE_DECODE=1 (B8): the same source, a warp per token instead of a launch per token,
// compiled with the same --use_fast_math (CMakeLists.txt), so a prompt token picks the experts and weights the token
// path would.  prefill_kernels_parity checks it against native_router_top10, bitwise.
//
// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
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
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
// native_router.cu's `route` for token blockIdx.x * 8 + threadIdx.y (the 32x8 block of the original, every row
// active); the arithmetic is copied unchanged, the barrier after the loads is dropped (it orders nothing).
__launch_bounds__(256, 1)
__global__ void route_native_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                    float* __restrict__ weights, int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * 8 + threadIdx.y;
    if (t >= T) return;
    logits += t * 512;
    ids += t * 10;
    weights += t * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
}  // namespace

void route_native(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream) {
    if (T <= 0) return;
    route_native_kernel<<<(unsigned) ((T + 7) / 8), dim3(32, 8), 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill route_native: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::prefill
