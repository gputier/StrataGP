// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
__device__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__global__ void append(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, const int32_t* __restrict__ mtab,
                        uint16_t* __restrict__ pooled16, uint16_t* __restrict__ dead16) {
    const int pos = *pos_dev, d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * D + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        // The spare's four gather indices all name cell zero. Completed blocks
        // use chronological slices; each graph ADD materializes an F32 sum.
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
        mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int b = pos / R;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const float theta = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        const float c = cosf(theta), s = sinf(theta);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    pooled[std::size_t(b) * D + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[std::size_t(b + 1) * D + d] = dead[d];
    if (d == 0 && pos != 0) *block_pos = rope_pos;
    if (pooled16 != nullptr) {   // O6 (#21): the fp16 shadow of the rows written above (integer rounding, exact)
        const uint16_t y16 = f16_from_f32(y);
        pooled16[std::size_t(b) * D + d] = y16;
        if (pos == 0) dead16[d] = y16;
        else pooled16[std::size_t(b + 1) * D + d] = f16_from_f32(dead[d]);
    }
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw,D*4},{relative_pos_device,4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    if ((b.pooled16 == nullptr) != (b.dead16 == nullptr))
        throw std::invalid_argument("native QSA indexer: the fp16 shadow needs both pooled16 and dead16");
    if (b.pooled16 != nullptr) {   // O6: the shadow is disjoint from everything above and from itself
        const Span shadow[] = {{b.pooled16,std::size_t(max_cells/R+1)*D*2},{b.dead16,D*2}};
        for (const auto& span : shadow) {
            validate(span);
            for (const auto& other : spans)
                if (overlaps(span, other)) throw std::invalid_argument("native QSA indexer buffers overlap");
        }
        if (overlaps(shadow[0], shadow[1])) throw std::invalid_argument("native QSA indexer buffers overlap");
    }
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    append<<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,relative_pos_device,pos_base,gamma,epsilon,
        b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table(),b.pooled16,b.dead16);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels

// Issue #40: the prompt path appended its chunk one launch per cell (8192 x 12 launches per chunk). These two
// kernels leave the buffers exactly as that loop does. Block i of `append_chunk` is one event of the chunk - the
// spare key when the chunk starts at cell 0, or one block the chunk completes - with `append`'s arithmetic,
// statement for statement, on the four raw values the loop would have pooled: the tail as the chunk found it for
// the cells before the chunk, the chunk's own keys (rounded through F16 as SET_ROWS stores them) for the rest.
// An event reads nothing another one writes, so they run side by side. What only the LAST event of the loop
// leaves behind - the spare's row after the last completed block, block_pos - and the tail (each slot's last
// cell) are written by `append_chunk_tail`, after every event has read the old tail.
namespace strata::kernels {
namespace {
__global__ void append_chunk(const float* __restrict__ raw, int cell0, int b0, int spare, int spare_row,
                             int pos_base, const float* __restrict__ gamma, float epsilon,
                             const float* __restrict__ tail, float* __restrict__ dead, float* __restrict__ pooled,
                             float theta_scale, const int32_t* __restrict__ mtab,
                             uint16_t* __restrict__ pooled16, uint16_t* __restrict__ dead16) {
    const int d = threadIdx.x;
    const bool is_spare = spare && blockIdx.x == 0;
    const int b = is_spare ? 0 : b0 + int(blockIdx.x) - spare;
    const int pos = is_spare ? 0 : R * b + R - 1;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        // The spare's four gather indices all name cell zero (the chunk's first key).
        float x[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const int cell = is_spare ? 0 : R * b + j;
            x[j] = cell >= cell0 ? __half2float(__float2half_rn(raw[std::size_t(cell - cell0) * D + d]))
                                 : tail[j * D + d];
        }
        float sum = x[0];
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, x[j]);
        mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const float theta = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        const float c = cosf(theta), s = sinf(theta);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    if (is_spare) {
        dead[d] = y;
        if (spare_row) pooled[d] = y; // row 0 is the spare until the chunk completes block 0
    } else {
        pooled[std::size_t(b) * D + d] = y;
    }
    if (pooled16 != nullptr) {   // O6 (#21): the fp16 shadow of the rows written above, as `append` keeps it
        const uint16_t y16 = f16_from_f32(y);
        if (is_spare) {
            dead16[d] = y16;
            if (spare_row) pooled16[d] = y16;
        } else {
            pooled16[std::size_t(b) * D + d] = y16;
        }
    }
}
__global__ void append_chunk_tail(const float* __restrict__ raw, int cell0, int n, int last_b, int pos_base,
                                  float* __restrict__ tail, const float* __restrict__ dead,
                                  float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                                  uint16_t* __restrict__ pooled16) {
    const int d = threadIdx.x;
    if (last_b >= 0) {
        pooled[std::size_t(last_b + 1) * D + d] = dead[d];
        if (pooled16 != nullptr) pooled16[std::size_t(last_b + 1) * D + d] = f16_from_f32(dead[d]);
        if (d == 0) *block_pos = pos_base + R * last_b;
    }
    const int last = cell0 + n - 1;
    for (int slot = 0; slot < R - 1; ++slot) {
        const int c = last - ((last - slot) % R + R) % R; // the chunk's last cell in this slot, if any
        if (c >= cell0) tail[slot * D + d] = __half2float(__float2half_rn(raw[std::size_t(c - cell0) * D + d]));
    }
}
} // namespace

void native_qsa_indexer_append_chunk(const float* raw, int64_t n, int64_t cell0, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f || n < 0 || cell0 < 0)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    // cells at or past the capacity are ignored, as the per-cell kernel ignores them
    if (cell0 >= max_cells) return;
    if (n > max_cells - cell0) n = max_cells - cell0;
    if (n == 0) return;
    const Span spans[] = {{raw,std::size_t(n)*D*4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    // O6 (#21, --idx-fp16): the chunk keeps the fp16 shadow as the per-cell kernel does
    if ((b.pooled16 == nullptr) != (b.dead16 == nullptr))
        throw std::invalid_argument("native QSA indexer: the fp16 shadow needs both pooled16 and dead16");
    if (b.pooled16 != nullptr) {
        const Span shadow[] = {{b.pooled16,std::size_t(max_cells/R+1)*D*2},{b.dead16,D*2}};
        for (const auto& span : shadow) {
            validate(span);
            for (const auto& other : spans)
                if (overlaps(span, other)) throw std::invalid_argument("native QSA indexer buffers overlap");
        }
        if (overlaps(shadow[0], shadow[1])) throw std::invalid_argument("native QSA indexer buffers overlap");
    }
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    const int64_t end = cell0 + n;
    const int spare = cell0 == 0 ? 1 : 0;
    const int64_t completed = end / R - cell0 / R; // blocks b with cell0 <= R*b + R-1 < end
    const auto cs = static_cast<cudaStream_t>(stream);
    if (spare + completed > 0)
        append_chunk<<<unsigned(spare + completed),THREADS,0,cs>>>(raw,int(cell0),int(cell0/R),spare,
            completed == 0 ? 1 : 0,pos_base,gamma,epsilon,b.tail,b.dead,b.pooled,theta_scale,mrope_table(),
            b.pooled16,b.dead16);
    append_chunk_tail<<<1,D,0,cs>>>(raw,int(cell0),int(n),completed > 0 ? int(end/R-1) : -1,pos_base,b.tail,
        b.dead,b.pooled,b.block_pos,b.pooled16);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
