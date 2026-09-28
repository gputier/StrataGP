// Numerical contract: pinned ggml/src/ggml-cuda/rope.cu, rope_multi/rope_yarn.
// Text positions are equal across the four IMRoPE sections, so section routing
// reduces to the one position associated with each contiguous row.
// MIT License
//
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
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
template <bool F64>
__global__ void apply(const float* x, float* out, int rows, int width,
                      int n_rot, float theta_scale, const int* positions, const int32_t* mtab,
                      double freq_base64) {
    const int row = blockIdx.y;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || pair >= width / 2) return;
    const size_t start = size_t(row) * width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    float c, s;
    if constexpr (F64) {
        // #3 opt-in (STRATA_ROPE_F64): the angle and its sine in float64, the table path's arithmetic
        // (rope.cu build_rope_table) - --use_fast_math does not touch double-precision functions.  A separate
        // instantiation, so the default kernel is compiled exactly as before.
        const double theta = mrope_pos(mtab, positions[row], pair) * pow(freq_base64, -2.0 * pair / n_rot);
        double sd, cd;
        sincos(theta, &sd, &cd);
        c = float(cd);
        s = float(sd);
    } else {
        const float theta = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, float(pair));
        c = cosf(theta);
        s = sinf(theta);
    }
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}
}
namespace { std::atomic<const int32_t*> mrope_tab{nullptr}; }
void mrope_table_set(const int32_t* device_table) { mrope_tab.store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return mrope_tab.load(std::memory_order_relaxed); }
void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
// Match pinned host-side float powf before device fast powf/trigonometry.
float native_rope_theta_scale(float freq_base, int n_rot) { return powf(freq_base, -2.0f / n_rot); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }
namespace { std::atomic<int> f64_angle{-1}; }   // -1: not set yet, STRATA_ROPE_F64 decides
void native_rope_set_f64_angle(bool value) { f64_angle.store(value ? 1 : 0, std::memory_order_relaxed); }
bool native_rope_f64_angle() {
    int v = f64_angle.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("STRATA_ROPE_F64");
        v = e != nullptr && e[0] != '\0' && e[0] != '0' ? 1 : 0;
        f64_angle.store(v, std::memory_order_relaxed);
    }
    return v != 0;
}
void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, float freq_base, const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 ||
        !std::isfinite(freq_base) || freq_base <= 1.0f ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    const float theta_scale = native_rope_theta_scale(freq_base, n_rot);
    const dim3 grid((head_dim / 2 + 127) / 128, rows);
    if (native_rope_f64_angle())
        apply<true><<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(x, out, rows, head_dim, n_rot, theta_scale,
                                                                          positions, mrope_table(), double(freq_base));
    else
        apply<false><<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(x, out, rows, head_dim, n_rot, theta_scale,
                                                                           positions, mrope_table(), 0.0);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
