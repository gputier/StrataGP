// src/kernels/cpu/native_expert_test_util.hpp - what the native expert tests share: random weight blocks.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>

// random bytes, then a sane fp16 scale at the front of every block (the i-quants and Q8_0 start their block with it)
inline void fill_blocks(uint8_t* p, size_t bytes, int type, std::mt19937_64& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((0.5f + (float) (rng() >> 40) * (1.f / 16777216.f)) * 1e-3f);
        std::memcpy(p + o, &h, 2);
    }
}
