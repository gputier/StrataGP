// src/kernels/cpu/native_expert_row_invariance_test.cpp - a token's expert rows must not depend on how many tokens share
// its expert.  The hybrid engine decodes one token at a time (every CPU expert is a one-token job) but verifies a window
// of several rows, where an expert chosen by two or more rows is one multi-token job; if the two take different kernels
// (the multi-token AVX-512 / AVX-2 kernels from mt_min tokens on, ggml-cpu's vec_dot below), the same row rounds
// differently in the two paths and a near-tie greedy pick flips between MTP 0 and MTP 2.  Ornith's layer pairs at its
// geometry (n_embd 3072, width 1024): each row of an nt = 2 or 3 call against the same token alone, bit for bit.
// No GPU, no model: random blocks with sane scales and random activations quantized the way the engine quantizes them.
// On a CPU with neither AVX-512 nor AVX-2 every call is ggml's dot and the test passes without exercising a kernel
// choice: it prints which kernels the CPU offers.
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int64_t kH = 3072, kFF = 1024;   // n_embd, expert width
constexpr int kMaxT = 3;                   // a verify window of three rows
constexpr int kExperts = 6;

// random bytes, then a sane fp16 scale at the front of every block (the i-quants and Q8_0 start their block with it)
void fill_blocks(uint8_t* p, size_t bytes, int type, std::mt19937_64& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((0.5f + (float) (rng() >> 40) * (1.f / 16777216.f)) * 1e-3f);
        std::memcpy(p + o, &h, 2);
    }
}

// how many floats differ at all (bit patterns) between a group's row and the same token alone
size_t differing(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++d;
    return d;
}

int check_pair(int gu_type, int d_type) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) {
        std::printf("  %-14s FAIL native_fmt: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    std::mt19937_64 rng(0x1a7e5000ull + (uint64_t) gu_type * 64 + (uint64_t) d_type);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::mt19937 frng(5u + (unsigned) gu_type);
    std::vector<float> x((size_t) kH), h((size_t) kFF);
    size_t gu_bad = 0, gu_all = 0, dn_bad = 0, dn_all = 0;
    for (int e = 0; e < kExperts; ++e) {
        std::vector<uint8_t> blob(f.bytes);
        fill_blocks(blob.data(), f.down_off, gu_type, rng);
        fill_blocks(blob.data() + f.down_off, f.bytes - f.down_off, d_type, rng);
        std::vector<std::vector<uint8_t>> act(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            hq(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void *ap[kMaxT], *hp[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            for (auto& v : x) v = nd(frng);
            for (auto& v : h) v = 0.05f * nd(frng);
            cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
            cpu::native_quant_h(f, h.data(), hq[(size_t) t].data());
            ap[t] = act[(size_t) t].data();
            hp[t] = hq[(size_t) t].data();
        }
        for (int nt = 2; nt <= kMaxT; ++nt) {
            std::vector<float> grp((size_t) nt * kFF), one(kFF);
            float* gp[kMaxT];
            for (int t = 0; t < nt; ++t) gp[t] = &grp[(size_t) t * kFF];
            cpu::native_gu_rows(f, blob.data(), ap, nt, gp, 0, (int) kFF);
            for (int t = 0; t < nt; ++t) {
                float* op = one.data();
                cpu::native_gu_rows(f, blob.data(), &ap[t], 1, &op, 0, (int) kFF);
                gu_bad += differing(gp[t], one.data(), (size_t) kFF);
                gu_all += (size_t) kFF;
            }
            std::vector<float> dgrp((size_t) nt * kH), done((size_t) kH);
            float* dp[kMaxT];
            for (int t = 0; t < nt; ++t) dp[t] = &dgrp[(size_t) t * kH];
            cpu::native_down_rows(f, blob.data(), hp, nt, dp, 0, (int) kH);
            for (int t = 0; t < nt; ++t) {
                float* op = done.data();
                cpu::native_down_rows(f, blob.data(), &hp[t], 1, &op, 0, (int) kH);
                dn_bad += differing(dp[t], done.data(), (size_t) kH);
                dn_all += (size_t) kH;
            }
        }
    }
    std::printf("  %-14s gate/up: %zu of %zu values differ between a group and the token alone (mt_min %d); down: %zu of %zu\n",
                name.c_str(), gu_bad, gu_all, cpu::native_gu_mt_min(gu_type), dn_bad, dn_all);
    return gu_bad == 0 && dn_bad == 0 ? 0 : 1;
}
}  // namespace

int main() {
    ggml_cpu_init();
    std::printf("native_expert_row_invariance_test: ggml-cpu avx512 %d, avx2 %d\n", ggml_cpu_has_avx512(), ggml_cpu_has_avx2());
    int failures = 0;
    // the pairs Ornith's quantization uses: IQ3_XXS gate/up with IQ4_XS down, and with IQ4_NL down
    failures += check_pair(18, 23);
    failures += check_pair(18, 20);
    std::printf("native_expert_row_invariance_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
