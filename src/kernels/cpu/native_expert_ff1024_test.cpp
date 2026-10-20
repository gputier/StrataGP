// src/kernels/cpu/native_expert_ff1024_test.cpp - the native (CPU) expert path at Ornith-1.5-122B's geometry: n_embd 3072,
// expert width 1024.  The down activation is then 1168 bytes as Q8_K (IQ4_XS down) or 1088 as Q8_0 (IQ4_NL down), above
// the 1024 the pool's buffers once held (kNativeHBytes), so native_fmt refused the layer.  No GPU, no model: random
// blocks with sane scales, random activations quantized the way the engine quantizes them, the rows against ggml-cpu's
// vec_dot, rel <= 1e-4.
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int64_t kH = 3072, kFF = 1024;   // n_embd, expert width
constexpr int kMaxT = 3;

// random bytes, then a sane fp16 scale at the front of every block (the i-quants and Q8_0 start their block with it)
void fill_blocks(uint8_t* p, size_t bytes, int type, std::mt19937_64& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((0.5f + (float) (rng() >> 40) * (1.f / 16777216.f)) * 1e-3f);
        std::memcpy(p + o, &h, 2);
    }
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}

int check_pair(int gu_type, int d_type) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) {
        std::printf("  %-14s FAIL native_fmt: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    std::printf("  %-14s native_fmt ok: activation %zu bytes (pool buffer %zu), down activation %zu bytes (pool buffer %zu)\n",
                name.c_str(), f.act_bytes, cpu::kNativeActBytes, f.h_bytes, cpu::kNativeHBytes);
    std::mt19937_64 rng(0x0a17e000ull + (uint64_t) gu_type * 64 + (uint64_t) d_type);
    std::vector<uint8_t> blob(f.bytes);
    fill_blocks(blob.data(), f.down_off, gu_type, rng);
    fill_blocks(blob.data() + f.down_off, f.bytes - f.down_off, d_type, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)),
        hq(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
    const void *ap[kMaxT], *hp[kMaxT];
    std::mt19937 frng(7u + (unsigned) gu_type);
    std::vector<float> x((size_t) kH), h((size_t) kFF);
    for (int t = 0; t < kMaxT; ++t) {
        for (auto& v : x) v = nd(frng);
        for (auto& v : h) v = 0.05f * nd(frng);
        cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
        cpu::native_quant_h(f, h.data(), hq[(size_t) t].data());
        ap[t] = act[(size_t) t].data();
        hp[t] = hq[(size_t) t].data();
    }
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) d_type);
    int failures = 0;

    // gate / up: silu(gate . a) * (up . a) for every row, 1..kMaxT tokens
    std::vector<float> ff((size_t) kMaxT * kFF), want_ff((size_t) kMaxT * kFF);
    float* fp[kMaxT];
    for (int t = 0; t < kMaxT; ++t) {
        fp[t] = &ff[(size_t) t * kFF];
        for (int64_t r = 0; r < kFF; ++r) {
            float g = 0.f, u = 0.f;
            tg->vec_dot((int) kH, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            tg->vec_dot((int) kH, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            want_ff[(size_t) t * kFF + (size_t) r] = g / (1.f + std::exp(-g)) * u;
        }
    }
    for (int nt = 1; nt <= kMaxT; ++nt) {
        cpu::native_gu_rows(f, blob.data(), ap, nt, fp, 0, (int) kFF);
        const std::vector<float> a(ff.begin(), ff.begin() + (long) nt * kFF), b(want_ff.begin(), want_ff.begin() + (long) nt * kFF);
        const double r = rel(a, b);
        std::printf("  %-14s gate/up, %d token%s: rel %.2e\n", name.c_str(), nt, nt > 1 ? "s" : "", r);
        if (!(r <= 1e-4)) ++failures;
    }

    // down: every row of the 3072 outputs against the 1024-wide activation
    std::vector<float> out((size_t) kMaxT * kH), want((size_t) kMaxT * kH);
    float* op[kMaxT];
    for (int t = 0; t < kMaxT; ++t) {
        op[t] = &out[(size_t) t * kH];
        for (int64_t r = 0; r < kH; ++r)
            td->vec_dot((int) kFF, &want[(size_t) t * kH + (size_t) r], 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[t], 0, 1);
    }
    cpu::native_down_rows(f, blob.data(), hp, kMaxT, op, 0, (int) kH);
    const double r = rel(out, want);
    std::printf("  %-14s down, %d tokens: rel %.2e\n", name.c_str(), kMaxT, r);
    if (!(r <= 1e-4)) ++failures;
    return failures;
}
}  // namespace

int main() {
    ggml_cpu_init();
    int failures = 0;
    // the pairs Ornith's quantization uses: IQ3_XXS gate/up with IQ4_XS down (Q8_K down activation, 1168 bytes), and
    // IQ3_XXS with IQ4_NL down (Q8_0, 1088 bytes)
    failures += check_pair(18, 23);
    failures += check_pair(18, 20);
    std::printf("native_expert_ff1024_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
