// src/kernels/cpu/native_expert_row_invariance_test.cpp - a token's expert rows must not depend on how many tokens share
// its expert.  The hybrid engine decodes one token at a time (every CPU expert is a one-token job) but verifies a window
// of several rows, where an expert chosen by two or more rows is one multi-token job; if the two take different kernels,
// the same row rounds differently in the two paths and a near-tie greedy pick flips between MTP 0 and MTP 2.
// The format is set the way the q36 engine sets it (NativeFmt::one_token_512), at Ornith's layer pairs and geometry
// (n_embd 3072, width 1024): each row of an nt = 2 .. MAXT call against the same token alone, bit for bit, for gate/up and
// for down.  On an AVX-512 CPU that proves the q36 rows are the same alone and in a group (IQ3_XXS gate/up, IQ4_XS down
// and IQ4_NL down all take the one-token kernels).  On a CPU without AVX-512 the IQ3_XXS gate/up is ggml-cpu's dot for one
// token and the AVX-2 kernel from two tokens on, so the test is expected to fail there: it only exercises the q36 rows
// where the one-token kernels exist.  With neither AVX-512 nor AVX-2 every call is ggml's dot and it passes without
// exercising a kernel choice.
// A second pass runs the engine's default (one_token_512 off, as without STRATA_Q36_ONE_TOKEN_CPU): a lone token's
// IQ3_XXS gate/up is ggml-cpu's dot and a group's is the AVX-512 kernel in ggml's float order (#74), so the rows must
// match bit for bit there too.  The down rows are checked in that pass for IQ4_XS only (ggml's dot at every token
// count); an IQ4_NL down keeps the #152 rule by default (ggml's dot alone, the AVX-2 kernel in a group), outside #74.
// The pass is skipped where native_gu_rows would not take the ggml-order kernel (STRATA_NO_IQ512,
// STRATA_NO_IQ512_GGML_ORDER, STRATA_IQ_MT_MIN=1, or a ggml-cpu without AVX2 and FMA).
// It prints which kernels the CPU offers.
// No GPU, no model: random blocks with sane scales and random activations quantized the way the engine quantizes them.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "native_expert_test_util.hpp"

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
constexpr int kMaxT = cpu::MAXT;           // every group size an expert serves (a verify window holds three rows today)
constexpr int kExperts = 6;

// how many floats differ at all (bit patterns) between a group's row and the same token alone
size_t differing(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++d;
    return d;
}

int check_pair(int gu_type, int d_type, bool one_token_512, bool check_down) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type) +
                             (one_token_512 ? " one_token_512" : "");
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) {
        std::printf("  %-14s FAIL native_fmt: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    f.one_token_512 = one_token_512;   // true: as q36_engine.cu sets it under STRATA_Q36_ONE_TOKEN_CPU
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
            if (!check_down) continue;
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
    std::printf("  %-28s gate/up: %zu of %zu values differ between a group and the token alone (mt_min %d); down: %zu of %zu%s\n",
                name.c_str(), gu_bad, gu_all, cpu::native_gu_mt_min(gu_type, f.one_token_512), dn_bad, dn_all,
                check_down ? "" : " (not checked)");
    return gu_bad == 0 && dn_bad == 0 ? 0 : 1;
}
}  // namespace

int main() {
    ggml_cpu_init();
    std::printf("native_expert_row_invariance_test: CPU avx512 %d; ggml-cpu avx512 %d, avx2 %d, fma %d\n", cpu::cpu_avx512_ok(),
                ggml_cpu_has_avx512(), ggml_cpu_has_avx2(), ggml_cpu_has_fma());
    // the kernels under test are AVX-512 (iq512_on: the CPU, not ggml-cpu's build): without them the test fails by
    // design (see above)
    if (!cpu::cpu_avx512_ok()) {
        std::printf("native_expert_row_invariance_test: no AVX-512: SKIPPED\n");
        return 77;
    }
    int failures = 0;
    // the pairs Ornith's quantization uses: IQ3_XXS gate/up with IQ4_XS down, and with IQ4_NL down
    failures += check_pair(18, 23, true, true);
    failures += check_pair(18, 20, true, true);
    // the engine's default (#74): native_gu_rows' own ggml-order condition
    if (cpu::native_gu_ggml_order(18, false)) {
        failures += check_pair(18, 23, false, true);
        failures += check_pair(18, 20, false, false);
    } else {
        std::printf("  default pass (one_token_512 off): no ggml-order kernel in this build or environment: SKIPPED\n");
    }
    std::printf("native_expert_row_invariance_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
