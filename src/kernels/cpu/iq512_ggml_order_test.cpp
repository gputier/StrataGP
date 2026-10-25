// src/kernels/cpu/iq512_ggml_order_test.cpp - #74: the AVX-512 IQ3_XXS gate/up rows in ggml's float order
// (iq512_iq3xxs_gu_rows_ggml) against ggml-cpu's own vec_dot for type 18, bit for bit, at every group size 1 .. MAXT:
// every gate row g and up row u of every token, then ff = SwiGLU(g) * u through native_gu_rows with the engine's default
// (one_token_512 off), against the same expression on ggml's g and u.  Where this holds, a token's IQ3_XXS gate/up rows in a
// verify window are the ones its lone-token decode step computes, so greedy MTP gives the tokens of MTP 0 on these rows.
// The kernel copies the order of ggml's AVX2+FMA dot: skipped (77) on a CPU without AVX-512 and where ggml-cpu was built
// without AVX2 or FMA (native_gu_rows does not take the kernel there).  The ff check is skipped where native_gu_rows would
// not take it either (STRATA_NO_IQ512, STRATA_NO_IQ512_GGML_ORDER, STRATA_IQ_MT_MIN=1).
// No GPU, no model: random blocks with sane scales, random activations quantized the way the engine quantizes them.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "native_expert_test_util.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int kMaxT = cpu::MAXT;
constexpr int kExperts = 3;

bool same(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// one geometry: n_embd, expert width.  Returns the number of differing values.
size_t check(int64_t n_embd, int64_t n_ff, bool check_ff) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(18, 23, n_embd, n_ff, f, err)) {
        std::printf("  %lld/%lld FAIL native_fmt: %s\n", (long long) n_embd, (long long) n_ff, err.c_str());
        return 1;
    }
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot;
    const int n = (int) n_embd, ff_rows = (int) n_ff;
    std::mt19937_64 rng(0x74a5e000ull + (uint64_t) n_embd);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::mt19937 frng(74u + (unsigned) n_ff);
    std::vector<float> x((size_t) n_embd);
    size_t g_bad = 0, u_bad = 0, ff_bad = 0, all = 0;
    for (int e = 0; e < kExperts; ++e) {
        std::vector<uint8_t> blob(f.bytes);
        fill_blocks(blob.data(), f.down_off, 18, rng);
        std::vector<std::vector<uint8_t>> act(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes));
        const void* ap[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            for (auto& v : x) v = nd(frng);
            cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
            ap[t] = act[(size_t) t].data();
        }
        // ggml's rows and ff for every token, once: the reference every group size is held to
        std::vector<float> rg((size_t) kMaxT * ff_rows), ru(rg.size()), rff(rg.size());
        for (int r = 0; r < ff_rows; ++r)
            for (int t = 0; t < kMaxT; ++t) {
                float g = 0.f, u = 0.f;
                dot(n, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                dot(n, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                const size_t i = (size_t) t * ff_rows + r;
                rg[i] = g;
                ru[i] = u;
                rff[i] = (g / (1.f + std::exp(-g))) * u;
            }
        for (int nt = 1; nt <= kMaxT; ++nt) {
            std::vector<float> kg((size_t) nt * ff_rows), ku(kg.size()), kff(kg.size());
            float *gp[kMaxT], *up[kMaxT], *fp[kMaxT];
            for (int t = 0; t < nt; ++t) {
                gp[t] = &kg[(size_t) t * ff_rows];
                up[t] = &ku[(size_t) t * ff_rows];
                fp[t] = &kff[(size_t) t * ff_rows];
            }
            // two row ranges, the way the pool splits an expert's rows across workers
            const int mid = ff_rows / 3;
            cpu::iq512_iq3xxs_gu_rows_ggml(blob.data(), f.gu_row, f.up_off, n, ap, nt, gp, up, 0, mid);
            cpu::iq512_iq3xxs_gu_rows_ggml(blob.data(), f.gu_row, f.up_off, n, ap, nt, gp, up, mid, ff_rows);
            if (check_ff) {
                cpu::native_gu_rows(f, blob.data(), ap, nt, fp, 0, mid);
                cpu::native_gu_rows(f, blob.data(), ap, nt, fp, mid, ff_rows);
            }
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < ff_rows; ++r) {
                    const size_t i = (size_t) t * ff_rows + r;
                    g_bad += !same(kg[i], rg[i]);
                    u_bad += !same(ku[i], ru[i]);
                    if (check_ff) ff_bad += !same(kff[i], rff[i]);
                    ++all;
                }
        }
    }
    std::printf("  n_embd %lld, width %lld, nt 1..%d: of %zu values, g %zu, u %zu, ff %zu%s differ from ggml-cpu's vec_dot\n",
                (long long) n_embd, (long long) n_ff, kMaxT, all, g_bad, u_bad, ff_bad, check_ff ? "" : " (not checked)");
    return g_bad + u_bad + ff_bad;
}
}  // namespace

int main() {
    ggml_cpu_init();
    std::printf("iq512_ggml_order_test: CPU avx512 %d; ggml-cpu avx512 %d, avx2 %d, fma %d\n", cpu::cpu_avx512_ok(),
                ggml_cpu_has_avx512(), ggml_cpu_has_avx2(), ggml_cpu_has_fma());
    if (!cpu::cpu_avx512_ok() || !ggml_cpu_has_avx2() || !ggml_cpu_has_fma()) {
        std::printf("iq512_ggml_order_test: no AVX-512 CPU, or ggml-cpu without AVX2+FMA: SKIPPED\n");
        return 77;
    }
    const bool check_ff = cpu::native_gu_ggml_order(18, false);   // native_gu_rows' own condition for the kernel
    size_t bad = 0;
    bad += check(3072, 1024, check_ff);   // Ornith-1.5-122B's experts
    bad += check(2048, 512, check_ff);    // fewer blocks per row, a narrower expert
    std::printf("iq512_ggml_order_test: %zu differing values\n", bad);
    return bad == 0 ? 0 : 1;
}
