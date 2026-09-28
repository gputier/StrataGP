// src/kernels/cpu/expert_multi_test.cpp - plan v0.3 P6: one expert, several tokens (CPU only, synthetic data).
//
//   expert_multi_test              bitwise check: s2_expert_vnni_multi vs s2_expert_vnni_q per token, n = 1..8,
//                                  in the default arithmetic and in the integer-correction one (#29), with and
//                                  without software prefetch (which must change nothing)
//   expert_multi_test --bench [E]  time per expert for n = 1..5 over E synthetic experts (default 256 = 354 MB,
//                                  far beyond L3, so the blob comes from DRAM as it does in the engine); 1 thread.
//                                  Then the O8c arms (#29), interleaved, best of 5: prefetch of the scales, of the
//                                  scales and codes, and the integer correction
//
// The bench measures the speculation model's `extra_use_cost`: the CPU time of each extra token routed to an
// expert, as a fraction of reading the expert once (tools/spec_economics.py assumes 0.2).
#include "strata/kernels/cpu/expert.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {
void make_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < c::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();              // codes
    for (size_t i = c::O_GU_SCALES; i < c::BLOB; i += 2) {                            // fp16 scales ~0.004-0.016
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    c::cpu_require_expert_support();
    std::mt19937 rng(9);
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int E = bench ? (argc > 2 ? std::atoi(argv[2]) : 256) : 4;
    std::vector<uint8_t> blobs((size_t) E * c::BLOB);
    for (int e = 0; e < E; ++e) make_blob(&blobs[(size_t) e * c::BLOB], rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    static c::ActQ acts[c::MAXT];
    static float xs[c::MAXT][c::H];
    for (int t = 0; t < c::MAXT; ++t) {
        for (float& v : xs[t]) v = nd(rng);
        c::act_quant_q8_1(xs[t], c::H, acts[t]);
    }
    const c::ActQ* a1[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) a1[t] = &acts[t];
    static float out_multi[c::MAXT][c::H], out_single[c::H];
    float* outs[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) outs[t] = out_multi[t];
    static c::ExpertScratchMulti wsm;
    static c::ExpertScratch ws;

    if (!bench) {
        int fail = 0;
        // default arithmetic, then the integer correction (O8c (b)); each with the prefetch off, then on (O8c (a))
        static float base_out[4][c::MAXT][c::H];
        for (int ic = 0; ic < 2; ++ic) {
            c::expert_set_int_corr(ic == 1);
            for (int t = 0; t < c::MAXT; ++t) c::act_quant_q8_1(xs[t], c::H, acts[t]);   // the seeds follow the mode
            for (int pf = 0; pf < 2; ++pf) {
                c::expert_set_prefetch(pf ? 2048 : 0, pf ? 1024 : 0);
                for (int e = 0; e < E; ++e) {
                    const uint8_t* blob = &blobs[(size_t) e * c::BLOB];
                    for (int n = 1; n <= c::MAXT; ++n) {
                        c::s2_expert_vnni_multi(blob, a1, n, outs, wsm);
                        for (int t = 0; t < n; ++t) {
                            c::s2_expert_vnni_q(blob, acts[t], out_single, ws);
                            if (std::memcmp(out_single, out_multi[t], sizeof out_single) != 0) {
                                std::fprintf(stderr, "FAIL expert %d n=%d token %d differs (int corr %d, prefetch %d)\n",
                                             e, n, t, ic, pf);
                                ++fail;
                            }
                        }
                        if (n == c::MAXT && e == 0) {
                            if (pf == 0) std::memcpy(base_out[ic], out_multi, sizeof base_out[ic]);
                            else if (std::memcmp(base_out[ic], out_multi, sizeof base_out[ic]) != 0) {
                                std::fprintf(stderr, "FAIL the prefetch changed a result (int corr %d)\n", ic);
                                ++fail;
                            }
                        }
                    }
                }
            }
        }
        // the integer correction is the same arithmetic up to rounding: against the default and against the
        // scalar oracle on the same int8 activations, it must be as close as the default is
        {
            double num = 0, den = 0, num_def = 0, num_ic = 0, den_o = 0;
            static float oracle[c::H];
            for (int t = 0; t < c::MAXT; ++t)
                for (int i = 0; i < c::H; ++i) {
                    num += std::fabs((double) base_out[1][t][i] - base_out[0][t][i]);
                    den += std::fabs((double) base_out[0][t][i]);
                }
            for (int t = 0; t < 2; ++t) {
                c::s2_expert_scalar(&blobs[0], xs[t], oracle, true);
                for (int i = 0; i < c::H; ++i) {
                    num_def += std::fabs((double) base_out[0][t][i] - oracle[i]);
                    num_ic += std::fabs((double) base_out[1][t][i] - oracle[i]);
                    den_o += std::fabs((double) oracle[i]);
                }
            }
            const double r = num / (den + 1e-30), rd = num_def / (den_o + 1e-30), ri = num_ic / (den_o + 1e-30);
            std::printf("integer correction vs default: rel %.2e; vs the scalar oracle: default %.2e, integer %.2e\n", r, rd, ri);
            if (!(r < 1e-5) || !(ri < 2.0 * rd + 1e-6)) {
                std::fprintf(stderr, "FAIL the integer correction is not the same arithmetic\n");
                ++fail;
            }
        }
        // the GGUF-layout Q2_0 rows (the native packs' down projection): one token vs MAXT at once, prefetch on and
        // off, in both arithmetics; and the integer correction against the default
        {
            constexpr int NB = c::SC_D, RB = NB * 18, R = 256;
            std::vector<uint8_t> w((size_t) R * RB);
            for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) rng();
            for (size_t i = 0; i < w.size(); i += 18) {
                const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
                std::memcpy(&w[i], &h, 2);
            }
            static c::ActQ h[c::MAXT];
            static float hx[c::MAXT][c::FF], q2g[2][2][c::MAXT][R];
            const c::ActQ* hp[c::MAXT];
            float* op[c::MAXT];
            float one[R];
            float* onep[1] = {one};
            for (int t = 0; t < c::MAXT; ++t) for (float& v : hx[t]) v = nd(rng);
            for (int ic = 0; ic < 2; ++ic) {
                c::expert_set_int_corr(ic == 1);
                for (int t = 0; t < c::MAXT; ++t) { c::act_quant_q8_1(hx[t], c::FF, h[t]); hp[t] = &h[t]; }
                for (int pf = 0; pf < 2; ++pf) {
                    c::expert_set_prefetch(pf ? 2048 : 0, pf ? 2048 : 0);
                    for (int t = 0; t < c::MAXT; ++t) op[t] = q2g[ic][pf][t];
                    c::q2_0_gguf_rows_multi(w.data(), RB, NB, hp, c::MAXT, op, 0, R);
                    for (int t = 0; t < c::MAXT; ++t) {
                        c::q2_0_gguf_rows_multi(w.data(), RB, NB, hp + t, 1, onep, 0, R);
                        if (std::memcmp(one, q2g[ic][pf][t], sizeof one) != 0) {
                            std::fprintf(stderr, "FAIL GGUF Q2_0 rows: token %d alone differs (int corr %d, prefetch %d)\n", t, ic, pf);
                            ++fail;
                        }
                    }
                }
                if (std::memcmp(q2g[ic][0], q2g[ic][1], sizeof q2g[ic][0]) != 0) {
                    std::fprintf(stderr, "FAIL GGUF Q2_0 rows: the prefetch changed a result (int corr %d)\n", ic);
                    ++fail;
                }
            }
            double num = 0, den = 0;
            for (int t = 0; t < c::MAXT; ++t)
                for (int r = 0; r < R; ++r) {
                    num += std::fabs((double) q2g[1][0][t][r] - q2g[0][0][t][r]);
                    den += std::fabs((double) q2g[0][0][t][r]);
                }
            const bool same = std::memcmp(q2g[0][0], q2g[1][0], sizeof q2g[0][0]) == 0;
            std::printf("GGUF Q2_0 rows, integer correction vs default: rel %.2e%s\n", num / (den + 1e-30),
                        same ? " (bitwise equal: the mode did not take effect)" : "");
            if (!(num / (den + 1e-30) < 1e-5) || same) ++fail;
        }
        c::expert_set_int_corr(false);
        c::expert_set_prefetch(0, 0);
        std::printf("expert_multi_test: %s\n", fail ? "FAILED" : "OK (every token bitwise equal to the single-token kernel)");
        return fail ? 1 : 0;
    }

    std::printf("%d experts x %.2f MB, one thread\n", E, c::BLOB / 1e6);
    double base = 0;
    for (int n = 1; n <= 5; ++n) {
        c::s2_expert_vnni_multi(&blobs[0], a1, n, outs, wsm);                     // warm the code path
        const double t0 = now_ms();
        for (int e = 0; e < E; ++e) c::s2_expert_vnni_multi(&blobs[(size_t) e * c::BLOB], a1, n, outs, wsm);
        const double us = 1000.0 * (now_ms() - t0) / E;
        if (n == 1) base = us;
        std::printf("tokens per expert %d: %7.1f us per expert (%.2fx one token; extra token = %.2f of one read), %.1f GB/s\n",
                    n, us, us / base, n > 1 ? (us / base - 1.0) / (n - 1) : 0.0, c::BLOB / (us * 1e3));
    }

    // O8c (#29): the arms interleaved, best of 5 passes over the E experts, per token count
    struct Arm { const char* name; int pf_sc, pf_co; bool ic; };
    const Arm arms[] = {{"default", 0, 0, false},
                        {"prefetch scales 2 KB", 2048, 0, false},
                        {"prefetch codes 512 B", 0, 512, false},
                        {"prefetch scales 2 KB + codes 1 KB", 2048, 1024, false},
                        {"prefetch scales 2 KB + codes 2 KB", 2048, 2048, false},
                        {"prefetch scales 4 KB + codes 4 KB", 4096, 4096, false},
                        {"integer correction", 0, 0, true},
                        {"integer correction + prefetch 2K/1K", 2048, 1024, true}};
    const int NA = (int) (sizeof arms / sizeof arms[0]);
    std::printf("\nO8c arms, one thread, best of 5 passes over %d experts (us per expert, gain vs default)\n", E);
    for (int n : {1, 2, 4}) {
        double best[NA];
        for (int a = 0; a < NA; ++a) best[a] = 1e30;
        for (int rep = 0; rep < 5; ++rep)
            for (int k = 0; k < NA; ++k) {
                const int a = (rep + k) % NA;
                c::expert_set_prefetch(arms[a].pf_sc, arms[a].pf_co);
                c::expert_set_int_corr(arms[a].ic);
                for (int t = 0; t < c::MAXT; ++t) c::act_quant_q8_1(xs[t], c::H, acts[t]);
                const double t0 = now_ms();
                for (int e = 0; e < E; ++e) c::s2_expert_vnni_multi(&blobs[(size_t) e * c::BLOB], a1, n, outs, wsm);
                best[a] = std::fmin(best[a], 1000.0 * (now_ms() - t0) / E);
            }
        for (int a = 0; a < NA; ++a)
            std::printf("  n=%d %-36s %7.1f us  %+6.1f%%\n", n, arms[a].name, best[a], 100.0 * (best[0] / best[a] - 1.0));
    }
    c::expert_set_prefetch(0, 0);
    c::expert_set_int_corr(false);
    return 0;
}
