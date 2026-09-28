// src/kernels/cpu/iq512_test.cpp - O8d (#30): the AVX-512 i-quant kernels on synthetic rows (CPU only, no model).
//
//   iq512_test               bitwise: the gathered decode (with and without the aligned activation copy) against
//                            the previous kernel, every format, 1..8 tokens, gate/up rows and plain rows; and the
//                            previous kernel against ggml's own vec_dot (float order only, rel <= 1e-5)
//   iq512_test --bench [E]   gate/up rows of E synthetic experts (default 16) per format and token count, one
//                            thread, the three builds interleaved, best of 7; the weights stay in the L2/L3, so this
//                            is the decode's own rate (the kernels measured ~5 GB/s per core, compute bound)
//
// Random bytes are valid encodings for all five formats (every grid, sign and scale index is in range); only the
// fp16 block scale is drawn from a sane range.  The activations are ggml's Q8_K, 4096 bytes apart and not 64-byte
// aligned, as the pool lays them out.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int N = 2560;      // n_embd: 10 Q8_K blocks
constexpr int ROWS = 640;    // gate rows (and as many up rows) per expert
constexpr int MAXT = 8;
const int kTypes[] = {16, 17, 18, 21, 22};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// `rows` random rows of `type`, each block's fp16 scale in [2^-6, 2^-5)
std::vector<uint8_t> make_rows(int type, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, N), bb = ggml_row_size((ggml_type) type, 256);
    std::vector<uint8_t> w((size_t) rows * rb);
    for (auto& v : w) v = (uint8_t) rng();
    for (size_t off = 0; off < w.size(); off += bb) {
        const uint16_t h = (uint16_t) (0x2400 + rng() % 0x0400);
        std::memcpy(&w[off], &h, 2);
    }
    return w;
}

double rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq512_test: this CPU has no AVX-512 VNNI/VBMI - SKIPPED\n");
        return 0;
    }
    ggml_cpu_init();
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::mt19937 rng(30);
    std::normal_distribution<float> nd(0.f, 1.f);

    // the activations: MAXT tokens of Q8_K, 4096 bytes apart, starting 16 bytes past a line (as a vector's might)
    std::vector<uint8_t> abuf((size_t) MAXT * cpu::kNativeActBytes + 128);
    uint8_t* abase = (uint8_t*) (((uintptr_t) abuf.data() + 63) & ~(uintptr_t) 63) + 16;
    const void* act[MAXT];
    {
        const auto* q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        std::vector<float> x(N);
        for (int t = 0; t < MAXT; ++t) {
            for (auto& v : x) v = nd(rng);
            q8k->from_float(x.data(), abase + (size_t) t * cpu::kNativeActBytes, N);
            act[t] = abase + (size_t) t * cpu::kNativeActBytes;
        }
    }

    if (!bench) {
        int fail = 0;
        const int R = 48;   // rows per matrix: enough to cover every code path, quick
        for (int type : kTypes) {
            const size_t rb = ggml_row_size((ggml_type) type, N);
            const std::vector<uint8_t> w = make_rows(type, 2 * R, rng);   // R gate rows, then R up rows
            // the previous kernel against ggml's vec_dot, one token (the contract the builds below inherit)
            {
                const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
                std::vector<float> ref(R), got(R);
                for (int r = 0; r < R; ++r) tc->vec_dot(N, &ref[(size_t) r], 0, w.data() + (size_t) r * rb, 0, act[0], 0, 1);
                float* gp[1] = {got.data()};
                cpu::iq512_set_path(2);
                cpu::iq512_rows(type, w.data(), rb, N, act, 1, gp, 0, R);
                const double e = rel(got.data(), ref.data(), R);
                if (e > 1e-5) {
                    std::printf("FAIL %s: the previous kernel differs from ggml's vec_dot (rel %.2e)\n",
                                ggml_type_name((ggml_type) type), e);
                    ++fail;
                }
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                std::vector<float> out[3], ffo[3];
                for (int path = 0; path < 3; ++path) {
                    out[path].assign((size_t) nt * R, -1.f);
                    ffo[path].assign((size_t) nt * R, -1.f);
                    float* op[MAXT];
                    float* fp[MAXT];
                    for (int t = 0; t < nt; ++t) { op[t] = out[path].data() + (size_t) t * R; fp[t] = ffo[path].data() + (size_t) t * R; }
                    cpu::iq512_set_path(path);
                    // rows 3..R-1 of the plain rows (an odd start), all R of the gate/up pairs
                    cpu::iq512_rows(type, w.data(), rb, N, act, nt, op, 3, R);
                    cpu::iq512_gu_rows(type, w.data(), rb, (size_t) R * rb, N, act, nt, fp, 0, R);
                }
                for (int path = 0; path < 2; ++path) {
                    if (std::memcmp(out[path].data(), out[2].data(), out[2].size() * sizeof(float)) != 0 ||
                        std::memcmp(ffo[path].data(), ffo[2].data(), ffo[2].size() * sizeof(float)) != 0) {
                        std::printf("FAIL %s nt=%d: build %d is not bitwise the previous kernel\n",
                                    ggml_type_name((ggml_type) type), nt, path);
                        ++fail;
                    }
                }
            }
        }
        cpu::iq512_set_path(0);
        std::printf("iq512_test: %s\n", fail ? "FAILED" : "OK (every build bitwise the previous kernel, which matches ggml)");
        return fail ? 1 : 0;
    }

    const int E = argc > 2 ? std::atoi(argv[2]) : 16;
    std::printf("gate/up rows of %d synthetic experts (2 x %d rows of %d), one thread, best of 7; GB/s of weights\n", E,
                ROWS, N);
    std::printf("%-8s %3s %12s %12s %12s %8s\n", "format", "nt", "previous", "gather", "gather+copy", "gain");
    for (int type : kTypes) {
        const size_t rb = ggml_row_size((ggml_type) type, N);
        const size_t eb = 2 * (size_t) ROWS * rb;
        std::vector<uint8_t> w((size_t) E * eb);
        for (int e = 0; e < E; ++e) {
            const std::vector<uint8_t> x = make_rows(type, 2 * ROWS, rng);
            std::memcpy(w.data() + (size_t) e * eb, x.data(), eb);
        }
        std::vector<float> ff((size_t) MAXT * ROWS);
        float* fp[MAXT];
        for (int t = 0; t < MAXT; ++t) fp[t] = ff.data() + (size_t) t * ROWS;
        for (int nt : {1, 2, 3, 4, 8}) {
            double best[3] = {1e30, 1e30, 1e30};
            for (int rep = 0; rep < 7; ++rep)
                for (int k = 0; k < 3; ++k) {
                    const int path = (rep + k) % 3;
                    cpu::iq512_set_path(path);
                    const double t0 = now_ms();
                    for (int e = 0; e < E; ++e)
                        cpu::iq512_gu_rows(type, w.data() + (size_t) e * eb, rb, (size_t) ROWS * rb, N, act, nt, fp, 0, ROWS);
                    best[path] = std::fmin(best[path], now_ms() - t0);
                }
            const double gb = (double) E * eb / 1e9;
            std::printf("%-8s %3d %8.2f GB/s %7.2f GB/s %7.2f GB/s %+7.1f%%", ggml_type_name((ggml_type) type), nt,
                        gb / (best[2] * 1e-3), gb / (best[1] * 1e-3), gb / (best[0] * 1e-3),
                        100.0 * (best[2] / best[0] - 1.0));
            if (nt == 1) {
                // what the engine runs for a single token (native_gu_rows): ggml-cpu's own vec_dot
                const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
                double bg = 1e30;
                for (int rep = 0; rep < 7; ++rep) {
                    const double t0 = now_ms();
                    for (int e = 0; e < E; ++e) {
                        const uint8_t* wb = w.data() + (size_t) e * eb;
                        for (int r = 0; r < ROWS; ++r) {
                            float g = 0.f, u = 0.f;
                            tc->vec_dot(N, &g, 0, wb + (size_t) r * rb, 0, act[0], 0, 1);
                            tc->vec_dot(N, &u, 0, wb + (size_t) (ROWS + r) * rb, 0, act[0], 0, 1);
                            ff[(size_t) r] = (g / (1.f + std::exp(-g))) * u;
                        }
                    }
                    bg = std::fmin(bg, now_ms() - t0);
                }
                std::printf("   (ggml vec_dot, one token: %.2f GB/s)", gb / (bg * 1e-3));
            }
            std::printf("\n");
        }
    }
    cpu::iq512_set_path(0);
    return 0;
}
