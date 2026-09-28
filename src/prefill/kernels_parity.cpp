// src/prefill/kernels_parity.cpp - the rewritten prompt kernels (docs/perf/prefill-kernels.md) against the kernels
// they replace, on synthetic data (GPU, no model).  Every comparison is BITWISE: the rewrites keep the old
// arithmetic in the old order, so any difference at all is a bug.  The outputs are pre-filled with different
// patterns on the two sides, so an element one side forgot to write shows up too.
//   1. gdn_conv (P2): the tiled conv + q/k norm and its history against the per-channel walk, over chunk lengths
//      around the tile (1..3 tokens: the history's own entries move) and two chunks in a row; plus a loose host
//      reference, so the two sides cannot agree on something wrong.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace p = strata::prefill;

namespace {
constexpr int S = 128, HK = 16, C = 10240;
int g_fail = 0;
cudaStream_t g_s = nullptr;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n, int fill = 0) {
    T* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(d, fill, n * sizeof(T) + 64), "memset");
    return d;
}
template <typename T> void h2d(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
}
template <typename T> std::vector<T> d2h(const T* d, size_t n) {
    ck(cudaStreamSynchronize(g_s), "sync");
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}
// bitwise; reports the count and the first mismatch
template <typename T> void same(const char* what, long long case_id, const std::vector<T>& a, const std::vector<T>& b) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0 && bad++ == 0) first = i;
    if (bad == 0) return;
    ++g_fail;
    double x = 0, y = 0;
    if constexpr (sizeof(T) == 4) { float f; std::memcpy(&f, &a[first], 4); x = f; std::memcpy(&f, &b[first], 4); y = f; }
    else { x = (double) a[first]; y = (double) b[first]; }
    std::fprintf(stderr, "FAIL %s (case %lld): %zu of %zu differ, first at %zu: old %.9g new %.9g\n", what, case_id, bad,
                 a.size(), first, x, y);
}
std::vector<float> normal(std::mt19937& rng, size_t n, float sd, float mean = 0.0f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// ---------------------------------------------------------------- 1. gdn_conv
void test_conv(std::mt19937& rng) {
    const float eps = 1e-6f;
    const std::vector<float> w = normal(rng, (size_t) C * 4, 0.5f);
    const int64_t Tmax = 300;
    float *dw = dalloc<float>((size_t) C * 4), *qkv = dalloc<float>((size_t) Tmax * C);
    float *hist_o = dalloc<float>((size_t) C * 3), *hist_n = dalloc<float>((size_t) C * 3);
    float *h_o = dalloc<float>((size_t) Tmax * C), *h_n = dalloc<float>((size_t) Tmax * C);
    h2d(dw, w);
    double worst = 0.0;
    const int64_t Ts[] = {1, 2, 3, 4, 15, 16, 17, 31, 33, 257};
    for (int64_t T1 : Ts) {
        std::vector<float> hist = normal(rng, (size_t) C * 3, 1.0f);
        h2d(hist_o, hist);
        h2d(hist_n, hist);
        const int64_t T2 = (T1 * 7) % 37 + 1;   // a second chunk after the first, as the prompt path runs them
        for (int64_t T : {T1, T2}) {
            std::vector<float> x = normal(rng, (size_t) T * C, 1.0f);
            for (size_t i = 0; i < x.size(); i += 97) x[i] *= 30.0f;   // the SiLU's saturated ends
            h2d(qkv, x);
            const std::vector<float> hist_before = d2h(hist_o, (size_t) C * 3);
            ck(cudaMemset(h_o, 0xff, (size_t) T * C * 4), "memset");
            ck(cudaMemset(h_n, 0x7f, (size_t) T * C * 4), "memset");
            p::gdn_conv(hist_o, qkv, dw, h_o, T, eps, g_s, p::KernelPath::Old);
            p::gdn_conv(hist_n, qkv, dw, h_n, T, eps, g_s, p::KernelPath::New);
            const std::vector<float> a = d2h(h_o, (size_t) T * C), b = d2h(h_n, (size_t) T * C);
            same("gdn_conv h", T, a, b);
            same("gdn_conv history", T, d2h(hist_o, (size_t) C * 3), d2h(hist_n, (size_t) C * 3));
            // host reference (double): conv + SiLU, then the L2 norm of the 32 q/k heads
            std::vector<double> ref((size_t) T * C);
            for (int c = 0; c < C; ++c)
                for (int64_t t = 0; t < T; ++t) {
                    double s = 0.0;
                    for (int k = 0; k < 4; ++k) {
                        const int64_t u = t - 3 + k;
                        const double in = u < 0 ? hist_before[(size_t) c * 3 + 3 + u] : x[(size_t) (u * C + c)];
                        s += in * w[(size_t) c * 4 + k];
                    }
                    ref[(size_t) (t * C + c)] = s / (1.0 + std::exp(-s));
                }
            for (int64_t t = 0; t < T; ++t)
                for (int hd = 0; hd < 2 * HK; ++hd) {
                    double ss = 0.0;
                    for (int j = 0; j < S; ++j) ss += ref[(size_t) (t * C + hd * S + j)] * ref[(size_t) (t * C + hd * S + j)];
                    for (int j = 0; j < S; ++j) ref[(size_t) (t * C + hd * S + j)] /= std::sqrt(ss + eps);
                }
            for (size_t i = 0; i < ref.size(); ++i)
                worst = std::fmax(worst, std::fabs((double) b[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
        }
    }
    if (!(worst < 1e-4)) { std::fprintf(stderr, "FAIL gdn_conv vs host reference: %.3g\n", worst); ++g_fail; }
    std::printf("gdn_conv: tiled vs per-channel bitwise over %zu chunk pairs; host reference %.2g\n",
                sizeof Ts / sizeof Ts[0], worst);
    cudaFree(dw); cudaFree(qkv); cudaFree(hist_o); cudaFree(hist_n); cudaFree(h_o); cudaFree(h_n);
}
}  // namespace

int main() {
    ck(cudaStreamCreate(&g_s), "stream");   // a blocking stream: ordered with the plain cudaMemcpy/cudaMemset
    std::mt19937 rng(20260928);
    test_conv(rng);
    std::printf("prefill_kernels_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
