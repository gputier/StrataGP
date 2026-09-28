// src/prefill/kernels_parity.cpp - the rewritten prompt kernels (docs/perf/prefill-kernels.md) against the kernels
// they replace, on synthetic data (GPU, no model).  Every comparison is BITWISE: the rewrites keep the old
// arithmetic in the old order, so any difference at all is a bug.  The outputs are pre-filled with different
// patterns on the two sides, so an element one side forgot to write shows up too.
//   1. gdn_conv (P2): the tiled conv + q/k norm and its history against the per-channel walk, over chunk lengths
//      around the tile (1..3 tokens: the history's own entries move) and two chunks in a row; plus a loose host
//      reference, so the two sides cannot agree on something wrong.
//   2. gdn_recurrence (P3): 4 blocks per head + the norm pass against a block per head: y, its FP16 bits and the
//      state after the chunk, over chunk lengths from 1 and two chunks in a row; plus a loose host reference.
//   3. the GR chain (P8): gr_norm_scale / gr_mix_scale / gr_write_norm against gr_norm / gr_mix / gr_write /
//      gr_norm: the BF16 xn, mixed in FP32, BF16 and FP16, the written R, then the next half's xn and mixed.
//   4. FP16 saturation (B7): off, every FP16 image is __float2half_rn (+-inf past 65520); on, finite values past the
//      range give +-65504, NaN and +-inf pass through, and every in-range value keeps its bits.
//   5. the router (B8): route_native over a chunk against native_router_top10 token by token (ids and weights
//      bitwise, with ties, a flat row and underflowing probabilities), route_decode's dispatch, and - reported, not
//      judged - how often the default prompt router (precise expf) picks a different set.
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"
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
constexpr int S = 128, HK = 16, HV = 48, C = 10240, N = 2560, D = 4 * N;
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

// ---------------------------------------------------------------- 2. gdn_recurrence
// host reference of one chunk (double), state [row][head][col] updated in place
void rec_reference(std::vector<double>& st, const std::vector<float>& h, const std::vector<float>& gate,
                   const std::vector<float>& beta, const std::vector<float>& z, const std::vector<float>& gamma,
                   float eps, int64_t T, std::vector<double>& y) {
    y.assign((size_t) T * HV * S, 0.0);
    std::vector<double> q(S), k(S), kv(S), o(S);
    for (int64_t t = 0; t < T; ++t)
        for (int hd = 0; hd < HV; ++hd) {
            const int qh = hd % HK;
            for (int i = 0; i < S; ++i) {
                q[i] = h[(size_t) (t * C + qh * S + i)];
                k[i] = h[(size_t) (t * C + HK * S + qh * S + i)];
            }
            const double g = std::exp((double) gate[(size_t) (t * HV + hd)]), b = beta[(size_t) (t * HV + hd)];
            auto at = [&](int row, int col) -> double& { return st[((size_t) row * HV + hd) * S + col]; };
            for (int col = 0; col < S; ++col) {
                double a = 0.0;
                for (int row = 0; row < S; ++row) a += at(row, col) * k[row];
                kv[col] = a;
            }
            double ss = 0.0;
            for (int col = 0; col < S; ++col) {
                const double delta = (h[(size_t) (t * C + 2 * HK * S + hd * S + col)] - g * kv[col]) * b;
                double a = 0.0;
                for (int row = 0; row < S; ++row) {
                    at(row, col) = g * at(row, col) + k[row] * delta;
                    a += at(row, col) * q[row];
                }
                o[col] = a / std::sqrt((double) S);
                ss += o[col] * o[col];
            }
            for (int col = 0; col < S; ++col) {
                const double zz = z[(size_t) (t * HV * S + hd * S + col)];
                y[(size_t) (t * HV * S + hd * S + col)] =
                    o[col] / std::sqrt(ss / S + eps) * gamma[(size_t) col] / (1.0 + std::exp(-zz));
            }
        }
}
void test_rec(std::mt19937& rng) {
    const float eps = 1e-6f;
    const int64_t Tmax = 200;
    const size_t n_state = (size_t) S * HV * S;
    float *st_o = dalloc<float>(n_state), *st_n = dalloc<float>(n_state), *h = dalloc<float>((size_t) Tmax * C);
    float *gate = dalloc<float>((size_t) Tmax * HV), *beta = dalloc<float>((size_t) Tmax * HV);
    float *z = dalloc<float>((size_t) Tmax * HV * S), *gamma = dalloc<float>(S);
    float *y_o = dalloc<float>((size_t) Tmax * HV * S), *y_n = dalloc<float>((size_t) Tmax * HV * S);
    uint16_t *y16_o = dalloc<uint16_t>((size_t) Tmax * HV * S), *y16_n = dalloc<uint16_t>((size_t) Tmax * HV * S);
    const std::vector<float> gm = normal(rng, S, 0.2f, 1.0f);
    h2d(gamma, gm);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    double worst = 0.0;
    const int64_t Ts[] = {1, 2, 5, 33, 130};
    for (int64_t T1 : Ts) {
        std::vector<float> st0 = normal(rng, n_state, 0.05f);
        h2d(st_o, st0);
        h2d(st_n, st0);
        std::vector<double> st_ref(st0.begin(), st0.end());
        const int64_t T2 = T1 % 7 + 3;
        for (int64_t T : {T1, T2}) {
            // q and k unit-norm per head (what the conv leaves), v and z free, decay gates in (0, 1]
            std::vector<float> hh = normal(rng, (size_t) T * C, 1.0f);
            for (int64_t t = 0; t < T; ++t)
                for (int hd = 0; hd < 2 * HK; ++hd) {
                    double ss = 0.0;
                    for (int j = 0; j < S; ++j) ss += (double) hh[(size_t) (t * C + hd * S + j)] * hh[(size_t) (t * C + hd * S + j)];
                    for (int j = 0; j < S; ++j) hh[(size_t) (t * C + hd * S + j)] = (float) (hh[(size_t) (t * C + hd * S + j)] / std::sqrt(ss));
                }
            std::vector<float> gt((size_t) T * HV), bt((size_t) T * HV);
            for (auto& v : gt) v = -2.0f * ud(rng);
            for (auto& v : bt) v = ud(rng);
            const std::vector<float> zz = normal(rng, (size_t) T * HV * S, 1.5f);
            h2d(h, hh); h2d(gate, gt); h2d(beta, bt); h2d(z, zz);
            ck(cudaMemset(y_o, 0xff, (size_t) T * HV * S * 4), "memset");
            ck(cudaMemset(y_n, 0x7f, (size_t) T * HV * S * 4), "memset");
            ck(cudaMemset(y16_o, 0xff, (size_t) T * HV * S * 2), "memset");
            ck(cudaMemset(y16_n, 0x7f, (size_t) T * HV * S * 2), "memset");
            p::gdn_recurrence(st_o, h, gate, beta, z, gamma, eps, y_o, y16_o, T, g_s, p::KernelPath::Old);
            p::gdn_recurrence(st_n, h, gate, beta, z, gamma, eps, y_n, y16_n, T, g_s, p::KernelPath::New);
            const std::vector<float> yn = d2h(y_n, (size_t) T * HV * S);
            same("gdn_recurrence y", T, d2h(y_o, (size_t) T * HV * S), yn);
            same("gdn_recurrence y16", T, d2h(y16_o, (size_t) T * HV * S), d2h(y16_n, (size_t) T * HV * S));
            same("gdn_recurrence state", T, d2h(st_o, n_state), d2h(st_n, n_state));
            std::vector<double> yr;
            rec_reference(st_ref, hh, gt, bt, zz, gm, eps, T, yr);
            for (size_t i = 0; i < yr.size(); ++i)
                worst = std::fmax(worst, std::fabs((double) yn[i] - yr[i]) / (1.0 + std::fabs(yr[i])));
        }
    }
    if (!(worst < 1e-3)) { std::fprintf(stderr, "FAIL gdn_recurrence vs host reference: %.3g\n", worst); ++g_fail; }
    std::printf("gdn_recurrence: split vs block per head bitwise over %zu chunk pairs; host reference %.2g\n",
                sizeof Ts / sizeof Ts[0], worst);
    cudaFree(st_o); cudaFree(st_n); cudaFree(h); cudaFree(gate); cudaFree(beta); cudaFree(z); cudaFree(gamma);
    cudaFree(y_o); cudaFree(y_n); cudaFree(y16_o); cudaFree(y16_n);
}

// ---------------------------------------------------------------- 3. the GR chain
void test_gr(std::mt19937& rng) {
    const float eps = 1e-6f;
    const int64_t Tmax = 70;
    const size_t nD = (size_t) Tmax * D, nN = (size_t) Tmax * N;
    float *R_o = dalloc<float>(nD), *R_n = dalloc<float>(nD), *xn = dalloc<float>(nD), *rs = dalloc<float>((size_t) Tmax * 4);
    uint16_t *x16_o = dalloc<uint16_t>(nD), *x16_n = dalloc<uint16_t>(nD);
    float *gated = dalloc<float>(nD), *bo = dalloc<float>(nN), *inj = dalloc<float>((size_t) Tmax * 4);
    float *w0 = dalloc<float>(D), *w1 = dalloc<float>(D), *mix_o = dalloc<float>(nN), *mix_n = dalloc<float>(nN);
    uint16_t *m16_o = dalloc<uint16_t>(nN), *m16_n = dalloc<uint16_t>(nN), *mh_o = dalloc<uint16_t>(nN), *mh_n = dalloc<uint16_t>(nN);
    h2d(w0, normal(rng, D, 0.1f, 1.0f));
    h2d(w1, normal(rng, D, 0.1f, 1.0f));
    for (int64_t T : {(int64_t) 1, (int64_t) 3, (int64_t) 64, Tmax}) {
        std::vector<float> R = normal(rng, (size_t) T * D, 1.0f);
        for (size_t i = 0; i < R.size(); i += 1001) R[i] *= 50.0f;   // outliers, as the residual stream has
        h2d(R_o, R);
        h2d(R_n, R);
        h2d(gated, normal(rng, (size_t) T * D, 2.0f));
        h2d(bo, normal(rng, (size_t) T * N, 1.0f));
        h2d(inj, normal(rng, (size_t) T * 4, 2.0f));
        auto fill = [&](int side) {
            const int b = side ? 0x7f : 0xff;
            ck(cudaMemset(side ? x16_n : x16_o, b, (size_t) T * D * 2), "memset");
            ck(cudaMemset(side ? mix_n : mix_o, b, (size_t) T * N * 4), "memset");
            ck(cudaMemset(side ? m16_n : m16_o, b, (size_t) T * N * 2), "memset");
            ck(cudaMemset(side ? mh_n : mh_o, b, (size_t) T * N * 2), "memset");
        };
        auto cmp = [&](const char* stage) {
            char what[96];
            std::snprintf(what, sizeof what, "gr %s xn16", stage);
            same(what, T, d2h(x16_o, (size_t) T * D), d2h(x16_n, (size_t) T * D));
            std::snprintf(what, sizeof what, "gr %s mixed", stage);
            same(what, T, d2h(mix_o, (size_t) T * N), d2h(mix_n, (size_t) T * N));
            std::snprintf(what, sizeof what, "gr %s mixed16", stage);
            same(what, T, d2h(m16_o, (size_t) T * N), d2h(m16_n, (size_t) T * N));
            std::snprintf(what, sizeof what, "gr %s mixed_h", stage);
            same(what, T, d2h(mh_o, (size_t) T * N), d2h(mh_n, (size_t) T * N));
        };
        // the first half: norm + mix
        fill(0); fill(1);
        p::gr_norm(R_o, w0, eps, xn, x16_o, T, g_s);
        p::gr_mix(xn, gated, mix_o, m16_o, T, g_s, mh_o);
        p::gr_norm_scale(R_n, w0, eps, rs, x16_n, T, g_s);
        p::gr_mix_scale(R_n, rs, w0, gated, mix_n, m16_n, T, g_s, mh_n);
        cmp("norm+mix");
        // its write, then the next half's norm + mix
        fill(0); fill(1);
        p::gr_write(R_o, bo, inj, 4, T, g_s);
        p::gr_norm(R_o, w1, eps, xn, x16_o, T, g_s);
        p::gr_mix(xn, gated, mix_o, m16_o, T, g_s, mh_o);
        p::gr_write_norm(R_n, bo, inj, 4, w1, eps, rs, x16_n, T, g_s);
        p::gr_mix_scale(R_n, rs, w1, gated, mix_n, m16_n, T, g_s, mh_n);
        same("gr write R", T, d2h(R_o, (size_t) T * D), d2h(R_n, (size_t) T * D));
        cmp("write+norm+mix");
    }
    std::printf("gr chain: row scales and the fused write+norm vs gr_norm/gr_mix/gr_write, bitwise\n");
    for (void* q : {(void*) R_o, (void*) R_n, (void*) xn, (void*) rs, (void*) x16_o, (void*) x16_n, (void*) gated,
                    (void*) bo, (void*) inj, (void*) w0, (void*) w1, (void*) mix_o, (void*) mix_n, (void*) m16_o,
                    (void*) m16_n, (void*) mh_o, (void*) mh_n})
        cudaFree(q);
}

// ---------------------------------------------------------------- 4. FP16 saturation
void test_f16_sat(std::mt19937& rng) {
    const float inf = INFINITY;
    const std::vector<float> x = {1.5f, -2.0f, 65504.0f, 65519.0f, 65520.0f, 70000.0f, -70000.0f, 1e30f, -1e30f,
                                  inf, -inf, NAN, 6e-8f, 0.0f};
    const std::vector<uint16_t> off = {0x3e00, 0xc000, 0x7bff, 0x7bff, 0x7c00, 0x7c00, 0xfc00, 0x7c00, 0xfc00,
                                       0x7c00, 0xfc00, 0x0000 /* NaN, checked apart */, 0x0001, 0x0000};
    std::vector<uint16_t> on = off;
    on[4] = on[5] = on[7] = 0x7bff;
    on[6] = on[8] = 0xfbff;
    const size_t n = x.size();
    float* dx = dalloc<float>(n);
    uint16_t* dy = dalloc<uint16_t>(n);
    h2d(dx, x);
    for (int sat = 0; sat < 2; ++sat) {
        p::set_f16_saturate(sat != 0);
        p::to_f16(dx, dy, (int64_t) n, g_s);
        const std::vector<uint16_t> y = d2h(dy, n);
        const std::vector<uint16_t>& want = sat ? on : off;
        for (size_t i = 0; i < n; ++i) {
            const bool ok = i == 11 ? ((y[i] & 0x7c00) == 0x7c00 && (y[i] & 0x03ff) != 0) : y[i] == want[i];
            if (!ok) { std::fprintf(stderr, "FAIL to_f16 sat %d: %g -> 0x%04x, want 0x%04x\n", sat, x[i], y[i], want[i]); ++g_fail; }
        }
    }
    // another kernel sees the flag too: the shared expert's SwiGLU with products past the range
    const int64_t rows = 4;
    std::vector<float> gg = normal(rng, (size_t) rows * 640, 3.0f), uu = normal(rng, (size_t) rows * 640, 3.0f);
    for (size_t i = 0; i < gg.size(); i += 5) { gg[i] = 400.0f; uu[i] = (i & 1) ? 300.0f : -300.0f; }
    float *dg = dalloc<float>(gg.size()), *du = dalloc<float>(uu.size());
    uint16_t *h_off = dalloc<uint16_t>(gg.size()), *h_on = dalloc<uint16_t>(gg.size());
    h2d(dg, gg);
    h2d(du, uu);
    p::set_f16_saturate(false);
    p::swiglu_pair(dg, du, h_off, rows, g_s);
    p::set_f16_saturate(true);
    p::swiglu_pair(dg, du, h_on, rows, g_s);
    p::set_f16_saturate(false);
    const std::vector<uint16_t> a = d2h(h_off, gg.size()), b = d2h(h_on, gg.size());
    size_t sat_seen = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool big = (i % 5) == 0;   // 400 * 300 = 120000
        const uint16_t want_on = big ? (uu[i] > 0 ? 0x7bff : 0xfbff) : a[i];
        if ((big && (a[i] & 0x7fff) != 0x7c00) || b[i] != want_on) {
            if (g_fail < 20) std::fprintf(stderr, "FAIL swiglu_pair sat %zu: off 0x%04x on 0x%04x\n", i, a[i], b[i]);
            ++g_fail;
        }
        sat_seen += big;
    }
    std::printf("fp16 saturation: to_f16 edge values off/on, swiglu_pair %zu saturated products, others unchanged\n",
                sat_seen);
    cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(du); cudaFree(h_off); cudaFree(h_on);
}

// ---------------------------------------------------------------- 5. the router
void test_route(std::mt19937& rng) {
    const int64_t T = 45, E = 512, K = 10;
    std::vector<float> lg = normal(rng, (size_t) (T * E), 2.0f);
    for (int64_t e = 0; e < E; ++e) lg[(size_t) (3 * E + e)] = 0.5f;                          // a flat row: ids 0-9
    lg[(size_t) (7 * E + 40)] = lg[(size_t) (7 * E + 300)] = 9.0f;                              // a tie at the top
    for (int64_t e = 0; e < E; ++e) lg[(size_t) (11 * E + e)] = e < 3 ? 80.0f - e : -80.0f;     // underflowing tail
    for (int64_t e = 0; e < E; ++e) lg[(size_t) (13 * E + e)] = 1.0f + 1e-7f * (float) (e % 17); // near-ties
    float* d_lg = dalloc<float>((size_t) (T * E));
    int32_t *id_a = dalloc<int32_t>((size_t) (T * K)), *id_b = dalloc<int32_t>((size_t) (T * K));
    float *w_a = dalloc<float>((size_t) (T * K)), *w_b = dalloc<float>((size_t) (T * K));
    h2d(d_lg, lg);
    for (int64_t t = 0; t < T; ++t) strata::kernels::native_router_top10(d_lg + t * E, id_a + t * K, w_a + t * K, g_s);
    p::route_native(d_lg, id_b, w_b, T, g_s);
    const std::vector<int32_t> ia = d2h(id_a, (size_t) (T * K));
    const std::vector<float> wa = d2h(w_a, (size_t) (T * K));
    same("route_native ids", T, ia, d2h(id_b, (size_t) (T * K)));
    same("route_native weights", T, wa, d2h(w_b, (size_t) (T * K)));
    for (int k = 0; k < K; ++k)
        if (ia[(size_t) (3 * K + k)] != k) { std::fprintf(stderr, "FAIL flat row: rank %d is expert %d\n", k, ia[(size_t) (3 * K + k)]); ++g_fail; }
    // route_decode follows the token path's switch
    const bool was = strata::kernels::native_router_enabled();
    strata::kernels::native_router_set_enabled(true);
    ck(cudaMemset(id_b, 0xff, (size_t) (T * K) * 4), "memset");
    p::route_decode(d_lg, id_b, w_b, T, E, g_s);
    same("route_decode (native) ids", T, ia, d2h(id_b, (size_t) (T * K)));
    same("route_decode (native) weights", T, wa, d2h(w_b, (size_t) (T * K)));
    strata::kernels::native_router_set_enabled(false);
    strata::kernels::router_top10(d_lg, (int) T, (int) E, (int) K, id_a, w_a, g_s);
    p::route_decode(d_lg, id_b, w_b, T, E, g_s);
    same("route_decode (router_top10) ids", T, d2h(id_a, (size_t) (T * K)), d2h(id_b, (size_t) (T * K)));
    same("route_decode (router_top10) weights", T, d2h(w_a, (size_t) (T * K)), d2h(w_b, (size_t) (T * K)));
    strata::kernels::native_router_set_enabled(was);
    // the default prompt router against the native one: different expert sets (reported only)
    p::route(d_lg, id_b, w_b, T, E, g_s);
    const std::vector<int32_t> ib = d2h(id_b, (size_t) (T * K));
    const std::vector<float> wb = d2h(w_b, (size_t) (T * K));
    int64_t rows = 0;
    double wmax = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        bool diff = false;
        for (int k = 0; k < K; ++k) {
            diff |= ia[(size_t) (t * K + k)] != ib[(size_t) (t * K + k)];
            wmax = std::fmax(wmax, std::fabs((double) wa[(size_t) (t * K + k)] - wb[(size_t) (t * K + k)]));
        }
        rows += diff;
    }
    std::printf("router: route_native == native_router_top10 bitwise over %lld tokens; default prompt router differs "
                "on %lld rows (ids or order), max weight difference %.2g\n", (long long) T, (long long) rows, wmax);
    cudaFree(d_lg); cudaFree(id_a); cudaFree(id_b); cudaFree(w_a); cudaFree(w_b);
}
}  // namespace

int main() {
    ck(cudaStreamCreate(&g_s), "stream");   // a blocking stream: ordered with the plain cudaMemcpy/cudaMemset
    std::mt19937 rng(20260928);
    test_conv(rng);
    test_rec(rng);
    test_gr(rng);
    test_f16_sat(rng);
    test_route(rng);
    std::printf("prefill_kernels_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
