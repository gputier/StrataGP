// src/kernels/gdn_fused_parity.cpp - the GDN kernels `--native` runs, against float64, over MANY steps (#8, #5).
//
// `gdn_parity` checks the legacy kernels for ONE step.  With `--native` the layer runs `fused_gdn_conv_l2`,
// `fused_gdn_ab` and `fused_gdn_step_norm` (and with `--no-fused-gdn` the native_gdn_* chain, llama.cpp's
// layout), and none of them had a test.  Four sections, all on synthetic data at the artifact's geometry:
//
//   1. THE RECURRENCE OVER 256 STEPS.  The three step kernels - legacy `gdn_step`, `native_gdn_step` and
//      `fused_gdn_step_norm` - each carry their own device state through the same 256 inputs, against ONE
//      float64 state.  A single step cannot show drift: an f32 rounding bias, or a decay computed with a biased
//      `__expf`, only becomes visible once the state has integrated it.  Two gate regimes: mixed decays, and a
//      long memory (decay 0.98-0.9995) where such a bias accumulates longest.  The error is printed at steps 1,
//      16, 64 and 256 so the growth itself is on record.
//   2. THE SOFTPLUS (#5).  Three kernels compute softplus(alpha + dt) three ways: the legacy `log1pf(expf(v))`,
//      the fused `log1pf(__expf(v))` (the production default with `--native`), and the native
//      `logf(1.0f + expf(v))` under --use_fast_math, which is ggml-cuda's `op_softplus` at the pinned llama.cpp
//      verbatim (threshold 20 included).  The last one is exact 0 for v <~ -17 and loses relative precision
//      below v ~ -5; it is asserted against llama.cpp's expression and the fused one against float64, and the
//      gap between them is printed per range of v.
//   3. `fused_gdn_ab`: the alpha/beta BF16 projections and their epilogues, against float64.
//   4. `fused_gdn_conv_l2` and the native preprocessing (conv+SiLU, L2 norm, beta sigmoid, output norm).
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T>
void up(T* d, const std::vector<T>& h) { check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

template <typename A, typename B>
double rel_l1(const std::vector<A>& ref, const std::vector<B>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        d += std::fabs((double) ref[i] - (double) got[i]);
        m += std::fabs((double) ref[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

double sigmoid64(double x) { return 1.0 / (1.0 + std::exp(-x)); }
/// softplus in float64, without a threshold: the function itself
double softplus64(double v) { return v > 0 ? v + std::log1p(std::exp(-v)) : std::log1p(std::exp(v)); }

constexpr int S = 128, HK = 16, HV = 48;       // the artifact's GDN geometry
constexpr float EPS = 1e-6f;                   // RMS_EPS in layer.cpp

// ================================ 1. the recurrence ================================

/// One float64 step, state in the kernels' (S, h_v, S) layout, q UNSCALED; returns o = S^T q (unscaled).
void ref_step(std::vector<double>& st, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              std::vector<double>& o) {
    o.assign((size_t) HV * S, 0.0);
    for (int h = 0; h < HV; ++h) {
        const int qh = h % HK;
        const double dec = std::exp((double) gate[(size_t) h]), b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            auto at = [&](int i) -> double& { return st[((size_t) i * HV + h) * S + j]; };
            double sk = 0;
            for (int i = 0; i < S; ++i) {
                at(i) *= dec;
                sk += at(i) * (double) k[(size_t) qh * S + i];
            }
            const double d = ((double) v[(size_t) h * S + j] - sk) * b;
            double dot = 0;
            for (int i = 0; i < S; ++i) {
                at(i) += (double) k[(size_t) qh * S + i] * d;
                dot += at(i) * (double) q[(size_t) qh * S + i];
            }
            o[(size_t) h * S + j] = dot;
        }
    }
}

int recurrence(const char* regime, float gate_lo, float gate_hi, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(gate_lo, gate_hi);
    const int steps = 256;
    const float qscale = 1.0f / std::sqrt((float) S);
    const size_t nst = (size_t) S * HV * S;

    std::vector<double> st(nst, 0.0);
    std::vector<float> gamma((size_t) S);
    for (auto& x : gamma) x = 1.0f + 0.1f * gauss(rng);

    float* d_st[3] = {dev<float>(nst), dev<float>(nst), dev<float>(nst)};   // legacy, native, fused
    for (float* p : d_st) check(cudaMemset(p, 0, nst * 4), "zero state");
    float *d_q = dev<float>((size_t) HK * S), *d_qs = dev<float>((size_t) HK * S), *d_k = dev<float>((size_t) HK * S);
    float *d_v = dev<float>((size_t) HV * S), *d_g = dev<float>(HV), *d_b = dev<float>(HV), *d_z = dev<float>((size_t) HV * S);
    float *d_gamma = dev<float>(S), *d_o_leg = dev<float>((size_t) HV * S), *d_o_nat = dev<float>((size_t) HV * S);
    float* d_y = dev<float>((size_t) HV * S);
    up(d_gamma, gamma);
    cudaStream_t cs = nullptr;
    check(cudaStreamCreate(&cs), "stream");
    const strata::kernels::GdnShapes sh{S, HK, HV};

    int bad = 0;
    std::printf("  recurrence, %s (gate in [%.4g, %.4g], decay %.4f-%.4f), rel L1 vs float64:\n", regime,
                (double) gate_lo, (double) gate_hi, std::exp((double) gate_lo), std::exp((double) gate_hi));
    std::printf("    %5s  %11s %11s  %11s %11s  %11s %11s  %10s\n", "step", "legacy S", "legacy o", "native S",
                "native o", "fused S", "fused y", "tolerance");
    std::vector<double> o64;
    for (int step = 1; step <= steps; ++step) {
        std::vector<float> q((size_t) HK * S), k((size_t) HK * S), v((size_t) HV * S), gate(HV), beta(HV),
            z((size_t) HV * S), qs((size_t) HK * S);
        // q and k are L2-normalised per head, as the layer's l2_norm leaves them
        for (auto* vec : {&q, &k}) {
            for (auto& x : *vec) x = gauss(rng);
            for (int h = 0; h < HK; ++h) {
                double ss = 0;
                for (int i = 0; i < S; ++i) ss += (double) (*vec)[(size_t) h * S + i] * (*vec)[(size_t) h * S + i];
                const float inv = (float) (1.0 / std::sqrt(ss + 1e-6));
                for (int i = 0; i < S; ++i) (*vec)[(size_t) h * S + i] *= inv;
            }
        }
        for (size_t i = 0; i < q.size(); ++i) qs[i] = q[i] * qscale;       // the legacy step takes q prescaled
        for (auto& x : v) x = gauss(rng);
        for (auto& x : gate) x = uni(rng);
        for (auto& x : beta) x = (float) sigmoid64((double) gauss(rng));
        for (auto& x : z) x = gauss(rng);
        up(d_q, q); up(d_qs, qs); up(d_k, k); up(d_v, v); up(d_g, gate); up(d_b, beta); up(d_z, z);

        strata::kernels::gdn_step(d_st[0], d_qs, d_k, d_v, d_g, d_b, d_o_leg, sh, cs);
        strata::kernels::native_gdn_step(d_st[1], d_q, d_k, d_v, d_g, d_b, d_o_nat, sh, cs);
        strata::kernels::fused_gdn_step_norm(d_st[2], d_q, d_k, d_v, d_g, d_b, d_z, d_gamma, EPS, d_y, HK, HV, cs);
        ref_step(st, q, k, v, gate, beta, o64);
        check(cudaStreamSynchronize(cs), "step");

        if (step != 1 && step != 16 && step != 64 && step != steps) continue;
        std::vector<double> o_ref((size_t) HV * S), y_ref((size_t) HV * S);
        for (size_t i = 0; i < o_ref.size(); ++i) o_ref[i] = o64[i] / std::sqrt((double) S);
        for (int h = 0; h < HV; ++h) {
            double ss = 0;
            for (int j = 0; j < S; ++j) ss += o_ref[(size_t) h * S + j] * o_ref[(size_t) h * S + j];
            const double inv = 1.0 / std::sqrt(ss / S + (double) EPS);
            for (int j = 0; j < S; ++j)
                y_ref[(size_t) h * S + j] = o_ref[(size_t) h * S + j] * inv * (double) gamma[(size_t) j] *
                                            sigmoid64((double) z[(size_t) h * S + j]);
        }
        const double e[6] = {rel_l1(st, down(d_st[0], nst)), rel_l1(o_ref, down(d_o_leg, o_ref.size())),
                             rel_l1(st, down(d_st[1], nst)), rel_l1(o_ref, down(d_o_nat, o_ref.size())),
                             rel_l1(st, down(d_st[2], nst)), rel_l1(y_ref, down(d_y, y_ref.size()))};
        // f32 rounding in a contraction (|k| = 1, 0 < beta < 1) walks, and a biased decay adds up for as long as
        // the state remembers.  A host model of the fused kernel's order (f32 fmaf chains per row group) measured
        // ~2e-7 at step 256 in the long-memory regime, and 1.3e-5 when EVERY decay is pushed 2 ulp the same way (a
        // worst-case __expf bias).  The tolerance is 4x that worst case at step 256 and 40x the unbiased drift at
        // step 1; a structural fault (a head, the decay order, a stale shared-memory reduction) is 1e-2 or more.
        const double tol = 5e-6 + 2e-7 * step;
        bool ok = true;
        for (double x : e) ok = ok && x <= tol;
        std::printf("    %5d  %11.3e %11.3e  %11.3e %11.3e  %11.3e %11.3e  %10.2e  %s\n", step, e[0], e[1], e[2], e[3],
                    e[4], e[5], tol, ok ? "ok" : "*** OVER ***");
        if (!ok) ++bad;
    }
    check(cudaStreamDestroy(cs), "stream destroy");
    for (float* p : d_st) cudaFree(p);
    for (float* p : {d_q, d_qs, d_k, d_v, d_g, d_b, d_z, d_gamma, d_o_leg, d_o_nat, d_y}) cudaFree(p);
    return bad;
}

// ================================ 2. the softplus ================================

int softplus() {
    // v on a grid over [-40, 30] plus the edges of the three regimes; alpha = v and dt = 0 for the native gate,
    // zero alpha weights and dt = v for the fused one (its dot product is then exactly 0), ssm_a = -1 for both
    std::vector<float> v;
    for (int i = 0; i <= 1400; ++i) v.push_back(-40.0f + 0.05f * (float) i);
    for (float x : {20.0f, std::nextafter(20.0f, 30.0f), std::nextafter(20.0f, 0.0f), -17.0f, -16.6f, -5.0f, 0.0f})
        v.push_back(x);
    const int n = (int) v.size();
    std::vector<float> zeros((size_t) n, 0.0f), minus1((size_t) n, -1.0f);

    float *d_alpha = dev<float>(n), *d_dt = dev<float>(n), *d_a = dev<float>(n), *d_gate = dev<float>(n);
    up(d_alpha, v); up(d_dt, zeros); up(d_a, minus1);
    cudaStream_t cs = nullptr;
    check(cudaStreamCreate(&cs), "sp stream");
    strata::kernels::native_gdn_gate(d_alpha, d_dt, d_a, d_gate, n, cs);
    check(cudaStreamSynchronize(cs), "native gate");
    const std::vector<float> g_nat = down(d_gate, (size_t) n);

    const int ne = 8;
    std::vector<float> x((size_t) ne, 1.0f);
    std::vector<uint16_t> w((size_t) n * ne, 0);
    float *d_x = dev<float>(ne), *d_beta = dev<float>(n);
    uint16_t* d_w = dev<uint16_t>(w.size());
    up(d_x, x); up(d_w, w); up(d_dt, v);
    strata::kernels::fused_gdn_ab(d_x, d_w, d_w, d_dt, d_a, d_gate, d_beta, ne, n, cs);
    check(cudaStreamSynchronize(cs), "fused ab");
    const std::vector<float> g_fus = down(d_gate, (size_t) n);
    const std::vector<float> b_fus = down(d_beta, (size_t) n);

    int bad_llama = 0, bad_exact = 0, bad_beta = 0;
    struct Band { float lo, hi; double nat_worst, fus_worst; };
    Band bands[] = {{-40.0f, -17.0f, 0, 0}, {-17.0f, -5.0f, 0, 0}, {-5.0f, 20.0f, 0, 0}, {20.0f, 31.0f, 0, 0}};
    float nat_zero_top = -1e30f;
    for (int i = 0; i < n; ++i) {
        const float vi = v[(size_t) i];
        const double exact = softplus64((double) vi);
        const float llama = vi > 20.0f ? vi : std::log(1.0f + std::exp(vi));     // ggml-cuda op_softplus, host f32
        const double nat = -(double) g_nat[(size_t) i], fus = -(double) g_fus[(size_t) i];
        // native vs llama.cpp's expression: the device's __expf/__logf against the host's libm.  __logf has an
        // absolute error of 2^-21.4 on [0.5, 2], 1 + e can round one ulp apart, and __expf's error is relative.
        if (!(std::fabs(nat - (double) llama) <= 1e-6 + 1e-5 * std::fabs((double) llama))) {
            if (bad_llama < 5) std::printf("    native v=%.6g: %.9g, llama.cpp expression %.9g\n", (double) vi, nat, (double) llama);
            ++bad_llama;
        }
        // fused vs the function: log1pf is accurate, __expf(v) is ~2 ulp plus |v log2 e| ulps of the scaled input
        if (!(std::fabs(fus - exact) <= 2e-5 * exact + 1e-37)) {
            if (bad_exact < 5) std::printf("    fused v=%.6g: %.9g, float64 %.9g\n", (double) vi, fus, exact);
            ++bad_exact;
        }
        if (b_fus[(size_t) i] != 0.5f) ++bad_beta;                               // sigmoid(0), exactly
        if (nat == 0.0) nat_zero_top = std::max(nat_zero_top, vi);
        for (Band& b : bands)
            if (vi >= b.lo && vi < b.hi) {
                b.nat_worst = std::max(b.nat_worst, std::fabs(nat - exact) / exact);
                b.fus_worst = std::max(b.fus_worst, std::fabs(fus - exact) / exact);
            }
    }
    std::printf("  softplus(v), %d values: native (llama.cpp's logf(1+expf), fast-math) vs its host expression: %d over;"
                " fused (log1pf(__expf)) vs float64: %d over\n", n, bad_llama, bad_exact);
    std::printf("    %-18s %16s %16s\n", "v range", "native rel err", "fused rel err");
    for (const Band& b : bands)
        std::printf("    [%6.1f, %6.1f)   %16.3e %16.3e\n", (double) b.lo, (double) b.hi, b.nat_worst, b.fus_worst);
    std::printf("    native softplus is exactly 0 up to v = %.4g (the decay exp(gate) is then exactly 1);"
                " the fused one is not\n", (double) nat_zero_top);
    if (bad_beta) std::printf("    *** fused beta != sigmoid(0) = 0.5 on %d rows ***\n", bad_beta);
    for (float* p : {d_alpha, d_dt, d_a, d_gate, d_x, d_beta}) cudaFree(p);
    cudaFree(d_w);
    check(cudaStreamDestroy(cs), "sp stream destroy");
    return (bad_llama ? 1 : 0) + (bad_exact ? 1 : 0) + (bad_beta ? 1 : 0);
}

// ================================ 3. fused_gdn_ab ================================

int alpha_beta() {
    const int NE = 2560;
    std::mt19937 rng(505);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) NE), dt(HV), a(HV);
    std::vector<uint16_t> wa((size_t) HV * NE), wb((size_t) HV * NE);
    for (auto& e : x) e = gauss(rng);
    for (auto& e : wa) e = strata::kernels::bf16_from_f32(0.02f * gauss(rng));
    for (auto& e : wb) e = strata::kernels::bf16_from_f32(0.02f * gauss(rng));
    // dt spreads the heads over every softplus regime (v from about -16 to 22); ssm_a = -exp(A_log)
    for (int h = 0; h < HV; ++h) {
        dt[(size_t) h] = -16.0f + 38.0f * (float) h / (float) (HV - 1);
        a[(size_t) h] = -std::exp(-1.0f + 3.5f * (float) ((rng() % 1000) / 1000.0));
    }
    float *d_x = dev<float>(NE), *d_dt = dev<float>(HV), *d_a = dev<float>(HV), *d_g = dev<float>(HV), *d_b = dev<float>(HV);
    uint16_t *d_wa = dev<uint16_t>(wa.size()), *d_wb = dev<uint16_t>(wb.size());
    up(d_x, x); up(d_dt, dt); up(d_a, a); up(d_wa, wa); up(d_wb, wb);
    strata::kernels::fused_gdn_ab(d_x, d_wa, d_wb, d_dt, d_a, d_g, d_b, NE, HV, nullptr);
    check(cudaDeviceSynchronize(), "fused ab");
    const std::vector<float> g = down(d_g, HV), b = down(d_b, HV);
    int bad = 0;
    double worst_g = 0, worst_b = 0;
    for (int h = 0; h < HV; ++h) {
        double da = 0, db = 0, sa = 0, sb = 0;
        for (int i = 0; i < NE; ++i) {
            const double xi = x[(size_t) i];
            const double wai = strata::kernels::f32_from_bf16(wa[(size_t) h * NE + i]);
            const double wbi = strata::kernels::f32_from_bf16(wb[(size_t) h * NE + i]);
            da += wai * xi; db += wbi * xi;
            sa += std::fabs(wai * xi); sb += std::fabs(wbi * xi);
        }
        const double vv = da + (double) dt[(size_t) h];
        const double g_ref = softplus64(vv) * (double) a[(size_t) h], b_ref = sigmoid64(db);
        // an f32 dot of 2560 terms (80 fmaf per lane, then a 5-level tree) is within 85 * 2^-24 ~ 5e-6 sum|w x|
        // of the exact one whatever the order; softplus' slope is sigmoid(v) <= 1 and the sigmoid's is <= 1/4
        const double tg = std::fabs((double) a[(size_t) h]) *
                          (sigmoid64(vv) * (5e-6 * sa + 1e-6 * std::fabs(vv)) + 2e-5 * softplus64(vv));
        const double tb = 0.25 * 5e-6 * sb + 2e-6;
        const double eg = std::fabs((double) g[(size_t) h] - g_ref), eb = std::fabs((double) b[(size_t) h] - b_ref);
        worst_g = std::max(worst_g, eg / tg);
        worst_b = std::max(worst_b, eb / tb);
        if (!(eg <= tg) || !(eb <= tb)) {
            if (bad < 5) std::printf("    head %d: gate %.9g vs %.9g, beta %.9g vs %.9g\n", h, (double) g[(size_t) h],
                                     g_ref, (double) b[(size_t) h], b_ref);
            ++bad;
        }
    }
    std::printf("  fused_gdn_ab (2560 x %d BF16, v over [-16, 22]): %d heads over; worst error / tolerance: gate %.3f,"
                " beta %.3f\n", HV, bad, worst_g, worst_b);
    for (float* p : {d_x, d_dt, d_a, d_g, d_b}) cudaFree(p);
    cudaFree(d_wa); cudaFree(d_wb);
    return bad ? 1 : 0;
}

// ================================ 4. conv + L2, and the native preprocessing ================================

int preprocessing() {
    const int C = 2 * HK * S + HV * S;          // 10240: q | k | v
    const int QK_HEADS = 2 * HK;
    std::mt19937 rng(606);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> hist((size_t) C * 3), qkv((size_t) C), w((size_t) C * 4);
    for (auto& e : hist) e = gauss(rng);
    for (auto& e : qkv) e = gauss(rng);
    for (auto& e : w) e = 0.5f * gauss(rng);

    // float64 conv + SiLU, and the history slide (oldest first, the new input last)
    std::vector<double> raw((size_t) C), silu((size_t) C), l2((size_t) C);
    std::vector<float> hist_want((size_t) C * 3);
    for (int c = 0; c < C; ++c) {
        const double in[4] = {hist[(size_t) c * 3], hist[(size_t) c * 3 + 1], hist[(size_t) c * 3 + 2], qkv[(size_t) c]};
        double s = 0;
        for (int t = 0; t < 4; ++t) s += in[t] * (double) w[(size_t) c * 4 + t];
        raw[(size_t) c] = s;
        silu[(size_t) c] = s * sigmoid64(s);
        hist_want[(size_t) c * 3] = hist[(size_t) c * 3 + 1];
        hist_want[(size_t) c * 3 + 1] = hist[(size_t) c * 3 + 2];
        hist_want[(size_t) c * 3 + 2] = qkv[(size_t) c];
    }
    // the L2 norm of each of the first 32 heads: x / sqrt(sum(x^2) + eps); v passes through
    l2 = silu;
    for (int h = 0; h < QK_HEADS; ++h) {
        double ss = 0;
        for (int i = 0; i < S; ++i) ss += silu[(size_t) h * S + i] * silu[(size_t) h * S + i];
        for (int i = 0; i < S; ++i) l2[(size_t) h * S + i] = silu[(size_t) h * S + i] / std::sqrt(ss + (double) EPS);
    }

    int bad = 0;
    cudaStream_t cs = nullptr;
    check(cudaStreamCreate(&cs), "pre stream");
    float *d_hist = dev<float>(hist.size()), *d_qkv = dev<float>(C), *d_w = dev<float>(w.size()), *d_h = dev<float>(C);
    float* d_raw = dev<float>(C);
    up(d_qkv, qkv); up(d_w, w);

    // ---- fused_gdn_conv_l2
    up(d_hist, hist);
    strata::kernels::fused_gdn_conv_l2(d_hist, d_qkv, d_w, d_h, C, QK_HEADS, EPS, cs);
    check(cudaStreamSynchronize(cs), "conv_l2");
    {
        const std::vector<float> h = down(d_h, (size_t) C), hs = down(d_hist, hist.size());
        const double e = rel_l1(l2, h);
        const bool slide = std::memcmp(hs.data(), hist_want.data(), hs.size() * 4) == 0;
        std::printf("  fused_gdn_conv_l2 (%d channels, %d normalised heads): rel %.3e, history slide %s\n", C, QK_HEADS, e,
                    slide ? "exact" : "*** WRONG ***");
        if (!(e <= 1e-5) || !slide) ++bad;
    }
    // ---- native: conv + SiLU, then the two L2 norms (q heads, k heads)
    up(d_hist, hist);
    strata::kernels::native_gdn_conv_silu(d_hist, d_qkv, d_w, d_raw, d_h, C, 4, cs);
    check(cudaStreamSynchronize(cs), "native conv");
    {
        const std::vector<float> r = down(d_raw, (size_t) C), s = down(d_h, (size_t) C), hs = down(d_hist, hist.size());
        const double er = rel_l1(raw, r), es = rel_l1(silu, s);
        const bool slide = std::memcmp(hs.data(), hist_want.data(), hs.size() * 4) == 0;
        std::printf("  native_gdn_conv_silu: raw rel %.3e, silu rel %.3e, history slide %s\n", er, es,
                    slide ? "exact" : "*** WRONG ***");
        if (!(er <= 2e-6) || !(es <= 1e-5) || !slide) ++bad;
    }
    strata::kernels::native_gdn_l2_norm(d_h, HK, S, EPS, cs);
    strata::kernels::native_gdn_l2_norm(d_h + (size_t) HK * S, HK, S, EPS, cs);
    check(cudaStreamSynchronize(cs), "native l2");
    {
        const std::vector<float> h = down(d_h, (size_t) C);
        const double e = rel_l1(l2, h);
        std::printf("  native_gdn_l2_norm (q and k heads): rel %.3e\n", e);
        if (!(e <= 1e-5)) ++bad;
    }
    // ---- native: beta sigmoid, and the output norm y = rms_norm(o) * gamma * sigmoid(z)
    {
        std::vector<float> beta(HV), o((size_t) HV * S), z((size_t) HV * S), gamma((size_t) S);
        for (auto& e : beta) e = 3.0f * gauss(rng);
        for (auto& e : o) e = gauss(rng);
        for (auto& e : z) e = gauss(rng);
        for (auto& e : gamma) e = 1.0f + 0.1f * gauss(rng);
        std::vector<double> beta_ref(HV), y_ref((size_t) HV * S);
        for (int h = 0; h < HV; ++h) beta_ref[(size_t) h] = sigmoid64((double) beta[(size_t) h]);
        for (int h = 0; h < HV; ++h) {
            double ss = 0;
            for (int j = 0; j < S; ++j) ss += (double) o[(size_t) h * S + j] * o[(size_t) h * S + j];
            const double inv = 1.0 / std::sqrt(ss / S + (double) EPS);
            for (int j = 0; j < S; ++j)
                y_ref[(size_t) h * S + j] = (double) o[(size_t) h * S + j] * inv * (double) gamma[(size_t) j] *
                                            sigmoid64((double) z[(size_t) h * S + j]);
        }
        float *d_beta = dev<float>(HV), *d_o = dev<float>(o.size()), *d_z = dev<float>(z.size());
        float *d_gm = dev<float>(S), *d_y = dev<float>(o.size());
        up(d_beta, beta); up(d_o, o); up(d_z, z); up(d_gm, gamma);
        strata::kernels::native_gdn_beta_gate(d_beta, HV, cs);
        strata::kernels::native_gdn_out_norm(d_o, d_z, d_gm, d_y, HV, S, EPS, cs);
        check(cudaStreamSynchronize(cs), "native beta/out");
        const double eb = rel_l1(beta_ref, down(d_beta, HV)), ey = rel_l1(y_ref, down(d_y, o.size()));
        std::printf("  native_gdn_beta_gate: rel %.3e;  native_gdn_out_norm: rel %.3e\n", eb, ey);
        if (!(eb <= 1e-5) || !(ey <= 1e-5)) ++bad;
        for (float* p : {d_beta, d_o, d_z, d_gm, d_y}) cudaFree(p);
    }
    for (float* p : {d_hist, d_qkv, d_w, d_h, d_raw}) cudaFree(p);
    check(cudaStreamDestroy(cs), "pre stream destroy");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gdn_fused_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    bad += recurrence("mixed decays", -3.0f, -0.01f, 2024);
    bad += recurrence("long memory", -0.02f, -0.0005f, 2025);
    std::printf("\n");
    bad += softplus();
    std::printf("\n");
    bad += alpha_beta();
    std::printf("\n");
    bad += preprocessing();

    std::printf("\ngdn_fused: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gdn_fused_parity OK\n");
    return 0;
}
