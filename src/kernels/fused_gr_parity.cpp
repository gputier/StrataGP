// src/kernels/fused_gr_parity.cpp - the fused hyper-connection read (`fused_gr.hpp`) against float64, and its
// T-token form against the single-token one (#8).
//
// `fused_gr_read` runs twice per layer per token in production (the `--native` default) and
// `fused_gr_read_multi` in every verify window and MTP draft, and neither had a test.  Two checks, on synthetic
// data at the artifact's geometry (n_embd 2560, hc 4, hc_lr 320):
//
//   1. THE SINGLE-TOKEN READ AGAINST A FLOAT64 TRANSCRIPTION OF THE HEADER'S CONTRACT, with and without the
//      folded write of the previous half (`apply`) and with and without the injection rows.  Every output is
//      compared: rs, lo, inject, mixed, and R itself (updated in place when `apply`).  The tolerances are f32
//      accumulation over 10,240-term dots and the __expf of the sigmoids - 1e-5 relative, where a wrong stream,
//      a transposed weight or a missing norm factor is O(1).
//   2. **THE MULTI-TOKEN READ IS BITWISE THE SINGLE ONE**, for T = 1..8, with a different R, pending write and
//      injection per token.  The header promises it ("every token's outputs are bitwise fused_gr_read(a[t])"),
//      and it is what makes a verify window reproduce plain decode exactly - so it is asserted, not measured.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr float EPS = 1e-6f;

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
double rel_l1(const std::vector<double>& ref, const std::vector<float>& got) {
    double d = 0, m = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        d += std::fabs(ref[i] - (double) got[i]);
        m += std::fabs(ref[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}
double sigmoid64(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Weights {
    std::vector<float> w_norm;
    std::vector<uint16_t> w_down, w_up, w_inject;
    float* d_norm = nullptr;
    uint16_t *d_down = nullptr, *d_up = nullptr, *d_inject = nullptr;
};

/// One token's inputs and its device buffers.
struct Token {
    std::vector<float> R, bo, inj;
    bool apply = false;
    float *d_R = nullptr, *d_bo = nullptr, *d_inj = nullptr, *d_lo = nullptr, *d_rs = nullptr, *d_inject = nullptr,
          *d_mixed = nullptr;
};

struct Out {
    std::vector<float> rs, lo, inject, mixed, R;
};

Token make_token(std::mt19937& rng, bool apply) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    Token t;
    t.apply = apply;
    t.R.resize(D);
    t.bo.resize(N);
    t.inj.resize(HC);
    for (auto& x : t.R) x = g(rng);
    for (auto& x : t.bo) x = g(rng);
    for (auto& x : t.inj) x = 2.0f * g(rng);
    t.d_R = dev<float>(D); t.d_bo = dev<float>(N); t.d_inj = dev<float>(HC); t.d_lo = dev<float>(LR);
    t.d_rs = dev<float>(HC); t.d_inject = dev<float>(HC); t.d_mixed = dev<float>(N);
    up(t.d_bo, t.bo); up(t.d_inj, t.inj);
    return t;
}

strata::kernels::FusedGrArgs args(const Weights& w, const Token& t, bool with_inject) {
    strata::kernels::FusedGrArgs a;
    a.R = t.d_R; a.R_out = t.d_R; a.apply = t.apply; a.bo_prev = t.d_bo; a.inj_prev = t.d_inj;
    a.w_norm = w.d_norm; a.w_down = w.d_down; a.w_up = w.d_up; a.w_inject = with_inject ? w.d_inject : nullptr;
    a.eps = EPS; a.lo = t.d_lo; a.rs = t.d_rs; a.inject_out = t.d_inject; a.mixed = t.d_mixed;
    return a;
}

Out read_back(const Token& t, bool with_inject) {
    Out o;
    o.rs = down(t.d_rs, HC); o.lo = down(t.d_lo, LR); o.mixed = down(t.d_mixed, N); o.R = down(t.d_R, D);
    o.inject = with_inject ? down(t.d_inject, HC) : std::vector<float>();
    return o;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// The header's contract in float64.
void reference(const Weights& w, const Token& t, bool with_inject, std::vector<double>& rs, std::vector<double>& lo,
               std::vector<double>& inject, std::vector<double>& mixed, std::vector<double>& Rp) {
    Rp.assign(D, 0.0);
    rs.assign(HC, 0.0);
    for (int c = 0; c < HC; ++c) {
        const double gw = t.apply ? 2.0 * sigmoid64((double) t.inj[(size_t) c] / HC) : 0.0;
        double ss = 0;
        for (int d = 0; d < N; ++d) {
            const double r = (double) t.R[(size_t) c * N + d] + (double) t.bo[(size_t) d] * gw;
            Rp[(size_t) c * N + d] = r;
            ss += r * r;
        }
        rs[(size_t) c] = 1.0 / std::sqrt(ss / N + (double) EPS);
    }
    std::vector<double> xn(D);
    for (int i = 0; i < D; ++i) xn[(size_t) i] = Rp[(size_t) i] * (double) w.w_norm[(size_t) i] * rs[(size_t) (i / N)];
    lo.assign(LR, 0.0);
    for (int k = 0; k < LR; ++k) {
        double acc = 0;
        for (int i = 0; i < D; ++i) acc += (double) strata::kernels::f32_from_bf16(w.w_down[(size_t) k * D + i]) * xn[(size_t) i];
        const double x = acc / HC;
        lo[(size_t) k] = x * sigmoid64(x);
    }
    inject.assign(with_inject ? HC : 0, 0.0);
    for (int c = 0; c < (with_inject ? HC : 0); ++c) {
        double acc = 0;
        for (int i = 0; i < D; ++i) acc += (double) strata::kernels::f32_from_bf16(w.w_inject[(size_t) c * D + i]) * xn[(size_t) i];
        inject[(size_t) c] = acc;
    }
    mixed.assign(N, 0.0);
    for (int i = 0; i < D; ++i) {
        double acc = 0;
        for (int k = 0; k < LR; ++k) acc += (double) strata::kernels::f32_from_bf16(w.w_up[(size_t) i * LR + k]) * lo[(size_t) k];
        mixed[(size_t) (i % N)] += xn[(size_t) i] * sigmoid64(acc) / HC;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: fused_gr_parity [--selftest]\n"); return 2; }
    }
    if (!strata::kernels::fused_gr_supported(N, HC, LR)) {
        std::fprintf(stderr, "fused_gr_parity: the kernel no longer supports the artifact's geometry\n");
        return 1;
    }
    std::mt19937 rng(4242);
    std::normal_distribution<float> g(0.0f, 1.0f);
    Weights w;
    w.w_norm.resize(D);
    w.w_down.resize((size_t) LR * D);
    w.w_up.resize((size_t) D * LR);
    w.w_inject.resize((size_t) HC * D);
    for (auto& x : w.w_norm) x = 1.0f + 0.1f * g(rng);
    for (auto& x : w.w_down) x = strata::kernels::bf16_from_f32(0.02f * g(rng));
    for (auto& x : w.w_up) x = strata::kernels::bf16_from_f32(0.05f * g(rng));
    for (auto& x : w.w_inject) x = strata::kernels::bf16_from_f32(0.02f * g(rng));
    w.d_norm = dev<float>(D); w.d_down = dev<uint16_t>(w.w_down.size()); w.d_up = dev<uint16_t>(w.w_up.size());
    w.d_inject = dev<uint16_t>(w.w_inject.size());
    up(w.d_norm, w.w_norm); up(w.d_down, w.w_down); up(w.d_up, w.w_up); up(w.d_inject, w.w_inject);
    cudaStream_t cs = nullptr;
    check(cudaStreamCreate(&cs), "stream");
    int bad = 0;

    // ---- 1. the single-token read against float64
    for (int variant = 0; variant < 4; ++variant) {
        const bool apply = (variant & 1) != 0, with_inject = (variant & 2) != 0;
        Token t = make_token(rng, apply);
        up(t.d_R, t.R);
        strata::kernels::fused_gr_read(args(w, t, with_inject), cs);
        check(cudaStreamSynchronize(cs), "fused_gr_read");
        const Out o = read_back(t, with_inject);
        std::vector<double> rs, lo, inject, mixed, Rp;
        reference(w, t, with_inject, rs, lo, inject, mixed, Rp);
        const double e_rs = rel_l1(rs, o.rs), e_lo = rel_l1(lo, o.lo), e_mx = rel_l1(mixed, o.mixed);
        const double e_in = with_inject ? rel_l1(inject, o.inject) : 0.0;
        // R is rewritten only when `apply`; otherwise it must come back bit for bit
        const double e_R = apply ? rel_l1(Rp, o.R) : (same_bits(t.R, o.R) ? 0.0 : 1.0);
        const bool ok = e_rs <= 1e-5 && e_lo <= 1e-5 && e_mx <= 1e-5 && e_in <= 1e-5 && e_R <= 2e-6;
        std::printf("  fused_gr_read apply=%d inject=%d: rs %.2e  lo %.2e  inject %.2e  mixed %.2e  R %.2e  %s\n",
                    (int) apply, (int) with_inject, e_rs, e_lo, e_in, e_mx, e_R, ok ? "ok" : "*** WRONG ***");
        if (!ok) ++bad;
        for (float* p : {t.d_R, t.d_bo, t.d_inj, t.d_lo, t.d_rs, t.d_inject, t.d_mixed}) cudaFree(p);
    }

    // ---- 2. T tokens at once, bitwise the single-token read of each
    {
        const int TM = strata::kernels::kFusedGrMaxT;
        std::vector<Token> toks;
        for (int t = 0; t < TM; ++t) toks.push_back(make_token(rng, t % 3 != 1));   // a mix of apply / no apply
        std::vector<Out> single;
        for (int t = 0; t < TM; ++t) {
            up(toks[(size_t) t].d_R, toks[(size_t) t].R);
            strata::kernels::fused_gr_read(args(w, toks[(size_t) t], true), cs);
            check(cudaStreamSynchronize(cs), "single");
            single.push_back(read_back(toks[(size_t) t], true));
        }
        float* xn = dev<float>((size_t) TM * D);
        int mismatches = 0;
        for (int T = 1; T <= TM; ++T) {
            strata::kernels::FusedGrArgs a[strata::kernels::kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                up(toks[(size_t) t].d_R, toks[(size_t) t].R);
                // poison the outputs, so an output the kernel forgets to write cannot pass on a stale value
                check(cudaMemset(toks[(size_t) t].d_lo, 0xFF, LR * 4), "poison lo");
                check(cudaMemset(toks[(size_t) t].d_mixed, 0xFF, N * 4), "poison mixed");
                check(cudaMemset(toks[(size_t) t].d_inject, 0xFF, HC * 4), "poison inject");
                check(cudaMemset(toks[(size_t) t].d_rs, 0xFF, HC * 4), "poison rs");
                a[t] = args(w, toks[(size_t) t], true);
            }
            strata::kernels::fused_gr_read_multi(a, T, xn, cs);
            check(cudaStreamSynchronize(cs), "multi");
            int bad_t = 0;
            for (int t = 0; t < T; ++t) {
                const Out o = read_back(toks[(size_t) t], true);
                const Out& s = single[(size_t) t];
                if (!same_bits(o.rs, s.rs) || !same_bits(o.lo, s.lo) || !same_bits(o.inject, s.inject) ||
                    !same_bits(o.mixed, s.mixed) || !same_bits(o.R, s.R))
                    ++bad_t;
            }
            if (bad_t) std::printf("    T=%d: %d of %d tokens differ from fused_gr_read\n", T, bad_t, T);
            mismatches += bad_t;
        }
        std::printf("  fused_gr_read_multi, T = 1..%d: %s\n", TM,
                    mismatches ? "*** NOT bitwise the single-token read ***" : "bitwise the single-token read");
        if (mismatches) ++bad;
        cudaFree(xn);
        for (Token& t : toks)
            for (float* p : {t.d_R, t.d_bo, t.d_inj, t.d_lo, t.d_rs, t.d_inject, t.d_mixed}) cudaFree(p);
    }
    cudaFree(w.d_norm); cudaFree(w.d_down); cudaFree(w.d_up); cudaFree(w.d_inject);
    check(cudaStreamDestroy(cs), "stream destroy");

    std::printf("\nfused_gr: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("fused_gr_parity OK\n");
    return 0;
}
