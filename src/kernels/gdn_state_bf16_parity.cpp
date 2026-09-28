// src/kernels/gdn_state_bf16_parity.cpp - issue #53 (S5): the GDN recurrent state in BF16, against FP32.
//
// `--gdn-state-bf16` changes the state's STORAGE and nothing else: the step kernels are the FP32 ones templated on
// the storage type (`fused_gdn.hpp`), so a load widens exactly and a store rounds to nearest even.  Two kinds of
// check:
//
//   1. EXACT, by construction.  From a state that BF16 represents exactly, the BF16 step's output equals the FP32
//      step's bit for bit, and the state it stores is the FP32 step's state rounded.  The same for the verify half
//      of the window kernel (outputs; the state is not written) and for its commit half (n_keep = 0..4), for the
//      switch that routes the `float*` entry points to the BF16 form, for a window of one token against the
//      single-token step, and for the widen / narrow pair the prompt path uses (all 65,536 BF16 patterns; ties,
//      denormals, infinities, NaN).
//   2. DRIFT, which the issue asks for before the option is used: the two forms run side by side on the same
//      synthetic inputs for 4,096 tokens, token by token (the path of a request's first token, and of `--spec`
//      off) and in windows of 4 with a random acceptance (verify + commit, the production path).  The relative L2
//      error of the output and of the state is printed at 1, 16, 64, 256, 1024 and 4096 tokens, and bounded
//      (`--tokens N` runs longer).
//
// `--cpu-drift` runs the drift on an emulation of the same arithmetic on the CPU (8 value heads, no GPU needed).
// The bounds below were set from it, with margin; the GPU numbers should land in the same range.
//
// The inputs imitate the real ones: q and k unit L2 per head, gate = log(decay) with the heads' decays spread
// log-uniformly from 0.37 to 0.9999 per token (the slow heads are where rounding accumulates), beta in (0, 1).
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/verify_kernels.hpp"

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

constexpr int S = 128;
constexpr float EPS = 1e-6f;
constexpr int WINDOW = 4;
int g_tokens = 4096;   // --tokens N

/// 1, 16, 64, 256, ... up to the run's length, and the length itself.
std::vector<int> checkpoints() {
    std::vector<int> c{1};
    for (int n = 16; n < g_tokens; n *= 4) c.push_back(n);
    if (c.back() != g_tokens) c.push_back(g_tokens);
    return c;
}

// The drift bounds (relative L2): the output's on the worst token so far, the state's at the checkpoint.
// `--cpu-drift` (8 heads) measured, at 4096 tokens, output 1.7e-2 and state 1.9e-2 rounded at every token, 1.2e-2
// and 1.2e-2 rounded at every commit, both flat from ~1024 tokens on (the decay forgets old rounding); the bounds
// are ~2.5x that.  A bound this loose still catches a wrong layout, a missed store or a rounding toward zero
// (biased, it grows with the slow heads' memory instead of levelling off).
constexpr double MAX_Y_STEP = 0.05, MAX_STATE_STEP = 0.05;       // rounded at every token
constexpr double MAX_Y_WINDOW = 0.05, MAX_STATE_WINDOW = 0.05;   // rounded at every commit

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float float_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

/// The kernels' rounding, on the host: nearest even, a NaN stays a NaN.
uint16_t bf16_rne(float f) {
    const uint32_t u = bits_of(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x0040u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf16_widen(uint16_t b) { return float_of((uint32_t) b << 16); }

/// An independent reading of "nearest, ties to even" for a finite f: the two BF16 values around it, compared in
/// double.  Used to check `bf16_rne` itself before it judges the device.
uint16_t bf16_rne_ref(float f) {
    const uint32_t u = bits_of(f);
    const uint16_t lo = (uint16_t) (u >> 16);          // toward zero
    if ((u & 0xffffu) == 0) return lo;
    const uint16_t hi = (uint16_t) (lo + 1);           // away from zero (carries into the exponent, or to inf)
    // |hi| as a number: one BF16 ulp above |lo|, which past the largest finite value is 2^128 (IEEE overflow)
    const int e = (lo >> 7) & 0xff;
    const double a = std::fabs((double) f), l = std::fabs((double) bf16_widen(lo));
    const double h = l + std::ldexp(1.0, (e == 0 ? 1 : e) - 127 - 7);
    if (a - l < h - a) return lo;
    if (h - a < a - l) return hi;
    return (lo & 1u) ? hi : lo;
}

struct Geo {
    int h_k, h_v;
    int qk() const { return S * h_k; }
    int value_dim() const { return S * h_v; }
    int conv_channels() const { return 2 * S * h_k + S * h_v; }
    size_t state() const { return (size_t) S * h_v * S; }
};

/// One token's inputs, packed as the window kernel reads them: h = [q | k | v], then gate, beta, z.
struct Tok {
    std::vector<float> h, gate, beta, z;
};

struct Source {
    Geo geo;
    std::mt19937 rng;
    std::normal_distribution<float> gauss{0.0f, 1.0f};
    std::vector<float> base_gate, gamma;

    Source(Geo g, uint32_t seed) : geo(g), rng(seed) {
        base_gate.resize((size_t) g.h_v);
        for (int h = 0; h < g.h_v; ++h)   // -gate from 1e-4 to 1: decays 0.9999 .. 0.37
            base_gate[(size_t) h] = -std::pow(10.0f, -4.0f + 4.0f * (float) h / (float) std::max(1, g.h_v - 1));
        gamma.resize(S);
        for (auto& x : gamma) x = 1.0f + 0.1f * gauss(rng);
    }

    void unit_heads(float* p, int heads) {
        for (int h = 0; h < heads; ++h) {
            double ss = 0;
            for (int i = 0; i < S; ++i) { p[h * S + i] = gauss(rng); ss += (double) p[h * S + i] * p[h * S + i]; }
            const float inv = (float) (1.0 / std::sqrt(ss));
            for (int i = 0; i < S; ++i) p[h * S + i] *= inv;
        }
    }

    Tok next() {
        Tok t;
        t.h.resize((size_t) geo.conv_channels());
        unit_heads(t.h.data(), geo.h_k);                       // q
        unit_heads(t.h.data() + geo.qk(), geo.h_k);            // k
        for (int i = 0; i < geo.value_dim(); ++i) t.h[(size_t) (2 * geo.qk() + i)] = gauss(rng);   // v
        t.gate.resize((size_t) geo.h_v);
        t.beta.resize((size_t) geo.h_v);
        for (int h = 0; h < geo.h_v; ++h) {
            t.gate[(size_t) h] = base_gate[(size_t) h] * std::exp(0.25f * gauss(rng));
            t.beta[(size_t) h] = 1.0f / (1.0f + std::exp(-gauss(rng)));
        }
        t.z.resize((size_t) geo.value_dim());
        for (auto& x : t.z) x = gauss(rng);
        return t;
    }
};

double rel_l2(const float* want, const float* got, size_t n) {
    double d = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double) got[i] - (double) want[i];
        d += e * e;
        m += (double) want[i] * want[i];
    }
    return std::sqrt(d / (m > 1e-300 ? m : 1e-300));
}

/// The state's relative L2 error over one head ([row][head][col]); head 0 has the slowest decay, the last the fastest.
double head_err(const std::vector<float>& want, const std::vector<float>& got, const Geo& geo, int head) {
    double d = 0, m = 0;
    for (int r = 0; r < S; ++r)
        for (int c = 0; c < S; ++c) {
            const size_t i = ((size_t) r * geo.h_v + head) * S + c;
            const double e = (double) got[i] - (double) want[i];
            d += e * e;
            m += (double) want[i] * want[i];
        }
    return std::sqrt(d / (m > 1e-300 ? m : 1e-300));
}

void report_heads(const std::vector<float>& want, const std::vector<float>& got, const Geo& geo) {
    std::printf("           state by head: slowest decay %.3e, fastest %.3e\n", head_err(want, got, geo, 0),
                head_err(want, got, geo, geo.h_v - 1));
}

bool all_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

// ================================ the CPU emulation ================================

/// Tokens [0, n) of the step on a state held in FP32 `st`, in the kernels' order: per column, four row groups of
/// 32 rows accumulate s.k, their sum gives delta, the rows are updated and s.q accumulates the same way; then the
/// head's RMS norm, gamma and sigmoid(z).  With `round`, the state is rounded to BF16 after the n tokens (after
/// every token, called with n = 1).  Plain multiply-adds: the emulation is for magnitudes, not for bits.
void cpu_tokens(std::vector<float>& st, const std::vector<Tok>& toks, int n, const std::vector<float>& gamma,
                const Geo& geo, bool round, std::vector<std::vector<float>>& ys) {
    const int h_v = geo.h_v, h_k = geo.h_k, qk = geo.qk();
    ys.assign((size_t) n, std::vector<float>((size_t) geo.value_dim()));
    float kv[4][S], o[4][S], delta[S], oc[S];
    for (int t = 0; t < n; ++t) {
        const Tok& tk = toks[(size_t) t];
        for (int head = 0; head < h_v; ++head) {
            const int qh = head % h_k;
            const float* q = tk.h.data() + qh * S;
            const float* k = tk.h.data() + qk + qh * S;
            const float* v = tk.h.data() + 2 * qk + head * S;
            const float g = std::exp(tk.gate[(size_t) head]);
            for (int rg = 0; rg < 4; ++rg) {
                for (int c = 0; c < S; ++c) kv[rg][c] = 0.0f;
                for (int r = 0; r < 32; ++r) {
                    const int row = rg * 32 + r;
                    const float* sr = st.data() + ((size_t) row * h_v + head) * S;
                    for (int c = 0; c < S; ++c) kv[rg][c] = sr[c] * k[row] + kv[rg][c];
                }
            }
            for (int c = 0; c < S; ++c)
                delta[c] = (v[c] - g * (kv[0][c] + kv[1][c] + kv[2][c] + kv[3][c])) * tk.beta[(size_t) head];
            for (int rg = 0; rg < 4; ++rg) {
                for (int c = 0; c < S; ++c) o[rg][c] = 0.0f;
                for (int r = 0; r < 32; ++r) {
                    const int row = rg * 32 + r;
                    float* sr = st.data() + ((size_t) row * h_v + head) * S;
                    for (int c = 0; c < S; ++c) {
                        sr[c] = g * sr[c] + k[row] * delta[c];
                        o[rg][c] = sr[c] * q[row] + o[rg][c];
                    }
                }
            }
            double ss = 0;
            for (int c = 0; c < S; ++c) {
                oc[c] = (o[0][c] + o[1][c] + o[2][c] + o[3][c]) / std::sqrt((float) S);
                ss += (double) oc[c] * oc[c];
            }
            const float scale = 1.0f / std::sqrt((float) (ss / S) + EPS);
            for (int c = 0; c < S; ++c) {
                const float zz = tk.z[(size_t) (head * S + c)];
                ys[(size_t) t][(size_t) (head * S + c)] = oc[c] * scale * gamma[(size_t) c] / (1.0f + std::exp(-zz));
            }
        }
    }
    if (round)
        for (auto& x : st) x = bf16_widen(bf16_rne(x));
}

struct Drift {
    double y_worst = 0;    // worst token's output error so far
    bool finite = true;
};

/// Prints one checkpoint row and judges it.
bool report(const char* what, int tok, const Drift& d, double state_err, double max_y, double max_state) {
    const bool ok = d.finite && d.y_worst <= max_y && state_err <= max_state;
    std::printf("  %-8s %5d tokens: y worst %.3e   state %.3e   %s\n", what, tok, d.y_worst, state_err,
                ok ? "ok" : "*** OVER ***");
    return ok;
}

int cpu_drift() {
    const Geo geo{2, 8};
    int bad = 0;
    std::printf("CPU emulation, %d value heads, decays 0.37 .. 0.9999\n", geo.h_v);
    const std::vector<int> cps = checkpoints();
    const size_t ncp = cps.size();
    for (int mode = 0; mode < 2; ++mode) {
        const bool window = mode == 1;
        Source src(geo, 53);
        std::mt19937 accept(7);
        std::vector<float> s32(geo.state(), 0.0f), s16(geo.state(), 0.0f);
        Drift d;
        int done = 0;
        size_t next_cp = 0;
        while (done < g_tokens) {
            const int T = window ? WINDOW : 1;
            std::vector<Tok> toks;
            for (int t = 0; t < T; ++t) toks.push_back(src.next());
            const int n = window ? 1 + (int) (accept() % WINDOW) : 1;
            std::vector<std::vector<float>> y32, y16;
            cpu_tokens(s32, toks, n, src.gamma, geo, false, y32);
            cpu_tokens(s16, toks, n, src.gamma, geo, true, y16);
            for (int t = 0; t < n; ++t) {
                const std::vector<float>& a = y32[(size_t) t];
                const std::vector<float>& b = y16[(size_t) t];
                d.y_worst = std::max(d.y_worst, rel_l2(a.data(), b.data(), a.size()));
                d.finite = d.finite && all_finite(b.data(), b.size());
            }
            done += n;
            // a window that crosses a checkpoint reports the state after its commit, as the GPU run does
            while (next_cp < ncp && done >= cps[next_cp]) {
                const double se = rel_l2(s32.data(), s16.data(), s32.size());
                if (!report(window ? "window" : "step", cps[next_cp], d, se,
                            window ? MAX_Y_WINDOW : MAX_Y_STEP, window ? MAX_STATE_WINDOW : MAX_STATE_STEP))
                    ++bad;
                if (++next_cp == ncp) report_heads(s32, s16, geo);
            }
        }
    }
    return bad;
}

// ================================ the rounding itself ================================

int rounding_host() {
    int bad = 0;
    std::mt19937 rng(11);
    int64_t n = 0;
    for (int i = 0; i < (1 << 20); ++i) {
        const float f = float_of((uint32_t) rng());
        if (std::isnan(f)) continue;
        ++n;
        if (bf16_rne(f) != bf16_rne_ref(f)) {
            if (bad < 5) std::printf("  host rne %08x -> %04x, want %04x\n", bits_of(f), bf16_rne(f), bf16_rne_ref(f));
            ++bad;
        }
    }
    // the exact ties, both parities, and the carry into the exponent
    const uint32_t ties[] = {0x3f808000u, 0x3f818000u, 0x3f7f8000u, 0x7f7f8000u, 0x00008000u, 0x00018000u};
    for (uint32_t u : ties)
        if (bf16_rne(float_of(u)) != bf16_rne_ref(float_of(u))) ++bad;
    if (bf16_rne(float_of(0x7f800001u)) != 0x7fc0u) ++bad;   // a NaN whose payload is in the low half only
    std::printf("  %-50s %s (%lld values)\n", "host rounding vs nearest-even in double", bad ? "*** FAIL ***" : "ok",
                (long long) n);
    return bad;
}

// ================================ the GPU checks ================================

struct Dev {
    Geo geo;
    cudaStream_t cs = nullptr;
    float *h = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *gamma = nullptr, *y = nullptr, *y2 = nullptr;
    int32_t* n_keep = nullptr;

    explicit Dev(Geo g) : geo(g) {
        check(cudaStreamCreate(&cs), "stream");
        check(cudaMalloc(&h, (size_t) WINDOW * g.conv_channels() * 4), "h");
        check(cudaMalloc(&gate, (size_t) WINDOW * g.h_v * 4), "gate");
        check(cudaMalloc(&beta, (size_t) WINDOW * g.h_v * 4), "beta");
        check(cudaMalloc(&z, (size_t) WINDOW * g.value_dim() * 4), "z");
        check(cudaMalloc(&gamma, (size_t) S * 4), "gamma");
        check(cudaMalloc(&y, (size_t) WINDOW * g.value_dim() * 4), "y");
        check(cudaMalloc(&y2, (size_t) WINDOW * g.value_dim() * 4), "y2");
        check(cudaMalloc(&n_keep, 4), "n_keep");
    }
    ~Dev() {
        cudaFree(h); cudaFree(gate); cudaFree(beta); cudaFree(z); cudaFree(gamma); cudaFree(y); cudaFree(y2);
        cudaFree(n_keep);
        cudaStreamDestroy(cs);
    }

    void upload(const std::vector<Tok>& toks, const std::vector<float>& gam) {
        const Geo& g = geo;
        for (size_t t = 0; t < toks.size(); ++t) {
            check(cudaMemcpy(h + t * g.conv_channels(), toks[t].h.data(), (size_t) g.conv_channels() * 4,
                             cudaMemcpyHostToDevice), "up h");
            check(cudaMemcpy(gate + t * g.h_v, toks[t].gate.data(), (size_t) g.h_v * 4, cudaMemcpyHostToDevice), "up g");
            check(cudaMemcpy(beta + t * g.h_v, toks[t].beta.data(), (size_t) g.h_v * 4, cudaMemcpyHostToDevice), "up b");
            check(cudaMemcpy(z + t * g.value_dim(), toks[t].z.data(), (size_t) g.value_dim() * 4, cudaMemcpyHostToDevice),
                  "up z");
        }
        check(cudaMemcpy(gamma, gam.data(), (size_t) S * 4, cudaMemcpyHostToDevice), "up gamma");
    }
    void set_keep(int n) { check(cudaMemcpy(n_keep, &n, 4, cudaMemcpyHostToDevice), "n_keep"); }

    // the single-token step on token 0 of the upload
    void step32(float* st, float* out) {
        strata::kernels::fused_gdn_step_norm(st, h, h + geo.qk(), h + 2 * geo.qk(), gate, beta, z, gamma, EPS, out,
                                             geo.h_k, geo.h_v, cs);
    }
    void step16(uint16_t* st, float* out) {
        strata::kernels::fused_gdn_step_norm_bf16(st, h, h + geo.qk(), h + 2 * geo.qk(), gate, beta, z, gamma, EPS,
                                                  out, geo.h_k, geo.h_v, cs);
    }
    // the window kernel; `keep` = nullptr for the verify half
    void multi32(float* st, int T, const int32_t* keep, float* out) {
        strata::kernels::gdn_step_norm_multi(st, h, geo.conv_channels(), gate, beta, z, gamma, EPS, out, geo.h_k,
                                             geo.h_v, T, keep, cs);
    }
    void multi16(uint16_t* st, int T, const int32_t* keep, float* out) {
        strata::kernels::gdn_step_norm_multi_bf16(st, h, geo.conv_channels(), gate, beta, z, gamma, EPS, out, geo.h_k,
                                                  geo.h_v, T, keep, cs);
    }
    void sync() { check(cudaStreamSynchronize(cs), "sync"); }
};

template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return v;
}
template <typename T>
void up(T* d, const std::vector<T>& v) {
    check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
}

std::vector<uint16_t> round_all(const std::vector<float>& f) {
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); ++i) b[i] = bf16_rne(f[i]);
    return b;
}
std::vector<float> widen_all(const std::vector<uint16_t>& b) {
    std::vector<float> f(b.size());
    for (size_t i = 0; i < b.size(); ++i) f[i] = bf16_widen(b[i]);
    return f;
}

/// Bit for bit (a -0 is not a +0, a NaN equals the same NaN).
template <typename T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

int line(const char* what, bool ok) {
    std::printf("  %-50s %s\n", what, ok ? "ok" : "*** FAIL ***");
    return ok ? 0 : 1;
}

int gpu_rounding() {
    int bad = 0;
    // widen: every BF16 pattern
    std::vector<uint16_t> all(65536);
    for (int i = 0; i < 65536; ++i) all[(size_t) i] = (uint16_t) i;
    uint16_t* d16 = nullptr;
    float* d32 = nullptr;
    const size_t n = (size_t) 1 << 20;
    check(cudaMalloc(&d16, n * 2), "d16");
    check(cudaMalloc(&d32, n * 4), "d32");
    up(d16, all);
    strata::kernels::gdn_state_widen(d16, d32, 65536, nullptr);
    check(cudaDeviceSynchronize(), "widen");
    const std::vector<float> w = down(d32, 65536);
    bool ok = true;
    for (int i = 0; i < 65536; ++i) ok = ok && bits_of(w[(size_t) i]) == ((uint32_t) i << 16);
    bad += line("widen: all 65,536 patterns, bit for bit", ok);

    // narrow: special values, then random bit patterns (NaNs included)
    std::vector<float> f;
    const uint32_t special[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x3f808000u, 0x3f818000u,
                                0x3f7f8000u, 0x7f7fffffu, 0x7f7f8000u, 0x00000001u, 0x00008000u, 0x00018000u,
                                0x807fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xff800001u,
                                0x7fffffffu};
    for (uint32_t u : special) f.push_back(float_of(u));
    std::mt19937 rng(5);
    while (f.size() < n) f.push_back(float_of((uint32_t) rng()));
    up(d32, f);
    strata::kernels::gdn_state_narrow(d32, d16, (int64_t) n, nullptr);
    check(cudaDeviceSynchronize(), "narrow");
    const std::vector<uint16_t> b = down(d16, n);
    ok = true;
    for (size_t i = 0; i < n; ++i) ok = ok && b[i] == bf16_rne(f[i]);
    bad += line("narrow: 1M values incl. ties, denormals, inf, NaN", ok);
    cudaFree(d16);
    cudaFree(d32);
    return bad;
}

int gpu_exact() {
    const Geo geo{16, 48};   // the model's geometry
    Dev dv(geo);
    Source src(geo, 101);
    int bad = 0;
    const size_t NS = geo.state(), NY = (size_t) geo.value_dim();
    float* s32 = nullptr;
    uint16_t *s16 = nullptr, *s16b = nullptr;
    check(cudaMalloc(&s32, NS * 4), "s32");
    check(cudaMalloc(&s16, NS * 2), "s16");
    check(cudaMalloc(&s16b, NS * 4), "s16b");   // FP32-sized, as a session slice is: the switch reads its first half

    // a start the BF16 form represents exactly
    std::mt19937 rng(3);
    std::normal_distribution<float> gauss(0.0f, 0.3f);
    std::vector<float> st0(NS);
    for (auto& x : st0) x = gauss(rng);
    std::vector<uint16_t> b0 = round_all(st0);
    const std::vector<float> f0 = widen_all(b0);

    // ---- 1. the single-token step, 8 tokens, each from the other form's rounded state
    {
        bool y_same = true, st_same = true, finite = true;
        std::vector<uint16_t> b = b0;
        for (int t = 0; t < 8; ++t) {
            dv.upload({src.next()}, src.gamma);
            up(s32, widen_all(b));
            up(s16, b);
            dv.step32(s32, dv.y);
            dv.step16(s16, dv.y2);
            dv.sync();
            const std::vector<float> ya = down(dv.y, NY), yb = down(dv.y2, NY);
            const std::vector<float> sa = down(s32, NS);
            b = down(s16, NS);
            y_same = y_same && same(ya, yb);
            st_same = st_same && same(round_all(sa), b);
            finite = finite && all_finite(ya.data(), NY);
        }
        bad += line("step: output bit for bit (8 tokens)", y_same && finite);
        bad += line("step: stored state = FP32 state rounded", st_same);
    }
    // ---- 2. the switch: `fused_gdn_step_norm(float*)` takes the BF16 path, reading the slice's first half
    {
        dv.upload({src.next()}, src.gamma);
        up(s16, b0);
        check(cudaMemcpy(s16b, b0.data(), NS * 2, cudaMemcpyHostToDevice), "s16b");
        dv.step16(s16, dv.y);
        strata::kernels::gdn_state_bf16_set_enabled(true);
        dv.step32(reinterpret_cast<float*>(s16b), dv.y2);
        strata::kernels::gdn_state_bf16_set_enabled(false);
        dv.sync();
        const bool ok = same(down(dv.y, NY), down(dv.y2, NY)) && same(down(s16, NS), down(s16b, NS));
        bad += line("switch on: fused_gdn_step_norm(float*) is the BF16 step", ok);
    }
    // ---- 3. the window kernel: verify (no write), commit n_keep = 0..4, the switch, one token vs the step
    {
        std::vector<Tok> win;
        for (int t = 0; t < WINDOW; ++t) win.push_back(src.next());
        dv.upload(win, src.gamma);
        up(s32, f0);
        up(s16, b0);
        dv.multi32(s32, WINDOW, nullptr, dv.y);
        dv.multi16(s16, WINDOW, nullptr, dv.y2);
        dv.sync();
        const bool y_ok = same(down(dv.y, NY * WINDOW), down(dv.y2, NY * WINDOW));
        const bool untouched = same(down(s32, NS), f0) && same(down(s16, NS), b0);
        bad += line("verify: 4 outputs bit for bit", y_ok);
        bad += line("verify: neither state written", untouched);
        bool commit_ok = true;
        for (int keep = 0; keep <= WINDOW; ++keep) {
            up(s32, f0);
            up(s16, b0);
            dv.set_keep(keep);
            dv.multi32(s32, WINDOW, dv.n_keep, dv.y);
            dv.multi16(s16, WINDOW, dv.n_keep, dv.y2);
            dv.sync();
            const std::vector<uint16_t> got = down(s16, NS);
            commit_ok = commit_ok && same(got, round_all(down(s32, NS)));
            if (keep == 0) commit_ok = commit_ok && same(got, b0);
            if (keep > 0)
                commit_ok = commit_ok && same(down(dv.y, NY * (size_t) keep), down(dv.y2, NY * (size_t) keep));
        }
        bad += line("commit n_keep 0..4: state = FP32 commit rounded", commit_ok);

        up(s16, b0);
        check(cudaMemcpy(s16b, b0.data(), NS * 2, cudaMemcpyHostToDevice), "s16b");
        dv.set_keep(3);
        dv.multi16(s16, WINDOW, dv.n_keep, dv.y);
        strata::kernels::gdn_state_bf16_set_enabled(true);
        dv.multi32(reinterpret_cast<float*>(s16b), WINDOW, dv.n_keep, dv.y2);
        strata::kernels::gdn_state_bf16_set_enabled(false);
        dv.sync();
        bad += line("switch on: gdn_step_norm_multi(float*) is the BF16 form",
                    same(down(dv.y, NY * 3), down(dv.y2, NY * 3)) && same(down(s16, NS), down(s16b, NS)));

        // a window of one committed token is the single-token step, in both storages
        dv.upload({win[0]}, src.gamma);
        dv.set_keep(1);
        up(s32, f0);
        dv.multi32(s32, 1, dv.n_keep, dv.y);
        dv.sync();
        const std::vector<float> ya = down(dv.y, NY), sa = down(s32, NS);
        up(s32, f0);
        dv.step32(s32, dv.y2);
        dv.sync();
        bad += line("FP32: window of 1 = the step (output and state)",
                    same(ya, down(dv.y2, NY)) && same(sa, down(s32, NS)));
        up(s16, b0);
        dv.multi16(s16, 1, dv.n_keep, dv.y);
        dv.sync();
        const std::vector<float> yb = down(dv.y, NY);
        const std::vector<uint16_t> sb = down(s16, NS);
        up(s16, b0);
        dv.step16(s16, dv.y2);
        dv.sync();
        bad += line("BF16: window of 1 = the step (output and state)",
                    same(yb, down(dv.y2, NY)) && same(sb, down(s16, NS)));
    }
    cudaFree(s32);
    cudaFree(s16);
    cudaFree(s16b);
    return bad;
}

int gpu_drift() {
    const Geo geo{16, 48};
    Dev dv(geo);
    int bad = 0;
    const size_t NS = geo.state(), NY = (size_t) geo.value_dim();
    float* s32 = nullptr;
    uint16_t* s16 = nullptr;
    check(cudaMalloc(&s32, NS * 4), "s32");
    check(cudaMalloc(&s16, NS * 2), "s16");
    std::printf("GPU, %d value heads, decays 0.37 .. 0.9999\n", geo.h_v);
    const std::vector<int> cps = checkpoints();
    const size_t ncp = cps.size();
    for (int mode = 0; mode < 2; ++mode) {
        const bool window = mode == 1;
        Source src(geo, 53);
        std::mt19937 accept(7);
        check(cudaMemset(s32, 0, NS * 4), "z32");
        check(cudaMemset(s16, 0, NS * 2), "z16");
        Drift d;
        int done = 0;
        size_t next_cp = 0;
        while (done < g_tokens) {
            const int T = window ? WINDOW : 1;
            std::vector<Tok> toks;
            for (int t = 0; t < T; ++t) toks.push_back(src.next());
            dv.upload(toks, src.gamma);
            int n = 1;
            if (window) {
                n = 1 + (int) (accept() % WINDOW);
                dv.set_keep(n);
                dv.multi32(s32, T, dv.n_keep, dv.y);   // the commit's outputs are the verify's (checked above)
                dv.multi16(s16, T, dv.n_keep, dv.y2);
            } else {
                dv.step32(s32, dv.y);
                dv.step16(s16, dv.y2);
            }
            dv.sync();
            const std::vector<float> ya = down(dv.y, NY * (size_t) n), yb = down(dv.y2, NY * (size_t) n);
            for (int t = 0; t < n; ++t) {
                d.y_worst = std::max(d.y_worst, rel_l2(ya.data() + (size_t) t * NY, yb.data() + (size_t) t * NY, NY));
                d.finite = d.finite && all_finite(yb.data() + (size_t) t * NY, NY);
            }
            done += n;
            while (next_cp < ncp && done >= cps[next_cp]) {
                const std::vector<float> want = down(s32, NS), got = widen_all(down(s16, NS));
                const double se = rel_l2(want.data(), got.data(), NS);
                if (!report(window ? "window" : "step", cps[next_cp], d, se, window ? MAX_Y_WINDOW : MAX_Y_STEP,
                            window ? MAX_STATE_WINDOW : MAX_STATE_STEP))
                    ++bad;
                if (++next_cp == ncp) report_heads(want, got, geo);
            }
        }
    }
    cudaFree(s32);
    cudaFree(s16);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--cpu-drift") cpu = true;
        else if (a == "--tokens" && i + 1 < argc && std::atoi(argv[i + 1]) >= 1) g_tokens = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: gdn_state_bf16_parity [--selftest | --cpu-drift] [--tokens N (default 4096)]\n");
            return 2;
        }
    }
    int bad = rounding_host();
    if (cpu) {
        bad += cpu_drift();
    } else {
        if (strata::kernels::gdn_state_bf16_enabled()) { std::printf("the switch starts on\n"); return 1; }
        bad += gpu_rounding();
        bad += gpu_exact();
        bad += gpu_drift();
    }
    std::printf("\ngdn_state_bf16: %d failures\n", bad);
    if (bad) return 1;
    if (selftest || cpu) std::printf("gdn_state_bf16_parity OK\n");
    return 0;
}
