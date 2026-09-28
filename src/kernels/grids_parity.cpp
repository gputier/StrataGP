// src/kernels/grids_parity.cpp - issue #26 (O7): the decode launches resized for the device, against the 0.1.20 ones.
//
// Every change in the issue is claimed BITWISE, so the new-vs-old checks here compare bytes, not a tolerance:
//
//   1. fused_gdn_ab: a block of four warps per row against a warp per row (STRATA_OLD_GDN_AB), several widths;
//   2. gdn_ab_multi: the same for T = 1..8 tokens;
//   3. fused_gdn_step_norm: a cluster of 4 blocks per head against a block per head (STRATA_OLD_GDN_STEP),
//      the state it writes and y;
//   4. gdn_step_norm_multi: the same, as verify (no n_keep; t_out_begin 0, 1, T-1) and as commit (n_keep 0, 1, T);
//   5. fetch_blobs (from mapped host memory) and gather_rows: the device-sized grid and 48 * 8 (STRATA_OLD_GRIDS),
//      both against a host copy.
//
// "New equals old" would also pass with both wrong, so 1-4 are ALSO held against a double-precision host
// reference (a tolerance: the kernels use __expf and fmaf chains).  On a device without clusters (sm_80/86/89) the
// step checks compare the old kernel with itself; the output says which path ran, so read the "cluster" line.
//
// --bench: old vs new time per launch of each kernel (cudaEvent over many launches).
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/launch_grid.hpp"
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

namespace sk = strata::kernels;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* dev_alloc(size_t n) {
    T* p = nullptr;
    check(cudaMalloc((void**) &p, n * sizeof(T) + 16), "cudaMalloc");
    return p;
}
template <typename T>
T* dev_copy(const std::vector<T>& h) {
    T* p = dev_alloc<T>(h.size());
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    return p;
}
template <typename T>
std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    return h;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
double from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return (double) f;
}

double rel_l1(const std::vector<float>& got, const std::vector<double>& want) {
    double d = 0, m = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        d += std::fabs((double) got[i] - want[i]);
        m += std::fabs(want[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

int g_bad = 0;

void report(bool ok, const char* fmt, const std::string& what, const std::string& detail = "") {
    std::printf(fmt, what.c_str(), ok ? "ok" : "*** FAIL ***", detail.c_str());
    if (!ok) ++g_bad;
}

bool same_bytes(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

// ------------------------------------------------------------------ 1-2. the alpha/beta projection
struct AbCase {
    int n = 0, h_v = 0;
    std::vector<float> x;                 // (T, n)
    std::vector<uint16_t> wa, wb;         // (h_v, n) bf16
    std::vector<float> dt, ssm_a;         // (h_v)
};

AbCase make_ab(int n, int h_v, int T, unsigned seed) {
    AbCase c;
    c.n = n;
    c.h_v = h_v;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    c.x.resize((size_t) T * n);
    for (float& v : c.x) v = g(rng);
    c.wa.resize((size_t) h_v * n);
    c.wb.resize((size_t) h_v * n);
    // Row 0 of alpha is large and positive-leaning, so softplus's x > 20 branch is taken somewhere.
    for (int r = 0; r < h_v; ++r)
        for (int i = 0; i < n; ++i) {
            const float big = r == 0 ? 0.5f * c.x[(size_t) i] : 0.0f;
            c.wa[(size_t) r * n + i] = to_bf16(g(rng) * 0.05f + big);
            c.wb[(size_t) r * n + i] = to_bf16(g(rng) * 0.05f);
        }
    c.dt.resize((size_t) h_v);
    c.ssm_a.resize((size_t) h_v);
    for (int r = 0; r < h_v; ++r) {
        c.dt[(size_t) r] = g(rng) * 0.5f;
        c.ssm_a[(size_t) r] = -(std::fabs(g(rng)) + 0.1f);
    }
    return c;
}

void ab_reference(const AbCase& c, int t, std::vector<double>& gate, std::vector<double>& beta) {
    gate.assign((size_t) c.h_v, 0.0);
    beta.assign((size_t) c.h_v, 0.0);
    const float* x = c.x.data() + (size_t) t * c.n;
    for (int r = 0; r < c.h_v; ++r) {
        double a = 0, b = 0;
        for (int i = 0; i < c.n; ++i) {
            a += from_bf16(c.wa[(size_t) r * c.n + i]) * (double) x[i];
            b += from_bf16(c.wb[(size_t) r * c.n + i]) * (double) x[i];
        }
        beta[(size_t) r] = 1.0 / (1.0 + std::exp(-b));
        const double v = a + (double) c.dt[(size_t) r];
        gate[(size_t) r] = (v > 20.0 ? v : std::log1p(std::exp(v))) * (double) c.ssm_a[(size_t) r];
    }
}

void test_ab(int n, int h_v) {
    const AbCase c = make_ab(n, h_v, 1, 100u + (unsigned) n + (unsigned) h_v);
    float *dx = dev_copy(c.x), *ddt = dev_copy(c.dt), *dsa = dev_copy(c.ssm_a);
    uint16_t *dwa = dev_copy(c.wa), *dwb = dev_copy(c.wb);
    float *dg = dev_alloc<float>((size_t) h_v), *db = dev_alloc<float>((size_t) h_v);
    std::vector<float> g_old, b_old, g_new, b_new;
    for (int pass = 0; pass < 2; ++pass) {
        sk::old_gdn_ab().set(pass == 0);
        check(cudaMemset(dg, 0xff, (size_t) h_v * 4), "memset");
        check(cudaMemset(db, 0xff, (size_t) h_v * 4), "memset");
        sk::fused_gdn_ab(dx, dwa, dwb, ddt, dsa, dg, db, n, h_v, nullptr);
        check(cudaDeviceSynchronize(), "fused_gdn_ab");
        (pass == 0 ? g_old : g_new) = host_copy(dg, (size_t) h_v);
        (pass == 0 ? b_old : b_new) = host_copy(db, (size_t) h_v);
    }
    sk::old_gdn_ab().set(false);
    const bool same = same_bytes(g_old.data(), g_new.data(), (size_t) h_v * 4) &&
                      same_bytes(b_old.data(), b_new.data(), (size_t) h_v * 4);
    std::vector<double> gr, br;
    ab_reference(c, 0, gr, br);
    const double rg = rel_l1(g_new, gr), rb = rel_l1(b_new, br);
    char name[96], det[96];
    std::snprintf(name, sizeof name, "fused_gdn_ab n=%d h_v=%d: new == old", n, h_v);
    report(same, "  %-58s %s%s\n", name);
    std::snprintf(name, sizeof name, "fused_gdn_ab n=%d h_v=%d: vs double reference", n, h_v);
    std::snprintf(det, sizeof det, " (gate %.2e, beta %.2e)", rg, rb);
    report(rg <= 1e-4 && rb <= 1e-4, "  %-58s %s%s\n", name, det);
    cudaFree(dx); cudaFree(ddt); cudaFree(dsa); cudaFree(dwa); cudaFree(dwb); cudaFree(dg); cudaFree(db);
}

void test_ab_multi(int n, int h_v) {
    const int MT = sk::kVerifyMaxT;
    const AbCase c = make_ab(n, h_v, MT, 200u + (unsigned) n + (unsigned) h_v);
    float *dx = dev_copy(c.x), *ddt = dev_copy(c.dt), *dsa = dev_copy(c.ssm_a);
    uint16_t *dwa = dev_copy(c.wa), *dwb = dev_copy(c.wb);
    float *dg = dev_alloc<float>((size_t) MT * h_v), *db = dev_alloc<float>((size_t) MT * h_v);
    int differ = 0;
    double worst = 0;
    for (int T = 1; T <= MT; ++T) {
        std::vector<float> g[2], b[2];
        for (int pass = 0; pass < 2; ++pass) {
            sk::old_gdn_ab().set(pass == 0);
            check(cudaMemset(dg, 0xff, (size_t) MT * h_v * 4), "memset");
            check(cudaMemset(db, 0xff, (size_t) MT * h_v * 4), "memset");
            sk::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, dg, db, n, h_v, T, nullptr);
            check(cudaDeviceSynchronize(), "gdn_ab_multi");
            g[pass] = host_copy(dg, (size_t) MT * h_v);
            b[pass] = host_copy(db, (size_t) MT * h_v);
        }
        if (!same_bytes(g[0].data(), g[1].data(), g[0].size() * 4) || !same_bytes(b[0].data(), b[1].data(), b[0].size() * 4))
            ++differ;
        for (int t = 0; t < T; ++t) {
            std::vector<double> gr, br;
            ab_reference(c, t, gr, br);
            const std::vector<float> gt(g[1].begin() + (size_t) t * h_v, g[1].begin() + (size_t) (t + 1) * h_v);
            const std::vector<float> bt(b[1].begin() + (size_t) t * h_v, b[1].begin() + (size_t) (t + 1) * h_v);
            worst = std::max(worst, std::max(rel_l1(gt, gr), rel_l1(bt, br)));
        }
    }
    sk::old_gdn_ab().set(false);
    char name[96], det[96];
    std::snprintf(name, sizeof name, "gdn_ab_multi n=%d h_v=%d T=1..%d: new == old", n, h_v, MT);
    std::snprintf(det, sizeof det, " (%d of %d windows differ)", differ, MT);
    report(differ == 0, "  %-58s %s%s\n", name, det);
    std::snprintf(name, sizeof name, "gdn_ab_multi n=%d h_v=%d: vs double reference", n, h_v);
    std::snprintf(det, sizeof det, " (worst %.2e)", worst);
    report(worst <= 1e-4, "  %-58s %s%s\n", name, det);
    cudaFree(dx); cudaFree(ddt); cudaFree(dsa); cudaFree(dwa); cudaFree(dwb); cudaFree(dg); cudaFree(db);
}

// ------------------------------------------------------------------ 3-4. the GDN step and its output norm
constexpr int S = 128;
constexpr float EPS = 1e-6f;

struct StepCase {
    int h_k = 16, h_v = 48, T = 1;
    int C = 0;                                   // conv channels: q | k | v
    std::vector<float> state;                    // (S rows, h_v, S cols) - the kernels' layout
    std::vector<float> h;                        // (T, C)
    std::vector<float> gate, beta;               // (T, h_v)
    std::vector<float> z;                        // (T, h_v * S)
    std::vector<float> gamma;                    // (S)
};

StepCase make_step(int h_k, int h_v, int T, unsigned seed) {
    StepCase c;
    c.h_k = h_k;
    c.h_v = h_v;
    c.T = T;
    c.C = 2 * h_k * S + h_v * S;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    c.state.resize((size_t) S * h_v * S);
    for (float& v : c.state) v = g(rng) * 0.1f;
    c.h.resize((size_t) T * c.C);
    for (int t = 0; t < T; ++t) {
        float* ht = c.h.data() + (size_t) t * c.C;
        for (int i = 0; i < c.C; ++i) ht[i] = g(rng);
        // q and k are L2-normalized per head, as the conv kernel leaves them
        for (int head = 0; head < 2 * h_k; ++head) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += (double) ht[head * S + i] * ht[head * S + i];
            const float inv = (float) (1.0 / std::sqrt(ss + 1e-6));
            for (int i = 0; i < S; ++i) ht[head * S + i] *= inv;
        }
    }
    c.gate.resize((size_t) T * h_v);
    c.beta.resize((size_t) T * h_v);
    for (float& v : c.gate) v = -std::fabs(g(rng)) * 0.5f;         // exp(gate) < 1: the state decays
    for (float& v : c.beta) v = 1.0f / (1.0f + std::exp(-g(rng)));
    c.z.resize((size_t) T * h_v * S);
    for (float& v : c.z) v = g(rng);
    c.gamma.resize((size_t) S);
    for (float& v : c.gamma) v = 1.0f + 0.1f * g(rng);
    return c;
}

/// One token in double: the state (kernel layout) is updated in place, y (h_v * S) is written.
void step_reference(std::vector<double>& st, const StepCase& c, int t, std::vector<double>& y) {
    const int qk = S * c.h_k;
    const float* ht = c.h.data() + (size_t) t * c.C;
    y.assign((size_t) c.h_v * S, 0.0);
    auto at = [&](int r, int head, int col) -> double& { return st[((size_t) r * c.h_v + head) * S + col]; };
    for (int head = 0; head < c.h_v; ++head) {
        const int qh = head % c.h_k;
        const double g = std::exp((double) c.gate[(size_t) t * c.h_v + head]);
        const double b = (double) c.beta[(size_t) t * c.h_v + head];
        std::vector<double> o((size_t) S);
        for (int col = 0; col < S; ++col) {
            double kv = 0;
            for (int r = 0; r < S; ++r) kv += at(r, head, col) * (double) ht[qk + qh * S + r];
            const double delta = ((double) ht[2 * qk + head * S + col] - g * kv) * b;
            double dot = 0;
            for (int r = 0; r < S; ++r) {
                at(r, head, col) = g * at(r, head, col) + (double) ht[qk + qh * S + r] * delta;
                dot += at(r, head, col) * (double) ht[qh * S + r];
            }
            o[(size_t) col] = dot / std::sqrt((double) S);
        }
        double ss = 0;
        for (double v : o) ss += v * v;
        const double scale = 1.0 / std::sqrt(ss / S + (double) EPS);
        for (int col = 0; col < S; ++col) {
            const double zz = (double) c.z[(size_t) t * c.h_v * S + head * S + col];
            y[(size_t) head * S + col] = o[(size_t) col] * scale * (double) c.gamma[(size_t) col] / (1.0 + std::exp(-zz));
        }
    }
}

void test_step(int h_k, int h_v, unsigned seed) {
    const StepCase c = make_step(h_k, h_v, 1, seed);
    const int qk = S * h_k;
    const size_t ns = c.state.size(), ny = (size_t) h_v * S;
    float *d_state0 = dev_copy(c.state), *d_state = dev_alloc<float>(ns), *d_h = dev_copy(c.h);
    float *d_gate = dev_copy(c.gate), *d_beta = dev_copy(c.beta), *d_z = dev_copy(c.z), *d_gamma = dev_copy(c.gamma);
    float* d_y = dev_alloc<float>(ny);
    std::vector<float> st[2], y[2];
    for (int pass = 0; pass < 2; ++pass) {
        sk::old_gdn_step().set(pass == 0);
        check(cudaMemcpy(d_state, d_state0, ns * 4, cudaMemcpyDeviceToDevice), "state0");
        check(cudaMemset(d_y, 0xff, ny * 4), "memset");
        sk::fused_gdn_step_norm(d_state, d_h, d_h + qk, d_h + 2 * qk, d_gate, d_beta, d_z, d_gamma, EPS, d_y, h_k, h_v,
                                nullptr);
        check(cudaDeviceSynchronize(), "fused_gdn_step_norm");
        st[pass] = host_copy(d_state, ns);
        y[pass] = host_copy(d_y, ny);
    }
    sk::old_gdn_step().set(false);
    std::vector<double> rs(c.state.begin(), c.state.end()), ry;
    step_reference(rs, c, 0, ry);
    char name[96], det[96];
    std::snprintf(name, sizeof name, "fused_gdn_step_norm h_k=%d h_v=%d: new == old", h_k, h_v);
    report(same_bytes(st[0].data(), st[1].data(), ns * 4) && same_bytes(y[0].data(), y[1].data(), ny * 4),
           "  %-58s %s%s\n", name);
    const double es = rel_l1(st[1], rs), ey = rel_l1(y[1], ry);
    std::snprintf(name, sizeof name, "fused_gdn_step_norm h_k=%d h_v=%d: vs double reference", h_k, h_v);
    std::snprintf(det, sizeof det, " (state %.2e, y %.2e)", es, ey);
    report(es <= 1e-4 && ey <= 1e-4, "  %-58s %s%s\n", name, det);
    cudaFree(d_state0); cudaFree(d_state); cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z);
    cudaFree(d_gamma); cudaFree(d_y);
}

void test_step_multi(int h_k, int h_v, unsigned seed) {
    const int MT = sk::kVerifyMaxT;
    const StepCase c = make_step(h_k, h_v, MT, seed);
    const size_t ns = c.state.size(), ny = (size_t) MT * h_v * S;
    float *d_state0 = dev_copy(c.state), *d_state = dev_alloc<float>(ns), *d_h = dev_copy(c.h);
    float *d_gate = dev_copy(c.gate), *d_beta = dev_copy(c.beta), *d_z = dev_copy(c.z), *d_gamma = dev_copy(c.gamma);
    float* d_y = dev_alloc<float>(ny);
    int32_t* d_keep = dev_alloc<int32_t>(1);
    // the reference: states and outputs after each token
    std::vector<std::vector<double>> ref_state((size_t) MT + 1), ref_y((size_t) MT);
    ref_state[0].assign(c.state.begin(), c.state.end());
    for (int t = 0; t < MT; ++t) {
        ref_state[(size_t) t + 1] = ref_state[(size_t) t];
        step_reference(ref_state[(size_t) t + 1], c, t, ref_y[(size_t) t]);
    }
    // one run: returns state and y
    auto run = [&](bool old_path, int T, const int32_t* keep, int t_out_begin, std::vector<float>& st,
                   std::vector<float>& y) {
        sk::old_gdn_step().set(old_path);
        check(cudaMemcpy(d_state, d_state0, ns * 4, cudaMemcpyDeviceToDevice), "state0");
        check(cudaMemset(d_y, 0xff, ny * 4), "memset");
        sk::gdn_step_norm_multi(d_state, d_h, c.C, d_gate, d_beta, d_z, d_gamma, EPS, d_y, h_k, h_v, T, keep, nullptr,
                                t_out_begin);
        check(cudaDeviceSynchronize(), "gdn_step_norm_multi");
        st = host_copy(d_state, ns);
        y = host_copy(d_y, ny);
    };
    int cases = 0, differ = 0, untouched_bad = 0;
    double worst = 0;
    // verify: the state is read, not written; outputs for t in [t_out_begin, T)
    for (int T = 1; T <= MT; ++T) {
        const int begins[3] = {0, T > 1 ? 1 : 0, T - 1};
        for (int bi = 0; bi < 3; ++bi) {
            if (bi > 0 && begins[bi] == begins[bi - 1]) continue;
            const int tb = begins[bi];
            std::vector<float> st0, y0, st1, y1;
            run(true, T, nullptr, tb, st0, y0);
            run(false, T, nullptr, tb, st1, y1);
            ++cases;
            if (!same_bytes(st0.data(), st1.data(), ns * 4) || !same_bytes(y0.data(), y1.data(), ny * 4)) ++differ;
            if (!same_bytes(st1.data(), c.state.data(), ns * 4)) ++untouched_bad;
            for (int t = tb; t < T; ++t) {
                const std::vector<float> yt(y1.begin() + (size_t) t * h_v * S, y1.begin() + (size_t) (t + 1) * h_v * S);
                worst = std::max(worst, rel_l1(yt, ref_y[(size_t) t]));
            }
        }
    }
    // commit: the first *n_keep tokens are run and the state is written (T = MT, as the verifier launches it)
    for (int keep : {0, 1, 3, MT}) {
        check(cudaMemcpy(d_keep, &keep, 4, cudaMemcpyHostToDevice), "keep");
        std::vector<float> st0, y0, st1, y1;
        run(true, MT, d_keep, 0, st0, y0);
        run(false, MT, d_keep, 0, st1, y1);
        ++cases;
        if (!same_bytes(st0.data(), st1.data(), ns * 4) || !same_bytes(y0.data(), y1.data(), ny * 4)) ++differ;
        worst = std::max(worst, rel_l1(st1, ref_state[(size_t) keep]));
    }
    sk::old_gdn_step().set(false);
    char name[96], det[96];
    std::snprintf(name, sizeof name, "gdn_step_norm_multi h_k=%d h_v=%d: new == old", h_k, h_v);
    std::snprintf(det, sizeof det, " (%d of %d verify/commit cases differ)", differ, cases);
    report(differ == 0, "  %-58s %s%s\n", name, det);
    std::snprintf(name, sizeof name, "gdn_step_norm_multi h_k=%d h_v=%d: verify leaves the state", h_k, h_v);
    report(untouched_bad == 0, "  %-58s %s%s\n", name);
    std::snprintf(name, sizeof name, "gdn_step_norm_multi h_k=%d h_v=%d: vs double reference", h_k, h_v);
    std::snprintf(det, sizeof det, " (worst %.2e)", worst);
    report(worst <= 1e-4, "  %-58s %s%s\n", name, det);
    cudaFree(d_state0); cudaFree(d_state); cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z);
    cudaFree(d_gamma); cudaFree(d_y); cudaFree(d_keep);
}

// ------------------------------------------------------------------ 5. the grid-stride copies
void test_fetch_blobs(int64_t blob_bytes, int cap) {
    // the blobs live in mapped host memory, as the PCIe share of a layer's missed experts does
    uint8_t* host = nullptr;
    check(cudaHostAlloc((void**) &host, (size_t) blob_bytes * cap, cudaHostAllocMapped), "cudaHostAlloc");
    std::mt19937 rng(7);
    for (int64_t i = 0; i < blob_bytes * cap; ++i) host[i] = (uint8_t) rng();
    uint8_t* host_dev = nullptr;
    check(cudaHostGetDevicePointer((void**) &host_dev, host, 0), "cudaHostGetDevicePointer");
    // the blobs in a scrambled order, so a copy that ignores src[k] is visible
    std::vector<unsigned long long> ptrs((size_t) cap);
    for (int k = 0; k < cap; ++k) ptrs[(size_t) k] = (unsigned long long) (host_dev + (size_t) ((k * 5 + 3) % cap) * blob_bytes);
    unsigned long long* d_ptrs = dev_copy(ptrs);
    int32_t* d_n = dev_alloc<int32_t>(1);
    uint8_t* d_dst = dev_alloc<uint8_t>((size_t) blob_bytes * cap);
    int bad = 0;
    for (int n : {0, 1, cap}) {
        for (int pass = 0; pass < 2; ++pass) {
            sk::old_grids().set(pass == 0);
            check(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice), "n");
            check(cudaMemset(d_dst, 0xab, (size_t) blob_bytes * cap), "memset");
            sk::fetch_blobs(d_ptrs, d_n, d_dst, blob_bytes, cap, nullptr);
            check(cudaDeviceSynchronize(), "fetch_blobs");
            const std::vector<uint8_t> got = host_copy(d_dst, (size_t) blob_bytes * cap);
            for (int k = 0; k < cap; ++k) {
                const uint8_t* g = got.data() + (size_t) k * blob_bytes;
                if (k < n) {
                    if (!same_bytes(g, host + (size_t) ((k * 5 + 3) % cap) * blob_bytes, (size_t) blob_bytes)) ++bad;
                } else {
                    for (int64_t i = 0; i < blob_bytes; ++i)
                        if (g[i] != 0xab) { ++bad; break; }
                }
            }
        }
    }
    sk::old_grids().set(false);
    char name[96];
    std::snprintf(name, sizeof name, "fetch_blobs %lld B x %d, both grids", (long long) blob_bytes, cap);
    report(bad == 0, "  %-58s %s%s\n", name);
    cudaFree(d_ptrs); cudaFree(d_n); cudaFree(d_dst); cudaFreeHost(host);
}

void test_gather_rows(int64_t row_bytes, int64_t rows, int64_t n) {
    std::mt19937 rng(9);
    std::vector<uint8_t> src((size_t) (row_bytes * rows));
    for (uint8_t& b : src) b = (uint8_t) rng();
    std::vector<int32_t> ids((size_t) n);
    for (int32_t& i : ids) i = (int32_t) (rng() % (uint32_t) rows);
    uint8_t* d_src = dev_copy(src);
    int32_t* d_ids = dev_copy(ids);
    uint8_t* d_dst = dev_alloc<uint8_t>((size_t) (row_bytes * n));
    int bad = 0;
    for (int pass = 0; pass < 2; ++pass) {
        sk::old_grids().set(pass == 0);
        check(cudaMemset(d_dst, 0, (size_t) (row_bytes * n)), "memset");
        sk::gather_rows(d_src, row_bytes, d_ids, n, d_dst, nullptr);
        check(cudaDeviceSynchronize(), "gather_rows");
        const std::vector<uint8_t> got = host_copy(d_dst, (size_t) (row_bytes * n));
        for (int64_t i = 0; i < n; ++i)
            if (!same_bytes(got.data() + i * row_bytes, src.data() + (int64_t) ids[(size_t) i] * row_bytes, (size_t) row_bytes))
                ++bad;
    }
    sk::old_grids().set(false);
    char name[96];
    std::snprintf(name, sizeof name, "gather_rows %lld B x %lld rows, both grids", (long long) row_bytes, (long long) n);
    report(bad == 0, "  %-58s %s%s\n", name);
    cudaFree(d_src); cudaFree(d_ids); cudaFree(d_dst);
}

// ------------------------------------------------------------------ --bench
template <typename Fn>
double time_us(Fn&& fn, int iters) {
    for (int i = 0; i < 10; ++i) fn();
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    check(cudaEventRecord(a, nullptr), "record");
    for (int i = 0; i < iters; ++i) fn();
    check(cudaEventRecord(b, nullptr), "record");
    check(cudaEventSynchronize(b), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return 1000.0 * ms / iters;
}

template <typename Fn>
void bench_pair(const char* what, sk::OldPath& sw, Fn&& fn, int iters = 2000) {
    sw.set(true);
    const double t_old = time_us(fn, iters);
    sw.set(false);
    const double t_new = time_us(fn, iters);
    std::printf("  %-44s old %8.2f us   new %8.2f us   (%+.1f%%)\n", what, t_old, t_new,
                100.0 * (t_new - t_old) / t_old);
}

void bench() {
    std::printf("\n--bench (back-to-back launches; in the engine each one follows other work, so read the "
                "difference, not the absolute)\n");
    const int N = 2560, HV = 48, HK = 16, MT = sk::kVerifyMaxT;
    {
        const AbCase c = make_ab(N, HV, MT, 5);
        float *dx = dev_copy(c.x), *ddt = dev_copy(c.dt), *dsa = dev_copy(c.ssm_a);
        uint16_t *dwa = dev_copy(c.wa), *dwb = dev_copy(c.wb);
        float *dg = dev_alloc<float>((size_t) MT * HV), *db = dev_alloc<float>((size_t) MT * HV);
        // a 192 MB buffer rewritten between launches (the 5090 has 96 MB of L2) keeps the weights out of it, as in a
        // decode step
        uint8_t* flush = dev_alloc<uint8_t>((size_t) 192 << 20);
        bench_pair("fused_gdn_ab (L2-hot)", sk::old_gdn_ab(),
                   [&] { sk::fused_gdn_ab(dx, dwa, dwb, ddt, dsa, dg, db, N, HV, nullptr); });
        bench_pair("fused_gdn_ab (+192 MB memset, cold)", sk::old_gdn_ab(), [&] {
            cudaMemsetAsync(flush, 0, (size_t) 192 << 20, nullptr);
            sk::fused_gdn_ab(dx, dwa, dwb, ddt, dsa, dg, db, N, HV, nullptr);
        }, 200);
        for (int T : {2, 4, 8}) {
            char name[64];
            std::snprintf(name, sizeof name, "gdn_ab_multi T=%d (+192 MB memset, cold)", T);
            bench_pair(name, sk::old_gdn_ab(), [&] {
                cudaMemsetAsync(flush, 0, (size_t) 192 << 20, nullptr);
                sk::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, dg, db, N, HV, T, nullptr);
            }, 200);
        }
        std::printf("  %-44s %8.2f us\n", "  (the 192 MB memset alone)",
                    time_us([&] { cudaMemsetAsync(flush, 0, (size_t) 192 << 20, nullptr); }, 200));
        cudaFree(dx); cudaFree(ddt); cudaFree(dsa); cudaFree(dwa); cudaFree(dwb); cudaFree(dg); cudaFree(db);
        cudaFree(flush);
    }
    {
        const StepCase c = make_step(HK, HV, MT, 6);
        const int qk = S * HK;
        float *d_state = dev_copy(c.state), *d_h = dev_copy(c.h), *d_gate = dev_copy(c.gate);
        float *d_beta = dev_copy(c.beta), *d_z = dev_copy(c.z), *d_gamma = dev_copy(c.gamma);
        float* d_y = dev_alloc<float>((size_t) MT * HV * S);
        int32_t* d_keep = dev_alloc<int32_t>(1);
        const int32_t keep = 4;
        check(cudaMemcpy(d_keep, &keep, 4, cudaMemcpyHostToDevice), "keep");
        bench_pair("fused_gdn_step_norm", sk::old_gdn_step(), [&] {
            sk::fused_gdn_step_norm(d_state, d_h, d_h + qk, d_h + 2 * qk, d_gate, d_beta, d_z, d_gamma, EPS, d_y, HK, HV,
                                    nullptr);
        });
        for (int T : {2, 4, 8}) {
            char name[64];
            std::snprintf(name, sizeof name, "gdn_step_norm_multi verify T=%d", T);
            bench_pair(name, sk::old_gdn_step(), [&] {
                sk::gdn_step_norm_multi(d_state, d_h, c.C, d_gate, d_beta, d_z, d_gamma, EPS, d_y, HK, HV, T, nullptr,
                                        nullptr, 0);
            });
        }
        bench_pair("gdn_step_norm_multi commit n_keep=4 of 8", sk::old_gdn_step(), [&] {
            sk::gdn_step_norm_multi(d_state, d_h, c.C, d_gate, d_beta, d_z, d_gamma, EPS, d_y, HK, HV, MT, d_keep,
                                    nullptr, 0);
        });
        cudaFree(d_state); cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z); cudaFree(d_gamma);
        cudaFree(d_y); cudaFree(d_keep);
    }
    {
        const int64_t blob = 1 << 20;
        const int cap = 8;
        uint8_t* host = nullptr;
        check(cudaHostAlloc((void**) &host, (size_t) blob * cap, cudaHostAllocMapped), "cudaHostAlloc");
        std::memset(host, 1, (size_t) blob * cap);
        uint8_t* host_dev = nullptr;
        check(cudaHostGetDevicePointer((void**) &host_dev, host, 0), "cudaHostGetDevicePointer");
        std::vector<unsigned long long> ptrs((size_t) cap);
        for (int k = 0; k < cap; ++k) ptrs[(size_t) k] = (unsigned long long) (host_dev + (size_t) k * blob);
        unsigned long long* d_ptrs = dev_copy(ptrs);
        int32_t* d_n = dev_alloc<int32_t>(1);
        uint8_t* d_dst = dev_alloc<uint8_t>((size_t) blob * cap);
        for (int n : {0, 2, cap}) {
            check(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice), "n");
            char name[64];
            std::snprintf(name, sizeof name, "fetch_blobs %d x 1 MiB (mapped host)", n);
            bench_pair(name, sk::old_grids(), [&] { sk::fetch_blobs(d_ptrs, d_n, d_dst, blob, cap, nullptr); },
                       n == 0 ? 2000 : 50);
        }
        cudaFree(d_ptrs); cudaFree(d_n); cudaFree(d_dst); cudaFreeHost(host);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") do_bench = true;
        else if (a == "--selftest") {}
        else { std::fprintf(stderr, "usage: grids_parity [--selftest] [--bench]\n"); return 2; }
    }
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    const sk::GridDevice& gd = sk::grid_device();
    // the switches may come from the environment; this test sets them itself
    sk::old_grids().set(false);
    sk::old_gdn_ab().set(false);
    sk::old_gdn_step().set(false);
    std::printf("grids_parity: %s, sm_%d%d, %d SMs, clusters %s\n", prop.name, prop.major, prop.minor, gd.sms,
                gd.clusters ? "yes" : "no");
    std::printf("  copy grids: %d blocks (48 * 8 = 384 before)\n", sk::grid_sms() * 8);
    std::printf("  cluster: fused_gdn_step_norm %s, gdn_step_norm_multi %s\n",
                sk::fused_gdn_step_cluster_path() ? "yes" : "NO (old kernel; its checks compare it with itself)",
                sk::gdn_step_norm_multi_cluster_path() ? "yes" : "NO (old kernel; its checks compare it with itself)");

    std::printf("\n1-2. alpha/beta: a block of four warps per row vs a warp per row\n");
    for (int n : {2560, 2048, 4096, 1032, 8, 4104}) test_ab(n, 48);
    test_ab(2560, 5);
    test_ab_multi(2560, 48);
    test_ab_multi(1032, 7);

    std::printf("\n3-4. the GDN step: a cluster of 4 blocks per head vs a block per head\n");
    test_step(16, 48, 31);
    test_step(4, 12, 32);
    test_step_multi(16, 48, 41);
    test_step_multi(2, 6, 42);

    std::printf("\n5. grid-stride copies: the device's SMs * 8 vs 48 * 8\n");
    test_fetch_blobs(1 << 20, 4);
    test_fetch_blobs(48 * 1024 + 16, 7);
    test_gather_rows(2100, 1000, 333);      // a Q6_K head row: 4-byte elements
    test_gather_rows(5120, 257, 1000);      // 16-byte elements
    test_gather_rows(21, 50, 77);           // bytes

    if (do_bench) bench();

    std::printf("\n%s (%d failed)\n", g_bad ? "FAIL" : "PASS", g_bad);
    return g_bad ? 1 : 0;
}
