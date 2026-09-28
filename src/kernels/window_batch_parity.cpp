// src/kernels/window_batch_parity.cpp - the verify window's batched launches (#19, #44) against the per-token
// launches they replace.  Every check is BITWISE (memcmp): each batched kernel is claimed to reproduce the old
// sequence value for value, not to within a tolerance.  Synthetic data, no model; GPU required.
//
//   1. bf16_gemv_fp32_mmvf_multi       vs one bf16_gemv_fp32_mmvf per column (router, indexer, gate shapes)
//   2. native_router_top10_multi       vs one native_router_top10 per token (ties included); router_top10 with
//                                      n_tokens > 1 vs one call per token (the non-native router's batch)
//   3. native_moe_combine_window / moe_combine_window (gate mode 0)
//                                      vs copy_from_mapped + moe_hit_add + one combine per token, with the CPU's
//                                      rows in mapped pinned memory, the GPU's rows listed in any order, -0 and
//                                      denormals in both, NaN in the rows that must not be read, and no hit list
//   4. shared_expert_multi_batched     vs shared_expert_multi (native and BF16 gate): the gated output and the
//                                      Q8_1 of the SwiGLU; then deferred gate + window combine vs the old chain
//                                      end to end, for both combine kernels
//   5. the QSA query rows              the q/gate split copy, norm, rotation, FWHT and output gate over a group's
//                                      heads in one call each vs per token (native and legacy kernels)

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;
cudaStream_t g_cs = nullptr;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "malloc");
    return p;
}
template <typename T>
void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    ck(cudaStreamSynchronize(g_cs), "sync");
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
template <typename T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
void check(bool ok, const std::string& what) {
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

float denormal(std::mt19937& rng) {   // a nonzero mantissa under a zero exponent, either sign
    const uint32_t bits = ((rng() & 1u) << 31) | (1u + rng() % 0x7FFFFFu);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------- 1. the multi-column MMVF
void test_mmvf(std::mt19937& rng) {
    std::printf("bf16_gemv_fp32_mmvf_multi vs one launch per column\n");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct Shape { int64_t n_in, n_out; };
    const Shape shapes[] = {{2560, 512}, {2560, 128}, {2560, 1}, {2560, 2048}, {640, 17}, {130, 9}, {6, 3}, {64, 1}};
    for (const Shape& s : shapes) {
        const int64_t xs = s.n_in + 6, ys = s.n_out + 3;   // padded strides: columns are not assumed packed
        std::vector<uint16_t> w((size_t) (s.n_in * s.n_out));
        for (auto& v : w) v = k::bf16_from_f32(nd(rng) * 0.05f);
        std::vector<float> x((size_t) (xs * k::kMmvfMaxCols));
        for (auto& v : x) v = nd(rng);
        uint16_t* dw = dalloc<uint16_t>(w.size());
        float* dx = dalloc<float>(x.size());
        float* d1 = dalloc<float>((size_t) (ys * k::kMmvfMaxCols));
        float* d2 = dalloc<float>((size_t) (ys * k::kMmvfMaxCols));
        up(dw, w);
        up(dx, x);
        bool ok = true;
        for (int n = 1; n <= k::kMmvfMaxCols; ++n) {
            ck(cudaMemset(d1, 0, (size_t) (ys * k::kMmvfMaxCols) * 4), "zero");
            ck(cudaMemset(d2, 0, (size_t) (ys * k::kMmvfMaxCols) * 4), "zero");
            for (int j = 0; j < n; ++j) k::bf16_gemv_fp32_mmvf(dx + j * xs, dw, d1 + j * ys, s.n_in, s.n_out, g_cs);
            k::bf16_gemv_fp32_mmvf_multi(dx, xs, dw, d2, ys, s.n_in, s.n_out, n, g_cs);
            ok = ok && same(down(d1, (size_t) (ys * k::kMmvfMaxCols)), down(d2, (size_t) (ys * k::kMmvfMaxCols)));
        }
        check(ok, "n_in " + std::to_string(s.n_in) + ", n_out " + std::to_string(s.n_out) + ", 1..8 columns");
        cudaFree(dw); cudaFree(dx); cudaFree(d1); cudaFree(d2);
    }
}

// ---------------------------------------------------------------- 2. the routers
void test_router(std::mt19937& rng) {
    std::printf("the router's top-10 for a window vs one launch per token\n");
    std::normal_distribution<float> nd(0.0f, 2.0f);
    constexpr int E = 512, K = 10, MAXT = 20;
    std::vector<float> logits((size_t) E * MAXT);
    for (auto& v : logits) v = nd(rng);
    // ties: whole groups of equal logits on some rows, so the lower-index rule is exercised
    for (int t = 0; t < MAXT; t += 3)
        for (int e = 0; e < E; e += 7) logits[(size_t) t * E + e] = logits[(size_t) t * E + (e / 49) * 49];
    float* dl = dalloc<float>(logits.size());
    up(dl, logits);
    int32_t *i1 = dalloc<int32_t>((size_t) K * MAXT), *i2 = dalloc<int32_t>((size_t) K * MAXT);
    float *w1 = dalloc<float>((size_t) K * MAXT), *w2 = dalloc<float>((size_t) K * MAXT);
    const int counts[] = {1, 2, 3, 4, 5, 8, 9, 17, 20};
    bool ok_native = true, ok_legacy = true;
    for (int n : counts) {
        ck(cudaMemset(i1, 0xFF, (size_t) K * MAXT * 4), "m"); ck(cudaMemset(i2, 0xFF, (size_t) K * MAXT * 4), "m");
        ck(cudaMemset(w1, 0, (size_t) K * MAXT * 4), "m"); ck(cudaMemset(w2, 0, (size_t) K * MAXT * 4), "m");
        for (int t = 0; t < n; ++t) k::native_router_top10(dl + (size_t) t * E, i1 + t * K, w1 + t * K, g_cs);
        k::native_router_top10_multi(dl, i2, w2, n, g_cs);
        ok_native = ok_native && same(down(i1, (size_t) K * MAXT), down(i2, (size_t) K * MAXT)) &&
                    same(down(w1, (size_t) K * MAXT), down(w2, (size_t) K * MAXT));
        ck(cudaMemset(i1, 0xFF, (size_t) K * MAXT * 4), "m"); ck(cudaMemset(i2, 0xFF, (size_t) K * MAXT * 4), "m");
        for (int t = 0; t < n; ++t) k::router_top10(dl + (size_t) t * E, 1, E, K, i1 + t * K, w1 + t * K, g_cs);
        k::router_top10(dl, n, E, K, i2, w2, g_cs);
        ok_legacy = ok_legacy && same(down(i1, (size_t) K * MAXT), down(i2, (size_t) K * MAXT)) &&
                    same(down(w1, (size_t) K * MAXT), down(w2, (size_t) K * MAXT));
    }
    check(ok_native, "native_router_top10_multi, 1..20 tokens (one to three blocks)");
    check(ok_legacy, "router_top10 with n_tokens > 1, 1..20 tokens");
    cudaFree(dl); cudaFree(i1); cudaFree(i2); cudaFree(w1); cudaFree(w2);
}

// ---------------------------------------------------------------- 3. the window combination (E1)
struct Window {
    int T = 0, K = 0;
    int64_t N = 0;
    float* h_ymiss = nullptr;   // mapped pinned, as the verifier's staging
    float* m_ymiss = nullptr;
    float *hit_out = nullptr, *weights = nullptr, *shared = nullptr, *parts = nullptr;
    int32_t *dst = nullptr, *count = nullptr;
    int n_hits = 0;
};

// The pool's contract: a GPU row of y_miss is zero, the CPU's rows are its results; the GPU rows of hit_out
// are the GPU's results and its other rows are whatever an earlier window left (NaN here: they must not be read).
Window make_window(std::mt19937& rng, int T, int K, int64_t N, double hit_rate) {
    Window w;
    w.T = T; w.K = K; w.N = N;
    const size_t rows = (size_t) T * K, elems = rows * (size_t) N;
    ck(cudaHostAlloc((void**) &w.h_ymiss, elems * 4, cudaHostAllocMapped), "host alloc");
    ck(cudaHostGetDevicePointer((void**) &w.m_ymiss, w.h_ymiss, 0), "device alias");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int32_t> hit_rows;
    std::vector<float> hit((size_t) elems, std::numeric_limits<float>::quiet_NaN());
    for (size_t r = 0; r < rows; ++r) {
        const bool gpu = u(rng) < hit_rate;
        if (gpu) hit_rows.push_back((int32_t) r);
        for (int64_t j = 0; j < N; ++j) {
            const size_t at = r * (size_t) N + (size_t) j;
            float v = nd(rng);
            const uint32_t pick = rng() % 64;
            if (pick == 0) v = -0.0f;
            else if (pick == 1) v = denormal(rng);
            else if (pick == 2) v = 0.0f;
            if (gpu) { hit[at] = v; w.h_ymiss[at] = 0.0f; }
            else w.h_ymiss[at] = v;
        }
    }
    std::shuffle(hit_rows.begin(), hit_rows.end(), rng);   // the plan lists groups, not rows in order
    w.n_hits = (int) hit_rows.size();
    std::vector<float> wt(rows), sh((size_t) T * (size_t) N);
    for (int t = 0; t < T; ++t) {
        double s = 0.0;
        for (int e = 0; e < K; ++e) { wt[(size_t) (t * K + e)] = std::fabs(nd(rng)) + 0.01f; s += wt[(size_t) (t * K + e)]; }
        for (int e = 0; e < K; ++e) wt[(size_t) (t * K + e)] = (float) (wt[(size_t) (t * K + e)] / s);
    }
    for (auto& v : sh) v = (rng() % 50 == 0) ? denormal(rng) : nd(rng) * 2.0f;
    w.hit_out = dalloc<float>(elems);
    w.parts = dalloc<float>(elems);
    w.weights = dalloc<float>(rows);
    w.shared = dalloc<float>(sh.size());
    w.dst = dalloc<int32_t>(rows);
    w.count = dalloc<int32_t>(1);
    up(w.hit_out, hit);
    up(w.weights, wt);
    up(w.shared, sh);
    if (!hit_rows.empty()) up(w.dst, hit_rows);
    ck(cudaMemcpy(w.count, &w.n_hits, 4, cudaMemcpyHostToDevice), "count");
    return w;
}
void free_window(Window& w) {
    cudaFreeHost(w.h_ymiss);
    cudaFree(w.hit_out); cudaFree(w.parts); cudaFree(w.weights); cudaFree(w.shared); cudaFree(w.dst); cudaFree(w.count);
}

// The sequence the window used to record: all rows over PCIe, the hits added, one combine per token.
void old_combine(const Window& w, bool native, bool hits, float* out) {
    const int64_t rows = (int64_t) w.T * w.K;
    k::copy_from_mapped(w.parts, w.m_ymiss, rows * w.N, g_cs);
    if (hits) k::moe_hit_add(w.parts, w.hit_out, w.dst, w.count, rows, w.N, g_cs);
    for (int t = 0; t < w.T; ++t) {
        const float* p = w.parts + (size_t) t * w.K * w.N;
        if (native) k::native_moe_combine(p, w.weights + t * w.K, w.shared + t * w.N, out + t * w.N, w.N, w.K, g_cs);
        else k::moe_combine(p, w.weights + t * w.K, w.shared + t * w.N, out + t * w.N, w.N, w.K, g_cs);
    }
}
void new_combine(const Window& w, bool native, bool hits, float* out, const float* gate = nullptr,
                 k::SharedGate mode = k::SharedGate::Applied) {
    const int32_t* dst = hits ? w.dst : nullptr;
    const int32_t* cnt = hits ? w.count : nullptr;
    if (native)
        k::native_moe_combine_window(w.m_ymiss, w.hit_out, dst, cnt, w.weights, w.shared, gate, (int) mode, out, w.N,
                                     w.K, w.T, g_cs);
    else
        k::moe_combine_window(w.m_ymiss, w.hit_out, dst, cnt, w.weights, w.shared, gate, mode, out, w.N, w.K, w.T, g_cs);
}

void test_combine(std::mt19937& rng) {
    std::printf("the window combination (rows read in place) vs copy + moe_hit_add + per-token combine\n");
    const int64_t N = 2560;
    const int K = 10;
    struct Case { int T; double hit_rate; };
    const Case cases[] = {{1, 0.93}, {4, 0.93}, {8, 0.5}, {8, 0.0}, {8, 1.0}, {3, 0.3}};
    for (const Case& c : cases) {
        Window w = make_window(rng, c.T, K, N, c.hit_rate);
        float* o1 = dalloc<float>((size_t) c.T * N);
        float* o2 = dalloc<float>((size_t) c.T * N);
        for (int native = 0; native < 2; ++native) {
            for (int hits = 0; hits < 2; ++hits) {   // without a hit list every row is the CPU's (zeros included)
                ck(cudaMemset(o1, 0xFF, (size_t) c.T * N * 4), "m");
                ck(cudaMemset(o2, 0x7F, (size_t) c.T * N * 4), "m");
                old_combine(w, native != 0, hits != 0, o1);
                new_combine(w, native != 0, hits != 0, o2);
                check(same(down(o1, (size_t) c.T * N), down(o2, (size_t) c.T * N)),
                      std::string(native ? "native" : "double") + " combine, " + std::to_string(c.T) + " tokens, " +
                          std::to_string(w.n_hits) + "/" + std::to_string(c.T * K) + " GPU rows" +
                          (hits ? "" : ", no hit list"));
            }
        }
        cudaFree(o1); cudaFree(o2);
        free_window(w);
    }
}

// ---------------------------------------------------------------- 4. the shared expert, then the whole chain
std::vector<uint8_t> random_q8_0(std::mt19937& rng, int64_t rows, int64_t n_in) {
    std::vector<uint8_t> b((size_t) (rows * (n_in / 32) * 34));
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t blk = 0; blk < b.size() / 34; ++blk) {
        const uint16_t d = k::f16_from_f32(ud(rng));
        std::memcpy(&b[blk * 34], &d, 2);
        for (int i = 0; i < 32; ++i) b[blk * 34 + 2 + (size_t) i] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
    }
    return b;
}

void test_shared(std::mt19937& rng) {
    std::printf("shared_expert_multi_batched vs shared_expert_multi, then the chain into the combination\n");
    constexpr int GGML_Q8_0 = 8;
    const int64_t N = 2560, FF = 640;
    const int K = 10;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const std::vector<uint8_t> hg = random_q8_0(rng, FF, N), hu = random_q8_0(rng, FF, N), hd = random_q8_0(rng, N, FF);
    std::vector<uint16_t> ginp((size_t) N);
    for (auto& v : ginp) v = k::bf16_from_f32(nd(rng) * 0.02f);
    uint8_t *dg = dalloc<uint8_t>(hg.size()), *du = dalloc<uint8_t>(hu.size()), *dd = dalloc<uint8_t>(hd.size());
    uint16_t* dginp = dalloc<uint16_t>(ginp.size());
    up(dg, hg); up(du, hu); up(dd, hd); up(dginp, ginp);
    const int MAXT = 8;
    std::vector<float> x((size_t) (MAXT * N));
    for (auto& v : x) v = nd(rng);
    std::vector<uint16_t> xb(x.size());
    for (size_t i = 0; i < x.size(); ++i) xb[i] = k::bf16_from_f32(x[i]);
    float* dx = dalloc<float>(x.size());
    uint16_t* dxb = dalloc<uint16_t>(xb.size());
    up(dx, x); up(dxb, xb);
    const size_t q8 = k::native_q8_1_bytes((int) N, MAXT);
    uint8_t *q1 = dalloc<uint8_t>(q8), *q2 = dalloc<uint8_t>(q8);
    float *gate = dalloc<float>((size_t) (MAXT * FF)), *upb = dalloc<float>((size_t) (MAXT * FF));
    float *g1 = dalloc<float>(MAXT), *g2 = dalloc<float>(MAXT);
    float *s1 = dalloc<float>((size_t) (MAXT * N)), *s2 = dalloc<float>((size_t) (MAXT * N));
    k::NativeSharedWeights nw;
    nw.gate_type = nw.up_type = nw.down_type = GGML_Q8_0;
    nw.gate_data = dg; nw.up_data = du; nw.down_data = dd;
    for (int native_gate = 0; native_gate < 2; ++native_gate) {
        k::shared_expert_set_native_bf16(native_gate != 0);
        const char* gname = native_gate ? "native gate" : "BF16 gate";
        for (int T : {1, 3, 4, 8}) {
            const size_t q8ff = k::native_q8_1_bytes((int) FF, T);   // the SwiGLU's T columns, written last
            ck(cudaMemset(s1, 0, (size_t) (MAXT * N) * 4), "m"); ck(cudaMemset(s2, 0, (size_t) (MAXT * N) * 4), "m");
            nw.q8_1 = q1;
            k::shared_expert_multi(T, dx, dxb, nw, dginp, gate, upb, g1, s1, N, FF, g_cs);
            const std::vector<uint8_t> swiglu_old = down(q1, q8ff);   // the last Q8_1: the SwiGLU's
            nw.q8_1 = q2;
            const k::SharedGate m = k::shared_expert_multi_batched(T, dx, dxb, nw, dginp, gate, upb, g2, s2, N, FF,
                                                                   g_cs, /*defer_gate=*/false);
            const std::vector<uint8_t> swiglu_new = down(q2, q8ff);
            check(m == k::SharedGate::Applied && same(down(s1, (size_t) (MAXT * N)), down(s2, (size_t) (MAXT * N))),
                  std::string(gname) + ", " + std::to_string(T) + " tokens: the gated output");
            check(same(swiglu_old, swiglu_new), std::string(gname) + ", " + std::to_string(T) +
                                                    " tokens: the SwiGLU's Q8_1 (fused SwiGLU + quantization)");
        }
        // the chain: old = gated shared + copy/hit_add/combine; new = deferred gate folded into the combination
        for (int T : {1, 4, 8}) {
            Window w = make_window(rng, T, K, N, 0.8);
            float *o1 = dalloc<float>((size_t) T * N), *o2 = dalloc<float>((size_t) T * N);
            for (int native = 0; native < 2; ++native) {
                nw.q8_1 = q1;
                k::shared_expert_multi(T, dx, dxb, nw, dginp, gate, upb, g1, w.shared, N, FF, g_cs);
                old_combine(w, native != 0, true, o1);
                nw.q8_1 = q2;
                const k::SharedGate m = k::shared_expert_multi_batched(T, dx, dxb, nw, dginp, gate, upb, g2, w.shared, N,
                                                                       FF, g_cs, /*defer_gate=*/true);
                new_combine(w, native != 0, true, o2, g2, m);
                const bool mode_ok = m == (native_gate ? k::SharedGate::Logit : k::SharedGate::Value);
                check(mode_ok && same(down(o1, (size_t) T * N), down(o2, (size_t) T * N)),
                      std::string(gname) + " folded into the " + (native ? "native" : "double") + " combination, " +
                          std::to_string(T) + " tokens");
            }
            cudaFree(o1); cudaFree(o2);
            free_window(w);
        }
    }
    k::shared_expert_set_native_bf16(false);
    cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dginp); cudaFree(dx); cudaFree(dxb); cudaFree(q1); cudaFree(q2);
    cudaFree(gate); cudaFree(upb); cudaFree(g1); cudaFree(g2); cudaFree(s1); cudaFree(s2);
}

// ---------------------------------------------------------------- 5. the QSA query rows and the output gate
// The window now treats a group's n * NH query heads as rows of one pitch: one q/gate split copy, one norm, one
// rotation (a position per row), one FWHT and one output gate.  Each kernel is per row (or per element), so the
// batched call must equal the per-token calls bit for bit - native and legacy variants both.
void test_qsa_rows(std::mt19937& rng) {
    std::printf("the QSA query rows (split, norm, rotation, FWHT) and the output gate, a group vs per token\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int NH = (int) s.n_head, HD = (int) s.head_dim, NR = (int) s.n_rot, MAXT = 8, MAXPOS = 4096;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qfull((size_t) MAXT * NH * 2 * HD), attn((size_t) MAXT * NH * HD), wn((size_t) HD);
    for (auto& v : qfull) v = nd(rng) * 2.0f;
    for (auto& v : attn) v = nd(rng);
    for (auto& v : wn) v = 1.0f + 0.1f * nd(rng);
    std::vector<int32_t> pos((size_t) MAXT * NH);
    for (int t = 0; t < MAXT; ++t)
        for (int h = 0; h < NH; ++h) pos[(size_t) (t * NH + h)] = 1000 + 37 * t;
    std::vector<float> cos_tab((size_t) MAXPOS * (NR / 2)), sin_tab((size_t) MAXPOS * (NR / 2));
    k::build_rope_table(NR, k::qsa_freq_base(), MAXPOS, cos_tab.data(), sin_tab.data());
    float *dq = dalloc<float>(qfull.size()), *da = dalloc<float>(attn.size()), *dwn = dalloc<float>(wn.size());
    float *dcos = dalloc<float>(cos_tab.size()), *dsin = dalloc<float>(sin_tab.size());
    int32_t* dpos = dalloc<int32_t>(pos.size());
    up(dq, qfull); up(da, attn); up(dwn, wn); up(dcos, cos_tab); up(dsin, sin_tab); up(dpos, pos);
    const size_t qn = (size_t) MAXT * NH * HD;
    float *q1 = dalloc<float>(qn), *q2 = dalloc<float>(qn), *g1 = dalloc<float>(qn), *g2 = dalloc<float>(qn);
    // the rows of `rows_tok` tokens starting at token t, exactly as the window records them
    auto rows = [&](float* qc, float* gc, int t, int rows_tok, bool native) {
        const int r = rows_tok * NH;
        ck(cudaMemcpy2DAsync(qc + (size_t) t * NH * HD, (size_t) HD * 4, dq + (size_t) t * NH * 2 * HD, (size_t) HD * 2 * 4,
                             (size_t) HD * 4, (size_t) r, cudaMemcpyDeviceToDevice, g_cs), "split");
        float* x = qc + (size_t) t * NH * HD;
        if (native) {
            k::native_qsa_rms_norm_weighted(x, dwn, x, HD, r, k::qsa_rms_eps(), g_cs);
            k::native_rope_apply(x, x, r, HD, NR, (float) k::qsa_freq_base(), dpos + t * NH, g_cs);
        } else {
            k::rms_norm_weighted(x, dwn, r, HD, k::qsa_rms_eps(), g_cs);
            k::rope_neox_apply(x, x, r, HD, NR, dcos, dsin, dpos + t * NH, g_cs);
        }
        k::fwht256_inplace_cuda(x, r, g_cs);
        if (native) {
            k::native_qsa_gate_apply(da + (size_t) t * NH * HD, dq + (size_t) t * NH * 2 * HD, gc + (size_t) t * NH * HD, r,
                                     HD, g_cs);
        } else {
            k::QsaShapes sg = s;
            sg.n_head = s.n_head * rows_tok;
            k::qsa_gate_apply_f32(da + (size_t) t * NH * HD, dq + (size_t) t * NH * 2 * HD, sg, gc + (size_t) t * NH * HD,
                                  g_cs);
        }
    };
    for (int native = 0; native < 2; ++native) {
        bool ok = true;
        for (int n : {1, 2, 5, 8}) {
            for (float* p : {q1, q2, g1, g2}) ck(cudaMemset(p, 0, qn * 4), "m");
            const int tb = n == 8 ? 0 : 1;   // a group need not start at token 0 (the split window's B)
            for (int t = tb; t < tb + n && t < MAXT; ++t) rows(q1, g1, t, 1, native != 0);
            rows(q2, g2, tb, std::min(n, MAXT - tb), native != 0);
            ok = ok && same(down(q1, qn), down(q2, qn)) && same(down(g1, qn), down(g2, qn));
        }
        check(ok, std::string(native ? "native" : "legacy") + " norm/rotation/FWHT and output gate, groups of 1..8");
    }
    cudaFree(dq); cudaFree(da); cudaFree(dwn); cudaFree(dcos); cudaFree(dsin); cudaFree(dpos);
    cudaFree(q1); cudaFree(q2); cudaFree(g1); cudaFree(g2);
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::strcmp(argv[1], "--selftest") == 0;
    ck(cudaSetDeviceFlags(cudaDeviceMapHost), "device flags");
    // a BLOCKING stream: the plain cudaMemset/cudaMemcpy calls below (legacy default stream) stay ordered with it
    ck(cudaStreamCreate(&g_cs), "stream");
    std::mt19937 rng(20260928);
    try {
        test_mmvf(rng);
        test_router(rng);
        test_combine(rng);
        test_shared(rng);
        test_qsa_rows(rng);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "window_batch_parity: %s\n", e.what());
        return 2;
    }
    ck(cudaStreamSynchronize(g_cs), "final sync");
    std::printf("\nwindow_batch_parity: %d failures\n", g_fail);
    if (g_fail) return 1;
    if (selftest) std::printf("window_batch_parity OK\n");
    return 0;
}
