// src/kernels/qsa_prep_parity.cpp - O6e/O6d: the fused QSA preparation (qsa_prep.hpp) against the separate launches
// it replaces, BITWISE (GPU, synthetic, no model).
//
// For every KV format (FP16, INT8, Q4_0), every arithmetic variant (canonical or native norm x table or native
// rotation), with and without a KV-streaming host copy (and non-resident pages), with and without the image position
// table, for one token and for a batch of three:
//   old: norm(K) ; rope(K) ; [fwht K ; fwht V] ; kv_append*_step           per token, as qsa_layer and verify.cpp do
//        cudaMemcpy2D(q) ; norm(q) ; rope(q) ; [fwht q] ; norm(q_idx) ; rope(q_idx)
//   new: kv_append*_prep_step ; qsa_q_prep                                  one launch each
// and every byte of the pools, the host copies, kcur/vcur, qcur and q_idx must match. The rows include adversarial
// ones - squares below FLT_MIN (the native norm flushes them, the canonical one does not), sub-normal inputs, zero
// rows, a lone non-zero, sums that overflow - because that is where an FTZ or fused-multiply-add mismatch shows.
// Also copy_i32x2_from_mapped (O6d) against its mapped sources.
//
// `--bench` then times one QSA layer's preparation both ways, captured as a graph of twelve layers the way the engine
// replays them (native norm and rotation, one token), and the per-token step upload both ways.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_prep.hpp"
#include "strata/kernels/rope.hpp"

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

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    return p;
}

template <typename T>
std::vector<uint8_t> bytes_of(const T* d, size_t n) {
    std::vector<uint8_t> h(n * sizeof(T));
    ck(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

/// Compares two device arrays byte for byte; reports the first difference.
template <typename T>
void same(const char* what, const std::string& cfg, const T* a, const T* b, size_t n) {
    const std::vector<uint8_t> x = bytes_of(a, n), y = bytes_of(b, n);
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {
            if (g_fail < 20)
                std::fprintf(stderr, "FAIL %s [%s]: byte %zu of %zu differs (old 0x%02x, fused 0x%02x)\n", what,
                             cfg.c_str(), i, x.size(), x[i], y[i]);
            ++g_fail;
            return;
        }
    }
}

/// One row of `n` values of kind `kind`: the realistic case and the adversarial ones.
void fill_row(float* r, int n, int kind, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) r[i] = nd(rng);
    switch (kind) {
    case 1: for (int i = 0; i < n; ++i) r[i] *= 1e-21f; break;            // squares far below FLT_MIN
    case 2: for (int i = 0; i < n; ++i) r[i] = 0.0f; break;               // a zero row
    case 3: for (int i = 0; i < n; ++i) r[i] = i == 17 ? 3.5f : 0.0f; break;   // a lone non-zero
    case 4: for (int i = 0; i < n; ++i) r[i] *= 1e19f; break;             // the sum of squares overflows
    case 5: for (int i = 0; i < n; i += 3) r[i] = (i & 1 ? 1.0f : -1.0f) * 1e-40f; break;   // sub-normal inputs
    case 6: for (int i = 0; i < n; ++i) r[i] *= (i % 5 == 0) ? 1e3f : 1e-3f; break;        // mixed magnitudes
    default: break;
    }
}

struct Pools {
    uint16_t *kp = nullptr, *vp = nullptr;
    int8_t *kq = nullptr, *vq = nullptr;
    uint16_t *ks = nullptr, *vs = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    size_t n_f16 = 0, n_q8 = 0, n_sc = 0, n_q4 = 0;
    void alloc(size_t rows, int D) {
        n_f16 = rows * D;
        n_q8 = rows * D;
        n_sc = rows * (D / k::KV_Q8_GROUP);
        n_q4 = rows * k::kv_q4_bytes_per_head(D);
        kp = dalloc<uint16_t>(n_f16); vp = dalloc<uint16_t>(n_f16);
        kq = dalloc<int8_t>(n_q8); vq = dalloc<int8_t>(n_q8);
        ks = dalloc<uint16_t>(n_sc); vs = dalloc<uint16_t>(n_sc);
        k4 = dalloc<uint8_t>(n_q4); v4 = dalloc<uint8_t>(n_q4);
    }
    void poison() {   // the same pattern on both sides, so the cells nobody writes compare too
        ck(cudaMemset(kp, 0xA5, n_f16 * 2), "poison"); ck(cudaMemset(vp, 0xA5, n_f16 * 2), "poison");
        ck(cudaMemset(kq, 0xA5, n_q8), "poison"); ck(cudaMemset(vq, 0xA5, n_q8), "poison");
        ck(cudaMemset(ks, 0xA5, n_sc * 2), "poison"); ck(cudaMemset(vs, 0xA5, n_sc * 2), "poison");
        ck(cudaMemset(k4, 0xA5, n_q4), "poison"); ck(cudaMemset(v4, 0xA5, n_q4), "poison");
    }
    k::KvHostPools host() const {
        k::KvHostPools h;
        h.k_pool = kp; h.v_pool = vp; h.k_q = kq; h.v_q = vq; h.k_scale = ks; h.v_scale = vs; h.k_q4 = k4; h.v_q4 = v4;
        return h;
    }
};

void same_pools(const std::string& cfg, const char* tag, const Pools& a, const Pools& b, int fmt) {
    const std::string c = cfg + " " + tag;
    if (fmt == 0) {
        same("K fp16", c, a.kp, b.kp, a.n_f16);
        same("V fp16", c, a.vp, b.vp, a.n_f16);
    } else if (fmt == 1) {
        same("K int8", c, a.kq, b.kq, a.n_q8);
        same("V int8", c, a.vq, b.vq, a.n_q8);
        same("K scales", c, a.ks, b.ks, a.n_sc);
        same("V scales", c, a.vs, b.vs, a.n_sc);
    } else {
        same("K q4_0", c, a.k4, b.k4, a.n_q4);
        same("V q4_0", c, a.v4, b.v4, a.n_q4);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: qsa_prep_parity [--bench]\n"); return 2; }
    }
    k::QsaShapes s = k::qsa_real_shapes();
    const int NH = (int) s.n_head, NKV = (int) s.n_head_kv, HD = (int) s.head_dim, IQ = (int) s.idx_n_head,
              ID = (int) s.idx_dim, NROT = (int) s.n_rot, P = (int) s.page_size;
    const float EPS = k::qsa_rms_eps();
    const float FREQ = (float) k::qsa_freq_base();
    const int max_t = 3, cells = 1024, pages = cells / P;
    const int max_pos = 131072;   // rope positions up to here (the table's rows), cells are separate
    std::mt19937 rng(20260928);

    cudaStream_t cs = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");

    // ---- which geometries the fused path takes
    {
        k::QsaNormRope nr;
        nr.eps = EPS;
        nr.native_rope = true;
        nr.freq_base = FREQ;
        if (!k::qsa_prep_supported(s, nr)) { std::fprintf(stderr, "FAIL: the real shapes are not supported\n"); ++g_fail; }
        k::QsaShapes bad = s;
        bad.head_dim = 128;
        if (k::qsa_prep_supported(bad, nr)) { std::fprintf(stderr, "FAIL: head_dim 128 claimed supported\n"); ++g_fail; }
        bad = s;
        bad.idx_n_head = s.n_head + 1;
        if (k::qsa_prep_supported(bad, nr)) { std::fprintf(stderr, "FAIL: idx_n_head > n_head claimed supported\n"); ++g_fail; }
        nr.native_rope = false;
        if (k::qsa_prep_supported(s, nr)) { std::fprintf(stderr, "FAIL: the table rotation without a table\n"); ++g_fail; }
    }

    // ---- the RoPE table and the image position table
    std::vector<float> hcos((size_t) max_pos * (NROT / 2)), hsin(hcos.size());
    k::build_rope_table(NROT, k::qsa_freq_base(), max_pos, hcos.data(), hsin.data());
    float* d_cos = dalloc<float>(hcos.size());
    float* d_sin = dalloc<float>(hsin.size());
    ck(cudaMemcpy(d_cos, hcos.data(), hcos.size() * 4, cudaMemcpyHostToDevice), "cos");
    ck(cudaMemcpy(d_sin, hsin.data(), hsin.size() * 4, cudaMemcpyHostToDevice), "sin");
    std::vector<int32_t> hmt((size_t) max_pos * 3);
    for (int c = 0; c < max_pos; ++c) {   // t, h, w: an image-like spread that stays inside the table
        hmt[(size_t) c * 3] = c;
        hmt[(size_t) c * 3 + 1] = std::min(max_pos - 1, c + c % 7);
        hmt[(size_t) c * 3 + 2] = std::max(0, c - c % 5);
    }
    int32_t* d_mt = dalloc<int32_t>(hmt.size());
    ck(cudaMemcpy(d_mt, hmt.data(), hmt.size() * 4, cudaMemcpyHostToDevice), "mrope");

    // ---- buffers: the old path's (a) and the fused path's (b)
    const size_t rows = (size_t) pages * NKV * P;
    Pools va, vb, ha, hb;   // VRAM pools and host copies
    va.alloc(rows, HD); vb.alloc(rows, HD); ha.alloc(rows, HD); hb.alloc(rows, HD);
    int32_t* d_table = dalloc<int32_t>(pages);
    int32_t* d_step = dalloc<int32_t>((size_t) max_t * k::kStepCount);
    int32_t* d_pos = dalloc<int32_t>((size_t) max_t * NH);
    float *kin = dalloc<float>((size_t) max_t * NKV * HD), *vin = dalloc<float>((size_t) max_t * NKV * HD);
    float *ka = dalloc<float>((size_t) max_t * NKV * HD), *kb = dalloc<float>((size_t) max_t * NKV * HD);
    float *vva = dalloc<float>((size_t) max_t * NKV * HD), *vvb = dalloc<float>((size_t) max_t * NKV * HD);
    float* qfull = dalloc<float>((size_t) max_t * NH * 2 * HD);
    float *qa = dalloc<float>((size_t) max_t * NH * HD), *qb = dalloc<float>((size_t) max_t * NH * HD);
    float* xin = dalloc<float>((size_t) max_t * IQ * ID);
    float *xa = dalloc<float>((size_t) max_t * IQ * ID), *xb = dalloc<float>((size_t) max_t * IQ * ID);
    float *kg = dalloc<float>(HD), *qg = dalloc<float>(HD), *ig = dalloc<float>(ID);

    long configs = 0;
    for (int trial = 0; trial < 4; ++trial) {
        // this trial's data: adversarial row kinds spread over tokens, heads and the three row sets
        std::vector<float> h((size_t) max_t * NKV * HD), hv(h.size()), hq((size_t) max_t * NH * 2 * HD),
            hx((size_t) max_t * IQ * ID), hkg(HD), hqg(HD), hig(ID);
        for (int r = 0; r < max_t * NKV; ++r) {
            fill_row(h.data() + (size_t) r * HD, HD, (r + trial * 3) % 7, rng);
            fill_row(hv.data() + (size_t) r * HD, HD, (r + trial * 5 + 2) % 7, rng);
        }
        for (int r = 0; r < max_t * NH; ++r) fill_row(hq.data() + (size_t) r * 2 * HD, 2 * HD, (r + trial) % 7, rng);
        for (int r = 0; r < max_t * IQ; ++r) fill_row(hx.data() + (size_t) r * ID, ID, (r + trial * 2) % 7, rng);
        std::normal_distribution<float> gd(1.0f, 0.3f);
        for (auto* g : {&hkg, &hqg, &hig})
            for (size_t i = 0; i < g->size(); ++i) (*g)[i] = (i % 61 == 7) ? 0.0f : (i % 53 == 3 ? 1e-30f : gd(rng));
        ck(cudaMemcpy(kin, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vin, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(qfull, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        ck(cudaMemcpy(xin, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMemcpy(kg, hkg.data(), HD * 4, cudaMemcpyHostToDevice), "kg");
        ck(cudaMemcpy(qg, hqg.data(), HD * 4, cudaMemcpyHostToDevice), "qg");
        ck(cudaMemcpy(ig, hig.data(), ID * 4, cudaMemcpyHostToDevice), "ig");
        // distinct cells, rope positions anywhere in the table (as a cell index when the image table is on)
        std::vector<int> cell(cells);
        for (int i = 0; i < cells; ++i) cell[i] = i;
        std::shuffle(cell.begin(), cell.end(), rng);
        std::vector<int32_t> hstep((size_t) max_t * k::kStepCount, 0), hpos((size_t) max_t * NH);
        for (int t = 0; t < max_t; ++t) {
            hstep[(size_t) t * k::kStepCount + k::kStepPos] = cell[t];
            hstep[(size_t) t * k::kStepCount + k::kStepNKv] = cell[t] + 1;
            const int32_t rp = trial == 0 && t == 0 ? 0 : (int32_t) (rng() % (max_pos - 8));
            for (int hh = 0; hh < NH; ++hh) hpos[(size_t) t * NH + hh] = rp;
        }
        ck(cudaMemcpy(d_step, hstep.data(), hstep.size() * 4, cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(d_pos, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice), "pos");

        for (int fmt = 0; fmt < 3; ++fmt)
            for (int variant = 0; variant < 4; ++variant)
                for (int streamed = 0; streamed < 2; ++streamed)
                    for (int image = 0; image < 2; ++image)
                        for (int n_tok : {1, max_t}) {
                            const bool native_norm = variant & 2, native_rope = variant & 1;
                            char buf[160];
                            std::snprintf(buf, sizeof buf, "trial %d, %s KV, %s norm, %s rope, %s, %s, %d token(s)",
                                          trial, fmt == 0 ? "fp16" : fmt == 1 ? "int8" : "q4_0",
                                          native_norm ? "native" : "canonical", native_rope ? "native" : "table",
                                          streamed ? "streamed" : "resident", image ? "image positions" : "text",
                                          n_tok);
                            const std::string cfg = buf;
                            ++configs;
                            // the page table: a permutation, with non-resident blocks when there is a host copy
                            std::vector<int32_t> table(pages);
                            for (int i = 0; i < pages; ++i) table[i] = (i * 37 + 11 + trial) % pages;
                            if (streamed)
                                for (int i = 0; i < pages; i += 3) table[i] = -1;
                            ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");
                            k::mrope_table_set(image ? d_mt : nullptr);
                            va.poison(); vb.poison(); ha.poison(); hb.poison();
                            const size_t nkv = (size_t) n_tok * NKV * HD;
                            ck(cudaMemcpy(ka, kin, nkv * 4, cudaMemcpyDeviceToDevice), "ka");
                            ck(cudaMemcpy(kb, kin, nkv * 4, cudaMemcpyDeviceToDevice), "kb");
                            ck(cudaMemcpy(vva, vin, nkv * 4, cudaMemcpyDeviceToDevice), "va");
                            ck(cudaMemcpy(vvb, vin, nkv * 4, cudaMemcpyDeviceToDevice), "vb");
                            ck(cudaMemcpy(xa, xin, (size_t) n_tok * IQ * ID * 4, cudaMemcpyDeviceToDevice), "xa");
                            ck(cudaMemcpy(xb, xin, (size_t) n_tok * IQ * ID * 4, cudaMemcpyDeviceToDevice), "xb");
                            ck(cudaMemset(qa, 0, (size_t) n_tok * NH * HD * 4), "qa");
                            ck(cudaMemset(qb, 0, (size_t) n_tok * NH * HD * 4), "qb");
                            const k::KvHostPools host_a = streamed ? ha.host() : k::KvHostPools{};
                            const k::KvHostPools host_b = streamed ? hb.host() : k::KvHostPools{};
                            // the poison, the copies and the memsets above run on the legacy stream, which the
                            // non-blocking `cs` does not wait for
                            ck(cudaDeviceSynchronize(), "setup");

                            // ---- the separate launches, token by token
                            auto norm_rope = [&](float* d, const float* g, int nrows, int ncols, const int32_t* p) {
                                if (native_norm) k::native_qsa_rms_norm_weighted(d, g, d, ncols, nrows, EPS, cs);
                                else k::rms_norm_weighted(d, g, nrows, ncols, EPS, cs);
                                if (native_rope) k::native_rope_apply(d, d, nrows, ncols, NROT, FREQ, p, cs);
                                else k::rope_neox_apply(d, d, nrows, ncols, NROT, d_cos, d_sin, p, cs);
                            };
                            for (int t = 0; t < n_tok; ++t) {
                                float* kt = ka + (size_t) t * NKV * HD;
                                float* vt = vva + (size_t) t * NKV * HD;
                                const int32_t* st = d_step + (size_t) t * k::kStepCount;
                                const int32_t* pt = d_pos + (size_t) t * NH;
                                norm_rope(kt, kg, NKV, HD, pt);
                                if (fmt == 2) {
                                    k::fwht256_inplace_cuda(kt, NKV, cs);
                                    k::fwht256_inplace_cuda(vt, NKV, cs);
                                    k::kv_append_q4_step(va.k4, va.v4, d_table, st, kt, vt, s, cs, &host_a);
                                } else if (fmt == 1) {
                                    k::kv_append_q8_step(va.kq, va.vq, va.ks, va.vs, d_table, st, kt, vt, s, cs, &host_a);
                                } else {
                                    k::kv_append_step(va.kp, va.vp, d_table, st, kt, vt, s, cs, &host_a);
                                }
                                float* qt = qa + (size_t) t * NH * HD;
                                ck(cudaMemcpy2DAsync(qt, (size_t) HD * 4, qfull + (size_t) t * NH * 2 * HD,
                                                     (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) NH,
                                                     cudaMemcpyDeviceToDevice, cs), "q split");
                                norm_rope(qt, qg, NH, HD, pt);
                                if (fmt == 2) k::fwht256_inplace_cuda(qt, NH, cs);
                                norm_rope(xa + (size_t) t * IQ * ID, ig, IQ, ID, pt);
                            }

                            // ---- the fused launches, all tokens at once (n_tok 1: stride 0, as qsa_layer passes)
                            k::QsaNormRope nr;
                            nr.eps = EPS;
                            nr.native_norm = native_norm;
                            nr.native_rope = native_rope;
                            nr.freq_base = FREQ;
                            nr.cos_tab = d_cos;
                            nr.sin_tab = d_sin;
                            const int64_t stride = n_tok == 1 ? 0 : NH;
                            if (fmt == 2)
                                k::kv_append_q4_prep_step(vb.k4, vb.v4, d_table, d_step, kb, vvb, kg, d_pos, stride,
                                                          n_tok, nr, s, cs, &host_b);
                            else if (fmt == 1)
                                k::kv_append_q8_prep_step(vb.kq, vb.vq, vb.ks, vb.vs, d_table, d_step, kb, vvb, kg,
                                                          d_pos, stride, n_tok, nr, s, cs, &host_b);
                            else
                                k::kv_append_prep_step(vb.kp, vb.vp, d_table, d_step, kb, vvb, kg, d_pos, stride,
                                                       n_tok, nr, s, cs, &host_b);
                            k::qsa_q_prep(qfull, qb, qg, xb, ig, d_pos, stride, n_tok, fmt == 2, nr, s, cs);
                            ck(cudaStreamSynchronize(cs), cfg.c_str());

                            same_pools(cfg, "VRAM", va, vb, fmt);
                            if (streamed) same_pools(cfg, "host copy", ha, hb, fmt);
                            same("kcur", cfg, ka, kb, nkv);
                            same("vcur", cfg, vva, vvb, nkv);
                            same("qcur", cfg, qa, qb, (size_t) n_tok * NH * HD);
                            same("q_idx", cfg, xa, xb, (size_t) n_tok * IQ * ID);
                        }
    }
    k::mrope_table_set(nullptr);

    // ---- O6d: the step and the positions from mapped host memory in one kernel
    {
        int32_t *h0 = nullptr, *h1 = nullptr, *m0 = nullptr, *m1 = nullptr;
        ck(cudaHostAlloc((void**) &h0, (k::kStepCount + 1) * 4, cudaHostAllocMapped | cudaHostAllocPortable), "pin");
        ck(cudaHostAlloc((void**) &h1, NH * 4, cudaHostAllocMapped | cudaHostAllocPortable), "pin");
        ck(cudaHostGetDevicePointer((void**) &m0, h0, 0), "map");
        ck(cudaHostGetDevicePointer((void**) &m1, h1, 0), "map");
        int32_t *d0 = dalloc<int32_t>(k::kStepCount), *d1 = dalloc<int32_t>(NH);
        for (int round = 0; round < 3; ++round) {   // the same launch re-reads the host values each time
            k::qsa_step_fill(h0, 1000 * round + 7, s);
            for (int i = 0; i < NH; ++i) h1[i] = 123456 + round * 1000 + i;
            k::copy_i32x2_from_mapped(d0, m0, k::kStepCount, d1, m1, NH, cs);
            ck(cudaStreamSynchronize(cs), "copy_i32x2_from_mapped");
            std::vector<int32_t> g0(k::kStepCount), g1(NH);
            ck(cudaMemcpy(g0.data(), d0, g0.size() * 4, cudaMemcpyDeviceToHost), "d0");
            ck(cudaMemcpy(g1.data(), d1, g1.size() * 4, cudaMemcpyDeviceToHost), "d1");
            if (std::memcmp(g0.data(), h0, g0.size() * 4) != 0 || std::memcmp(g1.data(), h1, g1.size() * 4) != 0) {
                std::fprintf(stderr, "FAIL: copy_i32x2_from_mapped round %d\n", round);
                ++g_fail;
            }
        }
        cudaFreeHost(h0);
        cudaFreeHost(h1);
    }

    // ---- --bench: the same work as graphs of 12 layers, replayed (what a token costs), microseconds per layer
    if (bench && g_fail == 0) {
        const int reps = 500, layers = 12;
        auto time_graph = [&](auto&& layer) -> double {
            cudaStream_t st = nullptr;
            ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "bench stream");
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
            for (int l = 0; l < layers; ++l) layer(st);
            cudaGraph_t graph = nullptr;
            ck(cudaStreamEndCapture(st, &graph), "end capture");
            cudaGraphExec_t exec = nullptr;
            ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            for (int r = 0; r < 20; ++r) ck(cudaGraphLaunch(exec, st), "warm-up");
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "event");
            ck(cudaEventCreate(&e1), "event");
            ck(cudaEventRecord(e0, st), "record");
            for (int r = 0; r < reps; ++r) ck(cudaGraphLaunch(exec, st), "launch");
            ck(cudaEventRecord(e1, st), "record");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(st);
            return 1000.0 * ms / (reps * layers);
        };
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = i;
        ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");
        k::QsaNormRope nr;
        nr.eps = EPS;
        nr.native_norm = nr.native_rope = true;   // what --native selects
        nr.freq_base = FREQ;
        std::printf("qsa_prep_parity --bench: one QSA layer, native norm and rotation, graphs of %d layers x %d:\n",
                    layers, reps);
        for (int fmt = 0; fmt < 3; ++fmt) {
            auto norm_rope = [&](cudaStream_t st, float* d, const float* g, int nrows, int ncols) {
                k::native_qsa_rms_norm_weighted(d, g, d, ncols, nrows, EPS, st);
                k::native_rope_apply(d, d, nrows, ncols, NROT, FREQ, d_pos, st);
            };
            auto separate = [&](cudaStream_t st) {
                norm_rope(st, ka, kg, NKV, HD);
                if (fmt == 2) {
                    k::fwht256_inplace_cuda(ka, NKV, st);
                    k::fwht256_inplace_cuda(vva, NKV, st);
                    k::kv_append_q4_step(va.k4, va.v4, d_table, d_step, ka, vva, s, st);
                } else if (fmt == 1) {
                    k::kv_append_q8_step(va.kq, va.vq, va.ks, va.vs, d_table, d_step, ka, vva, s, st);
                } else {
                    k::kv_append_step(va.kp, va.vp, d_table, d_step, ka, vva, s, st);
                }
                ck(cudaMemcpy2DAsync(qa, (size_t) HD * 4, qfull, (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) NH,
                                     cudaMemcpyDeviceToDevice, st), "q split");
                norm_rope(st, qa, qg, NH, HD);
                if (fmt == 2) k::fwht256_inplace_cuda(qa, NH, st);
                norm_rope(st, xa, ig, IQ, ID);
            };
            auto fused = [&](cudaStream_t st) {
                if (fmt == 2) k::kv_append_q4_prep_step(vb.k4, vb.v4, d_table, d_step, kb, vvb, kg, d_pos, 0, 1, nr, s, st);
                else if (fmt == 1)
                    k::kv_append_q8_prep_step(vb.kq, vb.vq, vb.ks, vb.vs, d_table, d_step, kb, vvb, kg, d_pos, 0, 1, nr, s, st);
                else k::kv_append_prep_step(vb.kp, vb.vp, d_table, d_step, kb, vvb, kg, d_pos, 0, 1, nr, s, st);
                k::qsa_q_prep(qfull, qb, qg, xb, ig, d_pos, 0, 1, fmt == 2, nr, s, st);
            };
            const double us_sep = time_graph(separate), us_fused = time_graph(fused);
            std::printf("  %-5s KV  separate %7.2f us   fused %7.2f us   saved %6.2f us per layer, %6.1f us per token\n",
                        fmt == 0 ? "fp16" : fmt == 1 ? "int8" : "q4_0", us_sep, us_fused, us_sep - us_fused,
                        (us_sep - us_fused) * layers);
        }
        // O6d: every QSA layer's two step/position copies against the first layer's one
        {
            int32_t *h0 = nullptr, *h1 = nullptr, *m0 = nullptr, *m1 = nullptr;
            ck(cudaHostAlloc((void**) &h0, (k::kStepCount + 1) * 4, cudaHostAllocMapped | cudaHostAllocPortable), "pin");
            ck(cudaHostAlloc((void**) &h1, NH * 4, cudaHostAllocMapped | cudaHostAllocPortable), "pin");
            ck(cudaHostGetDevicePointer((void**) &m0, h0, 0), "map");
            ck(cudaHostGetDevicePointer((void**) &m1, h1, 0), "map");
            k::qsa_step_fill(h0, 100, s);
            for (int i = 0; i < NH; ++i) h1[i] = 100;
            int32_t* d0 = dalloc<int32_t>((size_t) layers * k::kStepCount);
            int32_t* d1 = dalloc<int32_t>((size_t) layers * NH);
            int l_old = 0;
            const double us_old = time_graph([&](cudaStream_t st) {
                const int l = l_old++ % layers;
                k::copy_i32_from_mapped(d0 + (size_t) l * k::kStepCount, m0, k::kStepCount, st);
                k::copy_i32_from_mapped(d1 + (size_t) l * NH, m1, NH, st);
            });
            int l_new = 0;
            const double us_new = time_graph([&](cudaStream_t st) {
                if (l_new++ % layers == 0) k::copy_i32x2_from_mapped(d0, m0, k::kStepCount, d1, m1, NH, st);
            });
            std::printf("  step  upload  2 per layer %6.1f us per token   1 per token %6.1f us   saved %6.1f us per token\n",
                        us_old * layers, us_new * layers, (us_old - us_new) * layers);
            cudaFreeHost(h0);
            cudaFreeHost(h1);
        }
    }

    std::printf("qsa_prep_parity: %s (%ld configurations compared byte for byte)\n", g_fail ? "FAILED" : "OK", configs);
    return g_fail ? 1 : 0;
}
