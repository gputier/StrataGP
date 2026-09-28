// src/kernels/qsa_select_parity.cpp - the QSA block selection (qsa_select.hpp) and the fp16 indexer keys, on the
// device against host references (GPU, synthetic, no model).
//
//   1. qsa_block_scores against a double-precision host reference of the same formula (relu per indexer head,
//      summed; `dead` for block n_bid, +1e9 when the tail has cells), relative to the magnitude of the TERMS.
//   2. qsa_block_topk - the default kernel (O6b, #22) and the previous one (STRATA_OLD_TOPK) - against a host
//      selection that orders every CELL by (order key desc, index asc) and keeps `width` of them: EXACT, the two
//      kernels bitwise equal, and nothing written past `width`.  On the device's own scores and on adversarial
//      score vectors: ties across thousands of blocks (the budget of equal cells), all zero (a ReLU that fired
//      nowhere), -0 against +0, NaN, the +1e9 tail, empty tail blocks, and the identity below the width.
//   3. The fp16 shadow (O6, #21): both pooling kernels write `pooled16`/`dead16` equal to f16_from_f32 of every
//      fp32 row they write, and the fp32 rows stay bitwise what they are without the shadow;
//      qsa_block_scores_f16 equals qsa_block_scores on the fp32 keys rounded to fp16, bitwise.  How often the
//      fp16 keys select other cells is PRINTED only: on random keys it says nothing about a real prompt, which is
//      the measurement #21 asks for (STRATA_IDX_FP16_CHECK in the prompt path).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_bad = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void require(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-64s %-6s %s\n", name.c_str(), ok ? "ok" : "*** BAD ***", detail.c_str());
    if (!ok) ++g_bad;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        ck(cudaMalloc(&p, count * sizeof(T) + 64), "cudaMalloc");
        ck(cudaMemset(p, 0, count * sizeof(T) + 64), "cudaMemset");
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void put(const std::vector<T>& v) {
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }
    std::vector<T> get(size_t count) const {
        std::vector<T> v(count);
        ck(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
        return v;
    }
    std::vector<T> get() const { return get(n); }
};

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;

uint32_t order_key(float s) {   // the kernels' total order: -0 is +0, NaN is the lowest key
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct Query {
    int32_t n_kv, n_bid, width;
};

/// The selection by definition: every cell carries its block's key; order by (key desc, cell asc), keep `width`,
/// return them ascending.  Shares no code with either kernel's threshold search.
std::vector<int32_t> ref_select(const float* sc, const Query& q) {
    std::vector<int32_t> ids;
    if (q.n_kv <= q.width) {
        for (int32_t j = 0; j < q.n_kv; ++j) ids.push_back(j);
        return ids;
    }
    std::vector<std::pair<uint32_t, int32_t>> cells;
    for (int32_t b = 0; b <= q.n_bid; ++b) {
        const int w = b < q.n_bid ? R : q.n_kv - q.n_bid * R;
        for (int c = 0; c < w; ++c) cells.push_back({order_key(sc[b]), b * R + c});
    }
    std::stable_sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int32_t i = 0; i < q.width; ++i) ids.push_back(cells[(size_t) i].second);
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<int32_t> steps_of(const std::vector<Query>& qs) {
    std::vector<int32_t> st;
    for (const Query& q : qs) {
        st.push_back(q.n_kv - 1);
        st.push_back(q.n_kv);
        st.push_back(q.n_bid);
        st.push_back(q.width);
    }
    return st;
}

Query query(int32_t n_kv, const k::QsaShapes& s) {
    return Query{n_kv, n_kv / R, (int32_t) k::qsa_selection_width(n_kv, s)};
}

/// Both top-k kernels on `scores` [nq, max_blocks], each checked against the host selection; returns the fraction
/// of queries whose selections the two kernels wrote identically (must be 1).
void check_topk(const char* what, const std::vector<float>& scores, const std::vector<Query>& qs, int64_t max_blocks,
                int64_t cap, const k::QsaShapes& s) {
    const int64_t nq = (int64_t) qs.size();
    Dev<float> dsc(scores.size());
    dsc.put(scores);
    Dev<int32_t> dst((size_t) nq * k::kStepCount);
    dst.put(steps_of(qs));
    Dev<int32_t> dids((size_t) (nq * cap));
    const std::vector<int32_t> sentinel((size_t) (nq * cap), -7);
    std::vector<int32_t> got[2];
    for (int old = 0; old < 2; ++old) {
        k::qsa_block_topk_set_old(old != 0);
        dids.put(sentinel);
        k::qsa_block_topk(dsc.p, dst.p, nq, max_blocks, cap, s, dids.p, nullptr);
        ck(cudaDeviceSynchronize(), "qsa_block_topk");
        got[old] = dids.get((size_t) (nq * cap));
    }
    k::qsa_block_topk_set_old(false);
    int wrong[2] = {0, 0}, spill[2] = {0, 0};
    for (int64_t qi = 0; qi < nq; ++qi) {
        const std::vector<int32_t> want = ref_select(scores.data() + qi * max_blocks, qs[(size_t) qi]);
        for (int old = 0; old < 2; ++old) {
            const int32_t* g = got[old].data() + qi * cap;
            if (!std::equal(want.begin(), want.end(), g)) ++wrong[old];
            for (int64_t j = (int64_t) want.size(); j < cap; ++j)
                if (g[j] != -7) { ++spill[old]; break; }
        }
    }
    const std::string n = std::to_string(nq) + " queries";
    require(std::string(what) + ": default top-k = host selection", wrong[0] == 0 && spill[0] == 0,
            std::to_string(wrong[0]) + " wrong, " + std::to_string(spill[0]) + " wrote past width, of " + n);
    require(std::string(what) + ": previous top-k = host selection", wrong[1] == 0 && spill[1] == 0,
            std::to_string(wrong[1]) + " wrong, " + std::to_string(spill[1]) + " wrote past width, of " + n);
    require(std::string(what) + ": the two kernels are bitwise equal", got[0] == got[1]);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t MAXC = 131072 + 64;                         // 128K of context, and a little
    const int64_t max_blocks = MAXC / R + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(20260928u);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::printf("qsa_select_parity: block scores, block top-k (default and previous), fp16 indexer keys\n");

    // ---- the queries: identity below the width, just above it, every tail length, 32K and 128K
    std::vector<Query> qs;
    for (int32_t n : {1, 3, 7, 2051, 2052, 2053, 2054, 2055, 2056, 4096, 4099, 10001, 32768, 32771, 65538, 131072,
                      131071})
        qs.push_back(query(n, s));
    const int64_t nq = (int64_t) qs.size();

    // ================= 1. block scores =================
    std::printf("\n-- 1. qsa_block_scores against a host reference\n");
    std::vector<float> pooled((size_t) max_blocks * IDX_DIM), dead(IDX_DIM), q_idx((size_t) nq * IDX_HEADS * IDX_DIM);
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    for (auto& x : q_idx) x = nd(rng) * 0.3f;
    Dev<float> dpooled(pooled.size()), ddead(dead.size()), dq(q_idx.size()), dscores((size_t) (nq * max_blocks));
    dpooled.put(pooled);
    ddead.put(dead);
    dq.put(q_idx);
    Dev<int32_t> dsteps((size_t) nq * k::kStepCount);
    dsteps.put(steps_of(qs));
    k::qsa_block_scores(dpooled.p, ddead.p, dq.p, dsteps.p, nq, max_blocks, s, dscores.p, nullptr);
    ck(cudaDeviceSynchronize(), "qsa_block_scores");
    const std::vector<float> scores = dscores.get((size_t) (nq * max_blocks));
    {
        double worst = 0;
        int tail_bad = 0;
        for (int64_t qi = 0; qi < nq; ++qi) {
            const Query& q = qs[(size_t) qi];
            for (int32_t b = 0; b <= q.n_bid; ++b) {
                const float* key = b == q.n_bid ? dead.data() : pooled.data() + (size_t) b * IDX_DIM;
                double want = 0, terms = 0;
                for (int h = 0; h < IDX_HEADS; ++h) {
                    double d = 0;
                    for (int i = 0; i < IDX_DIM; ++i) {
                        const double t = (double) key[i] * (double) q_idx[((size_t) qi * IDX_HEADS + h) * IDX_DIM + i];
                        d += t;
                        terms += std::fabs(t);
                    }
                    want += d > 0 ? d : 0;
                }
                const bool tail = b == q.n_bid && q.n_kv % R != 0;
                const double got = scores[(size_t) (qi * max_blocks + b)];
                if (tail) {   // +1e9 in f32: a multiple of 64 there
                    if (std::fabs(got - (want + 1e9)) > 128.0) ++tail_bad;
                    continue;
                }
                worst = std::max(worst, std::fabs(got - want) / std::max(terms, 1e-30));
            }
        }
        require("scores within 1e-6 of the terms (fp32 dot of 128)", worst < 1e-6, "worst " + std::to_string(worst));
        require("the tail block scores dead + 1e9 exactly when it has cells", tail_bad == 0,
                std::to_string(tail_bad) + " wrong");
    }

    // ================= 2. block top-k =================
    std::printf("\n-- 2. qsa_block_topk, default and previous kernel, against the host selection\n");
    check_topk("device scores", scores, qs, max_blocks, cap, s);
    {
        std::vector<float> sc((size_t) (nq * max_blocks), 0.0f);
        std::uniform_int_distribution<int> small(0, 3);
        // ties: four values over all blocks, so the threshold class spans thousands of blocks
        for (int64_t qi = 0; qi < nq; ++qi)
            for (int32_t b = 0; b <= qs[(size_t) qi].n_bid; ++b) sc[(size_t) (qi * max_blocks + b)] = (float) small(rng);
        check_topk("four-valued scores (ties)", sc, qs, max_blocks, cap, s);
        // every score zero: the whole selection is the equal-cells budget, lowest cells first
        std::fill(sc.begin(), sc.end(), 0.0f);
        check_topk("all zero", sc, qs, max_blocks, cap, s);
        // -0 and +0 are one key; NaN is the lowest; negative values below zero; the +1e9 tail
        for (int64_t qi = 0; qi < nq; ++qi) {
            const Query& q = qs[(size_t) qi];
            for (int32_t b = 0; b <= q.n_bid; ++b) {
                const int r = (int) (rng() % 8);
                float v = r == 0 ? -0.0f : r == 1 ? 0.0f : r == 2 ? std::numeric_limits<float>::quiet_NaN()
                        : r == 3 ? -std::fabs(nd(rng)) : r == 4 ? 1e-40f : std::fabs(nd(rng)) * 10.0f;
                if (b == q.n_bid && q.n_kv % R != 0) v += 1e9f;
                sc[(size_t) (qi * max_blocks + b)] = v;
            }
        }
        check_topk("-0 / +0 / NaN / negative / subnormal / +1e9 tail", sc, qs, max_blocks, cap, s);
        // a few huge scores and a flat rest: the threshold lands in the flat rest after pass one
        for (int64_t qi = 0; qi < nq; ++qi)
            for (int32_t b = 0; b <= qs[(size_t) qi].n_bid; ++b)
                sc[(size_t) (qi * max_blocks + b)] = (rng() % 97 == 0) ? 1e6f + (float) (rng() % 1000) : 1.5f;
        check_topk("spikes over a flat floor", sc, qs, max_blocks, cap, s);
        // close keys: the last byte decides (pass four)
        for (int64_t qi = 0; qi < nq; ++qi)
            for (int32_t b = 0; b <= qs[(size_t) qi].n_bid; ++b) {
                uint32_t bits = 0x3F800000u + (uint32_t) (rng() % 300);
                float v;
                std::memcpy(&v, &bits, 4);
                sc[(size_t) (qi * max_blocks + b)] = v;
            }
        check_topk("keys that differ in the last byte only", sc, qs, max_blocks, cap, s);
    }

    // ================= 3. the fp16 shadow =================
    std::printf("\n-- 3. the fp16 shadow of the pooled keys (O6)\n");
    {
        // (a) qsa_block_scores_f16 == qsa_block_scores on the keys rounded to fp16, bitwise
        std::vector<uint16_t> p16(pooled.size()), d16(dead.size());
        std::vector<float> pr(pooled.size()), dr(dead.size());
        for (size_t i = 0; i < pooled.size(); ++i) { p16[i] = k::f16_from_f32(pooled[i]); pr[i] = k::f32_from_f16(p16[i]); }
        for (size_t i = 0; i < dead.size(); ++i) { d16[i] = k::f16_from_f32(dead[i]); dr[i] = k::f32_from_f16(d16[i]); }
        Dev<uint16_t> dp16(p16.size()), dd16(d16.size());
        dp16.put(p16);
        dd16.put(d16);
        Dev<float> dpr(pr.size()), ddr(dr.size()), dsc16((size_t) (nq * max_blocks)), dscr((size_t) (nq * max_blocks));
        dpr.put(pr);
        ddr.put(dr);
        k::qsa_block_scores_f16(dp16.p, dd16.p, dq.p, dsteps.p, nq, max_blocks, s, dsc16.p, nullptr);
        k::qsa_block_scores(dpr.p, ddr.p, dq.p, dsteps.p, nq, max_blocks, s, dscr.p, nullptr);
        ck(cudaDeviceSynchronize(), "scores f16");
        const std::vector<float> a = dsc16.get(), b = dscr.get();
        require("fp16 scores = fp32 scores of the rounded keys, bitwise",
                std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
        // the selection agreement on random keys (printed, not judged)
        Dev<int32_t> ids32((size_t) (nq * cap)), ids16((size_t) (nq * cap));
        k::qsa_block_topk(dscores.p, dsteps.p, nq, max_blocks, cap, s, ids32.p, nullptr);
        k::qsa_block_topk(dsc16.p, dsteps.p, nq, max_blocks, cap, s, ids16.p, nullptr);
        ck(cudaDeviceSynchronize(), "topk f16");
        const std::vector<int32_t> x = ids32.get(), y = ids16.get();
        long long shared = 0, cells = 0;
        for (int64_t qi = 0; qi < nq; ++qi) {
            const Query& q = qs[(size_t) qi];
            if (q.n_kv <= q.width) continue;
            const int32_t *u = x.data() + qi * cap, *v = y.data() + qi * cap;
            int64_t i = 0, j = 0;
            while (i < q.width && j < q.width) {
                if (u[i] == v[j]) { ++shared; ++i; ++j; } else if (u[i] < v[j]) ++i; else ++j;
            }
            cells += q.width;
        }
        std::printf("  %-64s %.3f%% of %lld cells (random keys: not a model measurement)\n",
                    "fp16 keys select the same cells as fp32", cells ? 100.0 * (double) shared / (double) cells : 100.0,
                    cells);
    }
    {
        // (b) both pooling kernels: the shadow is f16 of every row, and the fp32 rows do not move
        const int64_t NT = 4 * 60 + 3, MAXCELLS = 1024, rows = MAXCELLS / R + 2;
        std::vector<float> raw((size_t) NT * IDX_DIM), w_kn(IDX_DIM);
        for (auto& x : raw) x = nd(rng) * 3.0f;
        for (auto& x : w_kn) x = 1.0f + 0.1f * nd(rng);
        std::vector<float> cos_tab((size_t) MAXCELLS * s.n_rot / 2), sin_tab(cos_tab.size());
        k::build_rope_table((int) s.n_rot, k::qsa_freq_base(), (int) MAXCELLS, cos_tab.data(), sin_tab.data());
        Dev<float> dcos(cos_tab.size()), dsin(sin_tab.size()), dw(w_kn.size()), draw(raw.size());
        dcos.put(cos_tab);
        dsin.put(sin_tab);
        dw.put(w_kn);
        draw.put(raw);
        Dev<int32_t> dpos((size_t) NT);
        {
            std::vector<int32_t> pos((size_t) NT);
            for (int64_t t = 0; t < NT; ++t) pos[(size_t) t] = (int32_t) t;
            dpos.put(pos);
        }
        cudaStream_t cs;
        ck(cudaStreamCreate(&cs), "stream");
        for (int native = 0; native < 2; ++native) {
            std::vector<float> pooled_of[2], dead_of[2];
            std::vector<uint16_t> p16, d16;
            for (int shadow = 0; shadow < 2; ++shadow) {
                Dev<float> tail((size_t) (R - 1) * IDX_DIM), dd(IDX_DIM), pp((size_t) rows * IDX_DIM);
                Dev<int32_t> bpos(1);
                Dev<uint16_t> pp16((size_t) rows * IDX_DIM), dd16(IDX_DIM);
                k::QsaIndexerBuffers ib{tail.p, dd.p, pp.p, bpos.p};
                if (shadow) { ib.pooled16 = pp16.p; ib.dead16 = dd16.p; }
                for (int64_t t = 0; t < NT; ++t) {
                    if (native)
                        k::native_qsa_indexer_append(draw.p + t * IDX_DIM, dpos.p + t, 0, dw.p, k::qsa_rms_eps(), ib,
                                                     s, MAXCELLS, (float) k::qsa_freq_base(), cs);
                    else
                        k::indexer_key_append(draw.p + t * IDX_DIM, dpos.p + t, 0, dw.p, k::qsa_rms_eps(), ib, s,
                                              dcos.p, dsin.p, cs);
                }
                ck(cudaStreamSynchronize(cs), "pooling");
                pooled_of[shadow] = pp.get();
                dead_of[shadow] = dd.get();
                if (shadow) { p16 = pp16.get(); d16 = dd16.get(); }
            }
            const char* who = native ? "native_qsa_indexer_append" : "indexer_key_append";
            require(std::string(who) + ": fp32 rows unchanged by the shadow (bitwise)",
                    std::memcmp(pooled_of[0].data(), pooled_of[1].data(), pooled_of[0].size() * 4) == 0 &&
                        std::memcmp(dead_of[0].data(), dead_of[1].data(), IDX_DIM * 4) == 0);
            int bad = 0;
            for (size_t i = 0; i < p16.size(); ++i) bad += p16[i] != k::f16_from_f32(pooled_of[1][i]);
            for (size_t i = 0; i < d16.size(); ++i) bad += d16[i] != k::f16_from_f32(dead_of[1][i]);
            const int64_t n_bid = NT / R;
            bool nonzero = false;
            for (int64_t i = 0; i < (n_bid + 1) * IDX_DIM; ++i) nonzero |= pooled_of[1][(size_t) i] != 0.0f;
            require(std::string(who) + ": shadow = f16_from_f32 of every row and of dead", bad == 0 && nonzero,
                    std::to_string(bad) + " values differ");
        }
        cudaStreamDestroy(cs);
    }

    std::printf(g_bad ? "\nFAIL (%d)\n" : "\nPASS\n", g_bad);
    return g_bad ? 1 : 0;
}
