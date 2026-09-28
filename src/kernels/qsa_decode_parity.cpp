// src/kernels/qsa_decode_parity.cpp - the decode-time QSA kernels against float64 (#8).
//
// `qsa_block_scores`, `qsa_block_topk` and `qsa_decode_attn` run in every QSA layer of every token and every prompt
// chunk, and the only test that touched them (`kv_stream_parity`) compares the attention kernel with ITSELF over two
// pool layouts.  Here each one is checked against a host transcription of its header's contract, on synthetic data at
// the artifact's geometry (24 query heads over 2 KV heads of 256, indexer 4 x 128, blocks of 4 cells, pages of 4):
//
//   1. block scores: sum over the 4 indexer heads of relu(q . key), the key being the pooled block or, for the
//      incomplete tail block n_bid, the `dead` key plus 1e9 when that block holds cells;
//   2. block top-k: the `width` cells of highest block score, each block weighted by its cell count, ties to the
//      LOWEST cell index, emitted ascending - EXACT ids, on the device's own scores and on scores built with ties
//      (equal values, +0 against -0), plus the identity below the selection width;
//   3. split-K attention over the selected cells (chunks of 64, then the log-sum-exp merge) for FP16, INT8 and Q4_0
//      pools read through a permuted page table, against float64 softmax attention over the same dequantized
//      values; the batched form must be BITWISE the per-query form.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

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
    check(cudaMalloc(&p, n * sizeof(T) + 16), "cudaMalloc");
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

/// `order_key` of qsa_select.cu: +0 and -0 equal, NaN lowest, otherwise the float order as unsigned.
uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/// The selection the header specifies, from the block scores.
std::vector<int32_t> ref_select(const float* sc, int64_t n_kv, int64_t n_bid, int64_t width) {
    std::vector<int32_t> out;
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j) out.push_back((int32_t) j);
        return out;
    }
    std::vector<std::pair<uint32_t, int32_t>> cells;          // (key, cell)
    for (int64_t b = 0; b <= n_bid; ++b) {
        const int64_t w = b < n_bid ? 4 : n_kv - n_bid * 4;
        for (int64_t c = 0; c < w; ++c) cells.emplace_back(order_key(sc[b]), (int32_t) (b * 4 + c));
    }
    std::stable_sort(cells.begin(), cells.end(), [](const auto& x, const auto& y) {
        return x.first != y.first ? x.first > y.first : x.second < y.second;
    });
    for (int64_t i = 0; i < width; ++i) out.push_back(cells[(size_t) i].second);
    std::sort(out.begin(), out.end());
    return out;
}

// ================================ 1-2. block scores and the top-k ================================

int select_tests() {
    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(31);
    std::normal_distribution<float> g(0.0f, 1.0f);
    const std::vector<int64_t> positions = {2050, 2051, 5000, 9999, 30000};  // identity, just past it, tails 1 and 0
    const int64_t nq = (int64_t) positions.size();
    std::vector<int32_t> steps((size_t) nq * k::kStepCount);
    for (int64_t q = 0; q < nq; ++q) k::qsa_step_fill(&steps[(size_t) q * k::kStepCount], positions[(size_t) q], s);
    const int64_t max_blocks = (positions.back() + 1) / 4 + 8;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> pooled((size_t) max_blocks * 128), dead(128), qi((size_t) nq * 4 * 128);
    for (auto& v : pooled) v = g(rng);
    for (auto& v : dead) v = g(rng);
    for (auto& v : qi) v = g(rng);
    float *d_pooled = dev<float>(pooled.size()), *d_dead = dev<float>(128), *d_q = dev<float>(qi.size());
    float* d_sc = dev<float>((size_t) nq * max_blocks);
    int32_t *d_steps = dev<int32_t>(steps.size()), *d_ids = dev<int32_t>((size_t) nq * cap);
    up(d_pooled, pooled); up(d_dead, dead); up(d_q, qi); up(d_steps, steps);
    k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_sc, nullptr);
    check(cudaDeviceSynchronize(), "block scores");
    std::vector<float> sc = down(d_sc, (size_t) nq * max_blocks);

    int bad = 0;
    long long score_bad = 0;
    double worst = 0;
    for (int64_t q = 0; q < nq; ++q) {
        const int32_t* st = &steps[(size_t) q * k::kStepCount];
        const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid];
        for (int64_t b = 0; b <= n_bid; ++b) {
            const float* key = b == n_bid ? dead.data() : &pooled[(size_t) b * 128];
            double ref = 0, mag = 0;
            for (int h = 0; h < 4; ++h) {
                double d = 0;
                for (int i = 0; i < 128; ++i) {
                    d += (double) qi[(size_t) (q * 4 + h) * 128 + i] * (double) key[i];
                    mag += std::fabs((double) qi[(size_t) (q * 4 + h) * 128 + i] * (double) key[i]);
                }
                ref += d > 0 ? d : 0;
            }
            const bool bonus = b == n_bid && n_kv % 4 != 0;
            // an f32 dot of 128 terms (4 per lane, then 5 butterfly levels) is within ~10 ulp of sum|q k|; at 1e9
            // the float spacing is 64
            const double tol = 1e-5 * mag + (bonus ? 128.0 : 0.0);
            const double e = std::fabs((double) sc[(size_t) (q * max_blocks + b)] - (ref + (bonus ? 1e9 : 0.0)));
            worst = std::max(worst, e / tol);
            if (!(e <= tol)) ++score_bad;
        }
    }
    std::printf("  qsa_block_scores: %lld scores over tolerance (worst error / tolerance %.3f)  %s\n", score_bad, worst,
                score_bad ? "*** WRONG ***" : "ok");
    if (score_bad) ++bad;

    // the top-k, first on the device's own scores, then on scores full of ties
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            for (int64_t q = 0; q < nq; ++q)
                for (int64_t b = 0; b < max_blocks; ++b) {
                    const int v = (int) ((b * 7 + q) % 13);
                    sc[(size_t) (q * max_blocks + b)] = v == 0 ? ((b & 1) ? -0.0f : 0.0f) : (float) v * 0.25f;
                }
            up(d_sc, sc);
        }
        check(cudaMemset(d_ids, 0xFF, (size_t) nq * cap * sizeof(int32_t)), "poison ids");
        k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_ids, nullptr);
        check(cudaDeviceSynchronize(), "block topk");
        const std::vector<int32_t> ids = down(d_ids, (size_t) nq * cap);
        long long wrong = 0;
        for (int64_t q = 0; q < nq; ++q) {
            const int32_t* st = &steps[(size_t) q * k::kStepCount];
            const std::vector<int32_t> want = ref_select(&sc[(size_t) (q * max_blocks)], st[k::kStepNKv],
                                                         st[k::kStepNBid], st[k::kStepWidth]);
            for (size_t i = 0; i < want.size(); ++i) wrong += ids[(size_t) q * cap + i] != want[i];
        }
        std::printf("  qsa_block_topk (%s): %lld selected cells differ from the reference over %lld queries  %s\n",
                    pass == 0 ? "the device's scores" : "ties, +0 and -0", wrong, (long long) nq,
                    wrong ? "*** WRONG ***" : "ok");
        if (wrong) ++bad;
    }
    for (float* p : {d_pooled, d_dead, d_q, d_sc}) cudaFree(p);
    cudaFree(d_steps); cudaFree(d_ids);
    return bad;
}

// ================================ 3. attention over the selected cells ================================

int attention_test(int mode) {
    const char* names[3] = {"FP16", "INT8", "Q4_0"};
    const k::QsaShapes s = k::qsa_real_shapes();
    const int HD = 256, NH = 24, NKV = 2, G = NH / NKV, P = (int) s.page_size;
    const int cells = 6144, pages = cells / P;
    std::mt19937 rng(77 + (unsigned) mode);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    std::vector<int32_t> table((size_t) pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    const size_t rows = (size_t) pages * NKV * P;               // pool rows of HD values
    std::vector<uint16_t> kf, vf, ks, vs;
    std::vector<int8_t> kq, vq;
    std::vector<uint8_t> k4, v4;
    const size_t b4 = (size_t) (HD / k::QK4_0) * sizeof(k::block_q4_0);
    if (mode == 0) {
        kf.resize(rows * HD); vf.resize(rows * HD);
        for (auto& x : kf) x = k::f16_from_f32(g(rng));
        for (auto& x : vf) x = k::f16_from_f32(g(rng));
    } else if (mode == 1) {
        kq.resize(rows * HD); vq.resize(rows * HD);
        ks.resize(rows * (HD / k::KV_Q8_GROUP)); vs.resize(rows * (HD / k::KV_Q8_GROUP));
        for (auto& x : kq) x = (int8_t) ((int) (rng() % 255) - 127);
        for (auto& x : vq) x = (int8_t) ((int) (rng() % 255) - 127);
        for (auto& x : ks) x = k::f16_from_f32(0.004f + 0.012f * u(rng));
        for (auto& x : vs) x = k::f16_from_f32(0.004f + 0.012f * u(rng));
    } else {
        k4.resize(rows * b4); v4.resize(rows * b4);
        for (auto* pool : {&k4, &v4})
            for (size_t r = 0; r < rows * (HD / k::QK4_0); ++r) {
                k::block_q4_0 blk;
                blk.d = k::f16_from_f32((u(rng) < 0.5f ? -1.0f : 1.0f) * (0.1f + 0.3f * u(rng)));
                for (auto& b : blk.qs) b = (uint8_t) (rng() & 0xFF);
                std::memcpy(pool->data() + r * sizeof(blk), &blk, sizeof(blk));
            }
    }
    // one value of the pools as the kernel reads it: logical cell -> page table -> row
    auto value = [&](bool is_v, int cell, int kvh, int d) -> double {
        const size_t row = ((size_t) table[(size_t) (cell / P)] * NKV + kvh) * P + (size_t) (cell % P);
        if (mode == 0) return k::f32_from_f16((is_v ? vf : kf)[row * HD + d]);
        if (mode == 1) {
            const float sc = k::f32_from_f16((is_v ? vs : ks)[row * (HD / k::KV_Q8_GROUP) + d / k::KV_Q8_GROUP]);
            return (double) ((float) (is_v ? vq : kq)[row * HD + d] * sc);
        }
        k::block_q4_0 blk;
        std::memcpy(&blk, (is_v ? v4 : k4).data() + row * b4 + (size_t) (d / 32) * sizeof(blk), sizeof(blk));
        const int rem = d % 32;
        const int nib = rem < 16 ? (blk.qs[rem] & 0x0F) : (blk.qs[rem - 16] >> 4);
        return (double) ((float) (nib - 8) * k::f32_from_f16(blk.d));
    };

    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);   // 2051
    const std::vector<int> n_sel = {1, 65, (int) cap};                  // one cell, a 64+1 chunk split, the full width
    const int nq = (int) n_sel.size();
    std::vector<float> q((size_t) nq * NH * HD);
    for (auto& x : q) x = g(rng);
    std::vector<int32_t> ids((size_t) nq * cap, 0), steps((size_t) nq * k::kStepCount, 0);
    for (int j = 0; j < nq; ++j) {
        std::vector<int32_t> all((size_t) cells);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + n_sel[(size_t) j]);
        std::copy(all.begin(), all.begin() + n_sel[(size_t) j], ids.begin() + (ptrdiff_t) j * cap);
        k::qsa_step_fill(&steps[(size_t) j * k::kStepCount], cells - 1, s);
        steps[(size_t) j * k::kStepCount + k::kStepWidth] = n_sel[(size_t) j];
    }

    uint16_t *d_kf = nullptr, *d_vf = nullptr, *d_ks = nullptr, *d_vs = nullptr;
    int8_t *d_kq = nullptr, *d_vq = nullptr;
    uint8_t *d_k4 = nullptr, *d_v4 = nullptr;
    k::QsaAttnPools pools;
    if (mode == 0) {
        d_kf = dev<uint16_t>(kf.size()); d_vf = dev<uint16_t>(vf.size()); up(d_kf, kf); up(d_vf, vf);
        pools.k_pool = d_kf; pools.v_pool = d_vf;
    } else if (mode == 1) {
        d_kq = dev<int8_t>(kq.size()); d_vq = dev<int8_t>(vq.size()); d_ks = dev<uint16_t>(ks.size());
        d_vs = dev<uint16_t>(vs.size());
        up(d_kq, kq); up(d_vq, vq); up(d_ks, ks); up(d_vs, vs);
        pools.k_q = d_kq; pools.v_q = d_vq; pools.k_scale = d_ks; pools.v_scale = d_vs;
    } else {
        d_k4 = dev<uint8_t>(k4.size()); d_v4 = dev<uint8_t>(v4.size()); up(d_k4, k4); up(d_v4, v4);
        pools.k_q4 = d_k4; pools.v_q4 = d_v4;
    }
    int32_t *d_table = dev<int32_t>(table.size()), *d_ids = dev<int32_t>(ids.size()), *d_steps = dev<int32_t>(steps.size());
    up(d_table, table); up(d_ids, ids); up(d_steps, steps);
    pools.page_table = d_table;
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float *d_q = dev<float>(q.size()), *d_scr = dev<float>((size_t) nq * scr), *d_out = dev<float>(q.size());
    float* d_one = dev<float>((size_t) NH * HD);
    up(d_q, q);

    k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, s, d_scr, d_out, nq, nullptr);
    check(cudaDeviceSynchronize(), "attn batch");
    const std::vector<float> got = down(d_out, q.size());
    int batch_vs_step = 0;
    for (int j = 0; j < nq; ++j) {
        k::qsa_decode_attn_step(d_q + (size_t) j * NH * HD, pools, d_ids + (size_t) j * cap,
                                d_steps + (size_t) j * k::kStepCount, cap, s, d_scr, d_one, nullptr);
        check(cudaDeviceSynchronize(), "attn step");
        const std::vector<float> one = down(d_one, (size_t) NH * HD);
        batch_vs_step += std::memcmp(one.data(), got.data() + (size_t) j * NH * HD, one.size() * 4) != 0;
    }

    // float64 attention: softmax over the selected cells of (q . k) / 16, then the weighted values.  The error is
    // judged against sum_i p_i |v_i| - the scale of the f32 sums the kernel forms - so a head whose output nearly
    // cancels is not held to a relative bound it cannot meet.
    double worst = 0, l1 = 0, l1m = 0;
    for (int j = 0; j < nq; ++j) {
        const int n = n_sel[(size_t) j];
        const int32_t* cj = &ids[(size_t) j * cap];
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / G;
            std::vector<double> sco((size_t) n);
            double m = -1e300;
            for (int i = 0; i < n; ++i) {
                double d = 0;
                for (int e = 0; e < HD; ++e) d += (double) q[((size_t) j * NH + h) * HD + e] * value(false, cj[i], kvh, e);
                sco[(size_t) i] = d / 16.0;
                m = std::max(m, sco[(size_t) i]);
            }
            double L = 0;
            for (auto& x : sco) { x = std::exp(x - m); L += x; }
            for (int e = 0; e < HD; ++e) {
                double acc = 0, mag = 0;
                for (int i = 0; i < n; ++i) {
                    const double v = value(true, cj[i], kvh, e);
                    acc += sco[(size_t) i] * v;
                    mag += sco[(size_t) i] * std::fabs(v);
                }
                const double ref = acc / L, scale = mag / L;
                const double err = std::fabs((double) got[((size_t) j * NH + h) * HD + e] - ref);
                worst = std::max(worst, err / (scale > 1e-30 ? scale : 1e-30));
                l1 += err;
                l1m += std::fabs(ref);
            }
        }
    }
    // f32: 8-term lane sums and a butterfly for the scores, __expf, a 64-term fmaf chain per chunk and the merge
    // of up to 33 chunks: ~1e-6 of sum p|v| typically, ~2.5e-5 if every rounding lined up; a wrong page, head or
    // chunk is O(1)
    const bool ok = worst <= 5e-5 && batch_vs_step == 0;
    std::printf("  qsa_decode_attn %s pools, %d/%d/%d cells: worst error %.2e of sum p|v| (rel L1 %.2e), batch vs "
                "step %s  %s\n", names[mode], n_sel[0], n_sel[1], n_sel[2], worst, l1 / (l1m > 0 ? l1m : 1),
                batch_vs_step ? "*** DIFFER ***" : "bitwise", ok ? "ok" : "*** WRONG ***");
    for (void* p : {(void*) d_kf, (void*) d_vf, (void*) d_ks, (void*) d_vs, (void*) d_kq, (void*) d_vq, (void*) d_k4,
                    (void*) d_v4, (void*) d_table, (void*) d_ids, (void*) d_steps, (void*) d_q, (void*) d_scr,
                    (void*) d_out, (void*) d_one})
        if (p) cudaFree(p);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: qsa_decode_parity [--selftest]\n"); return 2; }
    }
    int bad = select_tests();
    for (int mode = 0; mode < 3; ++mode) bad += attention_test(mode);
    std::printf("\nqsa_decode: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("qsa_decode_parity OK\n");
    return 0;
}
