// src/kernels/qsa_decode_attn_parity.cpp - the split-K decode attention (qsa_decode_attn.hpp) on the device against a
// host reference, and its variants against each other (GPU, synthetic, no model).
//
//   1. every variant - the prefetching default (O6c, #23), the previous kernel, CHUNK 32 - against a double-precision
//      host softmax over the same decoded pool values, for fp16, int8 and q4_0 pools behind a permuted page table,
//      with a batch of queries whose widths cover one cell, partial chunks, whole chunks and the full 2,051;
//   2. the prefetching kernel BITWISE equal to the previous one, in the batch and the single-query forms;
//   3. cells whose page is -1 (a KV-streaming overflow) are masked in every variant - the output is the
//      reference without those cells, a chunk with no resolvable cell contributes nothing, and a query with none
//      at all gives zeros;
//   4. a V pool that is not 16-byte aligned falls back to the previous kernel, bitwise;
//   5. the output gate folded into the merge (both gates, opt-in STRATA_QSA_MERGE_GATE) equals the attention
//      followed by the separate gate kernel, bitwise - the condition for making it the default.
//
// The tolerance of (1) is set by `__expf` (a few ulp per weight) and fp32 accumulation over 256-term dots and up to
// 2,051 cells: 2e-4 of the largest |V|, the same order as qsa_parity's attention check.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
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
    std::printf("  %-66s %-6s %s\n", name.c_str(), ok ? "ok" : "*** BAD ***", detail.c_str());
    if (!ok) ++g_bad;
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T>
void put(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
}
template <typename T>
std::vector<T> get(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    return h;
}

constexpr int HD = 256, NKV = 2, NH = 24, G = NH / NKV, PS = 4;

/// One pool in one format, host copy and device copy, with its host decoder.
struct Pool {
    int fmt = 0;               // k::kKvF16 / kKvInt8 / kKvQ4
    int64_t rows = 0;          // physical rows: pages * NKV * PS
    std::vector<uint16_t> k16, v16;
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs;
    std::vector<uint8_t> k4, v4;
    k::QsaAttnPools dev;

    void make(int f, int64_t pages, std::mt19937& rng) {
        fmt = f;
        rows = pages * NKV * PS;
        std::normal_distribution<float> nd(0.f, 1.f);
        const size_t n = (size_t) rows * HD;
        if (fmt == k::kKvF16) {
            k16.resize(n); v16.resize(n);
            for (auto& x : k16) x = k::f16_from_f32(nd(rng));
            for (auto& x : v16) x = k::f16_from_f32(nd(rng));
            dev.k_pool = dalloc<uint16_t>(n); dev.v_pool = dalloc<uint16_t>(n);
            put((uint16_t*) dev.k_pool, k16); put((uint16_t*) dev.v_pool, v16);
        } else if (fmt == k::kKvInt8) {
            kq.resize(n); vq.resize(n);
            ks.resize((size_t) rows * (HD / k::KV_Q8_GROUP)); vs.resize(ks.size());
            for (auto& x : kq) x = (int8_t) ((int) (rng() % 255) - 127);
            for (auto& x : vq) x = (int8_t) ((int) (rng() % 255) - 127);
            for (auto& x : ks) x = k::f16_from_f32(0.004f + 0.008f * (float) (rng() % 1000) / 1000.0f);
            for (auto& x : vs) x = k::f16_from_f32(0.004f + 0.008f * (float) (rng() % 1000) / 1000.0f);
            dev.k_q = dalloc<int8_t>(n); dev.v_q = dalloc<int8_t>(n);
            dev.k_scale = dalloc<uint16_t>(ks.size()); dev.v_scale = dalloc<uint16_t>(vs.size());
            put((int8_t*) dev.k_q, kq); put((int8_t*) dev.v_q, vq);
            put((uint16_t*) dev.k_scale, ks); put((uint16_t*) dev.v_scale, vs);
        } else {
            const size_t bytes = (size_t) rows * k::kv_q4_bytes_per_head(HD);
            k4.resize(bytes); v4.resize(bytes);
            for (auto* buf : {&k4, &v4})
                for (int64_t r = 0; r < rows; ++r)
                    for (int b = 0; b < HD / k::QK4_0; ++b) {
                        uint8_t* blk = buf->data() + (size_t) r * k::kv_q4_bytes_per_head(HD) + (size_t) b * sizeof(k::block_q4_0);
                        const uint16_t d = k::f16_from_f32(0.05f + 0.2f * (float) (rng() % 1000) / 1000.0f);
                        std::memcpy(blk, &d, 2);
                        for (int j = 0; j < k::QK4_0 / 2; ++j) blk[2 + j] = (uint8_t) (rng() & 0xFF);
                    }
            dev.k_q4 = dalloc<uint8_t>(bytes); dev.v_q4 = dalloc<uint8_t>(bytes);
            put((uint8_t*) dev.k_q4, k4); put((uint8_t*) dev.v_q4, v4);
        }
        decode();
    }
    // every value decoded once, as the kernels decode it (exact fp16 widening, then one f32 multiply)
    std::vector<float> kd, vd;
    void decode() {
        kd.resize((size_t) rows * HD);
        vd.resize(kd.size());
        for (int64_t r = 0; r < rows; ++r)
            for (int d = 0; d < HD; ++d) {
                kd[(size_t) r * HD + d] = (float) decode_one(false, r, d);
                vd[(size_t) r * HD + d] = (float) decode_one(true, r, d);
            }
    }
    double val(bool value, int64_t row, int d) const { return (value ? vd : kd)[(size_t) row * HD + d]; }
    double decode_one(bool value, int64_t row, int d) const {
        if (fmt == k::kKvF16) return k::f32_from_f16((value ? v16 : k16)[(size_t) row * HD + d]);
        if (fmt == k::kKvInt8) {
            const float sc = k::f32_from_f16((value ? vs : ks)[(size_t) row * (HD / k::KV_Q8_GROUP) + d / k::KV_Q8_GROUP]);
            return (double) ((float) (value ? vq : kq)[(size_t) row * HD + d] * sc);
        }
        const uint8_t* blk = (value ? v4 : k4).data() + (size_t) row * k::kv_q4_bytes_per_head(HD) +
                             (size_t) (d / k::QK4_0) * sizeof(k::block_q4_0);
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const int rem = d % k::QK4_0;
        const uint8_t byte = blk[2 + (rem < 16 ? rem : rem - 16)];
        const int nib = rem < 16 ? (byte & 0x0F) - 8 : (byte >> 4) - 8;
        return (double) ((float) nib * k::f32_from_f16(dbits));
    }
    double vmax() const {
        double m = 0;
        for (int64_t r = 0; r < rows; r += 7)
            for (int d = 0; d < HD; d += 5) m = std::max(m, std::fabs(val(true, r, d)));
        return m;
    }
};

/// The reference: softmax over the selected cells whose page resolves, in double.
std::vector<double> reference(const Pool& pool, const std::vector<int32_t>& table, const std::vector<float>& q,
                              const std::vector<int32_t>& ids, const std::vector<int32_t>& steps, int n_q, int64_t cap) {
    std::vector<double> out((size_t) n_q * NH * HD, 0.0);
    for (int qi = 0; qi < n_q; ++qi) {
        const int width = steps[(size_t) qi * k::kStepCount + k::kStepWidth];
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / G;
            std::vector<double> sc;
            std::vector<int64_t> rows;
            for (int j = 0; j < width; ++j) {
                const int cell = ids[(size_t) qi * cap + j];
                const int page = table[(size_t) (cell / PS)];
                if (page < 0) continue;
                const int64_t row = ((int64_t) page * NKV + kvh) * PS + cell % PS;
                double s = 0;
                for (int d = 0; d < HD; ++d) s += (double) q[((size_t) qi * NH + h) * HD + d] * pool.val(false, row, d);
                sc.push_back(s / 16.0);
                rows.push_back(row);
            }
            if (sc.empty()) continue;
            const double m = *std::max_element(sc.begin(), sc.end());
            double l = 0;
            for (auto& s : sc) { s = std::exp(s - m); l += s; }
            for (size_t j = 0; j < sc.size(); ++j)
                for (int d = 0; d < HD; ++d)
                    out[((size_t) qi * NH + h) * HD + d] += sc[j] / l * pool.val(true, rows[j], d);
        }
    }
    return out;
}

const char* fmt_name(int f) { return f == k::kKvQ4 ? "q4_0" : f == k::kKvInt8 ? "int8" : "fp16"; }
const char* var_name(int v) {
    return v == k::kQsaAttnOld ? "previous" : v == k::kQsaAttnChunk32 ? "chunk 32" : "prefetch";
}

bool run(int fmt) {
    std::printf("\n-- %s pools\n", fmt_name(fmt));
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t N = 12000, pages = (N + PS - 1) / PS;
    std::mt19937 rng(11u + 3u * (unsigned) fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    Pool pool;
    pool.make(fmt, pages, rng);
    const double vmax = pool.vmax();

    // a permuted page table
    std::vector<int32_t> table((size_t) pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    int32_t* dtable = dalloc<int32_t>(table.size());
    put(dtable, table);

    // queries: widths 1, partial chunks, whole chunks, 2,051; ascending unique cells
    const std::vector<int> widths = {1, 5, 31, 32, 33, 63, 64, 65, 200, 1000, 2048, 2051};
    const int n_q = (int) widths.size();
    std::vector<int32_t> ids((size_t) n_q * cap, 0), steps((size_t) n_q * k::kStepCount, 0);
    for (int qi = 0; qi < n_q; ++qi) {
        const int64_t n_kv = std::max<int64_t>(widths[(size_t) qi], N - 17 * qi);
        std::set<int32_t> pick;
        while ((int) pick.size() < widths[(size_t) qi]) pick.insert((int32_t) (rng() % n_kv));
        std::copy(pick.begin(), pick.end(), ids.begin() + (size_t) qi * cap);
        int32_t* st = steps.data() + (size_t) qi * k::kStepCount;
        st[k::kStepPos] = (int32_t) (n_kv - 1); st[k::kStepNKv] = (int32_t) n_kv;
        st[k::kStepNBid] = (int32_t) (n_kv / 4); st[k::kStepWidth] = widths[(size_t) qi];
    }
    std::vector<float> q((size_t) n_q * NH * HD);
    for (auto& x : q) x = nd(rng);
    int32_t* dids = dalloc<int32_t>(ids.size());
    int32_t* dsteps = dalloc<int32_t>(steps.size());
    float* dq = dalloc<float>(q.size());
    put(dids, ids); put(dsteps, steps); put(dq, q);
    k::qsa_decode_attn_set_variant(k::kQsaAttnChunk32);                  // the largest scratch
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>((size_t) n_q * scr);
    float* dout = dalloc<float>((size_t) n_q * NH * HD);
    const size_t n_out = (size_t) n_q * NH * HD;

    auto attend = [&](int variant, const k::QsaAttnPools& pools, bool single, int qi) {
        k::qsa_decode_attn_set_variant(variant);
        ck(cudaMemset(dout, 0xFF, n_out * 4), "poison");   // NaN everywhere: every output must be written
        if (single)
            k::qsa_decode_attn_step(dq + (size_t) qi * NH * HD, pools, dids + (size_t) qi * cap,
                                    dsteps + (size_t) qi * k::kStepCount, cap, s, scratch, dout, nullptr);
        else
            k::qsa_decode_attn_batch(dq, pools, dids, dsteps, cap, s, scratch, dout, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "attention");
        return get(dout, single ? (size_t) NH * HD : n_out);
    };
    auto err_of = [&](const std::vector<float>& got, const std::vector<double>& want) {
        double e = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (!std::isfinite(got[i])) return 1e30;
            e = std::max(e, std::fabs((double) got[i] - want[i]));
        }
        return e / std::max(vmax, 1e-30);
    };

    k::QsaAttnPools pools = pool.dev;
    pools.page_table = dtable;
    bool ok = true;
    // 1. + 2. every variant against the reference; prefetch bitwise equal to the previous kernel
    {
        const std::vector<double> want = reference(pool, table, q, ids, steps, n_q, cap);
        std::vector<float> got[3];
        for (int v : {k::kQsaAttnPrefetch, k::kQsaAttnOld, k::kQsaAttnChunk32}) {
            got[v] = attend(v, pools, false, 0);
            const double e = err_of(got[v], want);
            char b[64];
            std::snprintf(b, sizeof b, "max err %.2e of max|V|", e);
            require(std::string(fmt_name(fmt)) + " " + var_name(v) + ": batch of 12 = host softmax", e < 2e-4, b);
            ok &= e < 2e-4;
        }
        const bool same = std::memcmp(got[k::kQsaAttnPrefetch].data(), got[k::kQsaAttnOld].data(), n_out * 4) == 0;
        require(std::string(fmt_name(fmt)) + " prefetch = previous kernel, bitwise (batch)", same);
        ok &= same;
        int single_bad = 0;
        for (int qi : {0, 6, 11}) {
            const std::vector<float> a = attend(k::kQsaAttnPrefetch, pools, true, qi);
            const std::vector<float> b = attend(k::kQsaAttnOld, pools, true, qi);
            single_bad += std::memcmp(a.data(), b.data(), a.size() * 4) != 0;
            single_bad += std::memcmp(a.data(), got[k::kQsaAttnOld].data() + (size_t) qi * NH * HD, a.size() * 4) != 0;
        }
        require(std::string(fmt_name(fmt)) + " prefetch = previous = batch row, bitwise (single query)",
                single_bad == 0);
        ok &= single_bad == 0;
    }
    // 3. unresolved pages are masked
    {
        std::vector<int32_t> holes = table;
        // every 5th page the selections name, a run of 16 consecutive pages (one whole 64-cell chunk of the 2,051
        // query), and every page of the width-5 query (which must then give zeros)
        std::set<int32_t> named;
        for (int qi = 0; qi < n_q; ++qi)
            for (int j = 0; j < widths[(size_t) qi]; ++j) named.insert(ids[(size_t) qi * cap + j] / PS);
        int c = 0;
        for (int32_t p : named) if (c++ % 5 == 0) holes[(size_t) p] = -1;
        const int32_t* last = ids.data() + (size_t) (n_q - 1) * cap;
        for (int j = 128; j < 192; ++j) holes[(size_t) (last[j] / PS)] = -1;
        for (int j = 0; j < 5; ++j) holes[(size_t) (ids[(size_t) 1 * cap + j] / PS)] = -1;
        put(dtable, holes);
        const std::vector<double> want = reference(pool, holes, q, ids, steps, n_q, cap);
        std::vector<float> got[3];
        for (int v : {k::kQsaAttnPrefetch, k::kQsaAttnOld, k::kQsaAttnChunk32}) {
            got[v] = attend(v, pools, false, 0);
            const double e = err_of(got[v], want);
            bool zeros = true;
            for (int i = 0; i < NH * HD; ++i) zeros &= got[v][(size_t) NH * HD + i] == 0.0f;
            char b[64];
            std::snprintf(b, sizeof b, "max err %.2e of max|V|", e);
            require(std::string(fmt_name(fmt)) + " " + var_name(v) + ": pages -1 masked (B14)", e < 2e-4 && zeros,
                    std::string(b) + (zeros ? "" : ", the all-masked query is not zero"));
            ok &= e < 2e-4 && zeros;
        }
        const bool same = std::memcmp(got[k::kQsaAttnPrefetch].data(), got[k::kQsaAttnOld].data(), n_out * 4) == 0;
        require(std::string(fmt_name(fmt)) + " with holes: prefetch = previous kernel, bitwise", same);
        ok &= same;
        put(dtable, table);
    }
    // 4. an unaligned V pool keeps the previous kernel
    {
        k::QsaAttnPools shifted = pools;
        const std::vector<float> want = attend(k::kQsaAttnOld, pools, false, 0);
        if (fmt == k::kKvF16) {
            uint16_t* v = dalloc<uint16_t>(pool.v16.size() + 8);
            put(v + 4, pool.v16);                     // 8 bytes past a 256-byte boundary
            shifted.v_pool = v + 4;
        } else if (fmt == k::kKvInt8) {
            int8_t* v = dalloc<int8_t>(pool.vq.size() + 16);
            put(v + 8, pool.vq);
            shifted.v_q = v + 8;
        } else {
            uint8_t* v = dalloc<uint8_t>(pool.v4.size() + 16);
            put(v + 4, pool.v4);
            shifted.v_q4 = v + 4;
        }
        const std::vector<float> got = attend(k::kQsaAttnPrefetch, shifted, false, 0);
        const bool same = std::memcmp(got.data(), want.data(), n_out * 4) == 0;
        require(std::string(fmt_name(fmt)) + " unaligned V pool: falls back, bitwise", same);
        ok &= same;
    }
    // 5. the gate folded into the merge (opt-in STRATA_QSA_MERGE_GATE) = the separate gate kernels, bitwise
    {
        std::vector<float> qf((size_t) n_q * NH * 2 * HD);
        for (auto& x : qf) x = nd(rng) * 4.0f;
        float* dqf = dalloc<float>(qf.size());
        float* dsep = dalloc<float>(n_out);
        float* dfold = dalloc<float>(n_out);
        put(dqf, qf);
        cudaStream_t cs;
        ck(cudaStreamCreate(&cs), "stream");
        k::qsa_decode_attn_set_variant(k::kQsaAttnPrefetch);
        for (int kind : {k::kQsaGateF64, k::kQsaGateNativeF32}) {
            k::qsa_decode_attn_batch(dq, pools, dids, dsteps, cap, s, scratch, dout, n_q, cs);
            for (int qi = 0; qi < n_q; ++qi) {
                const size_t o = (size_t) qi * NH * HD;
                if (kind == k::kQsaGateF64)
                    k::qsa_gate_apply_f32(dout + o, dqf + 2 * o, s, dsep + o, cs);
                else
                    k::native_qsa_gate_apply(dout + o, dqf + 2 * o, dsep + o, NH, HD, cs);
            }
            ck(cudaStreamSynchronize(cs), "separate gate");
            const std::vector<float> attn_sep = get(dout, n_out), sep = get(dsep, n_out);
            ck(cudaMemset(dout, 0xFF, n_out * 4), "poison");
            k::QsaAttnGate gate;
            gate.kind = kind;
            gate.q_full = dqf;
            gate.out = dfold;
            k::qsa_decode_attn_batch(dq, pools, dids, dsteps, cap, s, scratch, dout, n_q, cs, gate);
            ck(cudaStreamSynchronize(cs), "folded gate");
            const std::vector<float> attn_fold = get(dout, n_out), fold = get(dfold, n_out);
            const bool same = std::memcmp(sep.data(), fold.data(), n_out * 4) == 0 &&
                              std::memcmp(attn_sep.data(), attn_fold.data(), n_out * 4) == 0;
            require(std::string(fmt_name(fmt)) + (kind == k::kQsaGateF64 ? " fp64 gate" : " native f32 gate") +
                        " folded into the merge = separate kernel, bitwise", same);
            ok &= same;
        }
        cudaStreamDestroy(cs);
    }
    k::qsa_decode_attn_set_variant(k::kQsaAttnPrefetch);
    return ok;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("qsa_decode_attn_parity: split-K decode attention, every variant against a host reference\n");
    run(k::kKvF16);
    run(k::kKvInt8);
    run(k::kKvQ4);
    std::printf(g_bad ? "\nFAIL (%d)\n" : "\nPASS\n", g_bad);
    return g_bad ? 1 : 0;
}
