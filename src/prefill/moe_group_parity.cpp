// src/prefill/moe_group_parity.cpp - issue #36: the prompt path's expert grouping on the device and its batched
// gathers against what they replace (GPU, synthetic, no model).
//
//   1. moe_group against the host sort the prefill ran until 0.1.20 (count, prefix, `fill[e]++` in pair order):
//      the row tables `slot` and `src` and the counts must be EQUAL, entry for entry - the MMQ products and the
//      combine read them, and a row order that differs inside one expert changes the stream-k split of the MMQ
//      launch and so the last bit of the result.  Routings: top-10 distinct experts per token from a skewed
//      distribution (hot experts, experts nobody picks), 1 to 8192 tokens, sizes around the kernel's 2048-pair
//      chunks; one routing with an id out of range must raise the flag.
//   2. gather_native_batch / gather_strata_q2_batch against one gather_native / gather_strata_q2 per expert: the
//      group buffers (every byte, including the slots no expert of the batch lands in) must be EQUAL, for 16 and
//      for 5 experts in a shuffled slot order, 16-byte aligned (vector copies) and not (byte copies).
#include "strata/prefill/moe_group.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace pf = strata::prefill;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    return p;
}

// the host grouping of prefill.cpp until 0.1.20
void host_group(const std::vector<int32_t>& ids, int64_t k, int64_t n_expert, std::vector<int32_t>& slot,
                std::vector<int32_t>& src, std::vector<int32_t>& cnt) {
    const int64_t n = (int64_t) ids.size();
    cnt.assign((size_t) n_expert, 0);
    for (int32_t e : ids) ++cnt[(size_t) e];
    std::vector<int32_t> off((size_t) n_expert + 1, 0);
    for (int64_t e = 0; e < n_expert; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    slot.assign((size_t) n, 0);
    src.assign((size_t) n, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t p = fill[(size_t) ids[(size_t) i]]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
}

void group_case(const char* name, int64_t tokens, double skew, bool bad_id, uint32_t seed, cudaStream_t cs) {
    constexpr int64_t K = 10, NE = 512;
    std::mt19937 rng(seed);
    // a skewed routing: expert e drawn with weight 1 / (1 + e')^skew over a shuffled order, ten distinct per token
    std::vector<int32_t> perm(NE);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> wgt(NE);
    for (int64_t i = 0; i < NE; ++i) wgt[(size_t) i] = 1.0 / std::pow(1.0 + (double) i, skew);
    std::discrete_distribution<int> pick(wgt.begin(), wgt.end());
    std::vector<int32_t> ids((size_t) (tokens * K));
    for (int64_t t = 0; t < tokens; ++t) {
        int32_t* row = ids.data() + t * K;
        for (int64_t j = 0; j < K; ++j) {
            int32_t e;
            do e = perm[(size_t) pick(rng)]; while (std::find(row, row + j, e) != row + j);
            row[j] = e;
        }
    }
    const int64_t n = tokens * K;
    if (bad_id) ids[(size_t) (n / 2)] = (int32_t) NE + 3;
    int32_t* d_ids = dalloc<int32_t>((size_t) n);
    int32_t* d_slot = dalloc<int32_t>((size_t) n);
    int32_t* d_src = dalloc<int32_t>((size_t) n);
    void* scratch = dalloc<uint8_t>(pf::moe_group_scratch_bytes(n, NE));
    int32_t* host_out = nullptr;
    ck(cudaHostAlloc((void**) &host_out, (size_t) (NE + 1) * 4, cudaHostAllocDefault), "hostalloc");
    ck(cudaMemcpy(d_ids, ids.data(), (size_t) n * 4, cudaMemcpyHostToDevice), "ids");
    // junk in the scratch and the tables: the kernels must write what they own
    ck(cudaMemset(scratch, 0x5a, pf::moe_group_scratch_bytes(n, NE)), "junk");
    ck(cudaMemset(d_slot, 0x77, (size_t) n * 4), "junk");
    ck(cudaMemset(d_src, 0x77, (size_t) n * 4), "junk");
    // cudaMemset runs on the legacy stream, which a non-blocking `cs` does not wait for
    ck(cudaDeviceSynchronize(), "setup");
    pf::moe_group(d_ids, n, K, NE, d_slot, d_src, scratch, host_out, cs);
    ck(cudaStreamSynchronize(cs), "sync");
    bool ok = true;
    if (bad_id) {
        ok = host_out[NE] == 1;
        if (!ok) std::fprintf(stderr, "FAIL %s: an id out of range did not raise the flag\n", name);
    } else {
        std::vector<int32_t> slot, src, cnt, gs((size_t) n), gsrc((size_t) n);
        host_group(ids, K, NE, slot, src, cnt);
        ck(cudaMemcpy(gs.data(), d_slot, (size_t) n * 4, cudaMemcpyDeviceToHost), "slot");
        ck(cudaMemcpy(gsrc.data(), d_src, (size_t) n * 4, cudaMemcpyDeviceToHost), "src");
        if (host_out[NE] != 0) { ok = false; std::fprintf(stderr, "FAIL %s: flag raised\n", name); }
        if (!std::equal(cnt.begin(), cnt.end(), host_out)) { ok = false; std::fprintf(stderr, "FAIL %s: counts\n", name); }
        if (gs != slot) { ok = false; std::fprintf(stderr, "FAIL %s: slot table\n", name); }
        if (gsrc != src) { ok = false; std::fprintf(stderr, "FAIL %s: src table\n", name); }
    }
    int64_t unused = 0;
    for (int64_t e = 0; e < NE; ++e) unused += host_out[e] == 0;
    std::printf("%-40s %s (%lld pairs, %lld experts unpicked)\n", name, ok ? "equal" : "DIFFERS", (long long) n,
                (long long) unused);
    g_fail += !ok;
    cudaFree(d_ids); cudaFree(d_slot); cudaFree(d_src); cudaFree(scratch); cudaFreeHost(host_out);
}

#ifdef STRATA_PREFILL_MMQ
void gather_case(const char* name, bool strata_q2, int n, size_t misalign, uint32_t seed, cudaStream_t cs) {
    std::mt19937 rng(seed);
    // a native blob: gate | up | down; a Strata Q2_0 blob: 1382400 bytes (converted)
    const size_t half = strata_q2 ? 0 : 460800 + misalign, dbytes = strata_q2 ? 0 : 230400 + misalign;
    const size_t blob_bytes = strata_q2 ? 1382400 : 2 * half + dbytes;
    const size_t gu_stride = strata_q2 ? 921600 : 2 * half, d_stride = strata_q2 ? 460800 : dbytes;
    constexpr int G = pf::mmq::kGatherBatch;
    std::vector<uint8_t> h((size_t) G * (blob_bytes + 64));
    for (uint8_t& x : h) x = (uint8_t) rng();
    uint8_t* blobs = dalloc<uint8_t>(h.size());
    ck(cudaMemcpy(blobs, h.data(), h.size(), cudaMemcpyHostToDevice), "blobs");
    const size_t gu_bytes = (size_t) G * gu_stride + 4096 + misalign, d_bytes = (size_t) G * d_stride + 4096 + misalign;
    uint8_t *gu_a = dalloc<uint8_t>(gu_bytes), *gu_b = dalloc<uint8_t>(gu_bytes);
    uint8_t *d_a = dalloc<uint8_t>(d_bytes), *d_b = dalloc<uint8_t>(d_bytes);
    for (uint8_t* p : {gu_a, gu_b}) ck(cudaMemset(p, 0xa5, gu_bytes), "junk");
    for (uint8_t* p : {d_a, d_b}) ck(cudaMemset(p, 0xa5, d_bytes), "junk");
    ck(cudaDeviceSynchronize(), "setup");   // the legacy-stream memsets before the non-blocking `cs`
    std::vector<int32_t> q(G);
    std::iota(q.begin(), q.end(), 0);
    std::shuffle(q.begin(), q.end(), rng);
    pf::mmq::GatherBatch b;
    for (int i = 0; i < n; ++i) {
        const uint8_t* blob = blobs + (size_t) i * (blob_bytes + 64) + (strata_q2 ? 0 : misalign);
        b.blob[i] = blob;
        b.q[i] = q[(size_t) i];
        uint8_t* gdst = gu_a + (size_t) q[(size_t) i] * gu_stride;
        uint8_t* ddst = d_a + (size_t) q[(size_t) i] * d_stride;
        if (strata_q2) pf::mmq::gather_strata_q2(blob, gdst, ddst, cs);
        else pf::mmq::gather_native(blob, blob + half, half, blob + 2 * half, dbytes, gdst, ddst, cs);
    }
    b.n = n;
    if (strata_q2) pf::mmq::gather_strata_q2_batch(b, gu_b, gu_stride, d_b, d_stride, cs);
    else pf::mmq::gather_native_batch(b, half, 2 * half, half, dbytes, gu_b, gu_stride, d_b, d_stride, cs);
    ck(cudaStreamSynchronize(cs), "sync");
    std::vector<uint8_t> a1(gu_bytes), b1(gu_bytes), a2(d_bytes), b2(d_bytes);
    ck(cudaMemcpy(a1.data(), gu_a, gu_bytes, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(b1.data(), gu_b, gu_bytes, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(a2.data(), d_a, d_bytes, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(b2.data(), d_b, d_bytes, cudaMemcpyDeviceToHost), "d2h");
    const bool ok = a1 == b1 && a2 == b2;
    std::printf("%-40s %s (%d experts)\n", name, ok ? "bitwise equal" : "DIFFERS", n);
    g_fail += !ok;
    cudaFree(blobs); cudaFree(gu_a); cudaFree(gu_b); cudaFree(d_a); cudaFree(d_b);
}
#endif
}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "moe_group_parity: no CUDA device\n");
        return 2;
    }
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    group_case("one token", 1, 1.0, false, 1, cs);
    group_case("204 tokens (2040 pairs, one chunk)", 204, 1.0, false, 2, cs);
    group_case("205 tokens (2050 pairs, two chunks)", 205, 1.0, false, 3, cs);
    group_case("2048 tokens, flat routing", 2048, 0.0, false, 4, cs);
    group_case("8192 tokens, skewed routing", 8192, 1.1, false, 5, cs);
    group_case("8192 tokens, very skewed routing", 8192, 2.0, false, 6, cs);
    group_case("3000 tokens, an id out of range", 3000, 1.0, true, 7, cs);
#ifdef STRATA_PREFILL_MMQ
    gather_case("native gathers, 16 experts, aligned", false, 16, 0, 11, cs);
    gather_case("native gathers, 5 experts, aligned", false, 5, 0, 12, cs);
    gather_case("native gathers, 16 experts, byte copies", false, 16, 4, 13, cs);
    gather_case("Strata Q2_0 gathers, 16 experts", true, 16, 0, 14, cs);
    gather_case("Strata Q2_0 gathers, 3 experts", true, 3, 0, 15, cs);
#else
    std::printf("the batched gathers: not in this build (no MMQ)\n");
#endif
    std::printf("moe_group_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
