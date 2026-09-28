// src/kernels/s2_grouped_parity.cpp - the MTP draft layer's expert kernel (`moe_grouped_s2`, via
// `moe_group_resident`) against a double-precision CPU reference, with NEGATIVE weight scales (issue #52).
//
// WHY NEGATIVE.  Q2_0's grid is {-1, 0, 1, 2} x d: asymmetric, so a block whose large weights are negative is
// better served by d < 0, which mirrors the grid to {-2, -1, 0, 1} x |d|.  `tools/mtp_pack.py --q2-search
// wide|exact` stores such scales for the draft layer's experts, and the draft layer's experts run on exactly this
// kernel (`core/mtp.cpp`: all 512 resident, one `moe_grouped_s2` per round).  The kernel multiplies the fp16
// scale in as a float (`dw * dx * (s - hx)`), which is sign-agnostic by construction; this test is what shows it
// on the card, with the rows, the grouping and the intermediate quantization as the engine runs them.
//
// THE TWO STAGES ARE CHECKED SEPARATELY.  Gate/up and the SwiGLU against the reference from the kernel's own
// quantized activation; then the down projection against the reference from the kernel's OWN quantized
// intermediate (read back from the scratch, whose layout `moe_grouped_s2` fixes: gate_up floats, then the
// q8_0 blocks, then their fp32 scales).  Requantizing the intermediate on the CPU would add rounding
// disagreements that have nothing to do with the sign of d.
//
// NON-VACUITY.  The same blobs are also run with every scale made positive: both runs must match their
// references, and the two outputs must differ - a kernel that dropped the sign would pass one and fail the other.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int H = 2560, FF = 640, QK = 64;
constexpr int ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / QK, SC_D = FF / QK;
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr size_t BLOB = O_D_SCALES + (size_t) H * SC_D * 2;   // 1,382,400 (cpu/expert.hpp)
constexpr int NEXP = 16, T = 8, K = 10;                        // an MTP window: 8 tokens x top-10

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t rd16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

// sum over one 32-element chunk of (code - 1) * q, codes LSB-first 4 per byte
double chunk_dot(const uint8_t* codes, const int8_t* q) {
    long s = 0;
    for (int j = 0; j < 32; ++j) s += (long) (((codes[j / 4] >> (2 * (j % 4))) & 3) - 1) * q[j];
    return (double) s;
}

struct Result {
    double h_err = 0, h_max = 0, out_err = 0;   // worst |gpu - ref| / tolerance scale, per stage
    long bad = 0;
    std::vector<float> out;
};

Result run(const std::vector<uint8_t>& blobs, const std::vector<float>& x, const std::vector<int32_t>& ids) {
    namespace sk = strata::kernels;
    const int n = T * K;
    uint8_t *d_blobs = nullptr, *d_xq = nullptr, *d_scratch = nullptr;
    float *d_x = nullptr, *d_xs = nullptr, *d_out = nullptr;
    int32_t *d_ids = nullptr, *d_start = nullptr, *d_counts = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    unsigned long long* d_ptr = nullptr;
    const uint64_t scratch_bytes = sk::moe_hit_grouped_scratch_bytes(n, H, FF);
    check(cudaMalloc(&d_blobs, blobs.size()), "cudaMalloc blobs");
    check(cudaMalloc(&d_x, x.size() * 4), "cudaMalloc x");
    check(cudaMalloc(&d_xq, (size_t) T * (H / 32) * 34), "cudaMalloc xq");
    check(cudaMalloc(&d_xs, (size_t) T * (H / 32) * 4), "cudaMalloc xs");
    check(cudaMalloc(&d_ids, (size_t) n * 4), "cudaMalloc ids");
    check(cudaMalloc(&d_ptr, (size_t) n * 8), "cudaMalloc grp_ptr");
    check(cudaMalloc(&d_start, (size_t) (n + 1) * 4), "cudaMalloc grp_start");
    check(cudaMalloc(&d_counts, 4 * 4), "cudaMalloc counts");
    check(cudaMalloc(&d_dst, (size_t) n * 4), "cudaMalloc dst");
    check(cudaMalloc(&d_tok, (size_t) n * 4), "cudaMalloc tok");
    check(cudaMalloc(&d_scratch, scratch_bytes), "cudaMalloc scratch");
    check(cudaMalloc(&d_out, (size_t) n * H * 4), "cudaMalloc out");
    check(cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "copy blobs");
    check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_ids, ids.data(), (size_t) n * 4, cudaMemcpyHostToDevice), "copy ids");
    check(cudaMemset(d_out, 0xFF, (size_t) n * H * 4), "poison out");      // NaN: every row must be written

    sk::quantize_q8_0_scaled(d_x, d_xq, d_xs, (int64_t) T * H, nullptr);
    sk::moe_group_resident(d_ids, n, K, d_blobs, (int64_t) BLOB, d_ptr, d_start, d_counts, d_dst, d_tok, nullptr);
    sk::moe_grouped_s2(d_ptr, d_start, d_counts, d_dst, d_tok, n, n, d_xq, d_xs, d_scratch, d_out, nullptr);
    check(cudaDeviceSynchronize(), "kernels");

    std::vector<uint8_t> xq((size_t) T * (H / 32) * 34), scratch(scratch_bytes);
    std::vector<float> xs((size_t) T * (H / 32));
    std::vector<int32_t> counts(4), dst(n), tok(n);
    Result r;
    r.out.resize((size_t) n * H);
    check(cudaMemcpy(xq.data(), d_xq, xq.size(), cudaMemcpyDeviceToHost), "read xq");
    check(cudaMemcpy(xs.data(), d_xs, xs.size() * 4, cudaMemcpyDeviceToHost), "read xs");
    check(cudaMemcpy(counts.data(), d_counts, 16, cudaMemcpyDeviceToHost), "read counts");
    check(cudaMemcpy(dst.data(), d_dst, (size_t) n * 4, cudaMemcpyDeviceToHost), "read dst");
    check(cudaMemcpy(tok.data(), d_tok, (size_t) n * 4, cudaMemcpyDeviceToHost), "read tok");
    check(cudaMemcpy(scratch.data(), d_scratch, scratch_bytes, cudaMemcpyDeviceToHost), "read scratch");
    check(cudaMemcpy(r.out.data(), d_out, r.out.size() * 4, cudaMemcpyDeviceToHost), "read out");
    for (void* p : {(void*) d_blobs, (void*) d_x, (void*) d_xq, (void*) d_xs, (void*) d_ids, (void*) d_ptr,
                    (void*) d_start, (void*) d_counts, (void*) d_dst, (void*) d_tok, (void*) d_scratch, (void*) d_out})
        cudaFree(p);
    if (counts[1] != n) {
        std::fprintf(stderr, "moe_group_resident: %d entries, expected %d\n", counts[1], n);
        std::exit(1);
    }
    // the scratch as moe_grouped_s2 carves it
    const uint64_t gu_bytes = ((uint64_t) n * (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n * (FF / 32) * 34 + 15) & ~15ull;
    const float* h_gpu = (const float*) scratch.data();
    const uint8_t* hq = scratch.data() + gu_bytes;
    const float* hs = (const float*) (scratch.data() + gu_bytes + q8_bytes);

    for (int e = 0; e < n; ++e) {
        const int t = tok[e], row = dst[e], ex = ids[row];
        const uint8_t* blob = blobs.data() + (size_t) ex * BLOB;
        const uint8_t* xt = xq.data() + (size_t) t * (H / 32) * 34;
        const float* st = xs.data() + (size_t) t * (H / 32);
        // stage 1: gate (row slot 2r) and up (2r+1), then silu(gate) * up
        std::vector<double> h(FF), tol(FF);
        double hmax = 0;
        for (int rr = 0; rr < FF; ++rr) {
            double gu[2], mag[2];
            for (int half = 0; half < 2; ++half) {
                const int slot = 2 * rr + half;
                const uint8_t* codes = blob + (size_t) slot * ROW_GU;
                const uint8_t* sc = blob + O_GU_SCALES + (size_t) slot * SC_GU * 2;
                double acc = 0, m = 0;
                for (int c = 0; c < H / 32; ++c) {
                    const double dw = strata::fp16_to_fp32(rd16(sc + (c >> 1) * 2));
                    const double v = dw * st[c] * chunk_dot(codes + c * 8, (const int8_t*) (xt + c * 34 + 2));
                    acc += v;
                    m += std::fabs(v);
                }
                gu[half] = acc;
                mag[half] = m;
            }
            const double sig = 1.0 / (1.0 + std::exp(-gu[0]));
            h[rr] = gu[0] * sig * gu[1];
            // float accumulation over 80 chunks (~1e-6 of the magnitudes) and __expf, propagated
            tol[rr] = 1e-5 * (mag[0] * std::fabs(gu[1]) + mag[1] * std::fabs(gu[0] * sig)) + 1e-5 * std::fabs(h[rr]);
            hmax = std::max(hmax, std::fabs(h[rr]));
        }
        for (int rr = 0; rr < FF; ++rr) {
            const double g = h_gpu[(size_t) e * FF + rr];
            const double err = std::fabs(g - h[rr]) / (tol[rr] + 1e-6 * hmax + 1e-30);
            r.h_err = std::max(r.h_err, err);
            if (!(err <= 1.0)) ++r.bad;
        }
        r.h_max = std::max(r.h_max, hmax);
        // stage 2: down, from the kernel's own quantized intermediate
        const uint8_t* he = hq + (size_t) e * (FF / 32) * 34;
        const float* hse = hs + (size_t) e * (FF / 32);
        for (int rr = 0; rr < H; ++rr) {
            const uint8_t* codes = blob + O_D_CODES + (size_t) rr * ROW_D;
            const uint8_t* sc = blob + O_D_SCALES + (size_t) rr * SC_D * 2;
            double acc = 0, m = 0;
            for (int c = 0; c < FF / 32; ++c) {
                const double dw = strata::fp16_to_fp32(rd16(sc + (c >> 1) * 2));
                const double v = dw * hse[c] * chunk_dot(codes + c * 8, (const int8_t*) (he + c * 34 + 2));
                acc += v;
                m += std::fabs(v);
            }
            const double g = r.out[(size_t) row * H + rr];
            const double err = std::fabs(g - acc) / (1e-5 * m + 1e-30);
            r.out_err = std::max(r.out_err, err);
            if (!(err <= 1.0)) ++r.bad;
        }
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else {
            std::fprintf(stderr, "usage: s2_grouped_parity [--selftest]\n");
            return 2;
        }
    }
    std::mt19937 rng(52);
    // blobs: random codes; fp16 scales finite and normal (exponent 8..13: 2^-7 .. 2^-2), random sign
    std::vector<uint8_t> neg((size_t) NEXP * BLOB);
    for (auto& b : neg) b = (uint8_t) (rng() & 0xFF);
    long n_scales = 0, n_neg = 0;
    for (int ex = 0; ex < NEXP; ++ex) {
        uint8_t* blob = neg.data() + (size_t) ex * BLOB;
        for (size_t off = O_GU_SCALES; off < BLOB; off += 2) {
            const uint16_t sign = (ex % 4 == 3) ? 1 : (uint16_t) (rng() & 1);   // every 4th expert: all negative
            const uint16_t bits = (uint16_t) ((sign << 15) | ((8 + rng() % 6) << 10) | (rng() & 0x3FF));
            blob[off] = (uint8_t) (bits & 0xFF);
            blob[off + 1] = (uint8_t) (bits >> 8);
            ++n_scales;
            n_neg += sign;
        }
    }
    std::vector<uint8_t> pos = neg;                  // the same blobs with every scale made positive
    for (int ex = 0; ex < NEXP; ++ex)
        for (size_t off = O_GU_SCALES; off < BLOB; off += 2) pos[(size_t) ex * BLOB + off + 1] &= 0x7F;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = nd(rng);
    std::vector<int32_t> ids((size_t) T * K);        // per token 10 distinct experts: groups of up to T entries
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> all(NEXP);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + K, ids.begin() + (size_t) t * K);
    }

    const Result a = run(neg, x, ids), b = run(pos, x, ids);
    double diff = 0, mag = 0;
    long nan = 0;
    for (size_t i = 0; i < a.out.size(); ++i) {
        nan += std::isnan(a.out[i]) || std::isnan(b.out[i]);
        diff = std::max(diff, (double) std::fabs(a.out[i] - b.out[i]));
        mag = std::max(mag, (double) std::fabs(b.out[i]));
    }
    std::printf("s2_grouped: %d tokens x top-%d over %d experts, %ld of %ld scales negative\n", T, K, NEXP, n_neg,
                n_scales);
    std::printf("  signed scales:   swiglu worst %.3f of tolerance, down worst %.3f, %ld out of tolerance\n", a.h_err,
                a.out_err, a.bad);
    std::printf("  positive scales: swiglu worst %.3f of tolerance, down worst %.3f, %ld out of tolerance\n", b.h_err,
                b.out_err, b.bad);
    std::printf("  non-vacuity: outputs differ by up to %.4g (largest %.4g), %ld NaN\n", diff, mag, nan);
    if (a.bad || b.bad) {
        std::fprintf(stderr, "MISMATCH: moe_grouped_s2 disagrees with the reference%s\n",
                     a.bad && !b.bad ? " only with negative scales" : "");
        return 1;
    }
    if (n_neg * 4 < n_scales || nan || !(diff > 0.1 * mag) || a.h_max <= 0 || b.h_max <= 0) {
        std::fprintf(stderr, "VACUOUS: the comparison could not have told a dropped sign from a kept one\n");
        return 1;
    }
    if (selftest) std::printf("s2_grouped_parity OK\n");
    return 0;
}
