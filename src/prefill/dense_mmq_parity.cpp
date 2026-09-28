// src/prefill/dense_mmq_parity.cpp - issue #39: the prompt path's dense projections through MMQ
// (Gemm::native_mmq, `--prefill-dense-mmq`) against the FP16 cuBLAS path (Gemm::native) and an FP64 host product of
// the same dequantized weights.  Synthetic weights quantized by ggml's own quantizers; no model, needs the GPU.
//
// For every covered type and shape:
//   - the result is finite and close to the exact product: its only error is the q8_1 rounding of the activations
//     (~0.5% relative RMS on these data; the bound is 3%);
//   - for the D4-layout types it matches the product with the activations rounded as MMQ's quantizer rounds them
//     (emulated on the host) to FP32 summation order;
//   - rows of Y past N (ldy > N) are untouched, and a second run is bitwise identical;
//   - a weight whose rows are not a multiple of 512 values (the shared expert's down, K = 640) is read from a padded
//     copy: NaN bytes planted right after the matrix must not reach the output;
//   - token slices (a small scratch) agree with one pass up to FP32 summation order;
//   - an uncovered type (IQ1_M) or beta != 0 is declined with nothing written.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::prefill::Gemm;
namespace mmq = strata::prefill::mmq;

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_mmq_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void expect(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool d4_layout(int t) {   // the q8_1 activation layout MMQ uses against this weight type: one float scale per 32
    return !(t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K);
}

// MMQ's quantizer (ggml-cuda quantize.cu, quantize_mmq_q8_1): per 32 values, d = amax / 127, q = round(x * 127 / amax)
void q8_1_emulate(const float* x, float* out, int64_t K) {
    for (int64_t b = 0; b < K; b += 32) {
        float amax = 0.0f;
        for (int64_t i = b; i < b + 32 && i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
        if (amax == 0.0f) { for (int64_t i = b; i < b + 32 && i < K; ++i) out[i] = 0.0f; continue; }
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (int64_t i = b; i < b + 32 && i < K; ++i) out[i] = std::round(x[i] * d_inv) * d;
    }
}

// Y[t, n] = sum_k W[n, k] X[t, k] in FP64 at the sampled (t, n) (all of them for a small product), 4 threads
struct Ref {
    std::vector<int64_t> t, n;
    std::vector<double> y;
};
Ref reference(const std::vector<float>& W, const std::vector<float>& X, int64_t T, int64_t N, int64_t K) {
    Ref r;
    if (T * N <= 200000) {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t n = 0; n < N; ++n) { r.t.push_back(t); r.n.push_back(n); }
    } else {
        std::mt19937 rng(7);
        for (int i = 0; i < 4096; ++i) { r.t.push_back((int64_t) (rng() % T)); r.n.push_back((int64_t) (rng() % N)); }
    }
    r.y.assign(r.t.size(), 0.0);
    std::vector<std::thread> th;
    const size_t parts = 4;
    for (size_t p = 0; p < parts; ++p)
        th.emplace_back([&, p] {
            for (size_t i = p; i < r.t.size(); i += parts) {
                const float* w = W.data() + r.n[i] * K;
                const float* x = X.data() + r.t[i] * K;
                double s = 0.0;
                for (int64_t k = 0; k < K; ++k) s += (double) w[k] * (double) x[k];
                r.y[i] = s;
            }
        });
    for (auto& h : th) h.join();
    return r;
}

double rel_rms(const Ref& r, const std::vector<float>& Y, int64_t ldy) {
    double e = 0.0, s = 0.0;
    for (size_t i = 0; i < r.t.size(); ++i) {
        const double d = (double) Y[(size_t) (r.t[i] * ldy + r.n[i])] - r.y[i];
        e += d * d;
        s += r.y[i] * r.y[i];
    }
    return s > 0.0 ? std::sqrt(e / s) : std::sqrt(e);
}

bool all_finite(const std::vector<float>& Y, int64_t T, int64_t N, int64_t ldy) {
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n)
            if (!std::isfinite(Y[(size_t) (t * ldy + n)])) return false;
    return true;
}

struct Case {
    int type;
    int64_t N, K, T, ldy;
    bool x16 = false;             // quantize from the FP16 activations (the widened path)
    bool poison = false;          // NaN bytes right after the weight
    size_t scratch_bytes = 0;     // 0: the prompt path's 64 MiB
    bool vs_cublas = true;
};

// a scratch with room for `rows` activation rows next to what Gemm::native_mmq keeps there (the weight's padded copy
// when K is not a multiple of 512, the q8_1 tail, its alignment slack)
size_t scratch_for(int type, int64_t N, int64_t K, int64_t rows, bool x16) {
    const size_t q8_row = mmq::q8_bytes(1, K) - mmq::q8_bytes(0, K);
    const size_t w = K % 512 != 0 ? (mmq::matrix_bytes(type, N, K) + 4096 + 255) / 256 * 256 : 0;
    return w + mmq::q8_bytes(0, K) + 512 + (size_t) rows * (q8_row + (x16 ? (size_t) K * 4 : 0));
}

std::string label(const Case& c) {
    char b[160];
    std::snprintf(b, sizeof b, "%s N=%lld K=%lld T=%lld ldy=%lld%s%s%s", ggml_type_name((ggml_type) c.type),
                  (long long) c.N, (long long) c.K, (long long) c.T, (long long) c.ldy, c.x16 ? " x16" : "",
                  c.poison ? " nan-tail" : "", c.scratch_bytes ? " sliced" : "");
    return b;
}

void run_case(const Case& c, cudaStream_t s, mmq::Context& ctx) {
    std::printf("%s\n", label(c).c_str());
    const int64_t N = c.N, K = c.K, T = c.T, ldy = c.ldy;
    std::mt19937 rng((uint32_t) (c.type * 1000003 + N * 31 + K * 7 + T));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // weights: Gaussian, quantized by ggml and dequantized back (the exact values both paths multiply)
    std::vector<float> Wf((size_t) (N * K));
    for (float& v : Wf) v = nd(rng);
    const size_t row_bytes = ggml_row_size((ggml_type) c.type, K), w_bytes = row_bytes * (size_t) N;
    std::vector<uint8_t> Wq(w_bytes);
    std::vector<float> imatrix((size_t) K, 1.0f);
    ggml_quantize_chunk((ggml_type) c.type, Wf.data(), Wq.data(), 0, N, K, imatrix.data());
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) c.type);
    std::vector<float> Wd((size_t) (N * K));
    tr->to_float(Wq.data(), Wd.data(), N * K);
    // activations: FP32, their FP16 rounding (what `native` multiplies, and the x16 case's input)
    std::vector<float> X((size_t) (T * K)), Xh((size_t) (T * K));
    std::vector<uint16_t> X16((size_t) (T * K));
    for (size_t i = 0; i < X.size(); ++i) {
        X[i] = nd(rng) * (i % 97 == 0 ? 4.0f : 1.0f);
        const ggml_fp16_t h = ggml_fp32_to_fp16(X[i]);
        std::memcpy(&X16[i], &h, 2);
        Xh[i] = ggml_fp16_to_fp32(h);
    }
    const std::vector<float>& Xin = c.x16 ? Xh : X;   // what MMQ quantizes
    std::vector<float> Xq((size_t) (T * K));
    for (int64_t t = 0; t < T; ++t) q8_1_emulate(Xin.data() + t * K, Xq.data() + t * K, K);

    // device buffers: the weight (then 4 KiB of 0xFF - FP16 NaN - when poisoned), X32, X16, Y (sentinel-filled)
    uint8_t* dW = nullptr;
    float *dX = nullptr, *dY = nullptr;
    uint16_t* dX16 = nullptr;
    ck(cudaMalloc(&dW, w_bytes + 4096), "malloc W");
    ck(cudaMemset(dW, c.poison ? 0xFF : 0x00, w_bytes + 4096), "memset W");
    ck(cudaMemcpy(dW, Wq.data(), w_bytes, cudaMemcpyHostToDevice), "copy W");
    ck(cudaMalloc(&dX, X.size() * 4), "malloc X");
    ck(cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice), "copy X");
    ck(cudaMalloc(&dX16, X16.size() * 2), "malloc X16");
    ck(cudaMemcpy(dX16, X16.data(), X16.size() * 2, cudaMemcpyHostToDevice), "copy X16");
    const size_t ny = (size_t) (T * ldy);
    ck(cudaMalloc(&dY, ny * 4), "malloc Y");
    const float sentinel = 1234.5f;
    std::vector<float> fill(ny, sentinel);
    auto reset_y = [&] { ck(cudaMemcpy(dY, fill.data(), ny * 4, cudaMemcpyHostToDevice), "reset Y"); };

    const int64_t scratch_elems = c.scratch_bytes ? (int64_t) (c.scratch_bytes / 2) : (32ll << 20);
    Gemm gm;
    std::string err;
    if (!gm.init(s, scratch_elems, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(2); }
    gm.set_mmq(&ctx);

    std::vector<float> Y1(ny), Y2(ny), Yc(ny);
    const float* x32 = c.x16 ? nullptr : dX;
    reset_y();
    const bool took = gm.native_mmq(x32, dX16, c.type, dW, dY, T, N, K, ldy);
    ck(cudaStreamSynchronize(s), "mmq run");
    ck(cudaMemcpy(Y1.data(), dY, ny * 4, cudaMemcpyDeviceToHost), "read Y");
    expect(took, "native_mmq takes the call");
    if (!took) { cudaFree(dW); cudaFree(dX); cudaFree(dX16); cudaFree(dY); return; }
    reset_y();
    gm.native_mmq(x32, dX16, c.type, dW, dY, T, N, K, ldy);
    ck(cudaStreamSynchronize(s), "mmq rerun");
    ck(cudaMemcpy(Y2.data(), dY, ny * 4, cudaMemcpyDeviceToHost), "read Y");
    expect(std::memcmp(Y1.data(), Y2.data(), ny * 4) == 0, "a second run is bitwise identical");
    expect(all_finite(Y1, T, N, ldy), "every output is finite");
    bool untouched = true;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = N; n < ldy; ++n) untouched &= Y1[(size_t) (t * ldy + n)] == sentinel;
    expect(untouched, "columns past N are untouched");

    const Ref exact = reference(Wd, Xin, T, N, K);
    const double e_exact = rel_rms(exact, Y1, ldy);
    char b[160];
    std::snprintf(b, sizeof b, "vs the exact product: rel RMS %.3e (bound 3e-2)", e_exact);
    expect(e_exact < 3e-2, b);
    if (d4_layout(c.type)) {
        const Ref emu = reference(Wd, Xq, T, N, K);
        const double e_emu = rel_rms(emu, Y1, ldy);
        std::snprintf(b, sizeof b, "vs the q8_1-rounded product: rel RMS %.3e (bound 2e-3)", e_emu);
        expect(e_emu < 2e-3, b);
    }
    if (c.vs_cublas && !c.poison) {
        reset_y();
        gm.native(dX16, c.type, dW, dY, T, N, K, ldy);
        ck(cudaStreamSynchronize(s), "cublas run");
        ck(cudaMemcpy(Yc.data(), dY, ny * 4, cudaMemcpyDeviceToHost), "read Y");
        const Ref exact16 = c.x16 ? exact : reference(Wd, Xh, T, N, K);
        const double e16 = rel_rms(exact16, Yc, ldy);
        std::snprintf(b, sizeof b, "(the FP16 cuBLAS path vs its exact product: rel RMS %.3e, bound 5e-3)", e16);
        expect(e16 < 5e-3, b);
    }
    if (c.scratch_bytes) {   // the same product in one pass, with the full scratch
        Gemm big;
        if (!big.init(s, 32ll << 20, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(2); }
        big.set_mmq(&ctx);
        reset_y();
        big.native_mmq(x32, dX16, c.type, dW, dY, T, N, K, ldy);
        ck(cudaStreamSynchronize(s), "one pass");
        ck(cudaMemcpy(Yc.data(), dY, ny * 4, cudaMemcpyDeviceToHost), "read Y");
        double e = 0.0, n2 = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t n = 0; n < N; ++n) {
                const double d = (double) Y1[(size_t) (t * ldy + n)] - Yc[(size_t) (t * ldy + n)];
                e += d * d;
                n2 += (double) Yc[(size_t) (t * ldy + n)] * Yc[(size_t) (t * ldy + n)];
            }
        std::snprintf(b, sizeof b, "token slices vs one pass: rel RMS %.3e (bound 1e-5)", n2 > 0 ? std::sqrt(e / n2) : 0.0);
        expect(n2 > 0 && std::sqrt(e / n2) < 1e-5, b);
    }
    cudaFree(dW); cudaFree(dX); cudaFree(dX16); cudaFree(dY);
}

void declined(cudaStream_t s, mmq::Context& ctx) {
    std::printf("declined calls\n");
    Gemm gm;
    std::string err;
    if (!gm.init(s, 1 << 20, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(2); }
    // (the weight pointer is never read: every call below must return before launching anything)
    float* dY = nullptr;
    ck(cudaMalloc(&dY, 4), "malloc");
    uint16_t* dX = nullptr;
    ck(cudaMalloc(&dX, 256 * 2), "malloc");
    expect(!gm.native_mmq(nullptr, dX, GGML_TYPE_Q8_0, dY, dY, 1, 1, 256), "no context set: declined");
    gm.set_mmq(&ctx);
    expect(!gm.native_mmq(nullptr, dX, GGML_TYPE_IQ1_M, dY, dY, 1, 1, 256), "IQ1_M: declined");
    expect(!gm.native_mmq(nullptr, dX, GGML_TYPE_Q8_0, dY, dY, 1, 1, 256, 0, 1.0f), "beta = 1: declined");
    Gemm tiny;
    if (!tiny.init(s, 4096, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(2); }
    tiny.set_mmq(&ctx);
    expect(!tiny.native_mmq(nullptr, dX, GGML_TYPE_Q8_0, dY, dY, 1, 1, 256), "a scratch below one row: declined");
    ck(cudaStreamSynchronize(s), "declined");
    cudaFree(dY);
    cudaFree(dX);
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::fprintf(stderr, "dense_mmq_parity: no CUDA device\n");
        return 2;
    }
    if (!mmq::built()) {
        std::fprintf(stderr, "dense_mmq_parity: this build has no MMQ\n");
        return 2;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    mmq::Context ctx;
    const int types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL,
                         GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS,
                         GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS};
    for (int t : types) {
        if (!mmq::dense_supported(t)) { expect(false, std::string(ggml_type_name((ggml_type) t)) + " is covered"); continue; }
        // a row length that is a multiple of 512 (no copy) and, for the small blocks, the shared expert's down
        // shape (K = 640: the padded copy), with NaN bytes after the weight
        run_case({t, 256, 1024, 300, 256}, s, ctx);
        if (ggml_blck_size((ggml_type) t) <= 64) run_case({t, 384, 640, 77, 384, false, true}, s, ctx);
    }
    // N not a multiple of 128 (MMQ's bounds-checked tiles), one token, a whole tile, a tile and one
    run_case({GGML_TYPE_Q8_0, 200, 512, 129, 200}, s, ctx);
    run_case({GGML_TYPE_Q6_K, 200, 512, 129, 200}, s, ctx);
    run_case({GGML_TYPE_Q8_0, 128, 512, 1, 128}, s, ctx);
    run_case({GGML_TYPE_Q8_0, 128, 512, 128, 128}, s, ctx);
    // the output inside a wider row (ldy > N)
    run_case({GGML_TYPE_Q8_0, 256, 1024, 300, 320}, s, ctx);
    // the FP16 activations widened (the attention output and the shared expert's hidden state)
    run_case({GGML_TYPE_Q8_0, 256, 1024, 300, 256, true}, s, ctx);
    run_case({GGML_TYPE_Q4_K, 256, 1024, 300, 256, true}, s, ctx);
    run_case({GGML_TYPE_Q5_0, 2560, 640, 300, 2560, true, true}, s, ctx);
    // token slices: a scratch of 100 rows (X32) / 20 rows (widened X16) / 66 rows beside a weight copy, vs one pass
    run_case({GGML_TYPE_Q8_0, 256, 1024, 300, 256, false, false, scratch_for(GGML_TYPE_Q8_0, 256, 1024, 100, false)}, s, ctx);
    run_case({GGML_TYPE_Q4_K, 256, 1024, 300, 256, true, false, scratch_for(GGML_TYPE_Q4_K, 256, 1024, 20, true)}, s, ctx);
    run_case({GGML_TYPE_Q8_0, 384, 640, 300, 384, true, true, scratch_for(GGML_TYPE_Q8_0, 384, 640, 66, true)}, s, ctx);
    // the prompt path's shapes at a small chunk: GDN qkv, QSA q (sampled host reference)
    run_case({GGML_TYPE_Q8_0, 10240, 2560, 520, 10240}, s, ctx);
    run_case({GGML_TYPE_Q4_K, 12288, 2560, 300, 12288}, s, ctx);
    run_case({GGML_TYPE_Q8_0, 2560, 6144, 520, 2560, true}, s, ctx);
    declined(s, ctx);
    cudaStreamDestroy(s);
    if (g_fail) {
        std::printf("dense_mmq_parity: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("dense_mmq_parity: all checks passed\n");
    return 0;
}
