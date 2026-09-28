// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Issue #39: MMQ's quantizer reads FP32 rows; an FP16 activation (no FP32 copy kept) is widened first.
__global__ void widen_kernel(const __half* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// MMQ reads a weight row in K tiles of 256 values (llama.cpp pads every quantized row to MATRIX_ROW_PADDING = 512
// for it): with a row length that is not a multiple of 512 the last row's tile runs past the matrix, into whatever
// follows it.  Those values meet zero activations - harmless only if they decode to finite numbers - so such a
// weight is copied into the scratch with this many zeroed bytes after it (512 values of the widest block: 544 B).
constexpr int64_t kRowPad = 512;
constexpr size_t kTail = 4096;

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

bool Gemm::native_mmq(const float* X32, const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T,
                      int64_t N, int64_t K, int64_t ldy, float beta) {
    if (mmq_ == nullptr || beta != 0.0f || K % 4 != 0 || !mmq::dense_supported(ggml_type)) return false;
    if (T <= 0 || N <= 0) return true;
    if (ldy <= 0) ldy = N;
    const cudaStream_t s = (cudaStream_t) stream_;
    // the scratch: [W again with a zeroed tail, if its last K tile overruns] [the q8_1 rows] [X widened, if FP16]
    uint8_t* base = (uint8_t*) scratch_;
    const size_t cap = (size_t) scratch_elems_ * 2;
    const bool copy_w = K % kRowPad != 0;
    const size_t w_bytes = mmq::matrix_bytes(ggml_type, N, K);
    const size_t w_region = copy_w ? align256(w_bytes + kTail) : 0;
    // q8_bytes is linear in the rows: a per-row part and a fixed tail (MMQ's last column tile reads past the rows)
    const size_t q8_row = mmq::q8_bytes(1, K) - mmq::q8_bytes(0, K), q8_tail = mmq::q8_bytes(0, K);
    const size_t x_row = X32 != nullptr ? 0 : (size_t) K * 4;
    const size_t fixed = w_region + q8_tail + 2 * 256;
    if (base == nullptr || cap <= fixed) return false;
    int64_t rows = (int64_t) ((cap - fixed) / (q8_row + x_row));
    if (rows <= 0) return false;
    if (rows < T && rows >= 128) rows -= rows % 128;   // whole column tiles in every slice but the last
    rows = std::min(rows, T);
    const void* w = W_blocks;
    if (copy_w) {
        ck(cudaMemcpyAsync(base, W_blocks, w_bytes, cudaMemcpyDeviceToDevice, s), "mmq weight copy");
        ck(cudaMemsetAsync(base + w_bytes, 0, w_region - w_bytes, s), "mmq weight tail");
        w = base;
    }
    void* xq = base + w_region;
    float* xf = (float*) (base + w_region + align256(mmq::q8_bytes(rows, K)));
    // token slices (one when the chunk's rows fit): every output row depends on its own activation row only
    for (int64_t t0 = 0; t0 < T; t0 += rows) {
        const int64_t n = std::min(rows, T - t0);
        const float* x = X32 != nullptr ? X32 + t0 * K : xf;
        if (X32 == nullptr) {
            const int64_t e = n * K;
            widen_kernel<<<(unsigned) ((e + 255) / 256), 256, 0, s>>>((const __half*) (X + t0 * K), xf, e);
            ck(cudaGetLastError(), "widen");
        }
        mmq::quantize(x, nullptr, xq, ggml_type, K, K, n, s);
        mmq_->dense(w, ggml_type, N, K, xq, n, Y + t0 * ldy, ldy, s);
    }
    return true;
}

}  // namespace strata::prefill
