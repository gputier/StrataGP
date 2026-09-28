// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs, a tensor-core GEMM through cuBLAS with FP32 accumulation.  The pack's BF16 tensors are multiplied
// as they are, with activations rounded to BF16 (`bf16`); a quantized weight is dequantized from its native GGUF
// blocks to FP16 into a reusable scratch (`dequant_f16`) right before the product, with activations rounded to FP16
// (`native`).  Issue #39, opt-in (`--prefill-dense-mmq`): `native_mmq` multiplies the GGUF blocks as they are through
// llama.cpp's MMQ kernels instead - activations rounded to q8_1, int8 tensor cores.
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill {

namespace mmq { class Context; }

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: FP16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Issue #39: the product of `native` through MMQ (`set_mmq`): W stays in its GGUF blocks and X is rounded to
    /// q8_1 - from `X32` (FP32, K floats per row) when the caller has it, else from the FP16 `X` widened.  The
    /// scratch holds the quantized rows (in token slices when a chunk's do not fit) and, when MMQ's K tiles would
    /// read past the weight's end, a copy of it with a zeroed tail.  False, with nothing launched, when MMQ does not
    /// cover the call (no context, the weight type, beta != 0, a scratch too small): the caller runs `native`.
    bool native_mmq(const float* X32, const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T,
                    int64_t N, int64_t K, int64_t ldy = 0, float beta = 0.0f);
    /// The MMQ launch context of `native_mmq` (not owned; null, the default: `native_mmq` declines every call).
    void set_mmq(mmq::Context* ctx) { mmq_ = ctx; }

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    mmq::Context* mmq_ = nullptr;
};



}  // namespace strata::prefill
