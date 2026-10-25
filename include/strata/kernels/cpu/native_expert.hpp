// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized gate/up activation any native layer uses (3072 values as Q8_K: 12 x 292 = 3504).
inline constexpr size_t kNativeActBytes = 4096;
/// Bytes of the largest quantized down activation (1024 values as Q8_K: 4 x 292 = 1168, or Q8_0: 32 x 34 = 1088).
inline constexpr size_t kNativeHBytes = 2048;
/// The widest expert intermediate (n_ff) a native layer may have: the pool's gate/up row buffers are this wide.
inline constexpr int64_t kNativeMaxFF = 1024;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    /// NVFP4 (ggml type 40) only: the blob ends with {s_gate, s_up, s_down, 0} at this offset, the per-expert
    /// weight_scale_2 the block format has no room for (tools/iq_pack.py). 0 = no tail. Each scale goes on its own
    /// projection's FP32 output: gate * s_gate, up * s_up, down * s_down. Do not fold s_down into up: the hidden
    /// drops to ~1e-5 and its q8 block scale becomes an FP16 subnormal (expert error 2-12% instead of 1.1%).
    size_t tail_off = 0;
    /// Set by an engine that wants the one-token expert rows: IQ3_XXS gate/up served by iq512_gu_rows for a lone token
    /// too, an IQ4_XS down served by iq512_rows for any token count, and an IQ4_NL down served by iq4nl256_down_rows (AVX-2)
    /// for a lone token too.  Off (the default) keeps the selection the Flash-Next path has always had: ggml-cpu's dot
    /// for one token, and for an IQ4_XS down always.  Their rows round differently from ggml-cpu's (only the order of the
    /// float additions), so it is a choice of the engine, per layer; the q36 engine sets it only under
    /// STRATA_Q36_ONE_TOKEN_CPU, since on an AVX-512 CPU the one-token kernels measure slower than ggml-cpu's dots.  Once
    /// set, STRATA_IQ_MT_MIN takes all three back: the gate/up and IQ4_NL rows go by that variable's token threshold as
    /// before, and an IQ4_XS down goes back to ggml-cpu's dot for any token count.  STRATA_NO_IQ512=1 takes back the two
    /// AVX-512 kernels (IQ3_XXS gate/up, IQ4_XS down) but NOT the old selection: the gate/up rows then run on the AVX-2
    /// multi-token kernel instead of the AVX-512 one, and STRATA_NO_IQ4NL=1 removes the AVX-2 IQ4_NL down kernel, so
    /// neither reproduces a build without the flag; only STRATA_IQ_MT_MIN (or leaving the flag off) does.
    bool one_token_512 = false;
};
constexpr int kNvfp4Type = 40;
constexpr int kQ2_0Type = 42;
constexpr size_t kNvfp4Tail = 16;

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type, n_ff is above kNativeMaxFF,
/// or a Q2_0 (type 42) gate/up has n_embd above cpu::H.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.  `one_token_512`: NativeFmt::one_token_512.
int native_gu_mt_min(int gu_type, bool one_token_512 = false);
/// Whether a group of native_gu_mt_min tokens or more of this gate/up type takes the AVX-512 kernel in ggml's float
/// order (#74: IQ3_XXS where a lone token takes ggml-cpu's AVX2+FMA dot; STRATA_NO_IQ512_GGML_ORDER=1 turns it off).
bool native_gu_ggml_order(int gu_type, bool one_token_512 = false);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
