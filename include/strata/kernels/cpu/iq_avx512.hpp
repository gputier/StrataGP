// include/strata/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; IQ4_XS for iq512_rows only) against Q8_K activations (ggml's
// block_q8_K).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// The formats iq512_rows serves that iq512_gu_rows / iq512_supported leave to the other kernels: IQ4_XS (type 23), a
/// down projection (no AVX-512 gate/up kernel for it: an IQ4_XS gate/up keeps ggml-cpu's dot on an AVX-512 CPU).
bool iq512_down_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
