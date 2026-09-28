// include/strata/kernels/fused_gdn.hpp - plan v0.3 P3: the gated delta-net step and its output norm in ONE kernel.
//
// The native step (after llama.cpp's gated_delta_net.cu) gives each warp one (head, column) and strides its 32
// lanes over the 128 ROWS of the state, which is stored `[row][head][col]`: every lane's load is 24.5 KB from its
// neighbour's, so each 32 B sector delivers 4 useful bytes - 47.6 us per layer for 6.3 MB of state traffic.  Here a
// block owns one whole head, its 512 threads are (row group of 32, column), and a row's 128 columns are one
// contiguous 512 B load.  With the head in one block, the per-head RMS norm, the norm weight and the sigmoid(z)
// gate of `gdn_out_norm` run in the same kernel.
//
//   S <- g S + k (beta (v - g S^T k))^T      (g = exp(gate[head]), per value head; q/k shared by h_v / h_k heads)
//   o  = (S^T q) / sqrt(128)
//   y  = rmsnorm(o) * gamma * sigmoid(z)
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The 4-tap causal conv + SiLU over all `channels` (history is `[channel][3]`, updated), then the L2 norm of
/// each of the first `qk_heads` 128-channel heads (q then k), eps as the native `l2_norm` (x / sqrt(sum + eps)).
/// One block per 128-channel head; replaces conv_silu + two l2_norm launches.
void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream);

/// alpha and beta, both `(h_v, n_embd)` BF16 against the FP32 activation, with their epilogues:
/// gate = softplus(alpha + dt) * ssm_a,  beta = sigmoid(beta).  One warp per row; replaces two MMVF launches and
/// two elementwise kernels.
void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream);

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream);

// ---- Issue #53 (S5), OPT-IN: the recurrent state stored as BF16 (`--gdn-state-bf16`, default off).
//
// The step reads and writes the whole state every token (6.3 MB per layer in FP32); in BF16 that traffic halves.
// Only the STORAGE changes: a load widens (exact), all the arithmetic is the FP32 arithmetic above, and a store
// rounds to nearest even.  A layer's state slice keeps its FP32 size and its layout ([row][head][col]); the BF16
// state is the first half of the slice and the conv history stays where it was, so the session arena, the
// checkpoints and the plan's budget do not move.  Rounding once per step is a numerical change (drift over long
// sequences is the risk the issue names), hence opt-in; `gdn_state_bf16_parity` measures it against FP32.
//
// The switch is process-wide and must be set before any session or graph is captured, like
// `native_gdn_set_enabled`: `fused_gdn_step_norm` and `gdn_step_norm_multi` then read their `float* state` as
// the BF16 slice.  Only the fused native step and the verify/commit kernels know the BF16 form; the other token
// steps refuse it (`layer.cpp`), and the prompt path runs its FP32 recurrence on a widened copy
// (`gdn_state_widen` / `gdn_state_narrow`).
void gdn_state_bf16_set_enabled(bool enabled);
bool gdn_state_bf16_enabled();

/// `fused_gdn_step_norm` on a BF16 state, whatever the switch says (the parity test runs both forms side by side).
void fused_gdn_step_norm_bf16(uint16_t* state, const float* q, const float* k, const float* v, const float* gate,
                              const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                              int h_v, void* stream);

#if defined(__CUDACC__)
/// Round to nearest even, as `bf16_bits.hpp` does, except that a NaN stays a NaN instead of carrying into -0.
__device__ __forceinline__ uint16_t gdn_bf16_rne(float v) {
    const uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x0040u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/// One state element's load and store for either storage: FP32 is the plain access, so a kernel templated on it
/// compiles to what it was before the BF16 form existed.
__device__ __forceinline__ float gdn_state_load(const float* p) { return *p; }
__device__ __forceinline__ float gdn_state_load(const uint16_t* p) { return __uint_as_float((uint32_t) *p << 16); }
__device__ __forceinline__ void gdn_state_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void gdn_state_store(uint16_t* p, float v) { *p = gdn_bf16_rne(v); }
#endif

/// dst[i] = the BF16 src[i] as FP32 (exact).
void gdn_state_widen(const uint16_t* src, float* dst, int64_t n, void* stream);
/// dst[i] = src[i] rounded to BF16, nearest even (a NaN stays a NaN).
void gdn_state_narrow(const float* src, uint16_t* dst, int64_t n, void* stream);

}  // namespace strata::kernels
