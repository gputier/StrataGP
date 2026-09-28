#pragma once

namespace strata::kernels {
void native_rope_set_enabled(bool enabled);
bool native_rope_enabled();

// Pinned CUDA text-only IMRoPE: F32 rows, 64 rotated channels, no YaRN/frequency
// factors, equal text positions in all four IMRoPE sections. Each device position
// must be nonnegative. The position buffer remains live through graph replay.
// Supports head_dim 128/256 and exact x==out; partial overlap is rejected.
// Explicit stream required. No allocation or synchronization.
void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, float freq_base, const int* positions, void* stream);

// The per-pair frequency ratio the rotation raises to the pair index, computed on the
// host once per call: shared with the fused QSA preparation (qsa_prep.hpp) so both
// start from the same float.
float native_rope_theta_scale(float freq_base, int n_rot);
// Issue #3 A/B, OFF by default: the angle pos * base^(-2i/n_rot) and its sine/cosine in float64 (the table
// path's arithmetic, exact to the f32 rounding of cos/sin) instead of the pinned f32 fast-math angle, whose
// phase error grows with the position (rope_parity prints it). It no longer reproduces llama.cpp. Unset, the
// environment decides once: STRATA_ROPE_F64=1. Configure before capture (the choice is a kernel argument).
// The prompt path's own RoPE (prefill) is not affected.
void native_rope_set_f64_angle(bool enabled);
bool native_rope_f64_angle();
}
