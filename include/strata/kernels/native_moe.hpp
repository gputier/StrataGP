#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before constructing/capturing sessions. Existing captures keep their
// selected kernels; changing this flag does not rewrite an existing graph.
void native_moe_combine_set_enabled(bool enabled);
bool native_moe_combine_enabled();

// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);

// #44 (E1) and #19: n_tok tokens' combinations in ONE launch, each routed row read where it was left.
// Row i = t * k + e (n_embd floats) is the CPU's in `y_miss` - the mapped pinned staging, read over PCIe only
// for the CPU's rows - unless i is listed in hit_dst[0 .. *hit_count), in which case it is the GPU's in
// `hit_out` (hit_dst/hit_count null: every row is the CPU's).  Shared: `shared` + t * n_embd, scaled by
// shared_gate[t] when gate_mode is 1, by its sigmoid when 2 (the shared expert's gate folded in here), plain
// when 0.  Output token t at output + t * n_embd.  Bitwise equal, token for token, to copy_from_mapped of all
// rows + moe_hit_add + native_moe_combine (after the shared expert's own gate and scale for modes 1 and 2),
// given that the pool left the GPU's rows of y_miss zero, as it does.  k 2..15, n_tok * k <= 128, n_embd a
// multiple of 4, rows 16-byte aligned, output disjoint.  No allocation or synchronization.
void native_moe_combine_window(const float* y_miss, const float* hit_out, const int32_t* hit_dst,
                               const int32_t* hit_count, const float* weights, const float* shared,
                               const float* shared_gate, int gate_mode, float* output, int64_t n_embd, int64_t k,
                               int n_tok, void* stream);
}
