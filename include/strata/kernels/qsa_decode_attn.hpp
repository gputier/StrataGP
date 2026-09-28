// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const int32_t* page_table = nullptr;
};

/// The chunk kernel (O6c, #23).  The default starts the chunk's V rows as `cp.async` copies into shared memory as
/// soon as the page table has resolved them, preloads each warp's K rows, and waits for V only after QK and the
/// softmax; its arithmetic is the previous kernel's, expression for expression, so the output is bitwise the same.
///   kQsaAttnOld      the previous kernel, one load in flight per warp (A/B: STRATA_OLD_QSA_ATTN=1)
///   kQsaAttnChunk32  the prefetching kernel over 32 cells per block: twice the blocks, but a different split of the
///                    softmax, so NOT bitwise equal - opt-in (STRATA_QSA_CHUNK=32)
/// A cell whose page did not resolve (page table < 0: KV streaming overflow, `kv_stream.hpp` ctl[3]) is masked with
/// weight 0 in every variant instead of being read from before the pool (B14, #12).
enum QsaAttnVariant : int { kQsaAttnPrefetch = 0, kQsaAttnOld = 1, kQsaAttnChunk32 = 2 };
///   STRATA_QSA_ATTN_BATCH_OLD=1  the previous kernel for n_q > 1 only (batched prefill, verify), bitwise neutral:
///                    the prefetching kernel's 48 KB of shared memory allow 2 blocks per SM, to A/B on prefill
/// Read from the environment on first use.  The scratch size depends on it, so qsa_decode_attn_set_variant must come
/// BEFORE any qsa_decode_attn_scratch_floats: a test that switches variants sizes its scratch for kQsaAttnChunk32 (the
/// largest); the engine never switches.  A variant set later whose chunks are smaller than the smallest ever sized
/// runs with 64-cell chunks instead of overrunning that scratch (one warning).
int qsa_decode_attn_variant();
void qsa_decode_attn_set_variant(int variant);

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

/// O6c (#23): the layer's output gate applied by the merge itself, one launch less per QSA layer.  `out` [n_q,
/// n_head, 256] receives exactly what the gate kernel would write from `attn` (which is still written):
///   kQsaGateF64        `qsa_gate_apply_f32` (qsa.hpp), the default engine's gate
///   kQsaGateNativeF32  `native_qsa_gate_apply` (native_qsa.hpp), with its fast-math instructions reproduced
/// q_full [n_q, n_head, 2 * 256] is the raw `attn_q` projection (the gate is each head's second half).  Not for Q4_0
/// KV, whose output is rotated back between the attention and the gate.  Opt-in: STRATA_QSA_MERGE_GATE=1
/// (`qsa_decode_attn_gate_fold`), until `qsa_decode_attn_parity` has confirmed on the device that it is bitwise.
enum QsaGateFold : int { kQsaGateNone = 0, kQsaGateF64 = 1, kQsaGateNativeF32 = 2 };
struct QsaAttnGate {
    int kind = kQsaGateNone;
    const float* q_full = nullptr;
    float* out = nullptr;
};
bool qsa_decode_attn_gate_fold();

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream,
                          const QsaAttnGate& gate = QsaAttnGate{});

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                           const QsaAttnGate& gate = QsaAttnGate{});

}  // namespace strata::kernels
