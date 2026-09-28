// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : one warp per (query, block), FP32; relu per indexer head, summed; block n_bid (the
//                      incomplete tail) scores the `dead` key and gets +1e9 when it has cells - which is exactly
//                      what the per-token path reads there, because pooled[n_bid] holds `dead` at that time (in
//                      a chunk it may already hold a block completed later, so it is not read).
//   qsa_block_topk   : one block per query; a 4-pass radix select over the query's n_bid + 1 blocks, each
//                      weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);

/// O6 (#21): the same scores from the fp16 SHADOW of the pooled keys (`QsaIndexerBuffers::pooled16`/`dead16`, written
/// by the pooling kernels next to the fp32 rows): half the bytes per block, and exactly the fp32 arithmetic on the
/// keys rounded to fp16.  That rounding CAN CHANGE THE SELECTION (`qsa.hpp`, INDEXER KEYS), so it is opt-in
/// (`--idx-fp16` or STRATA_IDX_FP16=1).
void qsa_block_scores_f16(const uint16_t* pooled16, const uint16_t* dead16, const float* q_idx, const int32_t* steps,
                          int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
///
/// O6b (#22): by default a kernel with coalesced histogram passes, warp-aggregated per-warp histograms and scans in
/// place of thread 0's serial walks - the same integer counts, so the same selection.  STRATA_OLD_TOPK=1 (or
/// `qsa_block_topk_set_old`, before capture) keeps the previous kernel for A/B.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream);
bool qsa_block_topk_old();
void qsa_block_topk_set_old(bool old);

}  // namespace strata::kernels
