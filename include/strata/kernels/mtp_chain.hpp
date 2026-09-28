// include/strata/kernels/mtp_chain.hpp - the MTP draft chain as ONE graph launch (issue #16, O2).
//
// The draft chain continues while the last draft's probability is at least `min_p`.  Until now that test ran on
// the host: one graph and one stream synchronization per draft step.  Here the test runs on the device and drives
// a CUDA graph conditional WHILE node (CUDA 12.4+), so the whole chain is one launch and one synchronization:
//
//   init  : j = 1, *n_out = 1, cond = (1 < limit && probs[*row] >= min_p)
//   WHILE (cond) {
//       stage : the step and position records of row `row0 + j - 1` (mapped staging) -> the body's fixed row
//       ...     the draft layer for that row (the caller's kernels, unchanged)
//       next  : out[j] = ids[*row], out_p[j] = probs[*row]; j += 1; *n_out = j;
//               cond = (j < limit && probs[*row] >= min_p)
//   }
//
// which is the host loop `for (j = 1; j < limit && p >= min_p; ++j) { step j; p = probs[j]; ++n; }` step for step:
// the same float comparison, the same count, and the step kernels read the same values from other addresses.
// `params` is mapped host memory: [limit, min_p as float bits], written by the host before each launch.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace strata::kernels {

/// Before the WHILE node.  `ctl` is device memory (one int32: j), `n_out` mapped host memory (the drafts made).
void mtp_chain_init(int32_t* ctl, const float* probs, const int32_t* row_dev, const int32_t* params, int32_t* n_out,
                    unsigned long long cond_handle, void* stream);
/// The body's first kernel: step_dst[0..4) = step_src[(row0 + j - 1) * 4 ..], pos_dst[0..n_head) likewise.
void mtp_chain_stage(const int32_t* ctl, const int32_t* step_src, const int32_t* pos_src, int row0, int n_head,
                     int32_t* step_dst, int32_t* pos_dst, void* stream);
/// The body's last kernel (see above).  `out` / `out_p` may be mapped host memory.
void mtp_chain_next(int32_t* ctl, const int32_t* ids, const float* probs, const int32_t* row_dev, int32_t* out,
                    float* out_p, const int32_t* params, int32_t* n_out, unsigned long long cond_handle, void* stream);

/// Builds and instantiates [head] -> WHILE(cond) { body } on `stream`: `head` and `body` record their work into the
/// capturing stream and get the condition's handle.  On failure every capture is ended, nothing leaks, `err` says
/// why and false is returned (a driver without conditional nodes; the caller keeps its per-step path).
/// `*exec_out` is a cudaGraphExec_t, uploaded and ready to launch.
bool build_while_graph(void* stream, const std::function<bool(unsigned long long)>& head,
                       const std::function<bool(unsigned long long)>& body, void** exec_out, std::string& err);

}  // namespace strata::kernels
