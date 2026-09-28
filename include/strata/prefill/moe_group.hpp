// include/strata/prefill/moe_group.hpp - issue #36: the prompt path's routed (token, k) pairs grouped by expert
// on the GPU.
//
// Until 0.1.20 every MoE layer of a chunk copied the T x 10 routed ids to the host, synchronized the stream and
// sorted them there, then uploaded the two row tables (48 syncs and ~1 MB of copies per chunk).  This is the same
// counting sort on the device - counts per chunk of pairs, a scan, then each chunk's pairs placed in order by one
// warp - so the tables are the host sort's, entry for entry (stable: an expert's rows follow the pair index), and
// only the 512 counts come back, in one small async copy the host waits for while the shared expert runs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill {

/// Device scratch bytes `moe_group` needs for `n` pairs over `n_expert` experts.
size_t moe_group_scratch_bytes(int64_t n, int64_t n_expert);

/// The `n` routed ids (`k` per token) grouped by expert, exactly as the host sort grouped them: `slot[i]` = the row
/// of pair i in expert order, `src[row]` = its token (i / k).  The per-expert counts, then a flag (1 = an id
/// outside 0..n_expert-1: the tables are then incomplete), land in `host_out` [n_expert + 1] (pinned memory: the
/// copy is asynchronous) once the stream reaches them.  n_expert <= 1024; no allocation, no synchronization.
void moe_group(const int32_t* ids, int64_t n, int64_t k, int64_t n_expert, int32_t* slot, int32_t* src,
               void* scratch, int32_t* host_out, void* stream);

}  // namespace strata::prefill
