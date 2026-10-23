// src/qwen35/q36_delta.hpp - the running average of a layer's router logit range (strata-q36's --cache-prior), shared by the
// routing kernel (device) and its host test.
#pragma once

#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define Q36_HD __host__ __device__
#else
#define Q36_HD
#endif

namespace strata::q36 {

/// Delta_avg of arXiv 2412.00099 (Eq. 10): E[max(z) - min(z)] over the tokens of a layer, "a running average over sequences
/// and tokens" (the paper gives no decay): the cumulative mean of the ranges seen so far, one (mean, count) pair per layer.
/// A range that is not finite (NaN or infinite logits) is skipped, the mean stays what it was.
Q36_HD inline void delta_avg_step(float& mean, int32_t& count, float range) {
    if (!(range >= 0.0f && range <= FLT_MAX) || count == INT32_MAX) return;
    ++count;
    mean += (range - mean) / (float) count;
}

}  // namespace strata::q36
