// src/qwen35/q36_rules.hpp - the pure rules of strata-q36's routing and hybrid mode, shared by the routing kernel (device)
// and the host tests: the SER keep rule, the cache-prior bias and the graph guard of the hybrid layers.
#pragma once

#include "q36_delta.hpp"

namespace strata::q36 {

/// SER (ik_llama.cpp PR 239), k_route_group: the pick of rank `rank` is kept when it is one of the first ser_kmin picks or
/// its probability p exceeds ser_t times the top probability `top`: the row's largest unbiased probability, 1 / sum of the
/// exps (the max logit's).  Without cache-prior that is the rank-0 pick's probability; with it the rank-0 pick is the largest
/// biased logit, whose own probability may be lower than `top`.
Q36_HD inline bool ser_keep(int rank, int ser_kmin, float p, float ser_t, float top) {
    return rank < ser_kmin || p > ser_t * top;
}

/// cache-prior (arXiv 2412.00099, Eq. 9), k_route_group: the amount added to an expert's logit in the ranking.  It is
/// cache_prior * Delta_avg for an expert with a VRAM slot (`resident`) in a hybrid layer, and 0 otherwise or when
/// cache_prior is not positive.
Q36_HD inline float cache_prior_bias(bool hybrid, bool resident, float cache_prior, float delta_avg) {
    return (hybrid && resident && cache_prior > 0.0f) ? cache_prior * delta_avg : 0.0f;
}

/// place_experts' graph guard: the graphs (a decode step, a verify window) are off when the main layers are hybrid and a
/// batch past hybrid_t rows would stream the layer's experts, which a capture cannot hold: hybrid_t below verify_rows.
Q36_HD inline bool hybrid_graph_off(bool main_hybrid, int hybrid_t, int verify_rows) {
    return main_hybrid && hybrid_t < verify_rows;
}

}  // namespace strata::q36
