// src/qwen35/q36_threads.hpp - how many pool workers strata-q36 starts (host side, header only).
#pragma once

#include <algorithm>

namespace strata::q36 {

/// The cap on the automatic worker count of the CPU expert pool: the best fixed count of round v13 of 25 October 2026 on the
/// Ornith box, 48.5 tokens/s over three rounds against 48.6 for `--cpu-threads 0` and 45.2 for the former default of 6. 14
/// and 16 measured in another round (v14) did no better. The 6 default was measured before that, at 21.9 and 26.6 tokens/s
/// (undated).
inline constexpr int kAutoPoolWorkersCap = 12;

/// `requested` is --cpu-threads: negative (not given) takes one worker per processor of work the pool may use
/// (`worker_cores`: the topology's worker processors, which can include SMT siblings), capped at kAutoPoolWorkersCap; 0
/// keeps one worker per such processor with no cap (the pool's own default); positive is taken as given.
inline int pool_workers(int requested, int worker_cores) {
    if (requested < 0) return std::max(1, std::min(kAutoPoolWorkersCap, worker_cores));
    return requested;
}

}  // namespace strata::q36
