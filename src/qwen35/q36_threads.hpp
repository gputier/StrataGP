// src/qwen35/q36_threads.hpp - how many pool workers strata-q36 starts (host side, header only).
#pragma once

#include <algorithm>

namespace strata::q36 {

/// The default worker count of the CPU expert pool: measured best on the Ornith box (21.9 and 26.6 tokens/s on its two
/// requests, 17.4 and 21.8 with every physical core but the host's); more workers read the same RAM and slow the pool down.
inline constexpr int kDefaultPoolWorkers = 6;

/// `requested` is --cpu-threads: negative (not given) takes the default, capped at the logical processors of work the pool
/// may use (`worker_cores`: the topology's worker processors, which can include SMT siblings), 0 keeps one worker per such
/// processor (the pool's own default), positive is taken as given.
inline int pool_workers(int requested, int worker_cores) {
    if (requested < 0) return std::max(1, std::min(kDefaultPoolWorkers, worker_cores));
    return requested;
}

}  // namespace strata::q36
