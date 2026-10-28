// tests/core/q36_threads_test.cpp - strata-q36's pool worker count (src/qwen35/q36_threads.hpp), no GPU: the automatic
// count (not given) is one per worker processor, capped at kAutoPoolWorkersCap; 0 and a positive value are unchanged.
#include "check.hpp"
#include "q36_threads.hpp"

int main() {
    using namespace strata::q36;
    std::printf("pool workers:\n");
    check(kAutoPoolWorkersCap == 12, "the automatic cap is the measured 12");
    check(pool_workers(-1, 30) == 12, "not given, 30 worker processors: capped at 12");
    check(pool_workers(-1, 15) == 12, "not given, 15 worker processors: capped at 12");
    check(pool_workers(-1, 12) == 12, "not given, exactly 12 worker processors: 12");
    check(pool_workers(-1, 11) == 11, "not given, 11 worker processors: one per processor, under the cap");
    check(pool_workers(-1, 6) == 6, "not given, 6 worker processors: 6");
    check(pool_workers(-1, 3) == 3, "not given, 3 worker processors: no more workers than processors");
    check(pool_workers(-1, 0) == 1, "not given, no topology: 1");
    check(pool_workers(-5, 15) == 12, "any negative value is the automatic count: capped at 12");
    check(pool_workers(0, 15) == 0, "--cpu-threads 0 is the pool's own default (one per processor)");
    check(pool_workers(0, 30) == 0, "--cpu-threads 0 has no cap: the pool takes one per processor");
    check(pool_workers(8, 15) == 8, "--cpu-threads 8 is taken as given");
    check(pool_workers(16, 15) == 16, "an explicit count above the processors is taken as given, above the cap too");
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
