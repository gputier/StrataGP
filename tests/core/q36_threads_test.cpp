// tests/core/q36_threads_test.cpp - strata-q36's default pool worker count (src/qwen35/q36_threads.hpp), no GPU.
#include "check.hpp"
#include "q36_threads.hpp"

int main() {
    using namespace strata::q36;
    std::printf("pool workers:\n");
    check(kDefaultPoolWorkers == 6, "the default is the measured 6");
    check(pool_workers(-1, 15) == 6, "not given, 15 physical cores to use: 6");
    check(pool_workers(-1, 6) == 6, "not given, exactly 6 cores: 6");
    check(pool_workers(-1, 3) == 3, "not given, 3 cores: no more workers than cores");
    check(pool_workers(-1, 0) == 1, "not given, no topology: 1");
    check(pool_workers(0, 15) == 0, "--cpu-threads 0 is the pool's own default (one per core)");
    check(pool_workers(8, 15) == 8, "--cpu-threads 8 is taken as given");
    check(pool_workers(16, 15) == 16, "an explicit count above the cores is taken as given");
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
