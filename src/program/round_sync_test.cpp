// src/program/round_sync_test.cpp - the adaptive tier's persistent worker (round_sync.hpp), no GPU needed:
//   1. every job runs once, on one thread, and what it wrote is visible after wait();
//   2. wait() with nothing started returns at once; start() after start() runs both, in order;
//   3. the destructor waits for a running job;
//   4. STRATA_OLD_ADAPT_THREAD=1: a thread per job, the same results.
#include "round_sync.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void suite(const char* mode) {
    std::printf("mode: %s\n", mode);
    {
        strata::program::AdaptWorker w;
        std::vector<int> seen;
        std::set<std::thread::id> threads;
        int ok_rounds = 0;
        for (int r = 0; r < 1000; ++r) {
            bool flag = false;
            if (r % 4 == 3) w.start([&, r] { seen.push_back(r); threads.insert(std::this_thread::get_id()); flag = true; });
            w.wait();
            ok_rounds += (r % 4 == 3) == flag;
        }
        check(ok_rounds == 1000 && seen.size() == 250, "250 jobs of 1000 rounds, each seen by the caller after wait()");
        bool ordered = true;
        for (size_t i = 0; i < seen.size(); ++i) ordered &= seen[i] == (int) (4 * i + 3);
        check(ordered, "in order");
        const bool old = strata::program::env_on("STRATA_OLD_ADAPT_THREAD");
        check(old ? threads.size() >= 1 : threads.size() == 1, old ? "a thread per job" : "one thread for every job");
        check(threads.count(std::this_thread::get_id()) == 0, "never the caller's thread");
    }
    {
        strata::program::AdaptWorker w;
        const auto t0 = std::chrono::steady_clock::now();
        w.wait();
        check(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(100), "wait() with no job returns");
        int a = 0, b = 0;
        w.start([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); a = 1; });
        w.start([&] { b = a + 1; });   // waits for the first
        w.wait();
        check(a == 1 && b == 2, "start() after start(): both, in order");
    }
    {
        std::atomic<int> done{0};
        {
            strata::program::AdaptWorker w;
            w.start([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); done = 1; });
        }
        check(done.load() == 1, "the destructor waits for a running job");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--old") {
        suite("STRATA_OLD_ADAPT_THREAD=1 (a thread per job)");
    } else {
        suite("persistent worker");
#if defined(_WIN32)
        _putenv_s("STRATA_OLD_ADAPT_THREAD", "1");
#else
        setenv("STRATA_OLD_ADAPT_THREAD", "1", 1);
#endif
        suite("STRATA_OLD_ADAPT_THREAD=1 (a thread per job)");
    }
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
