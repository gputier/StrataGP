// tests/core/pool_lifecycle_test.cpp - the CPU expert pool's start, stop and diagnostic paths (issue #85).
//
//   1. run_split_multi_native refuses a token count outside 1..MAXT before it writes anything.  The per-token
//      scratch arrays are MAXT long, so nt = MAXT + 1 used to write past them.
//   2. A thread creation failure while the pool starts is reported as an error: the workers already started are
//      joined, the process does not terminate, and the diagnostic hook does not keep a pointer to the dead pool.
//   3. A diagnostic reading the pool while it is destroyed never touches the pool after its destructor has freed it.
//
// No job is published (run(jobs, 0) returns at once), so no expert kernel runs and the test needs no AVX.
// The failure injection for 2 is compiled only when NDEBUG is not defined (see g_pool_fail_thread_at in pool.cpp).
#include "check.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef NDEBUG
namespace strata::kernels::cpu {
extern std::atomic<int> g_pool_fail_thread_at;
}
#endif

namespace cpu = strata::kernels::cpu;

namespace {

void test_native_token_bound() {
    std::printf("native split, token count:\n");
    cpu::ExpertPool pool(2, false, false);
    cpu::NativeFmt f;
    f.gu_type = 0;   // not Q2_0: the ggml-cpu rows, which this test never reaches
    f.d_type = 0;
    f.n_embd = 64;
    f.n_ff = 64;
    std::vector<uint8_t> blob(64, 0);
    cpu::ExpertJobMulti job;
    job.blob = blob.data();

    auto refused = [&](int nt) {
        job.nt = nt;
        try {
            pool.run_split_multi_native(f, &job, 1);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(refused(cpu::MAXT + 1), "nt = MAXT + 1 is refused with an error");
    check(refused(0), "nt = 0 is refused with an error");
}

void test_thread_creation_failure() {
    std::printf("thread creation failure:\n");
#ifndef NDEBUG
    cpu::g_pool_fail_thread_at.store(2);   // the third worker's creation fails
    bool reported = false;
    try {
        cpu::ExpertPool pool(4, false, false);
    } catch (const std::runtime_error&) {
        reported = true;
    }
    cpu::g_pool_fail_thread_at.store(-1);
    check(reported, "the failed start is reported as an error, and the process survives");

    // The hook must not point at the pool that failed to start.  Calling it must be safe either way.
    if (strata::core::DiagFn fn = strata::core::diag_pool_fn().load()) {
        std::FILE* sink = std::tmpfile();
        fn(sink);
        std::fclose(sink);
    }
    check(true, "the diagnostic hook is callable after the failed start");

    {
        cpu::ExpertPool pool(2, false, false);
        check(pool.workers() == 2, "a pool started after the failure works");
    }
#else
    std::printf("  skipped: the failure injection is compiled out of NDEBUG builds\n");
#endif
}

void test_diag_during_destruction() {
    std::printf("diagnostic racing a destruction:\n");
    std::FILE* sink = std::tmpfile();
    for (int round = 0; round < 300; ++round) {
        std::atomic<bool> stop{false};
        std::thread reader([&] {
            while (!stop.load())
                if (strata::core::DiagFn fn = strata::core::diag_pool_fn().load()) fn(sink);
        });
        {
            cpu::ExpertPool pool(2, false, false);
        }
        stop.store(true);
        reader.join();
    }
    std::fclose(sink);
    check(true, "300 destructions with a diagnostic reader running");
}

}  // namespace

// Each case can run alone (`nt`, `thread`, `diag`), so a case that aborts does not hide the others.
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: a test that aborts still shows its checks
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() || only == "nt") test_native_token_bound();
    if (only.empty() || only == "thread") test_thread_creation_failure();
    if (only.empty() || only == "diag") test_diag_during_destruction();
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
