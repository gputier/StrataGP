// tests/core/q36_step_log_test.cpp - strata-q36's STRATA_Q36_STATS decode steps (src/qwen35/q36_stats.hpp), no GPU: the
// step log's partition of each step (pool, doorbell wait, host, PCIe copies), the copies tagged by step id, the bound on
// the steps listed, and the line's content.
#include "check.hpp"
#include "q36_stats.hpp"

#include <cmath>
#include <string>

namespace {
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("step log:\n");
    {
        StepLog log;
        const long long prompt_id = log.id();   // the prompt's copies carry the id before the decode starts
        log.begin(100.0, 40.0);                 // the counters at the decode's start: the pool's 100 ms, the wait's 40
        check(log.id() != prompt_id && log.steps().empty() && log.total() == 0, "begin empties the log and moves the id on");
        const long long first = log.id();
        log.open(1);
        log.add_pcie(first, 1.5);
        log.add_pcie(prompt_id, 9.0);   // a prompt's copy resolved in the first step: not the step's own
        log.close(112.0, 46.0, 25.0);   // 12 ms of pool, 6 of wait in 25 ms: 7 ms of host
        check(log.steps().size() == 1 && log.total() == 1, "a closed step is listed and counted");
        const StepStats& a = log.steps()[0];
        check(near(a.pool_ms, 12.0) && near(a.wait_ms, 6.0) && near(a.host_ms, 7.0), "pool and wait are the counters' moves, host the rest");
        check(near(a.pcie_ms, 1.5) && a.rows == 1, "the step's own copies only, one row");
        log.open(3);   // a verify window of 3 rows
        log.add_pcie(first, 0.5);   // the first step's last copies resolve one step later: they stay the first step's
        log.add_pcie(log.id(), 2.0);
        log.close(130.0, 50.0, 30.0);
        const StepStats& b = log.steps()[1];
        check(near(log.steps()[0].pcie_ms, 2.0), "a copy resolved late goes to the step that issued it");
        check(near(b.pool_ms, 18.0) && near(b.wait_ms, 4.0) && near(b.host_ms, 8.0) && near(b.pcie_ms, 2.0) && b.rows == 3,
              "the second step starts at the first one's close");
        log.add_pcie(log.id(), 4.0);   // the next step's copies, its entry not open yet: dropped
        check(near(b.pcie_ms, 2.0) && log.steps().size() == 2, "a copy of a step not listed is ignored");
        log.stop();
        log.open(1);
        log.close(200.0, 60.0, 40.0);
        check(log.steps().size() == 2 && log.total() == 2, "after stop no step opens or closes");
        log.begin(200.0, 60.0);
        log.add_pcie(first, 5.0);
        check(log.steps().empty() && log.total() == 0, "the next request starts empty; the last one's copies are not its own");
        log.open(1);
        log.close(203.0, 61.0, 2.0);   // host below zero when the counters moved more than the wall time: kept as measured
        check(near(log.steps()[0].host_ms, -2.0), "a negative host time is kept, not clamped");
    }
    {
        StepLog log;
        log.begin(0.0, 0.0);
        for (size_t i = 0; i < kStepLogMax + 5; ++i) {
            log.open(1);
            log.add_pcie(log.id(), 1.0);
            log.close((double) i + 1.0, 0.0, 2.0);
        }
        check(log.steps().size() == kStepLogMax && log.total() == (long long) kStepLogMax + 5,
              "the steps past kStepLogMax are counted, not listed");
        check(near(log.steps().back().pool_ms, 1.0) && near(log.steps().back().pcie_ms, 1.0), "the listed steps stay whole");
    }
    std::printf("line:\n");
    {
        StepLog log;
        log.begin(0.0, 0.0);
        log.open(1);
        log.add_pcie(log.id(), 2.6);
        log.close(15.06, 7.63, 24.12);
        log.open(2);
        log.close(30.0, 10.0, 20.0);
        const std::string s = format_steps(3, log);
        check(has(s, "strata serve: session 3: stats: decode steps (pool/wait/host/PCIe copies ms[/rows]), 2 steps, 2 listed:"),
              "the head names the session, the fields and the counts");
        check(has(s, ": 15.1/7.6/1.4/2.6 14.9/2.4/2.7/0.0/2") && s.back() == '2', "one field group per step, rows past one");
        StepLog none;
        none.begin(0.0, 0.0);
        check(format_steps(0, none).back() == ':', "no step: the head alone");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
