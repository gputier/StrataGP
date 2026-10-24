// tests/core/q36_pcie_step_test.cpp - STRATA_Q36_PCIE_AUTO_STEP, the step rule of strata-q36's automatic PCIe share
// (src/qwen35/q36_pcie.hpp), no GPU: the switch's value, one step's direction in the 256ths the engine stores, its bounds,
// its margin, the stop after two reversals, and the search on synthetic times per token whose minimum lies at a share of 0.55.
#include "check.hpp"
#include "q36_pcie.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using strata::q36::pcie_auto_step;
using strata::q36::PcieStep;
using strata::q36::PcieStepSearch;

bool same(const PcieStep& s, int num, int dir) { return s.num == num && s.dir == dir; }

// the share the engine stores for a share s with a rounding down or to nearest (strata::core::pcie_num_of is one of the two)
int floor_num(double s) { return (int) std::floor(s * 256.0); }
int round_num(double s) { return (int) std::lround(s * 256.0); }

struct Search {
    int last = 0;            // the share after the last window, 256ths
    int moves = 0;           // windows whose share changed
    int done_at = -1;        // the window after which the search had stopped (-1: never)
    bool held = true;        // the share never changed once the search had stopped
};

// the search as the engine runs it: each window measures the time per token at the share now, in 256ths
template <class F>
Search search(int start, F time_of, int windows) {
    Search r;
    PcieStepSearch st;
    int num = start;
    for (int k = 0; k < windows; ++k) {
        const bool was_done = st.done;
        const int next = st.next(num, time_of(num / 256.0));
        if (next != num) ++r.moves;
        if (was_done && next != num) r.held = false;
        if (st.done && r.done_at < 0) r.done_at = k;
        num = next;
    }
    r.last = num;
    return r;
}
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("pcie_step_enabled:\n");
    check(pcie_step_enabled("1"), "\"1\" is on");
    check(!pcie_step_enabled(nullptr) && !pcie_step_enabled("0") && !pcie_step_enabled("") && !pcie_step_enabled("10") &&
              !pcie_step_enabled("true"),
          "no variable, \"0\", empty, \"10\" and \"true\" are off");

    std::printf("pcie_auto_step (256ths; 154 is 0.60):\n");
    check(kPcieStepNumMin == 77 && kPcieStepNumMax == 230 && kPcieStepNum == 13, "the bounds and the step in 256ths: 77, 230, 13");
    check(same(pcie_auto_step(154, -1.0, 5.0, -1), 141, -1), "the first step (no window before) lowers the share");
    check(same(pcie_auto_step(154, 2.0, 1.0, -1), 141, -1), "the time fell after a step down: the share goes on down");
    check(same(pcie_auto_step(154, 1.0, 2.0, -1), 167, 1), "the time rose after a step down: the share goes up");
    check(same(pcie_auto_step(154, 2.0, 1.0, 1), 167, 1), "the time fell after a step up: the share goes on up");
    check(same(pcie_auto_step(154, 1.0, 2.0, 1), 141, -1), "the time rose after a step up: the share goes down");

    std::printf("the margin (3 %% of the window before):\n");
    check(same(pcie_auto_step(154, 1.0, 1.0, 1), 154, 1) && same(pcie_auto_step(154, 1.0, 1.0, -1), 154, -1),
          "an unchanged time does not move the share");
    check(same(pcie_auto_step(154, 1.0, 1.02, -1), 154, -1), "a time 2 % higher is a tie: no move, no turn");
    check(same(pcie_auto_step(154, 1.0, 0.98, -1), 154, -1), "a time 2 % lower is a tie: no move");
    check(same(pcie_auto_step(154, 1.0, 1.04, -1), 167, 1), "a time 4 % higher turns the direction");
    check(same(pcie_auto_step(154, 1.0, 0.96, -1), 141, -1), "a time 4 % lower keeps the direction");
    check(same(pcie_auto_step(154, 1.0, std::nan(""), -1), 154, -1), "a time that is not a number does not move the share");

    std::printf("the bounds, quantised to 1/256 as the engine stores the share:\n");
    for (const auto& q : {floor_num, round_num}) {
        const char* how = q == floor_num ? "rounded down" : "rounded to nearest";
        char what[160];
        const int top = q(kPcieStepMax), bottom = q(kPcieStepMin);
        std::snprintf(what, sizeof what, "0.90 stored %s (%d): a step up at the top turns down (217 or below)", how, top);
        const PcieStep up = pcie_auto_step(top, 2.0, 1.0, 1);
        check(up.dir == -1 && up.num == std::min(top, kPcieStepNumMax) - kPcieStepNum, what);
        std::snprintf(what, sizeof what, "0.30 stored %s (%d): two steps down at the bottom end inside the range, turned up", how, bottom);
        PcieStep down = pcie_auto_step(bottom, 2.0, 1.0, -1);
        if (down.dir == -1) down = pcie_auto_step(down.num, 2.0, 1.0, -1);   // 76 is first brought to 77, then turns
        check(down.dir == 1 && down.num == kPcieStepNumMin + kPcieStepNum, what);
    }
    check(same(pcie_auto_step(230, 2.0, 1.0, 1), 217, -1), "at the top bound (230) the step turns down");
    check(same(pcie_auto_step(77, 2.0, 1.0, -1), 90, 1), "at the bottom bound (77) the step turns up");
    check(same(pcie_auto_step(220, 2.0, 1.0, 1), 230, 1), "a step that passes the top stops on it (230)");
    check(same(pcie_auto_step(243, -1.0, 1.0, -1), 230, -1), "a share above the range (0.95) is brought to 230");
    check(same(pcie_auto_step(26, -1.0, 1.0, 1), 77, 1), "a share below the range (0.10) is brought to 77");
    {
        bool in = true;
        for (int c = 0; c <= 256; ++c)
            for (const int d : {-1, 1})
                for (const double w : {-1.0, 0.5, 1.0, 2.0}) {
                    const PcieStep s = pcie_auto_step(c, w, 1.0, d);
                    const bool moved_in = s.num >= kPcieStepNumMin && s.num <= kPcieStepNumMax;
                    in = in && (moved_in || (s.num == c && w == 1.0)) && (s.dir == 1 || s.dir == -1);
                }
        check(in, "every move lands in [77, 230] with a direction of -1 or +1 (a tie leaves the share as it was)");
    }

    std::printf("PcieStepSearch: the stop after two reversals:\n");
    {
        // 154 (first step) -> 141 faster -> 128 slower (turn 1) -> 141 faster -> 154 slower (turn 2): the best share, 141
        PcieStepSearch st;
        const int a = st.next(154, 10.0);
        const int b = st.next(a, 9.0);
        const int c = st.next(b, 10.0);
        const int d = st.next(c, 9.5);
        check(a == 141 && b == 128 && c == 141 && d == 154 && st.turns == 1 && !st.done, "the path down, the first turn, back up");
        const int e = st.next(d, 10.0);
        check(st.done && st.turns == 2 && e == 141, "the second turn stops the search on the best share seen (141, 9.0 ms)");
        check(st.next(e, 1.0) == 141 && st.next(141, 50.0) == 141, "once stopped, no window moves the share");
    }
    {
        // the stop returns the best share seen, not the step pcie_auto_step would take (154 here, the best is 141):
        //   next(154, 10.0): best 154 @ 10.0; first step down -> 141 (dir -1), t_prev 10.0
        //   next(141,  9.0): best 141 @ 9.0; 9.0 < 10.0 * 0.97 = 9.7, keep down -> 128, t_prev 9.0
        //   next(128, 10.0): 10.0 > 9.0 * 1.03 = 9.27, turn 1, up -> 141, t_prev 10.0
        //   next(141,  9.5): 9.5 < 10.0 * 0.97 = 9.7, keep up -> 154, t_prev 9.5
        //   next(154,  9.1): 9.1 < 9.5 * 0.97 = 9.215, keep up -> 167, t_prev 9.1 (9.1 > 9.0: best stays 141)
        //   next(167, 10.0): 10.0 > 9.1 * 1.03 = 9.373, turn 2: pcie_auto_step says 154 (dir -1), the search stops on 141
        PcieStepSearch st;
        const int a = st.next(154, 10.0);
        const int b = st.next(a, 9.0);
        const int c = st.next(b, 10.0);
        const int d = st.next(c, 9.5);
        const int e = st.next(d, 9.1);
        check(a == 141 && b == 128 && c == 141 && d == 154 && e == 167 && st.turns == 1 && !st.done && st.best_num == 141,
              "down to 128, one turn, up past the best share (141) to 167");
        check(same(pcie_auto_step(167, 9.1, 10.0, 1), 154, -1), "at the stopping window the step alone would go to 154");
        const int f = st.next(e, 10.0);
        check(st.done && st.turns == 2 && f == 141, "the second turn stops on the best share seen (141, 9.0 ms), not on 154");
        check(st.next(f, 1.0) == 141 && st.next(f, 50.0) == 141, "once stopped, the share stays 141");
    }
    {
        // a tie holds the share and keeps the window before: the next window at that share is still compared with it
        PcieStepSearch st;
        const int a = st.next(154, 10.0);
        const int b = st.next(a, 10.1);
        check(a == 141 && b == 141 && st.t_prev == 10.0 && st.turns == 0, "a tie (1 %) holds 141 and keeps 10.0 as the window before");
        check(st.next(b, 9.0) == 128, "a later window 10 % faster than that one moves on down");
    }
    {
        // a flat basin: every window within the margin of the first, so after the first step the share never moves
        const Search f = search(154, [](double s) { return 10.0 + 0.1 * s; }, 40);
        check(f.moves == 1 && f.last == 141 && f.done_at < 0, "a flat time per token: one first step, then no move in 40 windows");
    }

    std::printf("PcieStepSearch on synthetic basins (minimum at 0.55; times in ms per token):\n");
    {
        auto quad = [](double s) { return 1.0 + 40.0 * (s - 0.55) * (s - 0.55); };
        auto skew = [](double s) { return 1.0 + (s < 0.55 ? 3.0 * (0.55 - s) : (s - 0.55)); };
        for (const int start : {round_num(0.80), round_num(0.40)}) {
            char what[192];
            for (int which = 0; which < 2; ++which) {
                const Search r = which == 0 ? search(start, quad, 40) : search(start, skew, 40);
                const char* name = which == 0 ? "quadratic" : "asymmetric";
                std::snprintf(what, sizeof what, "%s basin from %.2f: the search stops (after window %d) within one step of 0.55 (%.3f)",
                              name, start / 256.0, r.done_at, r.last / 256.0);
                check(r.done_at >= 0 && std::fabs(r.last / 256.0 - 0.55) <= kPcieStepNum / 256.0 + 1e-9, what);
                std::snprintf(what, sizeof what, "%s basin from %.2f: the share stays put once stopped (%d moves in 40 windows)",
                              name, start / 256.0, r.moves);
                check(r.held && r.moves <= 12, what);
            }
        }
    }

    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
