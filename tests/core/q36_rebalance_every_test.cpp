// tests/core/q36_rebalance_every_test.cpp - STRATA_Q36_REBALANCE_EVERY, the rebalance cadence of strata-q36's hybrid mode
// (src/qwen35/q36_rebalance.hpp), no GPU: the value read from the environment and its bounds, and the draft depth a window
// may use so that no rebalance position falls inside it.
#include "check.hpp"
#include "q36_rebalance.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {
// the value read, and whether the message the engine prints for a refused value is the one it should be
bool refused(const char* v, int* got) {
    std::string err;
    *got = strata::q36::rebalance_every_from_env(v, &err);
    return *got == 8 && err.find("STRATA_Q36_REBALANCE_EVERY=") == 0 && err.find("using 8") != std::string::npos;
}
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("rebalance_every_from_env:\n");
    {
        std::string err;
        check(rebalance_every_from_env(nullptr, &err) == 8 && err.empty(), "unset: 8, no message");
        check(rebalance_every_from_env("1", &err) == 1 && err.empty(), "1 is accepted (a rebalance at every position)");
        check(rebalance_every_from_env(nullptr, nullptr) == kRebalanceEveryDefault && kRebalanceEveryDefault == 8, "the default is 8");
        check(rebalance_every_from_env("32", &err) == 32 && err.empty(), "32 is accepted");
        check(rebalance_every_from_env("1024", &err) == 1024 && err.empty(), "1024 is accepted");
        check(rebalance_every_from_env("512", &err) == 512 && err.empty(), "a value inside the range is taken as it is");
        check(rebalance_every_from_env("64", nullptr) == 64, "no message pointer: the value is still read");
        int got = 0;
        check(refused("0", &got) && got == 8, "0 is refused: 8 and the message");
        check(refused("1025", &got) && got == 8, "1025 is refused: 8 and the message");
        check(refused("-1", &got) && got == 8, "a negative value is refused");
        check(refused("", &got) && got == 8, "an empty value is refused");
        check(refused("abc", &got) && got == 8, "a value that is not a number is refused");
        check(refused("12x", &got) && got == 8, "trailing text after the number is refused");
        check(refused("3.5", &got) && got == 8, "a fraction is refused");
        check(refused("99999999999999999999", &got) && got == 8, "a value out of the long range is refused");
        check(refused("+8", &got) && got == 8, "a leading plus sign is refused (strtol alone would read 8)");
        check(refused(" 8", &got) && got == 8, "a leading space is refused (strtol alone would read 8)");
        check(refused("8 ", &got) && got == 8, "a trailing space is refused");
        check(rebalance_every_from_env("08", &err) == 8 && err.empty(), "a leading zero is read as decimal (08 is 8)");
        check(rebalance_every_from_env("0", nullptr) == 8 && rebalance_every_from_env("abc", nullptr) == 8 &&
                  rebalance_every_from_env("+8", nullptr) == 8,
              "a refused value with no message pointer: 8, and nothing is written through the null pointer");
    }
    std::printf("draft_depth_limit:\n");
    {
        // a window's rows are pos .. pos + D: no rebalance position (a multiple of `every`) may fall in (pos, pos + D], and the
        // bound is tight: D is the window's own depth or it stops right before a rebalance position
        bool bounded = true, tight = true, inside = true;
        for (const int every : {1, 2, 3, 8, 32, 1024})
            for (int64_t pos = 0; pos <= 3 * (int64_t) every + 1; ++pos)
                for (const int D0 : {0, 1, 2, 7, 2 * every + 3}) {
                    const int D = draft_depth_limit(pos, every, D0);
                    bounded = bounded && D >= 0 && D <= D0;
                    tight = tight && (D == D0 || (pos + D + 1) % every == 0);
                    for (int64_t p = pos + 1; p <= pos + D; ++p) inside = inside && p % every != 0;
                }
        check(bounded, "the depth stays within 0 .. the asked depth");
        check(inside, "no rebalance position falls inside a window (positions pos + 1 .. pos + D)");
        check(tight, "the bound is tight: the window ends right before the next rebalance position when it cuts");
        check(draft_depth_limit(0, 32, 7) == 7 && draft_depth_limit(30, 32, 7) == 1 && draft_depth_limit(31, 32, 7) == 0,
              "at 0 the asked depth stands; at 30 one draft; at 31 none (the window of the rebalance's own position)");
        check(draft_depth_limit(0, 8, 7) == 7 && draft_depth_limit(5, 8, 7) == 2 && draft_depth_limit(7, 8, 7) == 0 &&
                  draft_depth_limit(8, 8, 7) == 7,
              "the default every 8: 7 drafts fit from a multiple of 8, 2 at 5, none at 7");
        check(draft_depth_limit(5, 1, 7) == 0 && draft_depth_limit(0, 1024, 3) == 3, "every 1: no draft; 1024 leaves room");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
