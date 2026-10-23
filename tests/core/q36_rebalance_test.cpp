// tests/core/q36_rebalance_test.cpp - strata-q36's expert profile and global rebalance (src/qwen35/q36_rebalance.hpp), no GPU:
// the choice of the slot replacements on a known set of counts (global best gain, the count and gain thresholds, the
// budget), the slots' start from a profile, and the learned profile's round trip through the STRP file.
#include "check.hpp"
#include "q36_rebalance.hpp"

#include "strata/core/expert_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
using strata::q36::LayerCounts;
using strata::q36::SlotSwap;
using Pair = std::pair<int32_t, int32_t>;

bool is(const SlotSwap& s, int layer, int in, int out) { return s.layer == layer && s.in == in && s.out == out; }

// experts `a` and `b` in slots 0 and 1, the rest outside
std::vector<int> resident(int ne, int a, int b) {
    std::vector<int> v((size_t) ne, -1);
    v[(size_t) a] = 0;
    v[(size_t) b] = 1;
    return v;
}
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("choose_swaps:\n");
    // layer 0: residents 0 (count 1) and 1 (0.5); outside, 2 counts 9 and 3 counts 4  -> (2 for 1) gain 8.5, (3 for 0) gain 3
    // layer 1: residents 2 (count 1) and 3 (3);   outside, 0 counts 6 and 1 counts 2.5 -> (0 for 2) gain 5; (1 for 3) is 2.5 < 3 + 1.5
    const std::vector<double> s0 = {1, 0.5, 9, 4, 1, 0}, s1 = {6, 2.5, 1, 3, 0, 0};
    const std::vector<int> r0 = resident(6, 0, 1), r1 = resident(6, 2, 3);
    const std::vector<LayerCounts> two = {{&s0, &r0}, {&s1, &r1}};
    {
        const auto w = choose_swaps(two, 8);
        check(w.size() == 3, "three replacements pass the thresholds");
        check(w.size() == 3 && is(w[0], 0, 2, 1) && w[0].gain == 8.5, "best gain first: layer 0, expert 2 for 1");
        check(w.size() == 3 && is(w[1], 1, 0, 2) && w[1].gain == 5.0, "then layer 1, expert 0 for 2 (across layers)");
        check(w.size() == 3 && is(w[2], 0, 3, 0) && w[2].gain == 3.0, "then layer 0, expert 3 for 0");
    }
    {
        const auto w = choose_swaps(two, 2);
        check(w.size() == 2 && is(w[0], 0, 2, 1) && is(w[1], 1, 0, 2),
              "a budget of 2 keeps the two best gains, even from the last layer");
        check(choose_swaps(two, 0).empty() && choose_swaps(two, -3).empty(), "a budget of 0 (or less) swaps nothing");
    }
    {
        // the last layer alone, with a layer without slots before it: it is still reached
        const std::vector<LayerCounts> late = {{}, {&s1, &r1}};
        const auto w = choose_swaps(late, 1);
        check(w.size() == 1 && is(w[0], 1, 0, 2), "a layer without a slot table is skipped, the next one served");
    }
    {
        const std::vector<double> low = {0, 0, 1.9, 0, 0, 0}, mid = {2.0, 2.0, 3.0, 0, 0, 0}, edge = {2.0, 2.0, 3.5, 0, 0, 0};
        const std::vector<int> r = resident(6, 0, 1);
        check(choose_swaps({{&low, &r}}, 8).empty(), "a count under 2 is not a candidate");
        check(choose_swaps({{&mid, &r}}, 8).empty(), "a gain of 1 (3.0 over 2.0) is under 1.5");
        const auto w = choose_swaps({{&edge, &r}}, 8);
        check(w.size() == 1 && is(w[0], 0, 2, 0) && w[0].gain == 1.5, "a gain of exactly 1.5 passes");
    }
    {
        const std::vector<double> tie = {0, 0, 3, 3, 0, 0};
        const std::vector<int> r = resident(6, 0, 1);
        const auto w = choose_swaps({{&tie, &r}}, 8);
        check(w.size() == 2 && is(w[0], 0, 2, 0) && is(w[1], 0, 3, 1), "equal counts: the lower expert index first, victims likewise");
    }
    check(kRebalanceDecay == 0.7 && kRebalanceMinCount == 2.0 && kRebalanceMinGain == 1.5, "the thresholds are adapt()'s");

    std::printf("profile_slot_experts:\n");
    const std::vector<Pair> ranked = {{1, 5}, {0, 3}, {0, 1}, {1, 2}, {0, 3}, {0, 9}, {1, 5}};
    {
        const auto a = profile_slot_experts(ranked, 0, 3, 6);
        check(a == std::vector<int>({3, 1, 0}), "layer 0: the profile's order (3, 1), then the lowest unused (0); duplicate and out-of-range pairs ignored");
        const auto b = profile_slot_experts(ranked, 1, 2, 6);
        check(b == std::vector<int>({5, 2}), "layer 1: its own pairs only, in rank order");
        const auto c = profile_slot_experts(ranked, 2, 3, 6);
        check(c == std::vector<int>({0, 1, 2}), "a layer the profile does not rank keeps experts 0 .. S - 1");
        const auto d = profile_slot_experts({}, 0, 4, 6);
        check(d == std::vector<int>({0, 1, 2, 3}), "no profile: experts 0 .. S - 1");
        check(profile_slot_experts(ranked, 0, 1, 6) == std::vector<int>({3}), "fewer slots than ranked pairs: the best one");
    }

    std::printf("learned profile round trip:\n");
    {
        const int64_t L = 3, E = 4;
        std::vector<uint8_t> res((size_t) (L * E), 0);
        std::vector<double> heat((size_t) (L * E), 0.0);
        const int S = 2;
        // slots of layer l: experts (l + 2) % 4 and (l + 3) % 4; heat favours expert 1 of every layer among the rest
        for (int64_t l = 0; l < L; ++l) {
            res[(size_t) (l * E + (l + 2) % 4)] = 1;
            res[(size_t) (l * E + (l + 3) % 4)] = 1;
            heat[(size_t) (l * E + 1)] = 5.0 + (double) l;
        }
        const std::vector<Pair> prior = {{0, 0}, {0, 1}};
        const auto learned = strata::core::rank_learned_profile(L, E, res, heat, prior);
        const std::string path = (std::filesystem::temp_directory_path() / "q36_rebalance_test.strp").string();
        std::string err;
        check(strata::core::write_expert_profile(path, L, E, learned, err), "the learned ranking is written");
        std::vector<Pair> back;
        int64_t slots = 0;
        check(strata::core::read_expert_profile(path, L, E, back, slots, err) && back == learned,
              "and read back identical");
        bool same = true;
        for (int l = 0; l < (int) L; ++l) {
            std::vector<int> got = profile_slot_experts(back, l, S, (int) E), want = {(l + 2) % 4, (l + 3) % 4};
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            same = same && got == want;
        }
        check(same, "a start from it puts every layer's slots where the run left them");
        check(profile_slot_experts(back, 0, 3, (int) E).size() == 3 && profile_slot_experts(back, 0, 3, (int) E)[2] == 1,
              "one slot more takes the hottest expert outside");
        std::vector<Pair> other;
        check(!strata::core::read_expert_profile(path, L + 1, E, other, slots, err) &&
                  err.find("different artifact") != std::string::npos,
              "another layer count is refused (the engine then falls back to the default order)");
        check(!strata::core::read_expert_profile(path + ".absent", L, E, other, slots, err), "a missing file is refused");
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
