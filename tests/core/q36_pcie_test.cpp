// tests/core/q36_pcie_test.cpp - the PCIe share of strata-q36's hybrid mode, no GPU: which missed experts go over PCIe
// (src/qwen35/q36_pcie.hpp), the plan the graph's PCIe call reads, and the share's start and refinement
// (include/strata/core/pcie_share.hpp, shared with Flash-Next).
#include "check.hpp"
#include "q36_pcie.hpp"

#include "strata/core/pcie_share.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using strata::q36::kPcieMaxEntries;
using strata::q36::kPcieSlots;

// 16 experts, 0 .. 3 resident
std::vector<int> slots16() {
    std::vector<int> v(16, -1);
    for (int i = 0; i < 4; ++i) v[(size_t) i] = i;
    return v;
}

std::vector<int32_t> pick(const std::vector<int32_t>& ids, int num, const std::vector<int>& slot_of, int* announced = nullptr) {
    int32_t out[kPcieSlots] = {};
    const int m = strata::q36::pcie_pick(ids.data(), (int) ids.size(), slot_of, num, out, announced);
    return std::vector<int32_t>(out, out + m);
}
}  // namespace

int main() {
    using namespace strata::q36;
    static_assert(kPlanPtr % 2 == 0, "the group addresses are 8-byte aligned in the plan");
    static_assert(kPlanWords == kPlanTok + kPcieMaxEntries, "the plan ends with the last entry's token");

    std::printf("pcie_pick:\n");
    const std::vector<int> s16 = slots16();
    // misses in routing order: 5, 7, 9, 12, 14 (0, 2 and 3 are resident)
    const std::vector<int32_t> ids = {5, 0, 7, 9, 2, 12, 14, 3};
    check(pick(ids, 0, s16).empty(), "a share of 0 sends nothing");
    check(pick(ids, 128, s16) == std::vector<int32_t>({12, 14}),
          "half of 5 misses is 2: the LAST two in routing order");
    check(pick(ids, 256, s16) == std::vector<int32_t>({5, 7, 9, 12, 14}), "a share of 1 sends every miss");
    {
        // 20 distinct misses at a share of 1: the staging region holds kPcieSlots, the LAST ones, and the share asked for 20
        std::vector<int32_t> twenty;
        for (int i = 0; i < 20; ++i) twenty.push_back(20 + i);
        int announced = 0;
        const std::vector<int32_t> got = pick(twenty, 256, std::vector<int>(64, -1), &announced);
        check(got.size() == (size_t) kPcieSlots && got.front() == 24 && got.back() == 39 && announced == 20,
              "the staging region's slots cap the count and keep the last ones; the share asked for 20");
    }
    {
        // Ornith at one token: top-8, every expert missed, a share of 0.95 (243 / 256): 7 of the 8 go over PCIe, well under
        // the staging region's kPcieSlots, and a 32-row window (256 entries) is the widest step that carries a share
        std::vector<int32_t> top8;
        for (int i = 0; i < 8; ++i) top8.push_back(100 + i);
        int announced = 0;
        const std::vector<int32_t> got = pick(top8, strata::core::pcie_num_of(0.95), std::vector<int>(256, -1), &announced);
        check(strata::core::pcie_num_of(0.95) == 243 && got.size() == 7 && announced == 7,
              "a share of 0.95 at one token of top-8 is not cut by the staging region");
        static_assert(8 < kPcieSlots, "the eight experts of a token fit the staging region");
        check(kPcieMaxEntries / 8 == 32, "kPcieMaxEntries holds 32 rows of top-8");
    }
    check(pick(ids, 85, s16) == std::vector<int32_t>({14}), "the count rounds down: 5 misses x 85 / 256 = 1.66 is 1");
    check(pick({5, 0, 7}, 85, s16).empty(), "two misses at a third: 0.66 is none");
    // two tokens routing the same experts: a miss counts once, at its first appearance
    const std::vector<int32_t> twice = {5, 7, 9, 12, 5, 9, 12, 14};
    check(pick(twice, 128, s16) == std::vector<int32_t>({12, 14}),
          "an expert routed twice is one miss, ordered by its first appearance (misses 5, 7, 9, 12, 14)");
    check(pick({0, 1, 2, 3}, 256, s16).empty(), "resident experts are never sent");
    {
        std::vector<int32_t> many((size_t) kPcieMaxEntries + 1, 5);
        check(pick(many, 256, s16).empty(), "more entries than the plan holds: no share");
    }
    check(pick({99, -1, 5}, 256, s16) == std::vector<int32_t>({5}), "ids outside the experts are ignored");

    std::printf("pcie_plan:\n");
    {
        // two tokens, top-4: token 0 routes 5 9 12 0, token 1 routes 12 14 5 1
        const std::vector<int32_t> t = {5, 9, 12, 0, 12, 14, 5, 1};
        const int32_t picked[2] = {12, 14};
        std::vector<int32_t> plan((size_t) kPlanWords, -7);
        strata::q36::pcie_plan(t.data(), (int) t.size(), 4, picked, 2, 0x7000000000ull, 5210112ull, plan.data());
        unsigned long long p0 = 0, p1 = 0;
        std::memcpy(&p0, plan.data() + kPlanPtr, 8);
        std::memcpy(&p1, plan.data() + kPlanPtr + 2, 8);
        check(plan[kPlanGroups] == 2, "two groups");
        check(p0 == 0x7000000000ull && p1 == 0x7000000000ull + 5210112ull, "group q's blob is the staging region's slot q");
        check(plan[kPlanStart] == 0 && plan[kPlanStart + 1] == 2 && plan[kPlanStart + 2] == 3,
              "expert 12 has entries 0 and 1, expert 14 entry 2 (starts 0, 2, 3)");
        check(plan[kPlanDst] == 2 && plan[kPlanDst + 1] == 4 && plan[kPlanDst + 2] == 5,
              "the entries' rows: token * top_k + k (2, 4 and 5)");
        check(plan[kPlanTok] == 0 && plan[kPlanTok + 1] == 1 && plan[kPlanTok + 2] == 1, "the entries' tokens (0, 1, 1)");
        strata::q36::pcie_plan(t.data(), (int) t.size(), 4, picked, 0, 0x7000000000ull, 5210112ull, plan.data());
        check(plan[kPlanGroups] == 0 && plan[kPlanStart] == 0, "no share: an empty plan the graph's call passes over");
    }

    std::printf("PcieAuto (Flash-Next's rule):\n");
    {
        using strata::core::PcieAuto;
        check(strata::core::pcie_frac_for_gbps(28.0, 0.55) == 0.55, "a link of 20 GB/s or more keeps the base share");
        check(std::abs(strata::core::pcie_frac_for_gbps(10.0, 0.55) - 0.275) < 1e-12, "half the link, half the share");
        check(strata::core::pcie_frac_for_gbps(0.0, 0.55) == 0.55, "no reading: the base share");
        // 26 GB/s link, a 4.08 MB expert, the pool at 400 us per expert = 10.2 GB/s: 26 / (26 + 1.85 x 10.2) = 0.58 -> 0.60
        PcieAuto a;
        a.on = true;
        a.link_gbps = 26.0;
        a.blob_bytes = 4.08e6;
        a.mark(0.0, 0);
        double tc = 0.0, rate = 0.0, ms = 0.0;
        int64_t jobs = 0;
        double got = -1.0;
        int at = 0;
        for (int w = 1; w <= 48 && got < 0.0; ++w) {
            ms += 40.0;   // 100 experts a window at 400 us
            jobs += 100;
            got = a.window(ms, jobs, 0.55, &tc, &rate);
            at = w;
        }
        check(at == 48 && std::abs(got - 0.60) < 1e-9, "the first update comes at the 48th window: 0.60");
        check(std::abs(tc - 400.0) < 1e-6 && std::abs(rate - 10.2) < 1e-6, "from 400 us per expert = 10.2 GB/s");
        // the pool at 100 us per expert = 40.8 GB/s: 26 / (26 + 75.5) = 0.256 -> 0.25, after 96 more windows
        got = -1.0;
        for (int w = 1; w <= 96 && got < 0.0; ++w) {
            ms += 10.0;
            jobs += 100;
            got = a.window(ms, jobs, 0.60, &tc, &rate);
            at = w;
        }
        check(at == 96 && std::abs(got - 0.25) < 1e-9, "the second update comes 96 windows later: 0.25");
        got = -1.0;
        for (int w = 1; w <= 96 && got < 0.0; ++w) {
            ms += 10.0;
            jobs += 100;
            got = a.window(ms, jobs, 0.25, &tc, &rate);
        }
        check(got < 0.0 && !a.on, "the reading is within one step of the share: no change, and the third update ends it");
        check(a.window(ms + 1.0, jobs + 1000, 0.25, &tc, &rate) < 0.0, "after the third update it stays put");
    }
    {
        // Ornith on a 54 GB/s link, a 4.08 MB expert, the pool at 625 us per expert = 6.5 GB/s: 54 / (54 + 12.1) = 0.82 -> 0.80,
        // which Flash-Next's 0.60 cap would cut; a pool at 2500 us balances at 0.95, the q36 cap
        using strata::core::PcieAuto;
        for (const double cap : {0.60, 0.95}) {
            for (const double us : {625.0, 2500.0}) {
                PcieAuto a;
                a.on = true;
                a.link_gbps = 54.0;
                a.blob_bytes = 4.08e6;
                a.max_share = cap;
                a.mark(0.0, 0);
                double tc = 0.0, rate = 0.0, ms = 0.0, got = -1.0;
                int64_t jobs = 0;
                for (int w = 1; w <= 48 && got < 0.0; ++w) {
                    ms += us / 10.0;   // 100 experts a window
                    jobs += 100;
                    got = a.window(ms, jobs, 0.55, &tc, &rate);
                }
                const double want = cap == 0.60 ? 0.60 : (us == 625.0 ? 0.80 : 0.95);
                char what[120];
                std::snprintf(what, sizeof what, "a pool at %.0f us per expert under a cap of %.2f refines to %.2f", us, cap, want);
                check(std::abs(got - want) < 1e-9, what);
            }
        }
    }
    {
        // a 0.25 GB/s link against a pool at 10.2 GB/s balances at 0.013: Flash-Next's floor keeps 0.10, min_share 0 follows it
        using strata::core::PcieAuto;
        for (const double floor : {0.10, 0.0}) {
            PcieAuto a;
            a.on = true;
            a.link_gbps = 0.25;
            a.blob_bytes = 4.08e6;
            if (floor == 0.0) a.min_share = 0.0;   // (the default is Flash-Next's 0.10)
            a.mark(0.0, 0);
            double tc = 0.0, rate = 0.0, ms = 0.0, got = -1.0;
            int64_t jobs = 0;
            for (int w = 1; w <= 48 && got < 0.0; ++w) {
                ms += 40.0;
                jobs += 100;
                got = a.window(ms, jobs, 0.55, &tc, &rate);
            }
            char what[120];
            std::snprintf(what, sizeof what, "a slow link's refined share is %.2f with a floor of %.2f", floor, floor);
            check(std::abs(got - floor) < 1e-9, what);
        }
    }
    std::printf("first_of_row:\n");
    {
        // two rows of 4: the second row's picks 2 and 3 repeat its first (a dropped pick), 7 also sits in the first row
        const int32_t row_ids[8] = {7, 3, 5, 9, 7, 2, 7, 7};
        const bool want[8] = {true, true, true, true, true, true, false, false};
        bool same = true;
        for (int e = 0; e < 8; ++e) same = same && first_of_row(row_ids, e, 4) == want[e];
        check(same, "an entry repeating an earlier entry of its own row is not the first; another row's does not count");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
