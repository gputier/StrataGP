// tests/core/q36_pcie_test.cpp - the PCIe share of strata-q36's hybrid mode, no GPU: which missed experts go over PCIe
// (src/qwen35/q36_pcie.hpp), the plan the graph's PCIe call reads, and the share's start and refinement
// (include/strata/core/pcie_share.hpp, shared with Flash-Next).
#include "q36_pcie.hpp"

#include "strata/core/pcie_share.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

using strata::q36::kPcieMaxEntries;
using strata::q36::kPcieSlots;

// 16 experts, 0 .. 3 resident
std::vector<int> slots16() {
    std::vector<int> v(16, -1);
    for (int i = 0; i < 4; ++i) v[(size_t) i] = i;
    return v;
}

std::vector<int32_t> pick(const std::vector<int32_t>& ids, int num, int cap, const std::vector<int>& slot_of) {
    int32_t out[kPcieSlots] = {};
    const int m = strata::q36::pcie_pick(ids.data(), (int) ids.size(), slot_of, num, cap, out);
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
    check(pick(ids, 0, kPcieSlots, s16).empty(), "a share of 0 sends nothing");
    check(pick(ids, 128, kPcieSlots, s16) == std::vector<int32_t>({12, 14}),
          "half of 5 misses is 2: the LAST two in routing order");
    check(pick(ids, 256, kPcieSlots, s16) == std::vector<int32_t>({5, 7, 9, 12, 14}), "a share of 1 sends every miss");
    check(pick(ids, 256, 3, s16) == std::vector<int32_t>({9, 12, 14}), "the cap keeps the last ones");
    check(pick(ids, 85, kPcieSlots, s16) == std::vector<int32_t>({14}), "the count rounds down: 5 misses x 85 / 256 = 1.66 is 1");
    check(pick({5, 0, 7}, 85, kPcieSlots, s16).empty(), "two misses at a third: 0.66 is none");
    // two tokens routing the same experts: a miss counts once, at its first appearance
    const std::vector<int32_t> twice = {5, 7, 9, 12, 5, 9, 12, 14};
    check(pick(twice, 128, kPcieSlots, s16) == std::vector<int32_t>({12, 14}),
          "an expert routed twice is one miss, ordered by its first appearance (misses 5, 7, 9, 12, 14)");
    check(pick({0, 1, 2, 3}, 256, kPcieSlots, s16).empty(), "resident experts are never sent");
    {
        std::vector<int32_t> many((size_t) kPcieMaxEntries + 1, 5);
        check(pick(many, 256, kPcieSlots, s16).empty(), "more entries than the plan holds: no share");
    }
    check(pick({99, -1, 5}, 256, kPcieSlots, s16) == std::vector<int32_t>({5}), "ids outside the experts are ignored");

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
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
