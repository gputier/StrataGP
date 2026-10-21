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

std::vector<int32_t> pick(const std::vector<int32_t>& ids, int num, const std::vector<int>& slot_of, int* announced = nullptr,
                          int cap = kPcieSlots) {
    int32_t out[kPcieSlots] = {};
    const int m = strata::q36::pcie_pick(ids.data(), (int) ids.size(), slot_of, num, out, announced, cap);
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
        const unsigned long long ptr[2] = {0x7000000000ull, 0x7000000000ull + 5210112ull};
        strata::q36::pcie_plan(t.data(), (int) t.size(), 4, picked, 2, ptr, plan.data());
        unsigned long long p0 = 0, p1 = 0;
        std::memcpy(&p0, plan.data() + kPlanPtr, 8);
        std::memcpy(&p1, plan.data() + kPlanPtr + 2, 8);
        check(plan[kPlanGroups] == 2, "two groups");
        check(p0 == ptr[0] && p1 == ptr[1], "group q's blob is the address given for q");
        check(plan[kPlanStart] == 0 && plan[kPlanStart + 1] == 2 && plan[kPlanStart + 2] == 3,
              "expert 12 has entries 0 and 1, expert 14 entry 2 (starts 0, 2, 3)");
        check(plan[kPlanDst] == 2 && plan[kPlanDst + 1] == 4 && plan[kPlanDst + 2] == 5,
              "the entries' rows: token * top_k + k (2, 4 and 5)");
        check(plan[kPlanTok] == 0 && plan[kPlanTok + 1] == 1 && plan[kPlanTok + 2] == 1, "the entries' tokens (0, 1, 1)");
        strata::q36::pcie_plan(t.data(), (int) t.size(), 4, picked, 0, ptr, plan.data());
        check(plan[kPlanGroups] == 0 && plan[kPlanStart] == 0, "no share: an empty plan the graph's call passes over");
    }

    std::printf("prefetch:\n");
    {
        static_assert(kPcieStageBlobs == kPcieSlots + 2 * kPrefetchSlots, "the staging area holds the layer's slots and two regions");
        static_assert(kPrefetchSlots <= kPcieSlots, "a region is no wider than the layer's own slots");
        // layer L-1 predicts layer L's picks 7, 9, 12; layer L then picks 9, 12, 14
        const std::vector<int> s16 = slots16();
        const std::vector<int32_t> predicted = {5, 7, 9, 12, 5, 1};
        int32_t pf[kPrefetchSlots] = {};
        const int pf_n = strata::q36::pcie_pick(predicted.data(), (int) predicted.size(), s16, 192, pf, nullptr, kPrefetchSlots);
        check(pf_n == 3 && pf[0] == 7 && pf[1] == 9 && pf[2] == 12, "the prefetch takes the last 3 of the 4 predicted misses (192 / 256)");
        check(pick({5, 7, 9, 12, 5, 9, 12, 14}, 256, s16, nullptr, 2) == std::vector<int32_t>({12, 14}), "a cap of 2 keeps the last two");
        const int32_t real[3] = {9, 12, 14};
        int slot[3] = {};
        check(strata::q36::pcie_prefetch_match(real, 3, pf, pf_n, slot) == 2 && slot[0] == 1 && slot[1] == 2 && slot[2] == -1,
              "9 and 12 are in the region at slots 1 and 2, 14 is not: one copy on demand");
        check(strata::q36::pcie_prefetch_match(real, 3, pf, 0, slot) == 0 && slot[0] == -1 && slot[1] == -1 && slot[2] == -1,
              "an empty region matches nothing");
        const int32_t other[2] = {3, 8};
        check(strata::q36::pcie_prefetch_match(other, 2, pf, pf_n, slot) == 0, "a wrong prediction matches nothing and costs only its copy");
        // the plan points the prefetched groups at the region, the on-demand group at its own slot
        const std::vector<int32_t> t = {9, 12, 14, 0};
        const unsigned long long ptr[3] = {0x7000000000ull + 1 * 100, 0x7000000000ull + 2 * 100, 0x6000000000ull};
        std::vector<int32_t> plan((size_t) kPlanWords, -7);
        strata::q36::pcie_plan(t.data(), (int) t.size(), 4, real, 3, ptr, plan.data());
        unsigned long long p2 = 0;
        std::memcpy(&p2, plan.data() + kPlanPtr + 4, 8);
        check(p2 == ptr[2] && plan[kPlanGroups] == 3 && plan[kPlanStart + 3] == 3, "the on-demand group keeps its own slot's address");
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
    std::printf("PcieAuto (q36: the CPU path's own time, the launch counted on the host's side):\n");
    {
        using strata::core::PcieAuto;
        // a 54 GB/s link and a 4.08 MB expert: 75.6 us of wire per blob.  One first update after 48 windows of 100 jobs and
        // 20 blobs, the CPU path at `us` per job, the host's launch at `launch_us` per blob.
        const auto first = [](double us, double launch_us, double factor, double cap, PcieAuto* out = nullptr) {
            PcieAuto a;
            a.on = true;
            a.link_gbps = 54.0;
            a.blob_bytes = 4.08e6;
            a.pool_factor = factor;
            a.max_share = cap;
            a.min_share = 0.0;
            a.mark(0.0, 0, 0.0, 0);
            double tc = 0.0, rate = 0.0, ms = 0.0, lms = 0.0, got = -1.0;
            int64_t jobs = 0, blobs = 0;
            for (int w = 1; w <= 48 && got < 0.0; ++w) {
                ms += us / 10.0;
                jobs += 100;
                lms += launch_us * 20.0 / 1000.0;
                blobs += 20;
                got = a.window(ms, jobs, 0.55, &tc, &rate, lms, blobs);
            }
            if (out != nullptr) *out = a;
            return got;
        };
        // c = 200 us per job, no launch: 200 / (200 + 75.6) = 0.726 -> 0.75 (the fitted 1.85 would say 54 / (54 + 37.7) = 0.59 -> 0.60)
        check(std::abs(first(200.0, 0.0, 1.0, 0.95) - 0.75) < 1e-9, "a CPU path at 200 us per job against 75.6 us of wire: 0.75");
        check(std::abs(first(200.0, 0.0, PcieAuto::kPoolFactor, 0.95) - 0.60) < 1e-9, "the fitted 1.85 on the same reading would say 0.60");
        // the job build counts on the CPU path: 150 us of pool rows alone say 150 / 225.6 = 0.665 -> 0.65, and 50 us of job
        // build on top make the 200 us above (0.75)
        check(std::abs(first(150.0, 0.0, 1.0, 0.95) - 0.65) < 1e-9, "150 us of pool rows alone, no job build counted: 0.65");
        // 40 us of host launch per blob is host time: c / (c + w - l) = 200 / 235.6 = 0.849 -> 0.85
        {
            PcieAuto a;
            check(std::abs(first(200.0, 40.0, 1.0, 0.95, &a) - 0.85) < 1e-9, "40 us of launch per blob moves the balance to 0.85");
            check(std::abs(a.launch_us - 40.0) < 1e-6, "the reading's launch time is 40 us per blob");
        }
        check(std::abs(first(200.0, 80.0, 1.0, 0.95) - 0.95) < 1e-9, "a launch dearer than the wire leaves only the cap: 0.95");
        check(std::abs(first(200.0, 80.0, 1.0, 0.80) - 0.80) < 1e-9, "the cap holds");
        check(std::abs(first(1000.0, 0.0, 1.0, 0.95) - 0.95) < 1e-9, "a CPU path at 1000 us per job: 1000 / 1075.6 = 0.93 -> 0.95");
        // launch time with no blob counted (a window where nothing went over PCIe) is read as none
        {
            PcieAuto a;
            a.on = true;
            a.link_gbps = 54.0;
            a.blob_bytes = 4.08e6;
            a.pool_factor = 1.0;
            a.max_share = 0.95;
            a.mark(0.0, 0, 0.0, 0);
            double tc = 0.0, rate = 0.0, ms = 0.0, got = -1.0;
            int64_t jobs = 0;
            for (int w = 1; w <= 48 && got < 0.0; ++w) {
                ms += 20.0;
                jobs += 100;
                got = a.window(ms, jobs, 0.55, &tc, &rate, 5.0 * w, 0);
            }
            check(std::abs(got - 0.75) < 1e-9 && a.launch_us == 0.0, "a launch time without blobs counts as none");
        }
    }
    std::printf("PcieAuto (Flash-Next unchanged):\n");
    {
        using strata::core::PcieAuto;
        // the arithmetic of the rule before the launch time and the pool factor existed, written out, against the struct on a
        // grid of links, experts and pool times: the same share (bit for bit: the same double) on every one
        PcieAuto probe;
        check(probe.pool_factor == PcieAuto::kPoolFactor && PcieAuto::kPoolFactor == 1.85, "the default factor is Flash-Next's 1.85");
        bool same = true;
        int n = 0;
        for (const double link : {2.0, 6.0, 12.0, 26.0, 28.0, 54.0}) {
            for (const double blob : {2.0e6, 4.08e6, 5.2e6}) {
                for (const double us : {50.0, 100.0, 250.0, 400.0, 625.0, 1200.0, 2500.0}) {
                    for (const double cap : {0.60, 0.95}) {
                        PcieAuto a;
                        a.on = true;
                        a.link_gbps = link;
                        a.blob_bytes = blob;
                        a.max_share = cap;
                        a.mark(0.0, 0);
                        double tc = 0.0, rate = 0.0, ms = 0.0, got = -1.0;
                        int64_t jobs = 0;
                        for (int w = 1; w <= 48 && got < 0.0; ++w) {
                            ms += us / 10.0;
                            jobs += 100;
                            got = a.window(ms, jobs, 0.55, &tc, &rate);
                        }
                        const double r = blob / (tc * 1e-6) / 1e9;
                        const double f = std::min(cap, std::max(0.10, std::round(link / (link + PcieAuto::kPoolFactor * r) * 20.0) / 20.0));
                        const double want = std::fabs(f - 0.55) < 0.025 ? -1.0 : f;
                        same = same && got == want && a.launch_us == 0.0;
                        ++n;
                    }
                }
            }
        }
        char what[120];
        std::snprintf(what, sizeof what, "%d readings: the share is the old formula's, bit for bit, and no launch time is read", n);
        check(same, what);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
