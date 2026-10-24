// src/qwen35/q36_pcie.hpp - the PCIe share of the q36 hybrid mode's missed experts (header only, no CUDA).
//
// A hybrid layer's missed experts (no VRAM slot) run on the CPU pool.  With the share on, the host sends some of them to
// the GPU instead, as Flash-Next does: the LAST pcie_num/256 of
// the layer's distinct missed experts in routing order are copied from the pinned host arena into a staging region
// of the GPU while the pool computes the others, and the GPU's grouped expert kernel computes them from there.  The
// rule is Flash-Next's (src/core/expert_source.cpp, plan v0.3 P6), without its opt-in tail skip.  The
// host decides, so the captured graph needs no new decision: it carries a fixed "PCIe call" per hybrid layer whose
// groups come from a plan the host writes to mapped memory before it answers the layer's ring.
//
// The plan, in int32 words (what copy_i32_from_mapped moves to the device):
//   [kPlanGroups]            number of groups (PCIe experts)
//   [kPlanPtr + 2 q]         group q's blob, a device address (two words, 8-byte aligned)
//   [kPlanStart + q]         first entry of group q; [kPlanStart + groups] the entry count
//   [kPlanDst + i]           entry i's row in the layer's expert rows (token * top_k + k)
//   [kPlanTok + i]           entry i's token
//
// The prefetch: while the host serves layer L - 1 it copies the blobs layer L is predicted to send over PCIe (the router
// of L applied to the input of L - 1, published with that layer's routing) into the staging slots of its own region,
// two regions by the layer's parity.  The real routing of layer L still decides: a picked expert whose blob is in the
// region is not copied again, the plan's group points at it; a wrong prediction costs only its copy.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::q36 {

inline constexpr int kPcieSlots = 16;          // experts one layer step can send over PCIe (the staging region's slots)
inline constexpr int kPrefetchSlots = 8;       // blobs the prefetch copies ahead for one layer (one region; two regions by layer parity)
inline constexpr int kPcieStageBlobs = kPcieSlots + 2 * kPrefetchSlots;   // the staging area: the layer's own slots, then the two regions
inline constexpr int kPcieMaxEntries = 256;    // entries (rows x top_k) of a step that may carry a share (k_hy_split's limit)
inline constexpr int kPcieMaxExperts = 512;    // experts of a model that may carry a share (k_group's counts; the load refuses more)
inline constexpr int kPcieGroupRows = 4;       // the PCIe call's groups side by side (Flash-Next's kPcieGroupRows)
inline constexpr int kPlanGroups = 0;
inline constexpr int kPlanPtr = 2;
inline constexpr int kPlanStart = kPlanPtr + 2 * kPcieSlots;
inline constexpr int kPlanDst = kPlanStart + kPcieSlots + 1;
inline constexpr int kPlanTok = kPlanDst + kPcieMaxEntries;
inline constexpr int kPlanWords = kPlanTok + kPcieMaxEntries;

/// False for entry `e` when an earlier entry of its row (top_k entries per row) names the same expert: a pick the SER option
/// dropped is a copy of its row's first pick, and the usage counts of the slot rebalance count the expert once.
inline bool first_of_row(const int32_t* ids, int e, int top_k) {
    for (int q = e / top_k * top_k; q < e; ++q)
        if (ids[q] == ids[e]) return false;
    return true;
}

/// The experts of one layer step that go over PCIe: the distinct missed experts in routing order (first appearance),
/// the last (misses * pcie_num) >> 8 of them, at most `cap` (kPcieSlots: the staging region's slots; the prefetch asks for fewer).  `slot_of[x] < 0`: expert x has no VRAM slot.  Writes them
/// to `picked` (in routing order) and returns how many; `announced`, when given, gets the count the share asked for before
/// the cap cut it.  A step of more than kPcieMaxEntries entries gets no share (k_hy_split serves at
/// most that many): kPcieMaxEntries / top_k rows, 32 at the top-8 of Ornith and Qwen3.6, so a --hybrid-rows above that
/// runs its bigger batches on the CPU pool alone.  A model of more than kPcieMaxExperts experts gets none either: the
/// distinct experts are marked in a bit array of that size.
inline int pcie_pick(const int32_t* ids, int n_ent, const std::vector<int>& slot_of, int pcie_num, int32_t* picked,
                     int* announced = nullptr, int cap = kPcieSlots) {
    if (announced != nullptr) *announced = 0;
    if (pcie_num <= 0 || n_ent <= 0 || n_ent > kPcieMaxEntries || (int) slot_of.size() > kPcieMaxExperts) return 0;
    int32_t misses[kPcieMaxEntries];
    uint64_t seen[kPcieMaxExperts / 64] = {};   // the experts already in misses, one bit each
    int n_miss = 0;
    for (int e = 0; e < n_ent; ++e) {
        const int32_t x = ids[e];
        if (x < 0 || (size_t) x >= slot_of.size() || slot_of[(size_t) x] >= 0) continue;
        const uint64_t bit = (uint64_t) 1 << (x % 64);
        if (seen[x / 64] & bit) continue;
        seen[x / 64] |= bit;
        misses[n_miss++] = x;
    }
    const int want = (n_miss * pcie_num) >> 8;
    if (announced != nullptr) *announced = want;
    const int m = std::min(cap, want);
    for (int q = 0; q < m; ++q) picked[q] = misses[n_miss - m + q];
    return m;
}

/// For each of the `m` picked experts, the slot of the prefetch region (`pf_ids`, `pf_n` experts, in slot order) that already
/// holds its blob, or -1.  Returns how many are there.
inline int pcie_prefetch_match(const int32_t* picked, int m, const int32_t* pf_ids, int pf_n, int* slot) {
    int hits = 0;
    for (int q = 0; q < m; ++q) {
        const int32_t* at = std::find(pf_ids, pf_ids + pf_n, picked[q]);
        slot[q] = at != pf_ids + pf_n ? (int) (at - pf_ids) : -1;
        hits += slot[q] >= 0;
    }
    return hits;
}

/// The plan for `m` picked experts: group q is picked[q], its blob at the device address blob_ptr[q].  An entry belongs to
/// the group of its expert.  `plan` has kPlanWords words.
inline void pcie_plan(const int32_t* ids, int n_ent, int top_k, const int32_t* picked, int m,
                      const unsigned long long* blob_ptr, int32_t* plan) {
    plan[kPlanGroups] = m;
    int entries = 0;
    for (int q = 0; q < m; ++q) {
        const unsigned long long ptr = blob_ptr[q];
        std::memcpy(plan + kPlanPtr + 2 * q, &ptr, sizeof(ptr));
        plan[kPlanStart + q] = entries;
        for (int e = 0; e < n_ent; ++e)
            if (ids[e] == picked[q]) {
                plan[kPlanDst + entries] = e;
                plan[kPlanTok + entries] = e / top_k;
                ++entries;
            }
    }
    plan[kPlanStart + m] = entries;
}

// STRATA_Q36_PCIE_AUTO_STEP=1 (automatic share only): the share moves by one step every kPcieStepPositions decode tokens,
// against the wall time per decode token, instead of PcieAuto's refinement.  The rule works in 256ths, the unit the engine
// stores the share in (pcie_num), so a step and a bound compare what the engine really runs.  The bounds, the margin and the
// number of reversals are choices of the search; none is fitted to a machine's measure.
inline constexpr double kPcieStep = 0.05;            // the share's step
inline constexpr double kPcieStepMin = 0.30;         // the lowest share the step may set
inline constexpr double kPcieStepMax = 0.90;         // the highest share the step may set
inline constexpr int kPcieStepNum = 13;              // kPcieStep in 256ths (12.8 rounded)
inline constexpr int kPcieStepNumMin = 77;           // kPcieStepMin in 256ths (76.8 rounded up: the lowest 256th in range)
inline constexpr int kPcieStepNumMax = 230;          // kPcieStepMax in 256ths (230.4 rounded down: the highest 256th in range)
inline constexpr int kPcieStepPositions = 32;        // decode tokens between two steps
inline constexpr double kPcieStepMargin = 0.03;      // a window is slower or faster than the one before only past 3 %
inline constexpr int kPcieStepTurns = 2;             // reversals after which the search stops on the best share seen
static_assert(kPcieStepNum * 2 - 1 <= kPcieStep * 512 && kPcieStep * 512 < kPcieStepNum * 2 + 1, "kPcieStepNum is kPcieStep rounded");
static_assert(kPcieStepNumMin >= kPcieStepMin * 256 && kPcieStepNumMin - 1 < kPcieStepMin * 256, "the lowest 256th in range");
static_assert(kPcieStepNumMax <= kPcieStepMax * 256 && kPcieStepNumMax + 1 > kPcieStepMax * 256, "the highest 256th in range");

/// STRATA_Q36_PCIE_AUTO_STEP: "1" is on, anything else (or no variable) is off.
inline bool pcie_step_enabled(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

struct PcieStep {
    int num = 0;          // the new share in 256ths (what the engine stores in pcie_num)
    int dir = -1;         // the direction now: -1 lowers the share, +1 raises it
    double share() const { return num / 256.0; }
};

/// The next share under the step rule.  `cur`: the share now, in 256ths.  `t_prev`, `t_cur`: the wall time per decode token
/// over the window before the last step and over the window since it (t_prev < 0: no window before, the first step).
/// `dir_last`: the direction of the last step.  The first step moves in dir_last.  Afterwards a time that rose by more than
/// kPcieStepMargin (relative) turns the direction, one that fell by more than that keeps it, and anything in between (a tie,
/// or a time that is not a number) leaves the share where it is.  A move is one kPcieStepNum, kept in [kPcieStepNumMin,
/// kPcieStepNumMax]; a move the bound blocks turns the direction, so the search moves back inside the range.
inline PcieStep pcie_auto_step(int cur, double t_prev, double t_cur, int dir_last) {
    const auto clamp_num = [](int n) { return std::min(kPcieStepNumMax, std::max(kPcieStepNumMin, n)); };
    PcieStep out;
    out.num = cur;
    out.dir = dir_last < 0 ? -1 : 1;
    if (t_prev >= 0.0) {
        if (t_cur > t_prev * (1.0 + kPcieStepMargin)) out.dir = -out.dir;
        else if (!(t_cur < t_prev * (1.0 - kPcieStepMargin))) return out;   // within the margin: no move
    }
    out.num = clamp_num(cur + out.dir * kPcieStepNum);
    if (out.num == cur) {
        out.dir = -out.dir;
        out.num = clamp_num(cur + out.dir * kPcieStepNum);
    }
    return out;
}

/// The search the engine runs, one call per window of kPcieStepPositions decode tokens: pcie_auto_step until the direction has
/// turned kPcieStepTurns times (a slower window or a bound), then the best share seen (the window with the lowest time per
/// token) for good.  A window whose time is not a finite positive number changes nothing.
struct PcieStepSearch {
    int dir = -1;              // the last step's direction (the first step lowers the share)
    double t_prev = -1.0;      // the time per token of the window before the last move, ms (< 0: none yet)
    int turns = 0;             // reversals so far
    bool done = false;         // the search has stopped: the share stays
    int best_num = -1;         // the share of the fastest window seen, in 256ths (< 0: none yet)
    double best_ms = 0.0;      // that window's time per token, ms
    /// The window just ended ran at share `cur` (256ths) and took `t_ms` per decode token: the share for the next window.
    int next(int cur, double t_ms) {
        if (done || !(t_ms > 0.0 && t_ms < 1e300)) return cur;
        if (best_num < 0 || t_ms < best_ms) {
            best_num = cur;
            best_ms = t_ms;
        }
        const PcieStep s = pcie_auto_step(cur, t_prev, t_ms, dir);
        if (s.num == cur) return cur;   // a tie: the share and the window before stay
        if (s.dir != (dir < 0 ? -1 : 1) && ++turns >= kPcieStepTurns) {
            done = true;
            return best_num;
        }
        dir = s.dir;
        t_prev = t_ms;
        return s.num;
    }
};

/// The time the copy streams are busy: the length of the union of the copies' [begin, end] intervals (ms), so two streams
/// copying at once count once.  Sorts `iv`.
inline double interval_union_ms(std::vector<std::pair<double, double>>& iv) {
    std::sort(iv.begin(), iv.end());
    double total = 0.0, end = 0.0;
    bool open = false;
    for (const std::pair<double, double>& x : iv) {
        if (!open || x.first > end) {
            total += x.second - x.first;
            end = x.second;
            open = true;
        } else if (x.second > end) {
            total += x.second - end;
            end = x.second;
        }
    }
    return total;
}

}  // namespace strata::q36
