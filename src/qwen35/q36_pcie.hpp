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
#include <vector>

namespace strata::q36 {

inline constexpr int kPcieSlots = 16;          // experts one layer step can send over PCIe (the staging region's slots)
inline constexpr int kPrefetchSlots = 8;       // blobs the prefetch copies ahead for one layer (one region; two regions by layer parity)
inline constexpr int kPcieStageBlobs = kPcieSlots + 2 * kPrefetchSlots;   // the staging area: the layer's own slots, then the two regions
inline constexpr int kPcieMaxEntries = 256;    // entries (rows x top_k) of a step that may carry a share (k_hy_split's limit)
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
/// runs its bigger batches on the CPU pool alone.
inline int pcie_pick(const int32_t* ids, int n_ent, const std::vector<int>& slot_of, int pcie_num, int32_t* picked,
                     int* announced = nullptr, int cap = kPcieSlots) {
    if (announced != nullptr) *announced = 0;
    if (pcie_num <= 0 || n_ent <= 0 || n_ent > kPcieMaxEntries) return 0;
    int32_t misses[kPcieMaxEntries];
    int n_miss = 0;
    for (int e = 0; e < n_ent; ++e) {
        const int32_t x = ids[e];
        if (x < 0 || (size_t) x >= slot_of.size() || slot_of[(size_t) x] >= 0 || std::find(misses, misses + n_miss, x) != misses + n_miss) continue;
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

}  // namespace strata::q36
