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
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::q36 {

inline constexpr int kPcieSlots = 16;          // experts one layer step can send over PCIe (the staging region's slots)
inline constexpr int kPcieMaxEntries = 256;    // entries (rows x top_k) of a step that may carry a share (k_hy_split's limit)
inline constexpr int kPcieGroupRows = 4;       // the PCIe call's groups side by side (Flash-Next's kPcieGroupRows)
inline constexpr int kPlanGroups = 0;
inline constexpr int kPlanPtr = 2;
inline constexpr int kPlanStart = kPlanPtr + 2 * kPcieSlots;
inline constexpr int kPlanDst = kPlanStart + kPcieSlots + 1;
inline constexpr int kPlanTok = kPlanDst + kPcieMaxEntries;
inline constexpr int kPlanWords = kPlanTok + kPcieMaxEntries;

/// The experts of one layer step that go over PCIe: the distinct missed experts in routing order (first appearance),
/// the last (misses * pcie_num) >> 8 of them, at most `cap`.  `slot_of[x] < 0`: expert x has no VRAM slot.  Writes them
/// to `picked` (in routing order) and returns how many.
inline int pcie_pick(const int32_t* ids, int n_ent, const std::vector<int>& slot_of, int pcie_num, int cap, int32_t* picked) {
    if (pcie_num <= 0 || n_ent <= 0 || n_ent > kPcieMaxEntries) return 0;
    std::vector<char> seen(slot_of.size(), 0);
    std::vector<int32_t> misses;
    for (int e = 0; e < n_ent; ++e) {
        const int32_t x = ids[e];
        if (x < 0 || (size_t) x >= slot_of.size() || seen[(size_t) x] || slot_of[(size_t) x] >= 0) continue;
        seen[(size_t) x] = 1;
        misses.push_back(x);
    }
    const int m = std::min<int>(std::min(cap, kPcieSlots), ((int) misses.size() * pcie_num) >> 8);
    for (int q = 0; q < m; ++q) picked[q] = misses[misses.size() - (size_t) m + (size_t) q];
    return m;
}

/// The plan for `m` picked experts: group q is picked[q], its blob at stage + q * blob.  An entry belongs to the group of
/// its expert.  `plan` has kPlanWords words.
inline void pcie_plan(const int32_t* ids, int n_ent, int top_k, const int32_t* picked, int m, unsigned long long stage,
                      unsigned long long blob, int32_t* plan) {
    plan[kPlanGroups] = m;
    int entries = 0;
    for (int q = 0; q < m; ++q) {
        const unsigned long long ptr = stage + (unsigned long long) q * blob;
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
