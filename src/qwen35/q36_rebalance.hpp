// src/qwen35/q36_rebalance.hpp - the hybrid mode's choice of slot replacements (host side, header only).
//
// The routing counts of every hybrid layer decide which experts the VRAM slots should hold.  Flash-Next's adapt()
// rules, applied across layers: in a layer, the most-routed expert outside the slots takes the place of the
// least-routed one inside, while its count is at least kRebalanceMinCount and beats the one it replaces by at least
// kRebalanceMinGain; the pairs of all layers are then ranked by gain and the best ones, up to the call's budget, are
// swapped.  Every layer's counts decay by kRebalanceDecay afterwards (the caller's job).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::q36 {

inline constexpr double kRebalanceMinCount = 2.0;   // a candidate is routed at least this often (decayed count)
inline constexpr double kRebalanceMinGain = 1.5;    // and beats the expert it replaces by at least this much
inline constexpr double kRebalanceDecay = 0.7;      // the counts' factor after each rebalance, in every layer
inline constexpr int kRebalanceEveryDefault = 32;   // positions between two rebalances (STRATA_Q36_REBALANCE_EVERY unset)
inline constexpr int kRebalanceEveryMax = 1024;     // the largest value the variable accepts

/// STRATA_Q36_REBALANCE_EVERY: the positions between two rebalances, decimal digits only (no sign, no space, nothing after
/// them) for a value from 1 to 1024; leading zeros are read as decimal ("08" is 8).  Unset gives 32.  Any other value gives 32
/// as well, and `err` (when given) gets the line the engine prints for it; an accepted value leaves `err` alone.
inline int rebalance_every_from_env(const char* v, std::string* err) {
    if (v == nullptr) return kRebalanceEveryDefault;
    char* end = nullptr;
    const bool digit = v[0] >= '0' && v[0] <= '9';   // strtol alone would also take a leading space or sign
    const long n = digit ? std::strtol(v, &end, 10) : 0;
    if (digit && *end == '\0' && n >= 1 && n <= kRebalanceEveryMax) return (int) n;
    if (err != nullptr)
        *err = std::string("STRATA_Q36_REBALANCE_EVERY=") + v + " is not a whole number from 1 to 1024 in decimal digits: using 32";
    return kRebalanceEveryDefault;
}

/// The draft depth a window at position `pos` may use: its rows are the positions pos .. pos + D, so D stops short of the next
/// rebalance position (a multiple of `every`), and the rebalance falls at the window's end, never inside it.
inline int draft_depth_limit(int64_t pos, int every, int D) {
    return std::min<int>(D, (int) (every - pos % every) - 1);
}

/// One hybrid layer as the rebalance sees it; null pointers: the layer has no slot table (all its experts in VRAM).
struct LayerCounts {
    const std::vector<double>* score = nullptr;   // routing count per expert
    const std::vector<int>* slot_of = nullptr;    // expert -> its VRAM slot, or -1
};

struct SlotSwap {
    int layer = 0, in = 0, out = 0;   // `in` takes the slot `out` holds
    double gain = 0;                  // count(in) - count(out)
};

/// The experts of one layer's `S` slots at the start: the layer's first S experts in the profile's rank order, then, if
/// the profile ranks fewer, the lowest-numbered ones left (the placement without a profile is that alone).
inline std::vector<int> profile_slot_experts(const std::vector<std::pair<int32_t, int32_t>>& ranked, int layer, int S,
                                             int n_expert) {
    std::vector<int> out;
    std::vector<char> used((size_t) n_expert, 0);
    for (const auto& [l, e] : ranked) {
        if ((int) out.size() >= S) break;
        if (l != layer || e < 0 || e >= n_expert || used[(size_t) e]) continue;
        used[(size_t) e] = 1;
        out.push_back(e);
    }
    for (int e = 0; e < n_expert && (int) out.size() < S; ++e)
        if (!used[(size_t) e]) out.push_back(e);
    return out;
}

/// At most `budget` replacements over all layers, best gain first (ties: the lower layer, then the higher count).
inline std::vector<SlotSwap> choose_swaps(const std::vector<LayerCounts>& layers, int budget) {
    std::vector<SlotSwap> swaps;
    std::vector<std::pair<double, int>> cand, vict;
    for (size_t li = 0; li < layers.size(); ++li) {
        if (layers[li].score == nullptr || layers[li].slot_of == nullptr) continue;
        const std::vector<double>& score = *layers[li].score;
        const std::vector<int>& slot_of = *layers[li].slot_of;
        cand.clear();
        vict.clear();
        for (size_t x = 0; x < slot_of.size() && x < score.size(); ++x) {
            if (slot_of[x] < 0) {
                if (score[x] >= kRebalanceMinCount) cand.emplace_back(score[x], (int) x);
            } else {
                vict.emplace_back(score[x], (int) x);
            }
        }
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        std::sort(vict.begin(), vict.end());   // lowest count first, ties by expert index
        const size_t n = std::min(cand.size(), vict.size());
        for (size_t i = 0; i < n; ++i) {
            if (cand[i].first < vict[i].first + kRebalanceMinGain) break;
            swaps.push_back({(int) li, cand[i].second, vict[i].second, cand[i].first - vict[i].first});
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(), [](const SlotSwap& a, const SlotSwap& b) { return a.gain > b.gain; });
    if (budget < 0) budget = 0;
    if (swaps.size() > (size_t) budget) swaps.resize((size_t) budget);
    return swaps;
}

/// The placement change of one replacement: `in` takes the slot `out` held, and that slot's blob is then filled from the
/// host copy of expert `in`, which the caller does through copy(slot, in).  Returns the slot.
template <class Copy>
inline int apply_slot_swap(std::vector<int>& slot_of, const SlotSwap& w, Copy&& copy) {
    const int s = slot_of[(size_t) w.out];
    slot_of[(size_t) w.out] = -1;
    slot_of[(size_t) w.in] = s;
    copy(s, w.in);
    return s;
}

}  // namespace strata::q36
