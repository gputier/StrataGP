// src/qwen35/q36_rules.hpp - the pure rules of strata-q36's routing and hybrid mode, shared by the routing kernel (device)
// and the host tests: the SER keep rule, the cache-prior bias and the graph guard of the hybrid layers.
#pragma once

#include "q36_delta.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace strata::q36 {

/// SER (ik_llama.cpp PR 239), k_route_group: the pick of rank `rank` is kept when it is one of the first ser_kmin picks or
/// its probability p exceeds ser_t times the top probability `top`: the row's largest unbiased probability, 1 / sum of the
/// exps (the max logit's).  Without cache-prior that is the rank-0 pick's probability; with it the rank-0 pick is the largest
/// biased logit, whose own probability may be lower than `top`.
Q36_HD inline bool ser_keep(int rank, int ser_kmin, float p, float ser_t, float top) {
    return rank < ser_kmin || p > ser_t * top;
}

/// --ser-pool, k_route_group: SER only on the picks whose expert has no VRAM slot.  A pick with a slot (`resident`) is always
/// kept: that expert is read almost for free, only a pick served by the CPU pool costs.
Q36_HD inline bool ser_pool_keep(int rank, float p, float top, int ser_kmin, float ser_t, bool resident) {
    return resident || ser_keep(rank, ser_kmin, p, ser_t, top);
}

/// --ser-pool's `resident`: the expert has a VRAM slot.  A layer without a slot table (`slot_of` null: not hybrid, or the
/// MTP-only hybrid mode's main layers) holds every expert in VRAM, so each of its picks is resident and none is cut.
Q36_HD inline bool ser_pool_resident(const int* slot_of, int expert) {
    return slot_of == nullptr || slot_of[expert] >= 0;
}

/// cache-prior (arXiv 2412.00099, Eq. 9), k_route_group: the amount added to an expert's logit in the ranking.  It is
/// cache_prior * Delta_avg for an expert with a VRAM slot (`resident`) in a hybrid layer, and 0 otherwise or when
/// cache_prior is not positive.
Q36_HD inline float cache_prior_bias(bool hybrid, bool resident, float cache_prior, float delta_avg) {
    return (hybrid && resident && cache_prior > 0.0f) ? cache_prior * delta_avg : 0.0f;
}

/// place_experts' graph guard: the graphs (a decode step, a verify window) are off when the main layers are hybrid and a
/// batch past hybrid_t rows would stream the layer's experts, which a capture cannot hold: hybrid_t below verify_rows.
Q36_HD inline bool hybrid_graph_off(bool main_hybrid, int hybrid_t, int verify_rows) {
    return main_hybrid && hybrid_t < verify_rows;
}

/// The hybrid mode's prompt chunk.  free_gib is the free VRAM the chunk rule reads, counting the packed Q8_0 copies that can
/// be given back (the same fr as the chunk rule in q36_engine.cu).  A chunk streams every layer's experts over PCIe once, so a
/// bigger chunk pays that copy for more tokens.  4096 is taken only where it was measured: an RTX 5090 with 23.2 GiB free on
/// Ornith-1.5-122B, a 34,257-token prompt read 62 % faster than at 2048 for 4 % less decode at short context (issue #104,
/// 25 October 2026).  It is for one session only: with several sessions a chunk holds the other sessions' decode windows for
/// its whole length.  With slots or an expert VRAM budget given by hand (auto_slots false) the 4096 step is not taken, since
/// its buffers would take the room of those slots.  The 7.5 / 5.5 GiB steps are unchanged.
inline int hybrid_prompt_chunk(double free_gib, int sessions, bool auto_slots) {
    if (free_gib >= 20.0 && sessions <= 1 && auto_slots) return 4096;
    return free_gib >= 7.5 ? 2048 : free_gib >= 5.5 ? 1024 : 512;
}

/// The VRAM bytes a prompt chunk needs beside the engine: 2.75 GiB + 0.5 MiB per token over 512, plus `beside` (every expert's
/// bytes and the sessions' K / V past 8192 cells when the experts all fit in VRAM, 0 in hybrid mode, where the slots shrink
/// instead).  The auto rule picks the largest chunk under the free VRAM; STRATA_Q36_CHUNK above 4096 (#104) refuses to start
/// when the free VRAM is below it.
inline double forced_chunk_need(int chunk, double beside) {
    return beside + 2.75 * (double) (1ull << 30) + (double) (chunk - 512) * 0.5 * 1048576.0;
}

/// SER's Kmin per main layer, from STRATA_Q36_SER_TABLE's text: one "layer Kmin" line per layer, blank lines and whatever follows
/// a '#' ignored.  kmin gets n_layers entries: a layer the table does not name takes ser_kmin (--ser's).  Returns false with
/// err set (the line's number in it) when a line is not two integers, a layer is not in 0 .. n_layers - 1, a Kmin is not in
/// 1 .. 8, or a layer is named twice.
inline bool parse_ser_table(const std::string& text, int n_layers, int ser_kmin, std::vector<int>* kmin, std::string* err) {
    kmin->assign((std::size_t) n_layers, ser_kmin);
    std::vector<bool> named((std::size_t) n_layers, false);
    std::istringstream in(text);
    std::string line;
    for (int no = 1; std::getline(in, line); ++no) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream fields(line);
        long long layer = 0, k = 0;
        std::string extra;
        if (!(fields >> layer >> k) || (fields >> extra)) {
            *err = "line " + std::to_string(no) + ": expected two integers, the layer and its Kmin";
            return false;
        }
        if (layer < 0 || layer >= n_layers) {
            *err = "line " + std::to_string(no) + ": layer " + std::to_string(layer) + " is not in 0 .. " + std::to_string(n_layers - 1);
            return false;
        }
        if (k < 1 || k > 8) {
            *err = "line " + std::to_string(no) + ": Kmin " + std::to_string(k) + " is not in 1 .. 8";
            return false;
        }
        if (named[(std::size_t) layer]) {
            *err = "line " + std::to_string(no) + ": layer " + std::to_string(layer) + " is named twice";
            return false;
        }
        named[(std::size_t) layer] = true;
        (*kmin)[(std::size_t) layer] = (int) k;
    }
    return true;
}

/// The Kmin of layer `li`: its entry of the table, or ser_kmin past the table's end (the MTP block).
inline int layer_kmin(const std::vector<int>& kmin_layer, std::size_t li, int ser_kmin) {
    return li < kmin_layer.size() ? kmin_layer[li] : ser_kmin;
}

}  // namespace strata::q36
