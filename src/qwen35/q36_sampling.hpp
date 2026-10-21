// src/qwen35/q36_sampling.hpp - strata-q36's host sampler and the verify window's acceptance rule (no CUDA: tested on the host).
//
// Flash-Next samples inside its verifier (core/coupled_draft.hpp): row t of a window at position pos0 is picked with the
// draw of counter pos0 + t over the penalised history, and a draft is kept only when it EQUALS the pick of the row before
// it.  The text is then a function of the seed alone, whatever the drafts are: they only decide how many tokens a window
// yields.  Same rule here: draw_rng(seed, counter) gives every position its own generator (the draw of the token after
// position p does not depend on how many tokens were sampled before it, or on the window it came from), row t of a
// window follows the history plain decoding would hold there (the history, the fed-back token, drafts 0 .. t-1), and
// spec_accept keeps the longest run of drafts that equal the picks.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace strata::q36 {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The generator of the draw that picks the token after sequence position `counter` (splitmix64 of the pair).
inline std::mt19937_64 draw_rng(uint64_t seed, uint64_t counter) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull * (counter + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return std::mt19937_64(z ^ (z >> 31));
}

/// The pick from one row of logits (modified in place by the penalties).  The penalty history is `hist` followed by
/// tail[0 .. n_tail), so a verify row counts the window's drafts without copying the prompt.
inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                  const int32_t* tail = nullptr, size_t n_tail = 0) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t total = hist.size() + n_tail;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= total ? 0 : total - (size_t) s.last_n;
        for (size_t i = from; i < total; ++i) ++cnt[i < hist.size() ? hist[i] : tail[i - hist.size()]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    // top_p and min_p cut the untempered distribution (llama.cpp's and Flash-Next's order); the temperature only shapes the draw
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((double) lg[(size_t) idx[i]] - mx));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += (p[i] = std::exp(((double) lg[(size_t) idx[i]] - mx) / s.temperature));
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

/// A verify window's acceptance: pick(t) is the token plain decoding would pick after the window's row t; row t is
/// asked for only when the drafts before it were all kept.  Returns the kept drafts a; picks[0 .. a] hold the picks
/// (picks[a] is the token after the kept ones).
template <class Pick> int spec_accept(const int32_t* drafts, int n_drafts, int32_t* picks, Pick&& pick) {
    int a = 0;
    for (;;) {
        picks[a] = pick(a);
        if (a >= n_drafts || picks[a] != drafts[a]) return a;
        ++a;
    }
}

}  // namespace strata::q36
