// tests/core/q36_rebalance_swaps_test.cpp - the replacements strata-q36's rebalance() applies (src/qwen35/q36_rebalance.hpp),
// no GPU: over many generated sets of counts and several rounds, the replacements are applied in order by an apply step
// modelled in this test after rebalance() in src/qwen35/q36_engine.cu (the slot of `out` goes to `in`, `in`'s blob is
// copied into that slot); the engine code itself is not run.  Every replacement must take a slot that is held and give it
// to an expert that has none, the slots must stay a permutation, and every slot must hold the blob of the expert that maps
// to it.  After a call that took every pair, a second call on the same counts, or after the decay alone, replaces nothing.
#include "check.hpp"
#include "q36_rebalance.hpp"

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {
using strata::q36::LayerCounts;
using strata::q36::SlotSwap;

constexpr int kNoBudget = 1 << 30;   // every pair that passes the thresholds

// std::mt19937's output sequence is fixed by the standard, the distributions are not: only the raw engine output modulo n
// is used, so a fixed seed gives the same counts on every platform
int below(std::mt19937& r, int n) { return (int) (r() % (uint32_t) n); }

// one hybrid layer of the test: the slot table, the counts, and the expert whose blob each slot holds
struct TestLayer {
    std::vector<int> slot_of;
    std::vector<double> score;
    std::vector<int> blob;   // slot -> the expert copied there last; its size is the number of slots
};

// S distinct experts drawn at random in slots 0 .. S - 1, the counts at 0
TestLayer make_layer(std::mt19937& r, int ne, int S) {
    TestLayer L;
    L.slot_of.assign((size_t) ne, -1);
    L.score.assign((size_t) ne, 0.0);
    L.blob.assign((size_t) S, -1);
    std::vector<int> perm((size_t) ne);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < S; ++i) {
        std::swap(perm[(size_t) i], perm[(size_t) (i + below(r, ne - i))]);
        L.slot_of[(size_t) perm[(size_t) i]] = i;
        L.blob[(size_t) i] = perm[(size_t) i];
    }
    return L;
}

// one step of routing: a quarter of the experts counted, on a half-step grid up to 10 (ties between candidates, between
// victims and across the two)
void add_counts(std::mt19937& r, TestLayer& L) {
    for (double& v : L.score)
        if (below(r, 4) == 0) v += 0.5 * below(r, 21);
}

// rebalance()'s rule for a view: a layer with no slot or with every expert in a slot has none
bool has_view(const TestLayer& L) { return !L.blob.empty() && L.blob.size() < L.slot_of.size(); }

// the views rebalance() builds
std::vector<LayerCounts> views_of(const std::vector<TestLayer>& layers) {
    std::vector<LayerCounts> views(layers.size());
    for (size_t li = 0; li < layers.size(); ++li)
        if (has_view(layers[li])) views[li] = {&layers[li].score, &layers[li].slot_of};
    return views;
}

// every slot held by exactly one expert, and holding that expert's blob
bool consistent(const TestLayer& L) {
    const int S = (int) L.blob.size();
    std::vector<int> holder((size_t) S, -1);
    int residents = 0;
    for (size_t x = 0; x < L.slot_of.size(); ++x) {
        const int s = L.slot_of[x];
        if (s < 0) continue;
        if (s >= S || holder[(size_t) s] != -1 || L.blob[(size_t) s] != (int) x) return false;
        holder[(size_t) s] = (int) x;
        ++residents;
    }
    return residents == S;
}

// The replacements of one call, modelled after rebalance() in src/qwen35/q36_engine.cu, in its order: s = slot_of[out];
// slot_of[out] = -1; slot_of[in] = s; in's blob into slot s.  Returns false at the first replacement that could not be
// applied safely: a layer rebalance() gives no view (has_view), an expert out of range, an `out` with no slot (the copy
// would land at slot -1) or an `in` that already has one (its old slot would be lost).
bool apply(std::vector<TestLayer>& layers, const std::vector<SlotSwap>& swaps) {
    for (const SlotSwap& w : swaps) {
        if (w.layer < 0 || w.layer >= (int) layers.size() || !has_view(layers[(size_t) w.layer])) return false;
        TestLayer& L = layers[(size_t) w.layer];
        const int ne = (int) L.slot_of.size();
        if (w.in < 0 || w.in >= ne || w.out < 0 || w.out >= ne || w.in == w.out) return false;
        const int s = L.slot_of[(size_t) w.out];
        if (s < 0 || s >= (int) L.blob.size() || L.slot_of[(size_t) w.in] >= 0) return false;
        L.slot_of[(size_t) w.out] = -1;
        L.slot_of[(size_t) w.in] = s;
        L.blob[(size_t) s] = w.in;
    }
    return true;
}

// the list is ranked by gain, each gain is count(in) - count(out) and passes the thresholds
bool ranked_and_passing(const std::vector<TestLayer>& layers, const std::vector<SlotSwap>& swaps) {
    for (size_t i = 0; i < swaps.size(); ++i) {
        const SlotSwap& w = swaps[i];
        if (i > 0 && swaps[i - 1].gain < w.gain) return false;
        if (w.layer < 0 || w.layer >= (int) layers.size()) return false;
        const std::vector<double>& score = layers[(size_t) w.layer].score;
        if (w.in < 0 || w.in >= (int) score.size() || w.out < 0 || w.out >= (int) score.size()) return false;
        if (w.gain != score[(size_t) w.in] - score[(size_t) w.out]) return false;
        if (score[(size_t) w.in] < strata::q36::kRebalanceMinCount) return false;
        if (score[(size_t) w.in] < score[(size_t) w.out] + strata::q36::kRebalanceMinGain) return false;
    }
    return true;
}

bool same(const SlotSwap& a, const SlotSwap& b) { return a.layer == b.layer && a.in == b.in && a.out == b.out && a.gain == b.gain; }
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("targeted:\n");
    {
        // one slot, seven hot experts outside: a single replacement, the resident out, the hottest in
        std::mt19937 r(1);
        std::vector<TestLayer> layers = {make_layer(r, 8, 1)};
        const int resident = layers[0].blob[0];
        for (int x = 0; x < 8; ++x)
            if (x != resident) layers[0].score[(size_t) x] = 3.0 + x;
        const int hottest = resident == 7 ? 6 : 7;
        const std::vector<LayerCounts> views = views_of(layers);
        const std::vector<SlotSwap> w = choose_swaps(views, kNoBudget);
        check(w.size() == 1 && w[0].out == resident && w[0].in == hottest,
              "more candidates than slots: one replacement per slot, never a second one for the same slot");
        check(apply(layers, w) && consistent(layers[0]), "applied, the slot holds the new expert's blob");
    }
    {
        // every resident hotter than every expert outside: nothing moves
        std::mt19937 r(2);
        std::vector<TestLayer> layers = {make_layer(r, 16, 4)};
        for (int x = 0; x < 16; ++x) layers[0].score[(size_t) x] = layers[0].slot_of[(size_t) x] >= 0 ? 9.0 : 7.5;
        check(choose_swaps(views_of(layers), kNoBudget).empty(), "residents ahead of every candidate: no replacement");
    }
    {
        // a layer with every expert in a slot, and one with none: rebalance() gives them no view, and choose_swaps given
        // their tables would replace nothing either (no candidate, no victim)
        std::mt19937 r(3);
        TestLayer full = make_layer(r, 8, 8), none = make_layer(r, 8, 0);
        for (double& v : full.score) v = 5.0;
        for (double& v : none.score) v = 5.0;
        check(choose_swaps({{&full.score, &full.slot_of}}, kNoBudget).empty(), "every expert resident: no replacement");
        check(choose_swaps({{&none.score, &none.slot_of}}, kNoBudget).empty(), "no expert resident: no replacement");
    }
    {
        // four residents with different counts, two hot experts outside: the hottest evicts the coldest resident, the
        // next one the next coldest; residents 3 and 6 tie at 0.25, the lower index goes first; resident 1 (4.0) stays.
        // Expert 0 (1.875) is below kRebalanceMinCount but beats resident 6 by 1.625: only the count gate keeps it out
        TestLayer L;
        L.slot_of = {-1, 0, -1, 1, 2, -1, 3, -1};
        L.score = {1.875, 4.0, 5.0, 0.25, 0.0, 0.0, 0.25, 6.0};
        L.blob = {1, 3, 4, 6};
        std::vector<TestLayer> layers = {L};
        const std::vector<LayerCounts> views = views_of(layers);
        const std::vector<SlotSwap> w = choose_swaps(views, kNoBudget);
        check(w.size() == 2, "victims taken lowest count first: two replacements, one per hot expert outside");
        bool gated = true;
        for (const SlotSwap& s : w)
            if (s.in == 0) gated = false;
        check(gated, "an expert below kRebalanceMinCount is never taken, even with a gain above kRebalanceMinGain");
        check(w.size() == 2 && w[0].in == 7 && w[0].out == 4 && w[0].gain == 6.0,
              "victims taken lowest count first: the hottest (7) evicts the coldest resident (4)");
        check(w.size() == 2 && w[1].in == 2 && w[1].out == 3 && w[1].gain == 4.75,
              "victims taken lowest count first: then 2 evicts resident 3, ties by the lower expert index (3 before 6)");
        const std::vector<int>& slot_of = layers[0].slot_of;
        check(apply(layers, w), "applied");
        check(consistent(layers[0]), "applied, every slot is held by one expert and holds that expert's blob");
        check(slot_of[7] == 2 && slot_of[2] == 1, "applied, the two coldest residents' slots hold the new experts");
        check(slot_of[1] == 0 && slot_of[6] == 3, "applied, the other residents keep theirs");
        check(slot_of[4] == -1 && slot_of[3] == -1, "applied, the two evicted residents hold no slot");
    }
    {
        // one slot, held by expert 1 at 0.0; experts 0 and 2 outside, tied at 5.0, both above the thresholds
        const std::vector<double> score = {5.0, 0.0, 5.0};
        const std::vector<int> slot_of = {-1, 0, -1};
        const std::vector<SlotSwap> w = choose_swaps({{&score, &slot_of}}, kNoBudget);
        check(w.size() == 1 && w[0].in == 0 && w[0].out == 1 && w[0].gain == 5.0,
              "candidates tied on count: the lower expert index (0 before 2) takes the slot");
    }
    {
        // two layers whose best replacements tie on gain (6.0 - 1.0 and 7.0 - 2.0), a budget of 1: the lower layer wins,
        // although layer 1's candidate has the higher count
        const std::vector<double> score0 = {1.0, 6.0}, score1 = {7.0, 2.0, 0.0};
        const std::vector<int> slot_of0 = {0, -1}, slot_of1 = {-1, 0, -1};
        const std::vector<SlotSwap> w = choose_swaps({{&score0, &slot_of0}, {&score1, &slot_of1}}, 1);
        check(w.size() == 1 && w[0].layer == 0 && w[0].in == 1 && w[0].out == 0 && w[0].gain == 5.0,
              "replacements tied on gain across layers, a budget of 1: the lower layer's is kept");
    }

    std::printf("generated rounds:\n");
    bool safe = true, kept = true, ranked = true, prefix = true, settled = true, decayed = true;
    long long applied = 0, budget_cut = 0, took_all_calls = 0;   // took_all_calls: the calls the last two checks ran on
    const int sizes[] = {8, 16, 256};
    const int budgets[] = {1, 2, 3, 8, kNoBudget};
    std::mt19937 r(0x5eed);
    for (int trial = 0; trial < 300; ++trial) {
        const int ne = sizes[below(r, 3)];
        const int n_layers = 1 + below(r, 5);
        std::vector<TestLayer> layers;
        for (int li = 0; li < n_layers; ++li) {
            // mostly a partial layer; now and then one with no slot or every expert in a slot (no view)
            const int pick = below(r, 8);
            const int S = pick == 0 ? 0 : pick == 1 ? ne : 1 + below(r, ne - 1);
            layers.push_back(make_layer(r, ne, S));
        }
        const std::vector<LayerCounts> views = views_of(layers);   // the vectors stay in place from here on
        const auto replaces_any = [&views] { return !choose_swaps(views, kNoBudget).empty(); };
        for (int call = 0; call < 6; ++call) {
            for (TestLayer& L : layers) add_counts(r, L);
            const int budget = budgets[below(r, 5)];
            const std::vector<SlotSwap> all = choose_swaps(views, kNoBudget);
            const std::vector<SlotSwap> w = choose_swaps(views, budget);
            const size_t want = all.size() < (size_t) budget ? all.size() : (size_t) budget;
            if (w.size() != want) prefix = false;
            for (size_t i = 0; i < w.size() && i < all.size(); ++i)
                if (!same(w[i], all[i])) prefix = false;
            if (w.size() < all.size()) ++budget_cut;
            if (!ranked_and_passing(layers, w)) ranked = false;
            if (!apply(layers, w)) safe = false;
            applied += (long long) w.size();
            for (const TestLayer& L : layers)
                if (!consistent(L)) kept = false;
            const bool took_all = w.size() == all.size();
            if (took_all) ++took_all_calls;
            if (took_all && replaces_any()) settled = false;
            for (TestLayer& L : layers)
                for (double& v : L.score) v *= kRebalanceDecay;
            if (took_all && replaces_any()) decayed = false;
        }
    }
    check(applied > 300 && budget_cut > 30, "the generated counts made replacements, and budgets cut some lists");
    check(safe, "every replacement names a layer with a view, takes a held slot and gives it to an expert without one");
    check(kept, "after each call every slot is held by one expert and holds that expert's blob");
    check(ranked, "the replacements are ranked by gain, each gain is count(in) - count(out) and passes both thresholds");
    check(prefix, "a budget keeps the first replacements of the full list, in the same order");
    check(took_all_calls > 0, "some calls took every pair, so the two checks below ran");
    check(settled, "a call that took every pair: a second call on the same counts replaces nothing");
    check(decayed, "nor does a call after the decay alone (no replacement back and forth without new routing)");
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
