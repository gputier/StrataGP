// tests/core/q36_spec_sampling_test.cpp - strata-q36's sampled speculative decoding (src/qwen35/q36_sampling.hpp), no GPU.
//
// A small fake model (logits from a hash of the last two tokens) is decoded two ways: plain, one sampled token per
// step; and by verify windows, where row t of a window is picked with the draw of its own position over its own
// penalty history and a draft is kept when it equals the pick of the row before it.  Checked:
//  1. the token stream is identical for every seed, whatever the drafter proposes (argmax, a noisy model, garbage,
//     the perfect draft) and whatever the penalties are: drafting changes how many tokens a window yields, not which;
//  2. the drafts are really accepted and really rejected (the identity above is not vacuous);
//  3. with the logits independent of the context, the token drawn at each position (through windows) follows the
//     softmax within 5 sigma over 20000 seeds, and the acceptance rate of argmax drafts is the argmax' probability;
//  4. the cut (top_k, top_p, min_p on the untempered distribution, temperature for the draw only) matches a numeric
//     reference: same support, same frequencies within 5 sigma;
//  5. the penalties (repeat, frequency, presence, over the last_n window, drafts of a verify row included) match a
//     numeric reference;
//  6. slots decoded side by side, one token each in turn (the batch rule: the draw of the fed token's position), give
//     for every seed the text of the same request decoded alone.
#include "check.hpp"
#include "q36_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
using strata::q36::draw_rng;
using strata::q36::sample;
using strata::q36::Sampling;
using strata::q36::spec_accept;

constexpr int kV = 40;

// the next-token logits after a sequence; salt != 0 is the drafter, a different model close to the target
std::vector<float> row_logits(const std::vector<int32_t>& seq, bool context_free, uint32_t salt) {
    const uint32_t a = context_free ? 0u : (uint32_t) seq.back(), b = context_free || seq.size() < 2 ? 0u : (uint32_t) seq[seq.size() - 2];
    std::mt19937 g(a * 2654435761u + b * 40503u + 17u);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> lg(kV);
    for (auto& v : lg) v = nd(g);
    if (salt != 0) {
        std::mt19937 gn(salt + a * 977u + b);
        std::normal_distribution<float> noise(0.f, 0.8f);
        for (auto& v : lg) v += noise(gn);
    }
    return lg;
}

int argmax(const std::vector<float>& lg) { return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()); }

// plain decoding: the token after position p is sampled with the draw of counter p over the whole history
std::vector<int32_t> plain(std::vector<int32_t> seq, int n, const Sampling& s, bool cf) {
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        std::vector<float> lg = row_logits(seq, cf, 0);
        std::mt19937_64 r = draw_rng(s.seed, seq.size() - 1);
        const int t = sample(lg, s, seq, r);
        out.push_back(t);
        seq.push_back(t);
    }
    return out;
}

enum class Drafter { Argmax, Noisy, Garbage, Perfect };

struct Stats { long offered = 0, accepted = 0, judged = 0; };   // judged: the rows a draft was checked on

// windows of D drafts: x is the decided next token (not in the state), the drafter chains D tokens from it
std::vector<int32_t> windowed(std::vector<int32_t> seq, int n, int D, const Sampling& s, bool cf, Drafter dk, Stats& st) {
    std::vector<int32_t> out;
    auto first = [&] {
        std::vector<float> lg = row_logits(seq, cf, 0);
        std::mt19937_64 r = draw_rng(s.seed, seq.size() - 1);
        return (int32_t) sample(lg, s, seq, r);
    };
    int32_t x = first();
    std::mt19937 junk(s.seed * 31 + 5);
    while ((int) out.size() < n) {
        out.push_back(x);
        if ((int) out.size() >= n) break;
        seq.push_back(x);   // the history holds x; the window's rows follow it
        std::vector<int32_t> drafts;
        std::vector<int32_t> chain = seq;
        for (int d = 0; d < D; ++d) {
            int t;
            if (dk == Drafter::Garbage) t = (int) (junk() % kV);
            else if (dk == Drafter::Perfect) {   // what the target would pick: the oracle
                std::vector<float> lg = row_logits(chain, cf, 0);
                std::mt19937_64 r = draw_rng(s.seed, chain.size() - 1);
                std::vector<int32_t> h(seq.begin(), seq.end());
                t = sample(lg, s, h, r, drafts.data(), drafts.size());
            } else t = argmax(row_logits(chain, cf, dk == Drafter::Noisy ? 7u : 0u));
            drafts.push_back(t);
            chain.push_back(t);
        }
        // the verify rows: row t follows seq (x last) and the drafts 0 .. t-1, its draw is the one of its position
        int32_t picks[16];
        const int64_t pos_x = (int64_t) seq.size() - 1;
        const int a = spec_accept(drafts.data(), D, picks, [&](int t) {
            std::vector<int32_t> ctx = seq;
            ctx.insert(ctx.end(), drafts.begin(), drafts.begin() + t);
            std::vector<float> lg = row_logits(ctx, cf, 0);
            std::mt19937_64 r = draw_rng(s.seed, (uint64_t) (pos_x + t));
            return (int32_t) sample(lg, s, seq, r, drafts.data(), (size_t) t);
        });
        st.offered += D;
        st.accepted += a;
        st.judged += a + (a < D);
        for (int i = 0; i < a && (int) out.size() < n; ++i) out.push_back(drafts[(size_t) i]);
        seq.insert(seq.end(), drafts.begin(), drafts.begin() + a);
        x = picks[a];
    }
    out.resize((size_t) n);
    return out;
}

// the batch rule: S requests advance one token per turn each, the draw of a token being the one of the position that
// fed it (what the engine's batch windows do), and every request ends with the text it has alone
std::vector<std::vector<int32_t>> batched(const std::vector<Sampling>& ss, const std::vector<int32_t>& prompt, int n) {
    std::vector<std::vector<int32_t>> seq(ss.size(), prompt), out(ss.size());
    for (int i = 0; i < n; ++i)
        for (size_t b = 0; b < ss.size(); ++b) {
            std::vector<float> lg = row_logits(seq[b], false, 0);
            std::mt19937_64 r = draw_rng(ss[b].seed, seq[b].size() - 1);
            const int t = sample(lg, ss[b], seq[b], r);
            out[b].push_back(t);
            seq[b].push_back(t);
        }
    return out;
}

// the distribution a Sampling draws from one row of logits (indexed by token), written from the definition: top_k, then
// top_p and min_p on the untempered softmax of the candidates, then the temperature on the survivors
std::vector<double> cut_reference(const std::vector<float>& lg, const Sampling& s) {
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) std::min<int>(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    const double top = lg[(size_t) idx[0]];
    double z = 0;
    for (int i : idx) z += std::exp((double) lg[(size_t) i] - top);
    size_t keep = idx.size();
    double cum = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
        const double p = std::exp((double) lg[(size_t) idx[k]] - top) / z;
        if (p < s.min_p / z) { keep = k; break; }   // min_p is a fraction of the top token's probability (1 / z)
        cum += p;
        if (cum >= s.top_p) { keep = k + 1; break; }
    }
    keep = std::max<size_t>(keep, 1);
    std::vector<double> out(lg.size(), 0.0);
    double w = 0;
    for (size_t k = 0; k < keep; ++k) w += (out[(size_t) idx[k]] = std::exp(((double) lg[(size_t) idx[k]] - top) / s.temperature));
    for (auto& v : out) v /= w;
    return out;
}

// the logits after the penalties, written from the definition (the history is hist followed by tail)
std::vector<float> penalty_reference(std::vector<float> lg, const Sampling& s, const std::vector<int32_t>& hist, const std::vector<int32_t>& tail) {
    std::vector<int32_t> all = hist;
    all.insert(all.end(), tail.begin(), tail.end());
    if (s.last_n > 0 && (size_t) s.last_n < all.size()) all.erase(all.begin(), all.end() - s.last_n);
    for (size_t id = 0; id < lg.size(); ++id) {
        const int c = (int) std::count(all.begin(), all.end(), (int32_t) id);
        if (c == 0) continue;
        if (s.repeat != 1.f) lg[id] = lg[id] > 0 ? lg[id] / s.repeat : lg[id] * s.repeat;
        lg[id] -= (float) c * s.freq + s.present;
    }
    return lg;
}
}  // namespace

int main() {
    // 1 and 2: identity with plain sampling, the card's sampling (temperature 1, top_k 20, top_p 0.95, presence 1.5) and others
    struct Cfg { const char* name; Sampling s; };
    std::vector<Cfg> cfgs;
    {
        Sampling card; card.temperature = 1.f; card.top_k = 20; card.top_p = 0.95f; card.present = 1.5f; card.last_n = -1;
        Sampling pen; pen.temperature = 0.7f; pen.top_k = 0; pen.top_p = 0.9f; pen.min_p = 0.05f; pen.repeat = 1.15f; pen.freq = 0.3f; pen.present = 0.5f; pen.last_n = 6;
        Sampling cold; cold.temperature = 0.f; cold.present = 1.5f; cold.last_n = -1;   // temperature 0 keeps the penalty
        cfgs = {{"the model card's sampling", card}, {"windowed penalties + min_p", pen}, {"temperature 0 with the presence penalty", cold}};
    }
    const Drafter kinds[] = {Drafter::Argmax, Drafter::Noisy, Drafter::Garbage, Drafter::Perfect};
    const char* kind_names[] = {"argmax", "noisy model", "garbage", "perfect"};
    for (const Cfg& c : cfgs) {
        for (int k = 0; k < 4; ++k) {
            for (int D : {1, 2, 3}) {
                Stats st;
                int differ = 0;
                for (uint64_t seed = 1; seed <= 150; ++seed) {
                    Sampling s = c.s;
                    s.seed = seed;
                    const std::vector<int32_t> prompt = {3, 17, 5, 9};
                    if (plain(prompt, 24, s, false) != windowed(prompt, 24, D, s, false, kinds[k], st)) ++differ;
                }
                char what[160];
                std::snprintf(what, sizeof what, "%s, %s drafter, %d drafts: 150 seeds x 24 tokens identical to plain (%ld of %ld drafts kept)",
                              c.name, kind_names[k], D, st.accepted, st.offered);
                check(differ == 0, what);
                if (k < 2 && c.s.temperature > 0.f) check(st.accepted > 0 && st.accepted < st.offered, "  drafts both kept and rejected");
            }
        }
    }

    // 3: context-free logits, no penalties, the whole vocabulary: the token at each position is softmax-distributed
    {
        Sampling s;
        s.temperature = 1.f; s.top_k = 0; s.top_p = 1.f; s.min_p = 0.f;
        const std::vector<float> lg = row_logits({0}, true, 0);
        std::vector<double> p(kV);
        double z = 0;
        const double mx = lg[(size_t) argmax(lg)];
        for (int i = 0; i < kV; ++i) z += (p[(size_t) i] = std::exp((double) lg[(size_t) i] - mx));
        for (auto& v : p) v /= z;
        constexpr int N = 20000, kPos = 6;
        std::vector<std::vector<int>> cnt(kPos, std::vector<int>(kV, 0));
        Stats st;
        for (uint64_t seed = 1; seed <= N; ++seed) {
            s.seed = seed;
            const std::vector<int32_t> out = windowed({1, 2, 3}, kPos, 2, s, true, Drafter::Argmax, st);
            for (int k = 0; k < kPos; ++k) ++cnt[(size_t) k][(size_t) out[(size_t) k]];
        }
        double worst = 0;
        for (int k = 0; k < kPos; ++k)
            for (int i = 0; i < kV; ++i) {
                const double sd = std::sqrt(p[(size_t) i] * (1 - p[(size_t) i]) / N);
                worst = std::max(worst, std::fabs(cnt[(size_t) k][(size_t) i] / (double) N - p[(size_t) i]) / sd);
            }
        char what[160];
        std::snprintf(what, sizeof what, "token distribution at %d positions over %d seeds: worst deviation %.2f sigma (bound 5)", kPos, N, worst);
        check(worst < 5.0, what);
        const double pmax = p[(size_t) argmax(lg)], rate = st.accepted / (double) st.judged;
        std::snprintf(what, sizeof what, "argmax drafts kept per judged row at rate %.4f, argmax probability %.4f (5 sigma: %.4f)", rate, pmax,
                      5 * std::sqrt(pmax * (1 - pmax) / st.judged));
        check(std::fabs(rate - pmax) < 5 * std::sqrt(pmax * (1 - pmax) / st.judged), what);
    }

    // 4: the cut against the numeric reference: the support is exactly the reference's and the frequencies follow it
    {
        const std::vector<float> lg = row_logits({4, 8}, false, 0);
        struct Cut { const char* name; Sampling s; };
        std::vector<Cut> cuts;
        {
            Sampling a; a.temperature = 1.f; a.top_k = 5; a.top_p = 1.f;
            Sampling b; b.temperature = 0.6f; b.top_k = 0; b.top_p = 0.8f;
            Sampling c; c.temperature = 1.7f; c.top_k = 0; c.top_p = 1.f; c.min_p = 0.15f;
            Sampling d; d.temperature = 0.8f; d.top_k = 12; d.top_p = 0.9f; d.min_p = 0.05f;
            cuts = {{"top_k 5", a}, {"top_p 0.8 at temperature 0.6", b}, {"min_p 0.15 at temperature 1.7", c}, {"top_k 12 + top_p 0.9 + min_p 0.05 at temperature 0.8", d}};
        }
        constexpr int N = 40000;
        for (const Cut& c : cuts) {
            const std::vector<double> ref = cut_reference(lg, c.s);
            std::vector<int> cnt(kV, 0);
            for (uint64_t seed = 1; seed <= N; ++seed) {
                std::vector<float> row = lg;
                std::mt19937_64 r = draw_rng(seed, 3);
                ++cnt[(size_t) sample(row, c.s, {}, r)];
            }
            bool support = true;
            double worst = 0;
            int kept = 0;
            for (int i = 0; i < kV; ++i) {
                if ((ref[(size_t) i] == 0.0) != (cnt[(size_t) i] == 0)) support = false;
                kept += ref[(size_t) i] > 0.0;
                const double sd = std::sqrt(std::max(ref[(size_t) i] * (1 - ref[(size_t) i]), 1e-12) / N);
                worst = std::max(worst, std::fabs(cnt[(size_t) i] / (double) N - ref[(size_t) i]) / sd);
            }
            char what[200];
            std::snprintf(what, sizeof what, "cut, %s: support is the reference's (%d tokens), worst frequency deviation %.2f sigma (bound 5)", c.name, kept, worst);
            check(support && kept > 1 && worst < 5.0, what);
        }
    }

    // 5: the penalties against the numeric reference, through the pick (top_k 1: the argmax of the penalised row)
    {
        int bad = 0, flipped = 0, rows = 0;
        std::mt19937 g(99);
        for (int it = 0; it < 400; ++it) {
            Sampling s;
            s.temperature = 1.f; s.top_k = 1;
            s.repeat = it % 2 ? 1.3f : 1.f;
            s.freq = it % 3 ? 0.4f : 0.f;
            s.present = it % 5 ? 0.9f : 0.f;
            s.last_n = it % 4 == 0 ? -1 : (it % 4) * 3;
            std::vector<int32_t> hist(10), tail((size_t) (it % 4));
            for (auto& t : hist) t = (int32_t) (g() % 8);
            for (auto& t : tail) t = (int32_t) (g() % 8);
            const std::vector<float> lg = row_logits({(int32_t) (g() % 30), (int32_t) (g() % 30)}, false, 0);
            const std::vector<float> ref = penalty_reference(lg, s, hist, tail);
            std::vector<float> row = lg;
            std::mt19937_64 r = draw_rng(1, 0);
            const int got = sample(row, s, hist, r, tail.data(), tail.size());
            bad += got != argmax(ref);
            flipped += argmax(ref) != argmax(lg);
            ++rows;
        }
        char what[160];
        std::snprintf(what, sizeof what, "penalties: %d rows, the pick is the reference's argmax on all (the penalties change the argmax on %d)", rows, flipped);
        check(bad == 0 && flipped > 20, what);
    }

    // 6: slots side by side give each seed the text it has alone
    {
        int differ = 0;
        for (uint64_t seed = 1; seed <= 60; ++seed) {
            std::vector<Sampling> ss(3);
            for (size_t b = 0; b < ss.size(); ++b) {
                ss[b] = cfgs[b % cfgs.size()].s;
                ss[b].seed = seed * 7 + b;
            }
            const std::vector<int32_t> prompt = {3, 17, 5, 9};
            const auto together = batched(ss, prompt, 24);
            for (size_t b = 0; b < ss.size(); ++b) differ += together[b] != plain(prompt, 24, ss[b], false);
        }
        check(differ == 0, "3 slots side by side: 60 seeds x 24 tokens identical to each request alone");
    }
    std::printf("q36_spec_sampling_test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
