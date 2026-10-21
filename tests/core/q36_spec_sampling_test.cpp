// tests/core/q36_spec_sampling_test.cpp - strata-q36's sampled speculative decoding (src/qwen35/q36_sampling.hpp), no GPU.
//
// A small fake model (logits from a hash of the last two tokens) is decoded two ways: plain, one sampled token per
// step; and by verify windows, where row t of a window is picked with the draw of its own position over its own
// penalty history and a draft is kept when it equals the pick of the row before it.  Checked:
//  1. the token stream is identical for every seed, whatever the drafter proposes (argmax, a noisy model, garbage,
//     the perfect draft) and whatever the penalties are: drafting changes how many tokens a window yields, not which;
//  2. the drafts are really accepted and really rejected (the identity above is not vacuous);
//  3. with the logits independent of the context, the token drawn at each position (through windows) follows the
//     softmax within 5 sigma over 20000 seeds, and the acceptance rate of argmax drafts is the argmax' probability.
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
int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

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
    std::printf("q36_spec_sampling_test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
