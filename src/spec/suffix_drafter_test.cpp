// src/spec/suffix_drafter_test.cpp - plan v0.3 P6: suffix-lookup drafter tests and an offline simulation.
//
//   suffix_drafter_test                      unit tests
//   suffix_drafter_test --simulate [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens, accepts the
// prefix that matches the true next tokens, and commits that prefix plus one model token (the verify pass's
// bonus). tokens/step is the lookup drafter's ceiling speedup on that text when a K+1-token verify costs the same
// as a 1-token step. Two regimes per file:
//   continue  history = first half of the prompt; generate its second half (natural continuation)
//   copy      history = the whole prompt; generate its middle third again (output quoting input: edits)
// No model runs; this measures the drafter on text, not on the model's own outputs.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K) {
    SuffixDrafter d(3, 32, history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    Sim s;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++s.proposals; s.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        d.append(&target[i], commit);
        i += commit;
        s.tokens += (long) commit;
        ++s.steps;
    }
    return s;
}

// Issue #46: a drafter kept across requests (reset lazily by epoch, extended by sync) must propose exactly what a
// fresh drafter built from the same tokens proposes.  `a` and `b` are fed the same continuation, a token at a time,
// and must agree on every proposal (tokens and match length).
bool same_proposals(SuffixDrafter& a, SuffixDrafter& b, const std::vector<int32_t>& cont) {
    int32_t oa[16], ob[16];
    for (size_t i = 0; i <= cont.size(); ++i) {
        const int na = a.propose(16, oa), nb = b.propose(16, ob);
        if (na != nb || a.last_match() != b.last_match() || !std::equal(oa, oa + na, ob)) return false;
        if (i < cont.size()) { a.append(cont[i]); b.append(cont[i]); }
    }
    return a.size() == b.size();
}

// text-like tokens: a small vocabulary with repeated phrases, so trigrams recur and proposals are frequent
std::vector<int32_t> text_like(std::mt19937& rng, size_t n, int vocab) {
    std::vector<int32_t> v;
    std::uniform_int_distribution<int> tok(0, vocab - 1), coin(0, 3);
    while (v.size() < n) {
        if (v.size() > 16 && coin(rng) == 0) {   // quote an earlier passage
            std::uniform_int_distribution<size_t> at(0, v.size() - 8);
            const size_t s = at(rng);
            for (size_t i = 0; i < 8 && v.size() < n; ++i) v.push_back(v[s + i]);
        } else {
            v.push_back(tok(rng));
        }
    }
    return v;
}

void incremental_tests() {
    std::mt19937 rng(46);
    for (int round = 0; round < 20; ++round) {
        const size_t cap = round % 2 ? 4096 : 1024;   // 1024: the table fills past its nominal capacity
        const std::vector<int32_t> conv = text_like(rng, 3000, 40 + round), cont = text_like(rng, 300, 40 + round);
        // a conversation read in growing requests: every request continues the previous one
        SuffixDrafter kept(3, 32, cap);
        bool ok = true;
        for (size_t end : {size_t(0), size_t(1), size_t(700), size_t(701), size_t(1900), conv.size()}) {
            const bool extended = kept.sync(conv.data(), end);
            ok = ok && extended;
        }
        check(ok, "sync keeps a history that the request continues");
        SuffixDrafter fresh(3, 32, cap);
        fresh.append(conv.data(), conv.size());
        check(same_proposals(kept, fresh, cont), "a history extended by sync proposes as a fresh one");
        // a request that does not continue it (an edited message): rebuilt, lazily cleared
        std::vector<int32_t> edited(conv.begin(), conv.begin() + 1200);
        edited.push_back(-7);
        edited.insert(edited.end(), conv.begin() + 900, conv.begin() + 2500);
        check(!kept.sync(edited.data(), edited.size()), "sync rebuilds a history the request does not continue");
        SuffixDrafter fresh2(3, 32, cap);
        fresh2.append(edited.data(), edited.size());
        check(same_proposals(kept, fresh2, cont), "a rebuilt history proposes as a fresh one");
        // reset then the same text again: every trigram is already in the (stale) table
        kept.reset();
        kept.append(conv.data(), 1500);
        SuffixDrafter fresh3(3, 32, cap);
        fresh3.append(conv.data(), 1500);
        check(same_proposals(kept, fresh3, cont), "after a lazy reset the old slots are gone");
        // many resets in a row, then a short history (most of the table is stale)
        for (int i = 0; i < 1000; ++i) kept.reset();
        kept.sync(cont.data(), 40);
        SuffixDrafter fresh4(3, 32, cap);
        fresh4.append(cont.data(), 40);
        check(same_proposals(kept, fresh4, conv), "after a thousand resets");
    }
    {   // a small table filled in every epoch: a slot of an older epoch must be free again, or the table stays full
        SuffixDrafter kept(3, 32, 64);   // 128 slots
        std::uniform_int_distribution<int32_t> tok(0, 1 << 20);
        bool ok = true;
        for (int epoch = 0; epoch < 8 && ok; ++epoch) {
            std::vector<int32_t> h;
            for (int i = 0; i < 100; ++i) h.push_back(tok(rng));   // ~98 distinct trigrams
            std::vector<int32_t> cont(h.begin() + 30, h.begin() + 60);
            kept.reset();
            kept.append(h.data(), h.size());
            SuffixDrafter fresh(3, 32, 64);
            fresh.append(h.data(), h.size());
            ok = same_proposals(kept, fresh, cont);
            int32_t out[4];
            kept.append(h[30]);
            kept.append(h[31]);
            kept.append(h[32]);
            ok = ok && kept.propose(4, out) > 0;   // the repeated passage is still found
        }
        check(ok, "a table refilled after every reset");
    }
}

void unit_tests() {
    {   // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    {   // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // bounded memory: appending beyond the nominal capacity does not crash and still works
        SuffixDrafter d(3, 32, 1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 5000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    incremental_tests();
    std::printf("suffix_drafter unit tests: %s\n", g_fail ? "FAILED" : "OK");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 8, first = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); first = 4; }
        std::printf("%-28s %8s | %-44s | %-44s\n", "prompt", "tokens", "continue: tok/step  proposals  acc/proposal",
                    "copy: tok/step  proposals  acc/proposal");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2;
            const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K);
            const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
            const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K);
            const char* name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                        (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                        c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                        100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
        }
        return 0;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}
