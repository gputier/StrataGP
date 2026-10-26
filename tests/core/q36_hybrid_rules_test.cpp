// tests/core/q36_hybrid_rules_test.cpp - the pure rules of strata-q36's hybrid mode, no GPU: the SER keep rule, the Kmin of a
// layer and the cache-prior bias of the routing kernel (src/qwen35/q36_rules.hpp, the functions the kernel calls), the guard that turns
// the graphs off below the verify threshold, and the placement change of a rebalance swap (q36_rebalance.hpp).
#include "check.hpp"
#include "q36_rebalance.hpp"
#include "q36_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace strata::q36;
    // every value below is exact in binary, so the comparisons are exact
    std::printf("ser_keep:\n");
    {
        // top probability 0.5, ser_t 0.25: a pick past the first ser_kmin is kept when p > 0.125
        check(ser_keep(0, 1, 0.0f, 0.25f, 0.5f), "rank 0 below ser_kmin 1 is kept whatever its probability");
        check(ser_keep(1, 1, 0.25f, 0.25f, 0.5f), "rank 1 with p 0.25 (above 0.125) is kept");
        check(!ser_keep(1, 1, 0.125f, 0.25f, 0.5f), "rank 1 with p exactly 0.125 is dropped (the test is strict)");
        check(!ser_keep(2, 1, 0.1f, 0.25f, 0.5f) && !ser_keep(2, 1, 0.0f, 0.25f, 0.5f),
              "rank 2 with p below the threshold is dropped");
        check(ser_keep(2, 3, 0.0f, 0.25f, 0.5f) && !ser_keep(3, 3, 0.0f, 0.25f, 0.5f),
              "ser_kmin 3 keeps ranks 0 .. 2 and applies the threshold from rank 3");
        check(ser_keep(4, 0, 0.5f, 0.25f, 0.5f) && !ser_keep(4, 0, 0.1f, 0.25f, 0.5f),
              "ser_kmin 0: no rank is kept outright, rank 4 is kept at p 0.5 and dropped at p 0.1");
    }
    std::printf("cache_prior_bias:\n");
    {
        check(cache_prior_bias(true, true, 0.5f, 4.0f) == 2.0f,
              "a resident expert in a hybrid layer gets cache_prior * Delta_avg");
        check(cache_prior_bias(true, false, 0.5f, 4.0f) == 0.0f, "an expert outside the slots gets no bias");
        check(cache_prior_bias(false, true, 0.5f, 4.0f) == 0.0f, "a layer that is not hybrid gets no bias");
        check(cache_prior_bias(true, true, 0.0f, 4.0f) == 0.0f, "cache_prior 0 is no bias");
        check(cache_prior_bias(true, true, -1.0f, 4.0f) == 0.0f, "a negative cache_prior is no bias");
        check(cache_prior_bias(true, true, 0.5f, -4.0f) == -2.0f, "the bias keeps the sign of Delta_avg (no flip)");
        // cache_prior 0 (or -0) is no bias whatever Delta_avg holds: a test of cache_prior >= 0 would give 0 * inf or 0 * NaN,
        // a NaN; with a finite Delta_avg 0 * Delta_avg is 0 either way, so only a Delta_avg that is not finite tells them apart
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        check(cache_prior_bias(true, true, 0.0f, inf) == 0.0f && cache_prior_bias(true, true, 0.0f, -inf) == 0.0f,
              "cache_prior 0 with an infinite Delta_avg is still no bias (0, not NaN)");
        check(cache_prior_bias(true, true, 0.0f, nan) == 0.0f && cache_prior_bias(true, true, -0.0f, nan) == 0.0f,
              "cache_prior 0 or -0 with a NaN Delta_avg is still no bias (0, not NaN)");
        check(std::isinf(cache_prior_bias(true, true, 0.5f, inf)), "a positive cache_prior passes an infinite Delta_avg on");
        // the ranking the kernel makes: logits 1.0 (expert 0, resident) and 1.5 (expert 1); the bias of 1.0 lifts 0 past 1
        const std::vector<float> logit = {1.0f, 1.5f};
        const std::vector<bool> resident = {true, false};
        const int plain = logit[0] > logit[1] ? 0 : 1;
        const float key0 = logit[0] + cache_prior_bias(true, resident[0], 1.0f, 1.0f);
        const float key1 = logit[1] + cache_prior_bias(true, resident[1], 1.0f, 1.0f);
        const int biased = key0 > key1 ? 0 : 1;
        check(plain == 1 && biased == 0,
              "with the bias the resident expert 0 (2.0) ranks above expert 1 (1.5), without it expert 1");
    }
    std::printf("hybrid_graph_off:\n");
    {
        // the engine passes Engine::kVerify (8) as verify_rows: a hybrid window below 8 rows turns the graphs off
        check(hybrid_graph_off(true, 7, 8), "hybrid_t 7 under the verify threshold 8: the graphs are off");
        check(!hybrid_graph_off(true, 8, 8), "hybrid_t 8, exactly the threshold: the graphs stay on");
        check(!hybrid_graph_off(true, 9, 8), "hybrid_t 9 above the threshold: the graphs stay on");
        check(!hybrid_graph_off(false, 7, 8), "no hybrid main layers: the guard does not turn the graphs off");
    }
    std::printf("apply_slot_swap:\n");
    {
        // experts 0 and 1 in slots 0 and 1; expert 2 takes the slot of expert 1: the placement and the copy it asks for
        using Pair = std::pair<int, int>;
        std::vector<int> slot_of = {0, 1, -1, -1};
        std::vector<Pair> copies;
        const int s = apply_slot_swap(slot_of, {0, 2, 1, 2.0}, [&](int slot, int in) { copies.emplace_back(slot, in); });
        check(s == 1, "the swap returns the slot the expert leaves (1)");
        check(slot_of == std::vector<int>({0, -1, 1, -1}), "expert 1 leaves its slot, expert 2 holds it, the others are kept");
        check(copies == std::vector<Pair>({{1, 2}}), "one copy: expert 2 into slot 1 (not expert 1 into the slot)");
        // then expert 3 takes slot 0 from expert 0: still two slots, each held once
        const int s2 = apply_slot_swap(slot_of, {0, 3, 0, 1.0}, [&](int slot, int in) { copies.emplace_back(slot, in); });
        check(s2 == 0 && slot_of == std::vector<int>({-1, -1, 1, 0}), "a second swap moves slot 0 to expert 3");
        check(copies == std::vector<Pair>({{1, 2}, {0, 3}}), "and copies expert 3 into slot 0");
        std::vector<int> held;
        for (int v : slot_of)
            if (v >= 0) held.push_back(v);
        std::sort(held.begin(), held.end());
        check(held == std::vector<int>({0, 1}), "after both swaps the slots 0 and 1 are each held by exactly one expert");
        // the order of the two assignments: `out` loses its slot first, then `in` takes it.  They differ only when in == out
        // (choose_swaps never makes such a swap: `in` has no slot, `out` has one), where the expert must keep its slot
        std::vector<int> self = {0, 1, -1, -1};
        std::vector<Pair> self_copies;
        const int s3 = apply_slot_swap(self, {0, 1, 1, 0.0}, [&](int slot, int in) { self_copies.emplace_back(slot, in); });
        check(s3 == 1 && self == std::vector<int>({0, 1, -1, -1}), "a swap of an expert with itself keeps it in its slot (1)");
        check(self_copies == std::vector<Pair>({{1, 1}}), "and copies it into that slot again");
    }
    std::printf("hybrid_prompt_chunk:\n");
    check(hybrid_prompt_chunk(23.2, 1, true) == 4096, "23.2 GiB free, one session: 4096");
    check(hybrid_prompt_chunk(20.0, 1, true) == 4096, "20.0 GiB free, one session: 4096");
    check(hybrid_prompt_chunk(19.99, 1, true) == 2048, "19.99 GiB free, one session: 2048");
    check(hybrid_prompt_chunk(23.2, 2, true) == 2048, "23.2 GiB free, two sessions: 2048");
    check(hybrid_prompt_chunk(23.2, 0, true) == 4096, "23.2 GiB free, 0 sessions (one at a time): 4096");
    check(hybrid_prompt_chunk(7.5, 1, true) == 2048, "7.5 GiB free: 2048");
    check(hybrid_prompt_chunk(7.49, 1, true) == 1024, "7.49 GiB free: 1024");
    check(hybrid_prompt_chunk(5.5, 1, true) == 1024, "5.5 GiB free: 1024");
    check(hybrid_prompt_chunk(5.49, 1, true) == 512, "5.49 GiB free: 512");
    check(hybrid_prompt_chunk(23.2, 1, false) == 2048, "23.2 GiB free, one session, slots by hand: 2048");
    check(hybrid_prompt_chunk(20.0, 1, false) == 2048, "20.0 GiB free, one session, slots by hand: 2048");
    check(hybrid_prompt_chunk(5.49, 1, false) == 512, "5.49 GiB free, slots by hand: 512");
    std::printf("forced_chunk_need:\n");
    {
        const double GiB = (double) (1ull << 30);
        check(forced_chunk_need(8192, 0.0) == 6.5 * GiB, "8192 in hybrid mode: 2.75 GiB + 3.75 GiB of buffers");
        check(forced_chunk_need(4608, 0.0) == 4.75 * GiB, "4608 in hybrid mode: 2.75 GiB + 2 GiB");
        check(forced_chunk_need(8192, 18.0 * GiB) == 24.5 * GiB, "8192 with 18 GiB of experts all in VRAM: they keep their room");
        check(forced_chunk_need(8192, 18.0 * GiB) > 22.0 * GiB, "so 22 GiB free refuses 8192 when every expert must fit");
        check(forced_chunk_need(8192, 18.0 * GiB + 1.5 * GiB) == 26.0 * GiB, "the sessions' K / V past 8192 cells keep their room too");
        check(forced_chunk_need(4096, 0.0) == 4.5 * GiB, "4096, the auto cap: 2.75 GiB + 1.75 GiB");
    }
    std::printf("parse_ser_table:\n");
    {
        // 48 main layers, --ser's Kmin 7: a table line names a layer's own Kmin, the others keep 7
        std::vector<int> kmin;
        std::string err;
        const bool ok = parse_ser_table("# SER table\n\n5 6\n  12\t8   # all picks\r\n47 1\n", 48, 7, &kmin, &err);
        check(ok && err.empty(), "a table with comments, blank lines, tabs and CRs is read");
        check(ok && kmin.size() == 48 && kmin[5] == 6 && kmin[12] == 8 && kmin[47] == 1 && kmin[0] == 7 && kmin[46] == 7,
              "the named layers take their Kmin, the others --ser's Kmin 7");
        std::vector<int> none;
        check(parse_ser_table("", 4, 6, &none, &err) && none == std::vector<int>({6, 6, 6, 6}),
              "an empty table gives every layer --ser's Kmin");
        std::vector<int> two;
        check(parse_ser_table("0 5\n3 4\n", 4, 8, &two, &err) && two == std::vector<int>({5, 8, 8, 4}),
              "layers 0 and 3 are named, layers 1 and 2 take the default Kmin 8");
    }
    {
        // each refusal names its line: a bad line is never skipped
        std::vector<int> kmin;
        std::string err;
        const auto refused = [&](const char* text, int n_layers, const char* line) {
            err.clear();
            const bool ok = parse_ser_table(text, n_layers, 8, &kmin, &err);
            return !ok && err.find(line) != std::string::npos;
        };
        check(refused("5\n", 48, "line 1"), "a line with one number is refused");
        check(refused("5 6 7\n", 48, "line 1"), "a line with three numbers is refused");
        check(refused("5 6x\n", 48, "line 1"), "a Kmin with text after it is refused");
        check(refused("5 x\n", 48, "line 1"), "a non-number is refused");
        check(refused("1.5 6\n", 48, "line 1"), "a layer that is not an integer is refused");
        check(refused("3 6\n\n# c\n47 9\n", 48, "line 4"), "Kmin 9 is refused, the error names line 4");
        check(refused("0 0\n", 48, "line 1"), "Kmin 0 is refused");
        check(parse_ser_table("0 1\n", 48, 8, &kmin, &err), "Kmin 1 is accepted");
        check(refused("48 6\n", 48, "line 1"), "layer 48 is out of 0 .. 47 and refused");
        check(refused("-1 6\n", 48, "line 1"), "layer -1 is refused");
        check(parse_ser_table("47 8\n", 48, 8, &kmin, &err), "layer 47 of 48 is accepted");
        check(refused("2 6\n2 7\n", 48, "line 2"), "a layer named twice is refused, the error names line 2");
    }
    std::printf("layer_kmin:\n");
    {
        // 48 main layers, --ser's Kmin 7, the table names layer 5 (Kmin 6) and layer 47 (Kmin 1)
        std::vector<int> kmin;
        std::string err;
        parse_ser_table("5 6\n47 1\n", 48, 7, &kmin, &err);
        check(layer_kmin(kmin, 5, 7) == 6, "a layer the table names takes its own Kmin");
        check(layer_kmin(kmin, 47, 7) == 1, "the last main layer takes its own Kmin, 1");
        check(layer_kmin(kmin, 4, 7) == 7, "a layer the table does not name takes --ser's Kmin");
        check(layer_kmin(kmin, 48, 7) == 7 && layer_kmin(kmin, 49, 7) == 7,
              "the MTP block, past the main layers, takes --ser's Kmin whatever the table holds at 47");
        std::vector<int> none;
        check(parse_ser_table("", 48, 6, &none, &err) && layer_kmin(none, 0, 6) == 6 && layer_kmin(none, 48, 6) == 6,
              "an empty table gives every layer, the MTP block too, --ser's Kmin");
        check(layer_kmin(std::vector<int>(), 3, 5) == 5, "a layer list with no entries gives --ser's Kmin");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
