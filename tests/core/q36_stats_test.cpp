// tests/core/q36_stats_test.cpp - strata-q36's STRATA_Q36_STATS helpers (src/qwen35/q36_stats.hpp), no GPU: the expert union
// over windows of 2, 3 and 4 tokens, the difference of two snapshots, the GB/s arithmetic and the lines' content.
#include "check.hpp"
#include "q36_stats.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace {
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    using namespace strata::q36;
    std::printf("expert union:\n");
    {
        ExpertUnion u;
        u.reset(2, 2);
        const int32_t a[2] = {1, 2}, b[2] = {2, 3}, c[2] = {4, 5}, d[2] = {1, 2};
        u.add(0, a);
        check(u.mean(2) == 0.0 && u.layer_mean(0, 2) == 0.0, "one token makes no window yet");
        u.add(0, b);   // w2: {1,2,2,3} -> 3 distinct of 4
        check(near(u.layer_mean(0, 2), 0.75) && u.layer_mean(0, 3) == 0.0, "two tokens: one window of 2, 3 of 4 distinct");
        u.add(0, c);   // w2: {2,3,4,5} -> 1.0 (mean of .75 and 1) ; w3: {1,2,2,3,4,5} -> 5 of 6
        check(near(u.layer_mean(0, 2), 0.875), "the mean runs over every window that ended on a token");
        check(near(u.layer_mean(0, 3), 5.0 / 6.0), "a window of 3 counts its distinct experts over 3 * top-k");
        u.add(0, d);   // w4: {1,2,2,3,4,5,1,2} -> 5 of 8 ; w3: {2,3,4,5,1,2} -> 5 of 6
        check(near(u.layer_mean(0, 4), 5.0 / 8.0), "a window of 4 over repeated experts");
        check(near(u.layer_mean(0, 3), 5.0 / 6.0), "the ring keeps only the last tokens: the first one left the window of 3");
        check(u.layer_mean(1, 2) == 0.0 && near(u.mean(2), u.layer_mean(0, 2)), "a layer with no token is left out of the mean");
        u.add(7, a);
        check(u.layers() == 2, "a layer past the end is ignored");
        u.reset(2, 2);
        check(u.mean(2) == 0.0 && u.mean(4) == 0.0, "reset forgets the request");
    }
    {
        ExpertUnion u;
        u.reset(1, 3);
        const int32_t same[3] = {5, 6, 7};
        for (int i = 0; i < 6; ++i) u.add(0, same);
        check(near(u.mean(2), 0.5) && near(u.mean(3), 1.0 / 3.0) && near(u.mean(4), 0.25), "the same experts every token: 1 / window");
    }
    std::printf("snapshots and lines:\n");
    {
        DecodeStats a, b;
        a.wait_ms = 30;
        a.serve_ms = 50;
        a.steps = 200;
        a.gpu_bytes = 900;
        a.pcie_bytes = 400;
        a.cpu_bytes = 8'000'000'000LL;
        a.pool_gu_ms = 700;
        a.pool_q_ms = 100;
        a.pool_down_ms = 300;
        a.pcie_demand = 250;
        a.pf_issued = 400;
        a.pf_hits = 300;
        a.wait_layer_ms = {10, 20};
        b.wait_ms = 10;
        b.serve_ms = 20;
        b.steps = 100;
        b.gpu_bytes = 100;
        b.pcie_bytes = 100;
        b.cpu_bytes = 2'000'000'000LL;
        b.pool_gu_ms = 200;
        b.pool_q_ms = 50;
        b.pool_down_ms = 100;
        b.pcie_demand = 50;
        b.pf_issued = 200;
        b.pf_hits = 120;
        b.wait_layer_ms = {4, 5};
        a.rb_swaps = 7;
        a.rb_blob_bytes = 3'500'000LL;
        a.rb_slot_bytes = 9'000LL;
        b.rb_swaps = 4;
        b.rb_blob_bytes = 1'000'000LL;
        b.rb_slot_bytes = 4'000LL;
        a.rb_ms = 20.0;
        b.rb_ms = 7.5;
        const DecodeStats d = a - b;
        check(near(d.wait_ms, 20) && near(d.serve_ms, 30) && d.steps == 100, "the difference of the waits and steps");
        check(d.gpu_bytes == 800 && d.pcie_bytes == 300 && d.cpu_bytes == 6'000'000'000LL, "the difference of the bytes");
        check(d.wait_layer_ms.size() == 2 && near(d.wait_layer_ms[0], 6) && near(d.wait_layer_ms[1], 15), "the wait by layer");
        check(near(gbps(6'000'000'000LL, 1000.0), 6.0) && gbps(1, 0.0) == 0.0, "GB/s of bytes over ms, 0 with no time");
        const std::string line = format_decode_stats(3, d, 100);
        check(has(line, "session 3: stats: pool 750 ms (gate/up 500, quantize 50, down 200)"), "the pool ms by phase");
        check(has(line, "6.00 GB read, 8.57 GB/s over the row phases (8.00 over all three)"), "the bytes read and the pool GB/s");
        check(has(line, "GPU 0.0 MB, PCIe 0.0 MB, CPU 60.0 MB (100 tokens)"), "the bytes per token by path");
        check(has(line, "PCIe blobs per token: on demand 2.00, prefetched 2.00, prefetch hits 1.80 (90.0% of the prefetched)"),
              "the PCIe blobs by token: on demand, prefetched, hits and the hit rate");
        check(has(line, "host wait on the doorbell 20 ms over 100 layer steps (200.0 us each)"), "the doorbell wait");
        check(d.rb_swaps == 3 && d.rb_blob_bytes == 2'500'000LL && d.rb_slot_bytes == 5'000LL && near(d.rb_ms, 12.5),
              "the difference of the rebalance counters");
        check(has(line, "decode rebalances: 3 slot swaps, 2.5 MB of expert blobs, 5.0 KB of slot tables, 12.5 ms of host time"),
              "the decode cadence's rebalances: swaps, blob bytes, slot table bytes and host time");
        check(has(format_decode_stats(0, DecodeStats(), 0), "(0 tokens)"), "no token, no division by zero");
        check(!has(line, "swiglu skim"), "the skim's part is absent from the pool line with the skim off");
        DecodeStats k = d;
        k.skim_on = true;
        k.skim_seen = 400;
        k.skim_cut = 100;
        check(has(format_decode_stats(3, k, 100), "5.0 KB of slot tables, 12.5 ms of host time, swiglu skim 25.0 % of 400 covered neurons"),
              "the skim on: the cut share and the neurons seen, at the end of the pool line");
        check(has(format_wait_layers(3, d), "by layer (ms): 6.0 15.0"), "the wait by layer line");
        std::string err;
        check(swiglu_skim_tau_from_env(nullptr, &err) == 0.f && err.empty(), "STRATA_Q36_SWIGLU_SKIM unset: off, no line");
        check(swiglu_skim_tau_from_env("0.05", &err) == 0.05f && err.empty(), "a threshold is read as given");
        check(swiglu_skim_tau_from_env("0", &err) == 0.f && err.empty(), "0 is accepted, and is off");
        for (const char* bad : {"abc", "0.05x", "", "-0.1", "inf", "nan"}) {
            err.clear();
            check(swiglu_skim_tau_from_env(bad, &err) == 0.f && !err.empty(), "a value that is not a finite number >= 0 is refused");
        }
    }
    {
        ExpertUnion u;
        u.reset(2, 1);
        const int32_t x[1] = {1}, y[1] = {2};
        u.add(0, x);
        u.add(0, y);
        u.add(1, x);
        u.add(1, x);
        const std::string s = format_union(1, u);
        check(has(s, "expert union") && has(s, " w2 0.750 w3 0.000 w4 0.000"), "the union line: the mean per window");
        check(has(s, "per layer w2: 1.00 0.50"), "the union line: every layer");
    }
    std::printf("one decoding session:\n");
    {
        check(single_decoder(true, 4, 4), "no other session at the start and none mixed in: the stats are printed");
        check(!single_decoder(false, 4, 4), "another session decoding at the start: no line");
        check(!single_decoder(true, 4, 5), "a window of two sessions since the start: no line");
        check(!single_decoder(true, 4, 3), "a counter that went back (a new run) is not the start's count: no line");
    }
    std::printf("the switch:\n");
    {
        check(stats_flag("1"), "STRATA_Q36_STATS=1 is on");
        check(!stats_flag(nullptr) && !stats_flag("") && !stats_flag("0") && !stats_flag("11") && !stats_flag("true"),
              "no variable, empty, 0, 11 and true are off");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
