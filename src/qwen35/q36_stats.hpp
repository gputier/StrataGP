// src/qwen35/q36_stats.hpp - STRATA_Q36_STATS=1: the per-request measurements of the hybrid decode (host side, header only).
//
// What one request costs in bytes and in waiting, from counters the engine already keeps (the pool's phase timers and byte
// count, the hybrid entry totals) plus three small additions: the doorbell wait of the host, the bytes each path reads, and
// the expert union over 2, 3 and 4 consecutive tokens.  Nothing here runs unless the switch is set; the engine then reads
// the clock around the waits it already has and adds no synchronisation.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::q36 {

/// The switch's value: "1" is on, anything else (or no variable) is off.
inline bool stats_flag(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

/// STRATA_Q36_STATS=1
inline bool stats_enabled() {
    static const bool on = stats_flag(std::getenv("STRATA_Q36_STATS"));
    return on;
}

/// Whether a request's stats describe one session's decode: no other session decoded at its start (`alone`), and no decode
/// window carried rows of two or more sessions since (`mixed_at_start` and `mixed_now` count those windows and the starts
/// made beside a decoding session).  Otherwise the expert union and the counters mix sessions and no line is printed.
inline bool single_decoder(bool alone, long long mixed_at_start, long long mixed_now) {
    return alone && mixed_now == mixed_at_start;
}

inline constexpr int kUnionMinWindow = 2, kUnionMaxWindow = 4;

/// How many distinct experts a layer asks for over `w` consecutive tokens, as a share of the w * top_k asks: 1.0 when no
/// expert repeats, 1/w when every token picks the same ones.  Fed one token's routing at a time, per layer; the mean is
/// over every window of the run that ended on a token, so a request's first tokens only enter the larger windows once
/// they have enough predecessors.
class ExpertUnion {
public:
    void reset(size_t layers, int top_k) {
        k_ = top_k;
        ring_.assign(layers, std::vector<int32_t>((size_t) kUnionMaxWindow * (size_t) top_k, -1));
        seen_.assign(layers, 0);
        sum_.assign(layers, {});
        cnt_.assign(layers, {});
    }
    /// One token's routing (top_k expert ids) in `layer`, in decode order.
    void add(size_t layer, const int32_t* ids) {
        if (layer >= ring_.size()) return;
        std::vector<int32_t>& r = ring_[layer];
        const long long n = seen_[layer]++;
        std::copy(ids, ids + k_, r.begin() + (size_t) (n % kUnionMaxWindow) * (size_t) k_);
        for (int w = kUnionMinWindow; w <= kUnionMaxWindow && w <= n + 1; ++w) {
            std::vector<int32_t> u;
            u.reserve((size_t) w * (size_t) k_);
            for (int back = 0; back < w; ++back) {
                const size_t at = (size_t) ((n - back) % kUnionMaxWindow) * (size_t) k_;
                u.insert(u.end(), r.begin() + (std::ptrdiff_t) at, r.begin() + (std::ptrdiff_t) at + k_);
            }
            std::sort(u.begin(), u.end());
            const size_t distinct = (size_t) (std::unique(u.begin(), u.end()) - u.begin());
            sum_[layer][(size_t) w] += (double) distinct / (double) ((size_t) w * (size_t) k_);
            ++cnt_[layer][(size_t) w];
        }
    }
    size_t layers() const { return ring_.size(); }
    /// The mean union share of the windows of `w` tokens in `layer` (0 with no window yet).
    double layer_mean(size_t layer, int w) const {
        return layer < cnt_.size() && cnt_[layer][(size_t) w] > 0 ? sum_[layer][(size_t) w] / (double) cnt_[layer][(size_t) w] : 0.0;
    }
    /// The same over the layers that saw a window of `w` tokens.
    double mean(int w) const {
        double s = 0;
        size_t n = 0;
        for (size_t l = 0; l < cnt_.size(); ++l)
            if (cnt_[l][(size_t) w] > 0) {
                s += layer_mean(l, w);
                ++n;
            }
        return n ? s / (double) n : 0.0;
    }

private:
    int k_ = 0;
    std::vector<std::vector<int32_t>> ring_;
    std::vector<long long> seen_;
    // per layer, indexed by the window (2..kUnionMaxWindow)
    std::vector<std::array<double, kUnionMaxWindow + 1>> sum_;
    std::vector<std::array<long long, kUnionMaxWindow + 1>> cnt_;
};

/// The accumulated side of a request's measurements; the engine snapshots it at the start of the decode and prints the
/// difference.  Bytes are what each path reads from memory: the resident experts' distinct blobs per layer step (GPU), the
/// blobs copied to the staging region (PCIe), the blobs the pool streams (CPU, the pool's own count).
struct DecodeStats {
    double wait_ms = 0;         // the host waiting for the GPU to ring the doorbell (the GPU is still computing)
    double serve_ms = 0;        // the host serving a layer: the GPU waits for it
    long long steps = 0;        // layer steps served
    std::vector<double> wait_layer_ms;   // wait_ms by layer
    long long gpu_bytes = 0, pcie_bytes = 0, cpu_bytes = 0;
    double pool_gu_ms = 0, pool_q_ms = 0, pool_down_ms = 0;   // the pool's gate/up rows, intermediate quantization, down rows
    long long pcie_demand = 0;   // PCIe blobs copied when their layer asked (the layer's picks the prefetch did not hold)
    long long pf_issued = 0;     // blobs the prefetch copied ahead (the next layer's predicted picks)
    long long pf_hits = 0;       // picks of a layer step served from the prefetch (no copy)
};

inline DecodeStats operator-(const DecodeStats& a, const DecodeStats& b) {
    DecodeStats d;
    d.wait_ms = a.wait_ms - b.wait_ms;
    d.serve_ms = a.serve_ms - b.serve_ms;
    d.steps = a.steps - b.steps;
    d.wait_layer_ms = a.wait_layer_ms;
    for (size_t i = 0; i < d.wait_layer_ms.size() && i < b.wait_layer_ms.size(); ++i) d.wait_layer_ms[i] -= b.wait_layer_ms[i];
    d.gpu_bytes = a.gpu_bytes - b.gpu_bytes;
    d.pcie_bytes = a.pcie_bytes - b.pcie_bytes;
    d.cpu_bytes = a.cpu_bytes - b.cpu_bytes;
    d.pool_gu_ms = a.pool_gu_ms - b.pool_gu_ms;
    d.pool_q_ms = a.pool_q_ms - b.pool_q_ms;
    d.pool_down_ms = a.pool_down_ms - b.pool_down_ms;
    d.pcie_demand = a.pcie_demand - b.pcie_demand;
    d.pf_issued = a.pf_issued - b.pf_issued;
    d.pf_hits = a.pf_hits - b.pf_hits;
    return d;
}

/// GB/s of `bytes` over `ms` (0 with no time).
inline double gbps(long long bytes, double ms) { return ms > 0 ? (double) bytes / 1e9 / (ms / 1e3) : 0.0; }

/// The pool, bytes and doorbell line of a request that produced `tokens` tokens, then the PCIe blobs by token: copied on
/// demand, prefetched ahead, and the picks the prefetch served (the hit rate is over the prefetched blobs).
inline std::string format_decode_stats(int session, const DecodeStats& d, long long tokens) {
    const double t = tokens > 0 ? (double) tokens : 1.0;
    char b[1280];
    std::snprintf(b, sizeof b,
                  "strata serve: session %d: stats: pool %.0f ms (gate/up %.0f, quantize %.0f, down %.0f), %.2f GB read, "
                  "%.2f GB/s over the row phases (%.2f over all three); bytes per token: GPU %.1f MB, PCIe %.1f MB, CPU %.1f MB "
                  "(%lld tokens); host wait on the doorbell %.0f ms over %lld layer steps (%.1f us each), host serving %.0f ms "
                  "(%.1f us each); PCIe blobs per token: on demand %.2f, prefetched %.2f, prefetch hits %.2f (%.1f%% of the prefetched)",
                  session, d.pool_gu_ms + d.pool_q_ms + d.pool_down_ms, d.pool_gu_ms, d.pool_q_ms, d.pool_down_ms,
                  (double) d.cpu_bytes / 1e9, gbps(d.cpu_bytes, d.pool_gu_ms + d.pool_down_ms),
                  gbps(d.cpu_bytes, d.pool_gu_ms + d.pool_q_ms + d.pool_down_ms), (double) d.gpu_bytes / 1e6 / t,
                  (double) d.pcie_bytes / 1e6 / t, (double) d.cpu_bytes / 1e6 / t, tokens, d.wait_ms, d.steps,
                  d.steps ? 1e3 * d.wait_ms / (double) d.steps : 0.0, d.serve_ms,
                  d.steps ? 1e3 * d.serve_ms / (double) d.steps : 0.0, (double) d.pcie_demand / t, (double) d.pf_issued / t,
                  (double) d.pf_hits / t, d.pf_issued ? 100.0 * (double) d.pf_hits / (double) d.pf_issued : 0.0);
    return b;
}

/// The doorbell wait of each layer, summed over the request.
inline std::string format_wait_layers(int session, const DecodeStats& d) {
    std::string s = "strata serve: session " + std::to_string(session) + ": stats: host wait on the doorbell by layer (ms):";
    char b[32];
    for (const double v : d.wait_layer_ms) {
        std::snprintf(b, sizeof b, " %.1f", v);
        s += b;
    }
    return s;
}

/// The expert union line: the mean over the layers, then every layer's mean, for the windows of 2, 3 and 4 tokens.
inline std::string format_union(int session, const ExpertUnion& u) {
    std::string s = "strata serve: session " + std::to_string(session) + ": stats: expert union (distinct / (window * top-k))";
    char b[64];
    for (int w = kUnionMinWindow; w <= kUnionMaxWindow; ++w) {
        std::snprintf(b, sizeof b, " w%d %.3f", w, u.mean(w));
        s += b;
    }
    for (int w = kUnionMinWindow; w <= kUnionMaxWindow; ++w) {
        s += "; per layer w" + std::to_string(w) + ":";
        for (size_t l = 0; l < u.layers(); ++l) {
            std::snprintf(b, sizeof b, " %.2f", u.layer_mean(l, w));
            s += b;
        }
    }
    return s;
}

}  // namespace strata::q36
