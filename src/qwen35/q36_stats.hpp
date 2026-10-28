// src/qwen35/q36_stats.hpp - STRATA_Q36_STATS=1: the per-request measurements of the hybrid decode (host side, header only).
//
// What one request costs in bytes and in waiting, from counters the engine already keeps (the pool's phase timers and byte
// count, the hybrid entry totals) plus four small additions: the doorbell wait of the host, the bytes each path reads, the
// expert union over 2, 3 and 4 consecutive tokens, and the pool, wait, host and PCIe time of each decode step.  Nothing here
// runs unless the switch is set; the engine then reads the clock around the waits it already has and adds no synchronisation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
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

/// STRATA_Q36_SWIGLU_SKIM=<tau> (#110): the SwiGLU skim threshold of the CPU expert rows, a finite number >= 0 with nothing
/// after it.  Unset gives 0 (off).  Any other value gives 0 as well, and `err` (when given) gets the line the engine prints
/// for it; an accepted value leaves `err` alone.
inline float swiglu_skim_tau_from_env(const char* v, std::string* err) {
    if (v == nullptr) return 0.f;
    char* end = nullptr;
    const float t = std::strtof(v, &end);
    if (end != v && *end == '\0' && t >= 0.f && t < HUGE_VALF) return t;
    if (err != nullptr) *err = std::string("STRATA_Q36_SWIGLU_SKIM=") + v + " is not a finite number >= 0: the skim is off";
    return 0.f;
}

/// STRATA_Q36_SWIGLU_SKIM, read once; a rejected value is printed once and gives 0
inline float swiglu_skim_tau() {
    static const float tau = [] {
        std::string err;
        const float t = swiglu_skim_tau_from_env(std::getenv("STRATA_Q36_SWIGLU_SKIM"), &err);
        if (!err.empty()) std::fprintf(stderr, "strata: %s\n", err.c_str());
        return t;
    }();
    return tau;
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
    // the rebalances of the decode cadence (every rebalance_every positions or batch windows), not the one after a prompt
    long long rb_swaps = 0;      // slot replacements they applied
    long long rb_blob_bytes = 0; // bytes they copied as expert blobs (host to VRAM slots)
    long long rb_slot_bytes = 0; // bytes they uploaded as slot tables (slot_of), one per changed layer
    double rb_ms = 0;            // their host time: the decode step's stream sync before them (after_rows) and their own included
    // the SwiGLU skim (#110, STRATA_Q36_SWIGLU_SKIM): the pool's neurons seen and cut on the paths that apply it; shown only when on
    bool skim_on = false;
    long long skim_seen = 0, skim_cut = 0;
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
    d.rb_swaps = a.rb_swaps - b.rb_swaps;
    d.rb_blob_bytes = a.rb_blob_bytes - b.rb_blob_bytes;
    d.rb_slot_bytes = a.rb_slot_bytes - b.rb_slot_bytes;
    d.rb_ms = a.rb_ms - b.rb_ms;
    d.skim_on = a.skim_on;
    d.skim_seen = a.skim_seen - b.skim_seen;
    d.skim_cut = a.skim_cut - b.skim_cut;
    return d;
}

/// One decode step of a request (one forward of the decode loop: a token, or a verify window of `rows` tokens): the pool's
/// row phases, the host's wait on the doorbell, the host's own time (the step's wall time less those two: serving the
/// layers outside the pool, the PCIe copies' sync, the launches, the sampling, a rebalance) and the copy streams' busy time
/// on the step's PCIe copies (it overlaps the other three).
struct StepStats {
    double pool_ms = 0, wait_ms = 0, host_ms = 0, pcie_ms = 0;
    int rows = 1;
};

inline constexpr size_t kStepLogMax = 4096;   // the steps a request's line lists; the ones after are only counted

/// The decode steps of one request, for the line that tells which stage made a slow token slow.  The engine opens a step
/// when the decode loop's forward has ended (after_rows), adds the busy time of the step's PCIe copies as their events
/// complete (in that step or a later one: each copy carries the id of the step that issued it), and closes the step with
/// the pool and doorbell counters and the wall time since the last close (since the decode's start for the first step).
class StepLog {
public:
    /// The request's decode starts, at these counters: the log empties, and the copies tagged before (the prompt's) are
    /// not its own.
    void begin(double pool_ms, double wait_ms) {
        base_ = ++id_;
        steps_.clear();
        steps_.reserve(kStepLogMax);
        pool0_ = pool_ms;
        wait0_ = wait_ms;
        on_ = true;
    }
    /// The request's decode has ended: no step opens or closes until the next begin.
    void stop() { on_ = false; }
    /// The id of the step running now, for the copies it issues.
    long long id() const { return id_; }
    /// A forward of `rows` rows has ended: its entry, while the log has room.
    void open(int rows) {
        if (!on_ || steps_.size() >= kStepLogMax) return;
        steps_.emplace_back();
        steps_.back().rows = rows;
    }
    /// `ms` of copy-stream busy time on the copies of step `id` (ignored when that step is not listed).
    void add_pcie(long long id, double ms) {
        if (id >= base_ && (size_t) (id - base_) < steps_.size()) steps_[(size_t) (id - base_)].pcie_ms += ms;
    }
    /// The step ends at these counters, `wall_ms` after the last close.
    void close(double pool_ms, double wait_ms, double wall_ms) {
        if (!on_) return;
        const size_t at = (size_t) (id_ - base_);
        if (at < steps_.size()) {
            StepStats& s = steps_[at];
            s.pool_ms = pool_ms - pool0_;
            s.wait_ms = wait_ms - wait0_;
            s.host_ms = wall_ms - s.pool_ms - s.wait_ms;
        }
        pool0_ = pool_ms;
        wait0_ = wait_ms;
        ++id_;
    }
    const std::vector<StepStats>& steps() const { return steps_; }
    /// The steps closed since begin, the ones past kStepLogMax included.
    long long total() const { return id_ - base_; }

private:
    std::vector<StepStats> steps_;
    long long id_ = 0, base_ = 0;
    double pool0_ = 0, wait0_ = 0;
    bool on_ = false;
};

/// The decode steps line: per listed step, in order, pool / wait / host / PCIe copies in ms, then / rows when the step kept
/// more than one row (a verify window).  One line per request, whatever its length (kStepLogMax steps at most).
inline std::string format_steps(int session, const StepLog& log) {
    char b[160];
    std::snprintf(b, sizeof b, "strata serve: session %d: stats: decode steps (pool/wait/host/PCIe copies ms[/rows]), "
                  "%lld steps, %zu listed:", session, log.total(), log.steps().size());
    std::string s = b;
    for (const StepStats& t : log.steps()) {
        std::snprintf(b, sizeof b, " %.1f/%.1f/%.1f/%.1f", t.pool_ms, t.wait_ms, t.host_ms, t.pcie_ms);
        s += b;
        if (t.rows != 1) s += "/" + std::to_string(t.rows);
    }
    return s;
}

/// GB/s of `bytes` over `ms` (0 with no time).
inline double gbps(long long bytes, double ms) { return ms > 0 ? (double) bytes / 1e9 / (ms / 1e3) : 0.0; }

/// The pool, bytes and doorbell line of a request that produced `tokens` tokens, then the PCIe blobs by token: copied on
/// demand, prefetched ahead, and the picks the prefetch served (the hit rate is over the prefetched blobs), and the decode
/// rebalances' bytes as expert blobs and as slot tables (the share of picks each path served is the request's own line).
inline std::string format_decode_stats(int session, const DecodeStats& d, long long tokens) {
    const double t = tokens > 0 ? (double) tokens : 1.0;
    char b[1280];
    std::snprintf(b, sizeof b,
                  "strata serve: session %d: stats: pool %.0f ms (gate/up %.0f, quantize %.0f, down %.0f), %.2f GB read, "
                  "%.2f GB/s over the row phases (%.2f over all three); bytes per token: GPU %.1f MB, PCIe %.1f MB, CPU %.1f MB "
                  "(%lld tokens); host wait on the doorbell %.0f ms over %lld layer steps (%.1f us each), host serving %.0f ms "
                  "(%.1f us each); PCIe blobs per token: on demand %.2f, prefetched %.2f, prefetch hits %.2f (%.1f%% of the prefetched); "
                  "decode rebalances: %lld slot swaps, %.1f MB of expert blobs, %.1f KB of slot tables, %.1f ms of host time",
                  session, d.pool_gu_ms + d.pool_q_ms + d.pool_down_ms, d.pool_gu_ms, d.pool_q_ms, d.pool_down_ms,
                  (double) d.cpu_bytes / 1e9, gbps(d.cpu_bytes, d.pool_gu_ms + d.pool_down_ms),
                  gbps(d.cpu_bytes, d.pool_gu_ms + d.pool_q_ms + d.pool_down_ms), (double) d.gpu_bytes / 1e6 / t,
                  (double) d.pcie_bytes / 1e6 / t, (double) d.cpu_bytes / 1e6 / t, tokens, d.wait_ms, d.steps,
                  d.steps ? 1e3 * d.wait_ms / (double) d.steps : 0.0, d.serve_ms,
                  d.steps ? 1e3 * d.serve_ms / (double) d.steps : 0.0, (double) d.pcie_demand / t, (double) d.pf_issued / t,
                  (double) d.pf_hits / t, d.pf_issued ? 100.0 * (double) d.pf_hits / (double) d.pf_issued : 0.0, d.rb_swaps,
                  (double) d.rb_blob_bytes / 1e6, (double) d.rb_slot_bytes / 1e3, d.rb_ms);
    std::string s = b;
    if (d.skim_on) {
        std::snprintf(b, sizeof b, ", swiglu skim %.1f %% of %lld covered neurons",
                      d.skim_seen ? 100.0 * (double) d.skim_cut / (double) d.skim_seen : 0.0, d.skim_seen);
        s += b;
    }
    return s;
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
