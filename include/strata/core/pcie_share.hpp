// include/strata/core/pcie_share.hpp - the PCIe share of the missed experts: the link probe, the share a link
// bandwidth gives, and the refinement from the CPU pool's speed.  Header only; shared by strata generate
// (Flash-Next) and the q36 engine (src/qwen35), which both split a layer's missed experts between the CPU pool
// and the GPU, the GPU's part copied over PCIe.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace strata::core {

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// #485: a single timed burst read an x16 PCIe 4 link (RTX A3000 laptop) at 18.5, 6.9 and 5.8 GB/s in three starts -
// a link still in a low-power state, other DMA, a context not yet up to speed - and the low reading set the share.
// Such a disturbance only ever slows a copy: nothing makes a correctly timed copy faster than the link carries.  So
// the same 1 GiB is now copied as four bursts of 256 MiB, timed one by one, and the fastest is the link's figure (the
// median would still follow a disturbance that lasts through half the bursts).  Each burst takes ~10 ms on an x16
// PCIe 4 link, so the probe takes no longer than the one 1 GiB burst did.  `samples`, when given, gets every
// burst's reading for the log.
inline double probe_pcie_h2d_gbps(std::string* samples = nullptr) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess) {
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    float ms[kBursts] = {};
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // Windows HIP: the events do not bracket the copies there (an RX 6800 read 3,300-26,000 GB/s, so every link kept
    // the x16 share), so each burst is timed on the host between two synchronizes: 256 MiB takes ~10 ms, so the
    // synchronize around it hardly matters
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    for (int b = 0; b < kBursts && ok; ++b) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
        ok = cudaDeviceSynchronize() == cudaSuccess;
        ms[b] = (float) std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
#else
    // the bursts run back to back on the stream, an event between each two
    cudaEvent_t ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts && cudaEventCreate(&ev[n_ev]) == cudaSuccess) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        cudaEventRecord(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
            cudaEventRecord(ev[b + 1]);
        }
        ok = cudaEventSynchronize(ev[kBursts]) == cudaSuccess;
        for (int b = 0; b < kBursts && ok; ++b) ok = cudaEventElapsedTime(&ms[b], ev[b], ev[b + 1]) == cudaSuccess;
    }
    for (int i = 0; i < n_ev; ++i) cudaEventDestroy(ev[i]);
#endif
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    if (!ok) (void) cudaGetLastError();
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share measured on x16 links) from
// 20 GB/s up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about
// what the x16 share costs.  Continuous (#485): before, 19.9 GB/s gave 0.42 and 20.0 the full 0.55 (and 4.0 GB/s
// gave 0.08, 3.9 none), so a reading near either edge moved the share by a quarter of its range.  From 20 GB/s up
// (an x16 PCIe 4/5 link: ~26-28 GB/s) the share is unchanged.
inline double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

// A share as the 256ths the expert dispatch carries (rounded, clamped to 0 .. 256).
inline int pcie_num_of(double frac) { return std::max(0, std::min(256, (int) (frac * 256.0 + 0.5))); }

// 0.1.42, CUDA single-GPU serve, no --pcie-frac: the share refined from what the CPU pool measures in the first decode
// windows.  The balance share of the missed experts is link / (link + CPU rate).  The link is the start-up probe; the CPU
// rate is the expert bytes over the pool's wall time per missed expert (ExpertDispatch::ms_run / multi_misses), which the
// engine already times.  The best shares measured with the route tail skip on (RTX 3060 0.15-0.20, Tesla P100 0.20, RTX
// A4000 flat from 0.15 to 0.40; the 0.1.41 rule gave 0.55, 0.29, 0.30) lie within one 0.05 step of link / (link + 1.85 x pool rate).
// Three updates at most (after 48, then 96 and 96 more windows), each rounded to 0.05, so a noisy window cannot move
// the share by less than a step and it stops moving after the third.  STRATA_PCIE_FRAC_DEFAULT=old keeps the 0.1.41 rule.
//
// The q36 engine feeds it the CPU path's own cost per job (the pool's row phases plus the job build, not the PCIe copies
// the same host loop launches) and the host's launch cost per PCIe blob, and sets pool_factor to 1: share = c / (c + w - l)
// for c the CPU path's us per job, w = blob / link the PCIe us per blob, l the host us the launch of a blob takes.  The host
// serves one layer as launches, job build, pool, so a launch is host time on the CPU path's side of the balance (the wire
// time w of the blobs runs beside it): n w = J c + n l, n the blobs chosen for PCIe.  Flash-Next passes no launch time and
// keeps pool_factor 1.85, and its arithmetic is the one above (k = 1.85 - 0.0).  An engine that times the copy streams
// passes the busy time, the launch time and the blobs the dispatch chose for PCIe as well: w and l are then per chosen
// blob (the demand copies, the prefetch traffic and the link's real rate together) and the start-up probe is only the
// figure before the first reading.
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    static constexpr double kMinShare = 0.10;   ///< Flash-Next's lowest share
    static constexpr double kMaxShare = 0.60;   ///< Flash-Next's highest share
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double min_share = kMinShare;  ///< the lowest share a refinement may set (an engine that can do without the share passes 0)
    double max_share = kMaxShare;  ///< the highest share a refinement may set (an engine whose CPU pool is slow passes more)
    double pool_factor = kPoolFactor;   ///< what multiplies the pool's rate against the link's (an engine that times only the CPU path's own work passes 1)
    double mark_ms = 0.0, mark_launch_ms = 0.0, mark_wire_ms = 0.0;
    int64_t mark_jobs = 0, mark_chosen = 0;
    double acc_ms = 0.0, acc_launch_ms = 0.0, acc_wire_ms = 0.0;
    int64_t acc_jobs = 0, acc_chosen = 0;
    int acc_win = 0, updates = 0;
    double launch_us = 0.0;        ///< the last reading: the host's us per chosen PCIe blob (0 when the engine passes no launch time)
    double wire_us = 0.0;          ///< the last reading: the PCIe us per chosen blob (the probe's when the engine passes no wire time)
    bool wire_measured = false;    ///< the last reading's wire_us is the copy streams' time (false: the start-up probe's link figure)
    void mark(double ms_run, int64_t jobs, double launch_ms = 0.0, double wire_ms = 0.0, int64_t chosen = 0) {
        mark_ms = ms_run;
        mark_jobs = jobs;
        mark_launch_ms = launch_ms;
        mark_wire_ms = wire_ms;
        mark_chosen = chosen;
    }
    /// The share of a stage's missed experts for a link of `link_gbps` against a CPU pool of `pool_gbps`: the balance
    /// link / (link + pool_factor x pool), rounded to 1/20 and kept within [min_share, max_share].  The only copy of
    /// this formula: every stage of a split takes its share here.
    static double stage_share(double link_gbps, double pool_gbps, double pool_factor = kPoolFactor, double min_share = kMinShare,
                              double max_share = kMaxShare) {
        return std::min(max_share, std::max(min_share, std::round(link_gbps / (link_gbps + pool_factor * pool_gbps) * 20.0) / 20.0));
    }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    /// `launch_ms`: the host time spent launching PCIe copies, cumulative like `ms_run` (none: 0).  `wire_ms`/`chosen`: the
    /// time the copy streams were busy and the blobs the dispatch chose for PCIe, cumulative too (none: 0, then w is the
    /// probe's).  The launch time is read per chosen blob as well, so both sides of the balance count the same blobs.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate, double launch_ms = 0.0,
                  double wire_ms = 0.0, int64_t chosen = 0) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        acc_launch_ms += launch_ms - mark_launch_ms;
        acc_wire_ms += wire_ms - mark_wire_ms;
        acc_chosen += chosen - mark_chosen;
        mark(ms_run, jobs, launch_ms, wire_ms, chosen);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        const double launch = acc_chosen > 0 ? acc_launch_ms * 1000.0 / (double) acc_chosen : 0.0;
        const double measured_wire_us = acc_chosen > 0 ? acc_wire_ms * 1000.0 / (double) acc_chosen : 0.0;
        acc_wire_ms = 0.0;
        acc_chosen = 0;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_launch_ms = 0.0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double link = measured_wire_us > 0.0 ? blob_bytes / (measured_wire_us * 1e3) : link_gbps;
        const double wire = blob_bytes / (link * 1e3);
        const double k = std::max(0.0, pool_factor - launch / wire);   // launch 0: pool_factor, the same double
        const double f = stage_share(link, r, k, min_share, max_share);
        *tc_us = tc;
        *rate = r;
        launch_us = launch;
        wire_us = wire;
        wire_measured = measured_wire_us > 0.0;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace strata::core
