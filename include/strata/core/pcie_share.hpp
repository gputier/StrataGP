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
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double min_share = 0.10;       ///< the lowest share a refinement may set (Flash-Next's 0.10; an engine that can do without the share passes 0)
    double max_share = 0.60;       ///< the highest share a refinement may set (Flash-Next's 0.60; an engine whose CPU pool is slow passes more)
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;
    void mark(double ms_run, int64_t jobs) { mark_ms = ms_run; mark_jobs = jobs; }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double f = std::min(max_share, std::max(min_share, std::round(link_gbps / (link_gbps + kPoolFactor * r) * 20.0) / 20.0));
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace strata::core
