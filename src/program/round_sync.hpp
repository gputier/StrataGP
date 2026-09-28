// src/program/round_sync.hpp - issue #16: two pieces of the speculative round's host work without a thread
// creation or a synchronous copy (src/program/generate.cpp, the --serve and --spec loops).
//
// AdaptWorker: the adaptive VRAM tier's host work (ranking the experts, submitting the copies) on one thread that
// lives as long as the loop, woken by a condition variable, instead of a std::thread created and joined every
// `--adapt-every` rounds.  STRATA_OLD_ADAPT_THREAD=1 keeps a thread per job.
//
//   worker.start([&] { ok = adapt(); });   // beside the commit and the draft
//   ...
//   worker.wait();                          // where the thread was joined; returns at once when nothing runs
//
// start() and wait() are called from one thread; wait() gives the caller everything the job wrote (the mutex).
//
// ResidencyUpload: the device residency table after a swap, from a pinned copy with cudaMemcpyAsync on the refill
// stream (behind the swapped blobs), instead of a synchronous cudaMemcpy from pageable memory.  No verify window
// reads the device table (the dispatch plans from the host table), so the decode loop only waits for the upload at
// its end (`sync`).  The prompt path does read d_res, and with STRATA_ASYNC_REFILL=1 an upload can be issued
// between its windows: whoever writes d_res synchronously (the serve loop's lend() and refill()) calls `sync`
// first, so an older table cannot land last.  `put(..., wait = true)` is the old synchronous contract.
// STRATA_OLD_RES_UPLOAD=1 keeps cudaMemcpy.
#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace strata::program {

inline bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

class AdaptWorker {
public:
    AdaptWorker() : old_(env_on("STRATA_OLD_ADAPT_THREAD")) {}
    ~AdaptWorker() {
        wait();
        {
            std::lock_guard<std::mutex> l(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    AdaptWorker(const AdaptWorker&) = delete;
    AdaptWorker& operator=(const AdaptWorker&) = delete;

    void start(std::function<void()> job) {
        wait();
        if (old_) {
            once_ = std::thread(std::move(job));
            return;
        }
        {
            std::lock_guard<std::mutex> l(mu_);
            if (!th_.joinable()) th_ = std::thread([this] { loop(); });
            job_ = std::move(job);
            busy_ = true;
        }
        cv_.notify_all();
    }

    void wait() {
        if (once_.joinable()) once_.join();
        std::unique_lock<std::mutex> l(mu_);
        done_.wait(l, [&] { return !busy_; });
    }

private:
    void loop() {
        std::unique_lock<std::mutex> l(mu_);
        for (;;) {
            cv_.wait(l, [&] { return quit_ || busy_; });
            if (!busy_) return;   // quit with nothing to run
            std::function<void()> job = std::move(job_);
            job_ = nullptr;
            l.unlock();
            job();
            l.lock();
            busy_ = false;
            done_.notify_all();
        }
    }

    bool old_ = false;
    std::thread once_;   // STRATA_OLD_ADAPT_THREAD=1: one thread per job
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    std::function<void()> job_;
    bool busy_ = false, quit_ = false;
};

class ResidencyUpload {
public:
    ResidencyUpload() : old_(env_on("STRATA_OLD_RES_UPLOAD")) {}
    ~ResidencyUpload() {
        sync();
        if (ev_) cudaEventDestroy(ev_);
        if (pin_) cudaFreeHost(pin_);
    }
    ResidencyUpload(const ResidencyUpload&) = delete;
    ResidencyUpload& operator=(const ResidencyUpload&) = delete;

    void put(int32_t* d_res, const std::vector<int32_t>& host, cudaStream_t stream, bool wait) {
        const size_t bytes = host.size() * sizeof(int32_t);
        if (!old_ && stream != nullptr && pin_ == nullptr) {
            if (cudaHostAlloc((void**) &pin_, bytes, cudaHostAllocDefault) != cudaSuccess ||
                cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                old_ = true;   // no pinned memory: the synchronous copy, as before
            } else {
                bytes_ = bytes;
            }
        }
        if (old_ || stream == nullptr || bytes != bytes_) {
            sync();
            cudaMemcpy(d_res, host.data(), bytes, cudaMemcpyHostToDevice);
            return;
        }
        cudaEventSynchronize(ev_);   // the previous upload has read the pinned copy (never recorded: complete)
        std::memcpy(pin_, host.data(), bytes);
        cudaMemcpyAsync(d_res, pin_, bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev_, stream);
        if (wait) cudaEventSynchronize(ev_);
    }
    /// The last upload has landed.
    void sync() {
        if (ev_) cudaEventSynchronize(ev_);
    }

private:
    bool old_ = false;
    int32_t* pin_ = nullptr;
    size_t bytes_ = 0;
    cudaEvent_t ev_ = nullptr;
};

}  // namespace strata::program
