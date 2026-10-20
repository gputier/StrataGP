// src/qwen35/q36_doorbell.hpp - the host side of the q36 hybrid mode's doorbell (header only, no CUDA).
//
// The hybrid mode (experts split between VRAM and the CPU pool) keeps the decode graph with Flash-Next's scheme
// (include/strata/core/layer.hpp, Doorbell): a cudaEventRecord captured in a graph is silently dropped, so the layer
// hands its routing to the host with a kernel.  Per hybrid layer, in graph order:
//
//   GPU   doorbell_publish          payload (hidden, expert ids) to mapped pinned memory, then *seq += 1
//   host  wait_ring(k) ... answer(k)  sees seq >= k, runs the pool into mapped pinned memory, then *flag = k
//   GPU   copy_rows_from_mapped_after  spins until *flag >= k, then copies the pool's rows
//
// k is the layer's place among the hybrid layers of the graph (1, 2, ...), baked into the graph at capture.  A
// replay is made unambiguous by arm(): the host zeroes both words after the previous replay has drained and before
// it launches the next one, so ring k of this replay can never be satisfied by ring k of the last.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::q36 {

enum class RingWait { Rang, Ended, TimedOut };

struct HyDoorbell {
    // the HOST's view of two words in mapped pinned memory, a cache line apart (the GPU writes seq, the host flag)
    uint32_t* h_seq = nullptr;
    uint32_t* h_flag = nullptr;
    // the DEVICE's view of the same bytes
    uint32_t* d_seq = nullptr;
    uint32_t* d_flag = nullptr;

    static constexpr size_t kWords = 32;   // 128 bytes: seq at word 0, flag at word 16

    /// `words` / `d_words`: kWords zeroed words, as the host and as the device address them.
    void bind(uint32_t* words, uint32_t* d_words) {
        h_seq = words;
        h_flag = words + 16;
        d_seq = d_words;
        d_flag = d_words + 16;
    }

    /// Both words to 0: before a launch, with the previous launch's stream drained (no wait kernel can still be
    /// spinning on the old flag).
    void arm() const {
        store(h_seq, 0);
        store(h_flag, 0);
    }

    /// Spins until the GPU has rung ring `want` (seq >= want).  Every `poll_every` it calls `running()`, which asks the
    /// driver whether the stream is still busy (the call also lets the driver submit queued work): when the stream has
    /// ended and the ring is still missing the layer never rang.
    template <class Running>
    RingWait wait_ring(uint32_t want, Running&& running, std::chrono::milliseconds timeout,
                       std::chrono::microseconds poll_every = std::chrono::microseconds(2000)) const {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        auto last_poll = t0;
        uint32_t spins = 0;
        while (load(h_seq) < want) {
            pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_poll > poll_every) {
                last_poll = now;
                if (!running() && load(h_seq) < want) return RingWait::Ended;
            }
            if (now - t0 > timeout) return RingWait::TimedOut;
        }
        return RingWait::Rang;
    }

    /// The host's answer for ring `want`: what it wrote before (the pool's rows) is visible before the flag is.
    void answer(uint32_t want) const { store(h_flag, want); }

    /// Lets every wait of the graph pass (the flag is compared with >=): for a host that gives up on a launch.
    void release() const { store(h_flag, UINT32_MAX); }

private:
    static uint32_t load(uint32_t* p) { return std::atomic_ref<uint32_t>(*p).load(std::memory_order_acquire); }
    static void store(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }
    static void pause() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
};

}  // namespace strata::q36
