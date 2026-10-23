// tests/core/q36_doorbell_test.cpp - the sequencing of strata-q36's hybrid-mode doorbell (src/qwen35/q36_doorbell.hpp), no GPU:
// a thread plays the captured graph (publish: payload then seq += 1; wait: spin until flag >= the ring baked at capture, then
// read the host's rows) and the main thread plays hy_serve.  Replays back to back must never read a stale ring or stale rows.
#include "q36_doorbell.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

using strata::q36::HyDoorbell;
using strata::q36::RingWait;
using namespace std::chrono_literals;

// one buffer for both views: the replay threads share it, which a CPU address space allows.  The engine's two views of
// the mapped pinned memory are two addresses; main() checks that bind() wires them apart (split_views).
struct Words {
    alignas(64) uint32_t w[HyDoorbell::kWords] = {};
    HyDoorbell db;
    Words() { db.bind(w, w); }
};

uint32_t rd(uint32_t* p) { return std::atomic_ref<uint32_t>(*p).load(std::memory_order_acquire); }
void wr(uint32_t* p, uint32_t v) { std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release); }

// The graph: layer k writes its payload (replay * 1000 + k), rings, waits for ring k (a value fixed at capture), then reads
// the row the host produced for it.  `rows_seen[k]` is what it read.
void run_graph(HyDoorbell& db, std::vector<uint32_t>& payload, std::vector<uint32_t>& rows, int layers, uint32_t replay,
               std::vector<uint32_t>& rows_seen) {
    for (int k = 0; k < layers; ++k) {
        payload[(size_t) k] = replay * 1000 + (uint32_t) k;
        wr(db.d_seq, rd(db.d_seq) + 1);   // doorbell_publish: an increment, so a replay counts on from the armed 0
        while (rd(db.d_flag) < (uint32_t) k + 1) {
        }
        rows_seen[(size_t) k] = rows[(size_t) k];
    }
}

// hy_serve: ring k seen, the pool's answer (a function of the payload) written, then the flag
void serve(HyDoorbell& db, std::vector<uint32_t>& payload, std::vector<uint32_t>& rows, int layers,
           std::vector<uint32_t>& seen_payload) {
    for (int k = 0; k < layers; ++k) {
        if (db.wait_ring((uint32_t) k + 1, [] { return true; }, 5s) != RingWait::Rang) std::abort();
        seen_payload[(size_t) k] = payload[(size_t) k];
        rows[(size_t) k] = payload[(size_t) k] + 7;
        db.answer((uint32_t) k + 1);
    }
}

// the graph's wait for flag k, read through the device view and bounded: true once the flag reaches k
bool graph_wait_passes(const HyDoorbell& db, uint32_t k) {
    for (long spin = 0; spin < 10000000; ++spin)
        if (rd(db.d_flag) >= k) return true;
    return false;
}
}  // namespace

int main() {
    constexpr int kLayers = 40, kReplays = 200;
    Words ws;
    std::vector<uint32_t> payload(kLayers), rows(kLayers), seen(kLayers), seen_payload(kLayers);

    bool all_ok = true;
    for (uint32_t r = 1; r <= kReplays; ++r) {
        ws.db.arm();   // the previous replay has drained: its thread was joined
        std::thread gpu(run_graph, std::ref(ws.db), std::ref(payload), std::ref(rows), kLayers, r, std::ref(seen));
        serve(ws.db, payload, rows, kLayers, seen_payload);
        gpu.join();
        for (int k = 0; k < kLayers; ++k)
            all_ok = all_ok && seen_payload[(size_t) k] == r * 1000 + (uint32_t) k &&
                     seen[(size_t) k] == r * 1000 + (uint32_t) k + 7;
        all_ok = all_ok && rd(ws.w) == (uint32_t) kLayers && rd(ws.w + 16) == (uint32_t) kLayers;
    }
    check(all_ok, "200 replays of a 40-layer graph: every ring serves its own payload, every wait reads its own rows");

    // a flag left over from a replay would pass the next replay's waits before the host answered: arm() clears it
    wr(ws.w, 40);
    wr(ws.w + 16, 40);
    ws.db.arm();
    check(rd(ws.w) == 0 && rd(ws.w + 16) == 0, "arm() zeroes the sequence word and the flag");

    // a ring that never comes: the stream ended, then a timeout
    check(ws.db.wait_ring(1, [] { return false; }, 1s, 1ms) == RingWait::Ended, "a finished stream with no ring: Ended");
    check(ws.db.wait_ring(1, [] { return true; }, 30ms, 1ms) == RingWait::TimedOut, "a busy stream with no ring: TimedOut");
    wr(ws.w, 3);
    check(ws.db.wait_ring(3, [] { return false; }, 1s, 1ms) == RingWait::Rang, "a ring already there: Rang, whatever the stream says");
    check(ws.db.wait_ring(2, [] { return false; }, 1s, 1ms) == RingWait::Rang, "a later ring covers an earlier one (seq >= want)");

    // release lets every wait of the graph pass
    ws.db.arm();
    ws.db.release();
    bool passed = true;
    for (uint32_t k = 1; k <= 48; ++k) passed = passed && graph_wait_passes(ws.db, k);
    check(passed, "release() passes the waits of all layers");

    // the engine's views are two buffers: each word is bound to its own side, the flag a cache line after seq on both
    alignas(64) uint32_t host_words[HyDoorbell::kWords] = {}, dev_words[HyDoorbell::kWords] = {};
    HyDoorbell split;
    split.bind(host_words, dev_words);
    check(split.h_seq == host_words && split.h_flag == host_words + HyDoorbell::kFlagWord && split.d_seq == dev_words &&
              split.d_flag == dev_words + HyDoorbell::kFlagWord,
          "bind() takes the host and the device view of each word from their own buffer");

    std::printf("%s\n", g_fail == 0 ? "q36 doorbell: all checks passed" : "q36 doorbell: FAILED");
    return g_fail == 0 ? 0 : 1;
}
