// src/kernels/ple_stage_parity.cpp - issue #15: the verify window's late PLE staging (GPU, synthetic, no model).
//
// The window graph used to copy the mapped PLE rows at its head, so the host had to read them from the SSD before
// the launch. It now waits on a mapped flag (wait_flag_ge) just before layer 1 and copies them there, and the host
// fills them AFTER the launch (src/core/verify.cpp). On a captured graph replayed window after window, as the
// verifier uses it, this checks:
//   1. the graph really waits: it has not finished while the flag is down;
//   2. the late copy delivers exactly the bytes written after the launch, bitwise, for every window;
//   3. the head copy (STRATA_OLD_PLE_STAGING=1) and the late copy give identical device rows for the same data.

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void* mapped(size_t bytes, void** dev) {
    void* h = nullptr;
    ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(h, 0, bytes);
    ck(cudaHostGetDevicePointer(dev, h, 0), "cudaHostGetDevicePointer");
    return h;
}

}  // namespace

int main() {
    constexpr int N = 2560;                    // n_embd: 16 heads x 160
    constexpr int MT = 8;                      // kVerifyMaxT
    float* m_ple = nullptr;
    uint32_t* m_flag = nullptr;
    float* m_other = nullptr;
    float* h_ple = (float*) mapped((size_t) MT * N * 4, (void**) &m_ple);
    uint32_t* h_flag = (uint32_t*) mapped(64, (void**) &m_flag);
    float* h_other = (float*) mapped((size_t) MT * N * 4, (void**) &m_other);   // stands in for layer 0's inputs
    float *ple = nullptr, *other = nullptr, *seen_late = nullptr, *seen_head = nullptr;
    ck(cudaMalloc(&ple, (size_t) MT * N * 4), "malloc");
    ck(cudaMalloc(&other, (size_t) MT * N * 4), "malloc");
    ck(cudaMalloc(&seen_late, (size_t) MT * N * 4), "malloc");
    ck(cudaMalloc(&seen_head, (size_t) MT * N * 4), "malloc");
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(15);
    std::printf("ple_stage_parity (issue #15)\n");

    for (int T : {1, 3, 8}) {
        const int64_t n = (int64_t) T * N;
        // ---- the two graphs, as record_window captures them
        cudaGraphExec_t late = nullptr, head = nullptr;
        for (int arm = 0; arm < 2; ++arm) {
            const bool is_late = arm == 0;
            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
            if (!is_late) k::copy_from_mapped(ple, m_ple, n, cs);           // the head copy (the previous path)
            k::copy_from_mapped(other, m_other, n, cs);                      // "the embedding and layer 0"
            if (is_late) {
                k::wait_flag_ge(m_flag, 1u, cs);                             // "pre(1, first group)"
                k::copy_from_mapped(ple, m_ple, n, cs);
            }
            ck(cudaMemcpyAsync(is_late ? seen_late : seen_head, ple, (size_t) n * 4, cudaMemcpyDeviceToDevice, cs),
               "consumer");                                                  // "layer 1 reads the rows"
            cudaGraph_t g = nullptr;
            ck(cudaStreamEndCapture(cs, &g), "end capture");
            ck(cudaGraphInstantiate(is_late ? &late : &head, g, 0), "instantiate");
            ck(cudaGraphDestroy(g), "graph destroy");
        }
        bool waited = true, late_ok = true, head_ok = true, same = true;
        for (int w = 0; w < 6; ++w) {
            std::vector<float> data((size_t) n);
            for (auto& v : data) {
                const uint32_t bits = (uint32_t) rng();
                std::memcpy(&v, &bits, 4);                                   // any bit pattern, NaNs included
            }
            // ---- late: launched on stale rows and a lowered flag, filled while the graph runs
            for (int64_t i = 0; i < n; ++i) h_ple[i] = -1.0f;
            for (int64_t i = 0; i < n; ++i) h_other[i] = (float) i;
            *(volatile uint32_t*) h_flag = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ck(cudaGraphLaunch(late, cs), "launch late");
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            if (cudaStreamQuery(cs) != cudaErrorNotReady) waited = false;
            std::memcpy(h_ple, data.data(), (size_t) n * 4);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flag = 1u;
            ck(cudaStreamSynchronize(cs), "sync late");
            std::vector<float> a((size_t) n), b((size_t) n);
            ck(cudaMemcpy(a.data(), seen_late, (size_t) n * 4, cudaMemcpyDeviceToHost), "down");
            late_ok = late_ok && std::memcmp(a.data(), data.data(), (size_t) n * 4) == 0;
            // ---- head: the rows written before the launch
            std::memcpy(h_ple, data.data(), (size_t) n * 4);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ck(cudaGraphLaunch(head, cs), "launch head");
            ck(cudaStreamSynchronize(cs), "sync head");
            ck(cudaMemcpy(b.data(), seen_head, (size_t) n * 4, cudaMemcpyDeviceToHost), "down");
            head_ok = head_ok && std::memcmp(b.data(), data.data(), (size_t) n * 4) == 0;
            same = same && std::memcmp(a.data(), b.data(), (size_t) n * 4) == 0;
        }
        char what[128];
        std::snprintf(what, sizeof what, "T=%d: the graph waits for the flag", T);
        check(waited, what);
        std::snprintf(what, sizeof what, "T=%d: late copy = rows written after the launch (6 windows)", T);
        check(late_ok, what);
        std::snprintf(what, sizeof what, "T=%d: head copy = rows written before the launch", T);
        check(head_ok, what);
        std::snprintf(what, sizeof what, "T=%d: late and head copies bitwise identical", T);
        check(same, what);
        cudaGraphExecDestroy(late);
        cudaGraphExecDestroy(head);
    }
    cudaStreamDestroy(cs);
    cudaFree(ple);
    cudaFree(other);
    cudaFree(seen_late);
    cudaFree(seen_head);
    cudaFreeHost(h_ple);
    cudaFreeHost(h_flag);
    cudaFreeHost(h_other);
    std::printf("ple_stage_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
