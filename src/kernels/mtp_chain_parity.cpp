// src/kernels/mtp_chain_parity.cpp - the one-graph MTP draft chain (strata/kernels/mtp_chain.hpp) against the host
// loop it replaces (src/core/mtp.cpp, MtpDrafter::draft with STRATA_OLD_MTP_CHAIN=1):
//
//   n = 1; p = p0;  for (j = 1; j < limit && p >= min_p; ++j) { out[j] = id_j; p = prob_j; out_p[j] = p; ++n; }
//
// The body stands in for the draft layer: its "forward" reads the draft id and probability back from the record
// mtp_chain_stage copied for step j (the id as step[0], the probability as the bits of pos[0]), through a 2-D
// memcpy node and a memset node, as the real body has.  The head stands in for the round, which MtpDrafter records
// before mtp_chain_init in the same graph: a memset node and a copy of p0 from the pinned staging, so the chain
// only starts once that work is done.  Synthetic data, no model.
//   1. every (limit, min_p, probability pattern) of a seeded sweep: the draft count, the drafts and their
//      probabilities are exactly the host loop's; the outputs past the count are untouched.
//   2. the same graph launched again and again (the counter restarts at every launch).
//   --bench: one chain launch + one synchronization against one launch + one synchronization per step, staged once
//   (indicative only: a synthetic body, and the per-step side relaunches this graph with limit 2).

#include "strata/kernels/mtp_chain.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
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

template <typename T>
void mapped(size_t n, T** h, T** d) {
    ck(cudaHostAlloc((void**) h, n * sizeof(T) + 64, cudaHostAllocMapped), "host alloc");
    std::memset(*h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer((void**) d, *h, 0), "device alias");
}

int32_t bits(float f) { int32_t b; std::memcpy(&b, &f, 4); return b; }

constexpr int kMaxT = 8, kRows = 2 * kMaxT, kNH = 32, kRow0 = kMaxT;
constexpr int32_t kSentinel = -777;

struct Rig {
    cudaStream_t s = nullptr;
    int32_t *h_step = nullptr, *m_step = nullptr, *h_pos = nullptr, *m_pos = nullptr;
    int32_t *h_par = nullptr, *m_par = nullptr, *h_out = nullptr, *m_out = nullptr, *h_p0 = nullptr, *m_p0 = nullptr;
    float *h_outp = nullptr, *m_outp = nullptr;
    int32_t *ctl = nullptr, *step_dst = nullptr, *pos_dst = nullptr, *ids = nullptr, *row = nullptr, *scratch = nullptr;
    cudaGraphExec_t exec = nullptr;

    void init() {
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        mapped(kRows * 4, &h_step, &m_step);
        mapped(kRows * kNH, &h_pos, &m_pos);
        mapped(16, &h_par, &m_par);
        mapped(kMaxT, &h_out, &m_out);
        mapped(kMaxT, &h_outp, &m_outp);
        mapped(4, &h_p0, &m_p0);
        ck(cudaMalloc(&ctl, 16), "ctl");
        ck(cudaMalloc(&step_dst, 16), "step");
        ck(cudaMalloc(&pos_dst, kNH * 4), "pos");
        ck(cudaMalloc(&ids, 16), "ids");
        ck(cudaMalloc(&scratch, 256), "scratch");
        ck(cudaMalloc(&row, 16), "row");
        ck(cudaMemset(row, 0, 16), "row zero");
        // cudaMemset runs on the legacy stream, which the non-blocking `s` does not wait for
        ck(cudaDeviceSynchronize(), "setup");
        auto head = [&](unsigned long long h) {
            // the round's work: it leaves p0 in probs[*row], here the bits of pos_dst[0], copied at every launch
            if (cudaMemsetAsync(scratch, 0, 256, s) != cudaSuccess) return false;
            if (cudaMemcpyAsync(pos_dst, h_p0, 4, cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
            k::mtp_chain_init(ctl, (const float*) pos_dst, row, m_par, m_par + 2, h, s);
            return true;
        };
        auto body = [&](unsigned long long h) {
            k::mtp_chain_stage(ctl, m_step, m_pos, kRow0, kNH, step_dst, pos_dst, s);
            if (cudaMemsetAsync(scratch, 0, 256, s) != cudaSuccess) return false;
            if (cudaMemcpy2DAsync(ids, 16, step_dst, 16, 16, 1, cudaMemcpyDeviceToDevice, s) != cudaSuccess) return false;
            k::mtp_chain_next(ctl, ids, (const float*) pos_dst, row, m_out, m_outp, m_par, m_par + 2, h, s);
            return true;
        };
        std::string err;
        void* e = nullptr;
        if (!k::build_while_graph(s, head, body, &e, err)) {
            std::fprintf(stderr, "building the chain graph failed: %s\n", err.c_str());
            std::exit(2);
        }
        exec = (cudaGraphExec_t) e;
    }

    // one round's staging (mapped / pinned host memory, read by the graph when it runs)
    void stage(int limit, float min_p, float p0, const std::vector<int32_t>& id, const std::vector<float>& pr) {
        for (int j = 1; j < kMaxT; ++j) {
            const int r = kRow0 + j - 1;
            for (int c = 0; c < 4; ++c) h_step[r * 4 + c] = c == 0 ? id[(size_t) j] : 1000 * j + c;
            for (int c = 0; c < kNH; ++c) h_pos[r * kNH + c] = c == 0 ? bits(pr[(size_t) j]) : 5000 + c;
        }
        for (int j = 0; j < kMaxT; ++j) { h_out[j] = kSentinel; h_outp[j] = -1.0f; }
        h_par[0] = limit;
        h_par[1] = bits(min_p);
        h_par[2] = 0;
        h_p0[0] = bits(p0);
    }

    // one launch and one synchronization: returns the draft count; out / outp hold the chain's outputs
    int launch() {
        ck(cudaGraphLaunch(exec, s), "launch");
        ck(cudaStreamSynchronize(s), "sync");
        return ((volatile int32_t*) h_par)[2];
    }

    int run(int limit, float min_p, float p0, const std::vector<int32_t>& id, const std::vector<float>& pr) {
        stage(limit, min_p, p0, id, pr);
        return launch();
    }
};

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { std::fprintf(stderr, "no CUDA device\n"); return 2; }
    Rig rig;
    rig.init();
    std::mt19937 rng(20260928);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float vals[] = {0.0f, 0.05f, 0.3f, 0.5f, 0.50000006f, 0.9f, 1.0f, nan};
    const float min_ps[] = {0.0f, 0.3f, 0.5f, 1.0f};
    std::printf("mtp_chain_parity: the WHILE-node draft chain against the host loop\n");
    int trials = 0, bad = 0, bad_untouched = 0;
    int64_t steps = 0;
    for (int rep = 0; rep < 40; ++rep)
        for (int limit = 1; limit < kMaxT; ++limit)
            for (const float min_p : min_ps) {
                std::vector<int32_t> id(kMaxT);
                std::vector<float> pr(kMaxT);
                for (int j = 0; j < kMaxT; ++j) {
                    id[(size_t) j] = (int32_t) (rng() % 248320u);
                    pr[(size_t) j] = vals[rng() % (sizeof vals / sizeof vals[0])];
                }
                const float p0 = vals[rng() % (sizeof vals / sizeof vals[0])];
                // the host loop (mtp.cpp before issue #16)
                int n_ref = 1;
                float p = p0;
                std::vector<int32_t> out_ref(kMaxT, kSentinel);
                std::vector<float> outp_ref(kMaxT, -1.0f);
                for (int j = 1; j < limit && p >= min_p; ++j) {
                    out_ref[(size_t) j] = id[(size_t) j];
                    p = pr[(size_t) j];
                    outp_ref[(size_t) j] = p;
                    ++n_ref;
                }
                const int n = rig.run(limit, min_p, p0, id, pr);
                ++trials;
                steps += n - 1;
                bool ok = n == n_ref;
                for (int j = 1; ok && j < n; ++j)
                    ok = rig.h_out[j] == out_ref[(size_t) j] &&
                         std::memcmp(&rig.h_outp[j], &outp_ref[(size_t) j], 4) == 0;
                bool untouched = true;
                for (int j = std::max(n, 1); j < kMaxT; ++j) untouched &= rig.h_out[j] == kSentinel && rig.h_outp[j] == -1.0f;
                untouched &= rig.h_out[0] == kSentinel;
                if (!ok && bad < 5)
                    std::printf("    mismatch: limit %d min_p %g p0 %g: count %d, host %d\n", limit, min_p, p0, n, n_ref);
                bad += !ok;
                bad_untouched += !untouched;
            }
    char line[160];
    std::snprintf(line, sizeof line, "%d rounds (%lld chain steps): counts, drafts, probabilities", trials, (long long) steps);
    std::printf("  %-66s %s\n", line, bad == 0 ? "ok" : "FAIL");
    g_fail += bad != 0;
    std::printf("  %-66s %s\n", "outputs past the count untouched", bad_untouched == 0 ? "ok" : "FAIL");
    g_fail += bad_untouched != 0;

    if (bench) {
        // staged once; both sides time only the launches and the synchronizations.  Every step keeps p = 1, so the
        // staging stays valid: limit 7 runs 6 steps per launch, limit 2 one step per launch.
        using Clock = std::chrono::steady_clock;
        std::vector<int32_t> id(kMaxT, 7);
        std::vector<float> pr(kMaxT, 1.0f);
        const int R = 2000, S = kMaxT - 2;
        rig.stage(kMaxT - 1, 0.0f, 1.0f, id, pr);
        int n_one = 0, n_per = 0;
        for (int r = 0; r < 20; ++r) n_one = rig.launch();
        auto t0 = Clock::now();
        for (int r = 0; r < R; ++r) rig.launch();   // 6 steps in one launch
        const double one = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / R;
        rig.h_par[0] = 2;
        for (int r = 0; r < 20; ++r) n_per = rig.launch();
        t0 = Clock::now();
        for (int r = 0; r < R; ++r)
            for (int st = 0; st < S; ++st) rig.launch();   // one step per launch + sync
        const double per = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / R;
        if (n_one != S + 1 || n_per != 2)
            std::printf("  bench: unexpected draft counts %d and %d (want %d and 2)\n", n_one, n_per, S + 1);
        std::printf("  bench (indicative, synthetic body): %d steps, one launch %.1f us/round; one launch per step "
                    "%.1f us/round\n", S, one, per);
    }
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
