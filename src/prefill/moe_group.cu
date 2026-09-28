// src/prefill/moe_group.cu - see include/strata/prefill/moe_group.hpp.
#include "strata/prefill/moe_group.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int CH = 2048;      // pairs per chunk: one block counts them, one warp places them
constexpr int SCAN = 1024;    // the scan's block: one thread per expert

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// hist[c][e] = the pairs of chunk c routed to expert e; bad[c] = 1 if one of its ids is out of range
__global__ void count_kernel(const int32_t* __restrict__ ids, int64_t n, int n_expert, int32_t* __restrict__ hist,
                             int32_t* __restrict__ bad) {
    extern __shared__ int32_t sh[];   // n_expert counters, then the flag
    for (int e = threadIdx.x; e <= n_expert; e += blockDim.x) sh[e] = 0;
    __syncthreads();
    const int64_t i0 = (int64_t) blockIdx.x * CH, i1 = i0 + CH < n ? i0 + CH : n;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const int32_t e = ids[i];
        if ((uint32_t) e < (uint32_t) n_expert) atomicAdd(&sh[e], 1);
        else atomicOr(&sh[n_expert], 1);
    }
    __syncthreads();
    int32_t* h = hist + (size_t) blockIdx.x * n_expert;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) h[e] = sh[e];
    if (threadIdx.x == 0) bad[blockIdx.x] = sh[n_expert];
}

// One block, a thread per expert: hist[c][e] becomes the first row of chunk c's pairs of expert e (the rows of
// the experts before e, then of the chunks before c); out = the counts, then the flag.
__global__ void scan_kernel(int32_t* __restrict__ hist, const int32_t* __restrict__ bad, int nch, int n_expert,
                            int32_t* __restrict__ out) {
    __shared__ int32_t warp_tot[SCAN / 32];
    const int e = threadIdx.x, lane = e & 31, w = e >> 5;
    int32_t total = 0;
    if (e < n_expert)
        for (int c = 0; c < nch; ++c) {
            int32_t* h = hist + (size_t) c * n_expert + e;
            const int32_t v = *h;
            *h = total;
            total += v;
        }
    int32_t x = total;   // inclusive scan over the experts: in the warp, then across the warps
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int32_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_tot[w] = x;
    __syncthreads();
    if (w == 0) {
        int32_t t = warp_tot[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int32_t y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_tot[lane] = t;
    }
    __syncthreads();
    const int32_t first = x - total + (w > 0 ? warp_tot[w - 1] : 0);
    if (e < n_expert) {
        for (int c = 0; c < nch; ++c) hist[(size_t) c * n_expert + e] += first;
        out[e] = total;
    }
    if (e == 0) {
        int32_t b = 0;
        for (int c = 0; c < nch; ++c) b |= bad[c];
        out[n_expert] = b;
    }
}

// One warp per chunk, 32 pairs at a time in pair order: a pair's row is its expert's next row plus the number of
// lower lanes routed to the same expert, and the group's last lane moves that expert's next row on - so rows follow
// the pair index within an expert, as the host's `fill[e]++` did.
__global__ void scatter_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int n_expert,
                               const int32_t* __restrict__ hist, int32_t* __restrict__ slot, int32_t* __restrict__ src) {
    extern __shared__ int32_t next[];   // n_expert
    const int lane = threadIdx.x;
    const int32_t* h = hist + (size_t) blockIdx.x * n_expert;
    for (int e = lane; e < n_expert; e += 32) next[e] = h[e];
    __syncwarp();
    const unsigned below = (1u << lane) - 1u;
    const int64_t i0 = (int64_t) blockIdx.x * CH, i1 = i0 + CH < n ? i0 + CH : n;
    for (int64_t base = i0; base < i1; base += 32) {
        const int64_t i = base + lane;
        const int32_t e = i < i1 ? ids[i] : -1;
        const bool ok = (uint32_t) e < (uint32_t) n_expert;
        const unsigned peers = __match_any_sync(0xffffffffu, ok ? e : n_expert + lane);   // no pair: alone
        const int32_t row = ok ? next[e] + __popc(peers & below) : 0;
        __syncwarp();
        if (ok) {
            slot[i] = row;
            src[row] = (int32_t) (i / k);
            if ((peers >> lane) == 1u) next[e] = row + 1;
        }
        __syncwarp();
    }
}

}  // namespace

size_t moe_group_scratch_bytes(int64_t n, int64_t n_expert) {
    const int64_t nch = (n + CH - 1) / CH;
    return (size_t) (nch * n_expert + nch + n_expert + 1) * sizeof(int32_t);
}

void moe_group(const int32_t* ids, int64_t n, int64_t k, int64_t n_expert, int32_t* slot, int32_t* src,
               void* scratch, int32_t* host_out, void* stream) {
    if (n <= 0 || n_expert < 1 || n_expert > SCAN || k < 1) {
        std::fprintf(stderr, "prefill moe_group: %lld pairs over %lld experts is outside what it covers\n",
                     (long long) n, (long long) n_expert);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int64_t nch = (n + CH - 1) / CH;
    int32_t* hist = (int32_t*) scratch;
    int32_t* bad = hist + nch * n_expert;
    int32_t* out = bad + nch;
    count_kernel<<<(unsigned) nch, 256, (size_t) (n_expert + 1) * sizeof(int32_t), s>>>(ids, n, (int) n_expert, hist, bad);
    check("moe_group count");
    scan_kernel<<<1, SCAN, 0, s>>>(hist, bad, (int) nch, (int) n_expert, out);
    check("moe_group scan");
    scatter_kernel<<<(unsigned) nch, 32, (size_t) n_expert * sizeof(int32_t), s>>>(ids, n, (int) k, (int) n_expert, hist,
                                                                                   slot, src);
    check("moe_group scatter");
    const cudaError_t e = cudaMemcpyAsync(host_out, out, (size_t) (n_expert + 1) * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill moe_group counts: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::prefill
