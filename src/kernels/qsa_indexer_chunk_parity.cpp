// src/kernels/qsa_indexer_chunk_parity.cpp - issue #40: the prompt path's indexer append for a whole chunk
// (native_qsa_indexer_append_chunk) against the per-cell kernel it replaces (GPU, synthetic, no model).
//
// Two copies of one layer's indexer buffers (tail, spare, pooled rows, block_pos) start from the same random
// garbage.  A sequence of random raw keys (magnitudes spread over several decades, so the F16 rounding and the
// norm both matter) is appended to one copy a cell at a time, exactly as the prefill did it, and to the other a
// chunk at a time; after every chunk the two copies must be BITWISE equal, every byte of every buffer - including
// the rows neither should have touched.  Chunks of 1..7 cells and longer ones, starting at every alignment to
// the 4-cell block, one reaching past the capacity (those cells are ignored), with and without the multimodal
// position table, and a capacity that is not a multiple of four.  With --idx-fp16 (O6, #21) the pooling also keeps
// an fp16 shadow of the pooled rows and the spare; the "fp16 shadow" cases compare it too.
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

constexpr int D = 128, R = 4;

struct Bufs {
    float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
    int32_t* block_pos = nullptr;
    uint16_t *pooled16 = nullptr, *dead16 = nullptr;   // O6: the fp16 shadow, when the case keeps one
    int64_t rows = 0;
    void alloc(int64_t max_cells, bool shadow) {
        rows = max_cells / R + 1;
        tail = dalloc<float>((size_t) (R - 1) * D);
        dead = dalloc<float>(D);
        pooled = dalloc<float>((size_t) rows * D);
        block_pos = dalloc<int32_t>(1);
        if (shadow) {
            pooled16 = dalloc<uint16_t>((size_t) rows * D);
            dead16 = dalloc<uint16_t>(D);
        }
    }
    k::QsaIndexerBuffers view() const {
        k::QsaIndexerBuffers v{tail, dead, pooled, block_pos};
        v.pooled16 = pooled16;
        v.dead16 = dead16;
        return v;
    }
    size_t shadow_bytes() const { return pooled16 ? (size_t) (rows * D + D) * 2 : 0; }
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> h((size_t) ((R - 1) * D + D + rows * D) * 4 + 4 + shadow_bytes());
        uint8_t* p = h.data();
        ck(cudaMemcpy(p, tail, (size_t) (R - 1) * D * 4, cudaMemcpyDeviceToHost), "d2h tail");
        p += (size_t) (R - 1) * D * 4;
        ck(cudaMemcpy(p, dead, (size_t) D * 4, cudaMemcpyDeviceToHost), "d2h dead");
        p += (size_t) D * 4;
        ck(cudaMemcpy(p, pooled, (size_t) rows * D * 4, cudaMemcpyDeviceToHost), "d2h pooled");
        p += (size_t) rows * D * 4;
        ck(cudaMemcpy(p, block_pos, 4, cudaMemcpyDeviceToHost), "d2h block_pos");
        if (pooled16 != nullptr) {
            p += 4;
            ck(cudaMemcpy(p, pooled16, (size_t) rows * D * 2, cudaMemcpyDeviceToHost), "d2h pooled16");
            p += (size_t) rows * D * 2;
            ck(cudaMemcpy(p, dead16, (size_t) D * 2, cudaMemcpyDeviceToHost), "d2h dead16");
        }
        return h;
    }
    void fill(const std::vector<uint8_t>& h) {
        const uint8_t* p = h.data();
        ck(cudaMemcpy(tail, p, (size_t) (R - 1) * D * 4, cudaMemcpyHostToDevice), "h2d tail");
        p += (size_t) (R - 1) * D * 4;
        ck(cudaMemcpy(dead, p, (size_t) D * 4, cudaMemcpyHostToDevice), "h2d dead");
        p += (size_t) D * 4;
        ck(cudaMemcpy(pooled, p, (size_t) rows * D * 4, cudaMemcpyHostToDevice), "h2d pooled");
        p += (size_t) rows * D * 4;
        ck(cudaMemcpy(block_pos, p, 4, cudaMemcpyHostToDevice), "h2d block_pos");
        if (pooled16 != nullptr) {
            p += 4;
            ck(cudaMemcpy(pooled16, p, (size_t) rows * D * 2, cudaMemcpyHostToDevice), "h2d pooled16");
            p += (size_t) rows * D * 2;
            ck(cudaMemcpy(dead16, p, (size_t) D * 2, cudaMemcpyHostToDevice), "h2d dead16");
        }
    }
};

// one sequence: `cells` random keys appended in the chunks `sizes` (cycled), per cell into `a`, per chunk into `b`
void run_case(const char* name, int64_t max_cells, int64_t cells, const std::vector<int64_t>& sizes, bool mrope,
              uint32_t seed, cudaStream_t cs, bool shadow = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dec(-3, 2);
    k::QsaShapes s = k::qsa_real_shapes();
    s.idx_dim = D;
    s.idx_block = R;
    std::vector<float> raw_h((size_t) cells * D);
    for (float& v : raw_h) v = u(rng) * std::pow(10.0f, (float) dec(rng));
    std::vector<float> gamma_h(D);
    for (float& v : gamma_h) v = 0.5f + u(rng);
    std::vector<int32_t> pos_h((size_t) cells);
    for (int64_t c = 0; c < cells; ++c) pos_h[(size_t) c] = (int32_t) c;
    float* raw = dalloc<float>(raw_h.size());
    float* gamma = dalloc<float>(D);
    int32_t* pos = dalloc<int32_t>((size_t) cells);
    ck(cudaMemcpy(raw, raw_h.data(), raw_h.size() * 4, cudaMemcpyHostToDevice), "raw");
    ck(cudaMemcpy(gamma, gamma_h.data(), D * 4, cudaMemcpyHostToDevice), "gamma");
    ck(cudaMemcpy(pos, pos_h.data(), pos_h.size() * 4, cudaMemcpyHostToDevice), "pos");
    int32_t* tab = nullptr;
    if (mrope) {   // a picture's positions: t shared, h and w apart (any table the kernels index the same way)
        std::vector<int32_t> t((size_t) max_cells * 3);
        for (int64_t c = 0; c < max_cells; ++c) {
            t[(size_t) c * 3] = (int32_t) (c / 3);
            t[(size_t) c * 3 + 1] = (int32_t) (c / 3 + (c % 7));
            t[(size_t) c * 3 + 2] = (int32_t) (c / 3 + (c % 5));
        }
        tab = dalloc<int32_t>(t.size());
        ck(cudaMemcpy(tab, t.data(), t.size() * 4, cudaMemcpyHostToDevice), "table");
    }
    k::mrope_table_set(tab);
    Bufs a, b;
    a.alloc(max_cells, shadow);
    b.alloc(max_cells, shadow);
    std::vector<uint8_t> junk(a.bytes().size());
    for (uint8_t& x : junk) x = (uint8_t) rng();
    for (size_t i = 0; i + 4 <= junk.size(); i += 4) {   // finite floats: the kernels never read garbage rows, but
        float f = u(rng) * 8.0f;                           // a NaN there would still compare unequal to itself
        std::memcpy(&junk[i], &f, 4);
    }
    a.fill(junk);
    b.fill(junk);
    const float eps = 1e-6f, base = (float) k::qsa_freq_base();
    int64_t c0 = 0, chunks = 0;
    bool ok = true;
    for (size_t si = 0; c0 < cells; ++si) {
        const int64_t n = std::min<int64_t>(sizes[si % sizes.size()], cells - c0);
        for (int64_t t = 0; t < n; ++t)
            k::native_qsa_indexer_append(raw + (size_t) (c0 + t) * D, pos + c0 + t, 0, gamma, eps, a.view(), s,
                                         max_cells, base, cs);
        k::native_qsa_indexer_append_chunk(raw + (size_t) c0 * D, n, c0, 0, gamma, eps, b.view(), s, max_cells,
                                           base, cs);
        ck(cudaStreamSynchronize(cs), "sync");
        const std::vector<uint8_t> ha = a.bytes(), hb = b.bytes();
        if (ha != hb) {
            size_t first = 0;
            while (ha[first] == hb[first]) ++first;
            std::fprintf(stderr, "FAIL %s: chunk %lld (cells %lld..%lld) differs at byte %zu\n", name,
                         (long long) chunks, (long long) c0, (long long) (c0 + n - 1), first);
            ok = false;
            ++g_fail;
            break;
        }
        c0 += n;
        ++chunks;
    }
    k::mrope_table_set(nullptr);
    std::printf("%-44s %s (%lld cells in %lld chunks, capacity %lld)\n", name, ok ? "bitwise equal" : "DIFFERS",
                (long long) cells, (long long) chunks, (long long) max_cells);
    cudaFree(raw); cudaFree(gamma); cudaFree(pos);
    if (tab) cudaFree(tab);
    for (Bufs* x : {&a, &b}) {
        cudaFree(x->tail); cudaFree(x->dead); cudaFree(x->pooled); cudaFree(x->block_pos);
        if (x->pooled16) { cudaFree(x->pooled16); cudaFree(x->dead16); }
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void) argc; (void) argv;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "qsa_indexer_chunk_parity: no CUDA device\n");
        return 2;
    }
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    run_case("small chunks, every alignment", 4096, 700, {1, 2, 3, 5, 7, 1, 4, 6}, false, 1, cs);
    run_case("a first chunk of 1..3 cells", 4096, 40, {3, 1, 2, 8, 9}, false, 2, cs);
    run_case("prompt chunks (8192) after a short start", 40000, 20000, {5, 8192, 8192, 257}, false, 3, cs);
    run_case("long chunks, multimodal positions", 12288, 12000, {1023, 4096, 7, 2048}, true, 4, cs);
    run_case("past the capacity (ignored cells)", 1026, 1400, {100, 333, 600, 1, 366}, false, 5, cs);
    run_case("capacity not a multiple of four", 1023, 1023, {17, 64, 3, 1000}, true, 6, cs);
    run_case("fp16 shadow (--idx-fp16), small chunks", 4096, 700, {1, 2, 3, 5, 7, 1, 4, 6}, false, 7, cs, true);
    run_case("fp16 shadow (--idx-fp16), prompt chunks", 20000, 12000, {3, 8192, 257, 4096}, true, 8, cs, true);
    std::printf("qsa_indexer_chunk_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
