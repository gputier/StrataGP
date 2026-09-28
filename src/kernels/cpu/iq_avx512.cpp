// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu.
//
// O8d (#30): the decode was the bound (~5 GB/s per core, far below what the RAM gives): 8 or 16 scalar grid
// lookups inserted lane by lane and 8 scalar sign lookups per 64 values.  `decode_g` does the grid lookups with ONE
// gather and the eight sign lookups with ONE vpermi2b over the 128-byte sign table held in two registers, and the
// tokens' activations are read from a 64-byte-aligned copy (see `ActPack`).  The values are the same bytes, so the
// result is bitwise the previous kernel's, which stays behind STRATA_OLD_IQ512=1 for the A/B.
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// O8d (#30): 0 = gathers + aligned activation copy (default), 1 = gathers on the activations in place
// (STRATA_IQ512_NOPACK=1), 2 = the previous kernel (STRATA_OLD_IQ512=1).
std::atomic<int> g_path{[] {
    if (const char* e = std::getenv("STRATA_OLD_IQ512"); e != nullptr && e[0] == '1') return 2;
    if (const char* e = std::getenv("STRATA_IQ512_NOPACK"); e != nullptr && e[0] == '1') return 1;
    return 0;
}()};

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// O8d (#30): ksigns_iq2xs (128 bytes) in two registers, for `vpermi2b`.
struct SignLut {
    __m512i lo, hi;
};
inline SignLut sign_lut() {
    return {_mm512_loadu_si512((const void*) ksigns_iq2xs), _mm512_loadu_si512((const void*) (ksigns_iq2xs + 64))};
}
// Byte k of `ix` (k < 8) holds a 7-bit sign index in its low bits -> the 64-bit sign mask whose byte k is
// ksigns_iq2xs[index k].  vpermi2b reads bits 0-5 of an index byte and takes the upper table on bit 6; bit 7 is
// ignored, so whatever the truncation left there does not matter.
inline __mmask64 signs8(__m128i ix, const SignLut& L) {
    const __m512i s = _mm512_permutex2var_epi8(L.lo, _mm512_castsi128_si512(ix), L.hi);
    return _cvtu64_mask64((uint64_t) _mm_cvtsi128_si64(_mm512_castsi512_si128(s)));
}
// Two dwords of four 7-bit sign indices each (bits 0, 7, 14, 21), dwords `ia` and `ib` of `v` -> the eight
// indices in bytes 0-7.
template <int ia, int ib>
inline __m128i sign_fields(__m128i v) {
    const __m256i w = _mm256_permutexvar_epi32(_mm256_setr_epi32(ia, ia, ia, ia, ib, ib, ib, ib), _mm256_castsi128_si256(v));
    return _mm256_cvtepi32_epi8(_mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
    // O8d: grid bytes 0-3 and 8-11 of the 16 gathered at once, sign fields of dwords 1 and 3 looked up at once
    static inline void decode_g(const uint8_t* b, int j, const SignLut& L, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const __m128i v = _mm_loadu_si128((const __m128i*) q);
        const __m128i gi = _mm_shuffle_epi8(v, _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1));
        g = _mm512_i32gather_epi64(_mm256_cvtepu8_epi32(gi), (const void*) iq2xxs_grid, 8);
        m = signs8(sign_fields<1, 3>(v), L);
        const uint32_t a1 = u32(q + 4), b1 = u32(q + 12);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
    // O8d: the eight 9-bit grid indices gathered at once, the eight 7-bit sign indices (the top bits) looked up at once
    static inline void decode_g(const uint8_t* b, int j, const SignLut& L, __m512i& g, __mmask64& m, __m512i& sc) {
        const __m128i v = _mm_loadu_si128((const __m128i*) (b + 2 + 16 * j));
        g = _mm512_i32gather_epi64(_mm256_cvtepu16_epi32(_mm_and_si128(v, _mm_set1_epi16(511))), (const void*) iq2xs_grid, 8);
        m = signs8(_mm_cvtepi16_epi8(_mm_srli_epi16(v, 9)), L);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
    // O8d: index k = qs[k] | bits 2k, 2k+1 of (h0 | h1 << 8) above it, all eight gathered at once
    static inline void decode_g(const uint8_t* b, int j, const SignLut&, __m512i& g, __mmask64& m, __m512i& sc) {
        const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (b + 2 + 8 * j)));
        const __m256i h = _mm256_srlv_epi32(_mm256_set1_epi32((int) u16(b + 66 + 2 * j)),
                                            _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
        const __m256i gi = _mm256_or_si256(lo, _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(3)), 8));
        g = _mm512_i32gather_epi64(gi, (const void*) iq2s_grid, 8);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
    // O8d: the sixteen grid bytes gathered at once, the sign fields of `a` and `c` looked up at once
    static inline void decode_g(const uint8_t* b, int j, const SignLut& L, __m512i& g, __mmask64& m, __m512i& sc) {
        const __m128i q = _mm_loadu_si128((const __m128i*) (b + 2 + 16 * j));
        g = _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(q), (const void*) iq3xxs_grid, 4);
        m = signs8(sign_fields<0, 1>(_mm_loadl_epi64((const __m128i*) (b + 2 + 64 + 8 * j))), L);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
    // O8d: bit k of (h0 | h1 << 8) is index k's ninth bit, set with one masked OR; the sixteen gathered at once
    static inline void decode_g(const uint8_t* b, int j, const SignLut&, __m512i& g, __mmask64& m, __m512i& sc) {
        const __m512i lo = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (b + 2 + 16 * j)));
        const __mmask16 h = (__mmask16) u16(b + 66 + 2 * j);
        g = _mm512_i32gather_epi32(_mm512_mask_or_epi32(lo, h, lo, _mm512_set1_epi32(256)), (const void*) iq3s_grid, 4);
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

// ---- O8d (#30): the same row dot on `decode_g`, the activations given as each token's 256-value runs of `qs`
// (`qs[t] + QS * i` for block i) and its scales (`d[t]`, DS bytes apart).  Every operation after the decode is
// `row_dot`'s, on the same values, in the same order.
template <int TY, int NT, int QS, int DS>
inline void row_dot_g(const uint8_t* row, int nblocks, const int8_t* const* qs, const uint8_t* const* d, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    const SignLut L = sign_lut();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode_g(blk, j, L, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (qs[t] + (size_t) QS * i + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t) {
            float dy;
            std::memcpy(&dy, d[t] + (size_t) DS * i, 4);
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * dy), _mm512_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

// O8d (#30): the tokens' Q8_K values, copied once per call.  In a block_q8_K `qs` sits at offset 4 of 292 bytes,
// so every 64-value load of it straddled two cache lines; and the pool's activations are exactly kNativeActBytes =
// 4096 bytes apart, so the same load of every token mapped to the same L1 set.  Here each block's 256 values start
// on a cache line, and the tokens are kPackStride = 3,648 bytes (57 lines) apart, in different sets.
constexpr int kPackTokens = 8;   // gu_rows_nt's widest instance
constexpr int kPackBlocks = (int) (kNativeActBytes / sizeof(block_q8_K));
constexpr size_t kPackStride = (size_t) kPackBlocks * QK_K + 64;
static_assert(kPackStride % 64 == 0 && (kPackStride & 4095) != 0, "the copies must be aligned and not 4K apart");

template <int NT>
struct ActPack {
    alignas(64) int8_t qs[NT][kPackStride];
    float d[NT][kPackBlocks];
    const int8_t* q[NT];
    const uint8_t* dp[NT];
    ActPack(const block_q8_K* const* y, int nb) {
        for (int t = 0; t < NT; ++t) {
            for (int i = 0; i < nb; ++i) {
                for (int k = 0; k < QK_K; k += 64)
                    _mm512_store_si512((void*) (qs[t] + (size_t) QK_K * i + k), _mm512_loadu_si512((const void*) (y[t][i].qs + k)));
                d[t][i] = y[t][i].d;
            }
            q[t] = qs[t];
            dp[t] = (const uint8_t*) d[t];
        }
    }
};

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    const int path = g_path.load(std::memory_order_relaxed);
    if (path == 0 && nb <= kPackBlocks) {
        const ActPack<NT> P(y, nb);
        for (int r = r0; r < r1; ++r) {
            row_dot_g<TY, NT, QK_K, 4>(blob + (size_t) r * gu_row, nb, P.q, P.dp, g);
            row_dot_g<TY, NT, QK_K, 4>(blob + up_off + (size_t) r * gu_row, nb, P.q, P.dp, u);
            for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
        return;
    }
    if (path <= 1) {
        const int8_t* q[NT];
        const uint8_t* d[NT];
        for (int t = 0; t < NT; ++t) { q[t] = y[t]->qs; d[t] = (const uint8_t*) &y[t]->d; }
        for (int r = r0; r < r1; ++r) {
            row_dot_g<TY, NT, sizeof(block_q8_K), sizeof(block_q8_K)>(blob + (size_t) r * gu_row, nb, q, d, g);
            row_dot_g<TY, NT, sizeof(block_q8_K), sizeof(block_q8_K)>(blob + up_off + (size_t) r * gu_row, nb, q, d, u);
            for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
        return;
    }
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    const int nb = n / QK_K;
    const int path = g_path.load(std::memory_order_relaxed);
    if (path == 0 && nb <= kPackBlocks) {
        const ActPack<NT> P(y, nb);
        for (int r = r0; r < r1; ++r) {
            row_dot_g<TY, NT, QK_K, 4>(w + (size_t) r * row_bytes, nb, P.q, P.dp, res);
            for (int t = 0; t < NT; ++t) out[t][r] = res[t];
        }
        return;
    }
    if (path <= 1) {
        const int8_t* q[NT];
        const uint8_t* d[NT];
        for (int t = 0; t < NT; ++t) { q[t] = y[t]->qs; d[t] = (const uint8_t*) &y[t]->d; }
        for (int r = r0; r < r1; ++r) {
            row_dot_g<TY, NT, sizeof(block_q8_K), sizeof(block_q8_K)>(w + (size_t) r * row_bytes, nb, q, d, res);
            for (int t = 0; t < NT; ++t) out[t][r] = res[t];
        }
        return;
    }
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

}  // namespace

void iq512_set_path(int path) noexcept { g_path.store(path < 0 ? 0 : (path > 2 ? 2 : path), std::memory_order_relaxed); }
int iq512_path() noexcept { return g_path.load(std::memory_order_relaxed); }

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
