// tests/core/q36_kv_q4_reference_test.cpp - the math of strata-q36's `--kv q4` cache on one KV head, on the host: no model, no GPU.
//
// The engine rotates q, k and v by the orthonormal 256-point Hadamard transform H, stores k and v as q4_0 cells (kv_q4.hpp's
// layout: 8 blocks of 32 values, 144 bytes, element e of a block in the low nibble of qs[e] for e < 16 and the high nibble
// of qs[e - 16] after), scores the rotated q against the dequantized k, mixes the dequantized v and rotates the mix back
// with H before the output gate.  This test runs those steps with the kernels' own formulas (k_attn_prep's q4_group,
// k_attn_part's dequantization, k_attn_merge's inverse rotation) against float64 attention on the unrotated values:
//   - H is its own inverse and keeps q.k;
//   - a cell stored at row * 8 + block and read back through the nibble layout equals the quantizer's own values;
//   - the scores and the attention output stay within the stated tolerances on K / V with outlier channels, the case the
//     rotation is for.  A GPU run of the kernels (parity and KL against the fp32 cache) is the engine's own gate.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace k = strata::kernels;

constexpr int HD = 256;
constexpr int kBlocks = HD / k::QK4_0;
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// k_attn_prep's fwht256_block: the butterflies of every stage, then the 1/16 scale
void fwht256(float* x) {
    for (int h = 1; h < HD; h <<= 1) {
        float y[HD];
        for (int i = 0; i < HD; ++i) y[i] = (i & h) ? x[i ^ h] - x[i] : x[i] + x[i ^ h];
        std::memcpy(x, y, sizeof y);
    }
    for (int i = 0; i < HD; ++i) x[i] *= 1.0f / 16.0f;
}

// q4_group (kv_q4_group.cuh) on 32 values: ggml's q4_0, d = (the value of largest |x|) / -8 (the larger value on ties)
void store_group(k::block_q4_0* blk, const float* x) {
    float amax = 0.f, mval = 0.f;
    for (int e = 0; e < 32; ++e) {
        const float a = std::fabs(x[e]);
        if (a > amax || (a == amax && x[e] > mval)) { amax = a; mval = x[e]; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t q[32];
    for (int e = 0; e < 32; ++e) {
        const int qi = (int) (x[e] * id + 8.5f);   // truncation, as __float2int_rz
        q[e] = (uint8_t) (qi < 0 ? 0 : (qi > 15 ? 15 : qi));
    }
    blk->d = k::f16_from_f32(d);
    for (int e = 0; e < 16; ++e) blk->qs[e] = (uint8_t) (q[e] | (q[e + 16] << 4));
}
// q4_value: value e of a block
float block_value(const k::block_q4_0& b, int e) {
    const uint8_t v = b.qs[e & 15];
    return (float) ((e < 16 ? (v & 0x0F) : (v >> 4)) - 8) * k::f32_from_f16(b.d);
}

// the cache of one KV head: cell p = HD floats stored at row p (8 blocks); read back the way k_attn_part reads
struct Q4Cache {
    std::vector<k::block_q4_0> blocks;
    explicit Q4Cache(int cells) : blocks((size_t) cells * kBlocks) {}
    void put(int cell, const float* x) {
        for (int b = 0; b < kBlocks; ++b) store_group(&blocks[(size_t) cell * kBlocks + b], x + b * 32);
    }
    float at(int cell, int e) const { return block_value(blocks[(size_t) cell * kBlocks + e / 32], e % 32); }
};

double dot(const float* a, const float* b) {
    double s = 0;
    for (int i = 0; i < HD; ++i) s += (double) a[i] * b[i];
    return s;
}
}  // namespace

int main() {
    static_assert(sizeof(k::block_q4_0) == 18, "kv_q4.hpp's block");
    check(k::kv_q4_bytes_per_head(HD) == 144, "a cell of one KV head is 144 bytes (8 blocks of 18)");

    std::mt19937 rng(20261020);
    std::normal_distribution<float> gauss(0.f, 1.f);
    const int n = 1024;
    // K / V as the model writes them: a few channels far above the rest (what the rotation spreads); q likewise
    std::vector<float> K((size_t) n * HD), V((size_t) n * HD), q(HD);
    const int outliers[4] = {3, 77, 130, 201};
    for (int p = 0; p < n; ++p)
        for (int i = 0; i < HD; ++i) {
            K[(size_t) p * HD + i] = gauss(rng);
            V[(size_t) p * HD + i] = gauss(rng);
        }
    for (int p = 0; p < n; ++p)
        for (int c : outliers) {
            K[(size_t) p * HD + c] *= 12.f;
            V[(size_t) p * HD + c] *= 6.f;
        }
    for (int i = 0; i < HD; ++i) q[i] = gauss(rng);

    // H is its own inverse and keeps the scalar product
    {
        std::vector<float> x(HD), y(HD);
        for (int i = 0; i < HD; ++i) { x[i] = q[i]; y[i] = K[i]; }
        const double before = dot(x.data(), y.data());
        fwht256(x.data());
        fwht256(y.data());
        const double rotated = dot(x.data(), y.data());
        check(std::fabs(rotated - before) <= 1e-4 * std::fabs(before) + 1e-4, "H keeps q.k (relative 1e-4)");
        fwht256(x.data());
        double e = 0;
        for (int i = 0; i < HD; ++i) e = std::max(e, (double) std::fabs(x[i] - q[i]));
        check(e < 1e-5, "H applied twice gives the input back (max error < 1e-5)");
    }

    // the cells, rotated, stored and read back
    Q4Cache kc(n), vc(n);
    std::vector<float> Kr = K, Vr = V, qr = q;
    for (int p = 0; p < n; ++p) {
        fwht256(&Kr[(size_t) p * HD]);
        fwht256(&Vr[(size_t) p * HD]);
        kc.put(p, &Kr[(size_t) p * HD]);
        vc.put(p, &Vr[(size_t) p * HD]);
    }
    fwht256(qr.data());
    {
        k::block_q4_0 ref;
        store_group(&ref, &Kr[32]);   // cell 0, block 1
        bool same = std::memcmp(&ref, &kc.blocks[1], sizeof ref) == 0;
        for (int e = 0; e < 32; ++e) same = same && kc.at(0, 32 + e) == block_value(ref, e);
        check(same, "cell 0 block 1 is stored at row 0 * 8 + 1 and read back through the nibble layout");
        double worst = 0, se2 = 0, ss2 = 0;
        for (int p = 0; p < n; ++p)
            for (int b = 0; b < kBlocks; ++b) {
                const k::block_q4_0& blk = kc.blocks[(size_t) p * kBlocks + b];
                const double d = std::fabs((double) k::f32_from_f16(blk.d));
                for (int e = 0; e < 32; ++e) {
                    const double x = Kr[(size_t) p * HD + b * 32 + e], err = (double) block_value(blk, e) - x;
                    worst = std::max(worst, std::fabs(err) / (d > 0 ? d : 1.0));
                    se2 += err * err;
                    ss2 += x * x;
                }
            }
        // the nearest code is half a step away; the one value of a block that sits at +8 steps (the largest one's opposite)
        // is clamped to code 15, a full step away; the fp16 scale adds about 0.4%
        check(worst <= 1.0 + 0.02, "every k value is within one step (+ 2% for the fp16 scale) of its code");
        std::printf("  rotated k, q4_0 round trip: relative RMS error %.4f\n", std::sqrt(se2 / ss2));
    }

    // attention for many queries, float64 on the unrotated values against the engine's steps (rotated: q, k, v, the mix rotated
    // back; plain: the same q4_0 cells without the rotation, to show what it is for).  Errors are averaged over the queries.
    struct Err { double score = 0, out = 0; };
    const int n_q = 64;
    const double scale = 1.0 / std::sqrt((double) HD);
    Q4Cache kp(n), vp(n);   // the unrotated values in q4_0 cells
    for (int p = 0; p < n; ++p) {
        kp.put(p, &K[(size_t) p * HD]);
        vp.put(p, &V[(size_t) p * HD]);
    }
    auto softmax = [&](const std::vector<double>& sc, std::vector<double>& pr) {
        const double m = *std::max_element(sc.begin(), sc.end());
        double l = 0;
        for (int p = 0; p < n; ++p) l += (pr[p] = std::exp(sc[p] - m));
        for (int p = 0; p < n; ++p) pr[p] /= l;
    };
    auto run = [&](bool rotate) {
        const Q4Cache& kcache = rotate ? kc : kp;
        const Q4Cache& vcache = rotate ? vc : vp;
        Err acc;
        std::mt19937 qrng(7);
        for (int qi = 0; qi < n_q; ++qi) {
            std::vector<float> qv(HD), qs(HD);
            for (int i = 0; i < HD; ++i) qs[i] = qv[i] = gauss(qrng);
            for (int c : outliers) qs[c] = qv[c] *= 3.f;   // queries are large on the outlier channels too
            if (rotate) fwht256(qs.data());
            std::vector<double> s_ref(n), s_q4(n), p_ref(n), p_q4(n);
            for (int p = 0; p < n; ++p) {
                s_ref[p] = dot(qv.data(), &K[(size_t) p * HD]) * scale;
                double a = 0;
                for (int i = 0; i < HD; ++i) a += (double) qs[i] * kcache.at(p, i);
                s_q4[p] = a * scale;
            }
            softmax(s_ref, p_ref);
            softmax(s_q4, p_q4);
            std::vector<float> o_ref(HD), o_q4(HD);
            for (int i = 0; i < HD; ++i) {
                double a = 0, b = 0;
                for (int p = 0; p < n; ++p) {
                    a += p_ref[p] * V[(size_t) p * HD + i];
                    b += p_q4[p] * vcache.at(p, i);
                }
                o_ref[i] = (float) a;
                o_q4[i] = (float) b;
            }
            if (rotate) fwht256(o_q4.data());   // k_attn_merge: the mix of rotated values, rotated back
            double ss = 0, se = 0, os = 0, oe = 0;
            for (int p = 0; p < n; ++p) {
                ss += s_ref[p] * s_ref[p];
                se += (s_q4[p] - s_ref[p]) * (s_q4[p] - s_ref[p]);
            }
            for (int i = 0; i < HD; ++i) {
                os += (double) o_ref[i] * o_ref[i];
                oe += ((double) o_q4[i] - o_ref[i]) * ((double) o_q4[i] - o_ref[i]);
            }
            acc.score += std::sqrt(se / ss) / n_q;
            acc.out += std::sqrt(oe / os) / n_q;
        }
        return acc;
    };
    const Err rot = run(true), plain = run(false);
    std::printf("  %d queries, %d cells: with the rotation scores %.4f / output %.4f (relative error), without %.4f / %.4f\n", n_q, n,
                rot.score, rot.out, plain.score, plain.out);
    // (a gaussian K row rounded to 4 bits is 0.079 off, see above; the output also carries the scores' error through the softmax)
    check(rot.score < 0.08, "scores: mean relative RMS error under 0.08");
    check(rot.out < 0.25, "attention output: mean relative L2 error under 0.25");
    check(rot.score < plain.score && rot.out < plain.out, "the rotation lowers both errors on K / V with outlier channels");

    // the prompt's key windows (k_attn_win): a chunk of T queries at cells pos0 .. pos0 + T - 1 attends window by window with
    // the running max / sum and the rescale of the accumulated rows, then divides by the sum; against the one-shot softmax
    // over [0, pos] of the same dequantized cells.  Window 96 does not divide the cells, so the last one is short, and the
    // rows of the chunk end at different cells of it.  Every row sees at least 29 cells of every window it enters, so the
    // case of a row that sees none of a window is not exercised here.
    {
        const int T = 40, pos0 = 700, W = 96;
        const double fscale = scale;
        double worst = 0;
        std::vector<float> qrows((size_t) T * HD);
        std::vector<double> O((size_t) T * HD, -12345.0);   // O starts as garbage: the first window overwrites
        std::vector<double> mrun(T), lrun(T);
        for (float& v : qrows) v = gauss(rng);
        for (int c0 = 0; c0 < pos0 + T; c0 += W) {
            const int Pw = std::min(W, pos0 + T - c0);
            for (int t = 0; t < T; ++t) {
                const int nvis = std::min(Pw, pos0 + t + 1 - c0);
                std::vector<double> e(Pw, 0.0);
                double mx = -INFINITY;
                std::vector<double> sc(std::max(nvis, 0));
                for (int p = 0; p < nvis; ++p) {
                    double a = 0;
                    for (int i = 0; i < HD; ++i) a += (double) qrows[(size_t) t * HD + i] * kc.at(c0 + p, i);
                    sc[p] = a * fscale;
                    mx = std::max(mx, sc[p]);
                }
                const double mold = c0 == 0 ? -INFINITY : mrun[t];
                const double mnew = std::max(mold, mx);
                double sum = 0;
                for (int p = 0; p < nvis; ++p) sum += (e[p] = std::exp(sc[p] - mnew));
                const double alpha = std::exp(mold - mnew);
                lrun[t] = (c0 == 0 ? 0.0 : lrun[t]) * alpha + sum;
                mrun[t] = mnew;
                for (int d = 0; d < HD; ++d) {
                    double acc = c0 == 0 ? 0.0 : O[(size_t) t * HD + d] * alpha;   // beta = 0 on the first window's GEMM
                    for (int p = 0; p < nvis; ++p) acc += e[p] * vc.at(c0 + p, d);
                    O[(size_t) t * HD + d] = acc;
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            const int n_vis = pos0 + t + 1;
            std::vector<double> sc(n_vis);
            double mx = -1e300, l = 0;
            for (int p = 0; p < n_vis; ++p) {
                double a = 0;
                for (int i = 0; i < HD; ++i) a += (double) qrows[(size_t) t * HD + i] * kc.at(p, i);
                mx = std::max(mx, sc[p] = a * scale);
            }
            for (int p = 0; p < n_vis; ++p) l += (sc[p] = std::exp(sc[p] - mx));
            for (int d = 0; d < HD; ++d) {
                double ref = 0;
                for (int p = 0; p < n_vis; ++p) ref += sc[p] / l * vc.at(p, d);
                const double got = O[(size_t) t * HD + d] / lrun[t];
                worst = std::max(worst, std::fabs(got - ref) / (std::fabs(ref) + 1e-3));
            }
        }
        std::printf("  windowed prompt attention: %d queries, window %d, worst relative difference to the one-shot softmax %.2e\n", T, W,
                    worst);
        check(worst < 1e-9, "windowed online softmax equals the one-shot softmax (relative 1e-9, float64 arithmetic)");
    }

    std::printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
