// src/kernels/rope_parity.cpp - P2.S2's parity test for NEOX partial RoPE.
//
// TWO CHECKS, deliberately separated so each can be tight:
//
//   1. THE TABLE against the float64 reference - `theta ** (-2i/n_rot)`, `pos * inv`, `cos`/`sin`, all in
//      double exactly as `ref/qsa.py::rope_freqs` does.  Compared as float32 after the cast, so a mismatch
//      here is a real arithmetic difference and not a formatting one.
//   2. THE ROTATION against a host rotation written from `ref/qsa.py::rope_neox`, using the SAME table the
//      kernel is given.  Both sides then perform identical float32 operations, so this is BIT-EXACT - and a
//      bit-exact rotation plus a table that matches the reference means the composition matches too.
//
// A single end-to-end comparison would have had to carry one loose tolerance for both, and the interesting
// failure - the NEOX pairing being wrong - is a PERMUTATION that a loose tolerance over all 256 dims would
// happily accept.
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: rope_parity [--selftest]\n"); return 2; }
    }

    const int n_rot = 64;          // the artifact's rope.dimension_count
    const double theta = 1.0e7;    // rope.freq_base
    const int head_dim = 256;
    const int rows = 24 * 8;       // 24 heads over 8 positions, so several positions exercise the table
    const int max_pos = 32;
    const int half = n_rot / 2;
    int bad = 0;

    // ---- 1. the table against the float64 reference
    std::vector<float> hcos((size_t) max_pos * half), hsin((size_t) max_pos * half);
    strata::kernels::build_rope_table(n_rot, theta, max_pos, hcos.data(), hsin.data());
    long long table_bad = 0;
    double table_worst = 0.0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            const float rc = (float) std::cos(ang), rs = (float) std::sin(ang);
            if (std::memcmp(&rc, &hcos[(size_t) p * half + i], 4) != 0) ++table_bad;
            if (std::memcmp(&rs, &hsin[(size_t) p * half + i], 4) != 0) ++table_bad;
            table_worst = std::max(table_worst,
                                   std::fabs((double) rc - (double) hcos[(size_t) p * half + i]));
        }
    }
    std::printf("  rope table vs float64 reference   %s (%lld of %d entries differ, worst %.1e)\n",
                table_bad ? "*** WRONG ***" : "bit-exact", table_bad, max_pos * half * 2, table_worst);
    bad += (int) table_bad;

    // ---- 2. the rotation, against the same spec in float32 with the SAME table
    std::mt19937 rng(11);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * head_dim);
    for (auto& v : x) v = gauss(rng);
    std::vector<int> pos((size_t) rows);
    for (int r = 0; r < rows; ++r) pos[(size_t) r] = r % max_pos;

    std::vector<float> ref(x.size());
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[(size_t) r * head_dim];
        float* orow = &ref[(size_t) r * head_dim];
        for (int d = 0; d < head_dim; ++d) orow[d] = xr[d];          // the tail passes through
        for (int i = 0; i < half; ++i) {
            const float c = hcos[(size_t) pos[(size_t) r] * half + i];
            const float s = hsin[(size_t) pos[(size_t) r] * half + i];
            const float a = xr[i], b = xr[half + i];
            orow[i] = a * c - b * s;
            orow[half + i] = a * s + b * c;
        }
    }

    float *d_x = nullptr, *d_out = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    int* d_pos = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check(cudaMalloc(&d_out, ref.size() * sizeof(float)), "malloc out");
    check(cudaMalloc(&d_cos, hcos.size() * sizeof(float)), "malloc cos");
    check(cudaMalloc(&d_sin, hsin.size() * sizeof(float)), "malloc sin");
    check(cudaMalloc(&d_pos, pos.size() * sizeof(int)), "malloc pos");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_cos, hcos.data(), hcos.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cos");
    check(cudaMemcpy(d_sin, hsin.data(), hsin.size() * sizeof(float), cudaMemcpyHostToDevice), "copy sin");
    check(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pos");
    strata::kernels::rope_neox_apply(d_x, d_out, rows, head_dim, n_rot, d_cos, d_sin, d_pos, nullptr);
    std::vector<float> got(ref.size());
    check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");

    // A RELATIVE TOLERANCE, NOT BIT EQUALITY.  *c - b*s is one of the expressions the compiler is free to
    // contract into an FMA, and the host and device compilers choose differently - so a handful of values
    // differ in the last bit.  Round 167 established this the hard way: a bit-equality expectation that a
    // kernel cannot meet is a test that reports a correct kernel as broken.  The residual is reported, and if
    // it were structural (a wrong pairing, a wrong table entry) it would be O(1), not O(1 ULP).
    long long rot_bad = 0;
    double worst = 0.0;
    long long first_bad = -1;
    // The denominator is the ROW'S INPUT MAGNITUDE, not the element's own value.  *c - b*s cancels, so an
    // element whose result is near zero reports a large relative error for a 1-ULP absolute one - which is the
    // same metric mistake round 169 found on Q4_K's dot products, where the fix was to measure against
    // sum|term| instead of the result.  Here the natural scale of the computation is the largest input in the
    // row, because every output element is a combination of two inputs with |cos|,|sin| <= 1.
    std::vector<double> row_scale((size_t) rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        double m = 0;
        for (int d = 0; d < head_dim; ++d) m = std::max(m, (double) std::fabs(x[(size_t) r * head_dim + d]));
        row_scale[(size_t) r] = m > 1e-30 ? m : 1e-30;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = got[i];
        const double rel = std::fabs(a - b) / row_scale[i / (size_t) head_dim];
        if (!(rel <= worst)) worst = rel;
        if (!(rel <= 1e-6)) { if (first_bad < 0) first_bad = (long long) i; ++rot_bad; }
    }
    std::printf("  rope rotation vs host reference   %s (%lld of %zu over 1e-6, worst rel %.3e)\n",
                rot_bad ? "*** WRONG ***" : "agrees", rot_bad, ref.size(), worst);
    if (first_bad >= 0) std::printf("    first over-tolerance at element %lld\n", first_bad);
    bad += (int) rot_bad;

    // ---- 3. THE PAIRING, asserted structurally rather than as a tolerance.  NEOX pairs (i, i+half); the
    // adjacent-pair convention would pair (2i, 2i+1).  Feed an input that is 1 at dim 0 and 0 elsewhere, and
    // check WHICH dims move: under NEOX only dims 0 and half may change, under the adjacent convention only
    // dims 0 and 1 may.  A tolerance over all 256 dims cannot tell the two apart.
    {
        std::vector<float> e((size_t) head_dim, 0.0f);
        e[0] = 1.0f;
        std::vector<float> o((size_t) head_dim, 0.0f);
        check(cudaMemcpy(d_x, e.data(), e.size() * sizeof(float), cudaMemcpyHostToDevice), "copy e");
        // position 7, NOT position 0: at pos 0 sin is exactly 0, so dim half legitimately does not move and
        // the check would report the correct kernel as wrong.  The first version made exactly that mistake.
        int p7 = 7;
        check(cudaMemcpy(d_pos, &p7, sizeof(int), cudaMemcpyHostToDevice), "copy p7");
        strata::kernels::rope_neox_apply(d_x, d_out, 1, head_dim, n_rot, d_cos, d_sin, d_pos, nullptr);
        check(cudaMemcpy(o.data(), d_out, o.size() * sizeof(float), cudaMemcpyDeviceToHost), "back e");
        int moved[4] = {0, 0, 0, 0};       // dims 0, 1, half, half+1
        const float* chk[4] = {&e[0], &e[1], &e[half], &e[half + 1]};
        (void) chk;
        moved[0] = (o[0] != 0.0f || o[half] != 0.0f) ? 1 : 0;
        const bool adjacent_moved = (o[1] != 0.0f);
        const bool neox_moved = (o[half] != 0.0f);
        std::printf("  pairing: dim 0 set -> NEOX moves dim %d (o[half]=%.6f), adjacent would move dim 1 "
                    "(o[1]=%.6f)   %s\n", half, (double) o[half], (double) o[1],
                    (neox_moved && !adjacent_moved) ? "NEOX confirmed" : "*** WRONG CONVENTION ***");
        if (!(neox_moved && !adjacent_moved)) ++bad;
    }

    // ---- 4. #3: THE NATIVE RoPE (`--native`, the production path) AT LONG POSITIONS, against float64.
    //
    // `native_rope.cu` reproduces the pinned llama.cpp kernel: theta_scale = powf(base, -2/n_rot) on the host in
    // f32, then on the device theta = pos * powf(theta_scale, pair) and cosf/sinf - compiled with --use_fast_math,
    // so __powf, __cosf and __sinf.  The table path above is float64 end to end.  Sections 1-3 only reach
    // position 31; the error of the f32 angle grows with the angle, so this measures it where it is largest.
    //
    // THE PHASE IS READ BACK DIRECTLY: a row with x[pair] = 1 and x[pair + 32] = 0 for every pair comes back as
    // (cos theta', sin theta'), so atan2 gives the device's angle theta' for all 32 pairs of one position, and
    // theta' - theta64 (wrapped) is the phase error.  A second set of rows with random inputs gives the error of
    // the rotated values themselves, relative to the row's largest input.
    //
    // THE TOLERANCE IS A DOCUMENTED ERROR MODEL, not a measured number.  Relative to the angle, the f32 path
    // loses up to ~p * 2.6e-7 + 4.6e-7: theta_scale rounded to f32 and raised to the p-th power, __powf's
    // lg2 (2^-22 absolute on [0.5, 2]) and ex2 (2 ulp), the f32 product pos * inv, and __sinf/__cosf, which
    // multiply by an f32 1/(2 pi) - rounded toward zero - before the hardware sine takes the fraction of a turn.
    // Pair 0 has only the last two (1.6e-7).  Since inv(p) = base^(-p/32) falls faster than p grows, the worst
    // pair (p = 1) stays below ~4.3e-7 * pos radians.  The assertion allows 1e-4 + 1e-6 * pos radians: 2.3x that
    // sum of worst cases, and still ~4x under the 1 rad of the smallest structural error (the position off by
    // one on pair 0).  At 131072 the model allows up to ~0.056 rad; the audit estimated ~1e-2.  The measured
    // numbers are printed per position so the real figure is on record.
    {
        const int nat_rot = 64, nat_half = nat_rot / 2, nat_dim = 256;
        const float base = 1.0e7f;                                // rope.freq_base, as the engine passes it
        const std::vector<int> npos = {0, 1, 31, 1024, 4096, 8192, 16384, 32768, 65535, 65536, 100000, 131071,
                                       131072, 200000, 262143, 262144};
        const int np = (int) npos.size();
        // rows [0, np): the phase probes; rows [np, 2 np): random inputs at the same positions
        std::vector<float> nx((size_t) 2 * np * nat_dim);
        for (int r = 0; r < np; ++r)
            for (int d = 0; d < nat_dim; ++d)
                nx[(size_t) r * nat_dim + d] = d < nat_half ? 1.0f : d < nat_rot ? 0.0f : (float) d;
        for (size_t i = (size_t) np * nat_dim; i < nx.size(); ++i) nx[i] = gauss(rng);
        std::vector<int> nrow_pos((size_t) 2 * np);
        for (int r = 0; r < 2 * np; ++r) nrow_pos[(size_t) r] = npos[(size_t) (r % np)];

        float *d_nx = nullptr, *d_nout = nullptr;
        int* d_npos = nullptr;
        cudaStream_t nst = nullptr;
        check(cudaStreamCreate(&nst), "native stream");
        check(cudaMalloc(&d_nx, nx.size() * sizeof(float)), "malloc nx");
        check(cudaMalloc(&d_nout, nx.size() * sizeof(float)), "malloc nout");
        check(cudaMalloc(&d_npos, nrow_pos.size() * sizeof(int)), "malloc npos");
        check(cudaMemcpy(d_nx, nx.data(), nx.size() * sizeof(float), cudaMemcpyHostToDevice), "copy nx");
        check(cudaMemcpy(d_npos, nrow_pos.data(), nrow_pos.size() * sizeof(int), cudaMemcpyHostToDevice), "copy npos");
        // Twice: the default (llama.cpp's f32 fast-math angle, the error model above) and the opt-in float64 angle
        // (STRATA_ROPE_F64, the A/B of #3), which must be exact to the f32 rounding of cos/sin: 1e-6 rad.
        for (const bool f64 : {false, true}) {
            strata::kernels::native_rope_set_f64_angle(f64);
            strata::kernels::native_rope_apply(d_nx, d_nout, 2 * np, nat_dim, nat_rot, base, d_npos, nst);
            check(cudaStreamSynchronize(nst), "native rope");
            std::vector<float> nout(nx.size());
            check(cudaMemcpy(nout.data(), d_nout, nout.size() * sizeof(float), cudaMemcpyDeviceToHost), "back nout");
            // in place (x == out), which is how every engine caller uses it, must give the same bits
            check(cudaMemcpy(d_nout, nx.data(), nx.size() * sizeof(float), cudaMemcpyHostToDevice), "copy inplace");
            strata::kernels::native_rope_apply(d_nout, d_nout, 2 * np, nat_dim, nat_rot, base, d_npos, nst);
            check(cudaStreamSynchronize(nst), "native rope in place");
            std::vector<float> ninp(nx.size());
            check(cudaMemcpy(ninp.data(), d_nout, ninp.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "back inplace");
            const bool inplace_same = std::memcmp(ninp.data(), nout.data(), nout.size() * sizeof(float)) == 0;

            const double two_pi = 2.0 * 3.14159265358979323846;
            long long nat_bad = 0, tail_bad = 0;
            std::printf("\n  native RoPE (%s) vs float64, base %.0e:\n",
                        f64 ? "STRATA_ROPE_F64: the float64 angle" : "fast-math, pinned llama.cpp", (double) base);
            std::printf("    %8s  %12s %5s  %12s  %12s  %10s\n", "position", "max phase", "pair", "tolerance",
                        "values rel", "|c,s|-1");
            for (int r = 0; r < np; ++r) {
                const int pos = npos[(size_t) r];
                const float* o = &nout[(size_t) r * nat_dim];
                double worst = 0.0, worst_norm = 0.0;
                int worst_pair = 0;
                for (int i = 0; i < nat_half; ++i) {
                    const double th = (double) pos * std::pow((double) base, -2.0 * (double) i / (double) nat_rot);
                    const double got = std::atan2((double) o[nat_half + i], (double) o[i]);
                    const double err = std::fabs(std::remainder(got - th, two_pi));
                    if (!(err <= worst)) { worst = err; worst_pair = i; }
                    const double nrm = std::fabs(std::hypot((double) o[i], (double) o[nat_half + i]) - 1.0);
                    worst_norm = std::max(worst_norm, nrm);
                }
                for (int d = nat_rot; d < nat_dim; ++d) tail_bad += o[d] != (float) d;
                // the random rows: every rotated value against the float64 rotation, over the row's largest input
                const float* xr = &nx[(size_t) (np + r) * nat_dim];
                const float* orow = &nout[(size_t) (np + r) * nat_dim];
                double m = 1e-30, vworst = 0.0;
                for (int d = 0; d < nat_dim; ++d) m = std::max(m, (double) std::fabs(xr[d]));
                for (int i = 0; i < nat_half; ++i) {
                    const double th = (double) pos * std::pow((double) base, -2.0 * (double) i / (double) nat_rot);
                    const double c = std::cos(th), s = std::sin(th), a = xr[i], b = xr[nat_half + i];
                    vworst = std::max(vworst, std::fabs(a * c - b * s - (double) orow[i]) / m);
                    vworst = std::max(vworst, std::fabs(a * s + b * c - (double) orow[nat_half + i]) / m);
                }
                for (int d = nat_rot; d < nat_dim; ++d) tail_bad += orow[d] != xr[d];
                const double tol = f64 ? 1e-6 : 1e-4 + 1e-6 * (double) pos;
                // |value error| <= |(a, b)| * |phase error| + the sine's own amplitude error, and |(a, b)| <= sqrt(2) m
                const double amp = f64 ? 1e-6 : 1e-5;
                const bool ok = worst <= tol && vworst <= 1.5 * tol + amp && worst_norm <= amp;
                std::printf("    %8d  %12.3e %5d  %12.3e  %12.3e  %10.1e  %s\n", pos, worst, worst_pair, tol, vworst,
                            worst_norm, ok ? "ok" : "*** OVER ***");
                if (!ok) ++nat_bad;
            }
            std::printf("  native RoPE%s: %lld positions over tolerance, %lld pass-through values changed, "
                        "in place %s\n", f64 ? " (float64 angle)" : "", nat_bad, tail_bad,
                        inplace_same ? "bit-identical" : "*** DIFFERS ***");
            bad += (int) nat_bad + (tail_bad ? 1 : 0) + (inplace_same ? 0 : 1);
        }
        strata::kernels::native_rope_set_f64_angle(false);
        cudaFree(d_nx); cudaFree(d_nout); cudaFree(d_npos);
        check(cudaStreamDestroy(nst), "native stream destroy");
    }

    std::printf("\nrope: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("rope_parity OK\n");
    return 0;
}
