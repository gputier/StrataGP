// src/kernels/cpu/native_expert_ff1024_test.cpp - the native (CPU) expert path at Ornith-1.5-122B's geometry: n_embd 3072,
// expert width 1024.  The down activation is then 1168 bytes as Q8_K (IQ4_XS down) or 1088 as Q8_0 (IQ4_NL down), above
// the 1024 the pool's buffers once held (kNativeHBytes), so native_fmt refused the layer.  No GPU, no model: random
// blocks with sane scales, random activations quantized the way the engine quantizes them, the rows against ggml-cpu's
// vec_dot, rel <= 1e-4.  The same layer through ExpertPool::run_split_multi_native with MAXT tokens per expert must give
// native_gu_rows + native_quant_h + native_down_rows called directly, bit for bit: the pool's gate/up row buffers are
// kNativeMaxFF wide (they were FF = 640, so rows 640..1023 landed in the next token's buffer).  native_fmt refuses
// an expert wider than kNativeMaxFF.
//
// NativeFmt::one_token_512 (the q36 engine sets it under STRATA_Q36_ONE_TOKEN_CPU): on an AVX-512 CPU IQ3_XXS gate/up serve a lone token with the
// AVX-512 kernel and an IQ4_XS down is served by it for any token count.  Those rows share ggml-cpu's integer sums and
// differ only in the order of the float additions, so the test bounds them against ggml's dot (kMaxRatio, max abs
// error over the largest reference value) and requires a token's rows to be the same alone and in a group.  Where the
// CPU has no AVX-512 (or STRATA_NO_IQ512 / STRATA_IQ_MT_MIN is set) the flag changes nothing and the rows are ggml's or
// the AVX-2 kernels'; the test prints which path it took.  `--bench` adds ns per row, ggml's dot against the rows with
// the flag off and on, one token, over more expert bytes than a last-level cache holds.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "native_expert_test_util.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int64_t kH = 3072, kFF = 1024;   // n_embd, expert width
constexpr int kMaxT = cpu::MAXT;   // the widest group an expert serves: every token count the kernels split on
constexpr double kMaxRatio = 1e-6;   // max |row - ggml row| / max |ggml row|: float addition order only (see above)

// whether native_gu_rows / native_down_rows take the AVX-512 rows for a flagged layer on this CPU and environment
bool path_512() {
    return cpu::cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr && std::getenv("STRATA_IQ_MT_MIN") == nullptr;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}

// max |a - b| over max |b|
double max_ratio(const std::vector<float>& a, const std::vector<float>& b) {
    double err = 0, top = 0;
    for (size_t i = 0; i < a.size(); ++i) { err = std::max(err, std::fabs((double) a[i] - b[i])); top = std::max(top, std::fabs((double) b[i])); }
    return err / (top + 1e-30);
}

int check_pair(int gu_type, int d_type, bool one_token_512) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type) +
                             (one_token_512 ? " one_token_512" : "");
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) {
        std::printf("  %-14s FAIL native_fmt: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    f.one_token_512 = one_token_512;
    std::printf("  %-14s native_fmt ok: activation %zu bytes (pool buffer %zu), down activation %zu bytes (pool buffer %zu)\n",
                name.c_str(), f.act_bytes, cpu::kNativeActBytes, f.h_bytes, cpu::kNativeHBytes);
    std::mt19937_64 rng(0x0a17e000ull + (uint64_t) gu_type * 64 + (uint64_t) d_type);
    std::vector<uint8_t> blob(f.bytes);
    fill_blocks(blob.data(), f.down_off, gu_type, rng);
    fill_blocks(blob.data() + f.down_off, f.bytes - f.down_off, d_type, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)),
        hq(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
    const void *ap[kMaxT], *hp[kMaxT];
    std::mt19937 frng(7u + (unsigned) gu_type);
    std::vector<float> x((size_t) kH), h((size_t) kFF);
    for (int t = 0; t < kMaxT; ++t) {
        for (auto& v : x) v = nd(frng);
        for (auto& v : h) v = 0.05f * nd(frng);
        cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
        cpu::native_quant_h(f, h.data(), hq[(size_t) t].data());
        ap[t] = act[(size_t) t].data();
        hp[t] = hq[(size_t) t].data();
    }
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) d_type);
    int failures = 0;

    // gate / up: silu(gate . a) * (up . a) for every row, 1..kMaxT tokens
    std::vector<float> ff((size_t) kMaxT * kFF), want_ff((size_t) kMaxT * kFF);
    float* fp[kMaxT];
    for (int t = 0; t < kMaxT; ++t) {
        fp[t] = &ff[(size_t) t * kFF];
        for (int64_t r = 0; r < kFF; ++r) {
            float g = 0.f, u = 0.f;
            tg->vec_dot((int) kH, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            tg->vec_dot((int) kH, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            want_ff[(size_t) t * kFF + (size_t) r] = g / (1.f + std::exp(-g)) * u;
        }
    }
    // a flagged IQ3_XXS gate/up is the AVX-512 kernel for every token count on an AVX-512 CPU: a token's rows are then
    // the same alone and in a group, bit for bit
    const bool same_gu = one_token_512 && gu_type == 18 && path_512();
    const bool same_down = one_token_512 && ((d_type == 23 && path_512()) || (d_type == 20 && cpu::cpu_avx2_ok() &&
                                                                           std::getenv("STRATA_NO_IQ4NL") == nullptr && std::getenv("STRATA_IQ_MT_MIN") == nullptr));
    std::vector<float> alone((size_t) kFF);
    for (int nt = 1; nt <= kMaxT; ++nt) {
        cpu::native_gu_rows(f, blob.data(), ap, nt, fp, 0, (int) kFF);
        const std::vector<float> a(ff.begin(), ff.begin() + (long) nt * kFF), b(want_ff.begin(), want_ff.begin() + (long) nt * kFF);
        const double r = rel(a, b), q = max_ratio(a, b);
        std::printf("  %-14s gate/up, %d token%s: rel %.2e, max abs / max ref %.2e\n", name.c_str(), nt, nt > 1 ? "s" : "", r, q);
        if (!(r <= 1e-4) || !(q <= kMaxRatio)) ++failures;
        if (nt == 1) std::copy(ff.begin(), ff.begin() + kFF, alone.begin());
        else if (same_gu && std::memcmp(alone.data(), ff.data(), (size_t) kFF * sizeof(float)) != 0) {
            std::printf("  %-14s gate/up: token 0 differs between 1 and %d tokens\n", name.c_str(), nt);
            ++failures;
        }
    }

    // down: every row of the 3072 outputs against the 1024-wide activation
    std::vector<float> out((size_t) kMaxT * kH), want((size_t) kMaxT * kH);
    float* op[kMaxT];
    for (int t = 0; t < kMaxT; ++t) {
        op[t] = &out[(size_t) t * kH];
        for (int64_t r = 0; r < kH; ++r)
            td->vec_dot((int) kFF, &want[(size_t) t * kH + (size_t) r], 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[t], 0, 1);
    }
    cpu::native_down_rows(f, blob.data(), hp, kMaxT, op, 0, (int) kH);
    const double r = rel(out, want), q = max_ratio(out, want);
    std::printf("  %-14s down, %d tokens: rel %.2e, max abs / max ref %.2e\n", name.c_str(), kMaxT, r, q);
    if (!(r <= 1e-4) || !(q <= kMaxRatio)) ++failures;
    std::vector<float> down_alone((size_t) kH);
    float* dp[1] = {down_alone.data()};
    cpu::native_down_rows(f, blob.data(), hp, 1, dp, 0, (int) kH);
    if (same_down && std::memcmp(down_alone.data(), out.data(), (size_t) kH * sizeof(float)) != 0) {
        std::printf("  %-14s down: token 0 differs between 1 and %d tokens\n", name.c_str(), kMaxT);
        ++failures;
    }
    std::printf("  %-14s path: gate/up %s, down %s\n", name.c_str(), same_gu ? "AVX-512 for every token count" : "not the one-token AVX-512 rows",
                same_down ? "AVX-512 for every token count" : "not the AVX-512 rows");
    return failures;
}

// ExpertPool::run_split_multi_native at n_ff 1024 with MAXT tokens on several experts, against the direct calls
int check_pool(int gu_type, int d_type, bool one_token_512) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type) +
                             (one_token_512 ? " one_token_512" : "");
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) {
        std::printf("  %-14s pool FAIL native_fmt: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    f.one_token_512 = one_token_512;
    constexpr int kExperts = 5, kT = cpu::MAXT;
    std::mt19937_64 rng(0x900100ull + (uint64_t) gu_type * 64 + (uint64_t) d_type);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> blob(kExperts, std::vector<uint8_t>(f.bytes));
    std::vector<std::vector<std::vector<uint8_t>>> act(kExperts, std::vector<std::vector<uint8_t>>(kT, std::vector<uint8_t>(cpu::kNativeActBytes)));
    std::vector<std::vector<float>> pool_out((size_t) kExperts * kT, std::vector<float>((size_t) kH)),
        want_out((size_t) kExperts * kT, std::vector<float>((size_t) kH));
    std::mt19937 frng(11u + (unsigned) gu_type);
    std::vector<float> x((size_t) kH);
    cpu::ExpertJobMulti jobs[kExperts];
    for (int e = 0; e < kExperts; ++e) {
        fill_blocks(blob[(size_t) e].data(), f.down_off, gu_type, rng);
        fill_blocks(blob[(size_t) e].data() + f.down_off, f.bytes - f.down_off, d_type, rng);
        jobs[e].blob = blob[(size_t) e].data();
        jobs[e].nt = kT;
        for (int t = 0; t < kT; ++t) {
            for (auto& v : x) v = nd(frng);
            cpu::native_quant_act(f, x.data(), act[(size_t) e][(size_t) t].data());
            jobs[e].nact[t] = act[(size_t) e][(size_t) t].data();
            jobs[e].out[t] = pool_out[(size_t) (e * kT + t)].data();
        }
    }
    cpu::ExpertPool pool(3, false);
    pool.run_split_multi_native(f, jobs, kExperts);

    // the reference: the three row kernels called directly on their own buffers
    int bad = 0;
    for (int e = 0; e < kExperts; ++e) {
        std::vector<std::vector<float>> ff(kT, std::vector<float>((size_t) kFF));
        std::vector<std::vector<uint8_t>> hq(kT, std::vector<uint8_t>(cpu::kNativeHBytes));
        float* fp[kT];
        const void* hp[kT];
        float* op[kT];
        for (int t = 0; t < kT; ++t) { fp[t] = ff[(size_t) t].data(); hp[t] = hq[(size_t) t].data(); op[t] = want_out[(size_t) (e * kT + t)].data(); }
        cpu::native_gu_rows(f, blob[(size_t) e].data(), jobs[e].nact, kT, fp, 0, (int) kFF);
        for (int t = 0; t < kT; ++t) cpu::native_quant_h(f, ff[(size_t) t].data(), hq[(size_t) t].data());
        cpu::native_down_rows(f, blob[(size_t) e].data(), hp, kT, op, 0, (int) kH);
        for (int t = 0; t < kT; ++t) {
            const double r = rel(pool_out[(size_t) (e * kT + t)], want_out[(size_t) (e * kT + t)]);
            if (!(r == 0.0)) {
                std::printf("  %-14s pool expert %d token %d: rel %.2e\n", name.c_str(), e, t, r);
                ++bad;
            }
        }
    }
    std::printf("  %-14s run_split_multi_native, %d experts x %d tokens at n_ff %lld: %s\n", name.c_str(), kExperts, kT,
                (long long) kFF, bad == 0 ? "identical to the direct calls" : "DIFFERS");
    return bad == 0 ? 0 : 1;
}

// native_fmt refuses an expert wider than the pool's row buffers
int check_refusal() {
    cpu::NativeFmt f;
    std::string err;
    const bool ok = cpu::native_fmt(18, 23, kH, cpu::kNativeMaxFF + 256, f, err);
    std::printf("  native_fmt at n_ff %lld: %s\n", (long long) (cpu::kNativeMaxFF + 256), ok ? "accepted: FAIL" : err.c_str());
    return ok ? 1 : 0;
}

// ns per row for one token: ggml-cpu's dot, native_*_rows without the flag (what runs today) and with it.  Experts are
// cycled over kBenchExperts blobs (~4 MB each at this geometry, ~260 MB in all) so the weights stream from memory the way
// a decode does and not from a cache.
constexpr int kBenchExperts = 64, kBenchReps = 3;

template <class F>
double ns_per_row(F&& call, int rows) {
    for (int e = 0; e < kBenchExperts; ++e) call(e);   // one warm-up pass
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < kBenchReps; ++rep)
        for (int e = 0; e < kBenchExperts; ++e) call(e);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double) kBenchReps * kBenchExperts * rows);
}

void bench_pair(int gu_type, int d_type) {
    const std::string name = std::string(ggml_type_name((ggml_type) gu_type)) + "/" + ggml_type_name((ggml_type) d_type);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, kH, kFF, f, err)) { std::printf("  %-14s bench: native_fmt: %s\n", name.c_str(), err.c_str()); return; }
    std::mt19937_64 rng(0xbe9c4ull + (uint64_t) gu_type);
    std::vector<std::vector<uint8_t>> blob(kBenchExperts, std::vector<uint8_t>(f.bytes));
    for (auto& b : blob) {
        fill_blocks(b.data(), f.down_off, gu_type, rng);
        fill_blocks(b.data() + f.down_off, f.bytes - f.down_off, d_type, rng);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kH), h((size_t) kFF);
    for (auto& v : x) v = nd(rng);
    for (auto& v : h) v = 0.05f * nd(rng);
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
    cpu::native_quant_act(f, x.data(), act.data());
    cpu::native_quant_h(f, h.data(), hq.data());
    const void* ap[1] = {act.data()};
    const void* hp[1] = {hq.data()};
    std::vector<float> ff((size_t) kFF), out((size_t) kH);
    float* fp[1] = {ff.data()};
    float* op[1] = {out.data()};
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) d_type);
    volatile float sink = 0.f;   // keeps the reference loops from being dropped

    const double gu_ggml = ns_per_row([&](int e) {
        float g = 0.f, u = 0.f;
        for (int64_t r = 0; r < kFF; ++r) {
            tg->vec_dot((int) kH, &g, 0, blob[(size_t) e].data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
            tg->vec_dot((int) kH, &u, 0, blob[(size_t) e].data() + f.up_off + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
            ff[(size_t) r] = g / (1.f + std::exp(-g)) * u;
        }
        sink = ff[0];
    }, (int) kFF);
    f.one_token_512 = false;
    const double gu_off = ns_per_row([&](int e) { cpu::native_gu_rows(f, blob[(size_t) e].data(), ap, 1, fp, 0, (int) kFF); }, (int) kFF);
    f.one_token_512 = true;
    const double gu_on = ns_per_row([&](int e) { cpu::native_gu_rows(f, blob[(size_t) e].data(), ap, 1, fp, 0, (int) kFF); }, (int) kFF);
    const double dn_ggml = ns_per_row([&](int e) {
        for (int64_t r = 0; r < kH; ++r) td->vec_dot((int) kFF, &out[(size_t) r], 0, blob[(size_t) e].data() + f.down_off + (size_t) r * f.d_row, 0, hq.data(), 0, 1);
        sink = out[0];
    }, (int) kH);
    f.one_token_512 = false;
    const double dn_off = ns_per_row([&](int e) { cpu::native_down_rows(f, blob[(size_t) e].data(), hp, 1, op, 0, (int) kH); }, (int) kH);
    f.one_token_512 = true;
    const double dn_on = ns_per_row([&](int e) { cpu::native_down_rows(f, blob[(size_t) e].data(), hp, 1, op, 0, (int) kH); }, (int) kH);
    std::printf("  %-14s one token, ns per row (gate+up pair / down row), %d experts x %d passes, AVX-512 rows %s:\n", name.c_str(), kBenchExperts, kBenchReps,
                path_512() ? "available" : "NOT available here (the flag changes nothing)");
    std::printf("  %-14s   gate/up: ggml %.0f, flag off %.0f, flag on %.0f  (on / ggml %.2f)\n", name.c_str(), gu_ggml, gu_off, gu_on, gu_on / gu_ggml);
    std::printf("  %-14s   down:    ggml %.0f, flag off %.0f, flag on %.0f  (on / ggml %.2f)\n", name.c_str(), dn_ggml, dn_off, dn_on, dn_on / dn_ggml);
}
}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    int failures = 0;
    // the pairs Ornith's quantization uses: IQ3_XXS gate/up with IQ4_XS down (Q8_K down activation, 1168 bytes), and
    // IQ3_XXS with IQ4_NL down (Q8_0, 1088 bytes), each without and with NativeFmt::one_token_512
    for (const bool flag : {false, true}) {
        failures += check_pair(18, 23, flag);
        failures += check_pair(18, 20, flag);
        failures += check_pool(18, 23, flag);
        failures += check_pool(18, 20, flag);
    }
    failures += check_refusal();
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) {
            bench_pair(18, 23);
            bench_pair(18, 20);
        }
    std::printf("native_expert_ff1024_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
