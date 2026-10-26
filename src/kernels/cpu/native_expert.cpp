// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp / kq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool iq512_down_supported(int type) noexcept { return type == 23; }

bool kq256_supported(int type) noexcept { return type == 12 || type == 7 || type == 8; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    if (n_ff > kNativeMaxFF) {
        err = "native experts: expert width " + std::to_string(n_ff) + " is above the pool's " +
              std::to_string(kNativeMaxFF) + " row buffers";
        return false;
    }
    // Q2_0 gate/up reads its activation from an ActQ, sized for cpu::H; a wider n_embd would write past it. Refused
    // whatever the CPU, so a pack loads the same way everywhere.
    if (gu_type == kQ2_0Type && n_embd > cpu::H) {
        err = "native experts: Q2_0 gate/up with n_embd " + std::to_string(n_embd) + " is above the ActQ width " +
              std::to_string(cpu::H);
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 in any projection: one scale tail per blob; a mixed expert (e.g. NVFP4 gate/up + Q8_0 down) has 1.0 for
    // the other projection's scale (tools/iq_pack.py), so every path below multiplies by the tail unconditionally
    if (gu_type == kNvfp4Type || d_type == kNvfp4Type) {
        f.tail_off = f.bytes;
        f.bytes += kNvfp4Tail;
    }
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the older CPUs' copy (iq_avx1.cpp): byte-identical like the AVX-2 one, and only where AVX2 is absent (same
// policy as STRATA_NO_IQ128 above; STRATA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs).
bool q8k_avx1(int type) {
    static const bool on = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_Q8K_AVX1") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the AVX-512 i-quant kernels (iq_avx512.cpp) are on: an AVX-512 CPU, and STRATA_NO_IQ512 unset
bool iq512_on() {
    static const bool on = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    return on;
}
// SwiGLU as ggml-cpu's path applies it: one expression in this translation unit for every path that must match it
inline float swiglu(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }
// STRATA_Q36_SWIGLU_SKIM (#110): the totals, and one call's own count, added to them once when the call ends (no atomic
// per neuron; a call with the skim off has seen nothing and adds nothing)
std::atomic<uint64_t> skim_seen_total{0}, skim_cut_total{0};
struct SkimTally {
    uint64_t seen = 0, cut = 0;
    ~SkimTally() {
        if (seen == 0) return;
        skim_seen_total.fetch_add(seen, std::memory_order_relaxed);
        skim_cut_total.fetch_add(cut, std::memory_order_relaxed);
    }
};
// #110: counts the neuron, and cuts it (ff = 0) when its gate's |silu| is below tau
inline bool skim_cut(float g, float tau, float& ff, SkimTally& t) {
    ++t.seen;
    if (!swiglu_skim(g, tau)) return false;
    ff = 0.f;
    ++t.cut;
    return true;
}
// #110: one line per covered path and process, naming the path the skim runs on
void announce_skim(std::once_flag& once, float tau, const char* path) {
    std::call_once(once, [tau, path] {
        std::fprintf(stderr, "strata: STRATA_Q36_SWIGLU_SKIM %g: gate/up rows on %s\n", (double) tau, path);
    });
}
}  // namespace

void native_swiglu_skim_totals(uint64_t& seen, uint64_t& cut) {
    seen = skim_seen_total.load(std::memory_order_relaxed);
    cut = skim_cut_total.load(std::memory_order_relaxed);
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    if (q8k_avx1(f.gu_act)) { q8k_quant_avx1(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    if (q8k_avx1(f.d_act)) { q8k_quant_avx1(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type, bool one_token_512) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held (not IQ3_XXS on AVX-512 since #74: see native_gu_rows).
    // STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token kernels for every group: output independent of the
    // drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless STRATA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set keeps its rule.
    // Opt-in (STRATA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("STRATA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    // IQ3_XXS on the AVX-512 kernel, for the engines that ask (NativeFmt::one_token_512, the Ornith q36 engine): a token's
    // sums in that kernel do not depend on how many tokens share the decode, so a lone token's rows are the same as in a
    // verify window.  Speed at one token is NOT established for IQ3_XXS: the one measurement this file has (IQ3_S, Zen 4,
    // #152) has the multi-token kernel 1-3% slower than ggml-cpu's dot for a lone token, so the host decides with
    // native_expert_ff1024_test's timing.  STRATA_IQ_MT_MIN set keeps its rule; STRATA_NO_IQ512 leaves the kernel, and the
    // rule stays 2 (a lone token goes to ggml-cpu, as without the flag).
    if (one_token_512 && env == nullptr && gu_type == 18 && iq512_on()) return 1;
    return iq3s_one && gu_type == 21 ? 1 : mt_min;
}

bool native_gu_ggml_order(int gu_type, bool one_token_512) {
    // where a lone token takes ggml-cpu's dot (mt_min above 1, no one_token_512), and only where that dot is ggml's
    // AVX2+FMA one, the order the kernel copies
    static const bool on = iq512_on() && ggml_cpu_has_avx2() && ggml_cpu_has_fma() &&
                           std::getenv("STRATA_NO_IQ512_GGML_ORDER") == nullptr;
    return on && gu_type == 18 && !one_token_512 && native_gu_mt_min(gu_type, one_token_512) > 1;
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    SkimTally skim;   // #110: this call's count, when the skim is on
    const float tau = f.skim_tau;
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = iq512_on();
    // STRATA_NO_IQ256 drops the AVX-2 kernel.  cpu_avx2_ok() is ALSO required: iq_avx2.cpp and kq_avx2.cpp
    // are compiled /arch:AVX2 and use AVX2 and FMA3, so calling them on a CPU without either is an
    // illegal instruction, not a slow path.  The gate sat next to `avx512` above, which does test
    // cpu_avx512_ok(), and that asymmetry is what let a pre-Haswell CPU reach iq256_gu_rows.
    // Without it this falls through to ggml-cpu's vec_dot below, which is compiled for whatever
    // baseline the build selected (AVX1 here) and covers the same types - IQ2_XXS and IQ2_S among
    // them.  That path loops over tokens itself, so it is correct for any `nt`, not just one.
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    // The older-CPU tier (STRATA_ISA_FLOOR=avx builds): without it every expert on such a CPU runs on ggml-cpu's
    // scalar _generic dots.  Only where AVX2 is absent, so a modern CPU's rounding never moves (and
    // STRATA_NO_IQ256=1 there still means ggml, as before).  iq_avx1.cpp is compiled for AVX and gated on
    // cpu_avx1_ok().  Measured on an E5-2470 v2 (iq_avx1_parity --bench): 1.2-3.1x ggml's dot from two tokens on,
    // at one token ggml's auto-vectorized generic loop matches the 128-bit kernel - so the same #152 mt_min rule
    // as the AVX-2 kernels, and STRATA_IQ_MT_MIN=1 takes the kernel for one token too (its rows are then
    // independent of the drafting: iq_avx1_parity checks every width bit for bit).
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    // The exact rows are ggml-cpu's AVX2 single-token dot bit for bit, so they need ggml-cpu itself built for AVX2
    // (ggml_cpu_has_*): a STRATA_ISA_FLOOR build runs ggml's AVX1/SSE dots, which round differently (#2215's guard).
    // ggml-cpu built for AVX-512 runs the same AVX2 dots for these five formats (arch/x86/quants.c at the pinned
    // 3cf03257 has no AVX-512 branch for them), so the !ggml_cpu_has_avx512() check is a conservative guard only.
    static const bool exact_one = [] {
        const char* v = std::getenv("STRATA_CPU_IQ_ALL_EXACT1");
        return v && v[0] == '1' && v[1] == 0 && ggml_cpu_has_avx2() && !ggml_cpu_has_avx512();
    }();
    // only where ggml's dot runs today (below the #152 multi-token rule): with STRATA_IQ_MT_MIN=1 a lone token keeps the
    // multi-token kernel, so a token's rows still do not depend on how many tokens share its expert
    if (exact_one && nt == 1 && nt < native_gu_mt_min(f.gu_type, f.one_token_512) && avx2 && !cpu_avx512_ok() &&
        (f.gu_type == 16 || f.gu_type == 17 || f.gu_type == 18 || f.gu_type == 21 || f.gu_type == 22)) {
        static std::once_flag announced;
        std::call_once(announced, [] { std::fprintf(stderr, "strata ab: GATE=STRATA_CPU_IQ_ALL_EXACT1 active=1\n"); });
        iq256_gu_rows_exact_one(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act[0], ff[0], r0, r1);
        return;
    }
    const int mt_min = native_gu_mt_min(f.gu_type, f.one_token_512);   // #152
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return cpu_avx2_ok() && v != nullptr && std::atoi(v) != 0; }();
    if (kq && f.gu_type == 12 && nt >= 2) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 gate/up kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu
    // for it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    // #74: where a lone token takes ggml-cpu's dot below (mt_min above 1, no one_token_512), an IQ3_XXS group takes the
    // AVX-512 kernel in ggml's float order, then SwiGLU here with the expression and order of ggml's path below (the
    // gate/up scales are 1 for this format), so a token's rows are the same alone and in a verify window.  Only where the
    // linked dot is ggml's AVX2+FMA one, the order the kernel copies.  STRATA_NO_IQ512_GGML_ORDER=1 keeps iq512_gu_rows.
    if (nt >= mt_min && native_gu_ggml_order(f.gu_type, f.one_token_512)) {
        thread_local float gbuf[MAXT][kNativeMaxFF], ubuf[MAXT][kNativeMaxFF];
        float* gp[MAXT];
        float* up[MAXT];
        for (int t = 0; t < nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
        iq512_iq3xxs_gu_rows_ggml(blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, gp, up, r0, r1);
        for (int r = r0; r < r1; ++r) {
#ifdef _MSC_VER
#pragma loop(no_vector)   // a scalar exp, as in ggml's path below (MSVC may vectorize exp with other roundings)
#endif
            for (int t = 0; t < nt; ++t) ff[t][r] = swiglu(gbuf[t][r], ubuf[t][r]);
        }
        if (tau > 0.f) {   // #110: the cut is a pass after the SwiGLU above, so the expression and its order stay as they are
            static std::once_flag announced;
            announce_skim(announced, tau, "the ggml-order kernel");
            for (int r = r0; r < r1; ++r)
                for (int t = 0; t < nt; ++t) skim_cut(gbuf[t][r], tau, ff[t][r], skim);
        }
        return;
    }
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    if (avx1 && nt >= mt_min && iq128_supported(f.gu_type)) {
        iq128_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const int n = (int) f.n_embd;
    // NVFP4: the per-expert global scales from the blob's tail (1 and 1 otherwise). s_down goes on the down output:
    // folded into up it left the hidden near 1e-5, whose Q8_0 block scale is an FP16 subnormal (expert error 2-12%)
    float sg = 1.f, su = 1.f;
    if (f.tail_off) {
        float tail[4];
        std::memcpy(tail, blob + f.tail_off, sizeof tail);
        sg = tail[0];
        su = tail[1];
    }
    // NVFP4 in 512-bit lanes: ggml's arithmetic, blocks decoded once per window; 1.8x at one token, 3.7x at seven
    // (nvfp4_avx512_parity). STRATA_NO_NVFP4_512 falls back to ggml-cpu.
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    // Without AVX-512: the same rows in 256-bit lanes, bit-equal to ggml-cpu's AVX2 dot below; 2.0-2.4x it at 4-8
    // tokens, 2x under DRAM load with 6 threads (nvfp4_avx512_parity). STRATA_NO_NVFP4_256 falls back to ggml-cpu.
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    if (tau > 0.f) {   // #110: the neuron is cut on its gate, after the up row was read as without the skim
        static std::once_flag announced;
        announce_skim(announced, tau, "ggml-cpu's dot");
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            g *= sg;
            u *= su;
            if (tau > 0.f && skim_cut(g, tau, ff[t][r], skim)) continue;
            ff[t][r] = swiglu(g, u);
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const bool mt_env = std::getenv("STRATA_IQ_MT_MIN") != nullptr;
    static const int mt_min = mt_env ? std::atoi(std::getenv("STRATA_IQ_MT_MIN")) : 2;
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    // IQ4_XS down rows on AVX-512, for the engines that ask (NativeFmt::one_token_512): iq512_rows decodes the nibbles and
    // the signed sub-scales once for all tokens, for any token count including one, so a token's rows are the same alone
    // and in a verify window.  The reference is ggml's x86 dot (ggml-cpu/arch/x86/quants.c): the same integer sums, one fused
    // multiply-add per block; only the order of the float additions differs (native_expert_ff1024_test bounds it).  ggml-cpu's
    // dot otherwise, as before (an IQ4_XS down never had another kernel); STRATA_IQ_MT_MIN set takes the kernel back, as
    // it does the gate/up rows', and STRATA_NO_IQ512 drops it.
    if (f.one_token_512 && !mt_env && iq512_on() && iq512_down_supported(f.d_type)) {
        iq512_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // Both multi-token kernels below are /arch:AVX2 translation units (kq_avx2.cpp and iq_avx2.cpp),
    // so a CPU without AVX2 has to reach ggml-cpu's vec_dot instead - same reasoning as the gate/up
    // rows above, where `avx512` tested cpu_avx512_ok() and `avx2` did not.
    if (cpu_avx2_ok() && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // #152: the same rule as the gate/up rows; NativeFmt::one_token_512 serves a lone token too (the kernel's rows do not
    // depend on the token count), STRATA_IQ_MT_MIN set keeping its rule
    const int nl_min = f.one_token_512 && !mt_env ? 1 : mt_min;
    if (cpu_avx2_ok() && nt >= nl_min && f.d_type == 20 && iq4nl_mt) {
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // the older-CPU tier: IQ4_NL and Q2_0 down rows in 128-bit (ggml-cpu's dots for both are scalar here)
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    if (avx1 && iq128_down_supported(f.d_type)) {
        iq128_down_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const int n = (int) f.n_ff;
    float sd = 1.f;                                   // NVFP4: the expert's s_down (see native_gu_rows)
    if (f.tail_off) std::memcpy(&sd, blob + f.tail_off + 2 * sizeof(float), sizeof sd);
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.d_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;   // see above
    if (f.d_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s * sd;
        }
    }
}

}  // namespace strata::kernels::cpu
