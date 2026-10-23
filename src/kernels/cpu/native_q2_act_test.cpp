// src/kernels/cpu/native_q2_act_test.cpp - the pool's Q2_0 gate/up path reads each job's ActQ activation (ExpertJobMulti::act),
// not its ggml one (nact).  hy_cpu_compute (src/qwen35/q36_engine.cu) filled only nact, so a native Q2_0 pack in hybrid
// mode handed the pool a null act: q2_rows_any dereferenced it and the process died (issue #70).  The engine needs CUDA
// and does not link on a host, so this test builds the jobs the way the fixed hy_cpu_compute does (act and nact both set)
// and checks run_split_multi_native against the Q2_0 row kernels called directly, rel <= 1e-4.
//
// `--hy-job` builds the job the way hy_cpu_compute did before the fix (act null): on an AVX2 CPU the pool dies there, the
// red this test guards against.  Without AVX2 the pool takes ggml-cpu's Q2_0 dot, the Q2_0 kernels are not reached, and the
// test exits 77 (ctest reports it as skipped).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "native_expert_test_util.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
// the hidden size of an ActQ (expert.hpp's H, the size act_quant_any writes): the callers quantize with it, so the test does
constexpr int64_t kH = cpu::H, kFF = 1024;   // n_embd, expert width (kNativeMaxFF)
constexpr int kT = 2;                      // tokens in the job
constexpr int kGuType = 42, kDType = 23;   // Q2_0 gate/up (the pool's own kernels), IQ4_XS down (ggml-cpu's rows)
constexpr double kTol = 1e-4;
constexpr int kSkip = 77;

// sum of |a - b| over sum of |b|
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}
}   // namespace

int main(int argc, char** argv) {
    const bool hy_job = argc > 1 && std::strcmp(argv[1], "--hy-job") == 0;
    if (!cpu::cpu_avx2_ok()) {
        std::printf("native_q2_act_test: no AVX2 on this CPU, the Q2_0 kernels are not reached: skipped\n");
        return kSkip;
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(kGuType, kDType, kH, kFF, f, err)) {
        std::printf("native_q2_act_test: FAIL native_fmt: %s\n", err.c_str());
        return 1;
    }

    std::mt19937_64 rng(0x42c70ull);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> blob(f.bytes);
    fill_blocks(blob.data(), f.down_off, kGuType, rng);
    fill_blocks(blob.data() + f.down_off, f.bytes - f.down_off, kDType, rng);

    // the activations as the engine makes them: the ActQ for the Q2_0 kernels, the ggml one for the rest
    std::vector<cpu::ActQ> actq(kT);
    std::vector<uint8_t> nact((size_t) kT * cpu::kNativeActBytes);
    std::vector<float> x((size_t) kH);
    for (int t = 0; t < kT; ++t) {
        for (auto& v : x) v = nd(rng);
        cpu::act_quant_any(x.data(), (int) kH, actq[(size_t) t]);
        cpu::native_quant_act(f, x.data(), nact.data() + (size_t) t * cpu::kNativeActBytes);
    }

    std::vector<float> out((size_t) kT * kH), want((size_t) kT * kH);
    cpu::ExpertJobMulti job;
    job.blob = blob.data();
    job.nt = kT;
    for (int t = 0; t < kT; ++t) {
        job.act[t] = hy_job ? nullptr : &actq[(size_t) t];
        job.nact[t] = nact.data() + (size_t) t * cpu::kNativeActBytes;
        job.out[t] = out.data() + (size_t) t * kH;
    }
    cpu::ExpertPool pool(3, false);
    pool.run_split_multi_native(f, &job, 1);

    // the reference: the Q2_0 rows for gate and up, SwiGLU, then the down rows on the quantized hidden state
    const cpu::ActQ* ap[kT];
    std::vector<std::vector<float>> gate(kT, std::vector<float>((size_t) kFF)), up(kT, std::vector<float>((size_t) kFF)),
        ff(kT, std::vector<float>((size_t) kFF));
    std::vector<std::vector<uint8_t>> hq(kT, std::vector<uint8_t>(cpu::kNativeHBytes));
    float *gp[kT], *upp[kT], *op[kT];
    const void* hp[kT];
    for (int t = 0; t < kT; ++t) {
        ap[t] = &actq[(size_t) t];
        gp[t] = gate[(size_t) t].data();
        upp[t] = up[(size_t) t].data();
    }
    const int nbk = (int) (kH / 64);
    cpu::q2_rows_any(blob.data(), f.gu_row, nbk, ap, kT, gp, 0, (int) kFF);
    cpu::q2_rows_any(blob.data() + f.up_off, f.gu_row, nbk, ap, kT, upp, 0, (int) kFF);
    for (int t = 0; t < kT; ++t) {
        for (int64_t r = 0; r < kFF; ++r) {
            const float g = gate[(size_t) t][(size_t) r];
            ff[(size_t) t][(size_t) r] = g / (1.f + std::exp(-g)) * up[(size_t) t][(size_t) r];
        }
        cpu::native_quant_h(f, ff[(size_t) t].data(), hq[(size_t) t].data());
        hp[t] = hq[(size_t) t].data();
        op[t] = want.data() + (size_t) t * kH;
    }
    cpu::native_down_rows(f, blob.data(), hp, kT, op, 0, (int) kH);

    const double r = rel(out, want);
    std::printf("native_q2_act_test: %s, %d tokens, rel %.2e (tolerance %.0e)\n",
                hy_job ? "job as hy_cpu_compute built it before the fix (act null)" : "job with ActQ activations", kT, r, kTol);
    if (!(r <= kTol)) {
        std::printf("native_q2_act_test: FAIL\n");
        return 1;
    }
    std::printf("native_q2_act_test: ok\n");
    return 0;
}
