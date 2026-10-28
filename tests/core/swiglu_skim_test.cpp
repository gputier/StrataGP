// tests/core/swiglu_skim_test.cpp - the SwiGLU skim's rule (#110, include/strata/kernels/cpu/native_expert.hpp), no GPU: a
// neuron is cut when |silu(g)| < tau, on its gate alone.  tau <= 0 cuts nothing, a NaN gate or threshold is never cut, and
// the threshold is strict: a neuron whose |silu| equals tau is kept.
#include "check.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {
// the rule's own expression, so the threshold below is the value the rule compares against, bit for bit
float silu(float g) { return g / (1.f + std::exp(-g)); }
}  // namespace

int main() {
    using strata::kernels::cpu::swiglu_skim;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::printf("swiglu_skim:\n");
    {
        check(swiglu_skim(0.f, 0.1f), "g = 0 (silu 0) is cut under a positive threshold");
        check(!swiglu_skim(0.f, 0.f), "g = 0 with tau = 0 is kept: a threshold of 0 cuts nothing");
        check(!swiglu_skim(3.f, 0.5f), "a large positive gate (silu 2.86) is kept at tau 0.5");
        check(swiglu_skim(0.05f, 0.1f), "a small positive gate (silu 0.026) is cut at tau 0.1");
    }
    {
        check(!swiglu_skim(-1.f, 0.2f), "g = -1 (silu -0.269): |silu| is compared, so it is kept at tau 0.2");
        check(swiglu_skim(-1.f, 0.3f), "g = -1 (silu -0.269) is cut at tau 0.3: the sign does not matter");
        check(swiglu_skim(-20.f, 1e-6f), "a large negative gate (silu -4e-8) is cut at tau 1e-6");
        check(swiglu_skim(-100.f, 1e-6f), "exp overflows at g = -100 and silu computes as -0: it is cut");
    }
    {
        check(!swiglu_skim(1.f, 0.f), "tau = 0 cuts nothing, here at g = 1");
        check(!swiglu_skim(1.f, -1.f), "a negative tau cuts nothing");
        check(!swiglu_skim(0.f, -1.f), "a negative tau cuts nothing, here at g = 0");
        check(swiglu_skim(3.f, inf), "tau = inf cuts every finite gate");
    }
    {
        const float s = std::fabs(silu(1.f));
        check(!swiglu_skim(1.f, s), "|silu(1)| equal to tau is kept: the threshold is strict");
        check(swiglu_skim(1.f, std::nextafter(s, inf)), "a tau just above |silu(1)| cuts it");
        check(!swiglu_skim(1.f, std::nextafter(s, 0.f)), "a tau just below |silu(1)| keeps it");
    }
    {
        check(!swiglu_skim(nan, 0.5f), "a NaN gate is never cut");
        check(!swiglu_skim(nan, inf), "a NaN gate is never cut, even under tau = inf");
        check(!swiglu_skim(1.f, nan), "a NaN threshold cuts nothing");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
