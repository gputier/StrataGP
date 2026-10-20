// tests/core/rope_yarn_reference_test.cpp - the YaRN angles of rope_scaling.hpp (what strata-q36's k_attn_prep takes) against
// transformers' _compute_yarn_parameters (modeling_rope_utils.py), transcribed in float64: no model, no GPU.
//
// Ornith-1.5-122B's setting: factor 2, original context 262144, beta_fast 32, beta_slow 1, rope dim 64, base 1e7.  For every
// rotary pair and a set of positions the reference is cos / sin of position * inv_freq (inv_freq from the interpolation /
// extrapolation blend) times attention_factor = 0.1 ln(factor) + 1; the helper gets the same angle rounded to float32 (the
// kernel's own input), so the error reported is the helper's arithmetic and not the float32 rounding of position * inv_freq.
#include "strata/kernels/rope_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// transformers' find_correction_dim, float64
double corr_dim(double rot, double dim, double base, double max_pos) {
    return dim * std::log(max_pos / (rot * 2.0 * 3.14159265358979323846)) / (2.0 * std::log(base));
}
}  // namespace

int main() {
    const int dim = 64;
    const double base = 1e7, factor = 2.0, beta_fast = 32.0, beta_slow = 1.0, orig = 262144.0;

    // find_correction_range with truncate = True
    double low = std::floor(corr_dim(beta_fast, dim, base, orig)), high = std::ceil(corr_dim(beta_slow, dim, base, orig));
    low = std::max(low, 0.0);
    high = std::min(high, (double) dim - 1);
    const double attention_factor = 0.1 * std::log(factor) + 1.0;

    using namespace strata::kernels;
    RopeScaling cfg;
    cfg.type = RopeScalingType::YaRN;
    cfg.freq_base = base;
    cfg.factor = factor;
    cfg.orig_ctx = orig;
    cfg.ext_factor = 1.0;   // what strata-q36 resolves for yarn
    std::printf("rope_yarn_reference_test: factor %g, original context %.0f, betas %g / %g, dim %d, base %g\n", factor, orig,
                beta_fast, beta_slow, dim, base);
    std::printf("  reference correction range [%g, %g], attention factor %.9f\n", low, high, attention_factor);

    check(rope_scaling_invalid(cfg) == nullptr, "the resolved configuration is valid");
    double cd[2];
    cfg.corr_dims(dim, cd);
    check(cd[0] == low && cd[1] == high, "corr_dims equal transformers' find_correction_range");
    check(std::fabs(cfg.mscale() - attention_factor) < 1e-15, "mscale() equals transformers' attention_factor");

    const RopeKernelArgs a = cfg.kernel_args(dim);
    const std::vector<long> positions = {0, 1, 2, 17, 255, 1000, 4096, 32768, 65536, 131072, 262143, 400000};
    double max_err = 0.0, max_err_small = 0.0;
    int worst_pair = -1;
    long worst_pos = -1;
    bool within = true;
    for (int p = 0; p < dim / 2; ++p) {
        const double pos_freq = std::pow(base, (2.0 * p) / dim);
        const double extrap = 1.0 / pos_freq, interp = 1.0 / (factor * pos_freq);
        // linear_ramp_factor over the pair index, then inv_freq = interp * (1 - extrap_factor) + extrap * extrap_factor
        const double ramp = std::min(std::max((p - low) / (high - low), 0.0), 1.0);
        const double extrap_factor = 1.0 - ramp;
        const double inv_freq = interp * (1.0 - extrap_factor) + extrap * extrap_factor;
        for (const long pos : positions) {
            const float theta_extrap = (float) ((double) pos * extrap);   // the kernel's rp * base^(-2p / n_rot), float32
            // the reference at the angle the helper really gets: the blended inv_freq applied to that extrapolated angle
            const double theta = (double) theta_extrap * (inv_freq / extrap);
            float cs, sn;
            rope_scaled_angle(theta_extrap, a.freq_scale, a.corr_low, a.corr_high, a.ext_factor, a.attn_factor, p, cs, sn);
            const double e = std::max(std::fabs(std::cos(theta) * attention_factor - cs),
                                      std::fabs(std::sin(theta) * attention_factor - sn));
            if (e > max_err) {
                max_err = e;
                worst_pair = p;
                worst_pos = pos;
            }
            if (pos <= 4096) max_err_small = std::max(max_err_small, e);
            // the angle itself is a float32 product: its rounding error scales with its size
            if (e > 2e-6 + 4.0 * 5.97e-8 * std::fabs(theta)) within = false;
        }
    }
    std::printf("  max abs error of cos / sin: %.3e (pair %d, position %ld); positions <= 4096 only: %.3e\n", max_err, worst_pair,
                worst_pos, max_err_small);
    check(within, "every pair and position within 2e-6 + 4 ulp of the angle");
    check(max_err_small < 5e-6, "positions <= 4096: max abs error below 5e-6");

    const RopeKernelArgs id = RopeScaling{}.kernel_args(dim);
    check(id.freq_scale == 1.0f && id.ext_factor == 0.0f && id.attn_factor == 1.0f,
          "no scaling: the identity constants (the kernel keeps sincosf)");

    std::printf(g_fail ? "FAILED: %d\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
