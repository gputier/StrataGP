// tests/core/q36_pcie_option_test.cpp - strata-q36's --pcie-frac (src/qwen35/q36_pcie.hpp), no GPU: without the option the PCIe
// path is off, "auto" is the opt-in to the probed share, a number F > 0 fixes it, 0 turns it off, and anything else is refused.
#include "check.hpp"
#include "q36_pcie.hpp"

#include <cstdio>

int main() {
    using strata::q36::parse_pcie_option;
    using strata::q36::PcieMode;
    using strata::q36::PcieOption;
    std::printf("parse_pcie_option:\n");
    {
        const PcieOption d;
        check(d.mode == PcieMode::Default && d.frac == 0.0 && !d.on(), "no option: the PCIe path is off, no share");
        PcieOption o;
        check(parse_pcie_option("auto", &o) && o.mode == PcieMode::Auto && o.on(), "auto: the opt-in to the probed share");
        check(parse_pcie_option("0.55", &o) && o.mode == PcieMode::Fixed && o.frac == 0.55 && o.on(), "0.55: a fixed share");
        check(parse_pcie_option("1", &o) && o.mode == PcieMode::Fixed && o.frac == 1.0, "1: a fixed share");
        check(parse_pcie_option("0", &o) && o.mode == PcieMode::Off && !o.on(), "0: off, asked for");
        check(parse_pcie_option("0.0", &o) && o.mode == PcieMode::Off, "0.0 is off as well");
        PcieOption keep{PcieMode::Fixed, 0.3};
        bool refused = true;
        for (const char* bad : {"", "abc", "Auto", "0.3x", " 0.3", "-0.1", "-1", "nan", "inf", "auto ", "0x8", "1e0", ".", "1.2.3"})
            refused = refused && !parse_pcie_option(bad, &keep);
        check(refused, "empty, a typo, a negative value, nan, inf, hex, an exponent, a lone dot, two dots and text after the value are refused");
        check(keep.mode == PcieMode::Fixed && keep.frac == 0.3, "a refused value leaves the option as it was");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
