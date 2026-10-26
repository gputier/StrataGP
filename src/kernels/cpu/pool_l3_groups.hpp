// src/kernels/cpu/pool_l3_groups.hpp - which pool cores sit under which L3 cache, a pure rule with no OS call (tested by
// pool_l3_groups_test.cpp).  On a Ryzen 9 9950X3D one of the two CCDs has the 3D V-Cache (96 MiB of L3 against 32 MiB),
// so the pool says at start which cores are under the big one.  Reporting only: the order and the pinning are untouched.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// One L3 cache as the OS reports it: its size and the logical processors of processor group 0 that share it (bit i is
/// logical processor i).  A cache of another processor group is left out by the caller, as a mask holds 64 of them.
struct L3Group {
    uint64_t size_bytes = 0;
    uint64_t mask = 0;
};

/// The one-line report, largest cache first (equal sizes keep their order): `L3 96 MiB covers cores 16,18; L3 32 MiB
/// covers cores 0-3`.  `pool_cores` holds the first logical processor of each pool core; an ID outside 0-63 is under no
/// group.  Cores are written ascending in the grammar of STRATA_POOL_CORES (pool_core_list.hpp), so a set can be pasted
/// into it.  A group that covers no pool core is left out; empty when none is left.
inline std::string l3_report(std::vector<L3Group> groups, const std::vector<int>& pool_cores) {
    uint64_t pool = 0;
    for (int core : pool_cores)
        if (core >= 0 && core < 64) pool |= uint64_t(1) << core;
    std::stable_sort(groups.begin(), groups.end(),
                     [](const L3Group& x, const L3Group& y) { return x.size_bytes > y.size_bytes; });
    std::string line;
    for (const L3Group& g : groups) {
        const uint64_t m = g.mask & pool;
        if (m == 0) continue;
        constexpr uint64_t kMiB = uint64_t(1) << 20;
        line += line.empty() ? "L3 " : "; L3 ";
        line += g.size_bytes % kMiB == 0 ? std::to_string(g.size_bytes / kMiB) + " MiB"
                                         : std::to_string(g.size_bytes / 1024) + " KiB";
        line += " covers cores ";
        bool first = true;
        for (int i = 0; i < 64;) {
            if (((m >> i) & 1) == 0) {
                ++i;
                continue;
            }
            int j = i;
            while (j + 1 < 64 && ((m >> (j + 1)) & 1) != 0) ++j;
            if (!first) line += ',';
            line += std::to_string(i);
            if (j > i) line += "-" + std::to_string(j);
            first = false;
            i = j + 1;
        }
    }
    return line;
}

}  // namespace strata::kernels::cpu
