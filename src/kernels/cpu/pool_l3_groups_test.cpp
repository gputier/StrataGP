// pool_l3_groups_test - the L3 report of the pool cores (pool_l3_groups.hpp), no OS call, no topology.
#include "pool_core_list.hpp"
#include "pool_l3_groups.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using strata::kernels::cpu::L3Group;
using strata::kernels::cpu::l3_report;
using strata::kernels::cpu::parse_core_list;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "pool_l3_groups_test: FAILED: %s\n", what);
}

constexpr uint64_t kMiB = uint64_t(1) << 20;

void expect_report(const std::vector<L3Group>& groups, const std::vector<int>& pool, const std::string& want,
                   const char* what) {
    expect(l3_report(groups, pool) == want, what);
}

}  // namespace

int main() {
    // Two asymmetric CCDs, the 9950X3D: 32 logical processors, the first thread of each core is even.  The small cache
    // comes first in the OS's list, and the report puts the big one first.
    const std::vector<L3Group> ccds = {{32 * kMiB, 0x0000FFFFull}, {96 * kMiB, 0xFFFF0000ull}};
    const std::vector<int> all_cores = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    expect_report(ccds, all_cores,
                  "L3 96 MiB covers cores 16,18,20,22,24,26,28,30; L3 32 MiB covers cores 0,2,4,6,8,10,12,14",
                  "two asymmetric CCDs, big cache first");

    // The pool's order (STRATA_POOL_CORES can reorder it) does not reach the report, which writes the cores ascending.
    expect_report(ccds, {18, 2, 16, 0}, "L3 96 MiB covers cores 16,18; L3 32 MiB covers cores 0,2", "cores sorted");

    // Equal sizes keep the order of the groups.
    expect_report({{32 * kMiB, 0x3}, {32 * kMiB, 0xC}}, {0, 2}, "L3 32 MiB covers cores 0; L3 32 MiB covers cores 2",
                  "equal sizes keep their order");

    // One group: runs are written a-b, and the listed set reads back through the STRATA_POOL_CORES grammar.
    const std::string one = l3_report({{32 * kMiB, 0xFF}}, {0, 1, 2, 3, 5, 7});
    expect(one == "L3 32 MiB covers cores 0-3,5,7", "one group, runs written a-b");
    std::vector<int> back;
    expect(parse_core_list(one.substr(one.rfind(' ') + 1).c_str(), back, nullptr) &&
               back == std::vector<int>({0, 1, 2, 3, 5, 7}),
           "the listed set parses as STRATA_POOL_CORES");

    // Cores under no group, an ID outside 0-63 and a negative ID included, are left out; so is a group with no core.
    expect_report(ccds, {0, 40, 64, -1, 16}, "L3 96 MiB covers cores 16; L3 32 MiB covers cores 0",
                  "cores under no group");
    expect_report({{32 * kMiB, 0x1}}, {2, 3}, "", "no report when no core is covered");

    // An empty mask is left out; an empty pool or no group gives nothing to report.
    expect_report({{96 * kMiB, 0}, {32 * kMiB, 0xF}}, {0, 1}, "L3 32 MiB covers cores 0-1", "an empty mask");
    expect_report(ccds, {}, "", "an empty pool");
    expect_report({}, all_cores, "", "no group");

    // A core under two groups is in both, and the top bit of the mask is core 63.
    expect_report({{8 * kMiB, 0x3}, {16 * kMiB, 0x2}}, {1}, "L3 16 MiB covers cores 1; L3 8 MiB covers cores 1",
                  "a core under two groups");
    expect_report({{4 * kMiB, uint64_t(1) << 63}}, {63, 62}, "L3 4 MiB covers cores 63", "core 63");

    // A size that is not a whole number of MiB is written in KiB.
    expect_report({{1536 * 1024, 0x1}}, {0}, "L3 1536 KiB covers cores 0", "a size in KiB");

    if (failures == 0) std::printf("pool_l3_groups_test: ok\n");
    return failures == 0 ? 0 : 1;
}
