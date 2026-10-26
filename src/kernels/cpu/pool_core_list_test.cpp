// pool_core_list_test - the STRATA_POOL_CORES grammar (pool_core_list.hpp), no OS call, no topology.
#include "pool_core_list.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::kernels::cpu::parse_core_list;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "pool_core_list_test: FAILED: %s\n", what);
}

// A list that must parse to `want`, in this order.
void expect_parses(const char* text, const std::vector<int>& want) {
    std::vector<int> got;
    std::string err;
    const bool ok = parse_core_list(text, got, &err);
    const std::string what = std::string("'") + text + "' parses";
    expect(ok && got == want, what.c_str());
    if (!ok) std::fprintf(stderr, "  error: %s\n", err.c_str());
}

// A list that must be refused, with `out` left as it was and a reason given.
void expect_refused(const char* text) {
    std::vector<int> got{-7};
    std::string err;
    const bool ok = parse_core_list(text, got, &err);
    const std::string what = std::string("'") + text + "' is refused";
    expect(!ok && got == std::vector<int>{-7} && !err.empty(), what.c_str());
}

}  // namespace

int main() {
    // Plain numbers and ranges, kept in the order written.
    expect_parses("0", {0});
    expect_parses("5", {5});
    expect_parses("0-7,16-23", {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23});
    expect_parses("3,1-2", {3, 1, 2});
    expect_parses("9,8,7", {9, 8, 7});
    expect_parses("4-4", {4});
    expect_parses("10-12,0", {10, 11, 12, 0});

    // Spaces and tabs are tolerated around numbers, dashes and commas, and at both ends.
    expect_parses("  0-7 , 16 - 23  ", {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23});
    expect_parses("\t8\t-\t15\t", {8, 9, 10, 11, 12, 13, 14, 15});
    expect_parses("1,2 ,3", {1, 2, 3});

    // Refused: nothing to parse.
    expect_refused("");
    expect_refused("   ");

    // Refused: a stray or missing comma, or a missing number.
    expect_refused(",0");
    expect_refused("0,");
    expect_refused("0,,1");
    expect_refused("1 2");
    expect_refused("0-");
    expect_refused("-3");
    expect_refused("0-7-9");

    // Refused: not a CPU number.
    expect_refused("a");
    expect_refused("0,x");
    expect_refused("0.5");
    expect_refused("+3");
    expect_refused("-1-3");

    // Refused: a backwards range.
    expect_refused("7-0");
    expect_refused("0-3,9-8");

    // Refused: a CPU listed twice, within a range, across two entries, or inside a range that overlaps.
    expect_refused("2,2");
    expect_refused("0-3,2");
    expect_refused("0-7,4-9");
    expect_refused("5-5,5");

    // Refused: an ID that cannot be a CPU.
    expect_refused("99999999999");
    expect_refused("1048576");

    // The largest accepted ID, and an error text is optional.
    expect_parses("1048575", {1048575});
    std::vector<int> no_err_out;
    expect(!parse_core_list("7-0", no_err_out, nullptr) && no_err_out.empty(), "a refusal with no err pointer");

    if (failures == 0) std::printf("pool_core_list_test: ok\n");
    return failures == 0 ? 0 : 1;
}
