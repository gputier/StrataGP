// src/kernels/cpu/pool_core_list.hpp - the grammar of STRATA_POOL_CORES, a pure rule with no OS call (tested by
// pool_core_list_test.cpp).  The list names CPU IDs in the order the workers take them: `0-7,16-23`.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// Parses `v` into `out`: integers and ranges `a-b` separated by commas, spaces and tabs tolerated.  Refused, with a
/// reason in `*err` (when given) and `out` untouched: an empty list, a missing or non-numeric entry, a backwards range
/// (`7-0`), a CPU ID listed twice, and an ID above `kMaxCpuId`.  Order is kept: `3,1-2` gives 3, 1, 2.
inline bool parse_core_list(const char* v, std::vector<int>& out, std::string* err) {
    constexpr int kMaxCpuId = (1 << 20) - 1;   // far above any machine; a larger ID cannot be a CPU
    auto fail = [err](const std::string& why) {
        if (err != nullptr) *err = why;
        return false;
    };
    const char* p = v;
    auto skip_blanks = [&p] {
        while (*p == ' ' || *p == '\t') ++p;
    };
    auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
    auto read_id = [&](int& id, std::string* why) {
        if (!is_digit(*p)) {
            *why = *p == '\0' ? std::string("the list ends where a CPU number was expected")
                              : std::string("'") + *p + "' is not a CPU number";
            return false;
        }
        id = 0;
        while (is_digit(*p)) {
            id = id * 10 + (*p - '0');
            if (id > kMaxCpuId) {
                *why = "CPU number above " + std::to_string(kMaxCpuId) + " is not a CPU";
                return false;
            }
            ++p;
        }
        return true;
    };

    std::vector<int> parsed;
    std::string why;
    skip_blanks();
    if (*p == '\0') return fail("the list is empty");
    for (;;) {
        int first = 0;
        skip_blanks();
        if (!read_id(first, &why)) return fail(why);
        int last = first;
        skip_blanks();
        if (*p == '-') {
            ++p;
            skip_blanks();
            if (!read_id(last, &why)) return fail(why);
            if (last < first) return fail("range " + std::to_string(first) + "-" + std::to_string(last) + " runs backwards");
            skip_blanks();
        }
        for (int id = first; id <= last; ++id) parsed.push_back(id);
        if (*p == '\0') break;
        if (*p != ',') return fail(std::string("'") + *p + "' follows a CPU number, a comma was expected");
        ++p;
    }

    std::vector<int> sorted(parsed);
    std::sort(sorted.begin(), sorted.end());
    const auto twice = std::adjacent_find(sorted.begin(), sorted.end());
    if (twice != sorted.end()) return fail("CPU " + std::to_string(*twice) + " is listed twice");
    out = parsed;
    return true;
}

}  // namespace strata::kernels::cpu
