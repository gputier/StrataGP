// tests/core/check.hpp - the failure counter and the one-line check shared by the q36 host tests.
#pragma once

#include <cstdio>

namespace {
int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace
