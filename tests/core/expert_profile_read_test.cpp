// read_expert_profile on damaged files: a truncated header or pair list, a pair out of range, a version 2 file, more
// ranked pairs than slots, and a hostile header whose counts would size a huge allocation.  Each is refused with a
// message naming the cause.  CPU only: no device is touched.
#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
bool has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

// STRP, the five header words, then the given (layer, expert) pairs as uint16
std::string write_file(const std::filesystem::path& dir, const char* name, uint32_t version, uint32_t nl, uint32_t ne, uint32_t want,
                       uint32_t n_ranked, const std::vector<uint16_t>& pairs) {
    const std::string path = (dir / name).string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const uint32_t hdr[5] = {version, nl, ne, want, n_ranked};
    std::fwrite("STRP", 1, 4, f);
    std::fwrite(hdr, 4, 5, f);
    if (!pairs.empty()) std::fwrite(pairs.data(), 2, pairs.size(), f);
    std::fclose(f);
    return path;
}
}  // namespace

int main() {
    using Pair = std::pair<int32_t, int32_t>;
    const uint32_t L = 4, E = 8;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_profile_read_test";
    std::filesystem::create_directories(dir);
    std::vector<Pair> back;
    int64_t slots = 0;
    std::string err;
    auto refused = [&](const std::string& path, const char* cause, const char* what) {
        err.clear();
        const bool no = !strata::core::read_expert_profile(path, L, E, back, slots, err);
        check(no && has(err, "read_expert_profile: ") && has(err, cause), what);
    };

    // a good file loads
    const std::string good = write_file(dir, "good.bin", 1, L, E, 3, 3, {1, 2, 0, 0, 3, 7});
    check(strata::core::read_expert_profile(good, L, E, back, slots, err) && back == std::vector<Pair>{{1, 2}, {0, 0}, {3, 7}} && slots == 3,
          "a well-formed profile loads");

    // truncated: a header cut short, then a pair list cut short
    {
        const std::string p = (dir / "short_header.bin").string();
        std::FILE* f = std::fopen(p.c_str(), "wb");
        std::fwrite("STRP\x01\x00\x00\x00", 1, 8, f);
        std::fclose(f);
        refused(p, "too short", "a file shorter than its header is refused");
    }
    refused(write_file(dir, "short_pairs.bin", 1, L, E, 3, 3, {1, 2, 0, 0}), "truncated", "a pair list shorter than the header says is refused");

    // a pair out of range (layer, then expert)
    refused(write_file(dir, "bad_layer.bin", 1, L, E, 2, 2, {1, 2, (uint16_t) L, 0}), "out of range", "a pair whose layer is out of range is refused");
    refused(write_file(dir, "bad_expert.bin", 1, L, E, 2, 2, {1, 2, 0, (uint16_t) E}), "out of range", "a pair whose expert is out of range is refused");

    // version 2
    refused(write_file(dir, "v2.bin", 2, L, E, 3, 3, {1, 2, 0, 0, 3, 7}), "version 2", "a version 2 profile is refused");

    // more ranked pairs than slots
    refused(write_file(dir, "over_slots.bin", 1, L, E, 2, 3, {1, 2, 0, 0, 3, 7}), "more ranked pairs than slots", "n_ranked above the slot count is refused");

    // hostile counts: refused from the header alone, before any allocation sized by them
    refused(write_file(dir, "huge.bin", 1, L, E, 0xFFFFFFFFu, 0xFFFFFFFFu, {}), "more slots than the model has experts",
            "a header claiming 2^32 - 1 pairs is refused");
    refused(write_file(dir, "just_over.bin", 1, L, E, L * E + 1, L * E + 1, {}), "more slots than the model has experts",
            "a header claiming one pair more than the model has experts is refused");
    {
        std::vector<uint16_t> all;
        for (uint32_t l = 0; l < L; ++l)
            for (uint32_t e = 0; e < E; ++e) {
                all.push_back((uint16_t) l);
                all.push_back((uint16_t) e);
            }
        const std::string p = write_file(dir, "full.bin", 1, L, E, L * E, L * E, all);
        check(strata::core::read_expert_profile(p, L, E, back, slots, err) && back.size() == (size_t) (L * E),
              "a profile ranking every pair of the model loads");
    }

    std::filesystem::remove_all(dir);
    if (fails == 0) std::puts("expert_profile_read_test: OK");
    return fails == 0 ? 0 : 1;
}
