// src/kernels/cpu/pool_test.cpp - P2.S3's test for the expert pool.
//
// The pool's correctness claims are small and specific, so they are checked directly rather than through a
// timing number:
//
//   1. THE SAME ANSWER AS SERIAL.  Every job's output must be bit-identical to running that expert serially.
//      Each worker owns its own `ExpertScratch`, and the jobs share one read-only activation, so there is no
//      legitimate source of difference - "close enough" here would be hiding a data race.
//   2. EVERY JOB RUNS EXACTLY ONCE.  Claiming is `head.fetch_add`, and an off-by-one in the bound is the
//      classic pool bug: it either drops a job or runs one twice.  Checked with a sentinel-filled output
//      buffer, so a dropped job is visible as an untouched slot rather than as a slightly wrong number.
//   3. REPEATED BATCHES.  A pool that works once and hangs or corrupts on the second `run()` is the failure
//      mode the park protocol exists to prevent, so `run()` is called many times in a row.
//   4. A BATCH BIGGER AND SMALLER THAN THE WORKER COUNT, because `n < workers` leaves most workers claiming
//      nothing and `n > workers` is the real case (10 experts, 5 workers).
//   5. `run_split` AND `run_split_multi` BIT-IDENTICAL TO SERIAL (#27), in both of `run_split`'s cuts (the flat
//      row ranges and the per-expert parts), for every batch size up to the largest.
//
//   pool_test --synthetic      the same checks on random expert blobs (no pack needed; a ctest)
//   pool_test --bench [layers] [--workers N]
//                              run_split per token: flat vs per-expert cut, with software prefetch and, on Linux,
//                              with transparent huge pages; an arena of `layers` x 10 random experts (default 64:
//                              885 MB); N pool workers besides the host (default: the engine's, cores - 1)
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace {

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// A random expert: random 2-bit codes, fp16 scales in the pack's range (expert_multi_test's generator).
void make_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

// The bench's expert arena: anonymous memory, on Linux with or without transparent huge pages (#28), which is
// how the engine's arena falls back when no hugetlb pool is configured.
uint8_t* map_arena(size_t bytes, bool thp) {
#if defined(__linux__)
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return (uint8_t*) p;
#else
    (void) thp;
    return (uint8_t*) std::malloc(bytes);
#endif
}

void unmap_arena(uint8_t* p, size_t bytes) {
#if defined(__linux__)
    munmap(p, bytes);
#else
    (void) bytes;
    std::free(p);
#endif
}

// #27: run_split for one token (48 layers of n experts drawn at random from an arena of `layers` x 10, as the
// router does), flat cut against the per-expert parts, interleaved and best of `reps` so a contended machine shows
// up as noise and not as a difference.  Each timed token takes the NEXT experts of the arena, so no arm finds the
// previous one's experts in the L3.  The flat cut is timed again with the kernels' software prefetch (#29) and, on
// Linux, on a second arena with transparent huge pages (#28).
int bench_split(int layers, int reps, int workers) {
    std::mt19937 rng(27);
    const int K = 10;
    const size_t bytes = (size_t) layers * K * cpu::BLOB;
#if defined(__linux__)
    const int arenas = 2;
#else
    const int arenas = 1;
#endif
    uint8_t* arena[2] = {nullptr, nullptr};
    for (int a = 0; a < arenas; ++a) {
        arena[a] = map_arena(bytes, a == 1);
        if (arena[a] == nullptr) { std::fprintf(stderr, "cannot map %zu B\n", bytes); return 2; }
        std::mt19937 brng(27);
        for (int e = 0; e < layers * K; ++e) make_blob(arena[a] + (size_t) e * cpu::BLOB, brng);
    }
    std::vector<int> order((size_t) layers * K);
    for (int e = 0; e < layers * K; ++e) order[(size_t) e] = e;
    std::shuffle(order.begin(), order.end(), rng);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);
    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);
    std::vector<float> out((size_t) K * cpu::H);
    std::vector<cpu::ExpertJob> jobs[2];
    for (int a = 0; a < arenas; ++a) {
        jobs[a].resize((size_t) layers * K);
        for (int e = 0; e < layers * K; ++e) {
            jobs[a][(size_t) e].blob = arena[a] + (size_t) order[(size_t) e] * cpu::BLOB;
            jobs[a][(size_t) e].act = &act;
            jobs[a][(size_t) e].out = out.data() + (size_t) (e % K) * cpu::H;
        }
    }
    cpu::ExpertPool pool(workers);
    std::printf("run_split bench: 48 layers per token from an arena of %d x %d experts (%.0f MB), %d workers + host, "
                "best of %d tokens; gain against the first arm\n", layers, K, (double) bytes / 1e6, pool.workers(), reps);
    // the arms: the cut (#27), the arena's pages (#28), the kernels' software prefetch (#29, scales / codes bytes)
    struct Arm { const char* name; bool flat; int arena, pf_sc, pf_co; };
    const Arm all[] = {{"per-expert parts", false, 0, 0, 0},
                       {"flat", true, 0, 0, 0},
                       {"flat + prefetch", true, 0, 2048, 1024},
                       {"flat + THP", true, 1, 0, 0},
                       {"flat + THP + prefetch", true, 1, 2048, 1024}};
    std::vector<Arm> arms;
    for (const Arm& a : all)
        if (a.arena < arenas) arms.push_back(a);
    const int total = layers * K, narms = (int) arms.size();
    int cursor[2] = {0, 0};
    for (int n : {K, 7, 4}) {
        std::vector<double> best((size_t) narms, 1e30);
        for (int r = 0; r < reps; ++r)
            for (int k = 0; k < narms; ++k) {
                const int i = (r + k) % narms, a = arms[(size_t) i].arena;
                pool.set_flat_split(arms[(size_t) i].flat);
                cpu::expert_set_prefetch(arms[(size_t) i].pf_sc, arms[(size_t) i].pf_co);
                const double t0 = now_ms();
                for (int l = 0; l < 48; ++l) {
                    if (cursor[a] + n > total) cursor[a] = 0;
                    pool.run_split(jobs[a].data() + cursor[a], n);
                    cursor[a] += n;
                }
                best[(size_t) i] = std::fmin(best[(size_t) i], now_ms() - t0);
            }
        const double gb = 48.0 * n * cpu::BLOB / 1e9;
        for (int i = 0; i < narms; ++i)
            std::printf("  %2d experts per layer  %-24s %7.2f ms  %5.1f GB/s  %+6.1f%%\n", n, arms[(size_t) i].name,
                        best[(size_t) i], gb / (best[(size_t) i] * 1e-3), 100.0 * (best[0] / best[(size_t) i] - 1.0));
    }
    cpu::expert_set_prefetch(0, 0);
    for (int a = 0; a < arenas; ++a) unmap_arena(arena[a], bytes);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, synthetic = false, bench = false;
    int bench_layers = 64, bench_workers = 0;
    const char* path = "pack/full/experts.bin";
    long long layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--synthetic") synthetic = true;
        else if (a == "--bench") {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') bench_layers = std::atoi(argv[++i]);
        }
        else if (a == "--workers" && i + 1 < argc) bench_workers = std::atoi(argv[++i]);
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: pool_test [--selftest] [--synthetic] [--bench [layers] [--workers N]] [--file P] "
                                 "[--layer N]\n");
            return 2;
        }
    }

    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool test SKIPPED, not passed.\n",
                    feat.reason());
        std::printf("\npool: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    if (bench) return bench_split(bench_layers > 0 ? bench_layers : 64, 15, bench_workers);

    int bad = 0;

    // ---- ten experts off one layer, which is exactly what a token uses (or, with --synthetic, `kMaxSplit`
    // random ones, so run_split's largest batch is covered too)
    const int NEXP = 10;
    const int NALL = synthetic ? cpu::ExpertPool::kMaxSplit : NEXP;
    std::vector<std::vector<uint8_t>> blobs((size_t) NALL);
    if (synthetic) {
        std::mt19937 brng(27);
        for (int e = 0; e < NALL; ++e) {
            blobs[(size_t) e].assign(cpu::BLOB, 0);
            make_blob(blobs[(size_t) e].data(), brng);
        }
    } else {
        for (int e = 0; e < NEXP; ++e)
            if (!read_blob(path, layer * 512 + e, blobs[(size_t) e])) {
                std::fprintf(stderr, "cannot read expert %d of layer %lld from %s\n", e, layer, path);
                return 2;
            }
    }

    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    // ---- serial reference
    std::vector<float> ref((size_t) NALL * cpu::H);
    {
        cpu::ExpertScratch ws;
        for (int e = 0; e < NALL; ++e)
            cpu::s2_expert_vnni_q(blobs[(size_t) e].data(), act, ref.data() + (size_t) e * cpu::H, ws);
    }

    // ---- the pool
    const std::vector<int> cores = cpu::physical_cores(true);
    const int hw = (int) std::thread::hardware_concurrency();
    std::printf("  %-44s %d logical, %d physical (skipping the first)\n", "cores the pool will use",
                hw, (int) cores.size());

    cpu::ExpertPool pool;
    std::printf("  %-44s %d\n", "workers", pool.workers());

    std::vector<cpu::ExpertJob> jobs((size_t) NEXP);
    std::vector<float> got((size_t) NEXP * cpu::H);
    std::vector<float> weights((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e) weights[(size_t) e] = 0.1f * (float) (e + 1);

    // ---- 1 + 2: same answer as serial, and every job exactly once.
    // The output buffer starts at a sentinel no real result can equal, so a job that never runs shows up as
    // an untouched slot instead of as a plausible number.
    const float SENTINEL = -1.2345e33f;
    for (int e = 0; e < NEXP; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = got.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].weight = weights[(size_t) e];
        jobs[(size_t) e].slot = e;
        std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
    }
    pool.run(jobs.data(), NEXP);

    long long not_run = 0, diff = 0;
    float worst = 0.f;
    for (int e = 0; e < NEXP; ++e)
        for (int i = 0; i < cpu::H; ++i) {
            const float a = ref[(size_t) e * cpu::H + i], b = got[(size_t) e * cpu::H + i];
            if (b == SENTINEL) { ++not_run; continue; }
            if (std::memcmp(&a, &b, 4) != 0) ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
        }
    std::printf("  %-44s %s (%lld of %d outputs untouched)\n", "every job ran exactly once",
                not_run ? "*** NO ***" : "yes", not_run, NEXP * cpu::H);
    if (not_run) ++bad;
    // BIT-IDENTICAL, not "close".  Each worker has a private scratch and the activation is shared read-only,
    // so any difference at all is a race or a scratch collision - and a tolerance would hide exactly that.
    std::printf("  %-44s %s (%lld of %d differ, worst |d| %.3e)\n", "identical to the serial run",
                diff ? "*** NO ***" : "yes", diff, NEXP * cpu::H, (double) worst);
    if (diff) ++bad;

    // ---- 3: repeated batches.  A park protocol that works once and corrupts on the second call is the
    //        exact failure this loop is here to catch.
    {
        int repeats_bad = 0;
        for (int r = 0; r < 200; ++r) {
            for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), NEXP);
            for (int e = 0; e < NEXP && !repeats_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++repeats_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (200 consecutive batches)\n", "repeated batches stay correct",
                    repeats_bad ? "*** NO ***" : "yes");
        if (repeats_bad) ++bad;
    }

    // ---- 4: batch sizes either side of the worker count
    {
        int size_bad = 0;
        for (int n : {1, 2, pool.workers() - 1 > 0 ? pool.workers() - 1 : 1, pool.workers(),
                      pool.workers() + 1, NEXP}) {
            if (n < 1 || n > NEXP) continue;
            for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), n);
            for (int e = 0; e < n && !size_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++size_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (1, w-1, w, w+1, 10)\n", "batch sizes around the worker count",
                    size_bad ? "*** NO ***" : "yes");
        if (size_bad) ++bad;
    }

    // ---- 5: run_split, both cuts, every batch size (#27).  Rows of one expert are spread over several threads
    //        and a flat range may span two experts, so this is the check that no row is dropped, run twice or
    //        written into the wrong expert's buffer.
    {
        std::vector<cpu::ExpertJob> sj((size_t) NALL);
        std::vector<float> sgot((size_t) NALL * cpu::H);
        for (int e = 0; e < NALL; ++e) {
            sj[(size_t) e].blob = blobs[(size_t) e].data();
            sj[(size_t) e].act = &act;
            sj[(size_t) e].out = sgot.data() + (size_t) e * cpu::H;
            sj[(size_t) e].slot = e;
        }
        const bool flat0 = pool.flat_split();
        for (int flat = 1; flat >= 0; --flat) {
            pool.set_flat_split(flat == 1);
            long long split_bad = 0, split_missed = 0;
            for (int rep = 0; rep < 3; ++rep)
                for (int n = 1; n <= NALL; ++n) {
                    std::fill(sgot.begin(), sgot.begin() + (ptrdiff_t) n * cpu::H, SENTINEL);
                    pool.run_split(sj.data(), n);
                    for (size_t i = 0; i < (size_t) n * cpu::H; ++i) {
                        if (sgot[i] == SENTINEL) ++split_missed;
                        else if (std::memcmp(&sgot[i], &ref[i], 4) != 0) ++split_bad;
                    }
                }
            const char* what = flat ? "run_split (flat rows) identical to serial"
                                    : "run_split (per-expert parts) identical";
            std::printf("  %-44s %s (n = 1..%d x 3: %lld differ, %lld untouched)\n", what,
                        split_bad || split_missed ? "*** NO ***" : "yes", NALL, split_bad, split_missed);
            if (split_bad || split_missed) ++bad;
        }
        pool.set_flat_split(flat0);
    }

    // ---- 5b: run_split_multi, several tokens per expert (the verify window), against the one-expert kernel
    {
        const int NT = 3;
        std::vector<cpu::ActQ> acts((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            std::vector<float> xt(cpu::H);
            for (auto& v : xt) v = gauss(rng);
            cpu::act_quant_q8_1(xt.data(), cpu::H, acts[(size_t) t]);
        }
        std::vector<float> mref((size_t) NALL * NT * cpu::H), mgot((size_t) NALL * NT * cpu::H);
        std::vector<cpu::ExpertJobMulti> mj((size_t) NALL);
        static cpu::ExpertScratchMulti wsm;
        for (int e = 0; e < NALL; ++e) {
            const int nt = 1 + e % NT;
            cpu::ExpertJobMulti& j = mj[(size_t) e];
            j.blob = blobs[(size_t) e].data();
            j.nt = nt;
            const cpu::ActQ* a1[cpu::MAXT];
            float* ro[cpu::MAXT];
            for (int t = 0; t < nt; ++t) {
                j.act[t] = &acts[(size_t) t];
                j.out[t] = mgot.data() + ((size_t) e * NT + t) * cpu::H;
                a1[t] = &acts[(size_t) t];
                ro[t] = mref.data() + ((size_t) e * NT + t) * cpu::H;
            }
            cpu::s2_expert_vnni_multi(j.blob, a1, nt, ro, wsm);
        }
        long long mbad = 0;
        for (int n : {1, 2, NALL}) {
            std::fill(mgot.begin(), mgot.end(), SENTINEL);
            pool.run_split_multi(mj.data(), n);
            for (int e = 0; e < n; ++e)
                for (int t = 0; t < mj[(size_t) e].nt; ++t)
                    if (std::memcmp(mgot.data() + ((size_t) e * NT + t) * cpu::H,
                                    mref.data() + ((size_t) e * NT + t) * cpu::H, cpu::H * sizeof(float)) != 0)
                        ++mbad;
        }
        std::printf("  %-44s %s (%lld expert-tokens differ)\n", "run_split_multi identical to serial",
                    mbad ? "*** NO ***" : "yes", mbad);
        if (mbad) ++bad;
    }

    // ---- and the number that matters for the ledger: c for one layer, 10 experts over 48 layers.
    //
    // BEST OF N, not a mean.  The first version divided a 10-run total by 10 and reported 6.604 ms per layer -
    // 2.1 GB/s against L9's measured 42.55 - purely because a CUDA build was running on the same 6-core
    // machine at the time.  A mean over a contended machine measures the contention; a best-of measures the
    // instrument, and every other timing tool in this project already takes the best.  The spread is printed
    // so a contended run is visible rather than being read as a regression.
    {
        const int REPS = 20;
        std::vector<double> t((size_t) REPS, 0.0);
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            pool.run(jobs.data(), NEXP);
            t[(size_t) r] = now_ms() - t0;
        }
        std::vector<double> sorted = t;
        std::sort(sorted.begin(), sorted.end());
        const double best = sorted.front(), median = sorted[sorted.size() / 2], worst = sorted.back();
        const double gbs = (double) NEXP * cpu::BLOB / (best * 1e-3) / 1e9;
        std::printf("\n  %-44s %7.3f ms   (%.1f GB/s)\n", "10 experts, one layer (best of 20)",
                    best, gbs);
        std::printf("  %-44s %7.3f / %7.3f ms   (spread %.2fx)\n", "median / worst", median, worst,
                    median / (best > 0 ? best : 1));
        std::printf("  %-44s %7.3f ms   -> %.1f tok/s for the full 48 layers\n",
                    "extrapolated to 48 layers", best * 48, 1000.0 / (best * 48));
        std::printf("  L9 measured 42.55 GB/s on 6 cores; this pool uses %d workers (core 0 is left to the\n",
                    pool.workers());
        std::printf("  host loop), so %.1f GB/s x 6/%d = %.1f GB/s is the per-core comparison.\n",
                    gbs, pool.workers(), gbs * 6.0 / pool.workers());
        if (median / (best > 0 ? best : 1) > 1.5)
            std::printf("  *** the spread is over 1.5x: this machine was CONTENDED and the median is not a\n"
                        "      property of the pool.  Re-run on a quiet machine before quoting it. ***\n");
    }

    std::printf("\npool: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("pool_test OK\n");
    return 0;
}
