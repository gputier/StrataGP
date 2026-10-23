// tools/qwen35/logit_cmp.cpp - parity.py's top-1 / first-divergence report and kl.py's teacher-forced KL on two full-logits
// files ([step][vocab] f32, as --dump-full-logits writes them), for a box without numpy (the Windows GPU box).
//
//   cl /EHsc /O2 /std:c++17 logit_cmp.cpp   (or g++ -O2 -std=c++17 logit_cmp.cpp -o logit_cmp)
//   logit_cmp --ref FILE --test FILE --vocab 248320 [--steps N] [--stop-at-divergence] [--per-step]
//
// --ref is P (llama.cpp or the fp32 cache), --test is Q.  Reported: the first top-1 disagreement with both engines'
// top-1 / top-2 margins, the top-1 agreement, the rows that are bit-identical, and KL(P || Q) per step (mean, median, p99,
// max); --per-step adds one line per step.  --stop-at-divergence keeps only the steps before the first top-1 disagreement (free-running greedy: after it the
// two engines read different tokens).
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
struct Row { int a = 0, c = -1; float va = 0, vc = 0; };

Row top2(const float* x, int v) {
    Row r;
    for (int j = 1; j < v; ++j) if (x[j] > x[r.a]) r.a = j;
    for (int j = 0; j < v; ++j) if (j != r.a && (r.c < 0 || x[j] > x[r.c])) r.c = j;
    r.va = x[r.a];
    r.vc = x[r.c];
    return r;
}

void log_softmax(const float* x, int v, std::vector<double>& out) {
    double m = x[0];
    for (int j = 1; j < v; ++j) m = std::max(m, (double) x[j]);
    double s = 0;
    for (int j = 0; j < v; ++j) s += std::exp((double) x[j] - m);
    const double l = m + std::log(s);
    out.resize(v);
    for (int j = 0; j < v; ++j) out[j] = (double) x[j] - l;
}

std::vector<float> slurp(const std::string& p, int v) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    std::error_code ec;
    const std::uintmax_t bytes = std::filesystem::file_size(p, ec);
    if (ec) { std::fprintf(stderr, "cannot size %s: %s\n", p.c_str(), ec.message().c_str()); std::exit(2); }
    const std::uintmax_t row = 4 * (std::uintmax_t) v;
    if (bytes % row) {
        std::fprintf(stderr, "%s is %ju bytes, not a multiple of one logits row (%d floats, %ju bytes): truncated or not a logits file\n",
                     p.c_str(), bytes, v, row);
        std::exit(2);
    }
    std::vector<float> d(bytes / 4);
    if (std::fread(d.data(), 4, d.size(), f) != d.size()) { std::fprintf(stderr, "short read %s\n", p.c_str()); std::exit(2); }
    std::fclose(f);
    return d;
}

// A NaN or infinite logit would make its step's KL 0 (NaN fails the e > 0 test) and count as a top-1 agreement: refuse it.
void check_finite(const std::vector<float>& d, int v, int n, const char* role, const std::string& p) {
    for (size_t i = 0; i < (size_t) n * (size_t) v; ++i) {
        if (!std::isfinite(d[i])) {
            std::fprintf(stderr, "%s %s: step %zu, logit %zu is %s: refusing non-finite logits\n", role, p.c_str(), i / (size_t) v,
                         i % (size_t) v, std::isnan(d[i]) ? "NaN" : "infinite");
            std::exit(2);
        }
    }
}

// --vocab must be a whole number of at least 2: top2 needs two entries.
int parse_vocab(const char* s) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 2 || v > INT_MAX) {
        std::fprintf(stderr, "--vocab must be an integer of at least 2, got '%s'\n", s);
        std::exit(2);
    }
    return (int) v;
}

// --steps must be a whole number, 0 or more: 0 keeps every step, as when it is absent.
int parse_steps(const char* s) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0 || v > INT_MAX) {
        std::fprintf(stderr, "--steps must be a non-negative integer, got '%s'\n", s);
        std::exit(2);
    }
    return (int) v;
}
}  // namespace

int main(int argc, char** argv) {
    std::string ref, test;
    int V = 248320, max_steps = -1;
    bool stop = false, per_step = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--stop-at-divergence") stop = true;
        else if (a == "--per-step") per_step = true;
        else if (i + 1 < argc && a == "--ref") ref = argv[++i];
        else if (i + 1 < argc && a == "--test") test = argv[++i];
        else if (i + 1 < argc && a == "--vocab") V = parse_vocab(argv[++i]);
        else if (i + 1 < argc && a == "--steps") max_steps = parse_steps(argv[++i]);
        else { std::fprintf(stderr, "usage: logit_cmp --ref F --test F --vocab N [--steps N] [--stop-at-divergence] [--per-step]\n"); return 2; }
    }
    const std::vector<float> P = slurp(ref, V), Q = slurp(test, V);
    const int steps_p = (int) (P.size() / (size_t) V), steps_q = (int) (Q.size() / (size_t) V);
    int n = std::min(steps_p, steps_q);
    if (max_steps > 0) n = std::min(n, max_steps);
    if (steps_p != steps_q)
        std::fprintf(stderr, "warning: ref %s has %d steps (%zu bytes), test %s has %d steps (%zu bytes): comparing the first %d\n",
                     ref.c_str(), steps_p, P.size() * 4, test.c_str(), steps_q, Q.size() * 4, n);
    check_finite(P, V, n, "ref", ref);
    check_finite(Q, V, n, "test", test);
    std::vector<double> lp, lq, kls;
    int agree = 0, ident = 0, first = -1;
    double maxabs = 0;
    for (int s = 0; s < n; ++s) {
        const float *p = &P[(size_t) s * V], *q = &Q[(size_t) s * V];
        const Row rp = top2(p, V), rq = top2(q, V);
        if (rp.a != rq.a && first < 0) {
            first = s;
            std::printf("first divergence at step %d: ref top-1 %d (margin %.4f, top-2 %d), test top-1 %d (margin %.4f, top-2 %d)\n", s,
                        rp.a, rp.va - rp.vc, rp.c, rq.a, rq.va - rq.vc, rq.c);
            if (stop) break;
        }
        log_softmax(p, V, lp);
        log_softmax(q, V, lq);
        double kl = 0;
        for (int j = 0; j < V; ++j) {
            const double e = std::exp(lp[j]);
            if (e > 0) kl += e * (lp[j] - lq[j]);
        }
        kls.push_back(kl);
        if (per_step)
            std::printf("step %d: ref %d (margin %.4f) test %d (margin %.4f) KL %.6f\n", s, rp.a, rp.va - rp.vc, rq.a, rq.va - rq.vc, kl);
        agree += rp.a == rq.a;
        ident += std::memcmp(p, q, (size_t) V * 4) == 0;
        for (int j = 0; j < V; ++j) maxabs = std::max(maxabs, (double) std::fabs(p[j] - q[j]));
    }
    n = (int) kls.size();
    std::vector<double> sorted = kls;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (double k : kls) sum += k;
    std::printf("steps %d, top-1 agree %d (%.4f), identical rows %d, max |logit diff| %.5f\n", n, agree, n ? (double) agree / n : 0.0,
                ident, maxabs);
    if (n)
        std::printf("KL(ref||test): mean %.6f median %.6f p99 %.5f max %.5f\n", sum / n, sorted[n / 2],
                    sorted[std::min(n - 1, (int) (0.99 * n))], sorted[n - 1]);
    return 0;
}
