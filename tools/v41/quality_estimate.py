#!/usr/bin/env python3
"""quality_estimate.py - the quality verdict of the plan from the `llama-perplexity` logs, with an ESTIMATE of P3.

Inputs (logs of the four runs on the same text, 45 chunks of 2 048 tokens):
  R   the reference, experts in MXFP4 as converted: a `--save-all-logits` run, read from its
      "Final estimate: PPL over N chunks for n_ctx=C = P +/- U" line;
  Q   the Q2_K oracle,
  U2  every routed expert in IQ2_XS,
  U1  every routed expert in IQ1_M: `--kl-divergence` runs against R, read from the "Perplexity statistics" and
      "KL divergence statistics" blocks (PPL of the run and of the base, mean ln(PPL/PPL(base)) with its paired
      uncertainty, KLD mean, median and 99 %, RMS of the probability change, share of identical top tokens).
A log carrying `Oops`, another number of chunks or another context than asked, or a base perplexity that is not R's
(1 % tolerance) is refused: the runs would not be comparable.

The estimate of P3 (hot experts in MXFP4 in VRAM, warm in IQ2_XS in RAM, cold in IQ1_M on NVMe) is a first-order
model, never a measurement.  With the call shares of the tiers (`tier_budget.py` JSON, cross-validation, plan P3):

    ln PPL(P3) = ln PPL(R) + share_ram * (ln PPL(U2) - ln PPL(R)) + share_nvme * (ln PPL(U1) - ln PPL(R))

the VRAM share being at R's type.  It is computed for both cross-validation directions and the higher one is kept
(the less favourable).  Its uncertainty adds the paired uncertainties in quadrature and ignores covariances: it is
an order of magnitude.

Verdict, the plan's quality threshold, written before any measurement and never retouched.  "Passes" means a
perplexity that does not exceed PPL(Q), the oracle measured again on this text.  In this order:
  * U1 passes: "retenu", and the quality is acquired for any bit plan (P3 lies between R and U1);
  * else the estimate within 0.1 of PPL(Q), above or below, or the estimate passing while U1 fails: "zone grise"
    (P3 itself must be built and measured);
  * else "échec".
Verdict words are the plan's, in French.  The exit code is 0 whenever the report is written, the verdict is in the
output (and in the JSON `verdict`); it is 2 for a log that cannot be trusted.

    python3 -I tools/v41/quality_estimate.py --r r.log --q q.log --u2 u2.log --u1 u1.log --budget budget.json
        [--plan P3] [--out quality.json]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

N_CTX = 2048
CHUNKS = 45
GRAY_BAND = 0.1          # |estimate - PPL(Q)| below this is the grey zone (plan threshold)
BASE_TOLERANCE = 0.01    # relative gap allowed between PPL(base) of a KL run and PPL(R)

RETAINED, GRAY, FAILED = "retenu", "zone grise", "échec"

FINAL = re.compile(r"Final estimate: PPL over (\d+) chunks for n_ctx=(\d+) = ([\d.]+) \+/- ([\d.]+)")
# "Name   :  value ± unc %": the statistics lines of the KL run
STAT = re.compile(r"^(?P<key>[^:\n]+?)\s*:\s*(?P<val>-?\d+(?:\.\d+)?)(?:\s*(?:Â?±|\ufffd)\s*(?P<unc>\d+(?:\.\d+)?))?\s*%?\s*$",
                  re.MULTILINE)
# statistic -> key of the log line, once the spaces are collapsed and the non-ASCII characters dropped (a log
# redirected by Windows PowerShell may have them re-encoded)
KL_KEYS = {
    "ppl": "Mean PPL(Q)",
    "ppl_base": "Mean PPL(base)",
    "ln_ratio": "Mean ln(PPL(Q)/PPL(base))",
    "kld_mean": "Mean KLD",
    "kld_median": "Median KLD",
    "kld_p99": "99.0% KLD",
    "rms_dp_percent": "RMS Δp",
    "same_top_percent": "Same top p",
}


class LogError(ValueError):
    """A log that cannot be trusted."""


def read_log(path):
    """The text of a log: UTF-16 when it starts with a byte order mark (Windows PowerShell redirection), else UTF-8."""
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def ascii_key(key):
    """A statistic's name with spaces collapsed and non-ASCII characters dropped."""
    return re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7f]+", "", key)).strip()


def check_clean(text, name):
    if "Oops" in text:
        raise LogError(f"{name}: the log carries 'Oops'")


def parse_base(text, name="R", n_ctx=N_CTX, chunks=CHUNKS):
    """(ppl, uncertainty) of a perplexity run's final line."""
    check_clean(text, name)
    found = FINAL.findall(text)
    if not found:
        raise LogError(f"{name}: no 'Final estimate: PPL over ...' line")
    n, ctx, ppl, unc = found[-1]
    if int(n) != chunks or int(ctx) != n_ctx:
        raise LogError(f"{name}: {n} chunks at n_ctx={ctx}, expected {chunks} at {n_ctx}")
    return {"ppl": float(ppl), "ppl_unc": float(unc), "chunks": int(n), "n_ctx": int(ctx)}


def parse_kl(text, name):
    """The statistics of a `--kl-divergence` run as {statistic: value} and {statistic + '_unc': uncertainty}."""
    check_clean(text, name)
    lines = {ascii_key(m["key"]): (float(m["val"]), float(m["unc"]) if m["unc"] else None)
             for m in STAT.finditer(text)}
    out = {}
    for stat, key in KL_KEYS.items():
        if ascii_key(key) not in lines:
            raise LogError(f"{name}: no '{key}' line (is it a --kl-divergence log?)")
        out[stat], unc = lines[ascii_key(key)]
        if unc is not None:
            out[stat + "_unc"] = unc
    return out


def verdict(ppl_q, ppl_u1, estimate, band=GRAY_BAND):
    """The plan's quality verdict from PPL(Q), PPL(U1) and the estimate of PPL(P3), and whether the quality is
    acquired for any bit plan."""
    u1_passes = ppl_u1 <= ppl_q
    est_passes = estimate <= ppl_q
    if u1_passes:
        return RETAINED, True, "U1 does not exceed PPL(Q): P3 lies between R and U1"
    if abs(estimate - ppl_q) < band:
        return GRAY, False, f"the estimate is within {band:g} of PPL(Q)"
    if est_passes:
        return GRAY, False, "the estimate passes while U1 fails"
    return FAILED, False, f"the estimate exceeds PPL(Q) by {estimate - ppl_q:.4f} and U1 fails"


def shares(budget_path, plan):
    """{direction: {'vram', 'ram', 'nvme'}} cross-validation call shares of `plan` in a tier_budget.py JSON."""
    report = json.loads(Path(budget_path).read_text())
    for p in report["plans"]:
        if p["name"] == plan:
            return {d: p["cross_validation"][d]["call_share"] for d in ("A_to_B", "B_to_A")}
    raise LogError(f"{budget_path}: no plan {plan}")


def estimate_p3(ppl_r, ppl_r_unc, d_u2, d_u2_unc, d_u1, d_u1_unc, share):
    """(PPL(P3) estimate, its uncertainty) for one direction's call shares; `d_*` are ln(PPL(U)/PPL(R))."""
    ln_est = math.log(ppl_r) + share["ram"] * d_u2 + share["nvme"] * d_u1
    ln_unc = math.sqrt((ppl_r_unc / ppl_r) ** 2 + (share["ram"] * d_u2_unc) ** 2 + (share["nvme"] * d_u1_unc) ** 2)
    est = math.exp(ln_est)
    return est, est * ln_unc


def analyse(logs, budget_path, plan="P3", n_ctx=N_CTX, chunks=CHUNKS):
    """The full report from the texts of the four logs {'R', 'Q', 'U2', 'U1'}."""
    r = parse_base(logs["R"], "R", n_ctx, chunks)
    runs = {name: parse_kl(logs[name], name) for name in ("Q", "U2", "U1")}
    for name, run in runs.items():
        gap = abs(run["ppl_base"] - r["ppl"]) / r["ppl"]
        if gap > BASE_TOLERANCE:
            raise LogError(f"{name}: PPL(base) {run['ppl_base']} is not R's {r['ppl']} ({gap:.2%} apart)")
    by_direction = {}
    for direction, share in shares(budget_path, plan).items():
        est, unc = estimate_p3(r["ppl"], r["ppl_unc"], runs["U2"]["ln_ratio"], runs["U2"]["ln_ratio_unc"],
                               runs["U1"]["ln_ratio"], runs["U1"]["ln_ratio_unc"], share)
        by_direction[direction] = {"call_share": share, "estimate": est, "estimate_unc": unc}
    worst = max(by_direction, key=lambda d: by_direction[d]["estimate"])
    est = by_direction[worst]["estimate"]
    word, acquired, why = verdict(runs["Q"]["ppl"], runs["U1"]["ppl"], est)
    return {
        "R": r,
        "runs": runs,
        "estimate_p3": {"note": "estimation at first order, not a measurement", "plan": plan,
                        "less_favourable_direction": worst, "ppl": est,
                        "ppl_unc": by_direction[worst]["estimate_unc"], "by_direction": by_direction},
        "threshold": {"ppl_q": runs["Q"]["ppl"], "grey_band": GRAY_BAND},
        "verdict": word,
        "quality_acquired_for_any_bit_plan": acquired,
        "reason": why,
    }


def format_report(rep):
    r, runs = rep["R"], rep["runs"]
    lines = [f"R   PPL {r['ppl']:.4f} +/- {r['ppl_unc']:.5f}  ({r['chunks']} chunks, n_ctx {r['n_ctx']})"]
    for name in ("Q", "U2", "U1"):
        x = runs[name]
        lines.append(f"{name:<3} PPL {x['ppl']:.4f} +/- {x['ppl_unc']:.5f}  ln(PPL/PPL(R)) {x['ln_ratio']:.5f} +/- "
                     f"{x['ln_ratio_unc']:.5f}  KLD mean {x['kld_mean']:.5f} median {x['kld_median']:.5f} "
                     f"99% {x['kld_p99']:.5f}  RMS dp {x['rms_dp_percent']:.3f} %  same top {x['same_top_percent']:.3f} %")
    e = rep["estimate_p3"]
    lines.append(f"{e['plan']} estimation (first order, never a measurement): PPL {e['ppl']:.4f} +/- "
                 f"{e['ppl_unc']:.4f}, less favourable direction {e['less_favourable_direction']}")
    for d, v in e["by_direction"].items():
        s = v["call_share"]
        lines.append(f"  {d}: shares vram {s['vram']:.4f} ram {s['ram']:.4f} nvme {s['nvme']:.4f} -> estimate "
                     f"{v['estimate']:.4f}")
    lines.append(f"threshold: PPL(Q) {rep['threshold']['ppl_q']:.4f}, grey band {rep['threshold']['grey_band']:g}")
    lines.append(f"verdict: {rep['verdict']} ({rep['reason']})")
    if rep["quality_acquired_for_any_bit_plan"]:
        lines.append("quality acquired for any bit plan")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("r", "q", "u2", "u1"):
        ap.add_argument(f"--{name}", required=True, help=f"llama-perplexity log of {name.upper()}")
    ap.add_argument("--budget", required=True, help="JSON of tier_budget.py (call shares of the plan)")
    ap.add_argument("--plan", default="P3")
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--chunks", type=int, default=CHUNKS)
    ap.add_argument("--out", help="JSON report")
    args = ap.parse_args(argv)
    logs = {name: read_log(getattr(args, name.lower())) for name in ("R", "Q", "U2", "U1")}
    try:
        report = analyse(logs, args.budget, args.plan, args.n_ctx, args.chunks)
    except LogError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
