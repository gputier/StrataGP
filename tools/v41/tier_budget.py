#!/usr/bin/env python3
"""tier_budget.py - memory-tier budget of routed experts from the measured call counts (stats.json).

Reads the output of tools/v41/expert_stats.py (counts per layer and per expert, for half A, half B and the total).
The placement unit is the (layer, expert) pair (40 x 384 = 15 360).  For every bit plan (a type per tier) the pairs
are ranked by call count, the VRAM budget is filled first with the "hot" type, then the RAM budget with the "warm"
type, and the rest goes to NVMe at the "cold" type.  The placement is static: no dynamic cache is assumed, a pair
sits in one tier for the whole run.

Per plan it computes the pairs and the bytes per tier, the share of the calls served by each tier, the bytes read per
token on NVMe (routed experts, plus the engram tables on a separate line) and in RAM, the token/s ceilings those
bytes impose at the given bandwidths, and a verdict against the NVMe threshold written in the plan.

Cross-validation: the ranking is learned on one half and the traffic measured on the other, in both directions.  The
verdict is taken on the less favourable direction, never on the total, which is learned and measured on the same data.

All sizes are in bytes and all Go are 10^9 bytes.

    python3 -I tools/v41/tier_budget.py --stats stats.json --out budget.json
        [--vram-gb 26] [--ram-gb 100] [--plan NAME=hot,warm,cold ...]
"""
import argparse
import json
import sys
from pathlib import Path

GB = 10**9
PARAMS_PER_EXPERT = 3 * 2304 * 5120       # gate, up, down

# type -> (weights per block, bytes per block), from ggml-common.h (fp4 native: e8m0 scale per 32 weights)
BLOCKS = {
    "fp4": (32, 17),
    "q4_k": (256, 144),
    "iq3_xxs": (256, 98),
    "iq2_xs": (256, 74),
    "iq1_m": (256, 56),
}

# plan name -> (hot, warm, cold)
DEFAULT_PLANS = {
    "P1": ("fp4", "iq3_xxs", "iq2_xs"),
    "P2": ("q4_k", "iq2_xs", "iq1_m"),
    "P3": ("fp4", "iq2_xs", "iq1_m"),
}
TIERS = ("hot", "warm", "cold")
TIER_MEDIUM = {"hot": "vram", "warm": "ram", "cold": "nvme"}

# threshold written in the plan before any measurement, never retouched
NVME_BYTES_PER_TOKEN_MAX = 0.2 * GB

NVME_GBPS = 6.8
RAM_GBPS = (21.0, 83.0)

# engram: 2 tables, 48 rows read per token, read by 4096-byte pages, about 2 % of the rows straddle a page
ENGRAM_PAGES_PER_TOKEN = 48
ENGRAM_PAGE_BYTES = 4096
ENGRAM_STRADDLE = 0.02


class StatsError(ValueError):
    """A stats file that cannot be trusted."""


def expert_bytes(type_name, params=PARAMS_PER_EXPERT):
    """Bytes of one routed expert stored as `type_name`; the weights must fill whole blocks."""
    if type_name not in BLOCKS:
        raise ValueError(f"unknown type {type_name!r}, expected one of {sorted(BLOCKS)}")
    weights, size = BLOCKS[type_name]
    if params % weights:
        raise ValueError(f"{params} weights do not fill whole {type_name} blocks of {weights}")
    return params // weights * size


def engram_bytes_per_token(pages=ENGRAM_PAGES_PER_TOKEN, page_bytes=ENGRAM_PAGE_BYTES, straddle=ENGRAM_STRADDLE):
    """Physical engram bytes per token: one page per row, plus a page for the rows that straddle two."""
    return pages * (1 + straddle) * page_bytes


def load_stats(path):
    """(counts by half and total as flat lists in (layer, expert) order, tokens by half and total, n_layers,
    n_expert, n_used)."""
    d = json.loads(Path(path).read_text())
    try:
        n_layers, n_expert, n_used = d["n_layers"], d["n_expert"], d["n_used"]
        tokens = {"A": d["tokens"]["A"], "B": d["tokens"]["B"]}
        raw = {k: d["counts"][k] for k in ("A", "B", "total")}
    except KeyError as e:
        raise StatsError(f"{path}: missing key {e}") from e
    flat = {}
    for k, per_layer in raw.items():
        if len(per_layer) != n_layers or any(len(row) != n_expert for row in per_layer):
            raise StatsError(f"{path}: counts[{k}] is not {n_layers} x {n_expert}")
        flat[k] = [c for row in per_layer for c in row]
    tokens["total"] = tokens["A"] + tokens["B"]
    for k, t in tokens.items():
        if sum(flat[k]) != t * n_layers * n_used:
            raise StatsError(f"{path}: counts[{k}] sum to {sum(flat[k])}, expected "
                             f"{t} tokens x {n_layers} layers x {n_used} = {t * n_layers * n_used}")
    return flat, tokens, n_layers, n_expert, n_used


def split_tiers(n_pairs, sizes, vram_bytes, ram_bytes):
    """Pairs per tier: the VRAM holds floor(budget / hot size) pairs, then the RAM floor(budget / warm size), and
    the rest goes to NVMe; a budget larger than the pairs left takes them all."""
    n_hot = min(n_pairs, int(vram_bytes // sizes["hot"]))
    n_warm = min(n_pairs - n_hot, int(ram_bytes // sizes["warm"]))
    return {"hot": n_hot, "warm": n_warm, "cold": n_pairs - n_hot - n_warm}


def rank(counts):
    """Pair indices from the most to the least called; ties break on the pair index so the ranking is stable."""
    return sorted(range(len(counts)), key=lambda i: (-counts[i], i))


def traffic(order, n_pairs_by_tier, eval_counts, eval_tokens, sizes, engram):
    """Share of the calls and bytes per token of every tier, for the placement `order` (learned) measured on
    `eval_counts`."""
    total_calls = sum(eval_counts)
    calls, start = {}, 0
    for t in TIERS:
        end = start + n_pairs_by_tier[t]
        calls[t] = sum(eval_counts[i] for i in order[start:end])
        start = end
    share = {t: (calls[t] / total_calls if total_calls else 0.0) for t in TIERS}
    per_token = {t: calls[t] / eval_tokens * sizes[t] for t in TIERS}       # bytes per token
    nvme_total = per_token["cold"] + engram
    return {
        "call_share": {TIER_MEDIUM[t]: share[t] for t in TIERS},
        "nvme_experts_bytes_per_token": per_token["cold"],
        "nvme_engram_bytes_per_token": engram,
        "nvme_total_bytes_per_token": nvme_total,
        "ram_bytes_per_token": per_token["warm"],
        "nvme_ceiling_tok_s": ceiling(NVME_GBPS, nvme_total),
        "ram_ceiling_tok_s": {f"{bw:g}": ceiling(bw, per_token["warm"]) for bw in RAM_GBPS},
    }


def ceiling(gbps, bytes_per_token):
    """Tokens/s a bandwidth allows for `bytes_per_token`; None when nothing is read."""
    return gbps * GB / bytes_per_token if bytes_per_token > 0 else None


def evaluate_plan(name, types, flat, tokens, vram_gb, ram_gb, engram):
    """The tiers, the total (learned and measured on the same data), the two cross-validation directions, and the
    verdict taken on the less favourable direction."""
    sizes = {t: expert_bytes(ty) for t, ty in zip(TIERS, types)}
    n_pairs = len(flat["total"])
    split = split_tiers(n_pairs, sizes, vram_gb * GB, ram_gb * GB)
    tiers = {TIER_MEDIUM[t]: {"type": ty, "bytes_per_pair": sizes[t], "pairs": split[t],
                              "gb": split[t] * sizes[t] / GB} for t, ty in zip(TIERS, types)}
    orders = {h: rank(flat[h]) for h in ("A", "B", "total")}
    total = traffic(orders["total"], split, flat["total"], tokens["total"], sizes, engram)
    cv = {"A_to_B": traffic(orders["A"], split, flat["B"], tokens["B"], sizes, engram),
          "B_to_A": traffic(orders["B"], split, flat["A"], tokens["A"], sizes, engram)}
    worst_dir = max(cv, key=lambda k: cv[k]["nvme_total_bytes_per_token"])
    worst_nvme = cv[worst_dir]["nvme_total_bytes_per_token"]
    worst_ram = max(c["ram_bytes_per_token"] for c in cv.values())
    return {
        "name": name,
        "types": dict(zip(TIERS, types)),
        "tiers": tiers,
        "total": total,
        "cross_validation": {
            **cv,
            "worst": {
                "direction": worst_dir,
                "nvme_total_bytes_per_token": worst_nvme,
                "nvme_experts_bytes_per_token": max(c["nvme_experts_bytes_per_token"] for c in cv.values()),
                "ram_bytes_per_token": worst_ram,
                "nvme_ceiling_tok_s": ceiling(NVME_GBPS, worst_nvme),
                "ram_ceiling_tok_s": {f"{bw:g}": ceiling(bw, worst_ram) for bw in RAM_GBPS},
            },
        },
        "verdict": {
            "basis": "cross-validation, less favourable direction, experts plus engram",
            "threshold_bytes_per_token": NVME_BYTES_PER_TOKEN_MAX,
            "value_bytes_per_token": worst_nvme,
            "pass": worst_nvme <= NVME_BYTES_PER_TOKEN_MAX,
        },
    }


def build_report(flat, tokens, plans, vram_gb, ram_gb, engram):
    return {
        "placement": "static, no dynamic cache assumed",
        "unit": "(layer, expert) pair",
        "pairs": len(flat["total"]),
        "tokens": tokens,
        "budgets_gb": {"vram": vram_gb, "ram": ram_gb},
        "bandwidth_gbps": {"nvme": NVME_GBPS, "ram": list(RAM_GBPS)},
        "engram_bytes_per_token": engram,
        "plans": [evaluate_plan(n, ty, flat, tokens, vram_gb, ram_gb, engram) for n, ty in plans.items()],
    }


def rate(tok_s):
    """A ceiling for display; no ceiling when nothing is read."""
    return "no limit" if tok_s is None else f"{tok_s:.1f} tok/s"


def format_report(report):
    """A readable table of the report."""
    lines = [f"placement: {report['placement']}; unit: {report['unit']}, {report['pairs']} pairs; "
             f"tokens A={report['tokens']['A']} B={report['tokens']['B']}",
             f"budgets: VRAM {report['budgets_gb']['vram']:g} Go, RAM {report['budgets_gb']['ram']:g} Go; "
             f"engram {report['engram_bytes_per_token']:.0f} B/token on NVMe"]
    for p in report["plans"]:
        lines.append("")
        lines.append(f"{p['name']}: hot {p['types']['hot']} / warm {p['types']['warm']} / cold {p['types']['cold']}")
        for medium, t in p["tiers"].items():
            lines.append(f"  {medium:<5} {t['type']:<8} {t['pairs']:>6} pairs {t['gb']:>8.2f} Go")
        for label, r in (("total", p["total"]), ("A->B", p["cross_validation"]["A_to_B"]),
                         ("B->A", p["cross_validation"]["B_to_A"])):
            s = r["call_share"]
            lines.append(f"  {label:<5} calls vram {s['vram']:.2%} ram {s['ram']:.2%} nvme {s['nvme']:.2%}")
            lines.append(f"        nvme experts {r['nvme_experts_bytes_per_token'] / GB:.4f} Go/token, "
                         f"+ engram {r['nvme_engram_bytes_per_token'] / GB:.6f} = "
                         f"{r['nvme_total_bytes_per_token'] / GB:.4f} Go/token, ceiling "
                         f"{rate(r['nvme_ceiling_tok_s'])}")
            ram = ", ".join(f"{bw} Go/s: {rate(v)}" for bw, v in r["ram_ceiling_tok_s"].items())
            lines.append(f"        ram {r['ram_bytes_per_token'] / GB:.4f} Go/token, ceiling {ram}")
        v = p["verdict"]
        lines.append(f"  verdict (cross-validation worst, {p['cross_validation']['worst']['direction']}): "
                     f"{v['value_bytes_per_token'] / GB:.4f} Go/token vs {v['threshold_bytes_per_token'] / GB:.1f} "
                     f"-> {'PASS' if v['pass'] else 'FAIL'}")
    return "\n".join(lines)


def parse_plan(text):
    """`NAME=hot,warm,cold` into (NAME, (hot, warm, cold))."""
    try:
        name, types = text.split("=", 1)
        hot, warm, cold = (t.strip().lower() for t in types.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected NAME=hot,warm,cold") from None
    for ty in (hot, warm, cold):
        expert_bytes(ty)
    return name, (hot, warm, cold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", required=True, help="stats.json of expert_stats.py")
    ap.add_argument("--out", help="JSON report")
    ap.add_argument("--vram-gb", type=float, default=26.0)
    ap.add_argument("--ram-gb", type=float, default=100.0)
    ap.add_argument("--plan", action="append", type=parse_plan, metavar="NAME=hot,warm,cold",
                    help="a bit plan (repeatable); default: P1, P2 and P3")
    args = ap.parse_args(argv)
    plans = dict(args.plan) if args.plan else DEFAULT_PLANS
    flat, tokens, *_ = load_stats(args.stats)
    report = build_report(flat, tokens, plans, args.vram_gb, args.ram_gb, engram_bytes_per_token())
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
