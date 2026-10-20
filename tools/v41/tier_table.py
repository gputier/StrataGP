#!/usr/bin/env python3
"""tier_table.py - the tier table of the split-expert file: which routed experts go to which memory tier, per layer.

The table is the contract shared by tier_imatrix.py (the importance matrix per tier) and the tensor splitter:

    {"mode": "p3" | "identity", "n_expert": 384, "n_layer": 40, "tier_types": ["mxfp4", "iq2_xs", "iq1_m"],
     "layers": [[[e, ...], [e, ...], [e, ...]], ...]}

`layers[L][K]` is the increasing list of the expert ids of tier K in layer L, and the position of an expert in its
tier is its index in that list.  The three lists of a layer partition 0..n_expert-1.  The tensors of tier K in layer L
are `blk.L.tK.ffn_{gate,up,down}_exps.weight` and its lookup table is `blk.L.tK.ffn_exps_map.weight` (I32, shape
[1, n_expert], the position of the expert in tier K, -1 when it is not in it).

Modes:
  * p3: the (layer, expert) pairs are ranked by total call count with tier_budget.rank (the ranking the budget uses for
    its "total" traffic), the first `hot` pairs go to tier 0, the next `warm` to tier 1, the rest to tier 2.  The tier
    sizes are those the budget JSON (tier_budget.py --out) holds for the plan, under tiers.vram / ram / nvme .pairs.
  * identity: the tier of expert e is e % 3, whatever the layer; the split file then computes the same model as the
    unsplit one, which is what the identity test checks.

The JSON is written with sorted keys and compact separators, so the same inputs give the same bytes; the sha256 of
the file is printed.

    python3 -I tools/v41/tier_table.py --mode p3 --stats stats.json --budget budget.json --out tiers_p3.json
        [--plan P3]
    python3 -I tools/v41/tier_table.py --mode identity --out tiers_identity.json [--n-layer 40 --n-expert 384]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tier_budget as tb  # noqa: E402

N_TIERS = 3
TIER_TYPES = ["mxfp4", "iq2_xs", "iq1_m"]
# the type names of tier_budget.BLOCKS that carry another name in the file
BUDGET_TO_FILE_TYPE = {"fp4": "mxfp4"}
# the medium keys of the budget JSON, in the order of the tiers (hot, warm, cold)
BUDGET_MEDIA = ("vram", "ram", "nvme")
MODES = ("p3", "identity")


class TableError(ValueError):
    """A tier table or a budget file that cannot be trusted."""


def check_table(table, source="table"):
    """Raises TableError unless `table` follows the contract; returns it."""
    try:
        mode, n_expert, n_layer = table["mode"], table["n_expert"], table["n_layer"]
        tier_types, layers = table["tier_types"], table["layers"]
    except (KeyError, TypeError) as e:
        raise TableError(f"{source}: missing key {e}") from e
    if mode not in MODES:
        raise TableError(f"{source}: mode {mode!r}, expected one of {MODES}")
    if tier_types != TIER_TYPES:
        raise TableError(f"{source}: tier_types {tier_types}, expected {TIER_TYPES}")
    if len(layers) != n_layer:
        raise TableError(f"{source}: {len(layers)} layers, n_layer is {n_layer}")
    for layer, tiers in enumerate(layers):
        if len(tiers) != N_TIERS:
            raise TableError(f"{source}: layer {layer} has {len(tiers)} tiers, expected {N_TIERS}")
        for k, ids in enumerate(tiers):
            if any(not 0 <= e < n_expert for e in ids) or any(a >= b for a, b in zip(ids, ids[1:])):
                raise TableError(f"{source}: layer {layer} tier {k} is not an increasing list of ids in "
                                 f"0..{n_expert - 1}")
        if sorted(e for ids in tiers for e in ids) != list(range(n_expert)):
            raise TableError(f"{source}: layer {layer} is not a partition of 0..{n_expert - 1}")
    return table


def load_table(path):
    return check_table(json.loads(Path(path).read_text()), str(path))


def tier_sizes(budget_path, plan):
    """Pairs per tier (hot, warm, cold) of the plan `plan` in a tier_budget.py JSON; the plan's types must be the
    file's tier types."""
    budget = json.loads(Path(budget_path).read_text())
    plans = [p for p in budget.get("plans", []) if p.get("name") == plan]
    if len(plans) != 1:
        raise TableError(f"{budget_path}: {len(plans)} plans named {plan!r}")
    try:
        tiers = [plans[0]["tiers"][m] for m in BUDGET_MEDIA]
        types = [BUDGET_TO_FILE_TYPE.get(t["type"], t["type"]) for t in tiers]
        sizes = [t["pairs"] for t in tiers]
    except KeyError as e:
        raise TableError(f"{budget_path}: plan {plan!r} misses key {e}") from e
    if types != TIER_TYPES:
        raise TableError(f"{budget_path}: plan {plan!r} has types {types}, expected {TIER_TYPES}")
    return sizes


def build_p3(stats_path, budget_path, plan="P3"):
    """The table of the plan `plan`: pairs ranked by total call count, cut at the budget's tier sizes."""
    flat, _, n_layer, n_expert, _ = tb.load_stats(stats_path)
    sizes = tier_sizes(budget_path, plan)
    if sum(sizes) != n_layer * n_expert:
        raise TableError(f"{budget_path}: tier sizes {sizes} sum to {sum(sizes)}, stats hold {n_layer * n_expert} "
                         f"pairs")
    bounds = [sizes[0], sizes[0] + sizes[1]]
    layers = [[[] for _ in range(N_TIERS)] for _ in range(n_layer)]
    for position, pair in enumerate(tb.rank(flat["total"])):
        tier = sum(position >= b for b in bounds)
        layers[pair // n_expert][tier].append(pair % n_expert)
    for tiers in layers:
        for ids in tiers:
            ids.sort()
    return check_table({"mode": "p3", "n_expert": n_expert, "n_layer": n_layer, "tier_types": TIER_TYPES,
                        "layers": layers})


def build_identity(n_layer, n_expert):
    """Expert e in tier e % 3, in every layer."""
    one_layer = [list(range(k, n_expert, N_TIERS)) for k in range(N_TIERS)]
    return check_table({"mode": "identity", "n_expert": n_expert, "n_layer": n_layer, "tier_types": TIER_TYPES,
                        "layers": [[list(ids) for ids in one_layer] for _ in range(n_layer)]})


def dump(table):
    """The deterministic bytes of the table file."""
    return (json.dumps(table, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--stats", help="stats.json of expert_stats.py (p3)")
    ap.add_argument("--budget", help="JSON of tier_budget.py (p3)")
    ap.add_argument("--plan", default="P3", help="plan of the budget JSON (p3)")
    ap.add_argument("--n-layer", type=int, default=40, help="identity")
    ap.add_argument("--n-expert", type=int, default=384, help="identity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.mode == "p3":
            if not (args.stats and args.budget):
                ap.error("--mode p3 needs --stats and --budget")
            for path in (args.stats, args.budget):
                if os.path.exists(args.out) and os.path.samefile(args.out, path):
                    raise TableError(f"--out {args.out} is the input {path}: it would be overwritten")
            table = build_p3(args.stats, args.budget, args.plan)
        else:
            table = build_identity(args.n_layer, args.n_expert)
    except (TableError, tb.StatsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raw = dump(table)
    Path(args.out).write_bytes(raw)
    sizes = [[len(ids) for ids in tiers] for tiers in table["layers"]]
    print(f"{args.out}: mode {table['mode']}, {table['n_layer']} layers x {table['n_expert']} experts")
    for k, name in enumerate(TIER_TYPES):
        col = [s[k] for s in sizes]
        print(f"  tier {k} ({name}): {sum(col)} pairs, per layer min {min(col)} max {max(col)}")
    print(f"sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
