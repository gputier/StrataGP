#!/usr/bin/env python3
"""tier_imatrix.py - the importance matrix of the split-expert file: one up and one down entry per tier and layer.

`llama-quantize` looks up the matrix of a tensor by its name and needs exactly `ne[0] * ne[2]` values for it.  In the
split file the routed experts of layer L live in three tensors per family, `blk.L.tK.ffn_{gate,up,down}_exps.weight`,
holding `n_k` experts each, so the entries `blk.L.ffn_up_exps.weight` (ne0 * n_expert values) and
`blk.L.ffn_down_exps.weight` of the merged matrix (merge_imatrix.py) are replaced, in place, by three entries each:
`blk.L.tK.ffn_up_exps.weight` and `blk.L.tK.ffn_down_exps.weight`, K = 0, 1, 2, holding the slices
`[e * ne0, (e + 1) * ne0)` of the experts of tier K, concatenated in the order of their positions in the tier (the
increasing expert ids of tier_table.py).  `ncall` is copied as is, the original entry is dropped, and every other
entry is copied unchanged.  No gate entry is written: for a name holding `ffn_gate_exps.weight` and no entry of its
own, `llama-quantize` takes the entry of the matching `ffn_up_exps.weight` (src/llama-quantize.cpp:1880-1894 of
ik_llama at c446f1e0); the `tK.` prefix keeps that substring in the name.  The three tiers are always written, in
identity mode too.  The file format is that of merge_imatrix.py.

    python3 -I tools/v41/tier_imatrix.py --imatrix imatrix_ab.dat --tiers tiers_p3.json --out imatrix_tiers.dat
"""
import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tier_table  # noqa: E402
from merge_imatrix import EXPERT_ENTRY, ImatrixError, index, read_values, write_entry, write_trailer  # noqa: E402


def tier_name(layer, tier, family):
    return f"blk.{layer}.t{tier}.ffn_{family}_exps.weight"


def tier_slices(values, ids, width):
    """The slices of the experts `ids` of a `width`-wide expert tensor, concatenated in the order of `ids`."""
    out = []
    for e in ids:
        out += values[e * width:(e + 1) * width]
    return out


def check_entries(entries, table):
    """Number of entries the output holds; refuses an expert entry that does not fit the table, and a layer that
    lacks its up or its down entry."""
    n_out = 0
    present = set()
    for e in entries:
        m = EXPERT_ENTRY.match(e.name)
        if m is None:
            n_out += 1
            continue
        if m.group(2) == "gate":
            raise ImatrixError(f"{e.name}: a gate entry is not expected, the quantizer takes the up entry")
        if int(m.group(1)) >= table["n_layer"] or e.nval % table["n_expert"]:
            raise ImatrixError(f"{e.name}: layer {m.group(1)} of {table['n_layer']}, {e.nval} values for "
                               f"{table['n_expert']} experts")
        if any(not ids for ids in table["layers"][int(m.group(1))]):
            raise ImatrixError(f"{e.name}: layer {m.group(1)} has an empty tier, its entry would hold no value")
        present.add((int(m.group(1)), m.group(2)))
        n_out += tier_table.N_TIERS
    for layer in range(table["n_layer"]):
        for family in ("up", "down"):
            if (layer, family) not in present:
                raise ImatrixError(f"layer {layer} has no blk.{layer}.ffn_{family}_exps.weight entry")
    return n_out


def split_matrix(path_in, table, path_out):
    """Writes the per-tier matrix to `path_out`; returns the entry counts in and out and the tier entries written
    as (name, nval, experts, width)."""
    if os.path.exists(path_out) and os.path.samefile(path_out, path_in):
        raise ImatrixError(f"--out {path_out} is the input {path_in}: it would be emptied before being read")
    entries, trailer = index(path_in)
    n_out = check_entries(entries, table)
    written = []
    with open(path_in, "rb") as fin, open(path_out, "wb") as out:
        out.write(struct.pack("<i", n_out))
        for e in entries:
            m = EXPERT_ENTRY.match(e.name)
            values = read_values(fin, e)
            if m is None:
                write_entry(out, e.name, e.ncall, values)
                continue
            layer, family = int(m.group(1)), m.group(2)
            width = e.nval // table["n_expert"]
            for k, ids in enumerate(table["layers"][layer]):
                name = tier_name(layer, k, family)
                write_entry(out, name, e.ncall, tier_slices(values, ids, width))
                written.append((name, len(ids) * width, len(ids), width))
        write_trailer(out, trailer["last_call"], trailer["dataset"])
    return {"entries_in": len(entries), "entries_out": n_out, "tier_entries": written}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--imatrix", required=True, help="merged matrix (merge_imatrix.py)")
    ap.add_argument("--tiers", required=True, help="tier table JSON (tier_table.py)")
    ap.add_argument("--out", required=True, help="per-tier matrix")
    args = ap.parse_args(argv)
    try:
        table = tier_table.load_table(args.tiers)
        summary = split_matrix(args.imatrix, table, args.out)
    except (ImatrixError, tier_table.TableError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{args.out}: {summary['entries_in']} entries in, {summary['entries_out']} out, "
          f"{len(summary['tier_entries'])} tier entries")
    print(f"sha256 {hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
