#!/usr/bin/env python3
"""merge_imatrix.py - merges the importance matrices of the two corpus halves, expert by expert.

`llama-imatrix --in-file` would average the two files with equal weights.  That is wrong for the rarely called
routed experts: an expert that half B never called holds a filler of ones there, and an equal-weight average pollutes
the other half's value with it.  This script weights every expert slice by the number of calls it received in each
half, and leaves ones to an expert that neither half called.

Format (written by `save_imatrix`, read by `load_imatrix` in examples/quantize/quantize.cpp:221-282 of ik_llama at
c446f1e0), little-endian: an int32 entry count; per entry an int32 name length, the name, an int32 `ncall`, an int32
`nval` and `nval` float32; then an int32 last call count, an int32 length and the name of the calibration file.  The
loader divides every value by the entry's `ncall`.  `save_imatrix` writes `sum(x^2) / count * ncall`, where `count` is
the number of tokens that went through the element, so `value / ncall` is the mean of x^2 per token ("the mean").

Which counts weight what:
  * a routed expert tensor (`blk.N.ffn_{gate,up,down}_exps.weight`, `nval = n_expert * ne0`, the slice of expert e is
    `[e * ne0, (e + 1) * ne0)`): the .dat holds no per-expert count (one `ncall` per entry, and the means are already
    divided by the expert's own count), so the counts come from `stats.json` of expert_stats.py, which is the routing
    trace of the same two runs.  The slice of an expert with no call in a half is a filler (the mean is 1): the script
    refuses a file where that does not hold, which proves the counts and the matrix come from the same run.  Merged
    mean = (n_A * mean_A + n_B * mean_B) / (n_A + n_B); one half having no call, the other half's mean as is; no call
    in either half, 1.
  * any other tensor: weighted by the entry's own `ncall` (calls of the tensor in each half), which the .dat holds.
    Merged mean = (ncall_A * mean_A + ncall_B * mean_B) / (ncall_A + ncall_B).  This is exact only if every call
    carries the same number of tokens (batches of equal size): `count` of `save_imatrix` is in tokens, `ncall` in
    calls, and the .dat holds no token count to weight with.
The output `ncall` of an entry is the sum of the two, and its values are the merged mean times it, so the loader finds
the merged mean.  Both files must hold the same entries.  `--out` must not be one of the inputs.

The merged file has no `ffn_gate_exps` entry, only `ffn_up_exps` and `ffn_down_exps`, as the two inputs.
`llama-quantize` gives the `ffn_gate_exps` tensor the matrix of `ffn_up_exps` when it has none of its own
(src/llama-quantize.cpp:1880-1889 at c446f1e0).  The format above and this fallback were read in that source.

    python3 -I tools/v41/merge_imatrix.py --a imatrix_a.dat --b imatrix_b.dat --stats stats.json --out imatrix_ab.dat
        [--report report.json]
"""
import argparse
import json
import os
import re
import struct
import sys
from array import array
from collections import namedtuple
from pathlib import Path

EXPERT_ENTRY = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$")

# an entry of a file: where its floats start, so they can be read one entry at a time
Entry = namedtuple("Entry", "name ncall nval offset")


class ImatrixError(ValueError):
    """An importance matrix or a stats file that cannot be trusted."""


def read_exact(f, n, what):
    data = f.read(n)
    if len(data) != n:
        raise ImatrixError(f"{f.name}: file ends while reading {what}")
    return data


def read_int(f, what):
    return struct.unpack("<i", read_exact(f, 4, what))[0]


def index(path):
    """([Entry], trailer) of a matrix file, read without loading the floats.  The trailer is the last call count and
    the calibration file name; the whole file must be consumed."""
    entries = []
    with open(path, "rb") as f:
        n_entries = read_int(f, "the entry count")
        if n_entries < 1:
            raise ImatrixError(f"{path}: {n_entries} entries")
        for i in range(n_entries):
            length = read_int(f, f"the name length of entry {i}")
            name = read_exact(f, length, f"the name of entry {i}").decode("utf-8")
            ncall, nval = read_int(f, f"ncall of {name}"), read_int(f, f"nval of {name}")
            if nval < 1:
                raise ImatrixError(f"{path}: {name} has {nval} values")
            entries.append(Entry(name, ncall, nval, f.tell()))
            f.seek(nval * 4, 1)
        last_call = read_int(f, "the last call count")
        dataset = read_exact(f, read_int(f, "the dataset name length"), "the dataset name").decode("utf-8")
        if f.read(1):
            raise ImatrixError(f"{path}: bytes after the trailer")
    return entries, {"last_call": last_call, "dataset": dataset}


def read_values(f, entry):
    values = array("f")
    f.seek(entry.offset)
    values.frombytes(read_exact(f, entry.nval * 4, f"the values of {entry.name}"))
    if sys.byteorder != "little":
        values.byteswap()
    return values


def write_entry(f, name, ncall, values):
    raw = name.encode("utf-8")
    f.write(struct.pack("<i", len(raw)) + raw + struct.pack("<ii", ncall, len(values)))
    out = array("f", values)
    if sys.byteorder != "little":
        out.byteswap()
    out.tofile(f)


def load_counts(path):
    """({'A': [[calls of expert e in layer l]], 'B': ...}, n_layers, n_expert) from stats.json."""
    d = json.loads(Path(path).read_text())
    try:
        n_layers, n_expert = d["n_layers"], d["n_expert"]
        counts = {h: d["counts"][h] for h in ("A", "B")}
    except KeyError as e:
        raise ImatrixError(f"{path}: missing key {e}") from e
    for h, grid in counts.items():
        if len(grid) != n_layers or any(len(row) != n_expert for row in grid):
            raise ImatrixError(f"{path}: counts[{h}] is not {n_layers} x {n_expert}")
    return counts, n_layers, n_expert


def merge_dense(a, b):
    """Stored values of a tensor weighted by its calls: the merged mean times (ncall_a + ncall_b) is a + b, since
    the stored values are mean * ncall."""
    return [x + y for x, y in zip(a, b)]


def merge_expert(a, ncall_a, n_a, b, ncall_b, n_b):
    """The stored slice of one expert, merged: means weighted by the calls n_a and n_b the expert received, times
    the output ncall (ncall_a + ncall_b); ones when neither half called it."""
    ncall = ncall_a + ncall_b
    total = n_a + n_b
    if total == 0:
        return [float(ncall)] * len(a)
    ka = n_a / total * ncall / ncall_a
    kb = n_b / total * ncall / ncall_b
    return [x * ka + y * kb for x, y in zip(a, b)]


def check_filler(slice_, ncall, n, name, expert, half):
    """An expert with no call in a half carries a filler: stored value = ncall (mean 1).  Anything else means the
    counts are not those of this matrix."""
    if n == 0 and slice_.count(float(ncall)) != len(slice_):
        raise ImatrixError(f"{name}: expert {expert} has no call in half {half} but its slice is not the filler; "
                           f"the stats do not belong to this matrix")


def merge(path_a, path_b, stats_path, out_path):
    """Writes the merged matrix to `out_path`; returns a summary of what was merged."""
    for path in (path_a, path_b):
        if os.path.exists(out_path) and os.path.samefile(out_path, path):
            raise ImatrixError(f"--out {out_path} is the input {path}: it would be emptied before being read")
    entries_a, trailer_a = index(path_a)
    entries_b, trailer_b = index(path_b)
    by_name_b = {e.name: e for e in entries_b}
    if len(by_name_b) != len(entries_b) or set(by_name_b) != {e.name for e in entries_a}:
        raise ImatrixError("the two matrices do not hold the same entries")
    counts, n_layers, n_expert = load_counts(stats_path)
    summary = {"entries": len(entries_a), "expert_entries": 0, "dense_entries": 0,
               "experts_both": 0, "experts_one_half": 0, "experts_neither": 0}
    last_call = trailer_a["last_call"] + trailer_b["last_call"]
    dataset = f"{trailer_a['dataset']}+{trailer_b['dataset']}"
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb, open(out_path, "wb") as out:
        out.write(struct.pack("<i", len(entries_a)))
        for ea in entries_a:
            eb = by_name_b[ea.name]
            if ea.nval != eb.nval:
                raise ImatrixError(f"{ea.name}: {ea.nval} values in A, {eb.nval} in B")
            if ea.ncall < 1 or eb.ncall < 1:
                raise ImatrixError(f"{ea.name}: ncall {ea.ncall} in A, {eb.ncall} in B")
            va, vb = read_values(fa, ea), read_values(fb, eb)
            m = EXPERT_ENTRY.match(ea.name)
            if m is None:
                merged = merge_dense(va, vb)
                summary["dense_entries"] += 1
            else:
                layer = int(m.group(1))
                if layer >= n_layers or ea.nval % n_expert:
                    raise ImatrixError(f"{ea.name}: layer {layer} of {n_layers}, {ea.nval} values for "
                                       f"{n_expert} experts")
                width = ea.nval // n_expert
                merged = []
                for e in range(n_expert):
                    n_a, n_b = counts["A"][layer][e], counts["B"][layer][e]
                    sa, sb = va[e * width:(e + 1) * width], vb[e * width:(e + 1) * width]
                    check_filler(sa, ea.ncall, n_a, ea.name, e, "A")
                    check_filler(sb, eb.ncall, n_b, ea.name, e, "B")
                    merged += merge_expert(sa, ea.ncall, n_a, sb, eb.ncall, n_b)
                    summary["experts_both" if n_a and n_b else "experts_one_half" if n_a or n_b
                            else "experts_neither"] += 1
                summary["expert_entries"] += 1
            write_entry(out, ea.name, ea.ncall + eb.ncall, merged)
        out.write(struct.pack("<i", last_call))
        raw = dataset.encode("utf-8")
        out.write(struct.pack("<i", len(raw)) + raw)
    return summary


def spot_check(path_a, path_b, path_out, stats_path, tensor="ffn_up_exps"):
    """Recomputes by hand, from the raw floats of the three files, the first expert seen in both halves, the first
    seen in one half and the first seen in neither (in the `tensor` entries, layer order), and returns what was
    found."""
    counts, n_layers, n_expert = load_counts(stats_path)
    entries = [{e.name: e for e in index(p)[0]} for p in (path_a, path_b, path_out)]
    wanted = {"both": None, "one half": None, "neither": None}
    for layer in range(n_layers):
        for e in range(n_expert):
            n_a, n_b = counts["A"][layer][e], counts["B"][layer][e]
            kind = "both" if n_a and n_b else "one half" if n_a or n_b else "neither"
            if wanted[kind] is None:
                wanted[kind] = (layer, e, n_a, n_b)
    found = []
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb, open(path_out, "rb") as fo:
        for kind, hit in wanted.items():
            if hit is None:
                found.append({"kind": kind, "found": False})
                continue
            layer, e, n_a, n_b = hit
            name = f"blk.{layer}.{tensor}.weight"
            ea, eb, eo = (d[name] for d in entries)
            width = ea.nval // n_expert
            n = min(4, width)
            part = slice(e * width, e * width + n)
            a, b, o = (read_values(f, x)[part] for f, x in ((fa, ea), (fb, eb), (fo, eo)))
            ma, mb = [v / ea.ncall for v in a], [v / eb.ncall for v in b]     # the means the loader would use
            if n_a + n_b == 0:
                expect = [1.0] * n
            else:
                expect = [(n_a * x + n_b * y) / (n_a + n_b) for x, y in zip(ma, mb)]
            got = [v / eo.ncall for v in o]
            found.append({"kind": kind, "found": True, "tensor": name, "expert": e, "calls_a": n_a, "calls_b": n_b,
                          "mean_a": ma, "mean_b": mb, "expected_mean": expect, "merged_mean": got,
                          "max_abs_error": max(abs(x - y) for x, y in zip(expect, got))})
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="matrix of half A")
    ap.add_argument("--b", required=True, help="matrix of half B")
    ap.add_argument("--stats", required=True, help="stats.json of expert_stats.py (per-expert calls of both halves)")
    ap.add_argument("--out", required=True, help="merged matrix")
    ap.add_argument("--report", help="JSON summary, with the re-read of the output and a hand check")
    args = ap.parse_args(argv)
    try:
        summary = merge(args.a, args.b, args.stats, args.out)
        entries_out, trailer = index(args.out)
        entries_a, _ = index(args.a)
    except ImatrixError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sizes_ok = [(e.name, e.nval) for e in entries_a] == [(e.name, e.nval) for e in entries_out]
    summary["reread"] = {"entries": len(entries_out), "sizes_match_inputs": sizes_ok, "trailer": trailer}
    summary["spot_check"] = spot_check(args.a, args.b, args.out, args.stats)
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "spot_check"}, indent=1))
    for s in summary["spot_check"]:
        print(s)
    return 0 if sizes_ok else 2


if __name__ == "__main__":
    sys.exit(main())
