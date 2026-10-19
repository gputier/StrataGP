#!/usr/bin/env python3
"""expert_stats.py - per-expert call counts of a routing trace, split in two halves of a corpus.

Reads the RTR1 files written by the ik_llama.cpp routing patch (tools/v41/ik_routing.patch): little-endian, the
4 bytes `RTR1`, an int32 `n_used` (experts chosen per token, 6 for DeepSeek-V4.1), then records of `1 + n_used`
uint16: the layer, then the `n_used` expert ids.  One record per layer and per evaluated token.  A half can span
several files (a window cut and resumed): they are read in the order given and concatenated.

Per layer and per half it counts the calls of every expert, then computes:

  concentration  the share of the calls carried by the top 25 % experts (96 of 384), averaged over the layers,
                 for each half;
  stability      the share of half A's top 25 % that is also in half B's top 25 %, averaged over the layers.

Both are compared to the thresholds written in the plan before any measurement (THRESHOLDS below).  The JSON
output holds the counts per layer and per expert (A, B, total), the tokens per half, the concentration and the
stability per layer and on average, and the verdicts.

`--profile FILE` also writes a STRP expert profile (the format src/core/expert_cache.cpp reads) ranking every
(layer, expert) pair by its total calls, with `n_layers` taken from the data.

    python tools/v41/expert_stats.py --half-a a1.rtr a2.rtr --half-b b1.rtr [--n-expert 384] [--out stats.json]
                                     [--profile expert-profile.bin] [--trim-ragged]
"""
import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"RTR1"
HEADER_BYTES = 8
STRP_MAGIC, STRP_VERSION = b"STRP", 1
CHUNK_RECORDS = 1 << 16

# written in the plan before any measurement, never retouched
THRESHOLDS = {"concentration": 0.50, "stability": 0.70}
TOP_FRACTION = 0.25


class TraceError(ValueError):
    """A routing file that cannot be trusted."""


def read_header(path):
    """The `n_used` of an RTR1 file, after checking the magic and that the body is whole records."""
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES:
        raise TraceError(f"{path}: shorter than the {HEADER_BYTES}-byte header")
    if head[:4] != MAGIC:
        raise TraceError(f"{path}: bad magic {head[:4]!r}, expected {MAGIC!r}")
    (n_used,) = struct.unpack("<i", head[4:])
    if n_used <= 0 or n_used > 64:
        raise TraceError(f"{path}: n_used {n_used} is out of range")
    body = Path(path).stat().st_size - HEADER_BYTES
    record = 2 * (1 + n_used)
    if body % record:
        raise TraceError(f"{path}: body of {body} bytes is not a multiple of the {record}-byte record (truncated?)")
    return n_used


def iter_records(path, n_used):
    """(layer, ids...) tuples of one file, read by chunks."""
    unpack = struct.Struct(f"<{1 + n_used}H")
    size = unpack.size
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        while True:
            blob = f.read(size * CHUNK_RECORDS)
            if not blob:
                return
            yield from unpack.iter_unpack(blob)


def count_half(paths, n_expert, n_layer, limits=None):
    """({layer: [count per expert]}, {layer: records seen}, n_used) for the files of one half.

    `limits` ({layer: n}) keeps only the first n records of each layer."""
    n_used = None
    for p in paths:
        h = read_header(p)
        if n_used is not None and h != n_used:
            raise TraceError(f"{p}: n_used {h} differs from {n_used} in the other files")
        n_used = h
    counts, seen = {}, {}
    for p in paths:
        for rec in iter_records(p, n_used):
            layer = rec[0]
            if n_layer is not None and layer >= n_layer:
                raise TraceError(f"{p}: layer {layer} is out of range (n_layer {n_layer})")
            if limits is not None and seen.get(layer, 0) >= limits.get(layer, 0):
                continue
            ids = rec[1:]
            if max(ids) >= n_expert:
                raise TraceError(f"{p}: expert id {max(ids)} is out of range (n_expert {n_expert})")
            row = counts.get(layer)
            if row is None:
                row = counts[layer] = [0] * n_expert
            for e in ids:
                row[e] += 1
            seen[layer] = seen.get(layer, 0) + 1
    return counts, seen, n_used


def load_half(name, paths, n_expert, n_layer, trim_ragged):
    """The counts of one half as a list per layer, and its token count (records per layer)."""
    counts, seen, n_used = count_half(paths, n_expert, n_layer)
    if not seen:
        raise TraceError(f"half {name}: no record")
    layers = sorted(seen)
    if layers != list(range(len(layers))):
        raise TraceError(f"half {name}: layers {layers[0]}..{layers[-1]} are not contiguous from 0 "
                         f"({len(layers)} seen)")
    tokens = min(seen.values())
    if max(seen.values()) != tokens:
        if not trim_ragged:
            per = ", ".join(f"{l}:{seen[l]}" for l in layers if seen[l] != tokens)
            raise TraceError(f"half {name}: layers do not all hold {tokens} records (e.g. {per}); a window was "
                             f"cut mid-batch, pass --trim-ragged to keep the first {tokens} of every layer")
        counts, seen, _ = count_half(paths, n_expert, n_layer, limits={l: tokens for l in layers})
    return [counts[l] for l in layers], tokens, n_used


def top_set(row, k):
    """The k most called experts, ties broken by the lower id."""
    return set(sorted(range(len(row)), key=lambda e: (-row[e], e))[:k])


def concentration(row, k):
    total = sum(row)
    return sum(sorted(row, reverse=True)[:k]) / total


def stability(row_a, row_b, k):
    return len(top_set(row_a, k) & top_set(row_b, k)) / k


def mean(values):
    return sum(values) / len(values)


def analyse(counts_a, counts_b, tokens_a, tokens_b, n_used, n_expert, top_fraction=TOP_FRACTION):
    k = round(n_expert * top_fraction)
    if k < 1:
        raise TraceError("the top fraction selects no expert")
    conc_a = [concentration(r, k) for r in counts_a]
    conc_b = [concentration(r, k) for r in counts_b]
    stab = [stability(a, b, k) for a, b in zip(counts_a, counts_b)]
    total = [[x + y for x, y in zip(a, b)] for a, b in zip(counts_a, counts_b)]
    verdicts = {
        "concentration_A": {"value": mean(conc_a), "threshold": THRESHOLDS["concentration"]},
        "concentration_B": {"value": mean(conc_b), "threshold": THRESHOLDS["concentration"]},
        "stability": {"value": mean(stab), "threshold": THRESHOLDS["stability"]},
    }
    for v in verdicts.values():
        v["pass"] = v["value"] >= v["threshold"]
    verdicts["all_pass"] = all(v["pass"] for v in verdicts.values())
    return {
        "n_layers": len(counts_a), "n_expert": n_expert, "n_used": n_used, "top_k": k,
        "tokens": {"A": tokens_a, "B": tokens_b},
        "counts": {"A": counts_a, "B": counts_b, "total": total},
        "concentration": {"A": {"per_layer": conc_a, "mean": mean(conc_a)},
                          "B": {"per_layer": conc_b, "mean": mean(conc_b)}},
        "stability": {"per_layer": stab, "mean": mean(stab)},
        "verdicts": verdicts,
    }


def rank_pairs(total):
    """Every (layer, expert) pair, most called first.  Ties go expert by expert across the layers, the way
    tools/make_profile.py fills the pairs no trace used."""
    pairs = [(l, e) for l, row in enumerate(total) for e in range(len(row))]
    return sorted(pairs, key=lambda p: (-total[p[0]][p[1]], p[1], p[0]))


def write_profile(path, ranked, n_layers, n_expert):
    """A STRP profile as src/core/expert_cache.cpp write_expert_profile writes it: the magic, five uint32 (version,
    n_layers, n_expert, slots, n_ranked), the ranked (layer, expert) uint16 pairs, then the int32 rank of every
    pair, layer by layer, -1 for an unranked one."""
    table = [[-1] * n_expert for _ in range(n_layers)]
    for rank, (layer, e) in enumerate(ranked):
        table[layer][e] = rank
    with open(path, "wb") as f:
        f.write(STRP_MAGIC + struct.pack("<5I", STRP_VERSION, n_layers, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for row in table:
            f.write(struct.pack(f"<{n_expert}i", *row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--half-a", nargs="+", required=True, metavar="RTR1", help="routing file(s) of half A, in order")
    ap.add_argument("--half-b", nargs="+", required=True, metavar="RTR1", help="routing file(s) of half B, in order")
    ap.add_argument("--n-expert", type=int, default=384, help="experts per layer (default 384)")
    ap.add_argument("--n-layer", type=int, default=None,
                    help="refuse a layer id at or above this (default: layers are read from the data)")
    ap.add_argument("--trim-ragged", action="store_true",
                    help="when a cut window left layers with unequal record counts, keep the first n of every layer")
    ap.add_argument("--out", help="JSON output file (default: stdout)")
    ap.add_argument("--profile", help="also write a STRP profile ranking the pairs by total calls")
    a = ap.parse_args(argv)

    try:
        counts_a, tok_a, used_a = load_half("A", a.half_a, a.n_expert, a.n_layer, a.trim_ragged)
        counts_b, tok_b, used_b = load_half("B", a.half_b, a.n_expert, a.n_layer, a.trim_ragged)
        if used_a != used_b:
            raise TraceError(f"n_used {used_a} in half A, {used_b} in half B")
        if len(counts_a) != len(counts_b):
            raise TraceError(f"{len(counts_a)} layers in half A, {len(counts_b)} in half B")
    except TraceError as err:
        print(f"expert_stats: {err}", file=sys.stderr)
        return 2

    result = analyse(counts_a, counts_b, tok_a, tok_b, used_a, a.n_expert)
    text = json.dumps(result)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if a.profile:
        write_profile(a.profile, rank_pairs(result["counts"]["total"]), result["n_layers"], a.n_expert)
    v = result["verdicts"]
    print(f"tokens A={tok_a} B={tok_b}  concentration A={v['concentration_A']['value']:.4f} "
          f"B={v['concentration_B']['value']:.4f}  stability={v['stability']['value']:.4f}  "
          f"all_pass={v['all_pass']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
