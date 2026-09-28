"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order (issue #49): a blend of the base profile's ranking (default: the shipped data/expert-profile.bin) and how
often your routing traces used each pair, then every pair still missing, interleaved across the layers.  Both are
put on the traces' scale - a pair's base rank r counts as the r-th highest trace frequency - and a pair scores
    w * (its frequency in the traces) + (1 - w) * (the frequency of its base rank),      w = --trace-weight
ties going to the base rank.  w = 0.5 (the default) lets a pair your traces use a lot move ahead of base pairs
they barely touch, without a short trace's one-off pairs pushing out the base's best; w = 1 ranks by the traces
first and the base rank after; w = 0 (or no trace) keeps the base ranking unchanged.  The shipped profile already
ranks all 24,576 pairs, so the old order - base first, then the traces - could never let a trace change anything.

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--out PATH]
                                 [--trace-weight 0.5]
                                 [--n-expert 256]      (a pruned model: GSQ-RCO Coder keeps 256 of 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.
"""
import argparse
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
MAGIC, VERSION = b"STRP", 1


def read_profile(path, n_expert=N_EXPERT):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (N_LAYER, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {N_LAYER}x{n_expert}")
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def read_trace(path, n_expert=N_EXPERT):
    """(layer, k, k expert ids, k weights) records, as `--dump-routing` writes them."""
    blob = Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < N_LAYER and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k                                    # the ids and the weights
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT):
    table = [[-1] * n_expert for _ in range(N_LAYER)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, N_LAYER, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(N_LAYER):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def blend(base, freq, weight):
    """The base ranking and the trace frequencies as one order (see the module docstring): every pair of either,
    by w * freq + (1 - w) * (the trace frequency at the pair's base rank), then by base rank, then by pair."""
    base = [(int(p[0]), int(p[1])) for p in base]
    rank = {}
    for r, p in enumerate(base):
        rank.setdefault(p, r)
    by_rank = sorted(freq.values(), reverse=True)          # the trace's frequency at each rank

    def score(p):
        r = rank.get(p)
        return weight * freq.get(p, 0) + (1.0 - weight) * (by_rank[r] if r is not None and r < len(by_rank) else 0)
    pairs = set(rank) | set(freq)
    return sorted(pairs, key=lambda p: (-score(p), rank.get(p, len(base)), p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="the ranking the traces adjust")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--trace-weight", type=float, default=0.5,
                    help="0..1: the traces' share against the base ranking (1 = traces first, 0 = the base unchanged, "
                         "as before issue #49; default 0.5)")
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    a = ap.parse_args()
    if not 0.0 <= a.trace_weight <= 1.0:
        ap.error("--trace-weight is between 0 and 1")

    ranked, seen = [], set()

    def take(pairs):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)

    ne = a.n_expert
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, ne).items():
            freq[p] += c
    base = [] if a.no_base else read_profile(a.base, ne)
    if base and freq and a.trace_weight > 0.0:
        take(blend(base, freq, a.trace_weight))
        moved = sum(1 for p, q in zip(ranked, base) if p != (int(q[0]), int(q[1])))
        what = (f"{len(ranked)} from the base and the traces, blended at weight {a.trace_weight:g} "
                f"({moved} places moved)")
    else:                                              # the base first, then the traces (issue #49: the old order)
        take(base)
        n_base = len(ranked)
        take(p for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
        what = f"{n_base} from the base, {len(ranked) - n_base} from the traces"
    n_ranked = len(ranked)
    take((layer, e) for e in range(ne) for layer in range(N_LAYER))   # the rest, across the layers
    write_profile(a.out, ranked, ne)
    assert read_profile(a.out, ne) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({what}, {len(ranked) - n_ranked} filled in)")


if __name__ == "__main__":
    main()
