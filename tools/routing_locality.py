"""tools/routing_locality.py - how predictable the routed experts are from what came before (issue #51, S3).

Copying layer l+1's experts over PCIe (or prefetching them into the CPU's L3) BEFORE its router has run only pays
if they can be guessed.  This reads `--dump-routing` traces and measures, per layer and overall:

  * inter-token reuse: the share of token t+1's experts at layer l that the last W tokens also routed at layer l,
    and how many distinct experts those W tokens hold (the copies keeping them would cost);
  * next-layer prediction: the share of token t+1's experts at layer l+1 that fall in a predicted set of B experts:
      prev    token t's experts at layer l+1, known a whole token ahead (then the most frequent, up to B);
      xlayer  from token t+1's experts at layer l through P(expert at l+1 | expert at l), a table learned on the
              first part of each trace (--train-frac) and averaged over the k experts: known one layer ahead,
              once layer l's router has run;
      both    xlayer's estimate and prev's (token t's experts at the learned reuse rate) as a noisy OR;
      freq    the B most frequent experts at layer l+1: a static profile, what the VRAM cache already does.
    Chance is B / n_expert.  Everything is measured on the positions after the learning part;
  * with --resident N: the same restricted to the experts outside a static cache of the N most used (layer, expert)
    pairs of the learning part - the ones a prefetch would have to copy.  B is then the copies issued per layer and
    token (among non-resident experts only), and the result is the share of the misses they cover and their
    precision (useful copies / copies).

THE TRACE: a one-shot run WITHOUT --spec.  `drive_pool` (generate.cpp) writes one record per layer per position of
the token path; the verify window of --spec does not write it, so a --spec run traces its first token only.  The
prompt is traced too when it goes through the token path (no --prefill).  Record: int32 layer, int32 k, k int32
expert ids, k float32 router weights.  A position is a run of records whose layer index increases.

    python tools/routing_locality.py TRACE [TRACE ...] [--budget 2,5,10,20,40] [--window 1,2,4,8] [--train-frac 0.5]
                                     [--resident N] [--skip N] [--smooth 8] [--per-layer] [--json OUT]
                                     [--n-layer 48] [--n-expert 512]      (a pruned model: GSQ-RCO Coder keeps 256)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

N_LAYER, N_EXPERT = 48, 512
PREDICTORS = ("prev", "xlayer", "both", "freq")
CHUNK = 4096   # positions scored at once


def read_trace(path, n_layer=N_LAYER, n_expert=N_EXPERT):
    """(ids, weights, dropped, tail): ids (positions, n_layer, k) int32 with -1 for a layer the trace lacks at a
    position, weights (positions, n_layer, k) float32, the records dropped (layer or id out of range, another k)
    and the bytes left after the last whole record (a trace cut short)."""
    blob = Path(path).read_bytes()
    positions, cur, last, k0, off, dropped = [], None, None, None, 0, 0
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        if k < 1 or k > 64 or off + 8 + 8 * k > len(blob):
            break
        ids = struct.unpack_from("<%di" % k, blob, off + 8)
        w = struct.unpack_from("<%df" % k, blob, off + 8 + 4 * k)
        off += 8 + 8 * k
        if k0 is None:
            k0 = k
        if k != k0 or not 0 <= layer < n_layer or min(ids) < 0 or max(ids) >= n_expert:
            dropped += 1
            continue
        if cur is None or layer <= last:
            cur = {}
            positions.append(cur)
        cur[layer] = (ids, w)
        last = layer
    k = k0 or 1
    I = np.full((len(positions), n_layer, k), -1, np.int32)
    W = np.zeros((len(positions), n_layer, k), np.float32)
    for p, rec in enumerate(positions):
        for layer, (ids, w) in rec.items():
            I[p, layer] = ids
            W[p, layer] = w
    return I, W, dropped, len(blob) - off


def hot(ids, weights, n_expert):
    """(positions, k) ids -> (positions, n_expert) membership (bool) and router weight (float64); -1 ids are nobody."""
    P = ids.shape[0]
    col = np.where(ids < 0, n_expert, ids)
    m = np.zeros((P, n_expert + 1), bool)
    w = np.zeros((P, n_expert + 1), np.float64)
    rows = np.arange(P)[:, None]
    m[rows, col] = True
    w[rows, col] = weights
    return m[:, :n_expert], w[:, :n_expert]


class Acc:
    """Sums over (position, layer) samples: hits, weighted hits, samples, and whatever else is added by name."""

    def __init__(self):
        self.s = {}

    def add(self, key, v):
        self.s[key] = self.s.get(key, 0.0) + float(v)

    def get(self, key):
        return self.s.get(key, 0.0)

    def ratio(self, num, den):
        d = self.get(den)
        return self.get(num) / d if d > 0 else None


def resident_set(traces, n_train, n_layer, n_expert, n):
    """The static cache: the n most used (layer, expert) pairs of the learning part, as make_profile.py ranks them
    (count, then layer, then expert).  (n_layer, n_expert) bool."""
    counts = np.zeros((n_layer, n_expert), np.int64)
    for (I, _), nt in zip(traces, n_train):
        for layer in range(n_layer):
            ids = I[:nt, layer].reshape(-1)
            ids = ids[ids >= 0]
            counts[layer] += np.bincount(ids, minlength=n_expert)
    flat = counts.reshape(-1)
    order = np.lexsort((np.arange(flat.size), -flat))           # count desc, then (layer, expert) asc
    keep = order[:n]
    keep = keep[flat[keep] > 0]
    r = np.zeros(flat.size, bool)
    r[keep] = True
    return r.reshape(n_layer, n_expert)


def measure(traces, n_layer=N_LAYER, n_expert=N_EXPERT, budgets=(2, 5, 10, 20, 40), windows=(1, 2, 4, 8),
            train_frac=0.5, resident=0, skip=0, smooth=8.0):
    """traces: [(ids, weights)] as read_trace returns them.  Returns the result dict main() prints."""
    traces = [(I[skip:], W[skip:]) for I, W in traces]
    k = max((I.shape[2] for I, _ in traces), default=1)
    n_train = [int(round(I.shape[0] * train_frac)) for I, _ in traces]
    budgets = sorted(set(int(b) for b in budgets if 0 < int(b) <= n_expert))
    windows = sorted(set(int(w) for w in windows if int(w) > 0))
    res = resident_set(traces, n_train, n_layer, n_expert, resident) if resident > 0 else None
    reuse = [Acc() for _ in range(n_layer)]
    pred = [Acc() for _ in range(n_layer)]

    # ---- inter-token reuse, same layer
    for (I, W), nt in zip(traces, n_train):
        P = I.shape[0]
        first = max(nt, 1)
        for layer in range(n_layer):
            m, w = hot(I[:, layer], W[:, layer], n_expert)
            valid = (I[:, layer] >= 0).all(axis=1)
            cnt = np.vstack([np.zeros((1, n_expert), np.int32), np.cumsum(m, axis=0, dtype=np.int32)])
            vcnt = np.concatenate([[0], np.cumsum(valid)])
            for win in windows:
                t = np.arange(max(first, win), P)
                t = t[valid[t] & (vcnt[t] - vcnt[t - win] == win)]      # the target and its whole window traced
                if t.size == 0:
                    continue
                union = (cnt[t] - cnt[t - win]) > 0
                a = reuse[layer]
                a.add(f"n{win}", t.size)
                a.add(f"hit{win}", (union & m[t]).sum() / k)
                a.add(f"whit{win}", ((w[t] * union).sum(axis=1) / np.maximum(w[t].sum(axis=1), 1e-30)).sum())
                a.add(f"size{win}", union.sum())

    # ---- next layer: token t+1's experts at layer l+1
    bmax = max(budgets) if budgets else 0
    for layer in range(n_layer - 1):
        tgt_l = layer + 1
        per = []
        for (I, W), nt in zip(traces, n_train):
            src, _ = hot(I[:, layer], W[:, layer], n_expert)
            tgt, tw = hot(I[:, tgt_l], W[:, tgt_l], n_expert)
            ok = (I[:, layer] >= 0).all(axis=1) & (I[:, tgt_l] >= 0).all(axis=1)
            per.append((src, tgt, tw, ok, nt))
        # the tables, from the learning part of every trace
        trans = np.zeros((n_expert, n_expert), np.float64)
        seen = np.zeros(n_expert, np.float64)
        freq = np.zeros(n_expert, np.float64)
        kept, kept_n, n_rows = 0.0, 0, 0
        for src, tgt, _, ok, nt in per:
            rows = np.nonzero(ok[:nt])[0]
            s, g = src[rows].astype(np.float64), tgt[rows].astype(np.float64)
            trans += s.T @ g
            seen += s.sum(axis=0)
            freq += g.sum(axis=0)
            n_rows += rows.size
            rows = rows[(rows >= 1) & ok[np.maximum(rows - 1, 0)]]
            kept += (tgt[rows] & tgt[rows - 1]).sum()
            kept_n += rows.size
        # P(e' at l+1 | e at l), shrunk toward e''s base rate by `smooth` pseudo-counts: an expert seen twice
        # would otherwise predict its two companions with probability 1/2 each
        base = freq / max(1, n_rows)
        cond = (trans + smooth * base[None, :]) / (seen + smooth)[:, None]
        # P(e' at l+1 of token t+1 | e' at l+1 of token t), and for the others the rest of k spread evenly
        r_in = kept / (kept_n * k) if kept_n else 0.0
        r_out = (1.0 - r_in) * k / max(1, n_expert - k)
        tie = 1e-9 * freq / (freq.max() + 1.0)                        # ties go to the more frequent, then the lower id
        a = pred[tgt_l]
        for src, tgt, tw, ok, nt in per:
            ts = np.arange(max(nt, 1), src.shape[0])
            ts = ts[ok[ts] & ok[ts - 1]]
            if bmax == 0:
                continue
            for c in range(0, ts.size, CHUNK):                            # bounded memory on long traces
                t = ts[c:c + CHUNK]
                # P(e' routed) as the mean of the k source experts' estimates (a sum would count the same
                # evidence k times over), then combined with the previous token's as a noisy OR
                xl = (src[t].astype(np.float64) @ cond) / k
                prev = np.where(tgt[t - 1], r_in, r_out)
                keys = {"prev": tgt[t - 1] + tie, "xlayer": xl + tie, "both": 1.0 - (1.0 - xl) * (1.0 - prev) + tie,
                        "freq": np.broadcast_to(tie, (t.size, n_expert))}
                m, w = tgt[t], tw[t]
                wsum = np.maximum(w.sum(axis=1), 1e-30)
                a.add("n", t.size)
                if res is not None:
                    a.add("miss", (m & ~res[tgt_l]).sum())
                    a.add("chance_den", t.size * max(1, n_expert - int(res[tgt_l].sum())))
                for name, key in keys.items():
                    key = np.array(key, np.float64)
                    if res is not None:
                        key[:, res[tgt_l]] = -np.inf                   # a resident expert needs no copy
                    order = np.argsort(-key, axis=1, kind="stable")[:, :bmax]
                    live = np.isfinite(np.take_along_axis(key, order, axis=1))
                    got = np.take_along_axis(m, order, axis=1) & live   # with a cache: only misses are live
                    gotw = np.take_along_axis(w, order, axis=1) * live
                    for b in budgets:
                        a.add(f"{name}:{b}", got[:, :b].sum() if res is not None else got[:, :b].sum() / k)
                        a.add(f"{name}:w{b}", (gotw[:, :b].sum(axis=1) / wsum).sum())
                        a.add(f"{name}:copies{b}", live[:, :b].sum())

    # ---- the report
    out = {"traces": len(traces), "positions": int(sum(I.shape[0] for I, _ in traces)),
           "learn_positions": int(sum(n_train)), "n_layer": n_layer, "n_expert": n_expert, "k": k,
           "train_frac": train_frac, "smooth": smooth, "budgets": budgets, "windows": windows, "resident": resident}

    def total(accs, key):
        return sum(x.get(key) for x in accs)

    out["reuse"] = {}
    for win in windows:
        n = total(reuse, f"n{win}")
        out["reuse"][str(win)] = None if n == 0 else {
            "reuse": total(reuse, f"hit{win}") / n, "weighted": total(reuse, f"whit{win}") / n,
            "window_experts": total(reuse, f"size{win}") / n, "chance": 1.0 - (1.0 - k / n_expert) ** win,
            "samples": int(n)}
    n = total(pred, "n")
    out["next_layer"] = {}
    if res is None:
        for b in budgets:
            out["next_layer"][str(b)] = None if n == 0 else dict(
                {p: total(pred, f"{p}:{b}") / n for p in PREDICTORS},
                **{p + "_weighted": total(pred, f"{p}:w{b}") / n for p in PREDICTORS},
                chance=min(1.0, b / n_expert), samples=int(n))
    else:
        miss = total(pred, "miss")
        out["resident_pairs"] = int(res.sum())
        out["misses_per_layer_token"] = miss / n if n else None
        out["resident_hit_rate"] = 1.0 - miss / (n * k) if n else None
        for b in budgets:
            row = {"samples": int(n)}
            for p in PREDICTORS:
                copies = total(pred, f"{p}:copies{b}")
                row[p] = total(pred, f"{p}:{b}") / miss if miss else None
                row[p + "_precision"] = total(pred, f"{p}:{b}") / copies if copies else None
            row["chance"] = min(1.0, b * n / total(pred, "chance_den")) if n else None
            out["next_layer"][str(b)] = row
    out["per_layer"] = []
    w1 = windows[0] if windows else None
    bk = min(budgets, key=lambda b: abs(b - k)) if budgets else None
    for layer in range(n_layer):
        r, p = reuse[layer], pred[layer]
        row = {"layer": layer}
        if w1 is not None:
            row["reuse"] = r.ratio(f"hit{w1}", f"n{w1}")
        if bk is not None:
            for q in PREDICTORS:
                row[q] = p.ratio(f"{q}:{bk}", "n" if res is None else "miss")
        out["per_layer"].append(row)
    out["per_layer_window"], out["per_layer_budget"] = w1, bk
    return out


def fmt(v, nd=3):
    return "   -  " if v is None else f"{v:.{nd}f}"


def print_report(r, paths, notes):
    print(f"routing_locality: {r['traces']} trace(s), {r['positions']} positions ({r['learn_positions']} learn the "
          f"tables, the rest are measured), {r['n_layer']} layers, top-{r['k']} of {r['n_expert']} experts")
    for p, (dropped, tail) in zip(paths, notes):
        if dropped or tail:
            print(f"  {p}: {dropped} records dropped (out of range), {tail} trailing bytes (a cut trace)")
    print("\ninter-token reuse: token t+1's experts at layer l already routed at layer l by the last W tokens")
    print("  W    reuse   by weight   experts in the window   chance")
    for w in r["windows"]:
        x = r["reuse"][str(w)]
        if x is None:
            print(f"  {w:<4} (no positions)")
            continue
        print(f"  {w:<4} {fmt(x['reuse'])}   {fmt(x['weighted'])}       {x['window_experts']:7.1f}            "
              f"{fmt(x['chance'])}")
    if r["resident"] <= 0:
        print("\nnext layer: token t+1's experts at layer l+1 (l+1 = 1..%d) inside a predicted set of B experts"
              % (r["n_layer"] - 1))
        print("  B     prev            xlayer          both            freq            chance   (by weight)")
        for b in r["budgets"]:
            x = r["next_layer"][str(b)]
            if x is None:
                print(f"  {b:<5} (no positions)")
                continue
            cells = "  ".join(f"{fmt(x[p])} ({fmt(x[p + '_weighted'], 2)})" for p in PREDICTORS)
            print(f"  {b:<5} {cells}  {fmt(x['chance'])}")
    else:
        print(f"\noutside the static cache of the {r['resident_pairs']} most used (layer, expert) pairs: hit rate "
              f"{fmt(r['resident_hit_rate'])}, {fmt(r['misses_per_layer_token'], 2)} misses per layer and token")
        print("share of those misses covered by B copies per layer and token (precision = useful copies / copies)")
        print("  B     prev            xlayer          both            freq            chance")
        for b in r["budgets"]:
            x = r["next_layer"][str(b)]
            cells = "  ".join(f"{fmt(x[p])} ({fmt(x[p + '_precision'], 2)})" for p in PREDICTORS)
            print(f"  {b:<5} {cells}  {fmt(x['chance'])}")
    print("  prev = token t's experts at l+1 (a token ahead); xlayer = P(l+1 | token t+1's experts at l), learned on "
          f"the first {r['train_frac']:.0%}")
    print("  (a layer ahead); both = the two as a noisy OR; freq = the most frequent at l+1 (a static profile)")


def print_layers(r):
    w, b = r["per_layer_window"], r["per_layer_budget"]
    what = "share of the misses" if r["resident"] > 0 else "recall"
    print(f"\nper layer: reuse at W={w}, and next-layer {what} at B={b}")
    print("  layer  reuse   prev    xlayer  both    freq")
    for row in r["per_layer"]:
        print(f"  {row['layer']:<6} {fmt(row.get('reuse'))}  " + "  ".join(fmt(row.get(p)) for p in PREDICTORS))


def int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="routing traces from --dump-routing")
    ap.add_argument("--budget", type=int_list, default=[2, 5, 10, 20, 40], help="predicted-set sizes (default 2,5,10,20,40)")
    ap.add_argument("--window", type=int_list, default=[1, 2, 4, 8], help="reuse windows in tokens (default 1,2,4,8)")
    ap.add_argument("--train-frac", type=float, default=0.5, help="share of each trace that learns the tables (default 0.5)")
    ap.add_argument("--resident", type=int, default=0, help="measure outside a static cache of N (layer, expert) pairs")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N positions of each trace")
    ap.add_argument("--smooth", type=float, default=8.0,
                    help="pseudo-counts shrinking the cross-layer table toward the base rates (default 8)")
    ap.add_argument("--per-layer", action="store_true", help="also print the per-layer table")
    ap.add_argument("--json", help="write every number to this file")
    ap.add_argument("--n-layer", type=int, default=N_LAYER)
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    a = ap.parse_args(argv)
    if not 0.0 < a.train_frac < 1.0:
        ap.error("--train-frac must be in (0, 1)")
    loaded, notes = [], []
    for p in a.traces:
        I, W, dropped, tail = read_trace(p, a.n_layer, a.n_expert)
        loaded.append((I, W))
        notes.append((dropped, tail))
    if sum(I.shape[0] for I, _ in loaded) < 2:
        print("routing_locality: fewer than two positions in the traces (was --spec on? see the help)", file=sys.stderr)
        return 1
    r = measure(loaded, a.n_layer, a.n_expert, a.budget, a.window, a.train_frac, a.resident, a.skip, a.smooth)
    print_report(r, a.traces, notes)
    if a.per_layer:
        print_layers(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
