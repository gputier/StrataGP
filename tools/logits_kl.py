"""tools/logits_kl.py - how far apart two --dump-logits files are, position by position (issue #53).

Two runs of the same prompt with the same --logits-stride, one per arm (e.g. without and with --gdn-state-bf16).
The prompt is teacher-forced, so row i of both files is the same position in the same context and the distance
between them is the arm's effect alone.  Prints KL(a || b) and the top-1 agreement over all rows and by quarter of
the rows: a drift that grows along the sequence shows as a KL that rises from one quarter to the next.

File format (generate.cpp): int32 n_vocab, int32 n_rows, then n_rows rows of n_vocab float32 logits.

    python tools/logits_kl.py A.bin B.bin [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_logits(path):
    """(n_rows, n_vocab) float32, memory-mapped."""
    p = Path(path)
    n_vocab, n_rows = (int(x) for x in np.fromfile(p, dtype="<i4", count=2))
    if n_vocab <= 0 or n_rows < 0 or p.stat().st_size != 8 + 4 * n_vocab * n_rows:
        raise SystemExit(f"{path}: not a --dump-logits file (header {n_vocab} x {n_rows}, {p.stat().st_size} bytes)")
    return np.memmap(p, dtype="<f4", mode="r", offset=8, shape=(n_rows, n_vocab))


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def compare(a, b):
    """Per row: KL(softmax a || softmax b) in nats, and whether the argmaxes agree."""
    if a.shape != b.shape:
        raise SystemExit(f"the two dumps differ in shape: {a.shape} vs {b.shape} (same prompt, same --logits-stride?)")
    kl = np.zeros(a.shape[0])
    top1 = np.zeros(a.shape[0], bool)
    for i in range(a.shape[0]):
        la, lb = log_softmax(a[i]), log_softmax(b[i])
        kl[i] = max(0.0, float((np.exp(la) * (la - lb)).sum()))
        top1[i] = int(np.argmax(a[i])) == int(np.argmax(b[i]))
    return kl, top1


def summary(kl, top1):
    n = kl.size
    out = {"rows": int(n)}
    if n == 0:
        return out
    out.update(kl_mean=float(kl.mean()), kl_max=float(kl.max()), kl_p99=float(np.quantile(kl, 0.99)),
               top1=float(top1.mean()), first_disagreement=int(np.argmin(top1)) if not top1.all() else None)
    q = []
    for j in range(4):
        lo, hi = j * n // 4, (j + 1) * n // 4
        if hi > lo:
            q.append({"rows": [lo, hi], "kl_mean": float(kl[lo:hi].mean()), "top1": float(top1[lo:hi].mean())})
    out["quarters"] = q
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="the reference arm's --dump-logits file")
    ap.add_argument("b", help="the other arm's")
    ap.add_argument("--json", help="write the numbers to this file")
    args = ap.parse_args(argv)
    kl, top1 = compare(read_logits(args.a), read_logits(args.b))
    s = summary(kl, top1)
    if s["rows"] == 0:
        print("logits_kl: no rows")
        return 1
    print(f"logits_kl: {s['rows']} rows; KL(a||b) mean {s['kl_mean']:.3e}  p99 {s['kl_p99']:.3e}  max {s['kl_max']:.3e}"
          f"  nats; top-1 agreement {s['top1']:.4f}")
    if s["first_disagreement"] is not None:
        print(f"  first row whose argmax differs: {s['first_disagreement']}")
    for q in s["quarters"]:
        print(f"  rows {q['rows'][0]:>6} .. {q['rows'][1]:<6} KL mean {q['kl_mean']:.3e}  top-1 {q['top1']:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
