"""A/B of engine settings on one-shot runs (issues #50, #52): greedy, 256 tokens, --adapt-every 100000, the three
calibration prompts, N runs, the variants interleaved within each run.

    python tools/ab_oneshot.py strata-q2_0.json --runs 3 --variant base= --variant spec5="--spec 5" \\
                               --variant spec6="--spec 6"
    python tools/ab_oneshot.py strata-q2_0.json --variant grid= --variant exact="--mtp /path/to/rt-exact"

A variant is NAME=FLAGS: the run config's engine arguments with FLAGS set (a flag already there gets the new
value; a flag followed by another flag, or last, is a switch and is added).  Per variant: the median decode
tok/s over every run and prompt, the median tokens per round and the draft acceptance, and whether each prompt's
greedy output is the first variant's, token for token (speculative decoding must not change a greedy answer).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate as CAL  # noqa: E402

FIXED = ["--max-new", "256", "--greedy", "--adapt-every", "100000"]
DECODE = re.compile(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)
SPEC = re.compile(r"drafts accepted (\d+) of (\d+) \(([\d.]+)\), ([\d.]+) tokens per round")
OUTPUT = re.compile(r"^output\s*:(.*)$", re.M)


def set_flags(args: list[str], flags: list[str]) -> list[str]:
    out, i = list(args), 0
    while i < len(flags):
        flag = flags[i]
        if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            out = CAL.with_arg(out, flag, flags[i + 1])
            i += 2
        else:
            if flag not in out:
                out.append(flag)
            i += 1
    return out


def prompt_ids(cfg: dict) -> list[list[int]]:
    import strata_tokenizer as ST
    t = Path(cfg["tokenizer"])
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for s, i in vocab.items():
        toks[i] = s
    tok = ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((t / "token_type.json").read_text()))
    return [CAL.chat_ids(tok, p) for p in CAL.PROMPTS]


def parse(text: str) -> dict:
    d, s, o = DECODE.search(text), SPEC.search(text), OUTPUT.search(text)
    if d is None or o is None:
        raise RuntimeError("the engine's output has no decode/output line:\n" + text[-2000:])
    return {"tok_s": float(d.group(3)), "tokens_per_round": float(s.group(4)) if s else None,
            "accepted": int(s.group(1)) if s else 0, "offered": int(s.group(2)) if s else 0,
            "output": o.group(1).split()}


def split_flags(flags: str) -> list[str]:
    """FLAGS as arguments.  On Windows the non-POSIX split keeps the backslashes of a path (C:\\Strata\\rt-exact);
    it keeps the quotes around a quoted token too, so they are removed here."""
    if os.name != "nt":
        return shlex.split(flags)
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"" else t
            for t in shlex.split(flags, posix=False)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="a run config written by setup (strata-*.json)")
    ap.add_argument("--variant", action="append", required=True, help="NAME=FLAGS, e.g. spec5=\"--spec 5\"")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", help="also write every run to this file")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    from serve.server import child_env
    env = child_env(cfg)
    variants = []
    for v in a.variant:
        name, _, flags = v.partition("=")
        variants.append((name, set_flags(cfg["args"], split_flags(flags) + FIXED)))
    ids = prompt_ids(cfg)
    runs = {name: [] for name, _ in variants}
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for k, p in enumerate(ids):
            files.append(Path(tmp) / f"p{k}.ids")
            files[-1].write_text(",".join(str(t) for t in p))
        for r in range(a.runs):
            for name, args in variants:
                for k, f in enumerate(files):
                    res = subprocess.run([cfg["exe"], *args, "--tokens-file", str(f)], cwd=cfg.get("cwd"), env=env,
                                         capture_output=True, text=True, encoding="utf-8", errors="replace")
                    if res.returncode != 0:
                        raise SystemExit(f"{name}: the engine failed ({res.returncode}):\n{res.stderr[-2000:]}")
                    row = {"run": r, "prompt": k, **parse(res.stdout)}
                    runs[name].append(row)
                    print(f"  run {r + 1} {name:>10} prompt {k}: {row['tok_s']:.2f} tok/s, "
                          f"{row['tokens_per_round']} tokens/round", flush=True)
    first = variants[0][0]
    print(f"\n{'variant':>10} {'tok/s median':>12} {'tokens/round':>12} {'acceptance':>10}  same greedy output as "
          f"{first}")
    for name, _ in variants:
        rows = runs[name]
        tpr = [x["tokens_per_round"] for x in rows if x["tokens_per_round"] is not None]
        off = sum(x["offered"] for x in rows)
        same = all(x["output"] == runs[first][i]["output"] for i, x in enumerate(rows))
        print(f"{name:>10} {statistics.median(x['tok_s'] for x in rows):>12.2f} "
              f"{statistics.median(tpr) if tpr else float('nan'):>12.2f} "
              f"{sum(x['accepted'] for x in rows) / off if off else float('nan'):>10.3f}  {'yes' if same else 'NO'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"variants": dict(variants), "runs": runs}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
