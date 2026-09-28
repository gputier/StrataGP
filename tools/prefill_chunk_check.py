"""Read one prompt at several prompt-path chunk sizes and compare what each leaves behind (issue #9, audit B10).

bench/results/2026-09-28-prefill-speed takes "chunk 6144 against 8192, same code" as its yardstick: 89.8% top-1 and a
mean KL of 0.33 - a lot for a change of summation order.  If a chunk BOUNDARY changes the result (the QSA indexer's
tail block, the PLE history, the selection), the yardstick is noisy and every prefill A/B judged against it is too.
This reads the same prompt with each chunk size and compares, against the first one (or against the token-by-token
path with --token-path):

  * the GDN state hash after the prompt path, per recurrent layer (STRATA_STATE_HASH_GDN; bit-identical or not);
  * the logits of the last prompt positions (--tail N of them go through the token path, teacher-forced, with
    --prefill-until): same top-1, mean KL(ref || run), max |logit difference|;
  * the greedy tokens generated after the prompt.

The reference configuration is run twice first: a rerun must be bit-identical, or nothing below it means anything
(the adaptive expert tier is pinned with --adapt-every 100000 for that reason).

A native (IQ) pack runs its prompt tail in verify windows, which dump no logits, and --prefill-until would cut its
prompt short, so for it only the hash and the generated tokens are compared (the tool says so).

    python tools/prefill_chunk_check.py --config strata-q2_0.json
    python tools/prefill_chunk_check.py --config strata-q2_0.json --length 9000 --tail 256 --token-path
    python tools/prefill_chunk_check.py --exe build/strata -- --pack pack/full --ple-gguf M-2.gguf ...

The prompt is a deterministic pseudo-random id sequence (--length, --seed), or --ids FILE (commas or whitespace).
A chunk only splits the prompt when the prompt is longer than it: the default length, 9000, gives 2 chunks at 8192,
3 at 4096 and 9 at 1024 (a 2K prompt would read as ONE chunk at both 8192 and 4096).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# flags this tool sets itself: removed from the engine arguments (value-taking ones first)
OWN_FLAGS_WITH_VALUE = ("--prefill", "--prefill-until", "--tokens", "--tokens-file", "--max-new", "--dump-logits",
                        "--logits-stride", "--adapt-every", "--seed", "--temperature", "--top-k", "--top-p")
OWN_FLAGS = ("--serve", "--greedy", "--stop-eos")

HASH_RE = re.compile(r"GDN_HASH((?:\s+[0-9a-f]{4})+)")
CHUNKS_RE = re.compile(r"prefill (\d+) tokens in (\d+) chunks")
OUTPUT_RE = re.compile(r"^output\s*:(.*)$", re.M)


def strip_flags(args: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in OWN_FLAGS_WITH_VALUE:
            i += 2
            continue
        if a in OWN_FLAGS:
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def synthetic_prompt(length: int, seed: int) -> list[int]:
    """Deterministic ids with some structure: random runs, and earlier runs repeated (as real text repeats)."""
    rng = random.Random(seed)
    ids: list[int] = [248045]                       # <|im_start|>, as a chat prompt begins
    while len(ids) < length:
        if len(ids) > 64 and rng.random() < 0.2:
            start = rng.randrange(1, len(ids) - 32)
            ids += ids[start:start + rng.randrange(8, 32)]
        else:
            ids += [rng.randrange(200, 150000) for _ in range(rng.randrange(8, 48))]
    return ids[:length]


def read_ids(path: str) -> list[int]:
    return [int(t) for t in re.split(r"[\s,]+", Path(path).read_text().strip()) if t]


def read_logits(path: Path):
    """The dump's rows as a float32 array [rows, n_vocab].  The row count comes from the file size, not the header:
    the header counts every position, the token path writes only the ones it processed."""
    import numpy as np
    if not path.exists() or path.stat().st_size < 8:
        return None
    raw = path.read_bytes()
    n_vocab = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    body = np.frombuffer(raw[8:], dtype=np.float32)
    rows = body.size // n_vocab if n_vocab > 0 else 0
    if rows == 0:
        return None
    return body[:rows * n_vocab].reshape(rows, n_vocab)


def prompt_rows(logits, first_pos: int, n: int, tail: int):
    """The rows of positions n - tail .. n - 1 (the prompt's tail) of a dump whose first row is position first_pos;
    the rows after them are generated positions, which depend on the tokens each run generated."""
    if logits is None:
        return None
    lo, hi = n - tail - first_pos, n - first_pos
    if lo < 0 or hi > logits.shape[0]:
        return None
    return logits[lo:hi]


def compare_logits(ref, got) -> dict:
    """Same top-1, mean KL(ref || got) in float64, max |difference|, row by row (the same positions on both sides)."""
    import numpy as np
    n = min(ref.shape[0], got.shape[0])
    a = ref[:n].astype(np.float64)
    b = got[:n].astype(np.float64)
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    la -= np.log(np.exp(la).sum(axis=1, keepdims=True))
    lb -= np.log(np.exp(lb).sum(axis=1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    return {"rows": n, "top1": float((a.argmax(axis=1) == b.argmax(axis=1)).mean()), "kl_mean": float(kl.mean()),
            "kl_last": float(kl[-1]), "max_abs": float(np.abs(a - b).max()), "identical": bool(np.array_equal(a, b))}


def compare_hash(ref: list[str] | None, got: list[str] | None) -> str:
    if ref is None or got is None:
        return "n/a (no GDN_HASH line)"
    if ref == got:
        return f"identical ({len(ref)} layers)"
    diff = [i for i, (x, y) in enumerate(zip(ref, got)) if x != y]
    if len(ref) != len(got):
        return f"different layer count ({len(ref)} vs {len(got)})"
    return f"{len(diff)} of {len(ref)} layers differ, first at GDN layer {diff[0]}"


def compare_tokens(ref: list[int], got: list[int]) -> str:
    if ref == got:
        return f"identical ({len(ref)})"
    first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
    return f"differ from token {first} of {len(ref)}"


def run(exe: str, args: list[str], cwd: str | None, env: dict, label: str, workdir: Path, log, first_pos: int = 0,
        n: int = 0, tail: int = 1) -> dict:
    dump = workdir / (re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".logits")
    if dump.exists():
        dump.unlink()
    cmd = [exe] + args + ["--dump-logits", str(dump)]
    log(f"  running {label}: {' '.join(cmd[-8:])}")
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    (workdir / (dump.stem + ".log")).write_text(p.stdout + "\n---- stderr ----\n" + p.stderr)
    r: dict = {"label": label, "rc": p.returncode, "log": str(workdir / (dump.stem + ".log"))}
    m = HASH_RE.search(p.stderr)
    r["hash"] = m.group(1).split() if m else None
    m = CHUNKS_RE.search(p.stderr)
    r["chunks"] = int(m.group(2)) if m else None
    m = OUTPUT_RE.search(p.stdout)
    r["tokens"] = [int(t) for t in m.group(1).split()] if m else []
    r["logits"] = prompt_rows(read_logits(dump), first_pos, n, tail)
    r["native"] = "native pack:" in p.stderr
    return r


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    engine_args: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, engine_args = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="setup's strata-*.json (exe, args, cwd); engine arguments may also follow --")
    ap.add_argument("--exe", help="the strata binary (instead of --config)")
    ap.add_argument("--chunks", default="8192,4096,1024", help="prompt-path chunk sizes; the first is the reference")
    ap.add_argument("--length", type=int, default=9000, help="synthetic prompt length (default 9000)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ids", help="read the prompt ids from this file instead")
    ap.add_argument("--tail", type=int, default=1,
                    help="the last N prompt positions go through the token path and their logits are compared")
    ap.add_argument("--max-new", type=int, default=32, help="greedy tokens generated after the prompt")
    ap.add_argument("--token-path", action="store_true",
                    help="also read the whole prompt token by token and use it as the reference (slow: one "
                         "decode step per prompt token; canonical packs only)")
    ap.add_argument("--keep", help="keep the logs and dumps in this directory")
    a = ap.parse_args(argv)

    exe, cwd, base = a.exe, None, list(engine_args)
    if a.config:
        cfg = json.loads(Path(a.config).read_text())
        exe = exe or cfg["exe"]
        cwd = cfg.get("cwd")
        base = list(cfg.get("args", [])) + base
    if not exe:
        ap.error("--config or --exe is required")
    base = strip_flags(base) + ["--greedy", "--adapt-every", "100000"]
    ids = read_ids(a.ids) if a.ids else synthetic_prompt(a.length, a.seed)
    n = len(ids)
    chunks = [int(c) for c in a.chunks.split(",") if c]
    if n < 2 or not chunks:
        ap.error("need a prompt of 2+ tokens and at least one chunk size")
    tail = max(1, min(a.tail, n - 1))

    workdir = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="strata_chunk_"))
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = workdir / "prompt.ids"
    prompt.write_text(",".join(str(t) for t in ids))
    env = dict(os.environ, STRATA_STATE_HASH_GDN="1")
    print(f"prompt: {n} tokens ({'from ' + a.ids if a.ids else f'synthetic, seed {a.seed}'}); last {tail} position(s) "
          f"through the token path; {a.max_new} generated; files in {workdir}")

    common = base + ["--tokens-file", str(prompt), "--max-new", str(a.max_new)]
    # the prompt path reads positions [0, n - tail) and the token path the rest, whose logits land in the dump from
    # its first row; without --prefill-until the prompt path stops at n - 1 (tail 1)
    until = ["--prefill-until", str(n - tail)] if tail > 1 else []
    pos0 = n - tail
    ref_label = f"chunk {chunks[0]}"
    first = run(exe, common + ["--prefill", str(chunks[0])] + until, cwd, env, ref_label, workdir, print, pos0, n,
                tail)
    if first["native"] and tail > 1:
        print("  *** a native (IQ) pack: --prefill-until would cut the prompt short (its tail runs in verify windows, "
              "not teacher-forced); rerun with --tail 1")
        return 2
    runs = [first]
    runs.append(run(exe, common + ["--prefill", str(chunks[0])] + until, cwd, env, ref_label + " (rerun)", workdir,
                    print, pos0, n, tail))
    for c in chunks[1:]:
        runs.append(run(exe, common + ["--prefill", str(c)] + until, cwd, env, f"chunk {c}", workdir, print, pos0, n,
                        tail))
    if a.token_path and not first["native"]:
        runs.insert(0, run(exe, common, cwd, env, "token path", workdir, print, 0, n, tail))

    failed = [r for r in runs if r["rc"] != 0]
    for r in failed:
        print(f"  *** {r['label']} exited with {r['rc']}: see {r['log']}")
    ok = [r for r in runs if r["rc"] == 0]
    if not ok or first["rc"] != 0:
        return 1
    # logits and tokens against the token path when it ran, else the first chunk size; the hash (printed by the
    # prompt path only) always against the first chunk size
    ref = ok[0]
    print(f"\nreference: {ref['label']} (GDN hash: {ref_label})")
    if first["native"]:
        print("  (a native pack: its prompt tail runs in verify windows, which dump no logits - hash and tokens only)")
    print(f"{'run':<22} {'chunks':>6}  {'GDN hash':<46} {'top-1':>7} {'mean KL':>10} {'last KL':>10} "
          f"{'max |dlogit|':>12}  tokens")
    for r in ok:
        lg = compare_logits(ref["logits"], r["logits"]) if ref["logits"] is not None and r["logits"] is not None \
            else None
        cols = (f"{lg['top1'] * 100:6.1f}% {lg['kl_mean']:10.3e} {lg['kl_last']:10.3e} {lg['max_abs']:12.4g}"
                if lg else f"{'-':>7} {'-':>10} {'-':>10} {'-':>12}")
        chunks_s = str(r["chunks"]) if r["chunks"] is not None else "-"
        hash_s = compare_hash(first["hash"], r["hash"]) if r["label"] != "token path" else "-"
        print(f"{r['label']:<22} {chunks_s:>6}  {hash_s:<46} {cols}  {compare_tokens(ref['tokens'], r['tokens'])}")
    rerun = next((r for r in ok if r["label"].endswith("(rerun)")), None)
    if rerun and (rerun["hash"] != first["hash"] or rerun["tokens"] != first["tokens"] or
                  (rerun["logits"] is not None and first["logits"] is not None and
                   not compare_logits(first["logits"], rerun["logits"])["identical"])):
        print("\n*** the reference rerun is NOT bit-identical: the comparisons above measure run-to-run noise too")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
