"""tools/qwen35/parity.py - NEW-MODELS-PLAN.md T1's exit gate for Qwen3.6-35B-A3B: strata-q36ref against llama.cpp
(tools/qwen35/q36_llama_dump.cpp) on the same GGUF, the same box and the same token ids.

For each prompt (parity_prompts.json, ChatML with thinking off): llama.cpp tokenizes it and runs it (the reference
ids, layer outputs and greedy tokens); strata-q36ref runs the same ids.  Reported per prompt: the greedy tokens that
agree (of --n), the first divergence with both engines' top-1/top-2 logit margins (a near tie is a margin < 0.05), and
the per-layer hidden-state error max over positions of ||strata - llama|| / ||llama||.

    python tools/qwen35/parity.py --ref BIN --dump BIN --model GGUF --out DIR [--n 64] [--ngl 99] [--lock "flock ..."]

--logits-only compares the full logits of every step instead of the layer outputs: strata-q36 (the engine path) has no
layer dump, and strata-q36ref keeps the whole model resident, so a file larger than the card needs the engine path.
--ref-args passes extra arguments to the strata binary (--max-context, --kv, ...).  The layer mode reads the geometry
from --n-embd / --n-layer (Qwen3.6-35B-A3B: 2048 / 40); --vocab is the logits row size (--logits-only).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EOS = 248046                                    # <|im_end|>


def run(cmd: list[str], lock: str, log: Path) -> str:
    full = shlex.split(lock) + cmd if lock else cmd
    r = subprocess.run(full, capture_output=True, text=True, timeout=1800)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")
    return r.stdout


def ids_of(out: str, key: str) -> list[int]:
    for line in out.splitlines():
        if line.startswith(key + ":"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no '{key}:' line")


def top2(path: Path, n_vocab: int) -> np.ndarray:
    """[step][id1, logit1, id2, logit2] of a full-logits file ([step][vocab] f32)."""
    lg = np.fromfile(path, dtype=np.float32).reshape(-1, n_vocab)
    i2 = np.argpartition(-lg, 1, axis=1)[:, :2]
    v2 = np.take_along_axis(lg, i2, axis=1)
    o = np.argsort(-v2, axis=1)
    i2, v2 = np.take_along_axis(i2, o, axis=1), np.take_along_axis(v2, o, axis=1)
    return np.stack([i2[:, 0], v2[:, 0], i2[:, 1], v2[:, 1]], axis=1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="strata-q36ref")
    ap.add_argument("--dump", required=True, help="q36_llama_dump")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ngl", type=int, default=99, help="llama.cpp's GPU layers (0 = its CPU backend)")
    ap.add_argument("--lock", default="", help="a command prefix for the GPU runs (the box's flock)")
    ap.add_argument("--prompts", default=str(HERE / "parity_prompts.json"))
    ap.add_argument("--skip-ref", action="store_true", help="reuse strata-q36ref's outputs in --out")
    ap.add_argument("--n-embd", type=int, default=2048, help="the model's hidden size (layer mode)")
    ap.add_argument("--n-layer", type=int, default=40, help="the model's layer count (layer mode)")
    ap.add_argument("--vocab", type=int, default=248320, help="the logits row size (--logits-only)")
    ap.add_argument("--logits-only", action="store_true", help="compare the logits of every step, no layer outputs")
    ap.add_argument("--ref-args", default="", help="extra arguments for the strata binary")
    ap.add_argument("--past-eos", action="store_true", help="compare all --n tokens, not only up to the answer's end")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    summary = []
    for i, p in enumerate(prompts):
        tag = f"p{i:02d}"
        dumps = (["--dump-full-logits", str(out / f"{tag}.llama.full")] if a.logits_only else
                 ["--dump-layers", str(out / f"{tag}.llama.layers"), "--dump-logits", str(out / f"{tag}.llama.logits")])
        lo = run([a.dump, "--model", a.model, "--prompt", TEMPLATE.format(p=p), "--n", str(a.n), "--ngl", str(a.ngl),
                  *dumps], a.lock, out / f"{tag}.llama.log")
        prompt_ids, lgen = ids_of(lo, "prompt"), ids_of(lo, "gen")
        if not a.skip_ref:
            dumps = (["--dump-full-logits", str(out / f"{tag}.strata.full")] if a.logits_only else
                     ["--dump-layers", str(out / f"{tag}.strata.layers"), "--dump-logits", str(out / f"{tag}.strata.logits")])
            so = run([a.ref, "--model", a.model, "--tokens", ",".join(map(str, prompt_ids)), "--n", str(a.n), *dumps,
                      *shlex.split(a.ref_args)], a.lock, out / f"{tag}.strata.log")
            (out / f"{tag}.strata.gen").write_text(so, encoding="utf-8")
        sgen = ids_of((out / f"{tag}.strata.gen").read_text(encoding="utf-8"), "gen")
        # the answer ends at llama.cpp's first <|im_end|> (included); past it both engines run on in a context no chat
        # ever reaches, so the comparison stops there (--past-eos compares all --n tokens)
        upto = min(len(sgen), len(lgen))
        if not a.past_eos and EOS in lgen[:upto]:
            upto = lgen.index(EOS) + 1
        same = 0
        while same < upto and sgen[same] == lgen[same]:
            same += 1
        if a.logits_only:
            sl, ll = (top2(out / f"{tag}.{k}.full", a.vocab) for k in ("strata", "llama"))
        else:
            sl = np.fromfile(out / f"{tag}.strata.logits", dtype=np.float32).reshape(-1, 4)
            ll = np.fromfile(out / f"{tag}.llama.logits", dtype=np.float32).reshape(-1, 4)
        div = None
        if same < upto:
            div = {"step": same, "strata": [int(sl[same, 0]), float(sl[same, 1] - sl[same, 3])],
                   "llama": [int(ll[same, 0]), float(ll[same, 1] - ll[same, 3])]}
        rec = {"prompt": i, "tokens": len(prompt_ids), "same": same, "of": upto, "divergence": div,
               "max_rel_err": 0.0, "worst_layer": -1, "median_rel_err": 0.0}
        line = f"{tag}: {same}/{upto} greedy tokens equal"
        if not a.logits_only:
            # the layer outputs up to the divergence (after it the two engines read different tokens)
            sL = np.fromfile(out / f"{tag}.strata.layers", dtype=np.float32).reshape(-1, a.n_layer, a.n_embd)
            lL = np.fromfile(out / f"{tag}.llama.layers", dtype=np.float32).reshape(-1, a.n_layer, a.n_embd)
            npos = min(len(sL), len(lL), len(prompt_ids) + same)
            ln = np.linalg.norm(lL[:npos], axis=2)
            err = np.linalg.norm(sL[:npos] - lL[:npos], axis=2) / np.maximum(ln, 1e-30)   # [pos, layer]
            # llama.cpp keeps only the output rows of the last layer of a prompt batch: rows it did not compute are 0
            err = np.where(ln > 0, err, np.nan)
            per_layer = np.nanmax(err, axis=0)
            rec.update(max_rel_err=float(np.nanmax(per_layer)), worst_layer=int(np.nanargmax(per_layer)),
                       median_rel_err=float(np.nanmedian(err)), per_layer_max=[round(float(x), 5) for x in per_layer])
            line += (f"; layers max rel err {rec['max_rel_err']:.4f} (layer {rec['worst_layer']}), median "
                     f"{rec['median_rel_err']:.5f}")
        summary.append(rec)
        print(line + (f"; diverges at {div}" if div else ""), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    full = sum(r["same"] == r["of"] for r in summary)
    print(f"identical greedy answers (up to {a.n} tokens{'' if a.past_eos else ' or the end of the answer'}): {full} of "
          f"{len(summary)} prompts"
          + ("" if a.logits_only else f"; worst layer error {max(r['max_rel_err'] for r in summary):.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
