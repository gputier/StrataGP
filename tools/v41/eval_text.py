#!/usr/bin/env python3
"""eval_text.py - the held-out text for the quality measure: wikitext-2 validation, checked against the corpus.

The calibration corpus (build_corpus.py) holds wikitext-2-raw-v1 **test**, and the importance matrix was computed on
it, so a perplexity measured on test would be optimistic.  This script writes wikitext-2-raw-v1 **validation**
(`fetch_wikitext(path, split="validation")` of build_corpus.py, the same Hugging Face datasets server as the
corpus) as the plain text `llama-perplexity -f` reads, and proves it shares nothing with the corpus halves: no
paragraph of more than 80 characters of the validation text appears in `half_a.txt` or `half_b.txt` (compared with
the whitespace collapsed).  One overlap and the script stops with exit code 2 and writes no text.

The text is the rows of the dataset joined, an empty row becoming a bare newline (the convention of
build_corpus.load_wikitext).  Its sha256 is printed and written in the report, so every perplexity run can be tied
to the exact bytes.  `--control-rows` runs the same check with the test rows, which the corpus does hold: it must
find overlaps, otherwise the check cannot see and the script stops (exit code 2).

With `--tokenizer` (tokenizer.json of DeepSeek-V4.1, `tokenizers` library) it counts the tokens and the chunks of
`--n-ctx` tokens `llama-perplexity` cuts (`tokens // n_ctx`, one BOS token counted in front), and says whether the
`--chunks` wanted (45) are there.  If not it only reports the numbers.

    python3 -I tools/v41/eval_text.py --out-dir out --half-a half_a.txt --half-b half_b.txt --fetch
        [--rows wiki.validation.jsonl] [--control-rows wiki.test.jsonl] [--tokenizer tokenizer.json]
        [--n-ctx 2048] [--chunks 45]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_corpus  # noqa: E402

MIN_PARAGRAPH_CHARS = 80
N_CTX = 2048
CHUNKS_WANTED = 45
SPLIT = "validation"


class OverlapError(RuntimeError):
    """The held-out text shares a paragraph with the corpus."""


def read_rows(path):
    """The texts of a jsonl of dataset rows, in order."""
    return [json.loads(line)["text"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def to_text(rows):
    """The plain text of the rows: joined, an empty row being a bare newline."""
    return "".join(r if r else "\n" for r in rows)


def squash(text):
    """Whitespace collapsed to single spaces, so a different line wrapping cannot hide a shared paragraph."""
    return " ".join(text.split())


def long_paragraphs(rows, min_chars=MIN_PARAGRAPH_CHARS):
    """The distinct rows longer than `min_chars` characters once squashed (a row of wikitext is a paragraph;
    headings are short)."""
    return sorted({p for p in map(squash, rows) if len(p) > min_chars})


def find_overlaps(paragraphs, corpus_texts):
    """[(paragraph, name of the corpus text it appears in)] for every paragraph found in one of the texts
    {name: text}."""
    squashed = {name: squash(text) for name, text in corpus_texts.items()}
    return [(p, name) for p in paragraphs for name, text in squashed.items() if p in text]


def chunk_count(n_tokens, n_ctx=N_CTX):
    """Chunks `llama-perplexity` cuts from n_tokens text tokens: a BOS token is added in front and the number of
    whole windows of n_ctx tokens is taken (perplexity.cpp: `tokens.size() / n_ctx`)."""
    return (n_tokens + 1) // n_ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="where wiki_valid.txt and eval_text.json go")
    ap.add_argument("--half-a", required=True, help="half_a.txt of the corpus")
    ap.add_argument("--half-b", required=True, help="half_b.txt of the corpus")
    ap.add_argument("--rows", help="validation rows (jsonl); default: <out-dir>/wikitext-2-raw-v1.validation.jsonl")
    ap.add_argument("--fetch", action="store_true", help="download the validation rows to --rows first")
    ap.add_argument("--control-rows", help="test rows (jsonl): the same check must find overlaps in them")
    ap.add_argument("--tokenizer", help="tokenizer.json of DeepSeek-V4.1 (token and chunk count)")
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--chunks", type=int, default=CHUNKS_WANTED)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = Path(args.rows) if args.rows else out / f"wikitext-2-raw-v1.{SPLIT}.jsonl"
    if args.fetch:
        build_corpus.fetch_wikitext(rows_path, split=SPLIT)
    rows = read_rows(rows_path)
    corpus = {"half_a.txt": Path(args.half_a).read_text(encoding="utf-8"),
              "half_b.txt": Path(args.half_b).read_text(encoding="utf-8")}

    paragraphs = long_paragraphs(rows)
    report = {"split": SPLIT, "rows": len(rows), "paragraphs_checked": len(paragraphs),
              "min_paragraph_chars": MIN_PARAGRAPH_CHARS, "rows_sha256": build_corpus.sha256_of(rows_path)}
    if args.control_rows:
        control = long_paragraphs(read_rows(args.control_rows))
        hits = find_overlaps(control, corpus)
        report["control"] = {"split": "test", "paragraphs_checked": len(control), "overlaps": len(hits)}
        if not hits:
            print("error: the control (wikitext-2 test, which the corpus holds) found no overlap: the check is "
                  "blind", file=sys.stderr)
            return 2
    overlaps = find_overlaps(paragraphs, corpus)
    report["overlaps"] = len(overlaps)
    if overlaps:
        report["overlap_examples"] = [{"in": name, "paragraph": p[:120]} for p, name in overlaps[:5]]
        (out / "eval_text.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"error: {len(overlaps)} validation paragraphs appear in the corpus; no text written",
              file=sys.stderr)
        return 2

    text = to_text(rows)
    (out / "wiki_valid.txt").write_text(text, encoding="utf-8", newline="")
    raw = text.encode("utf-8")
    report.update({"file": "wiki_valid.txt", "bytes": len(raw), "chars": len(text),
                   "sha256": hashlib.sha256(raw).hexdigest()})
    if args.tokenizer:
        n_tokens = build_corpus.Counter(args.tokenizer).count(text)
        chunks = chunk_count(n_tokens, args.n_ctx)
        report["tokens"] = {"count": n_tokens, "n_ctx": args.n_ctx, "chunks": chunks, "chunks_wanted": args.chunks,
                            "enough": chunks >= args.chunks}
    (out / "eval_text.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
