"""A multi-turn conversation against a running StrataGP server, timed turn by turn (issues #31-#34).

    python tools/bench_turns.py [--port 8080] [--turns 6] [--context-chars 200000] [--max-tokens 256]
                                [--think high] [--send-reasoning] [--json out.json]

The first message carries a long document (docs/AUDIT-PERF.md repeated up to --context-chars) and a question, the
next ones short follow-ups, greedy.  Per turn: the time to the first streamed text, the whole turn, and what the
server's /metrics says of it - prompt tokens, how many the engine reused, its prompt and decode times.  Past answers
go back without their reasoning, as most OpenAI clients send them (--send-reasoning: with it, as chat.py does), so
running it against a server started with and without --recall-reasoning shows what #34 changes; against one started
with STRATA_OLD_PROMPT_ENCODE=1 STRATA_OLD_DETOK=1 and without, what #31/#32 change.  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLLOW_UPS = ["Summarize the three most important points in five lines.", "Which of them is the cheapest to do?",
              "Write the first step as a checklist.", "What could go wrong with it?", "And how would you measure it?",
              "Give a one-sentence conclusion."]


def post_stream(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0, first, reasoning, content = time.time(), None, [], []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if "error" in d:
                raise RuntimeError(d["error"].get("message"))
            delta = d["choices"][0]["delta"]
            if (delta.get("reasoning_content") or delta.get("content")) and first is None:
                first = time.time() - t0
            reasoning.append(delta.get("reasoning_content") or "")
            content.append(delta.get("content") or "")
    return first, time.time() - t0, "".join(reasoning), "".join(content)


def metrics(base, headers):
    try:
        req = urllib.request.Request(base + "/metrics", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return (json.loads(r.read()).get("requests") or [{}])[0]
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--context-chars", type=int, default=200000, help="size of the first message's document")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", default="high", choices=["none", "low", "medium", "high"])
    ap.add_argument("--send-reasoning", action="store_true", help="send past answers back with their reasoning")
    ap.add_argument("--salt", default=None, help="a line put before the document (default: the time), so a second "
                                                 "run is a new conversation the engine's cache knows nothing of")
    ap.add_argument("--json", help="also write the rows to this file")
    a = ap.parse_args(argv)
    base = f"http://{a.host}:{a.port}"
    headers = {"Content-Type": "application/json"}
    if a.api_key:
        headers["Authorization"] = "Bearer " + a.api_key
    doc = (ROOT / "docs" / "AUDIT-PERF.md").read_text(encoding="utf-8")
    doc = (doc * (a.context_chars // max(1, len(doc)) + 1))[:a.context_chars]
    salt = a.salt if a.salt is not None else f"run {time.time():.3f}"
    question = "\n\nWhat does this document recommend first, and why?"
    messages = [{"role": "user", "content": f"[{salt}]\n" + doc + question}]
    rows = []
    print(f"{'turn':>4} {'first text s':>12} {'turn s':>8} {'prompt tok':>10} {'reused':>8} {'prompt ms':>10} "
          f"{'decode tok/s':>12}")
    for turn in range(a.turns):
        body = {"model": "strata", "messages": messages, "stream": True, "max_tokens": a.max_tokens,
                "temperature": 0, "reasoning_effort": a.think}
        first, total, reasoning, content = post_stream(base + "/v1/chat/completions", body, headers)
        m = metrics(base, headers)
        row = {"turn": turn + 1, "first_text_s": round(first, 3) if first is not None else None,
               "turn_s": round(total, 3), "prompt_tokens": m.get("prompt_tokens"), "reused": m.get("reused"),
               "prompt_ms": m.get("prompt_ms"), "decode_tok_s": m.get("decode_tok_s")}
        rows.append(row)
        print(f"{row['turn']:>4} {row['first_text_s'] if first is not None else '-':>12} {row['turn_s']:>8} "
              f"{str(row['prompt_tokens']):>10} {str(row['reused']):>8} {str(row['prompt_ms']):>10} "
              f"{str(row['decode_tok_s']):>12}", flush=True)
        answer = {"role": "assistant", "content": content}
        if a.send_reasoning and reasoning:
            answer["reasoning_content"] = reasoning
        messages += [answer, {"role": "user", "content": FOLLOW_UPS[turn % len(FOLLOW_UPS)]}]
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
