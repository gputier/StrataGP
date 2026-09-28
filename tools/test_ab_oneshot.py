"""Tests for tools/ab_oneshot.py against a stand-in engine (a Python script that prints the engine's summary lines).

    python -m unittest tools.test_ab_oneshot
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_oneshot as AB  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

FAKE = r'''
import sys
a = sys.argv[1:]
spec = int(a[a.index("--spec") + 1]) if "--spec" in a else 0
assert "--greedy" in a and a[a.index("--max-new") + 1] == "256" and a[a.index("--adapt-every") + 1] == "100000"
ids = open(a[a.index("--tokens-file") + 1]).read()
out = "7 8 9" if "--broken" not in a else "7 8 10"
print("prompt  : " + ids.replace(",", " "))
print("output  : " + out)
print("speculation              40 rounds of %d, drafts accepted %d of 120 (%.3f), %.2f tokens per round"
      % (spec, 20 * spec, 20 * spec / 120, 1 + 20 * spec / 40))
print("decode                   256 tokens in 1000.0 ms  ->  %.2f tok/s" % (40.0 + spec))
'''


class AbOneshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        (d / "engine.py").write_text(FAKE)
        toks = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<think>",
                                                             "</think>"]
        (d / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(toks)}))
        (d / "merges.txt").write_text("a b", encoding="utf-8")
        (d / "token_type.json").write_text(json.dumps([1] * 257 + [3, 3, 4, 4]))
        self.cfg = d / "strata-x.json"
        self.cfg.write_text(json.dumps({"exe": sys.executable, "args": [str(d / "engine.py"), "--spec", "4",
                                                                        "--max-context", "8192"],
                                        "tokenizer": str(d), "cwd": str(d)}))

    def run_ab(self, *variants):
        out = io.StringIO()
        rows = Path(self.tmp.name) / "rows.json"
        argv = [str(self.cfg), "--runs", "2", "--json", str(rows)]
        for v in variants:
            argv += ["--variant", v]
        with contextlib.redirect_stdout(out):
            self.assertEqual(AB.main(argv), 0)
        return out.getvalue(), json.loads(rows.read_text())

    def test_variants(self):
        text, data = self.run_ab("base=", "spec6=--spec 6", "odd=--spec 6 --broken")
        self.assertEqual(len(data["runs"]["base"]), 2 * len(AB.CAL.PROMPTS))
        self.assertEqual(AB.CAL.arg_value(data["variants"]["spec6"], "--spec"), "6")
        self.assertEqual(AB.CAL.arg_value(data["variants"]["base"], "--spec"), "4")
        lines = {ln.split()[0]: ln for ln in text.splitlines()
                 if ln.strip() and ln.split()[0] in ("base", "spec6", "odd")}
        self.assertIn("44.00", lines["base"])
        self.assertIn("46.00", lines["spec6"])
        self.assertTrue(lines["spec6"].rstrip().endswith("yes"))
        self.assertTrue(lines["odd"].rstrip().endswith("NO"))                 # a different greedy output is flagged

    def test_set_flags(self):
        self.assertEqual(AB.set_flags(["--spec", "4", "--x"], ["--spec", "5", "--y", "--z", "1"]),
                         ["--x", "--spec", "5", "--y", "--z", "1"])


if __name__ == "__main__":
    unittest.main()
