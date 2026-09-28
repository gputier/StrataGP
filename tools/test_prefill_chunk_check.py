"""Tests for tools/prefill_chunk_check.py without a GPU: a stand-in engine that writes a logits dump, a GDN hash and
the generated tokens the way `strata generate` does, with a chunk-size effect it controls.

    python -m unittest tools.test_prefill_chunk_check
"""
from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prefill_chunk_check as PCC  # noqa: E402

# The stand-in: logits row of position p = a function of p (and of the chunk when the prompt is longer than it, the
# boundary effect under test); the token path and every chunk that holds the whole prompt give the same rows.
FAKE = r'''#!/usr/bin/env python3
import sys, struct, re
a = sys.argv[1:]
def val(f, d=None):
    return a[a.index(f) + 1] if f in a else d
ids = [int(t) for t in re.split(r"[\s,]+", open(val("--tokens-file")).read().strip()) if t]
n, max_new = len(ids), int(val("--max-new", "1"))
chunk = int(val("--prefill", "0"))
until = int(val("--prefill-until", "0"))
nv = 64
first = 0 if chunk == 0 else (until if 0 < until < n - 1 else n - 1)
split = chunk > 0 and chunk < n - 1
rows = []
for p in range(first, n - 1 + max_new):
    rows.append([((p * 7 + j * 3) % 11) * 0.5 + (0.01 * j if split and p < n else 0.0) for j in range(nv)])
with open(val("--dump-logits"), "wb") as f:
    f.write(struct.pack("<ii", nv, n - 1 + max_new))
    for r in rows:
        f.write(struct.pack("<%df" % nv, *r))
if chunk > 0:
    k = max(1, -(-(n - 1) // chunk))
    sys.stderr.write("strata generate: native pack: fake experts\n" if "--native-pack" in a else "")
    sys.stderr.write("strata generate: prefill %d tokens in %d chunks, 1.0 ms\n" % (n - 1, k))
    sys.stderr.write("strata prefill: GDN_HASH " + " ".join("%04x" % ((i * 31 + (k if split else 1)) & 0xffff) for i in range(36)) + "\n")
print("prompt  : " + " ".join(str(t) for t in ids))
print("output  : " + " ".join(str(100 + i + (1 if split and i > 3 else 0)) for i in range(max_new)))
'''


class PrefillChunkCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.exe = Path(self.dir.name) / "fake_strata"
        self.exe.write_text(FAKE)
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IEXEC)

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self, *args) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = PCC.main(["--exe", sys.executable, "--keep", str(Path(self.dir.name) / "work"), *args, "--",
                           str(self.exe), "--pack", "x"])
        return rc, out.getvalue()

    def test_strip_flags(self):
        args = ["--pack", "p", "--prefill", "auto", "--serve", "--max-new", "9", "--spec", "4", "--greedy"]
        self.assertEqual(PCC.strip_flags(args), ["--pack", "p", "--spec", "4"])

    def test_synthetic_prompt_is_deterministic(self):
        a, b = PCC.synthetic_prompt(9000, 1), PCC.synthetic_prompt(9000, 1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 9000)
        self.assertNotEqual(a, PCC.synthetic_prompt(9000, 2))

    def test_prompt_rows_alignment(self):
        logits = np.arange(10, dtype=np.float32).reshape(10, 1)       # row r = position first_pos + r
        np.testing.assert_array_equal(PCC.prompt_rows(logits, 0, 8, 3)[:, 0], [5, 6, 7])
        np.testing.assert_array_equal(PCC.prompt_rows(logits, 5, 8, 3)[:, 0], [0, 1, 2])
        self.assertIsNone(PCC.prompt_rows(logits, 6, 8, 3))

    def test_compare_logits(self):
        a = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        r = PCC.compare_logits(a, a.copy())
        self.assertTrue(r["identical"])
        self.assertEqual(r["top1"], 1.0)
        self.assertAlmostEqual(r["kl_mean"], 0.0, places=12)
        b = a.copy()
        b[1] = [0.0, 3.0, 0.0]
        r = PCC.compare_logits(a, b)
        self.assertEqual(r["top1"], 0.5)
        self.assertGreater(r["kl_last"], 0.1)

    def test_end_to_end_with_the_token_path(self):
        # the prompt (40 tokens) fits in chunk 64 but is split by 16: 64 matches the token path, 16 does not
        rc, out = self.run_tool("--ids", self.ids(40), "--chunks", "64,16", "--tail", "8", "--max-new", "6",
                                "--token-path")
        self.assertEqual(rc, 0, out)
        lines = {ln.split("  ")[0].strip(): ln for ln in out.splitlines() if ln.startswith(("chunk", "token"))}
        self.assertIn("identical (36 layers)", lines["chunk 64 (rerun)"])
        self.assertIn("100.0%", lines["chunk 64"])
        self.assertIn("0.000e+00", lines["chunk 64"])                  # the same rows as the token path
        self.assertIn("layers differ", lines["chunk 16"])
        self.assertIn("differ from token 4", lines["chunk 16"])
        self.assertNotIn("NOT bit-identical", out)

    def test_tail_one_compares_the_last_position(self):
        rc, out = self.run_tool("--ids", self.ids(40), "--chunks", "64,16")
        self.assertEqual(rc, 0, out)
        self.assertIn("last 1 position(s)", out)

    def ids(self, n: int) -> str:
        p = Path(self.dir.name) / f"ids{n}.txt"
        p.write_text(",".join(str(1000 + i) for i in range(n)))
        return str(p)


if __name__ == "__main__":
    unittest.main()
