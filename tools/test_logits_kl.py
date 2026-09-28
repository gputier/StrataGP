"""Tests for tools/logits_kl.py on small synthetic --dump-logits files.

    python -m unittest tools.test_logits_kl
"""
from __future__ import annotations

import io
import math
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import logits_kl as LK  # noqa: E402


def write_dump(path, rows):
    rows = np.asarray(rows, np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", rows.shape[1], rows.shape[0]))
        f.write(rows.tobytes())


class LogitsKL(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, rows):
        p = self.dir / name
        write_dump(p, rows)
        return p

    def test_identical_dumps_are_zero_apart(self):
        rows = np.random.default_rng(1).normal(size=(9, 50))
        a, b = self.dump("a.bin", rows), self.dump("b.bin", rows)
        kl, top1 = LK.compare(LK.read_logits(a), LK.read_logits(b))
        self.assertEqual(float(kl.max()), 0.0)
        self.assertTrue(top1.all())

    def test_kl_value_and_argmax(self):
        # p = (1/2, 1/2), q = (3/4, 1/4): KL = 1/2 ln(2/3) + 1/2 ln 2; the argmax differs only in row 1
        a = self.dump("a.bin", [[0.0, 0.0], [1.0, 0.0]])
        b = self.dump("b.bin", [[math.log(3.0), 0.0], [0.0, 1.0]])
        kl, top1 = LK.compare(LK.read_logits(a), LK.read_logits(b))
        self.assertAlmostEqual(kl[0], 0.5 * math.log(2 / 3) + 0.5 * math.log(2), places=6)   # ln 3 is stored as f32
        self.assertEqual(top1.tolist(), [True, False])
        s = LK.summary(kl, top1)
        self.assertEqual(s["first_disagreement"], 1)

    def test_a_growing_drift_shows_by_quarter(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=(40, 30))
        drift = base + rng.normal(size=base.shape) * np.linspace(0.0, 1.0, 40)[:, None]
        kl, top1 = LK.compare(LK.read_logits(self.dump("a.bin", base)), LK.read_logits(self.dump("b.bin", drift)))
        q = LK.summary(kl, top1)["quarters"]
        self.assertEqual(len(q), 4)
        self.assertLess(q[0]["kl_mean"], q[3]["kl_mean"])

    def test_refuses_a_wrong_size_and_mismatched_shapes(self):
        p = self.dir / "bad.bin"
        p.write_bytes(struct.pack("<ii", 10, 3) + b"\0" * 8)
        with self.assertRaises(SystemExit):
            LK.read_logits(p)
        a, b = self.dump("a.bin", np.zeros((2, 5))), self.dump("b.bin", np.zeros((3, 5)))
        with self.assertRaises(SystemExit):
            LK.compare(LK.read_logits(a), LK.read_logits(b))

    def test_cli(self):
        rows = np.random.default_rng(3).normal(size=(8, 16))
        a, b = self.dump("a.bin", rows), self.dump("b.bin", rows + 0.01)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(LK.main([str(a), str(b), "--json", str(self.dir / "o.json")]), 0)
        self.assertIn("top-1 agreement", buf.getvalue())
        self.assertTrue((self.dir / "o.json").exists())


if __name__ == "__main__":
    unittest.main()
