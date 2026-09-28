"""Tests for tools/make_profile.py (issue #49): routing traces change the profile, blended with the base ranking.

    python -m unittest tools.test_make_profile -v
"""
from __future__ import annotations

import contextlib
import io
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402

ALL = [(layer, e) for e in range(MP.N_EXPERT) for layer in range(MP.N_LAYER)]


def write_trace(path, records):
    """records: (layer, [expert ids]) - the ids, then as many weights, as --dump-routing writes them."""
    with open(path, "wb") as f:
        for layer, ids in records:
            f.write(struct.pack("<ii", layer, len(ids)))
            f.write(struct.pack("<%di" % len(ids), *ids))
            f.write(struct.pack("<%df" % len(ids), *([0.1] * len(ids))))


class MakeProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        rnd = random.Random(4)
        self.base = list(ALL)
        rnd.shuffle(self.base)                           # a base that ranks all 24,576 pairs, like the shipped one
        MP.write_profile(self.dir / "base.bin", self.base)
        # a trace that uses base rank 20000 most, then rank 15000, and rank 3 a little; the base's top pair never
        self.hot, self.warm, self.low = self.base[20000], self.base[15000], self.base[3]
        recs = [(self.hot[0], [self.hot[1]])] * 50 + [(self.warm[0], [self.warm[1]])] * 20 + \
               [(self.low[0], [self.low[1]])] * 2
        write_trace(self.dir / "t.bin", recs)

    def run_main(self, *args):
        out = self.dir / "out.bin"
        argv = ["make_profile.py", *[str(a) for a in args], "--base", str(self.dir / "base.bin"), "--out", str(out)]
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                MP.main()
        finally:
            sys.argv = old
        return MP.read_profile(out)

    def test_no_trace_keeps_the_base(self):
        self.assertEqual(self.run_main(), self.base)
        self.assertEqual((self.dir / "out.bin").read_bytes(), (self.dir / "base.bin").read_bytes())

    def test_weight_zero_is_the_old_order(self):
        self.assertEqual(self.run_main(self.dir / "t.bin", "--trace-weight", "0"), self.base)

    def test_traces_change_the_profile(self):
        got = self.run_main(self.dir / "t.bin")
        self.assertEqual(sorted(got), sorted(ALL))          # every pair once
        self.assertLess(got.index(self.hot), 5)             # a pair the trace uses a lot moves up
        self.assertLess(got.index(self.warm), 15000)
        self.assertLess(got.index(self.base[0]), 5)         # the base's best is not pushed out by a short trace
        self.assertNotEqual(got, self.base)

    def test_weight_one_is_traces_first(self):
        got = self.run_main(self.dir / "t.bin", "--trace-weight", "1")
        self.assertEqual(got[:3], [self.hot, self.warm, self.low])
        rest = [p for p in self.base if p not in (self.hot, self.warm, self.low)]
        self.assertEqual(got[3:], rest)                     # then the base rank

    def test_no_base(self):
        got = self.run_main(self.dir / "t.bin", "--no-base")
        self.assertEqual(got[:3], [self.hot, self.warm, self.low])
        self.assertEqual(got[3:], [p for p in ALL if p not in (self.hot, self.warm, self.low)])

    def test_partial_base(self):
        MP.write_profile(self.dir / "base.bin", self.base[:8000])    # an older profile that ranks 8,000 pairs
        got = self.run_main(self.dir / "t.bin")
        self.assertEqual(sorted(got), sorted(ALL))
        self.assertLess(got.index(self.hot), 5)                      # a pair the base did not rank at all
        self.assertEqual(got[-1], [p for p in ALL if p not in set(self.base[:8000]) | {self.hot, self.warm}][-1])

    def test_blend_scale(self):
        base = [(0, 0), (0, 1), (0, 2), (0, 3)]
        freq = {(0, 3): 10, (0, 2): 6, (1, 0): 8}
        # base ranks take the trace's sorted frequencies 10, 8, 6, 0: (0,0) 10, (0,1) 8, (0,2) 6, (0,3) 0
        self.assertEqual(MP.blend(base, freq, 0.5), [(0, 2), (0, 0), (0, 3), (0, 1), (1, 0)])
        self.assertEqual(MP.blend(base, freq, 1.0), [(0, 3), (1, 0), (0, 2), (0, 0), (0, 1)])
        self.assertEqual(MP.blend(base, freq, 0.0), [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)])


if __name__ == "__main__":
    unittest.main()
