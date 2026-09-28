"""Tests for tools/routing_locality.py on synthetic traces whose locality is known by construction.

    python -m unittest tools.test_routing_locality
"""
from __future__ import annotations

import io
import json
import random
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import routing_locality as RL  # noqa: E402


def write_trace(path, positions, weights=None, extra=b""):
    """positions[p][layer] = the ids routed there (None: the layer is not traced); the --dump-routing format."""
    with open(path, "wb") as f:
        for p, layers in enumerate(positions):
            for layer, ids in enumerate(layers):
                if ids is None:
                    continue
                w = weights[p][layer] if weights else [1.0 / len(ids)] * len(ids)
                f.write(struct.pack("<ii", layer, len(ids)))
                f.write(struct.pack("<%di" % len(ids), *ids))
                f.write(struct.pack("<%df" % len(ids), *w))
        f.write(extra)


def measure_files(paths, n_layer, n_expert, **kw):
    traces = []
    for p in paths:
        I, W, _, _ = RL.read_trace(p, n_layer, n_expert)
        traces.append((I, W))
    return RL.measure(traces, n_layer, n_expert, **kw)


class RoutingLocality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def trace(self, name, positions, weights=None, extra=b""):
        p = self.dir / name
        write_trace(p, positions, weights, extra)
        return p

    def test_positions_are_runs_of_increasing_layers(self):
        pos = [[[1, 2], [3, 4], [5, 6]], [[7, 8], None, [9, 10]], [[11, 12], [13, 14], [15, 0]]]
        # a record out of range (layer 5 of 3) and a record cut in the middle
        bad = struct.pack("<ii", 5, 2) + struct.pack("<2i", 1, 2) + struct.pack("<2f", 0.5, 0.5)
        cut = struct.pack("<ii", 0, 2) + struct.pack("<i", 3)
        p = self.trace("t.bin", pos, extra=bad + cut)
        I, W, dropped, tail = RL.read_trace(p, 3, 16)
        self.assertEqual(I.shape, (3, 3, 2))
        self.assertEqual(I[1, 0].tolist(), [7, 8])
        self.assertEqual(I[1, 1].tolist(), [-1, -1])           # position 1 has no layer 1
        self.assertEqual(I[2, 2].tolist(), [15, 0])
        self.assertAlmostEqual(float(W[0, 0, 0]), 0.5)
        self.assertEqual(dropped, 1)
        self.assertEqual(tail, len(cut))

    def test_same_experts_every_token_is_full_reuse(self):
        rng = random.Random(1)
        fixed = [rng.sample(range(64), 4) for _ in range(3)]
        p = self.trace("same.bin", [fixed] * 40)
        r = measure_files([p], 3, 64, windows=(1, 4), budgets=(4,))
        self.assertAlmostEqual(r["reuse"]["1"]["reuse"], 1.0)
        self.assertAlmostEqual(r["reuse"]["1"]["weighted"], 1.0)
        self.assertAlmostEqual(r["reuse"]["4"]["window_experts"], 4.0)
        self.assertAlmostEqual(r["next_layer"]["4"]["prev"], 1.0)   # token t's experts at l+1 are token t+1's

    def test_sliding_run_reuses_all_but_one(self):
        k, n = 5, 64
        pos = [[[(t + j) % n for j in range(k)] for _ in range(2)] for t in range(200)]
        r = measure_files([self.trace("slide.bin", pos)], 2, n, windows=(1, 2), budgets=(k,))
        self.assertAlmostEqual(r["reuse"]["1"]["reuse"], (k - 1) / k)
        self.assertAlmostEqual(r["reuse"]["2"]["reuse"], (k - 1) / k)
        self.assertAlmostEqual(r["reuse"]["2"]["window_experts"], k + 1)

    def test_window_records_without_weights_count_alike(self):
        # a verify window writes weights 0: the weighted figures must then equal the plain ones
        k, n = 4, 32
        pos = [[[(t + j) % n for j in range(k)] for _ in range(2)] for t in range(60)]
        zeros = [[[0.0] * k for _ in range(2)] for _ in range(60)]
        r = measure_files([self.trace("w0.bin", pos, zeros)], 2, n, windows=(1,), budgets=(k,))
        self.assertAlmostEqual(r["reuse"]["1"]["weighted"], r["reuse"]["1"]["reuse"])
        x = r["next_layer"][str(k)]
        self.assertAlmostEqual(x["prev_weighted"], x["prev"])

    def test_a_layer_map_is_learned(self):
        # layer l+1 = a fixed permutation of layer l, layer 0 random: only the cross-layer table can know it
        rng = random.Random(2)
        n, k, L = 64, 4, 4
        pos = []
        for _ in range(600):
            layers = [rng.sample(range(n), k)]
            for _ in range(L - 1):
                layers.append([(7 * e + 3) % n for e in layers[-1]])
            pos.append(layers)
        r = measure_files([self.trace("map.bin", pos)], L, n, budgets=(k,), windows=(1,))
        x = r["next_layer"][str(k)]
        self.assertAlmostEqual(x["xlayer"], 1.0)
        self.assertGreater(x["both"], 0.95)
        self.assertLess(x["prev"], 0.2)                    # chance is k / n = 0.0625
        self.assertLess(r["reuse"]["1"]["reuse"], 0.2)

    def test_random_routing_is_chance(self):
        rng = random.Random(3)
        n, k, L = 64, 4, 3
        pos = [[rng.sample(range(n), k) for _ in range(L)] for _ in range(3000)]
        r = measure_files([self.trace("rand.bin", pos)], L, n, budgets=(8,), windows=(1,))
        self.assertAlmostEqual(r["reuse"]["1"]["reuse"], k / n, delta=0.01)
        x = r["next_layer"]["8"]
        self.assertAlmostEqual(x["chance"], 8 / n)
        for p in RL.PREDICTORS:
            self.assertAlmostEqual(x[p], 8 / n, delta=0.02, msg=p)

    def test_resident_experts_are_not_copies(self):
        # layer 0 = {0, a}, layer 1 = {1, a}: experts 0 and 1 are the static cache; the miss a at layer 1 is
        # layer 0's a of the SAME token, so the cross-layer table finds it and the previous token does not
        rng = random.Random(4)
        n = 16
        pos = []
        for _ in range(400):
            a = rng.randrange(2, n)
            pos.append([[0, a], [1, a]])
        r = measure_files([self.trace("res.bin", pos)], 2, n, budgets=(1, 2), windows=(1,), resident=2)
        self.assertEqual(r["resident_pairs"], 2)
        self.assertAlmostEqual(r["resident_hit_rate"], 0.5)
        self.assertAlmostEqual(r["misses_per_layer_token"], 1.0)
        x = r["next_layer"]["1"]
        self.assertAlmostEqual(x["xlayer"], 1.0)
        self.assertAlmostEqual(x["xlayer_precision"], 1.0)
        self.assertLess(x["prev"], 0.2)
        self.assertAlmostEqual(x["chance"], 1 / (n - 1))    # one of the 15 non-resident experts of layer 1

    def test_cli_writes_json(self):
        rng = random.Random(5)
        pos = [[rng.sample(range(32), 3) for _ in range(4)] for _ in range(50)]
        p = self.trace("cli.bin", pos)
        out = self.dir / "r.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = RL.main([str(p), "--n-layer", "4", "--n-expert", "32", "--budget", "3,6", "--per-layer",
                          "--json", str(out)])
        self.assertEqual(rc, 0)
        r = json.loads(out.read_text())
        self.assertEqual(r["positions"], 50)
        self.assertEqual(sorted(r["next_layer"]), ["3", "6"])
        self.assertEqual(len(r["per_layer"]), 4)
        self.assertIn("inter-token reuse", buf.getvalue())

    def test_cli_refuses_a_single_position(self):
        p = self.trace("one.bin", [[[1, 2], [3, 4]]])
        with redirect_stdout(io.StringIO()):
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(RL.main([str(p), "--n-layer", "2", "--n-expert", "8"]), 1)


if __name__ == "__main__":
    unittest.main()
