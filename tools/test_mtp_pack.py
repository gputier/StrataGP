"""Tests for tools/mtp_pack.py's Q2_0 scale searches (issue #52).

    STRATA_GGUF_PY=<llama.cpp>/gguf-py python -m unittest tools.test_mtp_pack -v

`grid` (the default) must write exactly the bytes it always wrote; `wide` and `exact` may store negative scales and
must reconstruct better.  The relative RMS errors are printed for the three weight laws of the issue.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import mtp_pack as MP
except SystemExit:                                       # no gguf-py: _paths exits
    MP = None


def old_q2_0(w: np.ndarray) -> np.ndarray:
    """mtp_pack.q2_0 before issue #52, verbatim: the reference for the default search."""
    x = w.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def blocks(law: str, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if law == "normal":
        x = rng.standard_normal((n, 64))
    elif law == "laplace":
        x = rng.laplace(size=(n, 64))
    else:
        x = rng.standard_t(3, size=(n, 64))
    return (x * 0.02).astype(np.float32)


def rel_rms(x: np.ndarray, search: str) -> float:
    got = MP.dequant("q2_0", MP.q2_0(x, search), x.size).reshape(x.shape)
    return float(np.sqrt(((got - x) ** 2).mean()) / np.sqrt((x ** 2).mean()))


def block_err(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.clip(np.rint(x * inv), -1, 2)
    return ((q * d - x) ** 2).sum(axis=1)


@unittest.skipIf(MP is None, "llama.cpp's gguf-py is not here (set STRATA_GGUF_PY)")
class Q2Search(unittest.TestCase):
    def test_default_is_unchanged(self):
        for law in ("normal", "laplace", "t3"):
            x = blocks(law, 4000, seed=1)
            x[::7] = 0.0                                   # all-zero blocks too
            x[3::11, 5] *= 40                              # outliers
            with self.subTest(law=law):
                np.testing.assert_array_equal(MP.q2_0(x), old_q2_0(x))
                np.testing.assert_array_equal(MP.q2_0(x, "grid"), old_q2_0(x))

    def test_better_and_signed(self):
        table = {}
        for law in ("normal", "laplace", "t3"):
            x = blocks(law, 20000, seed=2)
            table[law] = [rel_rms(x, s) for s in MP.Q2_SEARCHES]
            grid, wide, exact = table[law]
            with self.subTest(law=law):
                self.assertLess(wide, grid * 0.9)
                self.assertLessEqual(exact, wide * 1.001)
            blob = MP.q2_0(x, "exact").reshape(-1, 18)
            d = blob[:, :2].copy().view(np.float16).astype(np.float32)[:, 0]
            self.assertGreater((d < 0).mean(), 0.2)            # the mirrored grid is used
        print("\n  relative RMS error, 64-weight blocks (grid / wide / exact):")
        for law, v in table.items():
            print(f"    {law:8s} " + " / ".join(f"{e:.3f}" for e in v))

    def test_exact_is_optimal(self):
        """Against a dense scan of d over both signs, before the fp16 rounding of the stored scale."""
        x = blocks("t3", 300, seed=3)
        d = MP.q2_0_exact_d(x)
        e_exact = block_err(x.astype(np.float64), d.astype(np.float64))
        amax = np.abs(x).max(axis=1, keepdims=True).astype(np.float64)
        best = np.full(x.shape[0], np.inf)
        for f in np.linspace(-1.5, 1.5, 6001):
            if f != 0:
                best = np.minimum(best, block_err(x.astype(np.float64), amax * f))
        self.assertTrue(np.all(e_exact <= best * (1 + 1e-9) + 1e-18), float((e_exact - best).max()))

    def test_negative_outlier_uses_a_negative_scale(self):
        # -0.2, -0.1, 0 and 0.1: exactly the mirrored grid {-2, -1, 0, 1} x 0.1, which no d > 0 can reach
        x = np.zeros((1, 64), dtype=np.float32)
        x[0, :8], x[0, 8:16], x[0, 16:24] = -0.2, -0.1, 0.1
        for search in ("wide", "exact"):
            blob = MP.q2_0(x, search)
            d = blob[:2].copy().view(np.float16).astype(np.float32)[0]
            with self.subTest(search=search):
                self.assertLess(d, 0)
                got = MP.dequant("q2_0", blob, 64)
                self.assertLess(np.abs(got - x[0]).max(), 1e-3)                     # fp16 rounding of d only
                self.assertGreater(np.abs(MP.dequant("q2_0", MP.q2_0(x), 64) - x[0]).max(), 0.04)
        # the byte layout is the format's: value = (code - 1) * d, whatever the sign of d
        blob = MP.q2_0(x, "exact").reshape(18)
        d = blob[:2].copy().view(np.float16).astype(np.float32)[0]
        codes = np.stack([(blob[2:] >> s) & 3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)
        np.testing.assert_array_equal(MP.dequant("q2_0", blob, 64), (codes.astype(np.float32) - 1) * d)

    def test_unknown_search(self):
        with self.assertRaises(ValueError):
            MP.q2_0(blocks("normal", 2), "fast")

    def test_cli(self):
        """mtp_pack.py --q2-search exact on a tiny fake checkpoint: the experts are packed with that search."""
        import contextlib
        import io
        import json
        import tempfile
        import gguf
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            vals = (rng.standard_t(3, size=(2, 4, 128)) * 0.02).astype(np.float32)
            bf = (vals.view(np.uint32) >> 16).astype("<u2")          # exact BF16: the low bits are zero
            (src / "gu.bin").write_bytes(bf.tobytes())
            (src / "norm.bin").write_bytes(np.ones(8, dtype="<u2").tobytes())
            manifest = [{"name": "mtp.layers.0.mlp.experts.gate_up_proj", "shape": [2, 4, 128], "file": "gu.bin",
                         "sha256": "x"},
                        {"name": "mtp.norm.weight", "shape": [8], "file": "norm.bin", "sha256": "y"}]
            (src / "mtp-manifest.json").write_text(json.dumps(manifest))
            for search in ("grid", "exact"):
                out = str(src / f"{search}.gguf")
                argv, sys.argv = sys.argv, ["mtp_pack.py", "--src", d, "--out", out, "--q2-search", search]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(MP.main(), 0)
                finally:
                    sys.argv = argv
                r = gguf.GGUFReader(out)
                t = next(t for t in r.tensors if t.name.endswith("gate_up_proj"))
                want = np.concatenate([MP.q2_0((bf[e].astype(np.uint32) << 16).view(np.float32), search)
                                       for e in range(2)])
                with self.subTest(search=search):
                    np.testing.assert_array_equal(np.asarray(t.data).reshape(-1), want)
                    q = r.fields["strata.mtp.expert_quantizer"]
                    text = bytes(q.parts[q.data[0]]).decode()
                    self.assertEqual("signed" in text, search != "grid")


if __name__ == "__main__":
    unittest.main()
