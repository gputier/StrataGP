"""Tests for tools/calibrate.py and setup's use of it, without a GPU: a stand-in engine whose decode speed is a
function of the settings it runs with.

    python -m unittest tools.test_calibrate
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate as CAL  # noqa: E402


class FakeEngine:
    """Speed = f(pcie_frac, spec_min_p, workers): the GEN line's tune keys arrive as `strata_tune`."""

    def __init__(self, args, speed, info_workers=6, starts=None):
        self.args = list(args)
        w = CAL.arg_value(args, "--pool-workers")
        self.workers = int(w) if w else info_workers
        self.info = {"pool_workers": info_workers, "pcie_frac": 0.55, "spec_min_p": float(CAL.arg_value(args, "--spec-min-p") or 0)}
        self.speed = speed
        self.last = {}
        self.proc = None
        if starts is not None:
            starts.append(list(args))

    def generate(self, ids, max_new, sampling, cancel):
        tune = sampling.get("strata_tune") or {}
        rate = self.speed(tune.get("pcie_frac", 0.55), tune.get("spec_min_p", self.info["spec_min_p"]), self.workers)
        for _ in range(max_new):
            yield 1
        self.last = {"generated": max_new, "decode_ms": max_new / rate * 1000.0}


BASE = ["--pack", "p", "--spec", "4", "--spec-min-p", "0.5", "--max-context", "8192"]


class Calibrate(unittest.TestCase):
    def run_with(self, speed, workers=6, base=BASE):
        starts = []
        res = CAL.measure(base, [[1, 2, 3]] * 3, lambda a: FakeEngine(a, speed, workers, starts), say=lambda *_: None)
        return res, starts

    def test_defaults_kept_when_flat(self):
        res, _ = self.run_with(lambda f, p, w: 50.0)
        self.assertEqual(res["settings"], {})

    def test_small_gain_is_noise(self):
        # 2% better at pcie 0.35: below MIN_GAIN, so the default stays
        res, _ = self.run_with(lambda f, p, w: 51.0 if abs(f - 0.35) < 1e-6 else 50.0)
        self.assertEqual(res["settings"], {})

    def test_finds_pcie_and_min_p(self):
        def speed(f, p, w):
            return 50.0 + (10.0 if abs(f - 0.2) < 1e-6 else 0.0) + (5.0 if abs(p - 0.7) < 1e-6 else 0.0)
        res, _ = self.run_with(speed)
        self.assertEqual(res["settings"].get("--pcie-frac"), "0.20")
        self.assertEqual(res["settings"].get("--spec-min-p"), "0.70")
        self.assertNotIn("--pool-workers", res["settings"])

    def test_fewer_workers(self):
        # a hybrid CPU: half the workers is 20% faster
        res, starts = self.run_with(lambda f, p, w: 60.0 if w == 3 else 50.0, workers=6)
        self.assertEqual(res["settings"].get("--pool-workers"), "3")
        self.assertEqual(len(starts), 1 + len(CAL.worker_candidates(6)))   # one start per worker count, plus the sweep
        self.assertEqual(CAL.arg_value(starts[0], "--spec-min-p"), "0.5")   # measured against the product default

    def test_old_calibration_is_the_baseline_reset(self):
        # a config tuned earlier: the measurement starts from the product defaults, not from those values
        base = CAL.apply(BASE, {"--pcie-frac": "0.20", "--pool-workers": "3", "--spec-min-p": "0.70"})
        _, starts = self.run_with(lambda f, p, w: 50.0, base=base)
        self.assertIsNone(CAL.arg_value(starts[0], "--pcie-frac"))
        self.assertIsNone(CAL.arg_value(starts[0], "--pool-workers"))
        self.assertEqual(CAL.arg_value(starts[0], "--spec-min-p"), "0.5")

    def test_apply(self):
        a = CAL.apply(BASE, {"--pcie-frac": "0.35", "--pool-workers": "4"})
        self.assertEqual(CAL.arg_value(a, "--pcie-frac"), "0.35")
        self.assertEqual(CAL.arg_value(a, "--pool-workers"), "4")
        self.assertEqual(CAL.arg_value(a, "--spec-min-p"), "0.5")
        b = CAL.apply(a, {})                                # back to the defaults
        self.assertIsNone(CAL.arg_value(b, "--pcie-frac"))
        self.assertIsNone(CAL.arg_value(b, "--pool-workers"))
        self.assertEqual(b.count("--spec-min-p"), 1)

    def test_worker_candidates(self):
        self.assertEqual(CAL.worker_candidates(6), [6, 4, 3])
        self.assertEqual(CAL.worker_candidates(23), [23, 15, 12])
        self.assertEqual(CAL.worker_candidates(3), [3, 2])
        self.assertEqual(CAL.worker_candidates(1), [1])

    def test_pick(self):
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [53, 52, 60]}, "a"), "b")
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [51, 51.5, 51]}, "a"), "a")
        self.assertEqual(CAL.pick({}, "a"), "a")


class SpecEngine(FakeEngine):
    """Speed also depends on the verify window the engine was started with (--spec)."""

    def __init__(self, args, speed, info_workers=6, starts=None):
        self.spec = int(CAL.arg_value(args, "--spec") or 0)
        full = speed
        super().__init__(args, lambda f, p, w: full(f, p, w, self.spec), info_workers, starts)


class CalibrateSpec(unittest.TestCase):
    """Issue #50: the verify window, measured only on request, kept only when it beats --spec 4 by more than 3%."""

    def run_with(self, speed, specs=CAL.SPECS, workers=6, base=BASE):
        starts = []
        res = CAL.measure(base, [[1, 2, 3]] * 3, lambda a: SpecEngine(a, speed, workers, starts),
                          say=lambda *_: None, specs=specs)
        return res, starts

    def test_not_measured_by_default(self):
        res, starts = self.run_with(lambda f, p, w, s: 50.0 + 10 * s, specs=())
        self.assertNotIn("--spec", res["settings"])
        self.assertTrue(all(CAL.arg_value(a, "--spec") == "4" for a in starts))
        self.assertNotIn("spec", res["report"])

    def test_longer_window_kept(self):
        res, starts = self.run_with(lambda f, p, w, s: {4: 50.0, 5: 56.0, 6: 53.0}[s])
        self.assertEqual(res["settings"].get("--spec"), "5")
        self.assertEqual(sorted(res["report"]["spec"]), ["4", "5", "6"])
        self.assertEqual(len(res["report"]["spec"]["5"]), CAL.SPEC_RUNS)
        self.assertEqual([CAL.arg_value(a, "--spec") for a in starts[1:4]], ["4", "5", "6"])
        # the worker counts are then measured with the chosen window
        self.assertTrue(all(CAL.arg_value(a, "--spec") == "5" for a in starts[4:]))
        self.assertEqual(res["report"]["tok_s"], 56.0)

    def test_small_gain_is_noise(self):
        res, _ = self.run_with(lambda f, p, w, s: {4: 50.0, 5: 51.0, 6: 51.4}[s])
        self.assertNotIn("--spec", res["settings"])

    def test_hand_edited_window_kept(self):
        # a config at --spec 6 where 6 wins: apply() must keep 6, the window the worker counts were measured with
        base = CAL.with_arg(BASE, "--spec", "6")
        res, starts = self.run_with(lambda f, p, w, s: {4: 50.0, 5: 50.5, 6: 55.0}[s], base=base)
        self.assertEqual(res["settings"].get("--spec"), "6")
        self.assertTrue(all(CAL.arg_value(a, "--spec") == "6" for a in starts[4:]))
        self.assertEqual(CAL.arg_value(CAL.apply(base, res["settings"]), "--spec"), "6")

    def test_with_the_chosen_pcie_share(self):
        res, starts = self.run_with(lambda f, p, w, s: 50.0 + (10.0 if abs(f - 0.2) < 1e-6 else 0.0) + (s == 6) * 5)
        self.assertEqual(res["settings"].get("--pcie-frac"), "0.20")
        self.assertEqual(res["settings"].get("--spec"), "6")
        self.assertTrue(all(CAL.arg_value(a, "--pcie-frac") == "0.20" for a in starts[1:]))

    def test_apply(self):
        a = CAL.apply(BASE, {"--spec": "6"})
        self.assertEqual(CAL.arg_value(a, "--spec"), "6")
        self.assertEqual(CAL.arg_value(CAL.apply(a, {}), "--spec"), "4")      # an older calibration does not linger
        no_spec = [x for x in BASE if x not in ("--spec", "4")]
        self.assertIsNone(CAL.arg_value(CAL.apply(no_spec, {"--spec": "6"}), "--spec"))   # never adds speculation

    def test_parse(self):
        self.assertEqual(CAL.parse_specs("6,4, 5,5"), (4, 5, 6))
        self.assertEqual(CAL.parse_specs(""), ())
        self.assertEqual(CAL.parse_specs(None), ())
        for bad in ("4,x", "1", "9"):
            with self.assertRaises(ValueError):
                CAL.parse_specs(bad)

    def test_env_asks_run_for_it(self):
        seen = {}
        saved = CAL.measure
        CAL.measure = lambda args, ids, start, say, specs=(): seen.setdefault("specs", specs) or {}
        os.environ["STRATA_CALIBRATE_SPEC"] = "4,6"
        try:
            with tempfile.TemporaryDirectory() as d:
                import strata_tokenizer as ST                   # a byte-level vocabulary, no merges but one
                t = Path(d)
                toks = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<think>",
                                                                     "</think>"]
                (t / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(toks)}))
                (t / "merges.txt").write_text("a b", encoding="utf-8")
                (t / "token_type.json").write_text(json.dumps([1] * 257 + [3, 3, 4, 4]))
                CAL.run({"tokenizer": d, "args": BASE}, start_engine=lambda a: None)
        finally:
            CAL.measure = saved
            del os.environ["STRATA_CALIBRATE_SPEC"]
        self.assertEqual(seen["specs"], (4, 6))

    def base_args_of(self, cfg: dict) -> list[str]:
        """The engine arguments run() hands to measure() for this config."""
        seen = {}
        saved = CAL.measure
        CAL.measure = lambda args, ids, start, say, specs=(): seen.setdefault("args", list(args)) and {}
        try:
            with tempfile.TemporaryDirectory() as d:
                import strata_tokenizer as ST
                t = Path(d)
                toks = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<think>",
                                                                     "</think>"]
                (t / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(toks)}))
                (t / "merges.txt").write_text("a b", encoding="utf-8")
                (t / "token_type.json").write_text(json.dumps([1] * 257 + [3, 3, 4, 4]))
                CAL.run({"tokenizer": d, "args": BASE, **cfg}, start_engine=lambda a: None, specs=())
        finally:
            CAL.measure = saved
        return seen["args"]

    def test_layer_split_config_is_measured_split(self):
        # 0.1.21: a config with several GPUs runs the layer split, and the calibration measures that setup
        self.assertEqual(self.base_args_of({}), BASE)
        self.assertEqual(self.base_args_of({"gpu": 1}), BASE)
        self.assertEqual(self.base_args_of({"gpu": [0, 2]}), BASE + ["--layer-split", "auto"])
        self.assertEqual(self.base_args_of({"gpu": "0,2", "layer_split": "18"}), BASE + ["--layer-split", "18"])


class SetupIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = self.tmp.name
        os.environ["XDG_CONFIG_HOME"] = self.tmp.name
        import setup as S
        self.S = S
        self.saved = (S.gpu_info, S.cpu_info, S.ram_gb, CAL.run)
        S.gpu_info = lambda pick=None: {"name": "RTX Test", "vram_gb": 12.0, "arch": "120", "count": 1, "index": 0}
        S.cpu_info = lambda: ("Test CPU", True, True)
        S.ram_gb = lambda: 64.0

    def tearDown(self):
        self.S.gpu_info, self.S.cpu_info, self.S.ram_gb, CAL.run = self.saved
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_calibrate_config_writes_and_remembers(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg = {"exe": "x", "args": list(BASE), "model_name": "qwen3.8-flash-next-q2_0"}
        cfg_path.write_text(json.dumps(cfg))
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {"--pcie-frac": "0.20"}, "report": {"tok_s": 61.2}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        written = json.loads(cfg_path.read_text())
        self.assertEqual(CAL.arg_value(written["args"], "--pcie-frac"), "0.20")
        saved = self.S.saved_calibration(written)
        self.assertEqual(saved["settings"], {"--pcie-frac": "0.20"})
        # another model on the same PC has no calibration yet
        self.assertIsNone(self.S.saved_calibration({**written, "model_name": "swift-iq2_xs"}))
        # another context size is another key (its KV cache changes the expert cache)
        other = dict(written, args=CAL.with_arg(written["args"], "--max-context", "131072"))
        self.assertIsNone(self.S.saved_calibration(other))

    def test_failed_calibration_keeps_defaults(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg_path.write_text(json.dumps({"exe": "x", "args": list(BASE), "model_name": "m"}))

        def boom(*a, **k):
            raise RuntimeError("the engine exited before it was ready")
        CAL.run = boom
        self.assertFalse(self.S.calibrate_config(cfg_path))
        self.assertEqual(json.loads(cfg_path.read_text())["args"], BASE)
        self.assertIsNone(self.S.saved_calibration({"args": BASE, "model_name": "m"}))


if __name__ == "__main__":
    unittest.main()
