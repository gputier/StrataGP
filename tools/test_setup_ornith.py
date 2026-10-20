"""Tests for setup.py's Ornith-1.5-122B install (the own-engine-path family OWN_PATH_FAMILIES["ornith"]): the entry and
its pinned source revision, a file of our own that is neither downloaded nor hashed, YaRN (factor 2) and the q4 KV cache
past the trained length, no images, and Qwen3.6's install unchanged.  Mocked engine and processes - nothing is built,
fetched or started.

    python -m unittest tools.test_setup_ornith
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_qwen36 import args, run  # noqa: E402

SOURCE = "ornith-ai/Ornith-1.5-122B"
FILE = "Ornith-1.5-122B-IQ3XXS-IQ4XS.gguf"


class OrnithEntry(unittest.TestCase):
    def test_entry(self):
        fam = setup.OWN_PATH_FAMILIES["ornith"]
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertEqual(setup.HF_REVISIONS[SOURCE], "dfa07bab90e90746ffbfc0b0cba3126eb9265de2")
        self.assertEqual(fam["source"], SOURCE)
        self.assertEqual(list(fam["models"]), ["IQ3XXS-IQ4XS"])
        self.assertEqual(fam["file"].format(q="IQ3XXS-IQ4XS"), FILE)
        self.assertFalse(fam["vision"])
        self.assertTrue(fam["name"].startswith("ornith"))

    def test_the_cache_is_counted_in_the_format_it_runs_in(self):
        fam = setup.OWN_PATH_FAMILIES["ornith"]
        self.assertEqual(setup.own_path_kv_bytes_per_token(fam), 13 * 2 * 2 * 144)       # q4: 144 bytes per head and cell
        self.assertEqual(setup.own_path_kv_bytes_per_token(setup.OWN_PATH_FAMILIES["qwen36"]), 11 * 2 * 256 * 2 * 4)

    def test_the_flash_next_cell_table_is_untouched(self):
        self.assertEqual(setup.KV_CELL_BYTES, {"q4_0": 576, "k8v4": 816})
        self.assertAlmostEqual(setup.kv_streaming_ram_gb(131072, "q4_0"), 131072 * 13 * 576 / 1e9)
        self.assertAlmostEqual(setup.kv_streaming_ram_gb(131072, "k8v4"), 131072 * 13 * 816 / 1e9)

    def test_the_tags_do_not_collide(self):
        tags = [d["tag"] for d in setup.OWN_PATH_FAMILIES.values()]
        self.assertEqual(len(set(tags)), len(tags))

    def test_estimated_size_and_recommendation(self):
        fam = setup.OWN_PATH_FAMILIES["ornith"]
        self.assertEqual(setup.own_path_file_bytes(fam, "IQ3XXS-IQ4XS"), 54e9)
        self.assertEqual(setup.own_path_size(fam, 128.0), "IQ3XXS-IQ4XS")
        self.assertEqual(setup.own_path_size(fam, 16.0), "IQ3XXS-IQ4XS")      # none fits: the only one
        q36 = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(setup.own_path_file_bytes(q36, "Q8_0"), 37801097504)  # a pinned file: its size

    def test_config_read_back(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "strata-ornith-iq3xxs-iq4xs.json"
            p.write_text('{"args": ["--native", "x.gguf", "--max-context", "409600"], "port": 8090}', encoding="utf-8")
            ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["model"], ch["context"]), ("ornith", "IQ3XXS-IQ4XS", 409600))


class InstallOrnith(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.cfg = {}
        eng = self.root / "engine"

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            return self.root / "llama", eng, {"version": "0.1.45"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.calls.append(("sha", s.name))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "engine_families", lambda e: {"qwen4exp", "qwen35moe"}),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda a: a),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.folder = self.root / "models" / "ornith-IQ3XXS-IQ4XS"

    def put_file(self):
        self.folder.mkdir(parents=True)
        (self.folder / FILE).write_bytes(b"x")

    def install(self, ram=128.0, vram=32.0, **kw):
        gpu = {"index": 0, "count": 1, "vram_gb": vram, "name": "Test GPU"}
        return run(setup.install_own_path, args(family="ornith", models_dir=str(self.root / "models"), **kw),
                   "ornith", gpu, False, [], 13, [], 8090, ram, None)

    def test_a_missing_file_is_asked_for_not_downloaded(self):
        rc, out = self.install(model="IQ3XXS-IQ4XS")
        self.assertEqual(rc, 1)
        self.assertIn(FILE, out)
        self.assertIn(f"{SOURCE} (revision dfa07bab90e9)", out)
        self.assertEqual([c for c in self.calls if c[0] in ("get", "sha")], [])

    def test_the_file_is_used_as_it_is_and_the_config_is_written(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=409600, kv="q4_0")
        self.assertEqual(rc, 0, out)
        self.assertEqual([c for c in self.calls if c[0] in ("get", "sha")], [])    # own file: no download, no hash
        self.assertIn(("engine", "none"), self.calls)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(self.folder / FILE))
        self.assertEqual(a[a.index("--max-context") + 1], "409600")
        self.assertEqual(a[a.index("--rope-scaling") + 1], "yarn")
        self.assertEqual(a[a.index("--rope-scale") + 1], "2")
        self.assertEqual(a[a.index("--rope-orig-ctx") + 1], "262144")
        self.assertEqual(a[a.index("--kv") + 1], "q4")
        self.assertNotIn("--vision", a)
        self.assertNotIn("vision", self.cfg)
        self.assertEqual(self.cfg["arch"], "qwen35moe")
        self.assertEqual(self.cfg["model_name"], "ornith-1.5-122b-iq3xxs-iq4xs")
        self.assertIn(("config", "strata-ornith-iq3xxs-iq4xs.json"), self.calls)
        self.assertNotIn("no rope scaling", out)

    def test_inside_the_trained_range_there_is_no_yarn(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=131072)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertNotIn("--rope-scaling", a)
        self.assertEqual(a[a.index("--kv") + 1], "q4")               # the family's own default

    def test_a_refused_rope_setting_stops_with_its_message(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=524288, rope_scaling="none")
        self.assertEqual(rc, 1, out)
        self.assertIn("--rope-scaling none", out)
        self.assertNotIn("Traceback", out)

    def test_explicit_yarn_at_the_trained_range_writes_factor_1(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=131072, rope_scaling="yarn")
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--rope-scaling") + 1], "yarn")
        self.assertEqual(a[a.index("--rope-scale") + 1], "1")
        self.assertEqual(a[a.index("--rope-orig-ctx") + 1], "262144")

    def test_a_factor_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            setup.resolve_rope(524288, "yarn", 0.5)
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=524288, rope_scale=0.5)
        self.assertEqual(rc, 1, out)
        self.assertIn("below 1", out)
        self.assertNotIn("Traceback", out)

    def test_a_context_past_the_family_limit_is_refused(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=1048576)
        self.assertEqual(rc, 1, out)
        self.assertIn("past Ornith-1.5-122B's limit of 524288", out)

    def test_an_asked_factor_wins(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", context=524288, rope_scale=4.0)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--rope-scale") + 1], "4")

    def test_flash_next_kv_modes_fall_back_with_a_note(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", kv="int8")
        self.assertEqual(rc, 0, out)
        self.assertIn("--kv int8 is a Flash-Next setting", out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--kv") + 1], "q4")

    def test_a_kv_mode_on_ornith_gets_one_warning(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", kv="q4_0")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count("always uses its q4 KV cache"), 1)
        self.assertNotIn("Flash-Next only", out)

    def test_images_are_off_even_when_asked(self):
        self.put_file()
        rc, out = self.install(model="IQ3XXS-IQ4XS", vision="gpu")
        self.assertEqual(rc, 0, out)
        self.assertIn("no image encoder on this engine path", out)
        self.assertIn(("engine", "none"), self.calls)
        self.assertNotIn("--vision", self.cfg["args"])

    def test_a_gguf_dir_holds_the_file(self):
        other = self.root / "mine"
        other.mkdir()
        (other / FILE).write_bytes(b"x")
        rc, out = self.install(model="IQ3XXS-IQ4XS", gguf_dir=str(other))
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(other / FILE))

    def test_qwen36_writes_no_kv_whatever_is_asked(self):
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        rc, out = run(setup.install_own_path, args(models_dir=str(self.root / "models"), model="UD-Q4_K_XL",
                                                    vision="no", context=262144, kv="q4_0"), "qwen36", gpu, False, [],
                      13, [], 8080, 64.0, None)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("--kv", self.cfg["args"])
        self.assertIn("--kv: Flash-Next only", out)

    def test_qwen36_keeps_its_install(self):
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        rc, out = run(setup.install_own_path, args(models_dir=str(self.root / "models"), model="UD-Q4_K_XL",
                                                    vision="no", context=262144), "qwen36", gpu, False, [], 13, [],
                      8080, 64.0, None)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertNotIn("--rope-scaling", a)
        self.assertNotIn("--kv", a)
        self.assertEqual(len([c for c in self.calls if c[0] == "get"]), 1)    # its GGUF is still downloaded
        self.assertIn(("sha", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), self.calls)


if __name__ == "__main__":
    unittest.main()
