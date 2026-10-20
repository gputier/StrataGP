"""serve/test_q36_config.py - what the server hands the Qwen3.6-35B-A3B engine (arch qwen35moe): the config's pictures
("vision" section -> --vision) and parallel requests ("parallel": N -> --batch N), as for Flash-Next's engine; and what
tells Ornith-1.5-122B (the same arch) from Qwen3.6: its API name and the model card's sampling.

    python -m unittest serve.test_q36_config
"""
from __future__ import annotations

import unittest
from unittest import mock

from serve.server import default_model_name, engine_args, model_card_sampling, model_family, parallel_args


def cfg(**kw):
    base = {"arch": "qwen35moe", "exe": "strata", "args": ["--native", "/m/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                                                           "--max-context", "32768", "--prefill", "auto", "--mtp", "2"]}
    base.update(kw)
    return base


class Q36Config(unittest.TestCase):
    def test_parallel_becomes_batch(self):
        a = engine_args(cfg(parallel=4))
        self.assertEqual(a[a.index("--batch") + 1], "4")
        self.assertNotIn("--batch", engine_args(cfg()))
        self.assertNotIn("--batch", engine_args(cfg(parallel=1)))

    def test_more_than_the_window_is_said(self):
        with mock.patch("builtins.print") as pr:
            a = engine_args(cfg(parallel=12))
        self.assertEqual(a[a.index("--batch") + 1], "12")          # the engine caps it at 8 and says so
        self.assertTrue(any("at most 8" in str(c) for c in pr.call_args_list))

    def test_a_vision_section_starts_the_engine_with_pictures(self):
        v = {"exe": "strata-vision", "mmproj": "/m/mmproj-F16.gguf", "model": "/m/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
             "gpu": False, "max_tokens": 300}
        self.assertIn("--vision", engine_args(cfg(vision=v)))
        self.assertIn("--vision", engine_args(cfg(vision=v, args=cfg()["args"] + ["--vision"])))
        self.assertEqual(engine_args(cfg(vision=v, args=cfg()["args"] + ["--vision"])).count("--vision"), 1)
        self.assertNotIn("--vision", engine_args(cfg()))

    def test_an_explicit_batch_in_the_args_wins(self):
        c = cfg(parallel=4, args=cfg()["args"] + ["--batch", "2"])
        self.assertEqual(parallel_args(c, c["args"]), [])

    def test_ornith_is_told_from_qwen36_by_its_model_name(self):
        ornith = cfg(model_name="ornith-1.5-122b-iq3xxs-iq4xs")
        self.assertEqual(model_family(ornith), "ornith")
        self.assertEqual(model_family(cfg(model_name="Ornith-1.5-122B")), "ornith")
        self.assertEqual(model_family(cfg(model_name="qwen3.6-35b-a3b-ud-q4_k_xl")), "qwen35moe")
        self.assertEqual(model_family(cfg()), "qwen35moe")
        self.assertEqual(model_family({"arch": "qwen4exp", "model_name": "ornith-x"}), "qwen4exp")
        self.assertEqual(model_family({}), "qwen4exp")

    def test_ornith_name_and_qwen36_name(self):
        self.assertEqual(default_model_name(cfg(model_name="ornith-1.5-122b-iq3xxs-iq4xs")),
                         "ornith-1.5-122b-iq3xxs-iq4xs")                  # the config's name wins
        self.assertEqual(default_model_name(cfg()), "qwen3.6-35b-a3b")
        self.assertEqual(default_model_name({"arch": "qwen35moe", "model_name": ""}), "qwen3.6-35b-a3b")

    def test_ornith_sampling_is_the_model_cards(self):
        card = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
        self.assertEqual(model_card_sampling("ornith", True), card)
        self.assertEqual(model_card_sampling("ornith", False), card)

    def test_qwen36_sampling_is_unchanged(self):
        self.assertEqual(model_card_sampling("qwen35moe", True),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5})
        self.assertEqual(model_card_sampling("qwen35moe", False),
                         {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})


if __name__ == "__main__":
    unittest.main()
