"""Tests for tools/strata_tokenizer.py's caches (issue #32): the piece cache and the incremental prompt encoder must
give exactly the ids of a plain encode.

    python -m unittest tools.test_strata_tokenizer -v

The real vocabulary comes from $STRATA_TOKENIZER (a pack's tokenizer/ directory) when set, else from llama.cpp's
`models/ggml-vocab-qwen35.gguf` next to the gguf-py the tools use ($STRATA_GGUF_PY, third_party/llama.cpp); the
tests that need it are skipped without one.  The chat template's USER_DEFINED tokens (`<think>`, `<tool_call>`,
...) are added to the llama.cpp vocabulary, which does not have them, as the pack's has.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402

ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
         "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]


def vocab_gguf() -> pathlib.Path | None:
    cands = [os.environ.get("STRATA_GGUF_PY"), ROOT / "third_party" / "llama.cpp" / "gguf-py", "/opt/llama.cpp/gguf-py"]
    for c in cands:
        if c and (pathlib.Path(c).parent / "models" / "ggml-vocab-qwen35.gguf").is_file():
            return pathlib.Path(c).parent / "models" / "ggml-vocab-qwen35.gguf"
    return None


def load_tokenizer():
    pack = os.environ.get("STRATA_TOKENIZER")
    if pack and (pathlib.Path(pack) / "vocab.json").is_file():
        t = pathlib.Path(pack)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                            json.loads((t / "token_type.json").read_text()))
    path = vocab_gguf()
    if path is None:
        return None
    from gguf_reader import GGUFFile
    md = GGUFFile(path).metadata
    tokens, types = list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.token_type"])
    for s in ADDED:
        if s not in tokens:
            tokens.append(s)
            types.append(4)
    return ST.Tokenizer(tokens, list(md["tokenizer.ggml.merges"]), types)


def plain_encode(tk, text: str, parse_special: bool) -> list[int]:
    """The encoder as it was before the caches: every piece through _bpe, every time."""
    pat = tk._special_re if parse_special else tk._always_re

    def plain(s):
        out = []
        for piece in tk._re.findall(s):
            mapped = "".join(ST.BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            out += [tk.ids[t] for t in tk._bpe(mapped)]
        return out
    if pat is None:
        return plain(text)
    out, pos = [], 0
    for m in pat.finditer(text):
        out += plain(text[pos:m.start()]) + [tk.special_tokens[m.group(0)]]
        pos = m.end()
    return out + plain(text[pos:])


def byte_tokenizer(specials: list[str]) -> ST.Tokenizer:
    """A vocabulary of the 256 byte tokens and `specials` (CONTROL), no merges: small enough to build per test."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + specials
    return ST.Tokenizer(tokens, [], [1] * 256 + [3] * len(specials))


WORDS = ["the", "tokenizer", " ", "  ", "\n", "\n\n", "\t", "def", "f(x):", "return", "x", "==", "->", "café",
         "你好", "\U0001f600", "é", "1234", "don't", "I'm", "<|im_", "<think", "</", "tool", "_call>",
         " ", "\r\n", "Hello,", "world!", "<", ">", "|", "      ", "x" * 70]


def random_text(rnd: random.Random, n: int, specials=()) -> str:
    parts = []
    for _ in range(n):
        parts.append(rnd.choice(WORDS) if not specials or rnd.random() > 0.05 else rnd.choice(specials))
        if rnd.random() < 0.5:
            parts.append(" ")
    return "".join(parts)


class Oracle(unittest.TestCase):
    """llama.cpp's own tokenizer test vectors for qwen35 (the piece cache is on)."""

    def test_llama_cpp_vectors(self):
        path = vocab_gguf()
        if path is None or not path.with_suffix(".gguf.inp").is_file():
            self.skipTest("llama.cpp's ggml-vocab-qwen35.gguf test vectors are not here")
        tk = load_tokenizer()
        inp = path.with_suffix(".gguf.inp").read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
        out = path.with_suffix(".gguf.out").read_text(encoding="utf-8").split("\n")
        n = 0
        for text, want in zip(inp, out):
            if text.endswith("\n__ggml_vocab_test__"):
                text = text[:-len("\n__ggml_vocab_test__")]
            want_ids = [int(x) for x in want.split()]
            for _ in range(2):                          # the second pass comes from the cache
                self.assertEqual(tk.encode(text), want_ids, repr(text[:40]))
            n += 1
        self.assertGreater(n, 40)


class PieceCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = load_tokenizer()
        if cls.tk is None:
            raise unittest.SkipTest("no tokenizer (set STRATA_TOKENIZER or STRATA_GGUF_PY)")

    def test_same_ids_as_the_plain_encoder(self):
        rnd = random.Random(7)
        specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>"]
        for i in range(60):
            text = random_text(rnd, rnd.randint(0, 300), specials)
            for ps in (False, True):
                with self.subTest(i=i, parse_special=ps):
                    self.assertEqual(self.tk.encode(text, parse_special=ps), plain_encode(self.tk, text, ps))

    def test_cached_value_is_not_shared_mutably(self):
        a = self.tk.encode("hello hello hello")
        a.append(-1)
        self.assertEqual(self.tk.encode("hello hello hello"), plain_encode(self.tk, "hello hello hello", False))

    def test_token_bytes_match_decode(self):
        for i in list(range(0, len(self.tk.tokens), 997)) + [len(self.tk.tokens) - 1]:
            self.assertEqual(self.tk.token_bytes(i).decode("utf-8", "replace"), self.tk.decode([i]))
        with self.assertRaises(IndexError):
            self.tk.token_bytes(len(self.tk.tokens))

    def test_marks(self):
        text = "a<|im_start|>user\nhi<|im_end|>\n<think>\n"
        ids, marks = self.tk.encode_marked(text, parse_special=True)
        self.assertEqual(ids, self.tk.encode(text, parse_special=True))
        self.assertEqual([m[0] for m in marks], [text.index("user"), text.index("<|im_end|>") + len("<|im_end|>"),
                                                  len(text) - 1])
        for end, n in marks:                            # each mark: the ids of the text up to it
            self.assertEqual(ids[:n], self.tk.encode(text[:end], parse_special=True))


class Prompts(unittest.TestCase):
    """PromptEncoder: the ids of a plain encode, whatever the earlier prompts were."""

    @classmethod
    def setUpClass(cls):
        cls.tk = load_tokenizer()
        if cls.tk is None:
            raise unittest.SkipTest("no tokenizer (set STRATA_TOKENIZER or STRATA_GGUF_PY)")
        from serve.frontend import ChatTemplate
        cls.template = ChatTemplate(ROOT / "serve" / "chat_template.jinja")

    def check(self, enc, text):
        got = enc.encode(text)
        self.assertEqual(got, self.tk.encode(text, parse_special=True))
        return got

    def test_chat_golden(self):
        cases = json.loads((ROOT / "serve" / "chat_golden.json").read_text(encoding="utf-8"))
        enc = ST.PromptEncoder(self.tk)
        for c in cases + cases:                         # the second round finds its own earlier prompts
            if c.get("rendered"):
                with self.subTest(case=c["name"]):
                    self.check(enc, c["rendered"])

    def test_long_conversations(self):
        rnd = random.Random(11)
        specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_response>"]
        enc = ST.PromptEncoder(self.tk)
        for conv in range(3):
            msgs = [{"role": "system", "content": random_text(rnd, 40)}]
            prev, prev_effort, edited = "", None, False
            for turn in range(12):
                msgs.append({"role": "user", "content": random_text(rnd, rnd.randint(1, 120), specials)})
                effort = rnd.choice(["low", "medium", "xhigh"]) if turn % 5 == 4 else "xhigh"
                prompt = self.template.render(msgs, reasoning_effort=effort)
                with self.subTest(conv=conv, turn=turn):
                    self.check(enc, prompt)
                    if effort == prev_effort and not edited:    # the previous prompt, but for its last few tokens
                        self.assertTrue(prompt.startswith(prev))
                        self.assertGreater(enc.last_reused, len(prev) - 100)
                prev, prev_effort, edited = prompt, effort, False
                answer = {"role": "assistant", "content": random_text(rnd, rnd.randint(1, 80), specials)}
                if rnd.random() < 0.7:
                    answer["reasoning_content"] = random_text(rnd, rnd.randint(1, 80))
                msgs.append(answer)
                if rnd.random() < 0.2:                  # a client that edits an earlier turn
                    i = rnd.randrange(1, len(msgs))
                    msgs[i] = dict(msgs[i], content=msgs[i]["content"] + " (edited)")
                    edited = True

    def test_prefix_of_an_earlier_prompt_and_unrelated_prompts(self):
        enc = ST.PromptEncoder(self.tk, keep=2)
        a = self.template.render([{"role": "user", "content": "first question " * 50}])
        b = self.template.render([{"role": "user", "content": "another conversation"}])
        c = self.template.render([{"role": "user", "content": "a third one"}])
        for text in (a, a[:len(a) // 2], b, c, a, a + "tail", "", "<|im_end|>", "<|im_end|><|im_end|>x"):
            with self.subTest(text=text[:30]):
                self.check(enc, text)


class Margin(unittest.TestCase):
    """A special literal that is a prefix of a longer one: cutting at the shorter one's end without the look-ahead
    margin would give different ids."""

    def test_longer_literal_past_the_common_prefix(self):
        tk = byte_tokenizer(["<|a|>", "<|a|>zz"])
        enc = ST.PromptEncoder(tk)
        first = "x<|a|>y" + "<|a|>" * 3 + "q"
        self.assertEqual(enc.encode(first), tk.encode(first, parse_special=True))
        second = "x<|a|>y" + "<|a|>" * 3 + "zz"             # shares everything up to the last <|a|>
        got = enc.encode(second)
        self.assertEqual(got, tk.encode(second, parse_special=True))
        self.assertEqual(got[-1], tk.special_tokens["<|a|>zz"])
        self.assertGreater(enc.last_reused, 0)              # the earlier boundaries were still reused

    def test_literal_ending_inside_another(self):
        tk = byte_tokenizer(["<|a|>", "b<|a|>c"])
        enc = ST.PromptEncoder(tk)
        for text in ("zzb<|a|>d", "zzb<|a|>c", "zzb<|a|>cz<|a|>", "zzb<|a|>cz<|a|>b<|a|>c"):
            with self.subTest(text=text):
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True))

    def test_random_specials(self):
        """Literals that extend or end inside one another, texts that change right after one of them."""
        rnd = random.Random(3)

        def word(lo, hi):
            return "".join(rnd.choice("<|ab>") for _ in range(rnd.randint(lo, hi)))
        for trial in range(60):
            a, b = word(2, 4), word(2, 4)
            lits = sorted({a, a + word(1, 3), b, word(1, 2) + b + word(0, 2)})
            tk = byte_tokenizer(lits)
            enc = ST.PromptEncoder(tk, keep=3)
            pieces = lits + [x[:rnd.randint(1, len(x))] for x in lits] + ["x", "y", "a", "|"]
            base = "".join(rnd.choice(pieces) for _ in range(60))
            for _ in range(30):
                cut = rnd.randint(0, len(base))
                text = base[:cut] + "".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 6)))
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True), (lits, text))
                if rnd.random() < 0.3:
                    base = text + base[cut:]


class CommonPrefix(unittest.TestCase):
    def test_against_a_loop(self):
        rnd = random.Random(5)
        for _ in range(500):
            a = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 9000)))
            k = rnd.randint(0, len(a))
            b = a[:k] + rnd.choice(["", "c", "cd"]) + a[k:k + rnd.randint(0, 50)]
            want = 0
            while want < min(len(a), len(b)) and a[want] == b[want]:
                want += 1
            self.assertEqual(ST.common_prefix_len(a, b), want)
            self.assertEqual(ST.common_prefix_len(b, a), want)


if __name__ == "__main__":
    unittest.main()
