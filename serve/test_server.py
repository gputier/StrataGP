"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (CTX_SLACK, ByteTokenizer, Detokenizer, EngineDied, MockEngine, Service,  # noqa: E402
                          StrataEngine, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "x" * 2000                              # longer than the old 1024 fallback: one token per byte


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_rejected(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, _, _ = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 400)
                self.assertIn("exceeds the context", b["error"]["message"])

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class FitMaxTokens(unittest.TestCase):
    """PR #24: --fit-max-tokens clamps an explicit budget that overshoots the context instead of a 400."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = MaxTokens.post
    call = MaxTokens.call

    def test_overshoot_is_clamped_to_the_room(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, ct = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_a_budget_that_fits_is_unchanged(self):
        s, b, _, ct = self.call("openai", max_tokens=50)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_no_room_is_still_a_400(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")
        s, b, _, _ = self.call("openai", text="y" * (CTX - CTX_SLACK - overhead), max_tokens=100)
        self.assertEqual(s, 400, b)
        self.assertIn("no room to answer", b["error"]["message"])


class ImageMarkers(unittest.TestCase):
    """#150: the text "<|image_pad|>" inside a message is text, not an image's place."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_literal_marker_with_an_image(self):
        import tempfile
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          vision=self.FakeVision(d))
            pad = tok.encode("<|image_pad|>", parse_special=True)[0]
            for text in ("the docs say <|image_pad|> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                                         {"type": "image", "source": "x.png"}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<|image_pad|> marks" if "docs" in text else "plain", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class StatusNeedsTheKey(unittest.TestCase):
    """#212: /status shows the end of the answer being written, so it needs the key like /v1/*."""

    def test_status(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = "k3y"
        httpd = serve(svc, port=0)
        root = f"http://127.0.0.1:{httpd.server_address[1]}"
        base = root + "/status"
        auth = {"Authorization": "Bearer k3y"}
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base, timeout=10)
            self.assertEqual(e.exception.code, 401)
            e.exception.close()
            # a whole request first: the answer's end is kept while it is written and must be gone once it is done
            body = json.dumps({"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]})
            with urllib.request.urlopen(urllib.request.Request(
                    root + "/v1/chat/completions", data=body.encode(),
                    headers={**auth, "Content-Type": "application/json"}), timeout=30) as r:
                self.assertEqual(r.status, 200)
                self.assertEqual(json.loads(r.read())["choices"][0]["finish_reason"], "stop")
            with urllib.request.urlopen(urllib.request.Request(base, headers=auth), timeout=10) as r:
                self.assertEqual(r.status, 200)
                status = json.loads(r.read())
            self.assertFalse(status["busy"])
            self.assertNotIn("tail", status)
        finally:
            httpd.shutdown()
            httpd.server_close()


class ToolCallTerminators(unittest.TestCase):
    """#210: a value that contains </parameter> or </tool_call> (a file documenting the call format) is kept whole."""
    CONTENT = ("Close each value with </parameter> and the call with </function></tool_call>.\n"
               "<parameter=x>\nnot a parameter\n</parameter>\nend")
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"},
                                                              "content": {"type": "string"}}}}]

    def run_parser(self, stream_tools, step):
        from serve.frontend import OutputParser
        text = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
                f"<parameter=content>\n{self.CONTENT}\n</parameter>\n</function>\n</tool_call>")
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish()
        return evs

    def test_values_keep_the_terminators(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.run_parser(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "content": self.CONTENT})
                    self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])
                    if stream_tools:
                        streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                        self.assertEqual(json.loads(streamed), {"path": "doc.md", "content": self.CONTENT})


class ClientShapes(unittest.TestCase):
    """What real clients send: Claude Code posts /v1/messages?beta=true (issue #55) and puts hook context into the
    conversation as a mid-conversation system message (issue #56); some OpenAI clients send a late developer message."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "</think>\n\n2", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt_text(self):
        return bytes(i for i in self.engine.last_ids if i < 256).decode("utf-8", "replace")

    def test_query_string(self):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        for path in ("/v1/messages?beta=true", "/v1/chat/completions?api-version=1", "/v1/messages/?beta=true"):
            status, b = self.post(path, body)
            self.assertEqual(status, 200, (path, b))
        status, _ = self.post("/v1/nothing?beta=true", body)
        self.assertEqual(status, 404)

    def test_anthropic_mid_conversation_system(self):
        status, b = self.post("/v1/messages?beta=true", {
            "model": "x", "max_tokens": 50,
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "1+1? digits only"}]},
                {"role": "system", "content": [{"type": "text", "text": "<system-reminder>answer in digits</system-reminder>"}]}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("You are terse.", text)
        self.assertIn("<system-reminder>answer in digits</system-reminder>", text)
        self.assertLess(text.index("You are terse."), text.index("1+1?"))          # the first system stays first
        self.assertLess(text.index("1+1?"), text.index("answer in digits"))        # the late one stays in place

    def test_openai_late_developer_and_system(self):
        status, b = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 50,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hi"}, {"role": "developer", "content": "Now use digits."},
                         {"role": "system", "content": "Also this."}, {"role": "user", "content": "1+1?"}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        for part in ("Be brief.", "Now use digits.", "Also this.", "1+1?"):
            self.assertIn(part, text)

    def test_leading_system_unchanged(self):
        from serve.frontend import anthropic_to_messages, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _, _ = anthropic_to_messages({"system": "S", "messages": [{"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class SamplingKeys(unittest.TestCase):
    """The GEN line's sampling keys: top_k 0 ("off") or wider than the engine's 64 get the widest list, 64 (they used
    to fall back to the engine default 20); a penalty always carries its window."""

    def keys(self, **sampling):
        return StrataEngine.sampling_keys(sampling).split()

    def test_top_k(self):
        self.assertIn("top_k=10", self.keys(temperature=0.7, top_k=10))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=64))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=100))
        for bad in (-1, True, 2.5, "20"):
            self.assertFalse([k for k in self.keys(temperature=0.7, top_k=bad) if k.startswith("top_k=")], bad)

    def test_tune_keys(self):
        k = self.keys(temperature=0, strata_tune={"pcie_frac": 0.2, "spec_min_p": 0.7})
        self.assertIn("pcie_frac=0.2", k)
        self.assertIn("spec_min_p=0.7", k)
        bad = self.keys(strata_tune={"pcie_frac": 3, "spec_min_p": True, "pool_workers": 2})
        self.assertFalse([x for x in bad if x.split("=")[0] in ("pcie_frac", "spec_min_p", "pool_workers")])

    def test_penalty_window(self):
        self.assertIn("penalty_last_n=64", self.keys(presence_penalty=1.5))
        self.assertIn("penalty_last_n=4096", self.keys(repetition_penalty=1.1, penalty_last_n=4096))
        self.assertFalse([k for k in self.keys(temperature=0.7) if k.startswith("penalty")])


class GpuChoice(unittest.TestCase):
    """Issue #51: the config's \"gpu\" reaches the engine as CUDA_VISIBLE_DEVICES, numbered like nvidia-smi."""

    def test_env(self):
        from serve.server import child_env
        env = child_env({"gpu": 1})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        plain = child_env({})                     # no choice: the environment as it was (existing installs)
        self.assertEqual(plain.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(plain.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_DEVICE_ORDER"))


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Issue #27: an engine that dies after a few tokens of its first answer, and comes back when restarted."""

    def __init__(self, tok, script, max_context):
        super().__init__(tok, script, max_context=max_context)
        self.dead, self.restarts, self.die_after = False, 0, 5

    def alive(self):
        return not self.dead

    def restart(self):
        self.dead, self.die_after = False, None
        self.restarts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.die_after is not None and i == self.die_after:
                self.dead = True
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class EngineDeath(unittest.TestCase):
    """Issue #27: a dead engine is an error (not "length"), and the next request starts it again."""

    def test_error_then_restart(self):
        tok = ByteTokenizer()
        eng = DyingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(body):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, r.read().decode()
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, e.read().decode()
            msgs = [{"role": "user", "content": "hi"}]
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"error"', text)
            self.assertIn("stopped unexpectedly", text)
            self.assertTrue(text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50})
            self.assertEqual(code, 200, text)
            self.assertEqual(eng.restarts, 1)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 50)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_err_mid_stream(self):
        """The engine's ERR line after the stream started reaches the client as an error event (it used to be a
        400 written into the open stream, which clients read as an empty answer)."""
        class ErrEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield None                                  # a prompt-progress heartbeat: the stream has started
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")

        tok = ByteTokenizer()
        svc = Service(ErrEngine(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in [("/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 20,
                                                          "messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"model": "m", "stream": True, "max_tokens": 20,
                                                 "messages": [{"role": "user", "content": "hi"}]})]:
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode()
                self.assertIn("illegal memory access", text, path)
                self.assertNotIn("HTTP/1", text, path)
                self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
        finally:
            httpd.shutdown()
            httpd.server_close()


class SharedSettings(unittest.TestCase):
    """The web app's "Use for other apps too": POST /settings makes its Chat settings every client's defaults."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        class Sampled(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = Sampled(tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc.shared_path = os.path.join(cls.tmp.name, "strata-x.shared-settings.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def req(self, path, body, headers=None, raw=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})

    def tearDown(self):
        self.svc.set_shared(None)

    def test_other_apps_get_the_chat_settings(self):
        d = {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 7, "max_tokens": 77,
             "reasoning_effort": "low", "experimental_speed_projection": False}
        code, b = self.req("/settings", {"defaults": d})
        self.assertEqual(code, 200, b)
        self.assertTrue(b["shared"])
        self.assertTrue(os.path.exists(self.svc.shared_path))
        code, _ = self.chat()                                        # a client that sets nothing
        self.assertEqual(code, 200)
        got = self.engine.last_sampling
        for k in ("temperature", "top_p", "top_k", "seed", "experimental_speed_projection"):
            self.assertEqual(got[k], d[k], k)
        self.assertEqual(self.engine.last_max_new, 77)
        code, _ = self.chat(temperature=0.9, max_tokens=5)          # its own values win
        self.assertEqual(self.engine.last_sampling["temperature"], 0.9)
        self.assertEqual(self.engine.last_max_new, 5)
        r = self.svc.with_shared({"messages": []}, "openai")
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertEqual(self.svc.with_shared({"reasoning_effort": "high"}, "openai")["reasoning_effort"], "high")
        self.assertEqual(self.svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})

    def test_off_again(self):
        self.req("/settings", {"defaults": {"temperature": 0.3}})
        code, b = self.req("/settings", {"defaults": None})
        self.assertEqual((code, b["shared"]), (200, False))
        self.assertFalse(os.path.exists(self.svc.shared_path))
        self.chat()
        self.assertNotIn("temperature", self.engine.last_sampling)

    def test_only_strata_s_own_page_may_set_them(self):
        code, _ = self.req("/settings", None, {"Content-Type": "text/plain"}, raw=b'{"defaults": {"temperature": 1}}')
        self.assertEqual(code, 415)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, b = self.req("/settings", {"defaults": {"temperature": 9}})
        self.assertEqual(code, 400)
        self.assertIn("temperature", b["error"]["message"])
        self.assertEqual(self.svc.shared, {})
        host = self.base.split("://", 1)[1]
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_they_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}})[0], 401)
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}},
                                      {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"/web/app.js", body)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


def old_deltas(tok, ids):
    d = Detokenizer(tok)
    d.fast = False
    return [d.push(t) for t in ids]


def new_deltas(tok, ids):
    d = Detokenizer(tok)
    assert d.fast
    return [d.push(t) for t in ids]


class StreamingDetokenizer(unittest.TestCase):
    """Issue #31: the incremental UTF-8 decoder gives the old re-decode's deltas, one for one."""

    def random_ids(self, rnd, n):
        out = []
        chars = "aé你\U0001f600 \n"
        while len(out) < n:
            r = rnd.random()
            if r < 0.5:
                out += list(rnd.choice(chars).encode())          # whole or (cut below) split characters
            elif r < 0.7:
                out.append(rnd.randrange(0x80, 0x100))            # stray continuation / lead / invalid bytes
            elif r < 0.75:
                out += list("�".encode())                     # a real U+FFFD in the text
            elif r < 0.8:
                out.append(256 + rnd.randrange(len(ByteTokenizer.SPECIALS)))
            else:
                out += list(rnd.choice(chars).encode())[:rnd.randint(1, 3)]   # a character cut short
        return out[:n]

    def test_same_deltas_as_the_re_decode(self):
        import random
        tok, rnd = ByteTokenizer(), random.Random(1)
        for i in range(400):
            ids = self.random_ids(rnd, rnd.randint(0, 60))
            with self.subTest(i=i):
                self.assertEqual(new_deltas(tok, ids), old_deltas(tok, ids))

    def test_real_vocabulary(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from test_strata_tokenizer import load_tokenizer
        tk = load_tokenizer()
        if tk is None:
            self.skipTest("no tokenizer (set STRATA_TOKENIZER or STRATA_GGUF_PY)")
        import random
        rnd = random.Random(2)
        text = "Café 你好世界 \U0001f600\U0001f680 é مرحبا def f(x):\n\treturn x\n"
        ids = tk.encode(text * 20)
        self.assertEqual("".join(new_deltas(tk, ids)), text * 20)
        self.assertEqual(new_deltas(tk, ids), old_deltas(tk, ids))
        for i in range(50):                              # any ids, byte tokens included
            ids = [rnd.randrange(len(tk.tokens) - 400) for _ in range(rnd.randint(1, 40))]
            ids += [rnd.randrange(0x80 - 0x21, 256) for _ in range(rnd.randint(0, 6))]
            rnd.shuffle(ids)
            with self.subTest(i=i):
                self.assertEqual(new_deltas(tk, ids), old_deltas(tk, ids))

    def test_old_path_on_request(self):
        os.environ["STRATA_OLD_DETOK"] = "1"
        try:
            self.assertFalse(Detokenizer(ByteTokenizer()).fast)
        finally:
            del os.environ["STRATA_OLD_DETOK"]


class IncrementalPrompts(unittest.TestCase):
    """Issue #32 over HTTP: every turn's prompt ids are those of a full encode, and a turn reuses the previous one's."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "Thinking.\n</think>\n\nThe answer.", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = ClientShapes.post

    def test_turns(self):
        self.assertIsNotNone(self.svc.prompts)
        msgs = [{"role": "system", "content": "Be brief."}]
        for turn in range(6):
            msgs.append({"role": "user", "content": f"question {turn} <|im_end|> é你 " * (turn + 1)})
            status, b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 64, "messages": msgs})
            self.assertEqual(status, 200, b)
            prompt = self.svc.template.render(msgs)
            self.assertEqual(self.engine.last_ids, self.svc.tok.encode(prompt, parse_special=True))
            if turn:
                self.assertGreater(self.svc.prompts.last_reused, len(prompt) // 3)
            msgs.append({"role": "assistant", "content": b["choices"][0]["message"]["content"]})

    def test_check_mode_and_old_path(self):
        import contextlib
        import io
        os.environ["STRATA_CHECK_PROMPT_IDS"] = "1"
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ids = self.svc.encode_prompt("<|im_start|>user\nhi<|im_end|>\n")
            self.assertEqual(ids, self.svc.tok.encode("<|im_start|>user\nhi<|im_end|>\n", parse_special=True))
            self.assertNotIn("DIFFER", out.getvalue())
        finally:
            del os.environ["STRATA_CHECK_PROMPT_IDS"]
        os.environ["STRATA_OLD_PROMPT_ENCODE"] = "1"
        try:
            svc = Service(self.engine, self.svc.tok, self.svc.template)
            self.assertIsNone(svc.prompts)
        finally:
            del os.environ["STRATA_OLD_PROMPT_ENCODE"]


class ThinkingLevelNote(unittest.TestCase):
    """Issue #33: a thinking level changed in the middle of a conversation is said in the server window."""

    def test_note(self):
        import contextlib
        import io
        tok = ByteTokenizer()
        svc = Service(RecordingPrompt(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            svc.prepare(conv[:1], None, {"reasoning_effort": "medium"})
            svc.prepare(conv + [{"role": "user", "content": "more"}], None, {"reasoning_effort": "medium"})
            self.assertNotIn("thinking level changed", out.getvalue())
            svc.prepare(conv + [{"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"},
                                {"role": "user", "content": "again"}], None, {"reasoning_effort": "low"})
            self.assertIn("thinking level changed (medium -> low)", out.getvalue())
            out.truncate(0)
            svc.prepare([{"role": "user", "content": "another conversation"}], None, {"enable_thinking": False})
            self.assertNotIn("thinking level changed", out.getvalue())


class RecallReasoning(unittest.TestCase):
    """Issue #34: with --recall-reasoning a client that returns only the answer still continues the engine's live
    sequence; without it, the prompt diverges right after <think> (the old behaviour, unchanged by default)."""

    SCRIPT = "Let me think.\nStill thinking.\n</think>\n\nThe answer is 4."

    def make(self, recall, script=None):
        from serve.server import ReasoningRecall
        tok = ByteTokenizer()
        engine = RecordingPrompt(tok, script or self.SCRIPT, max_context=CTX)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if recall:
            svc.recall = ReasoningRecall()
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return svc, engine, f"http://127.0.0.1:{httpd.server_address[1]}"

    def chat(self, base, body):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def two_turns(self, recall):
        svc, engine, base = self.make(recall)
        msgs = [{"role": "user", "content": "2+2?"}]
        b = self.chat(base, {"model": "m", "max_tokens": 200, "messages": msgs})
        live = engine.last_ids + list(engine.script)       # what the engine holds after turn 1
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["content"], "The answer is 4.")
        msgs += [{"role": "assistant", "content": msg["content"]}, {"role": "user", "content": "and 3+3?"}]
        self.chat(base, {"model": "m", "max_tokens": 200, "messages": msgs})
        return svc, engine, live

    def test_live_prefix_with_recall(self):
        _, engine, live = self.two_turns(True)
        self.assertEqual(engine.last_ids[:len(live)], live)

    def test_default_is_unchanged(self):
        svc, engine, live = self.two_turns(False)
        self.assertIsNone(svc.recall)
        self.assertNotEqual(engine.last_ids[:len(live)], live)
        text = bytes(t for t in engine.last_ids if t < 256).decode()
        self.assertIn("<think>\n\n</think>\n\nThe answer is 4.", text)

    def test_other_conversation_gets_nothing(self):
        svc, engine, base = self.make(True)
        self.chat(base, {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "2+2?"}]})
        self.chat(base, {"model": "m", "max_tokens": 200, "messages": [
            {"role": "user", "content": "something else"}, {"role": "assistant", "content": "The answer is 4."},
            {"role": "user", "content": "ok"}]})
        text = bytes(t for t in engine.last_ids if t < 256).decode()
        self.assertNotIn("Still thinking", text)

    def test_client_reasoning_wins_and_tool_calls(self):
        script = ("Need the tool.\n</think>\n\n<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n"
                  "</function>\n</tool_call>")
        svc, engine, base = self.make(True, script)
        tools = [{"type": "function", "function": {"name": "add", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}}}}}]
        msgs = [{"role": "user", "content": "add"}]
        b = self.chat(base, {"model": "m", "max_tokens": 300, "messages": msgs, "tools": tools})
        live = engine.last_ids + list(engine.script)
        call = b["choices"][0]["message"]["tool_calls"][0]
        msgs += [{"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"], "content": "4"}]
        self.chat(base, {"model": "m", "max_tokens": 300, "messages": msgs, "tools": tools})
        self.assertEqual(engine.last_ids[:len(live)], live)
        msgs[1] = dict(msgs[1], reasoning_content="My own words.")          # a client that sends its own
        self.chat(base, {"model": "m", "max_tokens": 300, "messages": msgs, "tools": tools})
        text = bytes(t for t in engine.last_ids if t < 256).decode()
        self.assertIn("My own words.", text)
        self.assertNotIn("Need the tool.", text)

    def test_chat_py_sends_its_reasoning_back(self):
        sys.path.insert(0, str(ROOT))
        import chat
        _, engine, base = self.make(False)                 # a default server: chat.py alone is enough
        url = base + "/v1/chat/completions"
        msgs = [{"role": "user", "content": "2+2?"}]
        parts = list(chat.stream(url, msgs, "high", 200))
        live = engine.last_ids + list(engine.script)
        msgs.append(chat.answer_message("".join(p[1] for p in parts), "".join(p[0] for p in parts)))
        self.assertEqual(msgs[-1]["reasoning_content"], "Let me think.\nStill thinking.\n")
        msgs.append({"role": "user", "content": "and 3+3?"})
        list(chat.stream(url, msgs, "high", 200))
        self.assertEqual(engine.last_ids[:len(live)], live)
        self.assertNotIn("reasoning_content", chat.answer_message("a", "b", drop_thinking=True))

    def test_bench_turns(self):
        """tools/bench_turns.py, the A/B client of the doc, against this server."""
        import contextlib
        import io
        import tempfile
        sys.path.insert(0, str(ROOT / "tools"))
        import bench_turns
        _, engine, base = self.make(True)
        port = int(base.rsplit(":", 1)[1])
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench_turns.main(["--port", str(port), "--turns", "3", "--context-chars", "600",
                                               "--max-tokens", "100", "--json", d + "/rows.json"]), 0)
            rows = json.loads(Path(d, "rows.json").read_text())
        self.assertEqual([r["turn"] for r in rows], [1, 2, 3])
        self.assertTrue(all(r["first_text_s"] is not None for r in rows))
        self.assertLess(rows[0]["prompt_tokens"], rows[1]["prompt_tokens"])
        self.assertIn("Still thinking", bytes(t for t in engine.last_ids if t < 256).decode())   # recalled

    def test_bounded(self):
        from serve.server import ReasoningRecall
        import hashlib
        r = ReasoningRecall(max_entries=3, max_chars=100)
        for i in range(10):
            h = hashlib.sha256(str(i).encode())
            r.remember(h, f"answer {i}", [], "x" * 30)
        self.assertLessEqual(len(r.table), 3)
        self.assertLessEqual(r.chars, 100)


class VisionCache(unittest.TestCase):
    """Issue #35: an image already encoded is found by its raw bytes, before any conversion, and a request with one
    image links its cached embeddings instead of copying them."""

    def make_vision(self):
        import tempfile
        from serve.server import Vision

        class FakeProc:
            def __init__(self):
                self.stdin, self.stdout, self.encodes = self, self, 0

            def write(self, line):
                _, img, out = line.split()
                Path(out).write_bytes(b"EMB" + Path(img).read_bytes())
                self.encodes += 1

            def flush(self):
                pass

            def readline(self):
                return "OK 3\n"
        v = Vision.__new__(Vision)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        v.dir, v.lock, v.cache, v.proc = Path(tmp.name), __import__("threading").Lock(), {}, FakeProc()
        v.normalized = 0

        def normalize(data):
            v.normalized += 1
            return b"PNG:" + data
        v.normalize = normalize
        return v

    def test_raw_bytes_hit_before_normalize(self):
        import base64
        v = self.make_vision()
        src = "data:image/webp;base64," + base64.b64encode(b"RIFF....WEBPVP8 fake").decode()
        a = v.encode(src)
        b = v.encode(src)
        self.assertEqual(a, b)
        self.assertEqual(v.normalized, 1)
        self.assertEqual(v.proc.encodes, 1)
        self.assertEqual(a[0].read_bytes(), b"EMBPNG:RIFF....WEBPVP8 fake")

    def test_one_image_request_links(self):
        import base64
        v = self.make_vision()
        tok = ByteTokenizer()
        svc = Service(RecordingPrompt(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        src = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
        msgs = [{"role": "user", "content": [{"type": "image", "source": src}, {"type": "text", "text": "what?"}]}]
        ids, _, _ = svc.prepare(msgs, None, {}, 16)
        self.assertEqual(ids.count(256 + ByteTokenizer.SPECIALS.index("<|image_pad|>")), 3)
        req = Path(svc.embeddings.path)
        cached = v.cache[next(iter(v.cache))][0]
        self.assertEqual(req.read_bytes(), cached.read_bytes())
        req.unlink()                                     # what Service.run does after the request
        self.assertTrue(cached.exists())
        msgs[0]["content"].append({"type": "image", "source": src})      # two images: one file, both in order
        ids, _, _ = svc.prepare(msgs, None, {}, 16)
        self.assertEqual(Path(svc.embeddings.path).read_bytes(), cached.read_bytes() * 2)
        self.assertEqual(v.proc.encodes, 1)


class _Reached(Exception):
    """raised by the patched serve(): main() got past the API key check."""


class EmptyApiKey(unittest.TestCase):
    """#213: a key that is present but empty is refused before the engine loads; the effective value counts."""

    def run_main(self, argv, env=None, cfg=None):
        """-> (return code, the API key the service got, or None when main() stopped before serving)."""
        import io
        import tempfile
        from unittest import mock
        import serve.server as S
        args = ["server.py", "--engine", "mock", "--port", "0", *argv]
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen["key"] = svc.api_key
            raise _Reached

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch.object(S, "serve", fake_serve), mock.patch.object(sys, "argv", args), \
                mock.patch.object(sys, "stderr", new=io.StringIO()):
            if env is None or "STRATA_API_KEY" not in env:
                os.environ.pop("STRATA_API_KEY", None)
            if cfg is not None:
                p = Path(d) / "cfg.json"
                p.write_text(json.dumps(cfg), encoding="utf-8")
                args += ["--config", str(p)]
            try:
                return S.main(), seen.get("key")
            except _Reached:
                return 0, seen["key"]

    def test_abbreviated_empty_flag_refused(self):
        self.assertEqual(self.run_main(["--api="]), (2, None))
        self.assertEqual(self.run_main(["--api", ""]), (2, None))
        self.assertEqual(self.run_main(["--api-key", "  "]), (2, None))

    def test_empty_env_alone_refused(self):
        self.assertEqual(self.run_main([], env={"STRATA_API_KEY": ""}), (2, None))

    def test_empty_env_with_key_on_the_command_line_accepted(self):
        self.assertEqual(self.run_main(["--api-key", "abc"], env={"STRATA_API_KEY": ""}), (0, "abc"))

    def test_env_key_accepted(self):
        self.assertEqual(self.run_main([], env={"STRATA_API_KEY": "fromenv"}), (0, "fromenv"))

    def test_empty_key_in_config_refused(self):
        self.assertEqual(self.run_main([], cfg={"api_key": ""}), (2, None))
        self.assertEqual(self.run_main([], cfg={"api_key": "   "}), (2, None))

    def test_no_key_at_all_still_starts(self):
        self.assertEqual(self.run_main([]), (0, ""))

    def test_config_key_used(self):
        self.assertEqual(self.run_main([], cfg={"api_key": "cfgkey"}), (0, "cfgkey"))


if __name__ == "__main__":
    unittest.main()
