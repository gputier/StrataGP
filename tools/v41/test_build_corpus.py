"""Tests of build_corpus.py on synthetic sources.

    python -m pytest tools/v41/test_build_corpus.py

V41_ENCODING_PY=<DeepSeek-V4.1 encoding.py> also runs the tests that render with the real chat template.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_corpus as bc  # noqa: E402

REAL_ENCODING = os.environ.get("V41_ENCODING_PY")
PARAMS = dict(tokens_per_half=6000, long_tokens=1500, file_chars=600, seed=7)


def fake_encode(messages):
    """A stand-in for encode_messages: role-tagged contents, tool calls and tool names included."""
    out = ["<BOS>"]
    for m in messages:
        out.append(f"[{m['role']}]")
        if m.get("tools"):
            out.append("tools:" + ",".join(t["function"]["name"] for t in m["tools"]))
        out.append(m.get("content") or "")
        for call in m.get("tool_calls", []):
            out.append("call:" + call["function"]["name"] + call["function"]["arguments"])
    return "\n".join(out)


def make_candidates():
    cands = []
    for i in range(40):
        body = "".join(f"int f{i}_{j}(int x) {{ return x * {j} + {i}; }}\n" for j in range(150))
        cands.append(("code", f"src/mod{i:02d}.cpp", body))
    for i in range(14):
        body = "".join(f"Ligne {j} du document {i} qui parle d'un cache d'experts.\n" for j in range(120))
        cands.append(("doc", f"docs/DOC{i:02d}.md", body))
    return cands


def make_articles():
    arts = []
    for i in range(36):
        title = f"Article {i:02d}"
        text = f" = {title} = \n" + "".join(f" Phrase numero {j} de l'article {i}, sur un sujet distinct . \n"
                                           for j in range(90))
        arts.append((title, text))
    return arts


def build(seed=7, candidates=None, **over):
    p = {**PARAMS, "seed": seed, **over}
    return bc.build(candidates or make_candidates(), make_articles(), bc.SHORT_QUESTIONS, fake_encode,
                    bc.Counter(), **p)


def kinds(parts):
    out = {}
    for it in parts:
        out[it["kind"]] = out.get(it["kind"], 0) + 1
    return out


def sources(parts):
    return {s for it in parts for s in it["sources"]}


def test_build_is_deterministic():
    t1, p1 = build()
    t2, p2 = build()
    assert t1 == t2 and p1 == p2
    t3, _ = build(seed=8)
    assert t3 != t1


def test_halves_share_no_source():
    _, parts = build()
    a, b = sources(parts["A"]), sources(parts["B"])
    assert a and b and not (a & b)
    files = {s for s in a | b if s.startswith(("src/", "docs/"))}
    assert len(files) > 10


def test_no_file_is_used_twice_inside_a_half():
    _, parts = build()
    for name in "AB":
        paths = [s for it in parts[name] if it["kind"] != "wikitext" for s in it["sources"]
                 if s.startswith(("src/", "docs/"))]
        assert len(paths) == len(set(paths))


def test_composition_is_balanced():
    texts, parts = build()
    ka, kb = kinds(parts["A"]), kinds(parts["B"])
    assert set(ka) == set(kb) == set(bc.KINDS) | {"long", "wikitext"}
    assert ka["long"] == kb["long"] == 1
    for k in bc.KINDS:
        assert abs(ka[k] - kb[k]) <= 1
    counter = bc.Counter()
    tokens = {n: counter.count(texts[n]) for n in "AB"}
    for n in "AB":
        assert abs(tokens[n] - PARAMS["tokens_per_half"]) < 0.06 * PARAMS["tokens_per_half"]
        wiki = sum(counter.count(it["text"]) for it in parts[n] if it["kind"] == "wikitext")
        assert abs(wiki / tokens[n] - 0.5) < 0.08
    assert abs(tokens["A"] - tokens["B"]) < 0.06 * PARAMS["tokens_per_half"]


def test_long_prompt_reaches_its_budget():
    _, parts = build()
    counter = bc.Counter()
    for n in "AB":
        long_text = next(it["text"] for it in parts[n] if it["kind"] == "long")
        assert abs(counter.count(long_text) - PARAMS["long_tokens"]) < 0.05 * PARAMS["long_tokens"]
        assert bc.LONG_ASK in long_text


def test_usage_items_carry_their_markers():
    _, parts = build()
    by = {k: next(it for it in parts["A"] if it["kind"] == k) for k in bc.KINDS}
    assert "tools:read_file,grep,list_dir" in by["tool_read"]["text"] and "call:read_file" in by["tool_read"]["text"]
    assert "call:grep" in by["tool_grep"]["text"] and "[tool]" in by["tool_grep"]["text"]
    assert "français" in by["explain_doc"]["text"] or "francais" in by["explain_doc"]["text"]
    assert by["short"]["sources"][0] in bc.SHORT_QUESTIONS


def test_wikitext_part_is_cut_at_a_line_and_whole_articles_come_first():
    _, parts = build()
    wiki = [it for it in parts["A"] if it["kind"] == "wikitext"]
    assert all(it["text"].endswith("\n") for it in wiki)
    assert sum(1 for it in wiki if it["sources"][0].endswith("(cut)")) <= 1


def test_not_enough_wikitext_stops_the_build():
    with pytest.raises(SystemExit):
        bc.build(make_candidates(), make_articles()[:2], bc.SHORT_QUESTIONS, fake_encode, bc.Counter(), **PARAMS)


# ---- sources

@pytest.mark.parametrize("path,size,expected", [
    ("src/core/expert_cache.cpp", 5000, "code"),
    ("include/strata/a.hpp", 5000, "code"),
    ("tools/make_profile.py", 5000, "code"),
    ("docs/FORK.md", 5000, "doc"),
    ("docs/benchmarks/x.md", 5000, "doc"),
    ("docs/media/x.md", 5000, None),
    ("tools/v41/build_corpus.py", 5000, None),
    ("third_party/lib/a.cpp", 5000, None),
    ("bench/results/a.json", 5000, None),
    ("src/core/tiny.cpp", 100, None),
    ("src/core/huge.cpp", 5_000_000, None),
    ("README.md", 5000, None),
    ("src/core/data.json", 5000, None),
])
def test_classify(path, size, expected):
    assert bc.classify(path, size) == expected


def test_clean_text_refuses_template_characters_and_binary():
    assert bc.clean_text("plain text\r\nsecond".encode()) == "plain text\nsecond"
    assert bc.clean_text("a <｜User｜> b".encode()) is None
    assert bc.clean_text("a▁b".encode()) is None
    assert bc.clean_text(b"a\0b") is None
    assert bc.clean_text(b"\xff\xfe") is None


def test_head_and_fit_prefix():
    assert bc.head("abc\ndef\nghi\n", 8) == "abc\ndef\n"
    assert bc.head("abc\ndef\nghi\n", 6) == "abc\n"
    assert bc.head("short", 100) == "short"
    assert bc.head("nolinebreakhere", 5) == "nolin"
    lines = [f"line {i}\n" for i in range(100)]
    got = bc.fit_prefix(bc.Counter(), lines, 50)
    assert got == "".join(lines[:len(got.splitlines())]) and bc.Counter().count(got) <= 50
    assert bc.Counter().count(got + lines[len(got.splitlines())]) > 50


def test_the_text_of_the_rows_is_joined_with_a_bare_newline_for_an_empty_row(tmp_path):
    rows = ["", " = A = \n", "", "text one\n"]
    assert bc.rows_to_text(rows) == "\n = A = \n\ntext one\n"
    p = tmp_path / "w.jsonl"
    p.write_text("".join(json.dumps({"text": r}) + "\n" for r in rows), encoding="utf-8")
    assert bc.read_rows(p) == rows


def test_load_wikitext_splits_at_level_one_headings(tmp_path):
    rows = ["", " = Alpha = \n", "", " Un texte . \n", "", " = = Section = = \n", " Suite . \n", "",
            " = Beta = \n", " Autre . \n"]
    p = tmp_path / "w.jsonl"
    p.write_text("".join(json.dumps({"text": r}) + "\n" for r in rows), encoding="utf-8")
    arts = bc.load_wikitext(p)
    assert [t for t, _ in arts] == ["Alpha", "Beta"]
    assert "Section" in arts[0][1] and "Autre" in arts[1][1] and "Autre" not in arts[0][1]


def test_tar_source_keeps_only_candidates(tmp_path):
    t = tmp_path / "repo.tar"
    with tarfile.open(t, "w") as tar:
        for name, text in [("src/a.cpp", "x" * 2000), ("src/small.cpp", "x"), ("docs/A.md", "y" * 2000),
                           ("tools/v41/me.py", "z" * 2000), ("src/bin.cpp", "\0" * 2000)]:
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    assert [(k, p) for k, p, _ in bc.tar_candidates(str(t))] == [("doc", "docs/A.md"), ("code", "src/a.cpp")]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_source_reads_tracked_files_only(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                       capture_output=True)
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tracked.cpp").write_text("int a;\n" * 400)
    (tmp_path / "src" / "untracked.cpp").write_text("int b;\n" * 400)
    git("add", "src/tracked.cpp")
    git("commit", "-q", "-m", "one")
    (tmp_path / "src" / "tracked.cpp").write_text("int changed;\n" * 400)      # the working tree moved on
    found = bc.git_candidates(tmp_path, "HEAD")
    assert [(k, p) for k, p, _ in found] == [("code", "src/tracked.cpp")]
    assert found[0][2].startswith("int a;")                                       # read with git show, not the disk


def test_render_item_shapes_the_tool_conversation():
    seen = []

    def spy(messages):
        seen.append(messages)
        return "x"

    rng = bc.random.Random(1)
    bc.render_item("tool_read", ("src/a.cpp", "int a;\n" * 500), rng, spy, 300)
    msgs = seen[-1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[0]["tools"] == bc.TOOLS
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "src/a.cpp"}
    assert msgs[3]["tool_call_id"] == "call_0" and len(msgs[3]["content"]) <= 300
    bc.render_item("tool_grep", ("src/a.cpp", "alpha_beta = 1\n" * 50), rng, spy, 300)
    args = json.loads(seen[-1][2]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "src/a.cpp" and "alpha_beta" in seen[-1][3]["content"]


def test_pick_word_needs_an_identifier_of_six_characters():
    rng = bc.random.Random(1)
    with pytest.raises(ValueError, match="no identifier"):
        bc.pick_word("1 22 333 4444 abc_d", rng)
    assert bc.pick_word("alpha_beta alpha_beta alpha_beta other_name", rng) == "alpha_beta"
    assert bc.pick_word("zeta_one alpha_two", rng) == "alpha_two"       # none thrice: the first in order


def test_manifest_counts_usage_and_wikitext_tokens_by_hand():
    # the stand-in counter is ceil(chars / 3)
    parts = {"A": [{"kind": "short", "sources": ["q"], "text": "abcdef"},
                   {"kind": "wikitext", "sources": ["t"], "text": "abcdefghi"}],
             "B": [{"kind": "long", "sources": ["f"], "text": "abcd"},
                   {"kind": "wikitext", "sources": ["u"], "text": "ab"}]}
    texts = {"A": "abcdef\nabcdefghi\n", "B": "abcd\nab\n"}
    m = bc.manifest_for(texts, parts, bc.Counter(), {"seed": 1})
    a, b = m["halves"]["A"], m["halves"]["B"]
    assert (a["tokens_usage"], a["tokens_wikitext"], a["tokens"], a["chars"]) == (2, 3, 6, 17)      # 6/3, 9/3, 17/3
    assert (b["tokens_usage"], b["tokens_wikitext"], b["tokens"], b["chars"]) == (2, 1, 3, 8)       # 4/3, 2/3, 8/3
    assert a["by_kind"]["short"] == {"items": 1, "chars": 6, "tokens": 2}
    assert m["tokens_exact"] is False and m["params"] == {"seed": 1}
    assert a["sha256"] == bc.hashlib.sha256(texts["A"].encode()).hexdigest()


def test_manifest_usage_and_wikitext_add_up_on_a_built_corpus():
    texts, parts = build()
    m = bc.manifest_for(texts, parts, bc.Counter(), {})
    for n in "AB":
        row = m["halves"][n]
        wiki = sum(-(-len(it["text"]) // 3) for it in parts[n] if it["kind"] == "wikitext")
        usage = sum(-(-len(it["text"]) // 3) for it in parts[n] if it["kind"] != "wikitext")
        assert (row["tokens_usage"], row["tokens_wikitext"]) == (usage, wiki)
        assert sum(r["items"] for r in row["by_kind"].values()) == len(parts[n]) == len(row["sources"])


def write_tar(path):
    with tarfile.open(path, "w") as t:
        for k, name, text in make_candidates():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_main_hashes_the_source_tar_and_refuses_stdin(tmp_path):
    tar = tmp_path / "repo.tar"
    write_tar(tar)
    enc = tmp_path / "encoding.py"
    enc.write_text("def encode_messages(messages, thinking_mode='chat'):\n"
                   "    return '\\n'.join(m['role'] + (m.get('content') or '') for m in messages)\n")
    wiki = tmp_path / "w.jsonl"
    wiki.write_text("".join(json.dumps({"text": line}) + "\n" for _, text in make_articles()
                            for line in text.splitlines(keepends=True)), encoding="utf-8")
    argv = ["--out-dir", str(tmp_path / "corpus"), "--encoding-py", str(enc), "--wikitext", str(wiki),
            "--rev", "label", "--tokens-per-half", "6000", "--long-tokens", "1500", "--file-chars", "600"]
    assert bc.main([*argv, "--source-tar", str(tar)]) == 0
    params = json.loads((tmp_path / "corpus" / "manifest.json").read_text())["params"]
    assert params["source_tar_sha256"] == bc.sha256_of(tar)
    assert params["source"] == {"tar": str(tar), "rev": "label"}
    with pytest.raises(SystemExit) as stdin:
        bc.main([*argv, "--source-tar", "-"])
    assert stdin.value.code == 2


# ---- with DeepSeek's real chat template

@pytest.mark.skipif(not REAL_ENCODING, reason="V41_ENCODING_PY is not set")
def test_real_template_renders_every_kind():
    encode = bc.load_encoder(REAL_ENCODING)
    texts, parts = bc.build(make_candidates(), make_articles(), bc.SHORT_QUESTIONS, encode, bc.Counter(), **PARAMS)
    tool = next(it["text"] for it in parts["A"] if it["kind"] == "tool_read")
    assert tool.startswith("<｜begin▁of▁sentence｜>")
    assert "DSML" in tool and 'invoke name="read_file"' in tool
    assert all(it["text"].startswith("<｜begin") for it in parts["A"] if it["kind"] != "wikitext")


@pytest.mark.skipif(not REAL_ENCODING, reason="V41_ENCODING_PY is not set")
def test_main_writes_halves_and_manifest(tmp_path):
    tar = tmp_path / "repo.tar"
    write_tar(tar)
    wiki = tmp_path / "w.jsonl"
    wiki.write_text("".join(json.dumps({"text": line}) + "\n" for _, text in make_articles()
                            for line in text.splitlines(keepends=True)), encoding="utf-8")
    out = tmp_path / "corpus"
    argv = ["--out-dir", str(out), "--encoding-py", REAL_ENCODING, "--wikitext", str(wiki),
            "--source-tar", str(tar), "--rev", "synthetic", "--tokens-per-half", "6000", "--long-tokens", "1500",
            "--file-chars", "600"]
    assert bc.main(argv) == 0
    first = {n: (out / n).read_bytes() for n in ("half_a.txt", "half_b.txt", "manifest.json")}
    m = json.loads(first["manifest.json"])
    assert m["tokens_exact"] is False and set(m["halves"]) == {"A", "B"}
    assert m["halves"]["A"]["by_kind"]["long"]["items"] == 1
    assert bc.main(argv) == 0
    assert first == {n: (out / n).read_bytes() for n in first}
