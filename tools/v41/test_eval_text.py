"""Tests of eval_text.py on synthetic rows, expected values worked out by hand.

    python -m pytest tools/v41/test_eval_text.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_text as et  # noqa: E402

# a paragraph of 81 characters once squashed, and one of exactly 80
P81 = "w" * 40 + " " + "x" * 40
P80 = "y" * 39 + " " + "z" * 40
SHARED = "The shared paragraph runs well beyond the limit of eighty characters, so any copy of it must be caught."
OWN = "A paragraph that only the held-out text holds, long enough to be checked against the corpus halves."


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps({"text": r}) + "\n" for r in rows), encoding="utf-8")
    return path


def run(tmp_path, rows, half_a, half_b, extra=()):
    write_rows(tmp_path / "valid.jsonl", rows)
    (tmp_path / "a.txt").write_text(half_a, encoding="utf-8")
    (tmp_path / "b.txt").write_text(half_b, encoding="utf-8")
    code = et.main(["--out-dir", str(tmp_path / "out"), "--rows", str(tmp_path / "valid.jsonl"),
                    "--half-a", str(tmp_path / "a.txt"), "--half-b", str(tmp_path / "b.txt"), *extra])
    return code, tmp_path / "out"


def test_the_limit_is_more_than_80_characters_once_whitespace_is_collapsed():
    assert len(P81) == 81 and len(P80) == 80
    assert et.long_paragraphs([P81, P80, " = Title = \n", ""]) == [P81]
    # a row wrapped over lines and padded counts for its collapsed length
    assert et.long_paragraphs(["w" * 40 + "\n\n   " + "x" * 40 + "\n"]) == [P81]
    # duplicates are checked once
    assert et.long_paragraphs([P81, P81]) == [P81]


def test_a_paragraph_found_in_a_half_is_an_overlap_whatever_its_wrapping():
    halves = {"half_a.txt": "intro\n" + SHARED.replace(", so", ",\n   so") + "\nend", "half_b.txt": "nothing here"}
    assert et.find_overlaps([et.squash(SHARED), et.squash(OWN)], halves) == [(et.squash(SHARED), "half_a.txt")]


def test_no_overlap_writes_the_text_and_its_sha256(tmp_path, capsys):
    rows = ["", " = Valid article = \n", "", OWN + "\n", "short\n"]
    code, out = run(tmp_path, rows, half_a="corpus a " + P81, half_b="corpus b")
    assert code == 0
    text = (out / "wiki_valid.txt").read_bytes()
    assert text == b"\n = Valid article = \n\n" + OWN.encode() + b"\nshort\n"
    report = json.loads((out / "eval_text.json").read_text())
    assert report["sha256"] == hashlib.sha256(text).hexdigest()
    assert report["bytes"] == len(text) and report["overlaps"] == 0 and report["paragraphs_checked"] == 1
    assert report["rows"] == 5
    assert report["sha256"] in capsys.readouterr().out


def test_an_overlap_stops_without_writing_the_text(tmp_path, capsys):
    code, out = run(tmp_path, ["", OWN + "\n", SHARED + "\n"], half_a="x", half_b="y " + SHARED + " z")
    assert code == 2
    assert not (out / "wiki_valid.txt").exists()
    report = json.loads((out / "eval_text.json").read_text())
    assert report["overlaps"] == 1 and report["overlap_examples"][0]["in"] == "half_b.txt"
    assert "1 validation paragraphs appear in the corpus" in capsys.readouterr().err


def test_without_a_control_the_report_says_so_and_states_the_length_limit(tmp_path, capsys):
    code, out = run(tmp_path, [OWN + "\n"], half_a="a", half_b="b")
    assert code == 0
    report = json.loads((out / "eval_text.json").read_text())
    assert report["control"] is None
    assert "more than 80 characters" in report["limit"]
    assert "no --control-rows" in capsys.readouterr().err


def test_a_control_that_finds_nothing_means_the_check_is_blind(tmp_path, capsys):
    write_rows(tmp_path / "test.jsonl", [OWN + "\n"])
    code, out = run(tmp_path, [SHARED + "\n"], half_a="a", half_b="b", extra=["--control-rows", str(tmp_path / "test.jsonl")])
    assert code == 2
    assert not (out / "wiki_valid.txt").exists()
    assert "the check is blind" in capsys.readouterr().err


def test_a_control_that_finds_the_test_paragraph_lets_the_check_stand(tmp_path):
    write_rows(tmp_path / "test.jsonl", [OWN + "\n", SHARED + "\n"])
    code, out = run(tmp_path, [P81 + "\n"], half_a="a " + OWN, half_b="b",
                    extra=["--control-rows", str(tmp_path / "test.jsonl")])
    assert code == 0
    assert json.loads((out / "eval_text.json").read_text())["control"] == {
        "split": "test", "paragraphs_checked": 2, "overlaps": 1}


def test_chunks_follow_the_perplexity_tool_with_a_bos_in_front():
    # n text tokens + 1 BOS, cut in whole windows of 2048
    assert et.chunk_count(4094) == 1
    assert et.chunk_count(4095) == 2
    assert et.chunk_count(45 * 2048 - 1) == 45
    assert et.chunk_count(45 * 2048 - 2) == 44
    assert et.chunk_count(0) == 0
    assert et.chunk_count(2999, n_ctx=1000) == 3


def test_the_tokens_and_chunks_are_reported_against_the_chunks_wanted(tmp_path, monkeypatch):
    class WordCounter:
        """One token per word."""
        def __init__(self, path):
            assert path == "tok.json"

        def count(self, text):
            return len(text.split())

    monkeypatch.setattr(et.build_corpus, "Counter", WordCounter)
    rows = ["a b c d e f g h i j\n"] * 3          # 30 words, 31 tokens with the BOS: 3 windows of 10
    for chunks, enough in ((3, True), (4, False)):
        code, out = run(tmp_path, rows, half_a="a", half_b="b",
                        extra=["--tokenizer", "tok.json", "--n-ctx", "10", "--chunks", str(chunks)])
        assert code == 0
        assert json.loads((out / "eval_text.json").read_text())["tokens"] == {
            "count": 30, "n_ctx": 10, "chunks": 3, "chunks_wanted": chunks, "enough": enough}


def test_without_a_tokenizer_there_is_no_token_count(tmp_path):
    code, out = run(tmp_path, [OWN + "\n"], half_a="a", half_b="b")
    assert code == 0
    assert "tokens" not in json.loads((out / "eval_text.json").read_text())


@pytest.mark.parametrize("name", ["half_a.txt", "half_b.txt"])
def test_each_half_is_checked(tmp_path, name):
    halves = {"half_a.txt": "a", "half_b.txt": "b"}
    halves[name] = SHARED
    code, _ = run(tmp_path, [SHARED + "\n"], half_a=halves["half_a.txt"], half_b=halves["half_b.txt"])
    assert code == 2
