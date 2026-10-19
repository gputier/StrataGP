"""Tests of expert_stats.py on synthetic routing files.

    python -m pytest tools/v41/test_expert_stats.py
"""
import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert_stats as es  # noqa: E402

N_EXPERT, N_USED, TOP = 16, 6, 4      # a 16-expert toy: the top 25 % are 4 experts


def write_rtr1(path, records, n_used=N_USED, magic=b"RTR1", cut=0):
    """An RTR1 file: header, then the (layer, ids...) records as uint16; `cut` drops that many trailing bytes."""
    body = b"".join(struct.pack(f"<{1 + n_used}H", *rec) for rec in records)
    Path(path).write_bytes(magic + struct.pack("<i", n_used) + body[:len(body) - cut])
    return path


def hot_records(hot, n_layers=3, tokens=96):
    """Every token calls the 4 `hot` experts plus 2 cold ones turning over the 12 others (16 per cold expert at
    96 tokens): hot experts carry 4/6 of the calls."""
    cold = [e for e in range(N_EXPERT) if e not in hot]
    recs = []
    for t in range(tokens):
        for layer in range(n_layers):
            recs.append((layer, *hot, cold[(2 * t) % 12], cold[(2 * t + 1) % 12]))
    return recs


def uniform_records(n_layers=3, tokens=64):
    """Token t calls experts 6t..6t+5 modulo 16: 384 calls on 16 experts, 24 each."""
    return [(layer, *[(6 * t + j) % N_EXPERT for j in range(N_USED)]) for t in range(tokens)
            for layer in range(n_layers)]


def run(tmp_path, a_records, b_records, extra=()):
    a = write_rtr1(tmp_path / "a.rtr", a_records)
    b = write_rtr1(tmp_path / "b.rtr", b_records)
    out = tmp_path / "stats.json"
    code = es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT), "--out", str(out),
                    *extra])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_uniform_routing_has_no_concentration(tmp_path):
    code, r = run(tmp_path, uniform_records(), uniform_records())
    assert code == 0
    assert r["n_layers"] == 3 and r["top_k"] == TOP and r["n_used"] == N_USED
    assert r["tokens"] == {"A": 64, "B": 64}
    assert r["counts"]["A"][0] == [24] * N_EXPERT
    assert r["concentration"]["A"]["per_layer"] == pytest.approx([0.25] * 3)
    assert r["concentration"]["A"]["mean"] == pytest.approx(0.25)
    assert not r["verdicts"]["concentration_A"]["pass"]
    assert not r["verdicts"]["all_pass"]


def test_concentrated_routing_and_identical_rankings(tmp_path):
    code, r = run(tmp_path, hot_records([0, 1, 2, 3]), hot_records([0, 1, 2, 3]))
    assert code == 0
    assert r["counts"]["A"][1][:4] == [96] * 4
    assert r["counts"]["A"][1][4:] == [16] * 12
    assert r["counts"]["total"][2][0] == 192
    assert r["concentration"]["A"]["mean"] == pytest.approx(2 / 3)
    assert r["concentration"]["B"]["per_layer"] == pytest.approx([2 / 3] * 3)
    assert r["stability"]["mean"] == pytest.approx(1.0)
    v = r["verdicts"]
    assert v["concentration_A"]["pass"] and v["concentration_B"]["pass"] and v["stability"]["pass"]
    assert v["all_pass"]
    assert v["concentration_A"]["threshold"] == 0.50 and v["stability"]["threshold"] == 0.70


def test_disjoint_rankings_have_zero_stability(tmp_path):
    code, r = run(tmp_path, hot_records([0, 1, 2, 3]), hot_records([8, 9, 10, 11]))
    assert r["stability"]["per_layer"] == [0.0] * 3
    assert r["concentration"]["A"]["mean"] > 0.5 and r["concentration"]["B"]["mean"] > 0.5
    assert not r["verdicts"]["stability"]["pass"]
    assert not r["verdicts"]["all_pass"]


def test_half_overlap_gives_half_stability(tmp_path):
    code, r = run(tmp_path, hot_records([0, 1, 2, 3]), hot_records([2, 3, 8, 9]))
    assert r["stability"]["mean"] == pytest.approx(0.5)


def test_several_files_per_half_concatenate(tmp_path):
    recs = hot_records([0, 1, 2, 3])
    mid = len(recs) // 2
    a1 = write_rtr1(tmp_path / "a1.rtr", recs[:mid])
    a2 = write_rtr1(tmp_path / "a2.rtr", recs[mid:])
    b = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    out = tmp_path / "s.json"
    assert es.main(["--half-a", str(a1), str(a2), "--half-b", str(b), "--n-expert", str(N_EXPERT),
                    "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    _, single = run(tmp_path, recs, hot_records([0, 1, 2, 3]))
    assert r["counts"] == single["counts"] and r["tokens"] == {"A": 96, "B": 96}


def test_layer_count_is_read_from_the_data(tmp_path):
    _, r = run(tmp_path, hot_records([0, 1, 2, 3], n_layers=5), hot_records([0, 1, 2, 3], n_layers=5))
    assert r["n_layers"] == 5 and len(r["counts"]["total"]) == 5


@pytest.mark.parametrize("magic", [b"RTR2", b"XXXX", b"\0\0\0\0"])
def test_wrong_magic_is_refused(tmp_path, magic, capsys):
    good = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    bad = write_rtr1(tmp_path / "a.rtr", hot_records([0, 1, 2, 3]), magic=magic)
    assert es.main(["--half-a", str(bad), "--half-b", str(good), "--n-expert", str(N_EXPERT)]) == 2
    assert "bad magic" in capsys.readouterr().err


def test_file_shorter_than_header_is_refused(tmp_path, capsys):
    short = tmp_path / "a.rtr"
    short.write_bytes(b"RTR1\x06")
    good = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(short), "--half-b", str(good), "--n-expert", str(N_EXPERT)]) == 2
    assert "header" in capsys.readouterr().err


def test_truncated_file_is_refused(tmp_path, capsys):
    bad = write_rtr1(tmp_path / "a.rtr", hot_records([0, 1, 2, 3]), cut=5)
    good = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(bad), "--half-b", str(good), "--n-expert", str(N_EXPERT)]) == 2
    assert "multiple" in capsys.readouterr().err


def test_n_used_must_match_between_files(tmp_path, capsys):
    a1 = write_rtr1(tmp_path / "a1.rtr", hot_records([0, 1, 2, 3]))
    a2 = write_rtr1(tmp_path / "a2.rtr", [(0, 0, 1, 2, 3)], n_used=4)
    b = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(a1), str(a2), "--half-b", str(b), "--n-expert", str(N_EXPERT)]) == 2
    assert "n_used" in capsys.readouterr().err


def test_n_used_must_match_between_halves(tmp_path, capsys):
    a = write_rtr1(tmp_path / "a.rtr", [(0, 0, 1, 2, 3), (0, 0, 1, 2, 4)], n_used=4)
    b = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3], n_layers=1))
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT)]) == 2
    assert "n_used" in capsys.readouterr().err


def test_expert_id_out_of_range_is_refused(tmp_path, capsys):
    recs = hot_records([0, 1, 2, 3])
    recs[7] = (recs[7][0], 0, 1, 2, 3, 4, N_EXPERT)
    a = write_rtr1(tmp_path / "a.rtr", recs)
    b = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT)]) == 2
    assert "expert id 16" in capsys.readouterr().err


def test_layer_out_of_range_is_refused(tmp_path, capsys):
    a = write_rtr1(tmp_path / "a.rtr", hot_records([0, 1, 2, 3]))
    b = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT), "--n-layer", "2"]) == 2
    assert "layer 2 is out of range" in capsys.readouterr().err


def test_layer_gap_is_refused(tmp_path, capsys):
    recs = [r for r in hot_records([0, 1, 2, 3]) if r[0] != 1]
    a = write_rtr1(tmp_path / "a.rtr", recs)
    b = write_rtr1(tmp_path / "b.rtr", recs)
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT)]) == 2
    assert "not contiguous" in capsys.readouterr().err


def test_ragged_layers_are_refused_unless_trimmed(tmp_path, capsys):
    recs = hot_records([0, 1, 2, 3])
    a = write_rtr1(tmp_path / "a.rtr", recs[:-2])          # the last token reached layer 0 only
    b = write_rtr1(tmp_path / "b.rtr", recs)
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT)]) == 2
    assert "--trim-ragged" in capsys.readouterr().err
    out = tmp_path / "s.json"
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--n-expert", str(N_EXPERT), "--trim-ragged",
                    "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["tokens"]["A"] == 95
    assert all(sum(row) == 95 * N_USED for row in r["counts"]["A"])


# ---- the STRP profile

def read_strp_like_the_engine(path, n_layers, n_expert):
    """A byte-level reader that follows src/core/expert_cache.cpp read_expert_profile and write_expert_profile."""
    blob = Path(path).read_bytes()
    assert blob[:4] == b"STRP"
    version, nl, ne, want, n_ranked = struct.unpack_from("<5I", blob, 4)
    assert version == 1 and (nl, ne) == (n_layers, n_expert) and n_ranked <= want
    assert len(blob) == 24 + 4 * n_ranked + 4 * nl * ne
    pairs = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n_ranked)]
    assert all(0 <= l < nl and 0 <= e < ne for l, e in pairs)
    table = struct.unpack_from(f"<{nl * ne}i", blob, 24 + 4 * n_ranked)
    for rank, (l, e) in enumerate(pairs):
        assert table[l * ne + e] == rank
    assert sorted(t for t in table if t >= 0) == list(range(n_ranked))
    return pairs, table, want


def test_profile_follows_the_engine_format(tmp_path):
    prof = tmp_path / "expert-profile.bin"
    code, r = run(tmp_path, hot_records([0, 1, 2, 3], n_layers=5), hot_records([0, 1, 2, 3], n_layers=5),
                  extra=["--profile", str(prof)])
    assert code == 0
    pairs, table, want = read_strp_like_the_engine(prof, 5, N_EXPERT)
    assert want == len(pairs) == 5 * N_EXPERT
    total = r["counts"]["total"]
    assert all(total[l][e] >= total[l2][e2] for (l, e), (l2, e2) in zip(pairs, pairs[1:]))
    assert {e for _, e in pairs[:5 * 4]} == {0, 1, 2, 3}          # the hot experts of every layer lead
    assert pairs[:5] == [(l, 0) for l in range(5)]                 # a tie goes expert by expert across layers


def test_profile_ranks_the_total_of_both_halves(tmp_path):
    prof = tmp_path / "p.bin"
    run(tmp_path, hot_records([0, 1, 2, 3], n_layers=1), hot_records([2, 3, 8, 9], n_layers=1),
        extra=["--profile", str(prof)])
    pairs, _, _ = read_strp_like_the_engine(prof, 1, N_EXPERT)
    assert [e for _, e in pairs[:2]] == [2, 3]                      # called by both halves: 192 calls each


def test_profile_bytes_match_make_profile(tmp_path):
    mp_dir = HERE.parent
    if not (mp_dir / "make_profile.py").exists():
        pytest.skip("tools/make_profile.py is not next to tools/v41")
    sys.path.insert(0, str(mp_dir))
    import make_profile
    n_layers = make_profile.N_LAYER
    total = [[(l * 7 + e * 3) % 11 for e in range(N_EXPERT)] for l in range(n_layers)]
    ranked = es.rank_pairs(total)
    ours, theirs = tmp_path / "ours.bin", tmp_path / "theirs.bin"
    es.write_profile(ours, ranked, n_layers, N_EXPERT)
    make_profile.write_profile(theirs, ranked, N_EXPERT)
    assert ours.read_bytes() == theirs.read_bytes()
    assert make_profile.read_profile(ours, N_EXPERT) == [tuple(p) for p in ranked]


def test_default_expert_count_is_384_and_the_top_quarter_is_96(tmp_path):
    recs = [(layer, *[(7 * t + j) % 384 for j in range(N_USED)]) for t in range(80) for layer in range(2)]
    a = write_rtr1(tmp_path / "a.rtr", recs)
    b = write_rtr1(tmp_path / "b.rtr", recs)
    out = tmp_path / "s.json"
    assert es.main(["--half-a", str(a), "--half-b", str(b), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["n_expert"] == 384 and r["top_k"] == 96 and len(r["counts"]["A"][0]) == 384


@pytest.mark.parametrize("n_used", [0, 65])
def test_n_used_outside_one_to_sixty_four_is_refused(tmp_path, capsys, n_used):
    bad = write_rtr1(tmp_path / "a.rtr", [], n_used=n_used)
    good = write_rtr1(tmp_path / "b.rtr", hot_records([0, 1, 2, 3]))
    assert es.main(["--half-a", str(bad), "--half-b", str(good), "--n-expert", str(N_EXPERT)]) == 2
    assert f"n_used {n_used} is out of range" in capsys.readouterr().err


@pytest.mark.parametrize("n_used", [1, 64])
def test_n_used_bounds_are_accepted(tmp_path, n_used):
    assert es.read_header(write_rtr1(tmp_path / "a.rtr", [], n_used=n_used)) == n_used


def test_top_set_breaks_ties_on_the_lower_id():
    # 5 leads, then four experts tied at 3: with k = 3 the two lowest ids of the tie (1 and 2) join it
    assert es.top_set([5, 3, 3, 3, 3, 1], 3) == {0, 1, 2}
    # all tied: the k lowest ids
    assert es.top_set([2, 2, 2, 2], 2) == {0, 1}
    # a higher count beats a lower id
    assert es.top_set([1, 1, 4], 1) == {2}
