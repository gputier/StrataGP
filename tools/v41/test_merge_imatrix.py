"""Tests of merge_imatrix.py on synthetic matrices, expected values worked out by hand.

    python -m pytest tools/v41/test_merge_imatrix.py

Toy model: 2 layers, 3 experts, 2 values per expert slice.  Both halves ran 10 calls of every expert tensor, so
the stored values (mean * ncall) are the means times 10.
"""
import json
import struct
import sys
from array import array
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import merge_imatrix as mi  # noqa: E402

N_EXPERT, WIDTH = 3, 2


def write_dat(path, entries, last_call=10, dataset="half.txt"):
    """A matrix file as save_imatrix writes it: entries [(name, ncall, [stored values])]."""
    out = struct.pack("<i", len(entries))
    for name, ncall, values in entries:
        raw = name.encode()
        out += struct.pack("<i", len(raw)) + raw + struct.pack("<ii", ncall, len(values))
        out += array("f", values).tobytes()
    raw = dataset.encode()
    out += struct.pack("<i", last_call) + struct.pack("<i", len(raw)) + raw
    Path(path).write_bytes(out)
    return path


def load_like_quantize(path):
    """The reader of examples/quantize/quantize.cpp load_imatrix: {name: values / ncall}, and the trailer."""
    data = Path(path).read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    (n,) = take("i")
    entries = {}
    for _ in range(n):
        (length,) = take("i")
        name = data[pos:pos + length].decode()
        pos += length
        ncall, nval = take("ii")
        values = list(struct.unpack_from(f"<{nval}f", data, pos))
        pos += 4 * nval
        entries[name] = [v / ncall for v in values] if ncall > 0 else values
    (last_call,) = take("i")
    (length,) = take("i")
    dataset = data[pos:pos + length].decode()
    assert pos + length == len(data)
    return entries, last_call, dataset


def write_stats(path, a, b):
    """stats.json with the calls of every expert, a and b as [layer][expert]."""
    Path(path).write_text(json.dumps({"n_layers": 2, "n_expert": N_EXPERT, "n_used": 1, "tokens": {"A": 1, "B": 1},
                                      "counts": {"A": a, "B": b, "total": a}}))
    return path


# calls of expert 0, 1, 2 of layer 0 and of layer 1, in half A and in half B
CALLS_A = [[3, 2, 0], [0, 4, 0]]
CALLS_B = [[1, 0, 0], [5, 0, 0]]
FILLER = [10.0, 10.0]      # a filler slice: mean 1 times ncall 10


def expert_entry(layer, slices):
    return (f"blk.{layer}.ffn_up_exps.weight", 10, [v for s in slices for v in s])


@pytest.fixture
def files(tmp_path):
    a = [expert_entry(0, [[2.0, 4.0], [5.0, 15.0], FILLER]),         # means .2 .4 | .5 1.5 | 1 1
         expert_entry(1, [FILLER, [3.0, 3.0], FILLER]),
         ("blk.0.attn_q_a.weight", 4, [8.0, 12.0, 16.0])]            # means 2 3 4
    b = [expert_entry(0, [[4.0, 8.0], FILLER, FILLER]),              # means .4 .8 | 1 1 | 1 1
         expert_entry(1, [[6.0, 2.0], FILLER, FILLER]),
         ("blk.0.attn_q_a.weight", 6, [6.0, 12.0, 18.0])]            # means 1 2 3
    return {"a": write_dat(tmp_path / "a.dat", a, 10, "A.txt"), "b": write_dat(tmp_path / "b.dat", b, 10, "B.txt"),
            "stats": write_stats(tmp_path / "stats.json", CALLS_A, CALLS_B), "out": tmp_path / "ab.dat"}


def merged(files):
    mi.merge(files["a"], files["b"], files["stats"], files["out"])
    return load_like_quantize(files["out"])


def test_an_expert_seen_in_both_halves_is_weighted_by_its_calls(files):
    # layer 0 expert 0: 3 calls in A (means .2 .4), 1 in B (.4 .8): (3 * .2 + .4) / 4 = .25, (3 * .4 + .8) / 4 = .5
    entries, _, _ = merged(files)
    assert entries["blk.0.ffn_up_exps.weight"][0:2] == pytest.approx([0.25, 0.5])


def test_an_expert_seen_in_one_half_keeps_that_halfs_mean(files):
    entries, _, _ = merged(files)
    # layer 0 expert 1: 2 calls in A only, the filler of B has no weight: means .5 and 1.5
    assert entries["blk.0.ffn_up_exps.weight"][2:4] == pytest.approx([0.5, 1.5])
    # layer 1 expert 0: 5 calls in B only (A holds the filler), means .6 and .2
    assert entries["blk.1.ffn_up_exps.weight"][0:2] == pytest.approx([0.6, 0.2])
    # layer 1 expert 1: 4 calls in A only, means .3 and .3
    assert entries["blk.1.ffn_up_exps.weight"][2:4] == pytest.approx([0.3, 0.3])


def test_an_expert_seen_in_neither_half_gets_ones(files):
    entries, _, _ = merged(files)
    assert entries["blk.0.ffn_up_exps.weight"][4:6] == [1.0, 1.0]
    assert entries["blk.1.ffn_up_exps.weight"][4:6] == [1.0, 1.0]


def test_a_dense_tensor_is_weighted_by_its_calls(files):
    # ncall 4 (means 2 3 4) and 6 (means 1 2 3): (4 * 2 + 6 * 1) / 10 = 1.4, 2.4, 3.4
    entries, _, _ = merged(files)
    assert entries["blk.0.attn_q_a.weight"] == pytest.approx([1.4, 2.4, 3.4])


def test_the_stored_ncall_is_the_sum_and_the_trailer_is_merged(files):
    summary = mi.merge(files["a"], files["b"], files["stats"], files["out"])
    entries, trailer = mi.index(files["out"])
    assert {e.name: e.ncall for e in entries} == {"blk.0.ffn_up_exps.weight": 20, "blk.1.ffn_up_exps.weight": 20,
                                                  "blk.0.attn_q_a.weight": 10}
    assert trailer == {"last_call": 20, "dataset": "A.txt+B.txt"}
    # expert 0 of layer 0 is stored as the merged mean times the output ncall: .25 * 20 = 5, .5 * 20 = 10
    with open(files["out"], "rb") as f:
        assert list(mi.read_values(f, entries[0])[0:2]) == [5.0, 10.0]
    # the experts by kind in CALLS_A and CALLS_B: one called in both halves, three in one, two in neither
    assert summary == {"entries": 3, "expert_entries": 2, "dense_entries": 1,
                       "experts_both": 1, "experts_one_half": 3, "experts_neither": 2}


def test_the_output_is_read_by_the_loader_of_the_quantizer_in_the_same_order(files):
    entries, last_call, dataset = merged(files)
    assert list(entries) == ["blk.0.ffn_up_exps.weight", "blk.1.ffn_up_exps.weight", "blk.0.attn_q_a.weight"]
    assert (last_call, dataset) == (20, "A.txt+B.txt")
    assert [len(v) for v in entries.values()] == [6, 6, 3]


def test_entries_in_another_order_are_matched_by_name(files, tmp_path):
    a = mi.index(files["a"])[0]
    with open(files["a"], "rb") as f:
        reordered = [(e.name, e.ncall, list(mi.read_values(f, e))) for e in reversed(a)]
    write_dat(files["a"], reordered, 10, "A.txt")
    entries, _, _ = merged(files)
    assert entries["blk.0.ffn_up_exps.weight"][0:2] == pytest.approx([0.25, 0.5])
    assert list(entries)[0] == "blk.0.attn_q_a.weight"      # the order of A


def test_stats_that_do_not_belong_to_the_matrix_are_refused(files, tmp_path):
    # the stats say expert 1 of layer 0 had no call in A, while its slice in A holds real values
    write_stats(files["stats"], [[3, 0, 0], [0, 4, 0]], CALLS_B)
    with pytest.raises(mi.ImatrixError, match="expert 1 has no call in half A"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_the_two_matrices_must_hold_the_same_entries(files, tmp_path):
    write_dat(files["b"], [expert_entry(0, [[4.0, 8.0], FILLER, FILLER])], 10)
    with pytest.raises(mi.ImatrixError, match="same entries"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_an_entry_of_another_size_is_refused(files):
    b = mi.index(files["b"])[0]
    with open(files["b"], "rb") as f:
        entries = [(e.name, e.ncall, list(mi.read_values(f, e))) for e in b]
    entries[2] = (entries[2][0], entries[2][1], [6.0, 12.0])
    write_dat(files["b"], entries)
    with pytest.raises(mi.ImatrixError, match="3 values in A, 2 in B"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_an_expert_tensor_that_does_not_split_into_experts_is_refused(files):
    write_dat(files["a"], [("blk.0.ffn_up_exps.weight", 10, [1.0] * 7)])
    write_dat(files["b"], [("blk.0.ffn_up_exps.weight", 10, [1.0] * 7)])
    with pytest.raises(mi.ImatrixError, match="7 values for 3 experts"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_a_layer_beyond_the_stats_is_refused(files):
    write_dat(files["a"], [expert_entry(5, [FILLER] * 3)])
    write_dat(files["b"], [expert_entry(5, [FILLER] * 3)])
    with pytest.raises(mi.ImatrixError, match="layer 5 of 2"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_a_zero_ncall_is_refused(files):
    write_dat(files["a"], [("blk.0.attn_q_a.weight", 0, [1.0])])
    write_dat(files["b"], [("blk.0.attn_q_a.weight", 6, [1.0])])
    with pytest.raises(mi.ImatrixError, match="ncall 0"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_damaged_files_are_refused(files, tmp_path):
    good = files["a"].read_bytes()
    for name, data, match in (("empty", b"", "file ends"),
                              ("no entries", struct.pack("<i", 0), "0 entries"),
                              ("cut in the values", good[:60], "file ends"),
                              ("extra bytes", good + b"x", "bytes after the trailer")):
        path = tmp_path / f"{name}.dat"
        path.write_bytes(data)
        with pytest.raises(mi.ImatrixError, match=match):
            mi.index(path)


def test_stats_with_the_wrong_shape_are_refused(files):
    write_stats(files["stats"], [[1, 2], [3, 4]], CALLS_B)
    with pytest.raises(mi.ImatrixError, match="not 2 x 3"):
        mi.merge(files["a"], files["b"], files["stats"], files["out"])


def test_the_hand_check_recomputes_one_expert_of_each_kind(files):
    mi.merge(files["a"], files["b"], files["stats"], files["out"])
    found = {s["kind"]: s for s in mi.spot_check(files["a"], files["b"], files["out"], files["stats"])}
    assert found["both"]["expert"] == 0 and (found["both"]["calls_a"], found["both"]["calls_b"]) == (3, 1)
    assert found["both"]["expected_mean"][:2] == pytest.approx([0.25, 0.5])
    assert found["one half"]["expert"] == 1 and found["one half"]["calls_b"] == 0
    assert found["neither"]["expected_mean"] == [1.0] * 2
    assert all(s["max_abs_error"] < 1e-6 for s in found.values())


def test_an_output_that_is_an_input_is_refused_and_the_input_is_left_intact(files, tmp_path, capsys):
    before = {k: files[k].read_bytes() for k in ("a", "b")}
    link = tmp_path / "link.dat"
    link.symlink_to(files["b"])
    for out in (files["a"], files["b"], link):
        assert mi.main(["--a", str(files["a"]), "--b", str(files["b"]), "--stats", str(files["stats"]),
                        "--out", str(out)]) == 2
        assert "is the input" in capsys.readouterr().err
    assert {k: files[k].read_bytes() for k in ("a", "b")} == before


def test_main_writes_the_report_and_returns_zero(files, tmp_path, capsys):
    report = tmp_path / "report.json"
    assert mi.main(["--a", str(files["a"]), "--b", str(files["b"]), "--stats", str(files["stats"]),
                    "--out", str(files["out"]), "--report", str(report)]) == 0
    doc = json.loads(report.read_text())
    assert doc["reread"]["entries"] == 3 and doc["reread"]["sizes_match_inputs"] is True
    assert "experts_neither" in capsys.readouterr().out
    # a refused input is an exit code 2 and a message, not a traceback
    write_stats(files["stats"], [[3, 0, 0], [0, 4, 0]], CALLS_B)
    assert mi.main(["--a", str(files["a"]), "--b", str(files["b"]), "--stats", str(files["stats"]),
                    "--out", str(files["out"])]) == 2
