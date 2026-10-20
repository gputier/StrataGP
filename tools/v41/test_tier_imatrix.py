"""Tests of tier_imatrix.py on synthetic matrices, expected values worked out by hand.

    python -m pytest tools/v41/test_tier_imatrix.py

Toy model: 2 layers, 4 experts; the up slice of an expert is 2 values wide and the down slice 3.  The value j of
expert e in layer L is L * 1000 + e * 10 + j (up) and 500 + L * 1000 + e * 10 + j (down), so a value says where it
came from.  Layer 0 tiers: [1, 3], [0], [2]; layer 1 tiers: [3], [0, 2], [1].
"""
import struct
import sys
from array import array
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import merge_imatrix as mi  # noqa: E402
import tier_imatrix as ti  # noqa: E402
import tier_table as tt  # noqa: E402

N_EXPERT = 4
UP_WIDTH, DOWN_WIDTH = 2, 3
TABLE = {"mode": "p3", "n_expert": N_EXPERT, "n_layer": 2, "tier_types": tt.TIER_TYPES,
         "layers": [[[1, 3], [0], [2]], [[3], [0, 2], [1]]]}


def up_value(layer, e, j):
    return float(layer * 1000 + e * 10 + j)


def down_value(layer, e, j):
    return float(500 + layer * 1000 + e * 10 + j)


def expert_values(value, layer, width):
    return [value(layer, e, j) for e in range(N_EXPERT) for j in range(width)]


def write_dat(path, entries, last_call=7, dataset="ab.txt"):
    out = struct.pack("<i", len(entries))
    for name, ncall, values in entries:
        raw = name.encode()
        out += struct.pack("<i", len(raw)) + raw + struct.pack("<ii", ncall, len(values))
        out += array("f", values).tobytes()
    raw = dataset.encode()
    out += struct.pack("<i", last_call) + struct.pack("<i", len(raw)) + raw
    Path(path).write_bytes(out)
    return path


def merged_matrix(path):
    return write_dat(path, [
        ("blk.0.ffn_up_exps.weight", 10, expert_values(up_value, 0, UP_WIDTH)),
        ("blk.0.ffn_down_exps.weight", 10, expert_values(down_value, 0, DOWN_WIDTH)),
        ("output.weight", 3, [5.0, 6.0]),
        ("blk.1.ffn_up_exps.weight", 20, expert_values(up_value, 1, UP_WIDTH)),
        ("blk.1.ffn_down_exps.weight", 20, expert_values(down_value, 1, DOWN_WIDTH)),
    ])


def read_all(path):
    """([(name, ncall, values)] in file order, trailer)."""
    entries, trailer = mi.index(path)
    with open(path, "rb") as f:
        return [(e.name, e.ncall, list(mi.read_values(f, e))) for e in entries], trailer


@pytest.fixture
def split(tmp_path):
    out = tmp_path / "tiers.dat"
    summary = ti.split_matrix(merged_matrix(tmp_path / "ab.dat"), TABLE, out)
    return read_all(out), summary


def test_every_expert_entry_becomes_three_in_place(split):
    (entries, _), summary = split
    assert [n for n, _, _ in entries] == [
        "blk.0.t0.ffn_up_exps.weight", "blk.0.t1.ffn_up_exps.weight", "blk.0.t2.ffn_up_exps.weight",
        "blk.0.t0.ffn_down_exps.weight", "blk.0.t1.ffn_down_exps.weight", "blk.0.t2.ffn_down_exps.weight",
        "output.weight",
        "blk.1.t0.ffn_up_exps.weight", "blk.1.t1.ffn_up_exps.weight", "blk.1.t2.ffn_up_exps.weight",
        "blk.1.t0.ffn_down_exps.weight", "blk.1.t1.ffn_down_exps.weight", "blk.1.t2.ffn_down_exps.weight",
    ]
    assert summary["entries_in"] == 5 and summary["entries_out"] == 13
    assert len(summary["tier_entries"]) == 12


def test_up_slices_follow_the_positions_of_the_tier(split):
    (entries, _), _ = split
    got = {n: v for n, _, v in entries}
    # layer 0, tier 0 = experts 1 then 3: slices [2, 4) then [6, 8) of the original
    assert got["blk.0.t0.ffn_up_exps.weight"] == [10.0, 11.0, 30.0, 31.0]
    assert got["blk.0.t1.ffn_up_exps.weight"] == [0.0, 1.0]
    assert got["blk.0.t2.ffn_up_exps.weight"] == [20.0, 21.0]
    assert got["blk.1.t0.ffn_up_exps.weight"] == [1030.0, 1031.0]
    assert got["blk.1.t1.ffn_up_exps.weight"] == [1000.0, 1001.0, 1020.0, 1021.0]
    assert got["blk.1.t2.ffn_up_exps.weight"] == [1010.0, 1011.0]


def test_down_slices_use_the_down_width(split):
    (entries, _), _ = split
    got = {n: v for n, _, v in entries}
    assert got["blk.0.t0.ffn_down_exps.weight"] == [510.0, 511.0, 512.0, 530.0, 531.0, 532.0]
    assert got["blk.0.t1.ffn_down_exps.weight"] == [500.0, 501.0, 502.0]
    assert got["blk.1.t1.ffn_down_exps.weight"] == [1500.0, 1501.0, 1502.0, 1520.0, 1521.0, 1522.0]


def test_sizes_are_width_times_experts_of_the_tier(split):
    _, summary = split
    sizes = {name: (nval, n, width) for name, nval, n, width in summary["tier_entries"]}
    assert sizes["blk.0.t0.ffn_up_exps.weight"] == (4, 2, 2)
    assert sizes["blk.1.t1.ffn_down_exps.weight"] == (6, 2, 3)
    assert sizes["blk.1.t2.ffn_down_exps.weight"] == (3, 1, 3)
    for layer in (0, 1):
        assert sum(sizes[f"blk.{layer}.t{k}.ffn_up_exps.weight"][1] for k in range(3)) == N_EXPERT


def test_ncall_other_entries_and_trailer_are_copied(split):
    (entries, trailer), _ = split
    ncall = {n: c for n, c, _ in entries}
    assert all(ncall[f"blk.0.t{k}.ffn_{f}_exps.weight"] == 10 for k in range(3) for f in ("up", "down"))
    assert all(ncall[f"blk.1.t{k}.ffn_{f}_exps.weight"] == 20 for k in range(3) for f in ("up", "down"))
    assert ncall["output.weight"] == 3
    assert {n: v for n, _, v in entries}["output.weight"] == [5.0, 6.0]
    assert trailer == {"last_call": 7, "dataset": "ab.txt"}


def test_no_original_and_no_gate_entry_is_kept(split):
    (entries, _), _ = split
    names = {n for n, _, _ in entries}
    assert not names & {"blk.0.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight"}
    assert not any("gate" in n for n in names)


def test_expert_e_at_position_p_of_tier_k_receives_the_slice_of_e(tmp_path):
    """Cross test of the contract with the splitter: the table says where expert e goes (tier k, position p = its
    index in layers[L][k]); the entry of tier k must hold, at [p * width, (p + 1) * width), the slice of e."""
    out = tmp_path / "tiers.dat"
    ti.split_matrix(merged_matrix(tmp_path / "ab.dat"), TABLE, out)
    got = {n: v for n, _, v in read_all(out)[0]}
    for layer, tiers in enumerate(TABLE["layers"]):
        for k, ids in enumerate(tiers):
            for p, e in enumerate(ids):
                up = got[ti.tier_name(layer, k, "up")][p * UP_WIDTH:(p + 1) * UP_WIDTH]
                down = got[ti.tier_name(layer, k, "down")][p * DOWN_WIDTH:(p + 1) * DOWN_WIDTH]
                assert up == [up_value(layer, e, j) for j in range(UP_WIDTH)]
                assert down == [down_value(layer, e, j) for j in range(DOWN_WIDTH)]


def test_identity_mode_writes_the_three_tiers(tmp_path):
    table = tt.build_identity(2, N_EXPERT)           # tiers [0, 3], [1], [2] in both layers
    out = tmp_path / "tiers.dat"
    ti.split_matrix(merged_matrix(tmp_path / "ab.dat"), table, out)
    got = {n: v for n, _, v in read_all(out)[0]}
    assert got["blk.0.t0.ffn_up_exps.weight"] == [0.0, 1.0, 30.0, 31.0]
    assert got["blk.0.t1.ffn_up_exps.weight"] == [10.0, 11.0]
    assert got["blk.1.t2.ffn_up_exps.weight"] == [1020.0, 1021.0]


def test_a_gate_entry_is_refused(tmp_path):
    src = write_dat(tmp_path / "g.dat", [("blk.0.ffn_gate_exps.weight", 1, [1.0] * 8)])
    with pytest.raises(mi.ImatrixError, match="gate"):
        ti.split_matrix(src, TABLE, tmp_path / "o.dat")


def test_an_entry_that_does_not_fit_the_table_is_refused(tmp_path):
    odd = write_dat(tmp_path / "odd.dat", [("blk.0.ffn_up_exps.weight", 1, [1.0] * 6)])
    with pytest.raises(mi.ImatrixError, match="6 values for 4 experts"):
        ti.split_matrix(odd, TABLE, tmp_path / "o.dat")
    far = write_dat(tmp_path / "far.dat", [("blk.2.ffn_up_exps.weight", 1, [1.0] * 8)])
    with pytest.raises(mi.ImatrixError, match="layer 2 of 2"):
        ti.split_matrix(far, TABLE, tmp_path / "o.dat")


def test_an_empty_tier_is_refused(tmp_path):
    table = {**TABLE, "layers": [[[0, 1, 2, 3], [], []], TABLE["layers"][1]]}
    with pytest.raises(mi.ImatrixError, match="empty tier"):
        ti.split_matrix(merged_matrix(tmp_path / "ab.dat"), table, tmp_path / "o.dat")


@pytest.mark.parametrize("family, other", [("up", "down"), ("down", "up")])
def test_a_layer_without_its_up_or_down_entry_is_refused(tmp_path, family, other):
    values = expert_values(up_value, 1, UP_WIDTH)
    src = write_dat(tmp_path / "missing.dat", [
        ("blk.0.ffn_up_exps.weight", 10, expert_values(up_value, 0, UP_WIDTH)),
        ("blk.0.ffn_down_exps.weight", 10, expert_values(down_value, 0, DOWN_WIDTH)),
        (f"blk.1.ffn_{other}_exps.weight", 20, values if other == "up" else expert_values(down_value, 1, DOWN_WIDTH)),
    ])
    with pytest.raises(mi.ImatrixError, match=f"layer 1 has no blk.1.ffn_{family}_exps.weight"):
        ti.split_matrix(src, TABLE, tmp_path / "o.dat")


def test_out_may_not_be_the_input(tmp_path):
    src = merged_matrix(tmp_path / "ab.dat")
    with pytest.raises(mi.ImatrixError, match="would be emptied"):
        ti.split_matrix(src, TABLE, src)
    assert src.stat().st_size > 0


def test_command_line(tmp_path, capsys):
    src = merged_matrix(tmp_path / "ab.dat")
    tiers = tmp_path / "tiers.json"
    tiers.write_bytes(tt.dump(TABLE))
    out = tmp_path / "tiers.dat"
    assert ti.main(["--imatrix", str(src), "--tiers", str(tiers), "--out", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "5 entries in, 13 out, 12 tier entries" in shown and "sha256" in shown


def test_a_bad_table_gives_exit_code_2(tmp_path, capsys):
    tiers = tmp_path / "bad.json"
    tiers.write_text('{"mode": "p3"}')
    assert ti.main(["--imatrix", str(merged_matrix(tmp_path / "ab.dat")), "--tiers", str(tiers),
                    "--out", str(tmp_path / "o.dat")]) == 2
    assert "error:" in capsys.readouterr().err
