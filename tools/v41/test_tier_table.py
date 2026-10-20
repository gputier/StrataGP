"""Tests of tier_table.py on synthetic counts, expected values worked out by hand.

    python -m pytest tools/v41/test_tier_table.py

Toy model: 2 layers, 4 experts, 2 experts per token.  Both halves hold the counts of pairs 0..7 (layer-major)
[4, 2, 0, 2, 0, 6, 2, 0], 4 tokens each, so the total is [8, 4, 0, 4, 0, 12, 4, 0].  Ranked by total count with ties
on the pair index: 5 (12), 0 (8), 1, 3, 6 (4 each), 2, 4, 7 (0 each).
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tier_table as tt  # noqa: E402

HALF = [4, 2, 0, 2, 0, 6, 2, 0]


def write_stats(path):
    def grid(flat):
        return [flat[:4], flat[4:]]
    doc = {"n_layers": 2, "n_expert": 4, "n_used": 2, "tokens": {"A": 4, "B": 4},
           "counts": {"A": grid(HALF), "B": grid(HALF), "total": grid([2 * c for c in HALF])}}
    Path(path).write_text(json.dumps(doc))
    return path


def write_budget(path, sizes=(1, 3, 4), types=("fp4", "iq2_xs", "iq1_m"), name="P3"):
    tiers = {m: {"type": ty, "pairs": n} for m, ty, n in zip(("vram", "ram", "nvme"), types, sizes)}
    Path(path).write_text(json.dumps({"plans": [{"name": "P2", "tiers": {}}, {"name": name, "tiers": tiers}]}))
    return path


@pytest.fixture
def files(tmp_path):
    return write_stats(tmp_path / "stats.json"), write_budget(tmp_path / "budget.json")


def test_p3_cuts_the_ranking_at_the_budget_sizes(files):
    # tier 0 = {5}; tier 1 = {0, 1, 3}; tier 2 = {6, 2, 4, 7}; pair p is expert p % 4 of layer p // 4
    table = tt.build_p3(*files)
    assert table["mode"] == "p3"
    assert table["n_expert"] == 4 and table["n_layer"] == 2
    assert table["tier_types"] == ["mxfp4", "iq2_xs", "iq1_m"]
    assert table["layers"] == [[[], [0, 1, 3], [2]], [[1], [], [0, 2, 3]]]


def test_p3_ids_are_increasing_whatever_the_rank_order(files):
    # tier 2 of layer 1 receives pairs 6, 4, 7 in that rank order, so experts 2, 0, 3: stored as [0, 2, 3]
    assert tt.build_p3(*files)["layers"][1][2] == [0, 2, 3]


def test_a_tier_of_the_budget_may_be_empty_in_a_layer_but_the_layer_stays_a_partition(files):
    layers = tt.build_p3(*files)["layers"]
    assert layers[0][0] == [] and layers[1][1] == []
    for tiers in layers:
        assert sorted(e for ids in tiers for e in ids) == [0, 1, 2, 3]


def test_identity_is_e_modulo_three_in_every_layer():
    table = tt.build_identity(2, 6)
    assert table["mode"] == "identity"
    assert table["layers"] == [[[0, 3], [1, 4], [2, 5]]] * 2
    assert tt.build_identity(1, 4)["layers"] == [[[0, 3], [1], [2]]]


def test_the_budget_sizes_must_cover_every_pair(tmp_path):
    stats = write_stats(tmp_path / "stats.json")
    budget = write_budget(tmp_path / "budget.json", sizes=(1, 3, 3))
    with pytest.raises(tt.TableError, match="sum to 7"):
        tt.build_p3(stats, budget)


def test_the_plan_types_must_be_the_file_types(tmp_path):
    stats = write_stats(tmp_path / "stats.json")
    budget = write_budget(tmp_path / "budget.json", types=("q4_k", "iq2_xs", "iq1_m"))
    with pytest.raises(tt.TableError, match="expected"):
        tt.build_p3(stats, budget)


def test_an_unknown_plan_is_refused(files):
    with pytest.raises(tt.TableError, match="0 plans named 'P9'"):
        tt.build_p3(*files, plan="P9")


@pytest.mark.parametrize("layers", [
    [[[0, 1], [2], [2, 3]]],            # expert 2 twice
    [[[0, 1], [2], []]],                # expert 3 missing
    [[[1, 0], [2], [3]]],               # not increasing
    [[[0, 1], [2, 4], [3]]],            # id out of range
    [[[0, 1], [2, 3]]],                 # two tiers
])
def test_a_table_that_is_not_a_partition_is_refused(layers):
    table = {"mode": "p3", "n_expert": 4, "n_layer": 1, "tier_types": tt.TIER_TYPES, "layers": layers}
    with pytest.raises(tt.TableError):
        tt.check_table(table)


def test_the_layer_count_must_match(files):
    table = tt.build_p3(*files)
    table["n_layer"] = 3
    with pytest.raises(tt.TableError, match="2 layers"):
        tt.check_table(table)


def test_the_file_is_deterministic_and_loads_back(files, tmp_path, capsys):
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    for out in (out_a, out_b):
        assert tt.main(["--mode", "p3", "--stats", str(files[0]), "--budget", str(files[1]), "--out", str(out)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    assert hashlib.sha256(out_a.read_bytes()).hexdigest() in capsys.readouterr().out
    assert tt.load_table(out_a) == tt.build_p3(*files)
    assert json.loads(out_a.read_text()).keys() == {"mode", "n_expert", "n_layer", "tier_types", "layers"}


def test_the_identity_mode_runs_from_the_command_line(tmp_path, capsys):
    out = tmp_path / "id.json"
    assert tt.main(["--mode", "identity", "--n-layer", "2", "--n-expert", "6", "--out", str(out)]) == 0
    assert tt.load_table(out)["layers"][1] == [[0, 3], [1, 4], [2, 5]]
    assert "sha256" in capsys.readouterr().out


@pytest.mark.parametrize("which", [0, 1])
def test_out_may_not_be_an_input(files, which, capsys):
    before = Path(files[which]).read_bytes()
    assert tt.main(["--mode", "p3", "--stats", str(files[0]), "--budget", str(files[1]),
                    "--out", str(files[which])]) == 2
    assert "would be overwritten" in capsys.readouterr().err
    assert Path(files[which]).read_bytes() == before


def test_a_bad_budget_gives_exit_code_2(files, tmp_path, capsys):
    bad = write_budget(tmp_path / "bad.json", sizes=(1, 1, 1))
    assert tt.main(["--mode", "p3", "--stats", str(files[0]), "--budget", str(bad),
                    "--out", str(tmp_path / "o.json")]) == 2
    assert "error:" in capsys.readouterr().err
