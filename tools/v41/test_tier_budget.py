"""Tests of tier_budget.py on synthetic call counts, expected values worked out by hand.

    python -m pytest tools/v41/test_tier_budget.py
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tier_budget as tb  # noqa: E402

# a toy placement: hot 4 bytes, warm 2 bytes, cold 3 bytes per pair
SIZES = {"hot": 4, "warm": 2, "cold": 3}


def write_stats(path, a, b, n_layers, n_expert, n_used):
    """A stats.json in the layout of expert_stats.py from two flat count lists (layer-major)."""
    def grid(flat):
        return [flat[l * n_expert:(l + 1) * n_expert] for l in range(n_layers)]
    per_token = n_layers * n_used
    total = [x + y for x, y in zip(a, b)]
    assert sum(a) % per_token == 0 and sum(b) % per_token == 0
    doc = {"n_layers": n_layers, "n_expert": n_expert, "n_used": n_used,
           "tokens": {"A": sum(a) // per_token, "B": sum(b) // per_token},
           "counts": {"A": grid(a), "B": grid(b), "total": grid(total)}}
    Path(path).write_text(json.dumps(doc))
    return path


def disjoint_halves():
    """4 layers x 4 experts, 3 experts per token and layer: half A calls only pairs 0..3 (30 calls each, 10 tokens),
    half B only pairs 12..15 (30 each, 10 tokens)."""
    a = [30] * 4 + [0] * 12
    b = [0] * 12 + [30] * 4
    return a, b


def test_expert_sizes_match_the_published_figures():
    assert tb.PARAMS_PER_EXPERT == 35_389_440
    assert tb.expert_bytes("fp4") == 18_800_640
    assert tb.expert_bytes("q4_k") == 19_906_560
    assert tb.expert_bytes("iq3_xxs") == 13_547_520
    assert tb.expert_bytes("iq2_xs") == 10_229_760
    assert tb.expert_bytes("iq1_m") == 7_741_440


def test_sizes_are_exact_integers_from_whole_blocks():
    # 256 weights of IQ2_XS take 74 bytes: 35 389 440 / 256 = 138 240 blocks
    assert tb.expert_bytes("iq2_xs") == 138_240 * 74
    # a count that does not fill whole blocks is refused, not rounded
    with pytest.raises(ValueError, match="whole"):
        tb.expert_bytes("q4_k", params=1000)
    with pytest.raises(ValueError, match="unknown type"):
        tb.expert_bytes("q2_k")


def test_engram_bytes_follow_the_page_reads():
    assert tb.engram_bytes_per_token(straddle=0) == 196_608
    assert tb.engram_bytes_per_token() == pytest.approx(200_540.16)


def test_budgets_are_respected_with_floor_rounding():
    # 11 bytes of VRAM hold 2 pairs of 4 bytes (8 bytes), not 3; 5 bytes of RAM hold 2 pairs of 2 bytes
    split = tb.split_tiers(16, SIZES, vram_bytes=11, ram_bytes=5)
    assert split == {"hot": 2, "warm": 2, "cold": 12}
    assert split["hot"] * SIZES["hot"] <= 11 and split["warm"] * SIZES["warm"] <= 5
    # a budget under one pair holds nothing
    assert tb.split_tiers(16, SIZES, vram_bytes=3, ram_bytes=1) == {"hot": 0, "warm": 0, "cold": 16}


def test_budgets_larger_than_the_pairs_take_them_all():
    assert tb.split_tiers(16, SIZES, vram_bytes=10**6, ram_bytes=10**6) == {"hot": 16, "warm": 0, "cold": 0}
    # the RAM only takes what the VRAM left
    assert tb.split_tiers(16, SIZES, vram_bytes=40, ram_bytes=10**6) == {"hot": 10, "warm": 6, "cold": 0}


def test_real_budgets_in_go_never_overflow():
    for hot, warm, _ in tb.DEFAULT_PLANS.values():
        sizes = {"hot": tb.expert_bytes(hot), "warm": tb.expert_bytes(warm), "cold": 1}
        split = tb.split_tiers(15_360, sizes, 26 * tb.GB, 100 * tb.GB)
        assert split["hot"] * sizes["hot"] <= 26 * tb.GB
        assert split["warm"] * sizes["warm"] <= 100 * tb.GB
        assert sum(split.values()) == 15_360
    # fp4 hot: 26e9 / 18 800 640 = 1382.9, IQ2_XS warm: 100e9 / 10 229 760 = 9775.5
    sizes = {"hot": 18_800_640, "warm": 10_229_760, "cold": 1}
    assert tb.split_tiers(15_360, sizes, 26 * tb.GB, 100 * tb.GB) == {"hot": 1382, "warm": 9775, "cold": 4203}


def test_ranking_orders_by_calls_then_by_index():
    assert tb.rank([5, 9, 5, 0, 9]) == [1, 4, 0, 2, 3]


def test_uniform_routing_spreads_the_calls_by_pair_count():
    # 16 pairs of 10 calls: 160 calls = 20 tokens x 2 layers x 4 experts
    counts = [10] * 16
    split = tb.split_tiers(16, SIZES, vram_bytes=12, ram_bytes=10)
    assert split == {"hot": 3, "warm": 5, "cold": 8}
    r = tb.traffic(tb.rank(counts), split, counts, eval_tokens=20, sizes=SIZES, engram=100)
    assert r["call_share"] == {"vram": 3 / 16, "ram": 5 / 16, "nvme": 8 / 16}
    # cold: 8 pairs x 10 calls / 20 tokens x 3 bytes = 12 ; warm: 5 x 10 / 20 x 2 = 5
    assert r["nvme_experts_bytes_per_token"] == pytest.approx(12)
    assert r["nvme_engram_bytes_per_token"] == 100
    assert r["nvme_total_bytes_per_token"] == pytest.approx(112)
    assert r["ram_bytes_per_token"] == pytest.approx(5)
    assert r["nvme_ceiling_tok_s"] == pytest.approx(6.8e9 / 112)
    assert r["ram_ceiling_tok_s"]["21"] == pytest.approx(21e9 / 5)
    assert r["ram_ceiling_tok_s"]["83"] == pytest.approx(83e9 / 5)


def test_concentrated_routing_puts_the_calls_in_the_fast_tiers():
    # 200 calls = 25 tokens x 2 layers x 4 experts, shuffled over the 16 pairs
    counts = [10, 0, 100, 5, 0, 20, 0, 50, 2, 0, 10, 0, 3, 0, 0, 0]
    assert sum(counts) == 200
    split = tb.split_tiers(16, SIZES, vram_bytes=8, ram_bytes=6)
    assert split == {"hot": 2, "warm": 3, "cold": 11}
    r = tb.traffic(tb.rank(counts), split, counts, eval_tokens=25, sizes=SIZES, engram=0)
    # hot: pairs with 100 and 50 = 150 calls ; warm: 20, 10, 10 = 40 ; cold: 5 + 3 + 2 = 10
    assert r["call_share"] == {"vram": 150 / 200, "ram": 40 / 200, "nvme": 10 / 200}
    assert r["nvme_experts_bytes_per_token"] == pytest.approx(10 / 25 * 3)
    assert r["ram_bytes_per_token"] == pytest.approx(40 / 25 * 2)


def test_nothing_read_on_nvme_gives_no_ceiling():
    counts = [1, 1, 1, 1]
    split = tb.split_tiers(4, SIZES, vram_bytes=100, ram_bytes=0)
    r = tb.traffic(tb.rank(counts), split, counts, eval_tokens=1, sizes=SIZES, engram=0)
    assert r["nvme_experts_bytes_per_token"] == 0
    assert r["nvme_ceiling_tok_s"] is None
    assert r["ram_ceiling_tok_s"] == {"21": None, "83": None}


def test_cross_validation_on_disjoint_halves_loses_the_hot_tiers(tmp_path):
    a, b = disjoint_halves()
    flat, tokens, *_ = tb.load_stats(write_stats(tmp_path / "s.json", a, b, 4, 4, 3))
    assert tokens == {"A": 10, "B": 10, "total": 20}
    # fp4 everywhere ; 0.038 Go of VRAM holds 2 pairs, 0.06 Go of RAM holds 3
    p = tb.evaluate_plan("T", ("fp4", "fp4", "fp4"), flat, tokens, vram_gb=0.038, ram_gb=0.06, engram=0)
    assert {m: t["pairs"] for m, t in p["tiers"].items()} == {"vram": 2, "ram": 3, "nvme": 11}
    assert p["tiers"]["vram"]["gb"] == pytest.approx(2 * 18_800_640 / 1e9)
    cold = 18_800_640
    # total: the order is 0..3 then 12..15 (all 30 calls) ; hot {0, 1}, warm {2, 3, 12}, cold {13, 14, 15}:
    # 90 of 240 calls, 12 calls per token
    assert p["total"]["call_share"] == {"vram": 60 / 240, "ram": 90 / 240, "nvme": 90 / 240}
    assert p["total"]["nvme_experts_bytes_per_token"] == pytest.approx(90 / 20 * cold)
    # learned on A, measured on B: hot {0, 1} and warm {2, 3, 4} are never called by B, so all 12 calls per token
    # fall on cold pairs
    r = p["cross_validation"]["A_to_B"]
    assert r["call_share"] == {"vram": 0.0, "ram": 0.0, "nvme": 1.0}
    assert r["nvme_experts_bytes_per_token"] == pytest.approx(12 * cold)
    # learned on B, measured on A: hot {12, 13}, warm {14, 15, 0}; only pair 0 (30 of 120 calls) is caught by the
    # warm tier, 9 calls per token stay cold
    r = p["cross_validation"]["B_to_A"]
    assert r["call_share"] == {"vram": 0.0, "ram": 0.25, "nvme": 0.75}
    assert r["nvme_experts_bytes_per_token"] == pytest.approx(9 * cold)
    # the verdict comes from the cross-validation: 12 x 18 800 640 = 225 607 680 > 0.2 Go, although the total
    # (84 602 880 per token) would have passed
    assert p["total"]["nvme_total_bytes_per_token"] < tb.NVME_BYTES_PER_TOKEN_MAX
    assert p["cross_validation"]["worst"]["nvme_total_bytes_per_token"] == pytest.approx(12 * cold)
    assert p["verdict"]["value_bytes_per_token"] == pytest.approx(12 * cold)
    assert p["verdict"]["pass"] is False


def test_cross_validation_takes_the_worse_direction(tmp_path):
    # A calls pairs 0..3 (30 each) ; B splits its calls over pairs 0..1 (45) and 12..13 (15)
    a, _ = disjoint_halves()
    b = [45, 45, 0, 0] + [0] * 8 + [15, 15, 0, 0]
    flat, tokens, *_ = tb.load_stats(write_stats(tmp_path / "s.json", a, b, 4, 4, 3))
    p = tb.evaluate_plan("T", ("fp4", "fp4", "fp4"), flat, tokens, vram_gb=0.038, ram_gb=0.06, engram=0)
    cv = p["cross_validation"]
    # A->B: hot {0, 1} serve 90 of B's 120 calls, warm {2, 3, 4} none, cold the 30 calls of 12..13 : 3 per token
    assert cv["A_to_B"]["nvme_experts_bytes_per_token"] == pytest.approx(3 * 18_800_640)
    # B->A: order 0, 1, 12, 13, then ties on 0 calls ; hot {0, 1}, warm {12, 13, 2}: A's 120 calls fall 60 hot, 30
    # warm (pair 2), 30 cold (pair 3) : 3 per token
    assert cv["B_to_A"]["nvme_experts_bytes_per_token"] == pytest.approx(3 * 18_800_640)
    assert cv["worst"]["nvme_experts_bytes_per_token"] == pytest.approx(3 * 18_800_640)
    # the worst direction is the one with the larger NVMe traffic
    assert cv["worst"]["nvme_total_bytes_per_token"] == max(
        cv["A_to_B"]["nvme_total_bytes_per_token"], cv["B_to_A"]["nvme_total_bytes_per_token"])


def test_verdict_threshold_is_inclusive():
    flat = {"total": [1], "A": [1], "B": [1]}
    tokens = {"A": 1, "B": 1, "total": 2}
    # no call at all: the traffic is the engram term alone, set on and just above the threshold
    p = tb.evaluate_plan("T", ("fp4", "fp4", "fp4"), {k: [0] for k in flat}, tokens, 0, 0,
                         engram=tb.NVME_BYTES_PER_TOKEN_MAX)
    assert p["verdict"]["pass"] is True
    p = tb.evaluate_plan("T", ("fp4", "fp4", "fp4"), {k: [0] for k in flat}, tokens, 0, 0,
                         engram=tb.NVME_BYTES_PER_TOKEN_MAX + 1)
    assert p["verdict"]["pass"] is False


def test_load_stats_refuses_inconsistent_counts(tmp_path):
    a, b = disjoint_halves()
    good = write_stats(tmp_path / "good.json", a, b, 4, 4, 3)
    doc = json.loads(good.read_text())
    doc["counts"]["B"][0][0] += 1
    (tmp_path / "sum.json").write_text(json.dumps(doc))
    with pytest.raises(tb.StatsError, match="sum to"):
        tb.load_stats(tmp_path / "sum.json")
    doc = json.loads(good.read_text())
    doc["counts"]["A"] = doc["counts"]["A"][:-1]
    (tmp_path / "shape.json").write_text(json.dumps(doc))
    with pytest.raises(tb.StatsError, match="not 4 x 4"):
        tb.load_stats(tmp_path / "shape.json")
    doc = json.loads(good.read_text())
    del doc["tokens"]
    (tmp_path / "key.json").write_text(json.dumps(doc))
    with pytest.raises(tb.StatsError, match="missing key"):
        tb.load_stats(tmp_path / "key.json")


def test_parse_plan():
    assert tb.parse_plan("X=FP4,iq2_xs,iq1_m") == ("X", ("fp4", "iq2_xs", "iq1_m"))
    with pytest.raises(Exception):
        tb.parse_plan("X=fp4,iq2_xs")
    with pytest.raises(ValueError, match="unknown type"):
        tb.parse_plan("X=fp4,iq2_xs,q9")


def test_default_plans_are_the_three_of_the_plan():
    assert tb.DEFAULT_PLANS == {"P1": ("fp4", "iq3_xxs", "iq2_xs"),
                                "P2": ("q4_k", "iq2_xs", "iq1_m"),
                                "P3": ("fp4", "iq2_xs", "iq1_m")}


def test_main_writes_the_report_and_says_the_placement_is_static(tmp_path, capsys):
    a, b = disjoint_halves()
    stats = write_stats(tmp_path / "s.json", a, b, 4, 4, 3)
    out = tmp_path / "budget.json"
    assert tb.main(["--stats", str(stats), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert [p["name"] for p in doc["plans"]] == ["P1", "P2", "P3"]
    assert doc["placement"] == "static, no dynamic cache assumed"
    assert doc["budgets_gb"] == {"vram": 26.0, "ram": 100.0}
    # the 16 toy pairs all fit in 26 Go of VRAM: nothing on NVMe but the engram, the ceilings are null
    p1 = doc["plans"][0]
    assert p1["tiers"]["vram"]["pairs"] == 16 and p1["tiers"]["nvme"]["pairs"] == 0
    assert p1["cross_validation"]["worst"]["nvme_experts_bytes_per_token"] == 0
    assert p1["total"]["ram_ceiling_tok_s"] == {"21": None, "83": None}
    assert p1["verdict"]["pass"] is True
    shown = capsys.readouterr().out
    assert "static, no dynamic cache assumed" in shown and "P3" in shown and "PASS" in shown


def test_main_takes_budgets_and_plans_from_the_arguments(tmp_path):
    a, b = disjoint_halves()
    stats = write_stats(tmp_path / "s.json", a, b, 4, 4, 3)
    out = tmp_path / "budget.json"
    tb.main(["--stats", str(stats), "--out", str(out), "--vram-gb", "0.038", "--ram-gb", "0.06",
             "--plan", "ONLY=fp4,fp4,fp4"])
    doc = json.loads(out.read_text())
    assert [p["name"] for p in doc["plans"]] == ["ONLY"]
    assert doc["plans"][0]["tiers"]["vram"]["pairs"] == 2
    assert doc["plans"][0]["verdict"]["pass"] is False
