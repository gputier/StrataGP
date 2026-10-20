"""Tests of quality_estimate.py on synthetic llama-perplexity logs in the real format, values worked out by hand.

    python -m pytest tools/v41/test_quality_estimate.py

Worked example: PPL(R) = 4, ln(PPL(U2)/PPL(R)) = 0.2, ln(PPL(U1)/PPL(R)) = 0.5, call shares vram 0.34, ram 0.60,
nvme 0.06 (direction A_to_B).  ln PPL(P3) = ln 4 + 0.6 * 0.2 + 0.06 * 0.5 = ln 4 + 0.15, so PPL(P3) = 4 * e^0.15
= 4 * 1.161834 = 4.647337.
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_estimate as qe  # noqa: E402

PPL_R = 4.0
EST = 4.0 * math.exp(0.15)           # 4.647337
SHARES = {"A_to_B": {"vram": 0.34, "ram": 0.60, "nvme": 0.06},
          "B_to_A": {"vram": 0.40, "ram": 0.55, "nvme": 0.05}}      # B_to_A: 0.11 + 0.025 = 0.135 < 0.15


def final_line(ppl, unc, chunks=45, n_ctx=2048):
    return (f"perplexity: calculating perplexity over {chunks} chunks, n_ctx={n_ctx}\n"
            f"[1]4.1,[2]4.0,\nFinal estimate: PPL over {chunks} chunks for n_ctx={n_ctx} = {ppl:.4f} +/- {unc:.5f}\n")


def kl_log(ppl, ppl_base, ln_ratio, kld=0.05, extra="", chunks=45, n_ctx=2048):
    """The statistics blocks of a --kl-divergence run, as perplexity.cpp prints them: the context line of the run,
    then the per-chunk table, whose last row index is the number of chunks."""
    table = "".join(f"{i:4d}       4.1000 ±    0.1000       0.01000 ±    0.00100       0.01000 ±    0.00100\n"
                    for i in range(1, chunks + 1))
    return f"""llm_load_print_meta: n_ctx_train      = 1048576
llama_init_from_model: n_ctx         = {n_ctx}
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence
{table}
====== Perplexity statistics ======
Mean PPL(Q)                   : {ppl:10.6f} ±   0.020000
Mean PPL(base)                : {ppl_base:10.6f} ±   0.010000
Cor(ln(PPL(Q)), ln(PPL(base))):  98.50%
Mean ln(PPL(Q)/PPL(base))     : {ln_ratio:10.6f} ±   0.003000
Mean PPL(Q)/PPL(base)         :   1.221403 ±   0.003000
Mean PPL(Q)-PPL(base)         :   0.885000 ±   0.012000

====== KL divergence statistics ======
Mean    KLD: {kld:10.6f} ±   0.000500
Maximum KLD:  12.000000
99.9%   KLD:   2.000000
99.0%   KLD:   0.900000
95.0%   KLD:   0.300000
90.0%   KLD:   0.150000
Median  KLD:   0.010000
10.0%   KLD:   0.000100

====== Token probability statistics ======
Mean    Δp: -1.200 ± 0.050 %
RMS Δp    :  9.500 ± 0.100 %
Same top p: 91.500 ± 0.200 %
{extra}"""


def logs(ppl_q=5.0, d_u2=0.2, d_u1=0.5, ppl_r=PPL_R):
    """Logs of the four runs; `ppl_q` is PPL(Q) and d_* the ln ratios to R of U2 and U1."""
    def kl(ln_ratio):
        return kl_log(ppl_r * math.exp(ln_ratio), ppl_r, ln_ratio)
    return {"R": final_line(ppl_r, 0.1), "Q": kl(math.log(ppl_q / ppl_r)), "U2": kl(d_u2), "U1": kl(d_u1)}


def budget(path, shares=SHARES, name="P3"):
    """A tier_budget.py report reduced to what quality_estimate reads."""
    doc = {"plans": [{"name": name, "cross_validation": {d: {"call_share": s} for d, s in shares.items()}}]}
    Path(path).write_text(json.dumps(doc))
    return path


@pytest.fixture
def bud(tmp_path):
    return budget(tmp_path / "budget.json")


def test_the_estimate_follows_the_formula_with_the_less_favourable_direction(bud):
    rep = qe.analyse(logs(), bud)
    e = rep["estimate_p3"]
    assert e["ppl"] == pytest.approx(4.647337, abs=1e-6)
    assert e["ppl"] == pytest.approx(EST)
    assert e["less_favourable_direction"] == "A_to_B"
    # B_to_A: ln 4 + 0.55 * 0.2 + 0.05 * 0.5 = ln 4 + 0.135
    assert e["by_direction"]["B_to_A"]["estimate"] == pytest.approx(4.0 * math.exp(0.135))
    assert "estimation" in e["note"]


def test_the_other_direction_is_kept_when_it_is_the_worse(tmp_path):
    flipped = {"A_to_B": SHARES["B_to_A"], "B_to_A": SHARES["A_to_B"]}
    rep = qe.analyse(logs(), budget(tmp_path / "b.json", flipped))
    assert rep["estimate_p3"]["less_favourable_direction"] == "B_to_A"
    assert rep["estimate_p3"]["ppl"] == pytest.approx(EST)


def test_the_uncertainty_adds_in_quadrature(bud):
    # ln PPL(R) +/- 0.1 / 4 = 0.025 ; the paired ln ratios are +/- 0.003 each
    # u(ln P3)^2 = 0.025^2 + (0.6 * 0.003)^2 + (0.06 * 0.003)^2 ; u(PPL) = PPL * u(ln)
    u = math.sqrt(0.025 ** 2 + (0.6 * 0.003) ** 2 + (0.06 * 0.003) ** 2)
    assert qe.analyse(logs(), bud)["estimate_p3"]["ppl_unc"] == pytest.approx(EST * u)


# the four cases of the threshold: PPL(U1) = 4 * e^0.5 = 6.594885 and the estimate is 4.647337 in all of them
def test_u1_under_the_oracle_is_retained_and_the_quality_is_acquired(bud):
    rep = qe.analyse(logs(ppl_q=7.0), bud)           # 6.5949 <= 7.0
    assert rep["verdict"] == "retenu" and rep["quality_acquired_for_any_bit_plan"] is True


def test_the_estimate_just_above_the_oracle_is_the_grey_zone(bud):
    rep = qe.analyse(logs(ppl_q=4.58), bud)          # 4.6473 - 4.58 = 0.067 < 0.1, U1 fails
    assert rep["verdict"] == "zone grise" and rep["quality_acquired_for_any_bit_plan"] is False
    assert "within 0.1" in rep["reason"]


def test_the_estimate_just_under_the_oracle_is_the_grey_zone(bud):
    rep = qe.analyse(logs(ppl_q=4.70), bud)          # 4.70 - 4.6473 = 0.053 < 0.1, U1 fails
    assert rep["verdict"] == "zone grise"
    assert "within 0.1" in rep["reason"]


def test_the_estimate_passing_while_u1_fails_is_the_grey_zone(bud):
    rep = qe.analyse(logs(ppl_q=5.5), bud)           # 5.5 - 4.6473 = 0.85, U1 6.5949 > 5.5
    assert rep["verdict"] == "zone grise"
    assert "passes while U1 fails" in rep["reason"]


def test_the_estimate_well_above_the_oracle_is_a_failure(bud):
    rep = qe.analyse(logs(ppl_q=4.4), bud)           # 4.6473 - 4.4 = 0.247 >= 0.1, U1 fails
    assert rep["verdict"] == "échec" and rep["quality_acquired_for_any_bit_plan"] is False


def test_the_grey_band_is_strict_and_u1_equal_to_the_oracle_passes():
    # binary-exact numbers: a gap of exactly 0.25 is not under a band of 0.25
    assert qe.verdict(4.0, 9.0, 4.25, band=0.25)[0] == "échec"
    assert qe.verdict(4.0, 9.0, 3.75, band=0.25)[2] == "the estimate passes while U1 fails"
    assert qe.verdict(4.0, 9.0, 3.5, band=0.25)[0] == "zone grise"       # passes (3.5 <= 4) while U1 fails
    assert qe.verdict(4.0, 4.0, 3.9)[0] == "retenu"                      # "does not exceed" includes equality


def test_the_default_band_is_the_plans_0_1():
    assert qe.GRAY_BAND == 0.1 and qe.CHUNKS == 45 and qe.N_CTX == 2048


def test_a_log_with_oops_is_refused(bud):
    bad = logs()
    bad["U2"] += "\nOops: inconsistent size for blk.0.ffn_up_exps.weight\n"
    with pytest.raises(qe.LogError, match="Oops"):
        qe.analyse(bad, bud)


def test_another_chunk_count_or_context_is_refused(bud):
    bad = logs()
    bad["R"] = final_line(PPL_R, 0.1, chunks=6)
    with pytest.raises(qe.LogError, match="6 chunks at n_ctx=2048, expected 45"):
        qe.analyse(bad, bud)
    bad["R"] = final_line(PPL_R, 0.1, n_ctx=512)
    with pytest.raises(qe.LogError, match="n_ctx=512"):
        qe.analyse(bad, bud)


@pytest.mark.parametrize("name", ["Q", "U2", "U1"])
def test_a_kl_log_with_another_chunk_count_or_context_is_refused(bud, name):
    bad = logs()
    bad[name] = kl_log(4.8, PPL_R, 0.18, chunks=6)
    with pytest.raises(qe.LogError, match=f"{name}: 6 chunks at n_ctx=2048, expected 45 at 2048"):
        qe.analyse(bad, bud)
    bad[name] = kl_log(4.8, PPL_R, 0.18, n_ctx=512)
    with pytest.raises(qe.LogError, match=f"{name}: 45 chunks at n_ctx=512"):
        qe.analyse(bad, bud)
    bad[name] = "\n".join(l for l in kl_log(4.8, PPL_R, 0.18).splitlines() if "llama_init" not in l)
    with pytest.raises(qe.LogError, match="no per-chunk table or no n_ctx line"):
        qe.analyse(bad, bud)


def test_the_chunks_and_context_of_a_kl_run_are_reported():
    s = qe.parse_kl(kl_log(4.8, 4.0, 0.18, chunks=3), "Q", chunks=3)
    assert s["chunks"] == 3 and s["n_ctx"] == 2048


def test_the_base_tolerance_is_1e_minus_4_relative(bud):
    assert qe.BASE_TOLERANCE == 1e-4
    ok = logs()
    ok["U1"] = kl_log(4.0 * math.exp(0.5), PPL_R * (1 + 5e-5), 0.5)          # 5e-5 apart: read
    qe.analyse(ok, bud)
    bad = logs()
    bad["U1"] = kl_log(4.0 * math.exp(0.5), PPL_R * (1 + 1e-3), 0.5)         # 1e-3 apart: refused
    with pytest.raises(qe.LogError, match="U1: PPL.base. .* is not R's"):
        qe.analyse(bad, bud)


def test_a_run_against_another_base_is_refused(bud):
    bad = logs()
    bad["U1"] = kl_log(6.6, 4.5, 0.5)                 # base 4.5 is 12.5 % away from R
    with pytest.raises(qe.LogError, match="U1: PPL.base. 4.5 is not R's"):
        qe.analyse(bad, bud)


def test_a_log_without_the_kl_block_is_refused(bud):
    bad = logs()
    bad["Q"] = final_line(5.0, 0.1)
    with pytest.raises(qe.LogError, match="Q: no per-chunk table or no n_ctx line"):
        qe.analyse(bad, bud)
    bad["Q"] = kl_log(5.0, PPL_R, 0.2).split("====== Perplexity")[0]       # the table, none of the statistics
    with pytest.raises(qe.LogError, match="Mean PPL.Q."):
        qe.analyse(bad, bud)
    with pytest.raises(qe.LogError, match="no 'Final estimate"):
        qe.parse_base("nothing", "R")


def test_the_statistics_are_read_with_their_uncertainty():
    s = qe.parse_kl(kl_log(4.8, 4.0, 0.182322, kld=0.0421), "Q")
    assert s["ppl"] == 4.8 and s["ppl_base"] == 4.0
    assert s["ln_ratio"] == pytest.approx(0.182322) and s["ln_ratio_unc"] == 0.003
    assert s["kld_mean"] == 0.0421 and s["kld_mean_unc"] == 0.0005
    assert s["kld_median"] == 0.01 and s["kld_p99"] == 0.9
    assert s["rms_dp_percent"] == 9.5 and s["same_top_percent"] == 91.5


def test_a_utf16_log_from_powershell_is_read(tmp_path):
    path = tmp_path / "r.log"
    path.write_bytes(b"\xff\xfe" + final_line(4.0, 0.1).encode("utf-16-le"))
    assert qe.parse_base(qe.read_log(path))["ppl"] == 4.0


def test_a_reencoded_plus_minus_sign_or_lost_delta_is_still_read():
    for sign in ("Â±", "\ufffd"):
        s = qe.parse_kl(kl_log(4.8, 4.0, 0.18).replace("±", sign), "Q")
        assert s["ln_ratio_unc"] == 0.003 and s["rms_dp_percent"] == 9.5
    # the Greek letter of the Δp lines may be lost: the names are compared without non-ASCII characters
    assert qe.parse_kl(kl_log(4.8, 4.0, 0.18).replace("Δ", ""), "Q")["rms_dp_percent"] == 9.5


def test_the_shares_are_read_from_a_real_tier_budget_report(tmp_path):
    import tier_budget as tb
    # 1 layer x 4 experts, 1 per token: A calls expert 0 (10x) and 1 (2x), B calls expert 0 (8x) and 2 (4x); the
    # plan keeps 1 pair per tier, so A_to_B (ranked 0, 1, 2, 3): vram = pair 0, ram = pair 1, nvme = pairs 2, 3
    a, b = [10, 2, 0, 0], [8, 0, 4, 0]
    tokens = {"A": 12, "B": 12, "total": 24}
    flat = {"A": a, "B": b, "total": [x + y for x, y in zip(a, b)]}
    sizes = tb.expert_bytes("fp4") + 1, tb.expert_bytes("iq2_xs") + 1        # budgets of exactly one pair
    plans = {"P3": ("fp4", "iq2_xs", "iq1_m")}
    report = tb.build_report(flat, tokens, plans, vram_gb=sizes[0] / 1e9, ram_gb=sizes[1] / 1e9, engram=0)
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(report))
    s = qe.shares(path, "P3")
    # A_to_B: ranking from A is 0, 1, 2, 3 ; calls in B: pair 0 = 8 (vram), pair 1 = 0 (ram), pairs 2, 3 = 4 (nvme)
    assert s["A_to_B"] == pytest.approx({"vram": 8 / 12, "ram": 0.0, "nvme": 4 / 12})
    # B_to_A: ranking from B is 0, 2, 1, 3 ; calls in A: pair 0 = 10 (vram), pair 2 = 0 (ram), pairs 1, 3 = 2 (nvme)
    assert s["B_to_A"] == pytest.approx({"vram": 10 / 12, "ram": 0.0, "nvme": 2 / 12})
    with pytest.raises(qe.LogError, match="no plan P9"):
        qe.shares(path, "P9")


def test_main_prints_the_word_estimation_and_the_verdict(tmp_path, bud, capsys):
    paths = {}
    for name, text in logs(ppl_q=5.5).items():
        paths[name] = tmp_path / f"{name}.log"
        paths[name].write_text(text, encoding="utf-8")
    out = tmp_path / "quality.json"
    code = qe.main(["--r", str(paths["R"]), "--q", str(paths["Q"]), "--u2", str(paths["U2"]),
                    "--u1", str(paths["U1"]), "--budget", str(bud), "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "estimation" in text and "verdict: zone grise" in text
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "zone grise" and doc["R"]["ppl"] == 4.0
    # an untrustworthy log is exit code 2, with a message
    paths["R"].write_text("Oops", encoding="utf-8")
    assert qe.main(["--r", str(paths["R"]), "--q", str(paths["Q"]), "--u2", str(paths["U2"]),
                    "--u1", str(paths["U1"]), "--budget", str(bud)]) == 2
    assert "Oops" in capsys.readouterr().err
