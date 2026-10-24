"""Tests for partition_summary.py on an engine log built from the format of src/qwen35/q36_engine.cu and q36_stats.hpp,
and on the sampler log, response.json and chunks.tsv built from the lines measure_partition.ps1 writes.  No engine, no GPU,
no server.

    python -I tools/ornith/test_partition_summary.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import partition_summary as ps  # noqa: E402

# 300 tokens in 10,000 ms: 30.00 tokens/s, 33.33 ms per token. Pool 6,000 ms (3,000 + 1,000 + 2,000) = 20 ms per token.
# GPU wait 2,400 ms = 8 ms per token, so the host outside the pool is 33.33 - 20 - 8 = 5.33 ms per token.
# Two wire windows: 48 steps with 480,000 us busy, then 96 steps with 960,000 us busy: 1,440,000 us over 144 steps = 10 ms.
# The engine prints them during the decode, before the request's own lines (generated, experts, stats).
ENGINE_LOG = (
    "strata-q36: loading the experts\n"
    "strata-q36: stats: wire window 1: share 0.45, 100 blobs chosen for PCIe (80 on demand, 20 prefetched), busy 480000 us "
    "on the copy streams: 4800.0 us per chosen blob, 4000.0 us per blob on the wire, 21.5 GB/s; CPU path 900 us per job, "
    "launch 12.0 us per chosen blob -> c / (c + w - l) = 0.214\n"
    "strata-q36: stats: wire window 2: share 0.45, 200 blobs chosen for PCIe (150 on demand, 50 prefetched), busy 960000 us "
    "on the copy streams: 4800.0 us per chosen blob, 4000.0 us per blob on the wire, 22.0 GB/s; CPU path 905 us per job, "
    "launch 12.0 us per chosen blob -> c / (c + w - l) = 0.216\n"
    "strata serve: session 1: prompt 27 tokens = 0 reused + 27 read in 900 ms (30.0 tok/s), 300 generated in 10000 ms "
    "(30.0 tok/s), drafts accepted 0 of 0 (length)\n"
    "strata serve: session 1: experts on the GPU 40.0% (1200 of 3000 rows), over PCIe 20.0% (600), on the CPU 40.0% (1200), "
    "7 slot replacements, PCIe share 0.45\n"
    "strata serve: session 1: stats: pool 6000 ms (gate/up 3000, quantize 1000, down 2000), 1.80 GB read, 0.60 GB/s over the "
    "row phases (0.30 over all three); bytes per token: GPU 150.0 MB, PCIe 70.0 MB, CPU 180.0 MB (300 tokens); host wait on "
    "the doorbell 2400 ms over 14400 layer steps (166.7 us each), host serving 1500 ms (104.2 us each); PCIe blobs per token: "
    "on demand 0.20, prefetched 0.10, prefetch hits 0.05 (12.3% of the prefetched)\n"
    "strata serve: session 1: stats: host wait on the doorbell by layer (ms): 50.0 49.0\n"
    "strata serve: session 1: stats: expert union (distinct / (window * top-k)) w2 0.890 w3 0.800 w4 0.750\n"
)

# the stats line's tail as the engine prints it since issue #96 (q36_stats.hpp): the decode rebalances' cost
REB_OLD = "(12.3% of the prefetched)\n"
REB_NEW = "(12.3% of the prefetched); decode rebalances: 42 slot swaps, 12.5 MB copied, 3.2 ms of host time\n"

MEM_LOG = (
    "0.0 30000000 -1 -1\n"
    "2.5 20000000 15000 32607\n"
    "5.0 13000000 16000 32607\n"
)


class ParseEngineLog(unittest.TestCase):
    def test_counters_of_one_request(self):
        launch = ps.Launch("launch_01")
        ps.parse_engine_log(ENGINE_LOG, launch)
        self.assertEqual(launch.problems, [])
        self.assertEqual(launch.tokens, 300)
        self.assertAlmostEqual(launch.tok_s, 30.0)
        self.assertAlmostEqual(launch.decode_ms, 10000.0)
        self.assertAlmostEqual(launch.pool_ms, 6000.0)
        self.assertAlmostEqual(launch.wait_ms, 2400.0)
        self.assertAlmostEqual(launch.gpu_mb, 150.0)
        self.assertAlmostEqual(launch.pcie_mb, 70.0)
        self.assertAlmostEqual(launch.cpu_mb, 180.0)
        self.assertAlmostEqual(launch.pcie_busy_ms, 10.0)
        self.assertAlmostEqual(launch.host_outside_ms, 10000.0 / 300 - 6000.0 / 300 - 2400.0 / 300)

    def test_the_last_complete_request_is_read(self):
        first = ENGINE_LOG.replace("300 generated in 10000 ms (30.0 tok/s)", "100 generated in 1000 ms (100.0 tok/s)")
        first = first.replace("(300 tokens)", "(100 tokens)", 1)
        log = first + ENGINE_LOG
        launch = ps.Launch("x")
        ps.parse_engine_log(log, launch)
        self.assertEqual(launch.tokens, 300)
        self.assertAlmostEqual(launch.tok_s, 30.0)

    def test_no_wire_window_is_a_problem_not_a_zero(self):
        log = "\n".join(line for line in ENGINE_LOG.splitlines() if "wire window" not in line)
        launch = ps.Launch("x")
        ps.parse_engine_log(log, launch)
        self.assertIsNone(launch.pcie_busy_ms)
        self.assertTrue(any("PCIe copy time" in p for p in launch.problems))

    def test_incomplete_log(self):
        launch = ps.Launch("x")
        ps.parse_engine_log("strata serve: session 1: stats: pool 6000 ms (gate/up", launch)
        self.assertIsNone(launch.tok_s)
        self.assertTrue(launch.problems)

    def test_percent_sign_in_the_stats_line(self):
        self.assertIsNotNone(ps.RE_STATS.search(ENGINE_LOG))
        self.assertEqual(ps.RE_STATS.search(ENGINE_LOG).group(21), "12.3")


    def test_wire_windows_of_the_request_before_are_not_read(self):
        # the warm-up's windows (a busy time that would give 100 ms per step) are left out: only the measured request's count
        first = ENGINE_LOG.replace("busy 480000 us", "busy 4800000 us").replace("busy 960000 us", "busy 9600000 us")
        launch = ps.Launch("x")
        ps.parse_engine_log(first + ENGINE_LOG, launch)
        self.assertAlmostEqual(launch.pcie_busy_ms, 10.0)

    def test_share_of_the_measured_request_and_the_one_before(self):
        first = ENGINE_LOG.replace("PCIe share 0.45", "PCIe share 0.55")
        launch = ps.Launch("x")
        ps.parse_engine_log(first + ENGINE_LOG, launch)
        self.assertAlmostEqual(launch.share, 0.45)
        self.assertAlmostEqual(launch.share_before, 0.55)
        one = ps.Launch("y")
        ps.parse_engine_log(ENGINE_LOG, one)
        self.assertAlmostEqual(one.share, 0.45)
        self.assertIsNone(one.share_before)

    def test_stats_line_of_the_request_before_is_not_used(self):
        # the measured request printed no stats line (another session decoded at the same time): the warm-up's stats line,
        # earlier in the log, is not taken for it, and the launch is not measured
        measured = "\n".join(line for line in ENGINE_LOG.splitlines() if ": stats: pool " not in line) + "\n"
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG + measured, launch)
        self.assertIsNone(launch.tok_s)
        self.assertFalse(launch.measured)
        self.assertTrue(any("no stats line after the measured request" in p for p in launch.problems))

    def test_stats_line_of_another_session_is_not_used(self):
        other = ENGINE_LOG.replace("strata serve: session 1: stats: pool", "strata serve: session 2: stats: pool")
        launch = ps.Launch("x")
        ps.parse_engine_log(other, launch)
        self.assertFalse(launch.measured)
        self.assertTrue(any("no stats line after the measured request" in p for p in launch.problems))

    def test_token_count_mismatch_is_not_measured(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG.replace("(300 tokens)", "(299 tokens)"), launch)
        self.assertIsNotNone(launch.tok_s)
        self.assertFalse(launch.short)
        self.assertFalse(launch.measured)
        self.assertTrue(any("token counts differ: 300 generated, 299 in the stats line" in p for p in launch.problems))

    def test_matching_stats_line_is_measured(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)
        self.assertTrue(launch.stats_tied)
        self.assertTrue(launch.measured)

    def test_rebalance_cost_is_read(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG.replace(REB_OLD, REB_NEW), launch)
        self.assertEqual(launch.problems, [])
        self.assertEqual(launch.rb_swaps, 42)
        self.assertAlmostEqual(launch.rb_mb, 12.5)
        self.assertAlmostEqual(launch.rb_ms, 3.2)
        self.assertEqual(launch.tokens, 300)
        self.assertEqual(ps.RE_STATS.search(ENGINE_LOG.replace(REB_OLD, REB_NEW)).group(21), "12.3")

    def test_old_stats_line_without_rebalances_still_parses(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)
        self.assertTrue(launch.measured)
        self.assertIsNone(launch.rb_swaps)
        self.assertIsNone(launch.rb_mb)
        self.assertIsNone(launch.rb_ms)


class ParseMemLog(unittest.TestCase):
    def test_minimum_free_and_vram(self):
        mem = ps.Mem()
        ps.parse_mem_log(MEM_LOG, mem)
        self.assertEqual(mem.samples, 3)
        self.assertEqual(mem.min_free_kib, 13000000)
        self.assertEqual(mem.max_vram_used_mib, 16000)
        self.assertEqual(mem.vram_total_mib, 32607)

    def test_comment_line_is_skipped(self):
        # an older guard's KILL line is only a comment now: the samples around it count, nothing else is read from it
        mem = ps.Mem()
        ps.parse_mem_log(MEM_LOG + "7.5 7000000 16000 32607\n# KILL: free 7000000 KiB below 8388608 KiB; stopped PIDs 12,34\n",
                         mem)
        self.assertEqual(mem.samples, 4)
        self.assertEqual(mem.min_free_kib, 7000000)
        self.assertFalse(hasattr(mem, "kill"))

    def test_missing_vram_reads_as_none(self):
        mem = ps.Mem()
        ps.parse_mem_log("0.0 30000000 -1 -1\n", mem)
        self.assertIsNone(mem.max_vram_used_mib)
        self.assertIsNone(mem.vram_total_mib)


class NoMemoryRule(unittest.TestCase):
    def test_low_free_memory_is_reported_not_discarded(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)
        launch.mem.min_free_kib = 2 * 1024 * 1024
        self.assertTrue(launch.measured)
        self.assertFalse(hasattr(ps, "MIN_FREE_GIB"))
        self.assertFalse(hasattr(launch, "discarded"))


# 21 text chunks: the first at 500 ms (TTFT), then 19 gaps of 30 ms and one of 130 ms. Nearest-rank p95 of the 20 gaps is the
# 19th smallest, 30 ms; the one gap above it (130 ms) is 130 / 700 of the decode time and 100 ms over the median (30 ms).
CHUNK_MS = [500.0 + 30.0 * i for i in range(20)] + [500.0 + 30.0 * 19 + 130.0]


def chunks_tsv(times: list) -> str:
    return "chunk\tt_ms\tchars\n" + "".join(f"{i}\t{t:.1f}\t4\n" for i, t in enumerate(times))


class InterToken(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        vals = list(range(1, 21))
        self.assertEqual(ps.percentile(vals, 95.0), 19)
        self.assertEqual(ps.percentile(vals, 50.0), 10)
        self.assertEqual(ps.percentile([7.0], 95.0), 7.0)
        self.assertIsNone(ps.percentile([], 95.0))

    def test_gaps_p95_and_slow_tokens(self):
        launch = ps.Launch("x")
        launch.chunk_ms = ps.parse_chunks(chunks_tsv(CHUNK_MS))
        self.assertEqual(len(launch.chunk_ms), 21)
        self.assertAlmostEqual(launch.ttft_ms, 500.0)
        self.assertEqual(len(launch.gaps_ms), 20)
        self.assertAlmostEqual(launch.itl_p50, 30.0)
        self.assertAlmostEqual(launch.itl_p95, 30.0)
        self.assertEqual([round(g, 6) for g in launch.slow_gaps], [130.0])
        self.assertAlmostEqual(launch.slow_share, 130.0 / 700.0)
        self.assertAlmostEqual(launch.slow_excess_ms, 100.0)

    def test_no_chunk_gives_none(self):
        launch = ps.Launch("x")
        self.assertIsNone(launch.itl_p95)
        self.assertIsNone(launch.slow_share)
        self.assertIsNone(launch.ttft_ms)


class Short(unittest.TestCase):
    def test_fewer_completion_tokens_than_asked(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)
        ps.parse_response(json.dumps({"max_tokens": 300, "completion_tokens": 120, "finish_reason": "stop"}), launch)
        self.assertTrue(launch.short)
        self.assertFalse(launch.measured)
        self.assertEqual(launch.finish_reason, "stop")

    def test_engine_count_below_asked_without_usage(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)   # 300 tokens generated
        ps.parse_response(json.dumps({"max_tokens": 400, "completion_tokens": None, "finish_reason": "stop"}), launch)
        self.assertIsNone(launch.completion_tokens)
        self.assertTrue(launch.short)

    def test_all_tokens_is_not_short(self):
        launch = ps.Launch("x")
        ps.parse_engine_log(ENGINE_LOG, launch)
        ps.parse_response(json.dumps({"max_tokens": 300, "completion_tokens": 300, "finish_reason": "length"}), launch)
        self.assertFalse(launch.short)
        self.assertTrue(launch.measured)


class Summary(unittest.TestCase):
    def make_launch(self, root: Path, name: str, tok_s: float, free_kib: int, completion: int = 300,
                    finish: str = "length", times: list = CHUNK_MS, warmup_share: str = "0.45",
                    own_stats: bool = True, rebalances: str = "") -> Path:
        d = root / name
        d.mkdir()
        # a warm-up request, then the measured one of 300 tokens at the given rate: decode time = 300 / rate seconds
        decode_ms = 300000.0 / tok_s
        text = ENGINE_LOG.replace("in 10000 ms (30.0 tok/s)", f"in {decode_ms:.0f} ms ({tok_s:.1f} tok/s)")
        if rebalances:   # the measured request's stats line with its decode rebalances (the tail after the prefetch hits)
            text = text.replace(REB_OLD, "(12.3% of the prefetched); decode rebalances: " + rebalances + "\n")
        if not own_stats:   # the measured request printed no stats line: only the warm-up's is in the log
            text = "\n".join(line for line in text.splitlines() if ": stats: pool " not in line) + "\n"
        warm = ENGINE_LOG.replace("PCIe share 0.45", "PCIe share " + warmup_share)
        (d / "engine.log").write_text(warm + text, encoding="utf-8")
        (d / "mem.log").write_text(f"0.0 {free_kib} 16000 32607\n", encoding="utf-8")
        resp = {"max_tokens": 300, "completion_tokens": completion, "finish_reason": finish}
        (d / "response.json").write_text(json.dumps(resp), encoding="utf-8")
        (d / "chunks.tsv").write_text(chunks_tsv(times), encoding="utf-8")
        return d

    def test_median_over_the_measured_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gib13 = 13 * 1024 * 1024
            self.make_launch(root, "launch_01", 30.0, gib13)
            self.make_launch(root, "launch_02", 32.0, gib13)
            self.make_launch(root, "launch_03", 50.0, 2 * 1024 * 1024)   # 2 GiB free: reported, still measured
            text = ps.summary(sorted(root.glob("launch_*")))
        self.assertIn("median over the measured launches (3 of 3)", text)
        self.assertNotIn("DISCARDED", text)
        self.assertIn("2.00", [l for l in text.splitlines() if l.startswith("launch_03")][0])
        tok_line = [l for l in text.splitlines() if l.strip().startswith("tokens/s (engine)")][0]
        self.assertIn("32.00", tok_line)
        p95_line = [l for l in text.splitlines() if l.strip().startswith("inter-token p95 ms")][0]
        self.assertIn("30.00", p95_line)
        share_line = [l for l in text.splitlines() if l.strip().startswith("tokens above the p95: share of decode")][0]
        self.assertIn("0.186", share_line)
        self.assertIn("per-token partition", text)

    def test_short_launch_is_flagged_and_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gib13 = 13 * 1024 * 1024
            self.make_launch(root, "launch_01", 30.0, gib13)
            self.make_launch(root, "launch_02", 32.0, gib13)
            self.make_launch(root, "launch_03", 50.0, gib13, completion=120, finish="stop")
            text = ps.summary(sorted(root.glob("launch_*")))
        self.assertIn("median over the measured launches (2 of 3)", text)
        line3 = [l for l in text.splitlines() if l.startswith("launch_03")][0]
        self.assertIn("SHORT (120 of 300 tokens, finish_reason stop)", line3)
        tok_line = [l for l in text.splitlines() if l.strip().startswith("tokens/s (engine)")][0]
        self.assertIn("31.00", tok_line)

    def test_moving_share_and_chunk_count_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_launch(root, "launch_01", 30.0, 13 * 1024 * 1024, warmup_share="0.55", times=CHUNK_MS[:10])
            text = ps.summary(sorted(root.glob("launch_*")))
        line = [l for l in text.splitlines() if l.startswith("launch_01")][0]
        self.assertIn("the PCIe share moved during the measured request (0.55 -> 0.45)", line)
        self.assertIn("10 text chunks for 300 tokens", line)

    def test_launch_without_its_own_stats_line_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gib13 = 13 * 1024 * 1024
            self.make_launch(root, "launch_01", 30.0, gib13, rebalances="10 slot swaps, 2.0 MB copied, 1.0 ms of host time")
            self.make_launch(root, "launch_02", 32.0, gib13, rebalances="20 slot swaps, 4.0 MB copied, 2.0 ms of host time")
            self.make_launch(root, "launch_03", 50.0, gib13, own_stats=False)   # only the warm-up's stats line
            text = ps.summary(sorted(root.glob("launch_*")))
        self.assertIn("median over the measured launches (2 of 3)", text)
        line3 = [l for l in text.splitlines() if l.startswith("launch_03")][0]
        self.assertIn("not measured; no stats line after the measured request", line3)
        tok_line = [l for l in text.splitlines() if l.strip().startswith("tokens/s (engine)")][0]
        self.assertIn("31.00", tok_line)
        swaps = [l for l in text.splitlines() if l.strip().startswith("decode rebalances: slot swaps")][0]
        self.assertEqual(swaps.split()[-1], "15")
        mb = [l for l in text.splitlines() if l.strip().startswith("decode rebalances: MB copied")][0]
        self.assertEqual(mb.split()[-1], "3.0")
        host = [l for l in text.splitlines() if l.strip().startswith("decode rebalances: host ms")][0]
        self.assertEqual(host.split()[-1], "1.5")

    def test_rebalances_of_older_lines_read_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_launch(root, "launch_01", 30.0, 13 * 1024 * 1024)
            text = ps.summary(sorted(root.glob("launch_*")))
        swaps = [l for l in text.splitlines() if l.strip().startswith("decode rebalances: slot swaps")][0]
        self.assertEqual(swaps.split()[-1], "n/a")


if __name__ == "__main__":
    unittest.main()
