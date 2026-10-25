"""Summarises the launches of tools/ornith/measure_partition.ps1 (issue #95): per launch and as a median, the decode rate,
the inter-token times of the streamed completion (p50, p95, the slow tokens above the p95) and the time per token that the
q36 engine's STRATA_Q36_STATS=1 counters give for the CPU pool, PCIe, the host's wait on the GPU and the host outside the pool.

Reads, per launch_NN folder: engine.log (the engine's stderr), mem.log (the passive sampler's free RAM and VRAM), and for the
measured request (the last one of the launch, after the warm-up) response.json and chunks.tsv (response_NN.json and
chunks_NN.tsv, the highest NN, with several prompts). Writes text to stdout.
The minimum free RAM is reported only: no launch is left out for its memory. A launch whose completion has fewer tokens than
asked is flagged SHORT and left out of the medians (issue #95 asks for 200 tokens or more). A launch whose measured request
has no stats line of its own, or whose stats line counts other tokens than its generated line, is not measured either.

    python -I tools/ornith/partition_summary.py <run folder>

Engine lines read (src/qwen35/q36_engine.cu and src/qwen35/q36_stats.hpp):
  strata serve: session S: prompt P tokens = R reused + F read in M ms (X tok/s), G generated in D ms (Y tok/s), ...
  strata serve: session S: experts on the GPU ..., N slot replacements, PCIe share H
  strata serve: session S: stats: pool ... ms (gate/up ..., quantize ..., down ...), ... bytes per token: GPU ... MB,
      PCIe ... MB, CPU ... MB (T tokens); host wait on the doorbell W ms over L layer steps (...), host serving ...
      ...; decode rebalances: N slot swaps, B MB copied, H ms of host time   (this last part is optional: older lines lack it)
  strata-q36: stats: wire window K: share ..., B blobs chosen for PCIe (...), busy U us on the copy streams: ...
  strata serve: session S: stats: decode steps (pool/wait/host/PCIe copies ms[/rows]), N steps, M listed: p/w/h/c ...
      (optional: older engines lack it; one p/w/h/c group per decode step, /r after it for a step of r > 1 rows)
The stats line is printed only for a single decoding session, right after its request's generated line: the one read is the
first of the same session after the last generated line, and so is the decode steps line. The wire window K covers 48 steps
for K = 1 and 96 after; only the windows printed during the measured request are read (a window that straddles the warm-up's
end is counted whole).
The "PCIe copies ms per decode step" figure is the busy time of those windows over the decode steps of the measured request,
taken from its stats line (the "(T tokens)" count: one step per token with --mtp 0), not over the printed windows' steps: the
engine counts every window and prints only those with copies. It is n/a without the stats line or without a wire window.

chunks.tsv: one line per streamed chunk that carried text, "index<TAB>t_ms<TAB>chars", t_ms counted from the moment the
request was sent. The inter-token time is the time between two consecutive such chunks (one chunk per token when the chunk
count equals usage.completion_tokens; otherwise the launch says so and the times are per chunk).

The per-token partition by path is the request's counters divided by its tokens; the slow tokens (above the p95) are
measured as times and as a share of the decode time, and their excess over the median is reported. With the decode steps
line, each slow token is matched to the decode step that produced it: the first token comes from the prompt, then decode
step k gives the next r_k tokens (r_k = 1 with --mtp 0), and the inter-token time before the first of them is the step's.
The match needs one text chunk per token and step rows that add up to the token count; otherwise the launch says why and
has no attribution. For each matched slow token the stage that made it slow is the one with the largest excess over the
launch's median step (pool, GPU wait, host, PCIe copies; the copies overlap the other three, so their excess can be the
pool's or the host's too). Without the decode steps line, which path made a given token slow is not in the logs.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NUM = r"(-?[\d.]+)"
RE_GEN = re.compile(
    r"strata serve: session (\d+): prompt (\d+) tokens = (\d+) reused \+ (\d+) read in " + NUM + r" ms \(" + NUM +
    r" tok/s\), (\d+) generated in " + NUM + r" ms \(" + NUM + r" tok/s\)")
RE_SHARE = re.compile(r"strata serve: session (\d+): experts on the GPU .*?, PCIe share " + NUM)
RE_STATS = re.compile(
    r"strata serve: session (\d+): stats: pool " + NUM + r" ms \(gate/up " + NUM + r", quantize " + NUM + r", down " + NUM +
    r"\), " + NUM + r" GB read, " + NUM + r" GB/s over the row phases \(" + NUM + r" over all three\); bytes per token: "
    r"GPU " + NUM + r" MB, PCIe " + NUM + r" MB, CPU " + NUM + r" MB \((\d+) tokens\); host wait on the doorbell " + NUM +
    r" ms over (\d+) layer steps \(" + NUM + r" us each\), host serving " + NUM + r" ms \(" + NUM + r" us each\); "
    r"PCIe blobs per token: on demand " + NUM + r", prefetched " + NUM + r", prefetch hits " + NUM + r" \(" + NUM +
    r"% of the prefetched\)"
    r"(?:; decode rebalances: (\d+) slot swaps, " + NUM + r" MB copied, " + NUM + r" ms of host time)?")
RE_WIRE = re.compile(
    r"strata-q36: stats: wire window (\d+): share " + NUM + r", (\d+) blobs chosen for PCIe \((\d+) on demand, (\d+) "
    r"prefetched\), busy " + NUM + r" us on the copy streams: " + NUM + r" us per chosen blob, " + NUM +
    r" us per blob on the wire, " + NUM + r" GB/s; CPU path " + NUM + r" us per job, launch " + NUM +
    r" us per chosen blob -> c / \(c \+ w - l\) = " + NUM)
RE_STEPS = re.compile(
    r"strata serve: session (\d+): stats: decode steps \(pool/wait/host/PCIe copies ms\[/rows\]\), (\d+) steps, (\d+) "
    r"listed:([^\n]*)")
RE_MEM = re.compile(r"^" + NUM + r" (\d+) (-?\d+) (-?\d+)$")


@dataclass
class Mem:
    min_free_kib: Optional[int] = None
    max_vram_used_mib: Optional[int] = None
    vram_total_mib: Optional[int] = None
    samples: int = 0


STAGES = ("pool", "wait", "host", "PCIe")


@dataclass
class Step:
    """One decode step of the decode steps line: the pool, the host's wait on the GPU, the host outside both, the copy streams'
    busy time on its PCIe copies (ms), and the rows (tokens) it kept."""
    pool: float
    wait: float
    host: float
    pcie: float
    rows: int = 1

    def stage(self, name: str) -> float:
        return {"pool": self.pool, "wait": self.wait, "host": self.host, "PCIe": self.pcie}[name]

    @property
    def wall(self) -> float:
        """The step's wall time: pool + wait + host (the copies overlap them)."""
        return self.pool + self.wait + self.host


def parse_steps(listed: str) -> Optional[list]:
    """The steps of a decode steps line's list (" p/w/h/c[/r] ..."); None when a group is malformed."""
    steps = []
    for group in listed.split():
        parts = group.split("/")
        if len(parts) not in (4, 5):
            return None
        try:
            steps.append(Step(*(float(x) for x in parts[:4]), rows=int(parts[4]) if len(parts) == 5 else 1))
        except ValueError:
            return None
    return steps


def percentile(values: list, q: float) -> Optional[float]:
    """The nearest-rank percentile: the smallest value with at least q percent of the values at or below it."""
    if not values:
        return None
    vals = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(vals)))
    return vals[rank - 1]


@dataclass
class Launch:
    name: str
    tokens: int = 0                     # the engine's count of generated tokens (the stats line's)
    tok_s: Optional[float] = None       # the engine's decode rate, from the generated line
    decode_ms: Optional[float] = None   # decode time of the request, from the generated line
    pool_ms: Optional[float] = None     # CPU pool, gate/up + quantize + down
    wait_ms: Optional[float] = None     # host waiting for the GPU (doorbell)
    gpu_mb: Optional[float] = None      # bytes per token read by the GPU path
    pcie_mb: Optional[float] = None
    cpu_mb: Optional[float] = None
    pcie_busy_ms: Optional[float] = None   # copy streams' busy time per decode step (overlaps the rest; not additive)
    rb_swaps: Optional[int] = None         # the decode rebalances of the request: slot swaps (None: an older stats line)
    rb_mb: Optional[float] = None          # their copies and slot table uploads, MB
    rb_ms: Optional[float] = None          # their host time, ms
    stats_tied: bool = False               # the stats line is the measured request's (after its generated line, same session
                                           # and same token count)
    steps: Optional[list] = None           # the measured request's decode steps (None: no decode steps line)
    steps_total: int = 0                   # the steps it ran, the ones the line does not list included
    share: Optional[float] = None          # the PCIe share at the end of the measured request
    share_before: Optional[float] = None   # the PCIe share at the end of the request before it (the warm-up)
    asked: Optional[int] = None            # max_tokens of the measured request
    completion_tokens: Optional[int] = None   # usage.completion_tokens of the measured request (None: not sent)
    finish_reason: Optional[str] = None
    chunk_ms: list = field(default_factory=list)   # arrival time of each streamed chunk that carried text, ms
    mem: Mem = field(default_factory=Mem)
    problems: list = field(default_factory=list)

    @property
    def host_outside_ms(self) -> Optional[float]:
        """Decode time per token less the pool and the GPU wait (the host's own share; can be negative when the
        pool and the wait overlap)."""
        if None in (self.decode_ms, self.pool_ms, self.wait_ms) or not self.tokens:
            return None
        return self.decode_ms / self.tokens - self.pool_ms / self.tokens - self.wait_ms / self.tokens

    @property
    def gaps_ms(self) -> list:
        """The inter-token times: between consecutive chunks that carried text (the first chunk is the TTFT, not a gap)."""
        return [b - a for a, b in zip(self.chunk_ms, self.chunk_ms[1:])]

    @property
    def ttft_ms(self) -> Optional[float]:
        return self.chunk_ms[0] if self.chunk_ms else None

    @property
    def itl_p50(self) -> Optional[float]:
        return percentile(self.gaps_ms, 50.0)

    @property
    def itl_p95(self) -> Optional[float]:
        return percentile(self.gaps_ms, 95.0)

    @property
    def slow_gaps(self) -> list:
        """The inter-token times above the p95."""
        p95 = self.itl_p95
        return [] if p95 is None else [g for g in self.gaps_ms if g > p95]

    @property
    def slow_share(self) -> Optional[float]:
        """The share of the streamed decode time spent in the tokens above the p95."""
        gaps = self.gaps_ms
        total = sum(gaps)
        return sum(self.slow_gaps) / total if gaps and total > 0 else None

    @property
    def slow_excess_ms(self) -> Optional[float]:
        """The mean excess of the tokens above the p95 over the median inter-token time."""
        slow = self.slow_gaps
        p50 = self.itl_p50
        return statistics.mean(slow) - p50 if slow and p50 is not None else None

    def step_of_gaps(self) -> tuple:
        """The decode step behind each inter-token time: (a list with, per gap, the index of the step whose first token ends
        it, or None for a step's later tokens; None and the reason when the steps and the streamed tokens do not match)."""
        if self.steps is None:
            return None, "no decode steps line"
        if self.completion_tokens is not None and self.completion_tokens != self.tokens:
            return None, f"usage.completion_tokens {self.completion_tokens} for {self.tokens} engine tokens"
        if len(self.chunk_ms) != self.tokens:
            return None, f"{len(self.chunk_ms)} text chunks for {self.tokens} tokens"
        rows = [s.rows for s in self.steps]
        if len(self.steps) == self.steps_total:
            # the last step's rows may run past the end of the answer (an end of answer or max_tokens inside a verify window)
            if not (1 + sum(rows[:-1]) < self.tokens <= 1 + sum(rows) if rows else self.tokens <= 1):
                return None, f"{self.steps_total} decode steps of {sum(rows)} rows for {self.tokens} tokens"
        elif 1 + sum(rows) > self.tokens:
            return None, f"{len(self.steps)} listed decode steps of {sum(rows)} rows for {self.tokens} tokens"
        of = [None] * len(self.gaps_ms)
        token = 1
        for k, r in enumerate(rows):
            if token - 1 < len(of):
                of[token - 1] = k
            token += r
        return of, None

    def slow_steps(self) -> list:
        """The tokens above the p95 that start a decode step: (token index, inter-token ms, step), in token order."""
        of, _ = self.step_of_gaps()
        p95 = self.itl_p95
        if of is None or p95 is None:
            return []
        return [(i + 1, g, self.steps[of[i]]) for i, g in enumerate(self.gaps_ms) if g > p95 and of[i] is not None]

    def median_step(self) -> Optional[Step]:
        """Each stage's median over the listed steps."""
        if not self.steps:
            return None
        return Step(*(statistics.median(s.stage(n) for s in self.steps) for n in STAGES))

    @property
    def short(self) -> bool:
        """Fewer tokens than asked, by usage.completion_tokens or by the engine's own count."""
        if self.asked is None:
            return False
        if self.completion_tokens is not None and self.completion_tokens < self.asked:
            return True
        return self.tok_s is not None and self.tokens < self.asked

    @property
    def measured(self) -> bool:
        return self.tok_s is not None and self.stats_tied and not self.short


def parse_engine_log(text: str, launch: Launch) -> None:
    """Fills `launch` from one engine log; the last request of the log (its generated line) is the one measured, and its stats
    line is the first one of the same session after that generated line (the engine prints it right after). No such line, or
    a token count that differs from the generated line's, leaves the launch not measured."""
    gens = list(RE_GEN.finditer(text))
    shares = list(RE_SHARE.finditer(text))
    if shares:
        launch.share = float(shares[-1].group(2))
        if len(shares) >= 2:
            launch.share_before = float(shares[-2].group(2))
    if not gens:
        launch.problems.append("no complete request in the engine log (no generated line)")
        return
    gen = gens[-1]
    stats = None
    for m in RE_STATS.finditer(text, gen.end()):
        if m.group(1) == gen.group(1):
            stats = m
            break
    if stats is None:
        launch.problems.append("no stats line after the measured request's generated line (another session decoded at the "
                               "same time, or the stats are off): not measured")
        return
    decode_ms = float(gen.group(8))
    launch.decode_ms = decode_ms
    launch.tok_s = float(gen.group(9))
    g = stats.groups()
    pool_total = float(g[1])
    launch.pool_ms = pool_total
    launch.gpu_mb = float(g[8])
    launch.pcie_mb = float(g[9])
    launch.cpu_mb = float(g[10])
    launch.tokens = int(g[11])
    launch.wait_ms = float(g[12])
    if g[21] is not None:
        launch.rb_swaps = int(g[21])
        launch.rb_mb = float(g[22])
        launch.rb_ms = float(g[23])
    if int(gen.group(7)) != launch.tokens:
        launch.problems.append(f"token counts differ: {gen.group(7)} generated, {launch.tokens} in the stats line: not measured")
    else:
        launch.stats_tied = True
    # the copy streams' busy time over the wire windows of the measured request (after the request before it), per decode step
    # of the request: the step count is the stats line's, because the engine counts every window and prints only those with
    # copies (the printed windows' steps would leave the others out); one decode step is one token with --mtp 0
    begin = gens[-2].end() if len(gens) >= 2 else 0
    busy_us = 0.0
    wires = 0
    for m in RE_WIRE.finditer(text):
        if m.start() < begin or m.start() > stats.end():
            continue
        busy_us += float(m.group(6))
        wires += 1
    if wires and launch.tokens > 0:
        launch.pcie_busy_ms = busy_us / 1000.0 / launch.tokens
    else:
        launch.problems.append("no wire window line or no step count in the stats line: the PCIe copy time is not measured")
    for m in RE_STEPS.finditer(text, gen.end()):
        if m.group(1) == gen.group(1):
            steps = parse_steps(m.group(4))
            if steps is None or len(steps) != int(m.group(3)):
                launch.problems.append("the decode steps line is malformed: no attribution of the slow tokens")
            else:
                launch.steps = steps
                launch.steps_total = int(m.group(2))
            break


def parse_mem_log(text: str, mem: Mem) -> None:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = RE_MEM.match(line)
        if not m:
            continue
        free = int(m.group(2))
        used, total = int(m.group(3)), int(m.group(4))
        mem.samples += 1
        mem.min_free_kib = free if mem.min_free_kib is None else min(mem.min_free_kib, free)
        if used >= 0:
            mem.max_vram_used_mib = used if mem.max_vram_used_mib is None else max(mem.max_vram_used_mib, used)
        if total >= 0:
            mem.vram_total_mib = total


def parse_chunks(text: str) -> list:
    """The arrival times (ms) of chunks.tsv, in file order; the header and malformed lines are skipped."""
    times = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        try:
            times.append(float(parts[1]))
        except ValueError:
            continue
    return times


def parse_response(text: str, launch: Launch) -> None:
    """max_tokens, usage.completion_tokens and finish_reason of the measured request (response.json of the script)."""
    try:
        r = json.loads(text)
    except ValueError:
        launch.problems.append("response file is not JSON")
        return
    if not isinstance(r, dict):
        launch.problems.append("response file is not a JSON object")
        return
    if isinstance(r.get("max_tokens"), int):
        launch.asked = r["max_tokens"]
    if isinstance(r.get("completion_tokens"), int):
        launch.completion_tokens = r["completion_tokens"]
    if isinstance(r.get("finish_reason"), str):
        launch.finish_reason = r["finish_reason"]


def last_of(folder: Path, single: str, pattern: str) -> Optional[Path]:
    """The measured request's file: `single` (one prompt) or the highest of `pattern` (several prompts)."""
    p = folder / single
    if p.is_file():
        return p
    many = sorted(q for q in folder.glob(pattern) if q.is_file())
    return many[-1] if many else None


def read_launch(folder: Path) -> Launch:
    launch = Launch(name=folder.name)
    engine = folder / "engine.log"
    if engine.is_file():
        parse_engine_log(engine.read_text(encoding="utf-8", errors="replace"), launch)
    else:
        launch.problems.append("no engine.log")
    mem = folder / "mem.log"
    if mem.is_file():
        parse_mem_log(mem.read_text(encoding="utf-8", errors="replace"), launch.mem)
    else:
        launch.problems.append("no mem.log")
    resp = last_of(folder, "response.json", "response_*.json")
    if resp is not None:
        parse_response(resp.read_text(encoding="utf-8", errors="replace"), launch)
    else:
        launch.problems.append("no response file")
    chunks = last_of(folder, "chunks.tsv", "chunks_*.tsv")
    if chunks is not None:
        launch.chunk_ms = parse_chunks(chunks.read_text(encoding="utf-8", errors="replace"))
    else:
        launch.problems.append("no chunks.tsv: no inter-token time")
    if launch.completion_tokens is None and launch.asked is not None:
        launch.problems.append("no usage.completion_tokens in the stream: the engine's count is used")
    if launch.completion_tokens is not None and launch.chunk_ms and len(launch.chunk_ms) != launch.completion_tokens:
        launch.problems.append(f"{len(launch.chunk_ms)} text chunks for {launch.completion_tokens} tokens: the inter-token "
                               "times are per chunk")
    if launch.share is not None and launch.share_before is not None and launch.share != launch.share_before:
        launch.problems.append(f"the PCIe share moved during the measured request ({launch.share_before:.2f} -> "
                               f"{launch.share:.2f})")
    return launch


def median_of(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


METRICS = [
    ("tokens/s (engine)", lambda l: l.tok_s, "{:.2f}"),
    ("decode ms per token", lambda l: (l.decode_ms / l.tokens) if l.decode_ms and l.tokens else None, "{:.2f}"),
    ("CPU pool ms per token", lambda l: (l.pool_ms / l.tokens) if l.pool_ms is not None and l.tokens else None, "{:.2f}"),
    ("host wait for GPU ms per token", lambda l: (l.wait_ms / l.tokens) if l.wait_ms is not None and l.tokens else None,
     "{:.2f}"),
    ("host outside pool ms per token", lambda l: l.host_outside_ms, "{:.2f}"),
    ("PCIe copies ms per decode step (overlaps)", lambda l: l.pcie_busy_ms, "{:.2f}"),
    ("GPU path MB per token", lambda l: l.gpu_mb, "{:.1f}"),
    ("PCIe MB per token", lambda l: l.pcie_mb, "{:.1f}"),
    ("CPU pool MB per token", lambda l: l.cpu_mb, "{:.1f}"),
    ("decode rebalances: slot swaps", lambda l: l.rb_swaps, "{:.0f}"),
    ("decode rebalances: MB copied", lambda l: l.rb_mb, "{:.1f}"),
    ("decode rebalances: host ms", lambda l: l.rb_ms, "{:.1f}"),
    ("TTFT ms (streamed)", lambda l: l.ttft_ms, "{:.0f}"),
    ("inter-token p50 ms", lambda l: l.itl_p50, "{:.2f}"),
    ("inter-token p95 ms", lambda l: l.itl_p95, "{:.2f}"),
    ("tokens above the p95: share of decode", lambda l: l.slow_share, "{:.3f}"),
    ("tokens above the p95: excess ms", lambda l: l.slow_excess_ms, "{:.2f}"),
]


def fmt(v: Optional[float], spec: str) -> str:
    return "n/a" if v is None else spec.format(v)


def dominant(step: Step, median: Step) -> str:
    """The stage with the largest excess over the median step."""
    return max(STAGES, key=lambda n: step.stage(n) - median.stage(n))


def stage_totals(slow: list, median: Step) -> tuple:
    """Over the slow tokens' steps: each stage's ms, its excess over the median step (ms), and how many it dominates."""
    ms = {n: sum(s.stage(n) for _, _, s in slow) for n in STAGES}
    excess = {n: sum(s.stage(n) - median.stage(n) for _, _, s in slow) for n in STAGES}
    count = {n: sum(1 for _, _, s in slow if dominant(s, median) == n) for n in STAGES}
    return ms, excess, count


def excess_text(excess: dict) -> str:
    """Each stage's excess and its share of the stages' positive excess."""
    pos = sum(v for v in excess.values() if v > 0)
    if pos <= 0:
        return ", ".join(f"{n} {excess[n]:.1f} ms" for n in STAGES)
    return ", ".join(f"{n} {excess[n]:.1f} ms ({100.0 * max(excess[n], 0.0) / pos:.0f}%)" for n in STAGES)


def slow_by_stage(launches: list) -> list:
    """The section on which stage made each slow token slow (the measured launches with a decode steps line)."""
    out = ["slow tokens by stage: the decode step behind each inter-token time above the p95 (ms; host = the step's wall time",
           "less the pool and the GPU wait; the PCIe copies overlap the other three). dominant: the stage with the largest excess",
           "over the launch's median step."]
    all_excess = {n: 0.0 for n in STAGES}
    all_count = {n: 0 for n in STAGES}
    matched = 0
    for l in launches:
        if not l.measured:
            out.append(f"{l.name}: not measured, left out")
            continue
        of, why = l.step_of_gaps()
        if of is None:
            out.append(f"{l.name}: no attribution: {why}")
            continue
        slow = l.slow_steps()
        med = l.median_step()
        out.append(f"{l.name}: p95 {fmt(l.itl_p95, '{:.2f}')} ms, {len(l.slow_gaps)} tokens above it, {len(slow)} matched to the "
                   f"first token of a decode step; median step: " + ", ".join(f"{n} {med.stage(n):.1f}" for n in STAGES))
        if l.steps_total != len(l.steps):
            out.append(f"  {l.steps_total - len(l.steps)} decode steps past the listed {len(l.steps)}: their tokens are not "
                       "attributed")
        if not slow:
            continue
        out.append(f"  {'token':>6}{'gap ms':>9}{'pool':>8}{'wait':>8}{'host':>8}{'PCIe':>8}  dominant")
        for token, gap, s in slow:
            out.append(f"  {token:>6}{gap:>9.2f}{s.pool:>8.1f}{s.wait:>8.1f}{s.host:>8.1f}{s.pcie:>8.1f}  {dominant(s, med)}")
        ms, excess, count = stage_totals(slow, med)
        wall = sum(s.wall for _, _, s in slow)
        share = (lambda n: f"{100.0 * ms[n] / wall:.0f}%") if wall > 0 else (lambda n: "n/a")
        out.append(f"  share of the slow steps' time: pool {share('pool')}, wait {share('wait')}, host {share('host')}; "
                   f"PCIe copies {share('PCIe')} (overlap)")
        out.append(f"  excess over the median step: {excess_text(excess)}")
        out.append("  dominant: " + ", ".join(f"{n} {count[n]}" for n in STAGES))
        matched += len(slow)
        for n in STAGES:
            all_excess[n] += excess[n]
            all_count[n] += count[n]
    out.append(f"all measured launches: {matched} slow tokens matched; dominant: " +
               ", ".join(f"{n} {all_count[n]}" for n in STAGES) + f"; excess over the median step: {excess_text(all_excess)}")
    return out


def summary(folders: list) -> str:
    launches = [read_launch(f) for f in folders]
    out = []
    out.append(f"{'launch':<11}{'tokens':>7}{'asked':>7}{'tok/s':>8}{'pool ms':>9}{'wait ms':>9}{'host ms':>9}{'PCIe ms':>9}"
               f"{'p50 ms':>8}{'p95 ms':>8}{'share':>7}{'min free GiB':>14}{'VRAM max MiB':>14}  status")
    for l in launches:
        free = "n/a" if l.mem.min_free_kib is None else f"{l.mem.min_free_kib / 1024 / 1024:.2f}"
        if l.short:
            got = l.completion_tokens if l.completion_tokens is not None else l.tokens
            status = f"SHORT ({got} of {l.asked} tokens, finish_reason {l.finish_reason or 'n/a'}): left out of the medians"
        elif not l.measured:
            status = "not measured"
        else:
            status = "measured"
        if l.problems:
            status += "; " + "; ".join(l.problems)
        ms_pt = (l.pool_ms / l.tokens) if (l.pool_ms is not None and l.tokens) else None
        wait_pt = (l.wait_ms / l.tokens) if (l.wait_ms is not None and l.tokens) else None
        asked = "n/a" if l.asked is None else str(l.asked)
        out.append(f"{l.name:<11}{l.tokens:>7}{asked:>7}{fmt(l.tok_s, '{:.2f}'):>8}{fmt(ms_pt, '{:.2f}'):>9}"
                   f"{fmt(wait_pt, '{:.2f}'):>9}{fmt(l.host_outside_ms, '{:.2f}'):>9}"
                   f"{fmt(l.pcie_busy_ms, '{:.2f}'):>9}{fmt(l.itl_p50, '{:.2f}'):>8}{fmt(l.itl_p95, '{:.2f}'):>8}"
                   f"{fmt(l.share, '{:.2f}'):>7}{free:>14}"
                   f"{'n/a' if l.mem.max_vram_used_mib is None else str(l.mem.max_vram_used_mib):>14}  {status}")
    kept = [l for l in launches if l.measured]
    out.append("")
    out.append(f"median over the measured launches ({len(kept)} of {len(launches)}):")
    if not kept:
        out.append("  no measured launch: nothing to report")
        return "\n".join(out)
    for label, get, spec in METRICS:
        med = median_of([get(l) for l in kept])
        out.append(f"  {label:<38}{fmt(med, spec):>10}")
    with_steps = any(l.steps is not None for l in launches)
    if with_steps:
        out.append("")
        out.extend(slow_by_stage(launches))
    out.append("")
    out.append("per-token partition: the request's counters over its tokens (pool, GPU wait, host outside the pool; the PCIe")
    out.append("copies overlap them). Slow tokens: the inter-token times above the p95 of the streamed completion, their share")
    if with_steps:
        out.append("of the decode time and their excess over the median. Which stage made each slow token slow: the section")
        out.append("above, from the engine's decode steps line.")
    else:
        out.append("of the decode time and their excess over the median. Which path made a given token slow is not in the logs:")
        out.append("the engine counts per request and per copy window, not per token.")
    return "\n".join(out)


def main(argv: list) -> int:
    if len(argv) != 2:
        print("usage: partition_summary.py <run folder>", file=sys.stderr)
        return 2
    run = Path(argv[1])
    folders = sorted(p for p in run.glob("launch_*") if p.is_dir())
    if not folders:
        print(f"no launch_* folder in {run}", file=sys.stderr)
        return 1
    print(summary(folders))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
