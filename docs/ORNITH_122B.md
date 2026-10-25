# Ornith-1.5-122B on StrataGP: engineering record

This is the record of bringing Ornith-1.5-122B (48 layers, 256 experts per layer, top-8 routing, about 10B active
parameters) to the q36 engine in hybrid mode, where the experts that do not fit in VRAM run on the CPU pool or are
copied over PCIe. It covers the protocols, the experiments in order, the final numbers, and the open leads. Work
branch `feat/ornith-122b`, pull request #83. Measurements of 22 and 23 October 2026.

All numbers come from the test box: RTX 5090 32 GB, 128 GB RAM, three CI VMs running during the measurements.
Figures marked "calculated" are arithmetic on measured values, not measurements.

## Result

| Measurement | Start (7bf040a5) | Final (ec3d4b63) |
|---|---|---|
| Decode, 1,024 tokens, four code requests per launch, median of three alternated launches, 24 GB working-set cap | 22.6 tokens/s (22.5 to 22.8) | 30.0 tokens/s (29.7 to 30.2) |
| Decode after a 127,685-token prompt, one launch per build | 10.4 tokens/s | 12.8 tokens/s |
| Prompt read, 127,685 tokens | 78.0 s | 78.0 s (1,637 tokens/s) |
| KL against llama.cpp b11756, 10 prompts of 32 steps | not measured | 0.0423 (bar 0.061) |
| Top-1 agreement, same test | not measured | 299 of 320 |

The branch is faster than its start under the 24 GB working-set cap (see below). No uncapped comparison with the start exists; the uncapped series of 24 October measured the branch alone (see its section). The N5 build (task 5) gave a KL of 0.0455 and 299 of 320; the start was not measured on this test. The target of 100 tokens/s is
not reached. By owner decision of 23 October 2026, there is no fixed throughput threshold: each result is judged case by case.

Prompt chunk size, issue #104. Measured on 25 October 2026 on the test PC without the cap (rev8, main's code, defaults otherwise, `STRATA_Q36_STATS=1`), one 34,257-token prompt then 300 decode tokens, three launches per setting alternated. `STRATA_Q36_CHUNK=2048` (the hybrid default here): 77 of 256 experts per layer in VRAM, prompt read in 15.21 s (2,252.3, 2,252.2 and 2,251.8 tokens/s), then 30.7, 30.6 and 30.8 tokens/s of decode. `STRATA_Q36_CHUNK=4096`: 68 per layer, prompt read in 9.37 s (3,656.2, 3,656.9 and 3,653.5 tokens/s, 62 % faster), then 30.4, 30.5 and 30.5 tokens/s. At short context the decode fell from 46.8 to 44.9 tokens/s on average (4 % slower, 9 fewer slots per layer). With fewer than half the experts of a layer in a slot, `stream_kick` copies the whole layer, 48.07 GiB per chunk (`q36_engine.cu:2826`): at chunk 2048 the read runs at the probed PCIe rate (calculated). The design note `docs/Q36_LAYER_MAJOR_PROMPT.md` weighs the next steps: giving back the slots the chunk buffers take, then a bigger default chunk, and a layer-major read only if the compute floor stays far below the copy time. The bigger default has shipped as a hybrid rule: with one session, the chunk is 4096 from 20 GiB free, 2048 from 7.5 GiB, 1024 from 5.5 GiB and 512 below (`hybrid_prompt_chunk`, `src/qwen35/q36_rules.hpp:36`, called at `src/qwen35/q36_engine.cu:5056`; `STRATA_Q36_CHUNK` still overrides it, `:5058`). With several sessions it stays 2048, because a chunk holds the other sessions' decode windows for its whole length. The verification series of 25 October 2026 ran on the test PC (RTX 5090, 128 GB RAM), new build against the previous one. Identity: the six parity prompts p00 to p05, forced greedy with full logits, `STRATA_Q36_FIXED_SLOTS=1 --expert-slots 52 --pcie-frac 0`: the old build at `--chunk 4096` and the new build at `--chunk auto` gave byte-identical dumps, 6 of 6. The new build's chunk rule line read "free 23.21 GiB (raw 19.41, with 3.80 packed), main experts 46.83 GiB, chunk 4096". The 62 slots, the count first planned, do not fit beside the 4096 chunk when the slots are given by hand (14,576 MiB needed, 13,645 free), because the packed Q8_0 copies are then kept. Server, `tools/ornith/measure_partition.ps1`, one launch per setting, alternated, three launches each, `STRATA_Q36_STATS=1`: the 34,257-token prompt was read in 9.37 s by default (3,656.6, 3,655.7 and 3,654.4 tokens/s) against 15.21 s with `STRATA_Q36_CHUNK=2048` (2,252.0, 2,251.9 and 2,251.9 tokens/s); 300 decode tokens after it ran at 30.5, 30.5 and 30.4 tokens/s against 30.7, 30.7 and 30.6. Experts per layer in VRAM: 67 by default against 76 at 2048 (engine line "67 instead of 47 per layer" and "76 instead of 56"). Long prompt, one launch each: the 128K request of an earlier run with its whitespace collapsed, 112,768 tokens, was read in 49.98 s (2,256.5 tokens/s) by default against 66.14 s (1,704.9 tokens/s) at 2048, 32 % faster; 128 decode tokens after it ran at 19.6 against 20.7 tokens/s. Short-context decode on the 256-token warm-up of those two launches: 44.8 against 47.0 tokens/s. The issue's criterion (chunk 8192 on the 127,685-token prompt) is not met, because `STRATA_Q36_CHUNK` and `--chunk` are capped at 4096. Issue #104 stays open.

Without the cap, one series run outside the protocol gave 43.2 tokens/s for the branch against 43.8 for the start at
1,024 tokens, and 18.8 against 17.2 at 128K. The memory guard was not armed in that run, free memory fell to 703 MB, and
the series does not count as a result. The first uncapped measurement under the protocol completed on 24 October 2026: 41.40 tokens/s for the default (see the uncapped series section).

## Protocols

Greedy identity. Two builds must produce the same tokens at `temperature 0`. The test runs six prompts (p00 to p05) for 32
tokens each, with `STRATA_Q36_FIXED_SLOTS=1 --pcie-frac 0 --expert-slots 62 --cpu-threads 6`. Without `FIXED_SLOTS`, the
placement of expert slots changes over the steps, and the output can change from one run to the next.

Logit determinism (issue #71). On the test PC, 40 forced greedy runs of p00 were made from the CLI (62 slots, chunk 2048, `--pcie-frac 0`, yarn scale 1), with the full logits dumped and each row compared bit for bit with the first run. This branch gave 40 of 40 identical runs with the CUDA graph, 40 of 40 with `--no-graph` and 40 of 40 with `--cpu-threads 16`. Main (ac27f23f) gave 40 of 40 with the graph. The divergence reported in #71 (one of eleven resumed runs, on branch feat/ornith-122b at e82d1fc3) did not reproduce in 160 runs, so its first differing step and its cause are not identified.

Server-side check for #71 (`tools/ornith/bench71_server.ps1`), run on 25 October 2026 on the test PC with main plus #96 (rev9), 8,192 cells per session (three sessions with 62 slots do not fit at 16,384: 14,384 MiB needed, 14,231 free). With the defaults, 5 of the 5 resumed runs under load differed from run 00, from the 21st, 40th or 127th character ("Thornwood Light for eleven years" against "Blackwood Light for thirty-one years"). With `STRATA_Q36_FIXED_SLOTS=1` and nothing else changed, 5 of 5 were identical to their run 00. A `-Trial` run of the script in its committed form diverged again at the 21st character. So the divergence follows the slot placement: the rebalances move experts between the VRAM slots and the CPU pool according to every session's routing (367 slot replacements after the prompt in one run), the two paths do not round an expert's output the same way, and the greedy choice flips when two logits are close. A single session repeats its own placement, which is why 160 CLI runs were all identical. The original divergence happened in the server while other sessions decoded; the CLI bench does not recreate that. The script starts one `serve/server.py` process (config with `"parallel": 3`, engine arguments `--expert-slots 62 --chunk 2048 --pcie-frac 0 --rope-scaling yarn --rope-scale 1`, plus `-ExtraArgs` and `-ExtraEnv`), keeps two greedy long-answer requests in flight (`-LoadTokens`, default 4096) and resumes p00 `-Runs` times (default 40, `-Trial` runs 2) in greedy decoding, each time with both loads decoding. Its start checks and its server stop live in `tools/ornith/ornith_server_common.ps1`, shared with `measure_partition.ps1`. It refuses to start when a `gitlab-runner*` VM is not Off, when `strata.exe`, `strata-q36.exe` or `llama-server` runs, when free RAM is under `-MinFreeGiB`, when the GPU holds more than `-MaxIdleGpuMiB`, when the port answers, when `-OutDir` exists or when a path parameter holds a double quote, and it stops the server by its process tree at the end and on any failure. Run: `powershell -File tools\ornith\bench71_server.ps1 -BuildDir <build> -OutDir <new folder> [-Trial]`. `status.txt` has one START, DONE and CMP line per run, then RESULT.

The server takes only chat messages (it templates and tokenizes itself, so `p00.ids` and the teacher forcing of `p00.force` cannot be sent) and returns text, `usage` counts and `finish_reason`, with no token ids and no logprobs. So the script compares text, not tokens: the answer text and `usage.completion_tokens` of each run against run 00, and it reports the first differing character. It sends p00's user message, checks that `p00.ids` is line 1 of `tools/qwen35/parity_prompts.ids` and that the server read as many prompt tokens as `p00.ids` holds, and asks for as many tokens as `p00.force` holds plus one. A SAME verdict shows that the greedy token choices were the same under load. It does not show that the logits were identical to the bit, which the CLI bench's full-logit dump does. A DIFF verdict is a real change in the server's greedy output. RESULT also counts the runs that had both loads decoding at their start and end.

Forced logits and KL. The engine receives the token sequence of llama.cpp b11756 and returns its logits at each step.
The comparison uses ten prompts of 32 steps and reports the mean KL divergence and the share of steps where the top-1
token agrees. The proposed bar, still to be validated, is a mean KL of 0.061, about 1.5 times the 0.041 of the T7 gate. Two passes on the same
build are identical to the byte. Between builds the mean KL moved by 0.0032.

Provenance of the KL (0.0423) and top-1 (299 of 320) figures. The reference is `q36_llama_dump` of llama.cpp b11756, on the CPU backend with 16 threads (`STRATA_DUMP_THREADS=16`) and without YaRN, on the same GGUF file. Ten prompts are tokenized with `llama-tokenize --no-bos`, and the reference decodes each one greedily and freely for 32 steps, keeping the full logits. The engine then receives the reference's 32-token greedy continuation through `--force`, so both sides read the same token at every step (teacher forcing on the reference's continuation). The engine runs with its default options and writes a full logit dump. `tools/qwen35/logit_cmp --ref <reference dump> --test <engine dump> --vocab 248320 --steps 32` compares them in default mode, without `--stop-at-divergence`. The KL is the mean over the ten prompts of each prompt's mean KL (0.042317), and the top-1 is counted over the 320 steps. The figures were first measured on 22 October 2026 at commit `93111239` plus the A2 changes. On 23 October the engine at `ec3d4b63`, then at `56f9fafc` (after the final review fixes), gave engine dumps byte-identical to each other on the ten prompts, so the figures hold at `56f9fafc`.

Throughput. Each measurement is several cold launches, alternated between builds, with four requests of 1,024 tokens per
launch. The reported figure is the median, with the range in brackets. The 128K figure is one launch per build with a
127,685-token prompt and 128 decoded tokens.

Working-set cap. `SetProcessWorkingSetSizeEx` sets a hard cap of 24 GB per `strata.exe` process (flags 6). The cap limits
the working set, not the commit. Without it, the working set of the process reached about 105 GB at the time of the stop, and was still rising, probably because it counts the
pages of the mapped GGUF file as well as the pinned experts (probable, not verified). Free memory then drops in under 30 seconds. A comparison
without the cap therefore measures memory pressure as much as the code, and every throughput comparison in this record
is made under the cap until 23 October.

Memory guard. A watcher samples free memory every few seconds. Below 8,192 MB it stops the process by its PID, never by
its name. Before each GPU window, `nvidia-smi` must show no foreign compute process and less than 200 MiB in use. Two
guard failures happened. In series 1 of the T8 run, 19 of the 21 watchers wrote nothing, because a leftover `watch.stop` file stopped them, and the two others had a filter that matched no `strata.exe`. Series 1 therefore ran without cap and without an effective guard. In a later attempt, memory fell faster than the
sampling interval, so the process was stopped at 4,272 MB, below the 8 GB floor. From 24 October 2026 the owner runs Ornith with no cap and no guard; the pre-start checks of the tool replace them (see the uncapped series section).

## Experiments, in order

### Porting (20 and 21 October)

The port added the hybrid mode to the q36 engine and quantized the model to IQ3_XXS for gate and up, IQ4_XS for down, with
the MTP block at Q4_K. The quantization ran for 30 minutes 33 seconds and produced a file of 3.74 bits per weight (quantization log). The KV cache
was moved to q4. A doorbell synchronizes the GPU and the host. The first identity and KL gate reached a KL of 0.0408, and
the MTP gate was not met. Raising the maximum PCIe share from 0.60 to 0.95 and routing MTP by phase came next. `--mtp` above
7 is refused with exit code 2, and the one-token CPU kernels are off by default. Three issues were opened during the port:
#71 (logits not identical to the bit; one of eleven resumed runs diverges), #72 (KV q4 above the bar) and #73 (a 99 % YaRN
gate that llama.cpp itself cannot meet).

### First server start (21 October)

The first server start, with `--max-context 409600 --kv q4 --mtp 2`, gave 15.4, 18.2, 19.7 and 20.1 tokens/s on four code (the server's timing reads 20.2)
requests and 9.2 tokens/s at 128K. The plan had assumed 19 GB of experts in VRAM. The engine placed 54 of 256 experts per
layer, 10.14 GiB. The goal of 100 tokens/s was missed by a factor of 5 to 10.

The gates of the next build, at commit `977dff70`, showed three things. The identity test failed without `FIXED_SLOTS`,
and passed with `STRATA_IQ_MT_MIN=2` and `STRATA_Q36_FIXED_SLOTS=1` on five of five prompts. The mean KL was 0.0652, above
the bar. And the test suite had five failures out of 140: fixtures absent on the box, and
two tests not built, which the report says predate the branch. With MTP on, decode fell to 7.5 to 8.5 tokens/s, against 18.1 to 20.9 without it. A speculative step
cost 133 ms against 50 ms for a plain step, and brought in about 1.5 tokens. The source report puts the cost at about four
plain steps, where the two timings give about 2.7. Acceptance was 52 to 63 %. The MTP
configuration was dropped from the deployed server on 23 October.

### Instrumentation and baseline (21 October)

Counters (`STRATA_Q36_STATS=1`) gave the per-token split: about 0.75 GB of experts missed per token, 391 MB over PCIe and
363 MB on the CPU at the best setting. The CPU pool ran at 18.9 to 20.7 GB/s with six worker threads, and slowed as threads
were added: 14 to 17 GB/s at eight, 10 to 15 GB/s at twelve and sixteen. Six threads gave the best result, 21.9 and 26.6
tokens/s. The plan had assumed 6.5 GB/s for the pool, which the measurement disproved. Sequential RAM read plateaued at
about 57 GB/s from four threads. A three-token window touches 79 % of the expert slots that three separate tokens would use, at best a 21 % saving. A token
took 37.7 ms: 16.6 ms in the pool, 8.9 ms waiting for the GPU. Large pages were not available: the process lacks
`SeLockMemoryPrivilege`.

### Identity bisection (21 October)

Four commits were built and compared to find where greedy output changed. `dfcfa856` and `0f4a3e90` match the reference.
`31e81c83` and `2cb7c736` differ under the three environment variables of the test. Only `STRATA_IQ_MT_MIN=2`, set explicitly and with fixed
slots, restored the reference, on five of five prompts. Unset, its default is also 2, but the `iq3s_one` branch then becomes active. Two causes were found:

- `STRATA_NO_IQ512=1` does not restore the old path. In the reference, the gate and up kernel for IQ3_XXS at more than one
  token ran on AVX-512; the variable moves it to AVX-2. `STRATA_NO_IQ4NL=1` removes the AVX-2 IQ4_NL kernel used at two
  tokens and more.
- The expert rebalancing changed. The branch made 1,514 swaps after prompt p00, against 1,973 in total for the reference, and left
  9,480 entries on the GPU against 9,763.

### Scheduling and copy experiments (21 October)

- Single-pass pool (one synchronization per layer instead of three barriers): 23.4 and 26.6 tokens/s against 23.4 and 26.9
  with three barriers. No measurable gain. The single-pass path was removed from the code.
- PcieAuto, the rule that sets the PCIe share from the measured cost of the CPU path: a host-only test of the new formula
  passed, with no throughput measurement at that stage.
- Prefetch of the next layer's experts and grouped copies: 15.2 and 24.0 tokens/s with prefetch off, 22.6 and 27.2 with it.
  About 47 % of prefetched blobs were lost. Identity held 6 of 6. The gain was not established, and a repeat was asked for
  but not run.

### Routing options, the cache prior and SER (21 October)

- `--cache-prior L` biases the router toward experts already in VRAM. With L 0.25 it reached 75.0 and 76.5 tokens/s, and with L 0.5 77.4 and 74.6, but KL was 0.6361 at L 0.25 and 0.9121 at L 0.5, and the text changed (`Interval = Tuple[int, int>`). Not usable.
- `--ser Kmin,t` drops low-probability picks. `7,0.4` gave KL 0.0864 and 16 % fewer bytes. `6,0.4` gave KL 0.1681.
  Both sit above the bar, and the speed gain stays within the run-to-run spread.
- Both stayed off by default.

### Pool and PCIe share (22 October)

The automatic share, set by the then-current rule, settled at 0.80 and gave 22.4 and 27.2 tokens/s. A fixed share of 0.65 gave
24.4 to 24.9 and 28.2 to 29.1. A sweep of fixed shares (issue #80) found the best values at 0.50 to 0.55, with 26.7 to 27.2
tokens/s on the first request. Feeding the measured wire time into the rule made it settle lower, near 0.40 to 0.45, below
that optimum. The copy streams moved about 21 to 25 GB/s, against the 57.8 GB/s of the link probe. A factor of 2.1 to 2.3
between the rule's formula and the measured cost remains unexplained.

The rule that was kept reads the measured wire time of the chosen blobs and settles at 0.45. On six alternated launches
it gave 24.1 to 24.7 tokens/s on the first request, against 22.4 to 22.8 for the 0.80 rule, and 27.5 to 28.5 against 26.5
to 27.4 on the second request. The first request's gain is clear. The second sits at the edge of the spread. The owner
accepted the 0.45 setting. The six servers ran with `STRATA_Q36_STATS=1`, so their counter overhead is included; the production setting without it has not been measured.

### MTP in hybrid mode (22 October)

With `STRATA_Q36_ONE_TOKEN_CPU=1`, `--mtp 2` and `--mtp 3` gave the same text as `--mtp 0` on six of six prompts. Without
the variable they diverged at step 6 on one prompt, as reported in issue #74. The variable itself changes the text without
MTP, so the identity holds only when both sides use the same environment. MTP at 2 drafts ran at 16.3 and 17.6 tokens/s, and at 3 drafts at 15.2 and 17.0
tokens/s, against 20.7 and 23.3 without it. The bytes missed per token did not fall with MTP (1.04 to 1.26 GB against 1.07
to 1.12 GB). The follow-up tasks A3 (doorbell) and A4 (MTP in hybrid) were cancelled on 22 October by owner decision. The installed configuration has no MTP.

### Cache prior, paper formula (22 October)

The cache prior of arXiv 2412.00099 (the Delta_avg variant) was implemented to match the paper. Its results on ten prompts:

| Setting | KL | Top-1 | Tokens/s, requests 1 and 2 |
|---|---|---|---|
| off | 0.0423 | 299/320 | 22.7, 27.5 |
| L 0.02 | 0.0707 | 298/320 | 23.1, 29.4 |
| L 0.05 | 0.1453 | 286/320 | 30.2, 35.4 |
| L 0.1 | 0.1893 | 278/320 | 32.5, 44.2 |

No value falls under the bar. The option stays off, and the follow-up task A3, which skipped the doorbell on layers without misses, was cancelled on 22 October by owner decision: without a usable bias, such layers are too rare to measure.
The first code tree of this task changed the output of the base (greedy identity 0 of 6 on that tree). Moving one kernel
launch into a member function restored identity. The cause was not established.

### Attention at long context (22 October)

Nsight Systems measured the attention cost per token at 2.42 ms at 1K, 6.82 ms at 32K and 18.93 ms at 128K. The out-of-context blocks of the grid of
`k_attn_part` (800 chunks) cost about 6 microseconds per instance at 1K, which is negligible. Kernel fusion was not worth
doing: the gap between kernels totals 242 microseconds per token, below a 0.7 ms threshold that the study's agent estimated, not one set by the plan. The cost at
128K is attributed to the per-head re-read of the K/V cache, an untested hypothesis. Only the slope of 5.6 microseconds per chunk is measured, as a linear fit on three points. Changing that loop changes the computation.

### Review of the branch (22 October)

Six read-only reviews covered the eleven commits from `7bf040a5` to `9ce40dca`. The security review found nothing above
the confidence threshold. The findings that were acted on: the PcieAuto launch count and the wire-time count used different
denominators; the cache prior kept its state at engine level instead of per session; the STATS counters were global under
`--batch` above 1. The fixes refuse `--cache-prior` at startup with `--batch` 2 or more, with `--mtp` N above 0, and off the fast routing path (coded, not run), print the
STATS line only for a single session, and move two tests out of the CUDA-only block. After the fixes, identity held on six
of six greedy prompts and ten of ten logit prompts.

### Kernels from ik_llama (22 October)

Reading the source of ik_llama's CPU MoE kernels (MIT licence, commit `c12b5ac` of the external ik_llama repository, not checked from this repository) shows that its IQ3_XXS path repacks
weights in R4 layout and its IQ4_XS path in R8 layout. The R8 path quantizes the activation to Q8_K32 instead of Q8_K,
which changes the bits of the output. A second copy of the repacked experts would cost about 50 GB of RAM (calculated).
No measurement was made. The recommended order is to measure the existing bench first, then the existing one-token path.

### Final measurement (22 and 23 October)

The final build, `ec3d4b63`, was checked against the start, `7bf040a5`:

- Identity: six of six greedy prompts identical across the final build, the base 9ce40dca, and the start.
  Logits identical on ten of ten prompts between the final build and 9ce40dca. The N5 build (task 5) was not identical to the base: none of six logit files matched.
- KL 0.0423, top-1 299 of 320 (93.4 %).
- Throughput under the cap: 30.0 tokens/s against 22.6, as in the result table.
- `STRATA_DMA_BATCH=1` against 0: 29.7 against 29.9 tokens/s, inside the spread. No verdict.
- `--pcie-frac 0.50` against the automatic share: 30.0 against 30.0. No difference.

Where a token's time goes. The counters were read on one launch at 30 tokens/s, and the per-token costs are calculated from them (33.3 ms per token):

| Cost | Time per token |
|---|---|
| CPU pool | 20.8 ms |
| Host waiting for the GPU | 9.3 ms |
| Host outside the pool | 3.1 ms |
| Sum | 33.2 ms |

Per token, 211 MB are copied over PCIe and 462 MB are read by the CPU pool, at 22.5 GB/s. Even with the pool at zero,
9.3 + 3.1 = 12.4 ms (calculated) exceeds the 10 ms that 100 tokens/s allows. Reaching 100 tokens/s would need the GPU wait
and the host cost to fall, not only the pool.

### Uncapped attempt (23 October)

A later attempt ran without the cap, under the guard. The process was stopped at 4,272 MB available, after 28.1 seconds of
falling memory, before it produced any token. No uncapped throughput existed at that date. The coordinator chose to stop and report; lifting the cap was the owner's decision. The series of 24 October are the first uncapped throughput measurements (41.40 tokens/s for the default, see the section below). The current uncapped throughput is the one of 25 October: 46.7 tokens/s with the new defaults, against 43.7 with the old ones on the same box (see the section New defaults below).

### Lower-bit expert quantization, IQ2_XXS (23 October)

The lower-bit candidate of the open leads (issue #76) was quantized and measured, then set aside. The BF16 source was downloaded again (revision 5514dc9e, checksum matched) and quantized with the importance matrix kept from the earlier work. Gate and up experts are IQ2_XXS, the block 48 gate and up are Q4_K, and down is IQ4_XS. The file is 48,577,355,616 bytes, 3.12 bits per weight over the file, and took 32 minutes on 16 threads.

Quality, same binary `ec3d4b63`, ten prompts of 32 steps against the llama.cpp reference:

| Quant | Mean KL | Top-1 |
|---|---|---|
| IQ2_XXS (new) | 0.4697 | 265 of 320 (82.8 %) |
| Current quant | 0.0423 | 299 of 320 (93.4 %) |

The bar of 0.061 is not met: the new quant is 7.7 times the bar. The worst prompt gives a KL of 1.84 and a top-1 of 19 of 32. The current quant reproduces the task A2 output byte for byte. This confirms the measurement chain.

Throughput, under the 24 GB working-set cap, four code requests of 1,024 tokens, three alternated launches per quant: 40.65 tokens/s (range 40.60 to 41.67) for the new quant, against 28.45 (27.80 to 28.85) for the current quant in the same session. The current quant measured 30.0 tokens/s on 22 and 23 October under the same protocol, so 28.45 is the same-session control.

Source of the gain. Missed expert bytes per token fall from 678 MB to 451 MB (minus 33 %). The pool time falls from 22.1 to 12.9 ms per token (calculated). The host wait on the GPU stays at 9.2 ms. Expert slots per layer in VRAM rise from 76 to 95. The plan had announced a lower bound of about 34 tokens/s without the slot gain; the measured figure sits above it thanks to the 19 extra slots. The quality run, launched with other settings, counted 104 slots per layer for the new quant and 83 for the current one.

Decision of the owner, 23 October 2026: IQ2_XXS is set aside and the current quant stays the default. The BF16 source is kept on the test box for a possible intermediate tier, for example gate and up in IQ2_S (2.5 bits per weight), which is not measured. The IQ2_XXS file stays on disk, unused.

### Uncapped series (24 October)

Measured on 24 October 2026 on the test PC (RTX 5090), with the binary of branch `fix-issues-simples`, build `E:\StrataGP-89\final`. It is the first throughput series run without the working-set cap. By owner decision, the series ran with no cap and no memory guard. The tool `tools/ornith/measure_partition.ps1` checks before each start: it refuses to start when a GitLab runner VM runs, when free RAM is under 100 GiB, or when the GPU holds more than 4096 MiB. During the run it only logs memory. Free memory fell to about 3.9 GiB in every launch (3.88 to 3.95 GiB), and no process was stopped.

Each series follows the same protocol. The summary is produced by `tools/ornith/partition_summary.py`. Each launch is a cold server start with `STRATA_Q36_STATS=1`. Each launch runs one code request, "Write a Python function that parses an ISO 8601 date-time string without the standard library, then explain each step in a short paragraph and give three unit tests.", for 300 decoded tokens, after a 256-token warm-up request. The figures are medians over the measured launches.

| Series | Launches | Tokens/s | ms per token | Pool / wait / host outside pool (ms) | PCIe copies per step (ms) | Inter-token p95 (ms) |
|---|---|---|---|---|---|---|
| Default (PCIe share 0.15) | 3 | 41.40 | 24.13 | 15.06 / 7.63 / 1.43 | 2.60 | 30.70 |
| Rebalance every 8 steps (r08) | 2 | 42.85 | 23.35 | 14.26 / 7.62 / 1.47 | 1.54 | 29.40 |
| Rebalance every 16 steps (r16) | 2 | 41.55 | 24.06 | 15.01 / 7.61 / 1.45 | 2.01 | 30.00 |
| Fixed PCIe share 0.50 (pf050) | 2 | 35.80 | 27.93 | 9.86 / 9.59 / 8.49 | 19.54 | 36.45 |
| Fixed PCIe share 0.55 (pf055) | 2 | 35.75 | 27.97 | 9.83 / 9.58 / 8.56 | 19.59 | 36.55 |
| Automatic PCIe step (pauto) | 2 | 35.10 | 28.47 | 9.97 / 9.67 / 8.83 | 26.26 | 38.15 |

The PCIe copies overlap the rest of each decode step. These counters are per request: pool, wait and host are the request's totals over its tokens, and the PCIe figure is the busy time of the copy windows over its decode steps. They do not say which stage made a slow token slow. The engine now also prints a decode steps line with the same partition per decode step (docs/QWEN36.md, `STRATA_Q36_STATS`), and `tools/ornith/partition_summary.py` uses it to attribute each token above the p95 to a stage. The first series with that line ran on 25 October 2026 on the same test PC, with the binary of this branch before the new defaults (probed PCIe share 0.55, rebalance every 32), `--mtp 0`, one 300-token request per launch and 256 warm-up tokens. Both launches gave 43.6 tokens/s. The p95 inter-token time was 30.0 ms, and 13 tokens lay above it; all 13 matched the first token of a decode step. In the median step the CPU pool took 14.3 ms, the wait on the doorbell 7.5 ms, the host 1.3 ms, and the PCIe copies 2.0 ms, which overlap the rest. The dominant stage of the 13 slow steps was the CPU pool in 11 of them, PCIe in 2, the GPU wait in none and the host in none. The slow steps spent 68 % of their time in the pool, 26 % in the wait and 5 % in the host, and their PCIe copies took 22 % of their time, overlapping the rest. Against the median step, the excess was 75.7 ms in the pool (54 %), 57.9 ms in the PCIe copies (41 %), 3.8 ms in the wait (3 %) and 3.0 ms in the host (2 %). The reading, an interpretation of these figures: the slow tokens are slow because of the CPU pool, and the PCIe copies come second. With the PCIe path now off by default, that second cause is gone. In the default series, the decode rebalances made 72 slot swaps, copied 293.8 MB and cost 9.3 ms of host time. Rebalancing every 8 steps was the only setting that beat the default: 304 swaps, 1240.5 MB copied and 36.8 ms of host time, for 42.85 tokens/s, or 3.5 % more over two launches. Rebalancing every 16 steps made 152 swaps, copied 620.2 MB and cost 18.8 ms of host time.

The automatic step started at 0.55. It saw four noisy windows of 36 to 41 ms and stayed at 0.55. It cannot go below 0.30. It stays off by default.

Parity: six greedy prompts in fixed slots, 128 tokens each, give identical content on this branch and on main. The single launches ran at 37.70 tokens/s on the branch and 36.10 on main, one launch each, which is within noise.

Reading (an interpretation of the figures above, not a separate measurement): without the cap, a larger PCIe share is slower. The default 0.15 beats 0.50 and 0.55 by about 5.6 tokens/s, because the host time outside the pool rises from 1.4 to 8.5 ms per token. The fixed optimum of 0.50 to 0.55 found on 22 October was measured under the 24 GB cap and does not carry over. Rebalancing every 8 steps is the only setting that beats the default, and only by 3.5 % over two launches.

### New defaults (owner decision, 25 October 2026)

Two defaults changed from these measurements, taken on the same test PC (RTX 5090, 128 GB RAM, Ryzen 9 9950X3D), decode, one 300-token prompt, three launches per setting, alternated. The table is the decision series: measured on 25 October, before this branch's code existed, with the settings passed by option and environment variable.

| Setting | Tokens/s |
|---|---|
| Old defaults (rebalance every 32, probed PCIe share) | 42.2 |
| Rebalance every 8 | 45.4 |
| `--pcie-frac 0` | 44.9 |
| Both | 46.5 |
| Any `--pcie-frac` from 0.05 to 0.30 | about 42.5 |

The PCIe path costs as soon as it is active, whatever the share. So the rebalance cadence default is now 8 (`STRATA_Q36_REBALANCE_EVERY`, bounds unchanged), and without `--pcie-frac` the PCIe share is 0: no probe at load, the path off. `--pcie-frac F` (F > 0) works as before. The former automatic share (probe, PcieAuto, and `STRATA_Q36_PCIE_AUTO_STEP=1` on top) now needs `--pcie-frac auto`; `STRATA_Q36_PCIE_AUTO_STEP=1` without it prints that it was ignored. The engine's load line says which share is used and why. The earlier sections of this record describe the old defaults. The check, on the test PC the same day with this branch's binary (not the table's series), three launches alternated per setting: the new defaults gave 46.7, 46.7 and 46.6 tokens/s, against 43.7, 43.7 and 43.6 for the old defaults (`--pcie-frac auto` and `STRATA_Q36_REBALANCE_EVERY=32`), which is 6.9 % more. The load line of the new defaults reads: "PCIe share off (the default: --pcie-frac F or --pcie-frac auto turns it on)". With `--pcie-frac auto`, the probe measured 57.7 GB/s and set the share to 0.55.

Greedy text is not identical with the new defaults. With slots moving, `--mtp 2` and `--mtp 0` do not give the same text on the six parity prompts: the first difference comes after 15 to 622 characters. The new defaults and the fixed-slot setting at `--mtp 0` differ on five of six prompts. All the texts read as coherent answers, and the differences are word choices. The likely cause, an interpretation not measured here: with moving slots, the slot assignment depends on the routing counts, which include the drafts of a verify window, so which experts run on the GPU and which on the CPU changes, and the two round differently. Identity holds only with fixed slots.

The full ctest of this build passed 148 of 151 on the test PC (MSVC, CUDA). The three failures (ple_parity, expert_parity and pool_test) fail on main too, and isa_guard_check was skipped.

### Attention kernel durations (25 October)

Measured on 25 October 2026 with Nsight Systems, kernel durations only, in bench mode with `--max-context 409600` and KV q4. The median `k_attn_part` per call is 199 µs at 1,024 tokens of context, 562 µs at 37,159 and 1,565 µs at 127,685. There are 12 calls per token, so the attention kernels take about 2.4 ms per token at 1K and 18.8 ms at 128K (calculated). The 22 October measurement gave 2.42 ms and 18.93 ms, so the two agree. The slope is about 10 to 11 µs per thousand tokens of context per call.

At 1,024 tokens with `--max-context 4096`, the median falls to 98 µs. That run also halves the chunk size (`chunk_tiles` 2 to 1), so it does not isolate the grid. The 22 October measurement already put a bounded grid at 6 µs per call, so most of the 100 µs gap comes from the chunk size (a deduction, not an isolated measurement). A smaller chunk changes how the sum is split, and so the output bits, which issue #81 rules out.

The per-head re-read of the K/V cache is still a hypothesis. Testing it needs read counters. Nsight Compute cannot provide them on this box (see the operational pitfalls). No kernel change has been made.

## Operational pitfalls

- A process started with `Start-Process` from an SSH session dies with the session. Start it detached instead.
- A command passed through `cmd.exe` over SSH breaks quoting and pipes. Use an encoded PowerShell command.
- A launcher copied from another experiment directory can overwrite that experiment's logs. Check paths before running.
- A stop file left from an earlier run closes a new watcher at once.
- A filter on a nested environment variable can match no process at all. The guard then does nothing.
- A background deletion can overwrite its own log. Verify in the foreground.
- Working set is not memory use: pages mapped from a file count toward it.
- Compare builds only under the same environment variables and the same `FIXED_SLOTS` setting: the variable itself changes the output.
- The thread count reported by the engine is one more than `--cpu-threads`.
- `--mtp` above 7 is refused at startup with exit code 2.
- Nsight Compute on the hybrid engine deadlocks it. It serializes the kernels while the host spins on a GPU doorbell (probable cause), and it can leave the GPU at 100 % with no process until a reboot. `nvidia-smi -rgc` does not clear it.
- Per-process GPU memory reads `[N/A]` under WDDM.

## Open leads, ranked

Estimated gains come from calculations in the sources, not from measurements on the box.

1. Lower-bit expert quantization, issue #76. Measured on 23 October and set aside by the owner. Gate and up in IQ2_XXS, down in IQ4_XS: KL 0.4697 against the 0.061 bar (current quant 0.0423), top-1 265 of 320 against 299. Throughput 40.65 tokens/s against 28.45 in the same session, with 95 expert slots per layer in VRAM against 76. The BF16 source is kept for a possible intermediate tier (gate and up in IQ2_S, 2.5 bits per weight), not measured. See the IQ2_XXS section above.
2. ik_llama CPU kernels, issue #77. No gain measured. The estimated ceiling of 23 to 29 tokens/s rests on a 6.5 GB/s pool, since measured at 20 to 22.5 GB/s, so it no longer applies. Cost: repacking, with about 50 GB of second copy in RAM, which needs the owner's approval. Quality risk: the R8 path changes output bits. First measurement: the one-token bench at six workers, then the existing one-token path.
3. Host memory, issue #82. No gain estimated. About 28 GB of private memory remains unidentified (calculated: 79.5 GB of private memory less 51.6 GB of pinned experts). First measurement: find its origin before any engine change.
4. NVMe tier, issue #84. The issue estimates no gain, because about 48 GiB of experts stay pinned in RAM and a storage tier does not lift the rate while they do. First step: confirm the architecture and the ceiling with the NVMe share, then decide.
5. MTP, issue #74. A speculative step costs 133 ms against 50 ms for a plain step, for about 1.5 tokens. The divergence came from the float order of the AVX-512 multi-token kernel for IQ3_XXS gate and up rows, which differed from ggml-cpu's dot; the ggml-order kernel removes it. It has been on by default since 25 October: with `--mtp 2` and fixed slots it gives the text of `--mtp 0` on six of six parity prompts, against one of six on main, and it runs about 3.3 % faster (49.0, 49.1 and 49.0 tokens/s against 47.6, 47.4 and 47.4 with `STRATA_NO_IQ512_GGML_ORDER=1`). The owner chose to keep it on, an exception to the opt-in rule of AGENTS.md. The parity holds for IQ3_XXS gate and up rows (Ornith's quant); an IQ4_NL down row still takes the AVX-2 group kernel against ggml for one token (the #152 rule), so that gap remains. The ggml-order kernel matches ggml-cpu only while the pinned ggml-cpu uses its AVX2 dot for IQ3_XXS even in an AVX-512 build (pin at `CMakeLists.txt:1538`, `GIT_TAG 3cf03257`), and `iq512_ggml_order_test` checks it on the build it runs on. The step cost above predates this kernel and has not been measured again.
6. Speculative decoding without MTP, issue #78 (AcceptMoE, Specter). Deferred. The gain depends on the union of experts across a verification window, where a two-token window touched 0.85 to 0.89 of the slots that two separate tokens would use. The published gains do not transfer without that measurement.
7. Attention per head, issue #81. No gain estimated. The durations of 25 October (see the section above) put the slope at 10 to 11 µs per thousand tokens of context per call, which gives about 18.8 ms of attention per token at 128K (calculated). The cache re-read per head remains a hypothesis: it needs read counters, which Nsight Compute cannot give on this box. No kernel change has been made.
8. PCIe share, issue #80. Under the 24 GB cap, the automatic share settled at 0.45, and the fixed optimum was 0.50 to 0.55 (22 October). Without the cap, measured on 24 October, the default share of 0.15 gives 41.40 tokens/s. Fixed 0.50 and 0.55 give 35.80 and 35.75 tokens/s, about 5.6 tokens/s less, and the automatic step gives 35.10, about 6.3 tokens/s less. A larger share raises the host time outside the pool from 1.4 to 8.5 ms per token. The optimum found under the cap does not carry over. The automatic step is off unless `--pcie-frac auto` is given; since 25 October the PCIe path itself is off by default (see "New defaults").
9. CPU pool, issue #79. The recommended path works without large pages, which need a privilege the box does not grant. First measurement: busy time against wall time per thread.
10. Expert pruning (REAP), issue #75. Not measured on Ornith. The published figure is 95.8 % kept on BoolQ alone at 20 % pruning, on another model. First measurement: KL on the ten prompts.
11. KV precision, issue #72. Accepted by the owner on 23 October 2026: KL 0.079 on the short prompts against the 0.061 bar, with `--kv q4`. The measurements are in the Ornith section of QWEN36.md.
12. YaRN gate, issue #73. A 99 % top-1 target cannot be reached even by llama.cpp (91.6 %). The owner must choose a criterion.
13. Q2_0 on CPU, issue #70. Fixed on branch `fix-issues-simples`: `hy_cpu_compute` now fills the ActQ that the Q2_0 kernels read. A Q2_0 pack whose n_embd exceeds 2560 is refused at load, so Ornith (3072) stays off native Q2_0 while the ActQ keeps its size. On the Windows test PC (MSVC 2022, CUDA 13.4, sm_120), `q36_engine.cu` compiles once it includes `strata/kernels/cpu/expert_layout.hpp`, `native_q2_act_test` passes on its AVX2 path, and its `--hy-job` mode, which builds the job without an ActQ as the old code did, crashes with an access violation. No q36 model was run with a Q2_0 pack.
14. Rebalance cadence, issue #96. Measured uncapped on 24 October 2026: rebalancing every 8 steps gives 42.85 tokens/s against 41.40 for the default, which is +3.5 % over two launches. Every 16 steps gives 41.55 tokens/s. Measured again on 25 October 2026 on the test PC with the rebalance byte counters (`STRATA_Q36_STATS=1`, defaults otherwise, PCIe path off, one 300-token request after 256 warm-up tokens, three launches per cadence alternated): every 8 steps gave 46.9, 46.5 and 46.7 tokens/s, every 16 gave 45.7, 45.5 and 45.7, every 32 gave 45.5, 45.5 and 45.6. Of the 114,816 expert picks of a request, the VRAM slots served 57.9 % at 8, 56.1 % at 16 and 55.8 % at 32, the CPU pool the rest and PCIe none: the 8-step cadence moves 2.1 % of the picks from the pool to VRAM against 32. Its cost per request: 304 slot swaps, 1240.2 MB of expert blobs and 287.7 KB of slot tables copied, 36.8 to 37.8 ms of host time, about 0.6 % of the 6.4 s decode; at 32, 72 swaps, 293.7 MB, 67.6 KB and 9.3 to 9.5 ms. Net, 8 is 2.6 % faster than 32 after that cost, so it stays the default (go).

## Resuming

- Branch `feat/ornith-122b` of this repository. Final measurement commit `ec3d4b63`, starting point `7bf040a5`, installed source `d9d6f634`. Pull request #83.
- Parity tools are under `tools/qwen35/`: `parity.py`, `logit_cmp.cpp`, `q36_tokfile.cpp`, `q36_llama_dump.cpp`.
- `STRATA_DUMP_THREADS` sets the CPU threads of the llama.cpp reference dump (`q36_llama_dump`). llama.cpp's default is 4; the reference dumps used 16.
- `STRATA_DUMP_YARN=FACTOR,ORIG_CTX` sets YaRN in that dump, the reference for `--rope-scaling yarn --rope-scale FACTOR --rope-orig-ctx ORIG_CTX`. Its other YaRN parameters stay at the model's and the library's defaults.
- The option list, the environment variables and the server settings are in the Ornith section of [QWEN36.md](QWEN36.md).
- The raw reports and logs of the measurements are kept in the reports folder of the test box, outside this repository: 2,910 files, copied and checked byte for byte.
