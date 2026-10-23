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

The branch is faster than its start, and only under the 24 GB working-set cap (see below). The N5 build (task 5) gave a KL of 0.0455 and 299 of 320; the start was not measured on this test. The target of 100 tokens/s is
not reached. By owner decision of 23 October 2026, there is no fixed throughput threshold: each result is judged case by case.

Without the cap, one series run outside the protocol gave 43.2 tokens/s for the branch against 43.8 for the start at
1,024 tokens, and 18.8 against 17.2 at 128K. The memory guard was not armed in that run, free memory fell to 703 MB, and
the series does not count as a result. No uncapped measurement under the protocol has completed.

## Protocols

Greedy identity. Two builds must produce the same tokens at `temperature 0`. The test runs six prompts (p00 to p05) for 32
tokens each, with `STRATA_Q36_FIXED_SLOTS=1 --pcie-frac 0 --expert-slots 62 --cpu-threads 6`. Without `FIXED_SLOTS`, the
placement of expert slots changes over the steps, and the output can change from one run to the next.

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
is made under the cap.

Memory guard. A watcher samples free memory every few seconds. Below 8,192 MB it stops the process by its PID, never by
its name. Before each GPU window, `nvidia-smi` must show no foreign compute process and less than 200 MiB in use. Two
guard failures happened. In series 1 of the T8 run, 19 of the 21 watchers wrote nothing, because a leftover `watch.stop` file stopped them, and the two others had a filter that matched no `strata.exe`. Series 1 therefore ran without cap and without an effective guard. In a later attempt, memory fell faster than the
sampling interval, so the process was stopped at 4,272 MB, below the 8 GB floor.

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
falling memory, before it produced any token. No uncapped throughput exists. The coordinator chose to stop and report; lifting the cap is the owner's decision.

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
## Open leads, ranked

Estimated gains come from calculations in the sources, not from measurements on the box.

1. Lower-bit expert quantization, issue #76. Measured on 23 October and set aside by the owner. Gate and up in IQ2_XXS, down in IQ4_XS: KL 0.4697 against the 0.061 bar (current quant 0.0423), top-1 265 of 320 against 299. Throughput 40.65 tokens/s against 28.45 in the same session, with 95 expert slots per layer in VRAM against 76. The BF16 source is kept for a possible intermediate tier (gate and up in IQ2_S, 2.5 bits per weight), not measured. See the IQ2_XXS section above.
2. ik_llama CPU kernels, issue #77. No gain measured. The estimated ceiling of 23 to 29 tokens/s rests on a 6.5 GB/s pool, since measured at 20 to 22.5 GB/s, so it no longer applies. Cost: repacking, with about 50 GB of second copy in RAM, which needs the owner's approval. Quality risk: the R8 path changes output bits. First measurement: the one-token bench at six workers, then the existing one-token path.
3. Host memory, issue #82. No gain estimated. About 28 GB of private memory remains unidentified (calculated: 79.5 GB of private memory less 51.6 GB of pinned experts). First measurement: find its origin before any engine change.
4. NVMe tier, issue #84. The issue estimates no gain, because about 48 GiB of experts stay pinned in RAM and a storage tier does not lift the rate while they do. First step: confirm the architecture and the ceiling with the NVMe share, then decide.
5. MTP, issue #74. A speculative step costs 133 ms against 50 ms for a plain step, for about 1.5 tokens. The cause of the divergence is not proven. First measurement: profile the multi-row path of the hybrid MoE step and the draft block's expert requests.
6. Speculative decoding without MTP, issue #78 (AcceptMoE, Specter). Deferred. The gain depends on the union of experts across a verification window, where a two-token window touched 0.85 to 0.89 of the slots that two separate tokens would use. The published gains do not transfer without that measurement.
7. Attention per head, issue #81. No gain estimated. At 128K, attention takes 18.93 ms per token. The cache re-read per head is a hypothesis, not checked. First measurement: the slope of `k_attn_part` across 2K, 32K and 128K.
8. PCIe share, issue #80. The automatic share settles at 0.45, while the fixed optimum is 0.50 to 0.55. The gap is within the noise on the capped median of the final series (30.0 against 30.0). On the first request, it is 2.5 tokens/s, outside the spreads. First measurement: three alternated automatic launches against three at 0.55.
9. CPU pool, issue #79. The recommended path works without large pages, which need a privilege the box does not grant. First measurement: busy time against wall time per thread.
10. Expert pruning (REAP), issue #75. Not measured on Ornith. The published figure is 95.8 % kept on BoolQ alone at 20 % pruning, on another model. First measurement: KL on the ten prompts.
11. KV precision, issue #72. Accepted by the owner on 23 October 2026: KL 0.079 on the short prompts against the 0.061 bar, with `--kv q4`. The measurements are in the Ornith section of QWEN36.md.
12. YaRN gate, issue #73. A 99 % top-1 target cannot be reached even by llama.cpp (91.6 %). The owner must choose a criterion.
13. Q2_0 on CPU, issue #70. Fixed on branch `fix-issues-simples`: `hy_cpu_compute` now fills the ActQ that the Q2_0 kernels read. A Q2_0 pack whose n_embd exceeds 2560 is refused at load, so Ornith (3072) stays off native Q2_0 while the ActQ keeps its size. The fix is not yet proven on an AVX2 CPU and not compiled under CUDA.

## Resuming

- Branch `feat/ornith-122b` of this repository. Final measurement commit `ec3d4b63`, starting point `7bf040a5`, installed source `d9d6f634`. Pull request #83.
- Parity tools are under `tools/qwen35/`: `parity.py`, `logit_cmp.cpp`, `q36_tokfile.cpp`, `q36_llama_dump.cpp`.
- `STRATA_DUMP_THREADS` sets the CPU threads of the llama.cpp reference dump (`q36_llama_dump`). llama.cpp's default is 4; the reference dumps used 16.
- `STRATA_DUMP_YARN=FACTOR,ORIG_CTX` sets YaRN in that dump, the reference for `--rope-scaling yarn --rope-scale FACTOR --rope-orig-ctx ORIG_CTX`. Its other YaRN parameters stay at the model's and the library's defaults.
- The option list, the environment variables and the server settings are in the Ornith section of [QWEN36.md](QWEN36.md).
- The raw reports and logs of the measurements are kept in the reports folder of the test box, outside this repository: 2,910 files, copied and checked byte for byte.
