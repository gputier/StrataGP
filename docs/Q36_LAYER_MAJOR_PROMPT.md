# Design note: a layer-major prompt read for the q36 hybrid engine (issue #104)

Status: design only, no code changed. All line numbers refer to `src/qwen35/q36_engine.cu` in this tree unless another
file is named. Measured figures are the ones of issue #104 (25 October 2026, test PC: RTX 5090 32 GB, PCIe 5.0 x16, probe
57.7 GB/s, Ryzen 9 9950X3D, 128 GB RAM, Ornith-1.5-122B, one 34,257-token prompt, 3 alternated launches each). Every other
number here is calculated, and says so.

## 0. A correction to the byte count of the issue

The issue counts the non-resident experts only (33.6 GiB per chunk at chunk 2048, 35.3 GiB at 4096). The code copies more.
`stream_kick` (2814) copies the whole layer from host memory when fewer than half the experts of the layer have a slot:
`if (all || 2 * L.n_slots < ne)` (2826), then one `cudaMemcpyAsync` of `L.stride * ne` bytes (2827). The comment above it
(2824 to 2825) says why: on an RTX 3060 at 37 of 256 the one big copy was faster than many small ones. Ornith has 77 slots
(chunk 2048) or 68 slots (chunk 4096) of 256, both under 128, so every chunk copies all 48.07 GiB of experts, the resident
ones included. The run-by-run path (2828 to 2844), which copies only the experts without a slot over PCIe, does not run.

Calculated with this count (57.7 GB/s = 53.74 GiB/s; 48.07 GiB / 48 layers = 1.0015 GiB per layer, 18.64 ms per layer):

| Chunk | Chunks for 34,257 tokens | Bytes over PCIe | Copy time at the probe rate | Measured read | Effective rate |
|---|---|---|---|---|---|
| 2048 | 17 | 817.2 GiB | 15.21 s | 15.21 s | 57.7 GB/s (the probe rate) |
| 4096 | 9 (the last one 1,489 tokens) | 432.6 GiB | 8.05 s | 9.37 s | 49.6 GB/s |

At chunk 2048 the read runs at the probe rate to within the rounding of the figures: it is PCIe-bound and the compute is
hidden behind the copies. At chunk 4096 it is 1.32 s slower than its copies, so some compute is visible. This changes the
conclusion of the issue for chunk 4096: that read is no longer purely PCIe-bound. Assumption to check: the MTP block is
not loaded (`mtp_d` 0; the installed Ornith configuration has no MTP, docs/ORNITH_122B.md:146), so `forward_mtp` (3355
to 3363) streams no extra layer.

## 1. The compute floor (assumption stated)

The double-buffered stream overlaps the copy of layer i + 2 with the compute of layer i: `stream_done` records `ev_used`
and kicks the copy two layers ahead (2857 to 2862), `stream_take` waits for the copy of the layer it is about to use
(2848 to 2856). So a layer step costs about max(copy, compute), not their sum.

Model A (the assumption used below): every (layer, chunk) step costs max(copy of one layer, compute of that step), the
compute per token is the same in every layer and at chunk sizes from 2048 to 4096. Fitted on the chunk 4096 run:
8 full chunks x 48 steps x c + 48 steps x max(18.64 ms, c x 1489 / 4096) = 9.37 s gives c = 22.07 ms per layer per
4096-token chunk, and a compute total of 8.86 s for the prompt (258.6 us per token, a ceiling of about 3,870 tok/s).
Check on the other run: at chunk 2048 the same rate gives 11.0 ms per step, under the 18.64 ms copy, so the read would be
copy-bound at exactly 15.21 s, which is what was measured. One fitted value, one independent point that agrees.

What the data alone allow: the compute total is at least 1.32 s (the time the 4096 read spends beyond its copies) and at
most about 8.9 s (model A; if the layers differ in cost, for example the 12 attention layers costing more at long
context, the total can be lower). Model A is the only one of the two that predicts the 2048 result without a second
parameter. A rough cross-check, not a measurement: the R9700 reads Qwen3.6-35B-A3B at 5,750 tok/s with every expert
resident (5032 to 5033); Ornith has about three times the active parameters, so a few thousand tok/s on a 5090 is the
expected order. The floor must be measured before the large change below is written (section 6, step 1).

## 2. Where the per-chunk forward is, and what a layer step needs

The loop today (chunk-major):

- `process()` (3317) reads a prompt in chunks of `tmax` tokens: `for (int64_t i = 0; i < n; i += tmax)` (3347). Per
  chunk it syncs the stream (3349), uploads the tokens and the first position (`hpin[1]`, 3350 to 3352), sets the picture
  rows (`chunk_pictures`, 3353), runs `forward(T, pos, ...)` (3354), the MTP block when drafts are on (3355 to 3363),
  keeps the last row's normed hidden in `hlast` (3364), advances `pos` (3365), syncs (3366), and calls `on_chunk` (3371).
  The rebalance after the prompt runs at the end (3376) or at a cancel (3372).
- `forward()` (2721) starts the expert stream over every hybrid layer (2724 to 2729), embeds the chunk into `x` (2730),
  norms it into `h` (2732), runs the 48 layers (2733 to 2737), and runs the head on the last row of the last chunk only
  (`heads` is 1 only when `i + T == n`, 3354; head at 2738).
- `layer()` (2497) is where the experts of a hybrid layer are used: the routing counts go to the host with a stream sync
  (2669 to 2673), `stream_take` gives the staged layer (2675), MMQ runs gate/up and down on it, and `stream_done` frees the
  buffer and kicks the next copy (2689).

The loop for layer-major: for each layer l, for each chunk c (the same partition as today), run `layer(l, c)`; the
embedding is a pass over all chunks before layer 0, the head is one call after the last layer on the last row. The
stream visits each hybrid layer once per prompt: `stream_take` at the first chunk of layer l, `stream_done` after its
last chunk. `stream_take` checks the order (`stream_seq[stream_pos] != li` exits, 2849 to 2852), so the take and done
calls must move out of the per-chunk `layer()` call or become idempotent within a layer.

State each (layer, chunk) step needs:

| State | Where it lives today | Layer-major |
|---|---|---|
| Residual `x` and normed input `h` of the chunk's rows | Engine buffers of `tmax` rows (1904); `moe_end` writes the next layer's `h` (2715, via `k_moe_out`) | Must persist for every row of the prompt between layer steps: a whole-prompt buffer (below) |
| GDN conv and recurrent state | Per session, per layer: `conv[li]`, `S[li]` (`k_conv_seq` at 2534, `k_gdn_seq_cols` at 2536) | Unchanged: within a layer the chunks still run in order, so chunk c + 1 finds the state chunk c left |
| KV writes of attention layers | `attn_prep` writes the chunk's cells into `kc[li]` / `vc[li]` (2559) | Unchanged: per layer |
| Attention over earlier chunks of the same layer | The prompt attention reads cells 0 .. p0 + T of `kc[li]` (2569, gathered by `kv_f16` at 2579) | Unchanged: those cells were written by the earlier chunks of the same layer, in both orders; later chunks of the layer are not yet written in either order |
| Position | One device scalar `pos0`, uploaded per chunk from `hpin[1]` (3350 to 3352) | A small device array of chunk start positions; `layer()` already takes the position pointer as a parameter (`dpos`, 2497) |
| Picture rows and M-RoPE table | `chunk_pictures` per chunk (3295 to 3311): `map_on` is used at the embedding only (2731), `mtab_now` by every attention layer | `map_on` in the embedding pass; `mtab_now` set per chunk inside every layer step |
| Cross-layer quantized input | For T <= `kGemmT` (8, at 2142) `moe_end` leaves the next layer's q8_1 image in `xq` and sets `xq_ready` (2468 to 2472); `prep_in` of the next layer uses it (2289 to 2295) | A tail chunk of 8 rows or fewer would find `xq` overwritten by the other chunks of the next layer. Keep that chunk's `xq` image beside its `h`, or check that `prep` gives the same bytes (VERIFY compares them, `verify_q8` at 2474) |
| Tail chunk of <= `hybrid_t` rows (128 by default, 4886 and 3840) | Takes the hybrid rows path (slots plus the CPU pool, 2648), not the stream | Same path, inside the layer step; the staged layer is just held a little longer |
| Routing counts for the rebalance | Added per layer per chunk (2673) | Same additions in another order; the counts are integer-valued doubles, so the sums are exact and identical |
| MTP block (only with `--mtp` N > 0) | Per chunk after the main forward, seeded by `hlast` and the chunk's `hmain` rows (3355 to 3363) | A 49th layer pass over all chunks after the last main layer; needs the final normed hidden of every prompt row (the whole-prompt `h` buffer after the last layer), and streams the MTP experts once |
| Final norm and lm_head | The last layer's `moe_end` writes the final norm into `hmain` (2735, `out_norm`); `head_rows` on the last row (2738) | Same, for the last chunk only; `hlast` from its last row (3364) |

Where the hidden states of the whole prompt live. Per token the persistent state is `x` and `h` in fp32: 2 x 3,072 x 4 B =
24 KiB (n_embd 3,072, docs/ORNITH_122B.md:314). Keeping both is the bitwise-safe choice: `h` comes out of the fused
`k_moe_out` (2470, 2481), and recomputing it from `x` with a separate `k_rms` may round differently.

| Prompt | x and h, fp32 | As expert slots (one slot in every layer = 48 x 4.006 MiB = 192 MiB) |
|---|---|---|
| 32,768 tokens | 768 MiB | 4 slots per layer |
| 34,257 tokens | 803 MiB | 4.2 slots per layer |
| 131,072 tokens | 3.0 GiB | 16 slots per layer |

VRAM is the simple place. Pinned host memory is possible: each layer step reads and writes x and h, 48 KiB per token per
layer, so 75 GiB over the 34K prompt (37.6 GiB each way). That is less than one chunk-major read of chunk 4096 moves
(432.6 GiB) but about as much as the experts themselves (48.07 GiB), and it shares the link. The bounded form is a
super-chunk: layer-major over S tokens at a time (for example S = 32,768, a fixed 768 MiB), chunk-major between
super-chunks. The experts then cross PCIe ceil(n / S) times: twice at 34K, four times at 128K (192 GiB, 3.6 s at the
probe rate, calculated, hidden behind compute under model A).

The whole-prompt buffers must be separate allocations, not the engine's `x` / `h`: the decode windows that run between
prompt steps in serve mode use those buffers (section 3), and the captured graphs hold the engine's buffer pointers as
kernel arguments (1465, 3239 to 3240), so swapping the members would let a graph captured mid-read keep a wrong pointer.
Copying the chunk's rows in and out of `x` / `h` around each step costs 4 copies of T x 12 KiB per step: about 75 GiB of
device-to-device traffic for the 34K prompt, under 0.1 s at the 5090's VRAM bandwidth (calculated).

The staging region keeps its size. It is two whole layers: `max_layer + max_main` plus tails (3755, allocated at 3971
to 3976), about 2.0 GiB on Ornith. Layer-major holds one staged layer for all chunks of the layer and prefetches the next
into the other buffer; a layer copy (18.64 ms) is much shorter than a layer's compute over the whole prompt (about 0.18 s
under model A), so the two buffers hide it.

## 3. What breaks or must change

Decode windows of other sessions (serve mode). Today the server runs `read_windows` decode windows (default 1, 4201 to
4205) between two chunks of a read (4595 to 4598), and never after the last chunk, whose logits must stay (comment at
4597). Under layer-major:

- The read's state is incomplete until the last layer step. `run_rows` saves and reloads `pos` of the current session
  (3400, 3439), so `pos` must stay at the start of the read until the end; it must not advance per chunk as at 3365.
- Rebalance race. `run_rows` rebalances every `rebalance_every` windows (3444) without a stream sync before
  (`rebalance_cadence(false)`, 1835 to 1840); the rebalance writes slot blobs on the main stream (3163 to 3166). Today the
  copy stream is idle between chunks: every kick of a forward has been consumed before `process` syncs (3366). Under
  layer-major the copy stream may be staging the next layer from slot blobs (the device-to-device copies at 2838) while a
  window's rebalance overwrites one of those slots. With Ornith's whole-layer copies (2826) no slot is read, but on a card
  with half the experts resident or more the staged layer would hold a wrong expert. Fix: wait for `ev_copy` before a
  rebalance during a read, or defer the window rebalances to the end of the read.
- Cadence. A window costs about one decode step (25 to 30 ms here). One window per (layer, chunk) step would add 432
  windows to a chunk 4096 read, about 13 s (calculated). The windows need a budget: one per chunk-equivalent of work (every
  48 layer steps) keeps today's share; a time-based slice is better for long reads.
- `window()` overwrites `tok`, `pick`, `logits` and the activation buffers (3418 to 3433); layer-major uses `tok` only in
  the embedding pass and must run the head after the last window.

Cancel. Today a stop between chunks keeps a valid state for the tokens read: `on_chunk` returns false (4602), `process`
returns `i + T` (3371 to 3374), and the server appends those tokens to `live` (4605) so a continuation reuses them. Under
layer-major a stop in the middle leaves layers 0 .. l - 1 past the stop point and layers l .. 47 behind it: no prefix of
the prompt has a valid state, and the GDN states of the early layers have also moved past the reused prefix, so even the
`reuse` part is lost. The cancel path must reset the session (`E.reset()`, 2083) and clear `ss.live` and `ss.imgs`
instead of appending (4580 to 4614). With super-chunks, a stop at a super-chunk boundary keeps today's behaviour. The stop
must still be polled between layer steps (the server sends STOP after a PP line, serve/test_control_cancel.py:1), and the
server's silence watchdog expects a PP line per chunk (serve/server.py:229 to 234): emit PP lines with the tokens-equivalent
done so far.

Rebalance after the prompt. `rebalance_after_prompt` (3575) runs `rebalance(2048)` (3572) on the counts the prompt added.
In single-session mode the counts are the same integer sums in another order (2673), so the swaps are the same. In serve
mode the window rebalances during the read see partial prompt counts in a different order (layers above l have none yet),
so the slots after the read can differ, which changes which experts run on the GPU and which on the CPU during the decode
(docs/ORNITH_122B.md:271: identity then holds only with fixed slots).

Prompt cache and reuse. The q36 engine has no mid-prompt checkpoint (no `prompt_cache` or `checkpoint` in
q36_engine.cu; docs/PROMPT_CACHE_TAIL.md and docs/MESSAGE_BOUNDARY_CACHE.md describe the other engine). Reuse works on
`ss.live` (`match_len`, 4434) and `copy_session` (3543), both after a complete read, so a complete layer-major read leaves
them unchanged. Only the cancel path above changes.

CUDA graphs. Prompt chunks run eagerly (`forward(..., false, false)` at 3354); graphs are captured for the decode step,
verify windows and batch windows (`graph_run` at 3241, `run_graph_key` at 3390). Layer-major touches only the eager path.
The one constraint is the pointer rule above: keep the engine buffers' addresses fixed.

Staging region. Unchanged in size (section 2).

Pictures. `chunk_pictures` reuses the pinned `h_map` per chunk (3304 to 3309); an embedding pass over all chunks needs a
sync per chunk or one region per chunk.

## 4. A cheaper middle step

Chunk picked per prompt. `tmax` is fixed at init (`E.init(m, ctx, chunk, ...)` at 5066; every per-row buffer is sized by it,
1904 to 1973), and the slots are sized after it (`place_experts`, 3923). The decode loses its slots to the buffers'
allocation, not to the chunk a prompt uses, so using 2048 for short prompts with 4096-row buffers gives the decode
nothing back. A per-prompt chunk only helps if the buffers above the short size can be lent to the slots between long
prompts; that means reallocating `L.blobs` (one allocation per layer, 3923), resetting the graphs (they hold the slot
blobs' addresses, 3239 to 3240) and a device sync per long prompt. Not recommended.

A bound above 4096. The environment variable is clamped to 256 .. 4096 (5061), but `--chunk N` is not (4913), so
`--chunk 8192` is reachable today (untested at that size). Under model A it gains nothing on this prompt: 4 full chunks
and a tail of 1,489 tokens, 192 steps x 44.1 ms + 48 x 18.64 ms = 9.37 s, the same as 4096, because the 4096 read is
already compute-bound except for its tail chunk. It would cost about 3.4 GiB more VRAM (0.845 MiB per token, below), about
18 more slots per layer. Not recommended.

Equal chunks. The tail chunk is the part of the 4096 read that is still copy-bound: 1,489 rows pay a full 48.07 GiB copy.
Splitting n into ceil(n / tmax) equal chunks (9 chunks of 3,806 or 3,807 rows) makes every step compute-bound: 432 x
20.5 ms = 8.86 s (calculated, model A). This changes the chunk partition, so the attention sums and the GDN chunk splits
change and the output bits change (docs/ORNITH_122B.md:279; issue #81 rules out such changes on the default path): opt-in
only.

Copying only the experts without a slot. Lowering or removing the whole-layer rule at 2826 for this card would send 33.6
GiB per chunk instead of 48.07 GiB at chunk 2048. It changes nothing in the arithmetic (the staged bytes are the same), so
the output is bitwise the same. Its speed depends on the run copies reaching the probe rate, which the RTX 3060 case says
they may not (2824 to 2825); it must be measured. Today `STRATA_Q36_STREAM_ALL` forces the whole copy, nothing forces the
runs.

Giving back slots lost to the chunk buffers. Measured: chunk 4096 has 9 slots fewer per layer than chunk 2048 (77 to 68),
that is 9 x 192 MiB = 1.69 GiB for 2,048 more rows, about 0.845 MiB per token (calculated; a slot is a whole unit, so 0.75 to
0.94). The rule at 5033 to 5034 assumes 0.5 MiB per token, measured on a 35B model with n_embd 2,048. Two buffers are sized
for `tmax` rows but used for at most `hybrid_t` rows when every main layer is hybrid and MMQ runs the prompt:

- `ecpu`, `T * Kp * N * 4` (3964): 96 KiB per row. Used only by `moe_hybrid_cpu` (3082, 3095 to 3097), on the hybrid rows
  path (2648), which takes at most `hybrid_t` rows when `main_hybrid` (only the MTP-only case sends a whole chunk there).
- `escr`, `native_expert_scratch_bytes(T * Kp, FF)` (1973; formula at src/kernels/cuda/iq_kernels.cu:3559): 3 x 8 x FF x 4
  B plus the q8_1 blocks, about 105 KiB per row with FF 1,024 (the size `native_expert_ff1024_test` names, 3937 to 3939).
  Used by the native grouped path (2695: T <= `kGemmT` when MMQ is on), the hybrid rows path (2927) and the PCIe path
  (3109, at most `kPcieMaxEntries` 256 entries, src/qwen35/q36_pcie.hpp:36).

Sized for `hybrid_t` (128) rows instead of `tmax`: (4096 - 128) x 201 KiB = 779 MiB at chunk 4096, about 4 of the 9 slots
per layer back; 377 MiB at chunk 2048, about 2 slots per layer more than today's 77 (calculated). If the decode cost of
lost slots is linear, the short-context decode at chunk 4096 would go from 44.9 to about 45.7 tok/s (interpolated, not
measured). The outputs of every kernel are unchanged; only the slot count changes. `escr` is allocated in `init` before
`place_experts` decides `main_hybrid` (3834), so the shrink must happen in `place_experts` or the allocation must move there.

## 5. Expected prompt throughput, 34,257 tokens (calculated)

Model A of section 1 (compute 8.86 s for the prompt at 4096-row chunks; copies at 53.74 GiB/s; each step max(copy,
compute)). "Low floor" is the same option if the compute were 3 s, to show what depends on the measurement.

| Option | Bytes over PCIe | Read time, model A | tok/s, model A | Against measured 4096 | Low floor (3 s) | Output bits |
|---|---|---|---|---|---|---|
| Chunk 2048, today (measured) | 817.2 GiB | 15.21 s | 2,252 | -38 % | 15.21 s | reference |
| Chunk 4096, today (measured) | 432.6 GiB | 9.37 s | 3,655 | 0 | 8.05 s (copy-bound) | different from 2048 |
| 2048, copy only the experts without a slot | 571 GiB | 10.63 s (copy-bound) | 3,222 | -12 % | 10.63 s | same as 2048 |
| 4096, copy only the experts without a slot | 318 GiB | 9.13 s | 3,752 | +2.6 % | 5.92 s | same as 4096 |
| `--chunk 8192` | 240.4 GiB | 9.37 s | 3,655 | 0 | 4.47 s | changed |
| Equal chunks, cap 4096 | 432.6 GiB | 8.86 s | 3,866 | +5.8 % | 8.05 s | changed |
| Layer-major, 4096-row steps | 48.07 GiB | 8.88 s | 3,858 | +5.5 % | 3.0 s (11,400 tok/s) | same as 4096 |
| Layer-major, 2048-row steps | 48.07 GiB | 8.86 to 9.52 s | 3,600 to 3,866 | -1.5 to +5.8 % | 3.0 to 3.2 s | same as 2048 |
| Layer-major, super-chunks of 32,768, 2048-row steps | 96.1 GiB | 8.9 to 9.6 s | 3,570 to 3,850 | -2.3 to +5.3 % | about 3.2 s | same as 2048 |

The 2048-row range: model A assumes the compute rate at 2048 rows equals that at 4096; the R9700 table at 5033 measured
2048-row chunks 7 % slower than 4096 with every expert resident (5,353 against 5,750 tok/s), which gives the upper end.

Reading under model A: the 4096 read is already near its compute floor. Layer-major then buys about 5 % of prompt speed at
4096, and its real gain is to give the chunk 4096 prompt speed at the VRAM cost of chunk 2048 (77 slots, short-context
decode 46.8 tok/s) less the whole-prompt buffer (4 slots per layer at 32K, bounded by super-chunks). Under a low floor,
layer-major is two to three times faster than anything else. The difference between the two readings is the first thing
to measure.

## 6. Recommendation, ranked

1. Measure the compute floor (no product change, no risk). On the 34K prompt at chunk 4096, an Nsight Systems trace gives
   the kernel busy time on the main stream against the copy time on the copy stream (`cst`, 2814 to 2827) per chunk. A
   test-only switch that skips the host-to-device copies in `stream_kick` gives the same number as a wall time (the output
   is then wrong; timing only). If the compute is near 8.9 s, skip step 5; if it is near 3 s, step 5 moves up to second.
   Test: none needed for output; record the per-chunk split.
2. Size `ecpu` and `escr` for `hybrid_t` rows when every main layer is hybrid and MMQ runs the prompt (section 4). About 4
   slots per layer back at chunk 4096, 2 at 2048. Risk: low; a buffer too small is an out-of-bounds write, so the guard is
   the path conditions at 2648 and 2695 (assert T <= the sized rows where the buffers are used). Test: with
   `STRATA_Q36_FIXED_SLOTS=1 --expert-slots 62 --pcie-frac 0` (the same slot count on both builds), byte-identical
   `--dump-full-logits` on the six parity prompts and on a long prompt at `--chunk 256`, `--chunk 2048` and `--chunk 4096`;
   then the decode at short context and after the 34K prompt with automatic slots, 3 alternated launches.
3. With step 2 in, re-measure chunk 4096 against 2048 on this card and, if it holds (+62 % prompt read, a smaller decode
   loss than 4 %), make 4096 the hybrid default where the room allows it (5056 to 5059 picks 2048 today). Risk: a default
   change, so the AGENTS.md rule applies: the numbers on the record, and the smaller cards (7.5 GiB and 5.5 GiB steps at
   5059) left as they are unless measured. Test: the same dumps; a chunk change alters bits only against another chunk size,
   so the check is that the new default build matches the old build run with `--chunk 4096`.
4. Opt-in equal chunks (+5.8 % under model A) and an opt-in run-by-run copy for cards that stay copy-bound (chunk 2048 or
   smaller: +43 % if the runs reach the probe rate). Risk: low, both opt-in; equal chunks change bits by design and need
   the KL check of docs/ORNITH_122B.md (ten prompts of 32 steps against llama.cpp b11756, bar 0.061). The run copy must give
   byte-identical dumps against the whole copy.
5. Layer-major with super-chunks (default off, an environment variable). Worth it in full only if step 1 finds a low
   floor; under model A its gain is the VRAM of chunk 2048 at the speed of chunk 4096. Risk: high. It changes the
   read loop, the stream's take and done points, the cancel semantics (a stop loses the session state), the window cadence
   and a rebalance race in serve mode (section 3), the MTP prompt pass, and the cross-layer `xq_ready` carry of tail chunks.
   Tests:
   - CLI, byte-identical `--dump-full-logits` against the chunk-major build at the same chunk size, with
     `STRATA_Q36_FIXED_SLOTS=1`, with and without `--no-graph`. The parity prompts (tools/qwen35/parity_prompts.ids, 20 to
     40 tokens) fit in one chunk and never reach the layer-major loop, so the test needs long prompts: the 34K prompt at
     its chunk, and prompts at `--chunk 256` whose length gives a tail of 8 rows or fewer (3 x 256 + 5) and of 9 to 128
     rows (3 x 256 + 100, the hybrid rows path), plus one longer than a super-chunk.
   - `--mtp 2` against `--mtp 0` on the same long prompts (docs/ORNITH_122B.md:306 protocol), if the MTP pass is changed.
   - Serve: tools/qwen35/q36_serve_par.py (concurrent greedy texts equal to the solo texts) and
     tools/qwen35/q36_interleave_probe.py (the decoding slots' pauses during a long read), with fixed slots; and a
     cancel in the middle of a read followed by the same request, which must read from the start and give the same text.

What this note did not check: no code was run or built; the compute floor is an inference from two measurements; the
per-token buffer cost of 0.845 MiB comes from a slot difference, not from an allocation ledger; the speed of the
run-by-run copies at 77 of 256 on PCIe 5.0 is unknown.
