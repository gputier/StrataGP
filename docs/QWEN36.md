# Qwen3.6-35B-A3B: pictures, several requests at once and speed

Strata runs [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of experts, 3B active, 256 experts, top-8) from
[Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own
engine path (`src/qwen35`; `strata` switches to it from the file's architecture). Pick it in setup (`--family qwen36`). Besides text it
reads **pictures** and serves **several requests at once**, the same two options the Flash-Next models have.

Measured on an AMD Radeon AI PRO R9700 32 GB (HIP, Linux, UD-Q4_K_XL, every expert in VRAM) and on an RTX 3060 12 GB (CUDA, 20 threads,
UD-Q4_K_XL in the hybrid mode: 22-37 of 256 experts per layer in VRAM, the rest on the CPU), llama.cpp 972b5fc on the same files.

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT with a 2x2 merger,
the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width differs: 2048 here, 2560 there). Strata runs
it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the encoder choices are the Flash-Next ones
([Images](DETAILS.md#images-vision)): **setup asks "Do you want images?"**, the default answer is yes on an NVIDIA card of 16 GB or more
(GPU encoder, ~1.2 GB of VRAM at up to 1,024 image tokens) and on AMD under Linux (CPU encoder, no VRAM); `--vision yes|no|gpu|cpu`
and `--vision-tokens N` work as there. A card under 16 GB is recommended the CPU encoder (the GPU one takes room from the experts);
`--vision gpu` is kept as asked, with a note.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

A picture becomes up to 1,024 tokens of the context (the CPU default is 300). The engine puts the encoder's rows where the prompt has
`<|image_pad|>` tokens and gives every row its 2-D position, interleaved M-RoPE exactly as llama.cpp does for this model (rotary sections
11 / 11 / 10, a picture's rows at (t, h, w) = (p, p + row, p + column), the text after it continuing at p + max(columns, rows)).

**Checked against llama.cpp** (its `mtmd` and its own graph, the encoder on the CPU on both sides, the same prompt ids, the answer
llama.cpp gives fed to both so the logits are compared step by step, 24 steps; the "floor" is llama.cpp's CPU backend against its own
HIP run, the noise two correct engines have on this model):

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor (llama.cpp CPU vs HIP) |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, read in 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request, the text in each | 655 | differs (a near tie: the floor's KL at this step is 0.16 too) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's: `STRATA 42` (with llama.cpp's own spelling ` STRATA42`), `CODE 4217`, `Red circle / Blue square
/ Green triangle`, `A red square is positioned on the left ... A blue circle is positioned on the right ...`. With the picture options on,
a request without pictures gives the same tokens as without them (10 prompts x 128 greedy tokens, identical).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache: the same conversation sent again (a chat
app resends its pictures) continues from the held state when its pictures are the same (they are matched by the server's key for each
picture); a different picture starts that conversation again.

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`, or `--parallel N` at setup) makes the engine keep N conversations open
(`--batch N`, up to 8) and decode them together: every window carries one token of each conversation, so the dense weights, the head and
every routed expert two conversations share are read once for all of them. The server needs no change: it speaks the same engine protocol
as for Flash-Next ([BATCHING.md](BATCHING.md)). A request that arrives while another is read waits only for that read: the prompt of a
new request is read **between** the decode windows of the others, and a conversation continues in the slot that holds its start (or the
engine copies the longer held start from another slot). A 3,008-token prompt read in 0.95 s (3,200 tok/s, 6 chunks) beside three decoding
slots: each of them paused at most 160 ms (one chunk) instead of the whole read, and produced 20 tokens meanwhile (one window per chunk,
the default); `STRATA_Q36_READ_WINDOWS=4` gives them 65 tokens during a 1.14 s read (more of the time, a slower read).
A request that runs alone (the solo path, with its MTP drafts) and is moved into a batch slot when another arrives, and back when it is
alone again, gives the text it gives running alone (the server's hand-over reuses the held state).

Aggregate decode speed, N requests at once, same prompts, greedy, 128 new tokens each, wall time from the first request to the last
answer, **R9700 32 GB, UD-Q4_K_XL, llama.cpp `llama-server -np N`** (5 interleaved pairs, median; Strata `--batch 8 --mtp 0`):

| Requests at once | Strata | llama.cpp | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.9 tok/s | 73.8 | 1.14 | 5/5 |
| 2 | 173.5 | 107.4 | 1.61 | 5/5 |
| 4 | 252.0 | 165.0 | 1.53 | 5/5 |
| 8 | 325.4 | 208.1 | 1.56 | 5/5 |

**RTX 3060 12 GB, UD-Q4_K_XL in the hybrid mode** (each engine with the slots it needs: `--batch N` takes room from the expert slots; llama.cpp
`llama-server -np N -ncmoe 28 -t 8`, the best split of a sweep on this card; 5 interleaved pairs, medians; the box was shared with other
jobs between runs, never during one):

| Requests at once | Strata | llama.cpp | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64.0 tok/s | 41.1 | 1.56 | 5/5 |
| 2 | 85.1 | 52.5 | 1.62 | 5/5 |
| 4 | 99.3 | 63.0 | 1.58 | 5/5 |

Every slot's tokens equal the tokens of the same request alone (the base engine's), at 2, 4 and 8 slots, with a 2.5K-token prompt read beside
decoding slots, with slots reused for later requests, with pictures and with sampling (same seed). Through the server: four concurrent chat
requests give the texts of the same four one after the other.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts (the server sends it alone while nothing else runs). With
several requests the batch window has no drafts unless `--batch-mtp` is in the config's `args` (opt-in): every slot's window then also
carries the proposal its MTP block made (one per slot, up to the rows to spare in the 8-row window) and a confirmed proposal moves two
tokens in the window. Same prompts, 5 interleaved pairs, R9700:

| Requests at once | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 | 5/5 |
| 2 | 173.1 | 207.0 | 1.20 | 5/5 |
| 4 | 251.0 | 275.0 | 1.10 | 5/5 |
| 8 | 324.0 | 320.5 | 0.99 | 0/5 (no row to spare: no proposals) |

The text is the same with and without it (every slot equal to its solo tokens).

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10 attention layers
and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. Setup writes the context for one conversation, so N slots take N times
that from the room the experts would have: on a card that does not hold every expert (12 and 16 GB, the hybrid mode) that is room the
expert slots lose, and a batch reads more distinct experts from RAM. Setup **offers `--parallel`** where the sessions do not eat the experts' room: up to 3-4 where every expert still fits (3 on a 32 GB card at
32K), and on a card that does not (the hybrid mode) the count whose sessions take at most 35% of the room the experts would have, so it
depends on the context you chose (12 GB: 4 slots at 8K, none at 32K; 16 GB: 2 slots at 32K). It says the price in the same breath, asks
"How many requests at once?" (Enter keeps one at a time), and any number you give is kept. Measured on the hybrid cards (UD-Q4_K_XL, medians of 5
interleaved pairs, `--mtp 2` as setup writes it; "together" = total tokens per second of N simultaneous requests, against ONE request at a time
with the draft layer):

| card, context | slots | one request alone, vs 1 slot | N together, vs one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26-1.64x together (3060 at 8K: 4 slots 1.64x for -5.5% alone), but a single request
with drafts is already about 1.6x as fast as one without, so on a small card the main gain is that a second user starts at once; the
speed gain is small. On a card that holds every expert (R9700 32 GB, 8K prompts) 4 slots run 1.45x one-at-a-time-with-drafts (1.59x with
`--batch-mtp`). The engine says what the sessions take before it allocates them and refuses (with the numbers) a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others: the experts that sit on the
GPU or on the CPU change with the routing of all the requests, and the two compute with different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its
solo run there too, with and without `--batch-mtp`. Under load the intended behavior is not a fixed text: its measured bound and the test setting are in `docs/ORNITH_122B.md`, in the server-side check for #71.

## Speed (0.1.45)

Decode and prompt kernels of this engine path were rewritten for speed; **every change on the decode path gives the old kernels' bits**
(`STRATA_Q36_VERIFY=1 strata ...` runs the old kernel beside each new one on the real activations and compares them bit for bit,
with or without CUDA graphs: the comparisons are counted on the device, so every graph replay counts (issue #92).
On an RTX 5090, p00 and p05 give 16,599 comparisons and 0 mismatches each, with and without CUDA graphs (this change). The older
18,360 / 8,187 / 6,399 comparisons, 0 mismatches each, were measured with `--no-graph` before this change, on the R9700 and the RTX 3060;
`STRATA_Q36_OLD_KERNELS=1` switches them all off). The 10 parity prompts (greedy 128) give the base engine's tokens, plain and with `--mtp 2`. Where the prompt's chunk size changes, the numbers
change a little: see "Prompt chunk".

**What was slow** (rocprofv3, R9700, UD-Q4_K_XL, plain decode, 11.25 ms of kernel time per token): the big Q8_0 products (attention, GDN
projections, head: 1.9 GB per token) run at 530 GB/s, the memory bound. The rest did not: the shared expert's and the attention k/v products
(140 per token) at 18 us, the Q4_K gate/up kernel at 250 GB/s, `k_moe_out` (one block) 17 us, `k_group` (one thread walking 256 experts) 9 us.
Now: those products one warp per row (a launch of 2048 warps took 21 us on gfx1201 where 1024 warps took 5 us, so the grid is capped), shared
gate+up+SwiGLU+q8_1 in one launch, the MoE end with 1024 threads and the next layer's q8_1 image, add+norm+q8_1 in one launch, router top-k and
grouping in one launch, the router and the shared gate in one product, Q4_K gate/up on the V2K kernels (`STRATA_EXPERT_V2K`, gfx12 default).
The prompt's GDN conv runs in tiles of 16 tokens side by side and the delta rule loads one token ahead with two barriers per token (both exact).

R9700 32 GB (p620, HIP gfx1201), UD-Q4_K_XL, base = the engine before this work, 5 interleaved pairs, 3 repetitions each, medians; llama.cpp
972b5fc HIP on the same file (`-fa 1`, best of `-ub` 512 / 2048):

| | base | 0.1.45 | ratio | pairs faster | llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode tg128 | 83.2 tok/s | 117.8 | 1.42 | 5/5 | 81.9-82.1 |
| prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 1.80 | 5/5 | 3,794 (-ub 2048) / 3,042 (-ub 512) |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 | 183.7 | 1.10 | 3/3 | 129 (draft-mtp, n-max 2, earlier run) |
| 1 / 2 / 4 / 8 requests at once (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 | 5/5 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, so the prompt chunk stays 512): decode 80.4 -> 106.9 tok/s (1.33), pp4096
2,160 -> 2,200 (llama.cpp 3,486 at -ub 2048, 2,738 at -ub 512; decode 81.1), `--mtp 2` 154.7 -> 167.3. `--chunk 2048` there moves the experts
to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94 - an option, not the default.

RTX 3060 12 GB (P2003, CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer in VRAM): the GPU part of a
token is the dense Q8_0 at the card's bandwidth (6 ms of the 17) and the rest waits for the CPU pool (9 ms of experts), so only the small
kernels' share is gained: decode tg128 58.5 -> 59.7 tok/s (+2%), pp2048 1,958 -> 2,022 (+3.3%), 10/10 identical, plain and `--mtp 2`
(84.3 -> 84.7). llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48-52.

**Prompt chunk (the default changed).** The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per
expert: R9700 pp4096 at chunk 512 / 1024 / 2048 / 4096 = 3,453 / 4,786 / 5,766 / 6,223 tok/s. With every expert in VRAM the engine now takes the
largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve and the sessions' caches without giving the packed decode
copies back (8K-64K context: 4096, 100K: 2048, 131K: 512; `--batch 2..4`: 1024 at most so the other slots are not held for a whole chunk,
`--batch 8`: 512). The hybrid mode takes 4096 from 20 GiB free with one session, 2048 from 7.5 GiB, 1024 from 5.5 GiB and 512 below. The 20 GiB counts the packed Q8_0 copies that can be given back (19.41 GiB raw on the measured RTX 5090). The 4096 step needs automatic slots: with `--expert-slots` or `--expert-vram` given by hand, the old steps hold.
(`hybrid_prompt_chunk`, `src/qwen35/q36_rules.hpp:42`; with several sessions the chunk stays 2048). The 4096 step was measured on an
RTX 5090 with 23.2 GiB free (docs/ORNITH_122B.md, "Prompt chunk size, issue #104"). A bigger chunk takes expert slots: on an RTX 3060,
chunk 4096 read prompts 5% faster and decoded 13% slower, the reason the hybrid mode kept 2048 before that step. **`--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the old size, and
then the text is the base engine's, bit for bit.** The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the
prompt state: KL against the 512 chunk on six real 1.5-3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3, 2048 4.2e-3,
4096 3.6e-3 (per prompt 2.3e-3 .. 5.9e-3), top-1 agreement 44-48 of 48 rows - the size of the llama.cpp CPU-vs-HIP difference on the same
file (4.2e-3), not a quality loss. Greedy 64 tokens after those six documents: the default chunk gives the base engine's text for 2 of 6,
`--prefill 512` for 6 of 6. A prompt shorter than the chunk (every chat turn after the first) is one chunk either way: unchanged.
The new default's goldens are `phaseA-tests/q36speed/golden/` (six documents, greedy 64): later changes are checked against them.

**Tried and dropped.** The MTP drafts' head over a subset of the vocabulary (`--draft-vocab N`, kept as an opt-in; the answer never
changes): the first 16K-131K rows gain 5-8% on English and code but lose 10-15% on Chinese / Russian / Arabic / Hindi (70% of a Chinese
prompt's token ids are above 65,536), and a candidate list built from the tokens seen so far loses on both. The shared expert on a second
stream (captured into the decode graph) decodes at 36 tok/s instead of 117 with this ROCm. Chunked (WY) GDN and the PCIe-assisted hybrid decode
were not attempted.

## Ornith-1.5-122B (hybrid, 400K context)

Full engineering record (protocols, experiments in order, open leads): [ORNITH_122B.md](ORNITH_122B.md).

Ornith-1.5-122B runs on the q36 engine path (`src/qwen35`), in hybrid mode: the experts that do not fit in VRAM run on the CPU pool, or are copied over PCIe and computed on the GPU. It is a mixture of experts of 48 layers, 256 experts per layer, top-8 routing, about 10B active parameters per token. The measured file is `Ornith-1.5-122B-IQ3XXS-IQ4XS.gguf` (58,241,032,032 bytes): IQ3_XXS for the gate and up experts, IQ4_XS for the down experts, and the MTP block (blk.48) at Q4_K for its gate and up.

The server listens on port 8090. The Windows launcher (`start-ornith.ps1`, `stop-ornith.ps1`, `stratagp-ornith.json`) passes `--native <gguf> --max-context 409600 --prefill auto --rope-scaling yarn --rope-scale 2 --rope-orig-ctx 262144 --kv q4`. MTP is off in its config (see below). Two other settings were measured against it on 26 October 2026 (build rev19, one 34,257-token prompt, 300 tokens, three alternated rounds) and are not adopted: `--max-context 40960` cut the request time by 5.2 % (38.8 tokens/s of decode against 34.6, 83 expert slots per layer against 68) but cannot hold a prompt above 40,960 tokens, and `--batch 2` took 18 % longer than the same two requests run one after the other (about 32.4 s against 27.4 s for a 34K request and a short one). The figures and what was not measured are in docs/ORNITH_122B.md, "Series of 26 October 2026, build rev19".

Options added by the branch:

- `--expert-profile <file>` starts the expert slots from a learned profile (STRP format, version 1). `--expert-profile-save <file>` writes it at QUIT. The server passes the saved file back as `--expert-profile` at the next start, once it holds a whole header (`serve/test_q36_config.py`). On the q36 engine (Qwen3.6, Ornith) the profile is written at QUIT only: the server does not pass `--expert-profile-save-every` there, so a crash loses what was learned since the start.
- `--pcie-frac F`: the share of the missed experts that are copied over PCIe and computed on the GPU. `0` turns it off. Without the option the PCIe path is off (share 0) and nothing is probed at load: on Ornith-1.5-122B (RTX 5090, 128 GB RAM, Ryzen 9 9950X3D, decode, one 300-token prompt, three alternated launches per setting; the decision series of 25 October, before this branch's code existed) the default measured 42.2 tokens/s with the old probed share, 44.9 with `--pcie-frac 0`, and about 42.5 with any share from 0.05 to 0.30, so the path costs as soon as it is active. The check, same day with this branch's binary, gave 46.7, 46.7 and 46.6 tokens/s in three alternated launches, against 43.7, 43.7 and 43.6 with the old defaults (`--pcie-frac auto` and `STRATA_Q36_REBALANCE_EVERY=32`), so the new defaults are 6.9 % faster. With `--pcie-frac auto` the probe measured 57.7 GB/s and set the share to 0.55. `--pcie-frac auto` opts in to the former automatic share: it starts from the probed PCIe bandwidth and is then refined from the CPU pool's speed, between 0 and a cap of 0.95. A fixed value is taken as given, up to 1. Any other word, a negative number, an exponent, hex or text after the number is refused with exit code 2. On a model with a hybrid main layer, the engine prints one line at load saying which share it uses and why; with the new defaults it reads: "PCIe share off (the default: --pcie-frac F or --pcie-frac auto turns it on)".
- `--mtp N`: number of MTP drafts. The start refuses N above 7 with `strata-q36: --mtp takes 0 .. 7 drafts` and exit code 2.

Options that change the output (off by default):

- `--cache-prior L` (0 to 1, 0 is off) biases the logits of resident experts, with the Delta_avg version of arXiv 2412.00099. Measured: L 0.02 gives KL 0.0707 at 23.1 and 29.4 tokens/s; L 0.05 gives KL 0.1453 at 30.2 and 35.4; L 0.1 gives KL 0.1893 at 32.5 and 44.2. No value falls under the bar of 0.061. The earlier version of this option reached 75 to 77 tokens/s at L 0.25 and 0.5, with KL 0.64 and 0.91, and altered the text.
- `--ser Kmin,t` keeps a lower-ranked pick only when its probability is above t times the top probability. `--ser 7,0.4` gives KL 0.086 and `--ser 6,0.4` gives KL 0.168. The speed gain is within run-to-run noise.
- `--ser-pool Kmin,t` is `--ser` restricted to the picks whose expert has no VRAM slot: a pick with a slot is always kept, since a resident expert is read almost for free and only a pick served by the CPU pool costs. Same bounds and error message as `--ser`; with both given the start stops with exit code 2. Without the hybrid expert mode (every expert has a slot) it acts as `--ser`, and the start says so. Measured on 26 October 2026 (build rev19, test PC RTX 5090 and Ryzen 9 9950X3D, `--mtp 1`, 300 tokens, median of three alternated rounds, mean KL on 10 prompts of 32 steps, bar 0.061; the full series is in docs/ORNITH_122B.md, "Series of 26 October 2026, build rev19"): without the option the KL is 0.0513. `--ser-pool 7,0.4` gives KL 0.0711 (above the bar), `--ser-pool 7,0.6` gives 0.0555 (below it), and `--ser 7,0.6` gives 0.0889 (above it). The request time falls by 10.3 %, 10.7 % and 12.3 % on the short prompt, and by 1.4 %, 0.6 % and 3.3 % after a 34,257-token prompt. None of the three is the default: each changes the output, and the decision is pending.
- `--kv q4` (set by the Windows launcher) stores the KV cache in 4 bits. Measured during gate T7 on the .99, branch `feat/ornith-122b` at `e82d1fc3`, fp32 against q4 on the same model and the same prompts. On 10 short prompts of 33 steps, top-1 agreement is 93.3 % (308/330) and the mean KL is 0.079 (0.013 to 0.361). On a prompt of 20,914 tokens, agreement is 30/32 and KL 0.093. On a prompt of 65,760 tokens, agreement is 31/32 and KL 0.073. The engine against llama.cpp gives KL 0.041, so q4 adds about 0.04, with no growth in KL as the context gets longer. The error comes from the prompt pass in q4: on the 20,914-token prompt, the output already diverges at step 0. The gap was accepted on 23 October 2026. The fp32 cache is unchanged.

Refusals at start (exit code 2, messages prefixed by `strata-q36:`):

- `--cache-prior` with two or more sessions (`--batch` 2 or more): it "needs one session".
- `--cache-prior` with MTP drafts (`--mtp N`, N above 0) on a model with an MTP block: "cannot be combined".
- `--cache-prior` when every expert fits in VRAM: it "needs the hybrid expert mode".
- `--cache-prior`, `--ser` with Kmin below the 8 picks of the model, or `STRATA_Q36_SER_TABLE`: "need the fast routing path". That path is taken when `STRATA_Q36_OLD_KERNELS` is unset, the model has 256 experts and 8 picks. With `--ser 8,t`, the option is not refused.
- `--ser` not given as `Kmin,t` with 1 <= Kmin <= 8 and 0 <= t <= 1: "--ser must be Kmin,t".
- `STRATA_Q36_SER_TABLE` (a file, read once the layer count is known): a line "layer Kmin" per layer, decimal integers, blank lines and whatever follows a `#` ignored. A layer must be in 0 to n_layers-1 and Kmin in 1 to 8, and no layer may be named twice. A layer the table does not name keeps `--ser`'s Kmin (8 without `--ser`). `t` is `--ser`'s for every layer, so without `--ser` (t = 0) a table prunes only picks of probability 0. The MTP block keeps `--ser`'s Kmin. A file that cannot be read or a line that fails a check stops the start with exit code 2 and the line's number. Measured gain: none. The map (`tools/qwen35/ser_layer_map.ps1`) ran on 26/10/2026, and the table that passes its quality bar gives no gain (see docs/ORNITH_122B.md, "Per-layer SER table").
- `--cache-prior` outside [0, 1]: "--cache-prior must be in [0, 1]".

Environment variables. The q36 engine reads 22 of its own: 21 `STRATA_Q36_*` and `STRATA_DMA_BATCH`. It also reads `STRATA_BATCH_MTP`, `STRATA_MMQ_KQ` and `STRATA_Q8_PACKED`, which other engines share (on Windows it sets the last two to 1 when unset). Unset means the default shown.

- `STRATA_Q36_BLAS_F32`: off. Set, the router product uses the hipBLAS / cuBLAS SGEMM instead of the native kernel (`q36_engine.cu:2116`, A/B check).
- `STRATA_Q36_CHUNK=N`: forces the prompt chunk, from 256 to 8192 (`src/qwen35/q36_engine.cu:5095`; above 4096 only for the #104 measurement). Unset, `--chunk auto` picks it from the free VRAM (`:5047`). On Ornith-1.5-122B in hybrid mode, 4096 read a 34,257-token prompt 62 % faster than 2048, the hybrid default before 25 October 2026, and cost 4 % of decode at short context (docs/ORNITH_122B.md, "Prompt chunk size, issue #104"; next steps in docs/Q36_LAYER_MAJOR_PROMPT.md). In hybrid mode the unset default is now 4096 with one session and automatic slots, from `hybrid_prompt_chunk`.
- `STRATA_Q36_DRAFT_VOCAB=N`: the MTP drafts pick among the first N token ids, used when `--draft-vocab` is absent (`:5168`). Unset, 0: the whole vocabulary (`:2750`).
- `STRATA_Q36_FIXED_SLOTS=1`: freezes the expert slots as placed. The identity gate uses it (`:3148`).
- `STRATA_Q36_KEEP_PACKED`: unset, the packed Q8_0 copies are freed when that gives more expert slots (`:3760`). Set, they are kept.
- `STRATA_Q36_NO_EXT`: unset, a decode batch of up to 8 rows takes the router and the shared expert's gate in one product (`:2643`). Set, separate products.
- `STRATA_Q36_NO_MMQ`: unset, the MMQ kernels are used when built (`:1932`). Set, they are not.
- `STRATA_Q36_NO_MTP_CPU`: unset, the experts of the MTP block may go to the CPU pool (`:3738`, `:3789`). Set, they may not.
- `STRATA_Q36_NO_PREFETCH`: unset, the experts of the next layer are prefetched (`:3997`). Set, no prefetch.
- `STRATA_Q36_OLD_KERNELS`: unset, the fast decode kernels and the fast routing path are used (`:2154`). Set, the old kernels, and `--cache-prior` and `--ser` are refused.
- `STRATA_Q36_ONE_TOKEN_CPU`: off. Set, the one-token AVX-512 CPU kernels are used for every token count: the IQ3_XXS gate/up and IQ4_XS down rows on AVX-512, an IQ4_NL down on the AVX-2 kernel. They ran slower than ggml-cpu on the Ornith box (comment at `:3935`, `native_expert_ff1024_test --bench`). Unset, a lone token takes ggml-cpu's dot, and a group of IQ3_XXS gate/up rows takes the AVX-512 kernel in ggml's float order (issue #74, below).
- `STRATA_Q36_PCIE_AUTO_STEP=1`: with `--pcie-frac auto` only (without it the variable prints a line and does nothing), the share moves by 0.05 every 32 decode tokens against the measured wall time per decode token, between 0.30 and 0.90. A change under 3 percent does not move it, and the search stops on the best share seen after two reversals. PcieAuto's refinement is off then. Unset, the default behaviour is unchanged (the constants and the comment at `q36_pcie.hpp:144` to `:159`, `pcie_step_enabled` at `q36_pcie.hpp:162`, `PcieStepSearch` at `q36_pcie.hpp:196`, `pcie_step_window` at `q36_engine.cu:1790`, enabled at `q36_engine.cu:4021`).
- `STRATA_Q36_PLACE_DEBUG`: off. Set, prints the placement (`:5059`).
- `STRATA_Q36_PROMPT_SWAPS=N`: the slot replacements of the rebalance after a prompt, decimal digits only, from 0 to 2147483647; 0 means no cap. Unset or empty, 4096 (2048 before round v25: at 34K and 112K, 4096 cut the request time by 0.4 to 0.5 % over three alternated rounds, above the 33 ms spread between rounds, and the decode after the 112K prompt went from 22.5 to 23.2 tok/s; 8192 and 0 did no better: against 2048, 8192 cut the request time by 0.34 % at 34K and 0.35 % at 112K, and no cap by 0.45 % and 0.37 %, same series, build rev19). Any other value prints a line and uses 4096. Read once in init (`q36_engine.cu:1912`, `prompt_swaps_from_env` at `q36_rebalance.hpp:47`), kept in `hybrid_swaps_after_prompt` (`q36_engine.cu:3586`), used by `rebalance_after_prompt` (`q36_engine.cu:3590`).
- `STRATA_Q36_READ_WINDOWS=N`: default 1, from 0 to 64 (`:4201`). The decode windows run between two chunks of a prompt read.
- `STRATA_Q36_REBALANCE_EVERY=N`: the positions between two rebalances of the hybrid expert slots, decimal digits only, 1 to 1024. Unset, 4 (from 25 October 2026; it was 8 before, and 32 before that). On Ornith-1.5-122B, same setup as `--pcie-frac` above, 8 measured 45.4 tokens/s against 42.2, and 46.5 with the PCIe path off as well: the decision series of 25 October, before this branch's code existed; with the rebalance counters below, 8 served 57.9 % of the expert picks from VRAM against 55.8 % at 32, for 46.7 tokens/s against 45.5 after its 37 ms of rebalance host time per 300-token request, docs/ORNITH_122B.md item 14). Cadence 2 was measured on 26 October 2026 (build rev19, one 34,257-token prompt, 300 tokens, `--mtp 1`, three alternated rounds, prompt swaps at 4096): 18,174 ms against 18,356 at 4, which is 1.0 % less, and 35.8 tokens/s of decode against 35.0. One of its three launches stopped in the prompt read with `prefill mmq: mul_mat_q: unknown error`, and the Windows log shows driver events at that moment. The launch passed when run again. Cadence 2 needs a new test before any change, and the default stays 4; 8 gave 18,418 ms (+0.3 % against 4). Any other value prints a line and uses 4. The MTP draft depth stops short of the next rebalance position (`rebalance_every_from_env` at `q36_rebalance.hpp:32`, `draft_depth_limit` at `q36_rebalance.hpp:61`, read in `q36_engine.cu:1909`, applied at `q36_engine.cu:3613`).
- `STRATA_Q36_STATS=1`: off. Prints one line per window, 48 steps then 96: the bytes each path reads and the host's wait on the doorbell (`src/qwen35/q36_stats.hpp:24`, `q36_engine.cu:1753`). The share of the expert picks each path served (VRAM slots, PCIe, CPU pool) is on the request's own line, printed for every hybrid request with or without this setting ("experts on the GPU ... over PCIe ... on the CPU", `hy_gpu`, `hy_pcie`, `hy_cpu`). The stats line reports the decode-cadence rebalances: slot swaps, MB of expert blobs copied, KB of slot tables (`slot_of`) uploaded, and host time (`rb_swaps`, `rb_blob_bytes`, `rb_slot_bytes`, `rb_ms` in `DecodeStats`), not the rebalance after a prompt. At the end of a request decoded by a single session, one decode steps line follows the request's stats lines. It lists each decode step of the request: one forward of the decode loop, a token, or a verify window with its row count. For each step it gives the pool's time, the host's wait on the doorbell, the host's own time (the step's wall time less those two) and the copy streams' busy time on the step's PCIe copies, which overlaps the other three. The line lists at most 4,096 steps and counts the rest (`StepLog` and `format_steps` at `q36_stats.hpp:151` and `:204`, `after_rows` at `q36_engine.cu:1820`). Unset, no step is recorded. `tools/ornith/partition_summary.py` matches the steps to the streamed tokens and, for each token above the inter-token p95, names the stage with the largest excess over the median step.
- `STRATA_Q36_STREAM_ALL`: off. Set, the whole layer streams from host memory, the first version, PCIe-bound (`:2821`).
- `STRATA_Q36_SWIGLU_SKIM=<tau>`: off. Set, the CPU pool's SwiGLU skim (#110) cuts each expert neuron whose |silu(gate)| is below tau: its output is 0, and its up row is still read. Read once (`src/qwen35/q36_stats.hpp`, `swiglu_skim_tau`): a value that is not a finite number >= 0 prints one line and leaves the skim off. The section "SwiGLU skim, option (#110, 26 October)" of `docs/ORNITH_122B.md` lists the paths it covers.
- `STRATA_Q36_VERIFY`: off. Set, the decode kernels are compared with the old kernel's bits (`:2167`), on the device and with CUDA graphs on; its mismatch lines print at the end of each window (decode window, MTP step or prompt chunk), not at the comparison, so a fatal CUDA error inside a window loses them. This also disables the fused router path (`:2643`).
- `STRATA_DMA_BATCH=N`: the grouping of PCIe copies, modes 0, 1 and 2 (`include/strata/core/dma_batch.hpp:30`). The q36 default is 1, one `cudaMemcpyBatchAsync` per group (`q36_engine.cu:102`). 0 gives one copy each.
- `STRATA_POOL_CORES=<list>`: the CPU pool's workers take these CPU IDs in this order (`8-15`, `0-7,16-23`), not the OS's order; `skip_first` has no effect on the list, and a host core that is listed also runs a worker. Read once; a list that does not parse or names a CPU the topology does not have prints one line on stderr and keeps the default order. The grammar is `src/kernels/cpu/pool_core_list.hpp`, the override `src/kernels/cpu/pool.cpp:137` (`explicit_pool_cores`, `:91`), tested by `pool_core_list_test`. Measured on 26 October 2026 (build rev19, short prompt, 300 tokens, `--mtp 1`, median of three alternated rounds, Ryzen 9 9950X3D): 12 threads, the default, took 6,041 ms (52.7 tokens/s). With 7 threads and no list, 6,379 ms (50.4). With a list of 7 cores on the logical processors 0 to 12 (even numbers), 6,382 ms (50.3); on 16 to 28 (even numbers), 6,307 ms (51.0); on 0, 4, 8, 12, 16, 20 and 24, 6,382 ms (50.3). The 16-to-28 set beats the 0-to-12 set by 1.2 %, and every 7-core run is slower than the default (the best one by about 4 %). Which of the two groups carries the 3D V-Cache is not established by this measurement: the L3 line below will say it at the next build. The pool log of the 7-thread runs reads "8 CPU threads": 7 pinned workers plus the host thread, which takes jobs during a drain and is not pinned (the `hybrid experts` line now says so), so a list fixes the workers only.
- On Windows, a pinned pool prints one line at start saying which of its cores sit under which L3 cache, largest first: `strata cpu pool: L3 96 MiB covers cores ...; L3 32 MiB covers cores ...`. On a Ryzen 9 9950X3D the 96 MiB group is the CCD with the 3D V-Cache. The IDs are first logical processors (even numbers with SMT on), written in the `STRATA_POOL_CORES` grammar so a set can be pasted into it. Reporting only: the order and the pinning do not change, and Linux prints nothing. Only processor group 0 is read (64 logical processors). The rule is `src/kernels/cpu/pool_l3_groups.hpp`, the OS read `src/kernels/cpu/pool.cpp:467` (`read_l3_groups`), the log `:634`, tested by `pool_l3_groups_test`.

`STRATA_Q36_CPU_POOL` is a compile-time macro (`CMakeLists.txt:801`), not an environment variable.

Measured on 22 and 23/10/2026, on an RTX 5090 32 GB with 128 GB of RAM, three runner VMs running. Commit ec3d4b63 is compared with 7bf040a5, the start of the work, and with 9ce40dca. The working set is capped at 24 GB per process, unless noted.

- Greedy identity: 6 of 6 prompts (p00 to p05, 32 tokens) identical between ec3d4b63, 9ce40dca and 7bf040a5, with `STRATA_Q36_FIXED_SLOTS=1 --pcie-frac 0 --expert-slots 62 --cpu-threads 6`.
- Forced logits at default settings: 10 of 10 files identical between ec3d4b63 and 9ce40dca.
- KL against llama.cpp b11756 logits, 10 prompts of 32 steps: 0.0423, against a bar of 0.061. The top token matches llama.cpp on 299 of 320 steps (93.4 %). Repeated runs are identical to the byte. Between builds the KL moves by 0.0032 (0.0455 for the build of task 5, 0.0423 now). Perplexity was not measured.
- Decode, 4 code requests of 1,024 tokens per launch, three cold launches per build, alternated: 30.0 tokens/s (29.7 to 30.2), against 22.6 (22.5 to 22.8) at 7bf040a5.
- Decode after a 127,685-token prompt, one launch per build: 12.8 tokens/s, against 10.4. The prompt is read in 78.0 s on both builds, 1,636 tokens/s.

Capped and uncapped runs differ, and only the capped figures are reproduced:

- The first uncapped series ran by mistake, with the memory guard inactive: 43.2 tokens/s for the branch against 43.8 for the start at 1,024 tokens, and 18.8 and 17.2 at 128K. Free memory fell to 703 MB. On that series alone the branch is no faster than the start at 1,024 tokens.
- A dedicated uncapped run loaded the model: available memory went from 108.7 GB to 4.3 GB in 25 s. The guard stopped the engine with a working set of 105 GB and 79.5 GB of private memory, still rising. The pinned experts take 48.07 GiB; about 28 GB of private memory is not identified. No uncapped series ran to the end.
- Without the cap, a 128 GB machine running three VMs keeps less than 8 GB free.

Where a token's time goes (`STRATA_Q36_STATS=1`, one launch at 30 tokens/s, so 33.3 ms per token, computed as 1,000 / 30):

- CPU pool on the missed experts: 20.8 ms, 62 % of the token (computed).
- Host waiting for the GPU (attention, GPU layers, GPU experts): 9.3 ms.
- Host outside the pool: 3.1 ms.

The three parts add to 33.2 ms, the rounding of 33.3. Even without the pool, 9.3 + 3.1 = 12.4 ms is above the 10 ms that 100 tokens/s allows (computed). Per token, 211 MB are copied over PCIe and 462 MB are read by the pool, at 22.5 GB/s.

Attention costs 2.42 ms per token at 1k context, 6.82 ms at 32k and 18.93 ms at 128k (base 9ce40dca, `a5.md`). Bounding the grid of `k_attn_part` removes nothing: the time follows the chunks processed.

Other levers measured: `STRATA_DMA_BATCH=1` against 0 has no measurable effect, and neither has `--pcie-frac 0.50` against the automatic share (issue #80). The automatic share now settles at 0.45. The former default settled at 0.80 and gave 22.4 tokens/s, against 24.6. The best fixed value measured is 0.50 to 0.55. The branch also prefetches the experts of the next layer, groups the copies, and drops a single-pass CPU pool per layer, which gave no gain. Without `--cpu-threads`, the pool runs one worker per worker processor present, capped at 12 (`kAutoPoolWorkersCap`, `src/qwen35/q36_threads.hpp:11`) since 25 October 2026; before, it ran 6. `--cpu-threads 0` keeps one per processor with no cap.

Not reached: 100 tokens/s. The rate is 30.0 tokens/s at 1,024 tokens. Its open issues are #77 (CPU kernels of ik_llama: the pool stays the first cost, 20.8 ms), #79 (CPU pool), #80 (PCIe share), #81 (attention at long context), #74 (MTP), #75 (REAP) and #78 (speculative decoding). #76 (experts in 2 bits) was measured and set aside: 40.65 against 28.45 tokens/s, KL 0.4697 against 0.0423 (docs/ORNITH_122B.md).

MTP is off in the configuration. Measured on 22/10/2026 (issue #74), before the change below: with `STRATA_Q36_ONE_TOKEN_CPU=1`, its output is identical to `--mtp 0` with the same variable on 6 of 6 prompts, but 20 to 35 % slower, with no drop in missed bytes. Without the variable, `--mtp 2` gave the text of `--mtp 0` on 1 prompt of 6.

The cause, read in the code: in a verify window, an expert picked by two or more rows is one CPU job of several tokens. By default its IQ3_XXS gate/up rows took the AVX-512 multi-token kernel (`iq_avx512.cpp`), while a lone token, the only case at `--mtp 0`, takes ggml-cpu's dot. Both compute the same integer sums per block. The float part differed: 16 float lanes and `_mm512_reduce_add_ps` against ggml's 8 lanes and `hsum_float_8`. A near tie then flipped. The IQ4_XS down rows are ggml-cpu's dot for every token count by default, so they never differed.

The change: for IQ3_XXS, a group of two or more tokens now takes `iq512_iq3xxs_gu_rows_ggml`. It decodes the weights once for all the tokens, as before. For each token it adds the two 256-bit halves of the integer sums (ggml's `sumi1 + sumi2`, lane for lane), accumulates one FMA per block into 8 float lanes with `d = fp16(d_x) * d_y`, and ends with `0.25f * hsum_float_8`: the order of ggml's AVX2 dot `ggml_vec_dot_iq3_xxs_q8_K`, which has no AVX-512 branch at the pinned commit 3cf03257. It returns the raw gate and up values, and `native_gu_rows` applies SwiGLU in `native_expert.cpp` with the expression of the lone-token path. The kernel is used only where ggml-cpu's dot is its AVX2+FMA one (`ggml_cpu_has_avx2()`, `ggml_cpu_has_fma()`), not with `STRATA_Q36_ONE_TOKEN_CPU` (its kernels already give one result for every token count) and not with `STRATA_IQ_MT_MIN=1`. `STRATA_NO_IQ512_GGML_ORDER=1` gives the previous multi-token kernel. A lone token's path is unchanged, so the `--mtp 0` text cannot change.

Measured on 25 October 2026 on the Ornith box (RTX 5090, 128 GB RAM, Ryzen 9 9950X3D), with the new defaults and `--mtp 2`, decode, three alternated launches per setting: with the ggml-order kernel (the default) 49.0, 49.1 and 49.0 tokens/s; with `STRATA_NO_IQ512_GGML_ORDER=1` 47.6, 47.4 and 47.4 tokens/s. The kernel is about 3.3 % faster. The owner chose on 25 October to keep it on by default, an exception to the opt-in rule of AGENTS.md (line 30); `STRATA_NO_IQ512_GGML_ORDER=1` turns it off. The parity holds for IQ3_XXS gate and up rows (Ornith's quant); an IQ4_NL down row still takes the AVX-2 group kernel against ggml for one token (the #152 rule), so that gap remains. The ggml-order kernel matches ggml-cpu only while the pinned ggml-cpu uses its AVX2 dot for IQ3_XXS even in an AVX-512 build (pin at `CMakeLists.txt:1538`, `GIT_TAG 3cf03257`), and `iq512_ggml_order_test` checks it on the build it runs on.

Greedy identity with fixed slots (`STRATA_Q36_FIXED_SLOTS=1 --pcie-frac 0`, six parity prompts, 128 tokens): `--mtp 2` and `--mtp 3` give the same text as `--mtp 0` on six of six prompts with this branch, against one of six on main (ac27f23f). The build with the new defaults gives the same text as the build before them on six of six prompts under the same setting.

On a real AVX-512 CPU, `iq512_ggml_order_test` and the default pass of `native_expert_row_invariance_test` pass, with 0 differing values.

Checked without AVX-512 (an AVX2 transcription of the new per-token arithmetic, run under qemu against ggml-cpu's AVX2 dot): 0 of 32,768 dots differ, against 22,863 of 32,768 for the previous 16-lane order.

## Non-finite logits

A row of logits with no finite maximum, or a NaN, draws no token. The engine prints `ERR the logits hold no finite maximum or a NaN: no token drawn` (`src/qwen35/q36_engine.cu:728`, printed at `:4646` and `:4741`) and prints no DONE line for that request. In the batch path the slot ends with `BDONE <slot> <tokens> error <ms>` (`q36_engine.cu:4204`), and the server does not reuse that slot's state (`serve/server.py:1782`).

The server turns the ERR line into HTTP 400 when it comes before the first token, on the non-streaming endpoints of `/v1/chat/completions`, `/v1/messages` and `/v1/responses` (`serve/server.py:5665`, `:6070`). A streaming request has already sent its 200 status and headers, so it gets no 400. It receives an error event, then the stream ends: `data: {"error": {"type": "server_error", ...}}` followed by `data: [DONE]` on chat completions (`serve/server.py:6001`), a `response.failed` event on `/v1/responses` (`:6102`), and an `event: error` on `/v1/messages` (`:6221`).

## Not done / limits

- Ornith-1.5-122B does not reach 100 tokens/s (30.0 at 1,024 tokens). Its limits and open issues are in its section above.
- Eight slots at most (the window holds 8 rows); `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals. `logit_bias` is not available on this engine path: the
  server refuses a request that has it (as it does with `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over; the MTP block's cache has holes for
  the tokens a slot decoded in batch windows (the drafts of a later solo stretch are a little worse; the text is not affected).
- Intel Arc (SYCL) does not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed); run them under the box's GPU lock:

- `q36_solo_gate.py` + `q36_cmp_gate.py`: the 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's.
- `q36_batch_gate.py --slots N --waves --long 2500`: every slot's tokens equal its solo tokens, slots reused, a long prompt read beside
  decoding slots. `q36_proto_edge.py`: `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests,
  the context's end. `q36_interleave_probe.py`: what the decoding slots see while a long prompt is read.
- `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp): the picture comparison above; `q36_vision_proto.py`: pictures through
  `GENI` / `BGENI` (solo, in a batch, the conversation cache); `q36_vision_serve.py` and `q36_serve_par.py` / `q36_serve_handoff.py`:
  the same through the server.
- `q36_par_bench.py`: the aggregate-throughput A/B against `llama-server -np N`.
