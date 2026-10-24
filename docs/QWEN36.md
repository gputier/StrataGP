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
solo run there too, with and without `--batch-mtp`.

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
`--batch 8`: 512). The hybrid mode keeps 2048 (a bigger chunk takes expert slots: RTX 3060 chunk 4096 reads prompts 5% faster and decodes 13%
slower) and goes down to 1024 / 512 on a card with 7.5 / 5.5 GiB free. **`--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the old size, and
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

The server listens on port 8090. The Windows launcher (`start-ornith.ps1`, `stop-ornith.ps1`, `stratagp-ornith.json`) passes `--native <gguf> --max-context 409600 --prefill auto --rope-scaling yarn --rope-scale 2 --rope-orig-ctx 262144 --kv q4`. MTP is off in its config (see below).

Options added by the branch:

- `--expert-profile <file>` starts the expert slots from a learned profile (STRP format, version 1). `--expert-profile-save <file>` writes it at QUIT. The server passes the saved file back as `--expert-profile` at the next start, once it holds a whole header (`serve/test_q36_config.py`). On the q36 engine (Qwen3.6, Ornith) the profile is written at QUIT only: the server does not pass `--expert-profile-save-every` there, so a crash loses what was learned since the start.
- `--pcie-frac F`: the share of the missed experts that are copied over PCIe and computed on the GPU. `0` turns it off. Without the option, the share starts from the probed PCIe bandwidth and is then refined from the CPU pool's speed, between 0 and a cap of 0.95. A fixed value is taken as given, up to 1.
- `--mtp N`: number of MTP drafts. The start refuses N above 7 with `strata-q36: --mtp takes 0 .. 7 drafts` and exit code 2.

Options that change the output (off by default):

- `--cache-prior L` (0 to 1, 0 is off) biases the logits of resident experts, with the Delta_avg version of arXiv 2412.00099. Measured: L 0.02 gives KL 0.0707 at 23.1 and 29.4 tokens/s; L 0.05 gives KL 0.1453 at 30.2 and 35.4; L 0.1 gives KL 0.1893 at 32.5 and 44.2. No value falls under the bar of 0.061. The earlier version of this option reached 75 to 77 tokens/s at L 0.25 and 0.5, with KL 0.64 and 0.91, and altered the text.
- `--ser Kmin,t` keeps a lower-ranked pick only when its probability is above t times the top probability. `--ser 7,0.4` gives KL 0.086 and `--ser 6,0.4` gives KL 0.168. The speed gain is within run-to-run noise.
- `--kv q4` (set by the Windows launcher) stores the KV cache in 4 bits. Measured during gate T7 on the .99, branch `feat/ornith-122b` at `e82d1fc3`, fp32 against q4 on the same model and the same prompts. On 10 short prompts of 33 steps, top-1 agreement is 93.3 % (308/330) and the mean KL is 0.079 (0.013 to 0.361). On a prompt of 20,914 tokens, agreement is 30/32 and KL 0.093. On a prompt of 65,760 tokens, agreement is 31/32 and KL 0.073. The engine against llama.cpp gives KL 0.041, so q4 adds about 0.04, with no growth in KL as the context gets longer. The error comes from the prompt pass in q4: on the 20,914-token prompt, the output already diverges at step 0. The gap was accepted on 23 October 2026. The fp32 cache is unchanged.

Refusals at start (exit code 2, messages prefixed by `strata-q36:`):

- `--cache-prior` with two or more sessions (`--batch` 2 or more): it "needs one session".
- `--cache-prior` with MTP drafts (`--mtp N`, N above 0) on a model with an MTP block: "cannot be combined".
- `--cache-prior` when every expert fits in VRAM: it "needs the hybrid expert mode".
- `--cache-prior`, or `--ser` with Kmin below the 8 picks of the model: it "needs the fast routing path". That path is taken when `STRATA_Q36_OLD_KERNELS` is unset, the model has 256 experts and 8 picks. With `--ser 8,t`, the option is not refused.
- `--ser` not given as `Kmin,t` with 1 <= Kmin <= 8 and 0 <= t <= 1: "--ser must be Kmin,t".
- `--cache-prior` outside [0, 1]: "--cache-prior must be in [0, 1]".

Environment variables. The q36 engine reads 18 of them: 17 `STRATA_Q36_*` and `STRATA_DMA_BATCH`. Unset means the default shown.

- `STRATA_Q36_BLAS_F32`: off. Set, the router product uses the hipBLAS / cuBLAS SGEMM instead of the native kernel (`q36_engine.cu:1933`, A/B check).
- `STRATA_Q36_CHUNK=N`: forces the prompt chunk, from 256 to 4096 (`:4768`). Unset, `--chunk auto` picks it from the free VRAM (`:4721`).
- `STRATA_Q36_DRAFT_VOCAB=N`: the MTP drafts pick among the first N token ids, used when `--draft-vocab` is absent (`:4835`). Unset, 0: the whole vocabulary (`:2548`).
- `STRATA_Q36_FIXED_SLOTS=1`: freezes the expert slots as placed. The identity gate uses it (`:2942`).
- `STRATA_Q36_KEEP_PACKED`: unset, the packed Q8_0 copies are freed when that gives more expert slots (`:3541`). Set, they are kept.
- `STRATA_Q36_NO_EXT`: unset, a decode batch of up to 8 rows takes the router and the shared expert's gate in one product (`:2441`). Set, separate products.
- `STRATA_Q36_NO_MMQ`: unset, the MMQ kernels are used when built (`:1749`). Set, they are not.
- `STRATA_Q36_NO_MTP_CPU`: unset, the experts of the MTP block may go to the CPU pool (`:3519`, `:3570`). Set, they may not.
- `STRATA_Q36_NO_PREFETCH`: unset, the experts of the next layer are prefetched (`:3771`). Set, no prefetch.
- `STRATA_Q36_OLD_KERNELS`: unset, the fast decode kernels and the fast routing path are used (`:1971`). Set, the old kernels, and `--cache-prior` and `--ser` are refused.
- `STRATA_Q36_ONE_TOKEN_CPU`: off. Set, the one-token AVX-512 CPU kernels are used. They ran slower than ggml-cpu on the Ornith box (comment at `:3716`, `native_expert_ff1024_test --bench`).
- `STRATA_Q36_PLACE_DEBUG`: off. Set, prints the placement (`:4769`).
- `STRATA_Q36_READ_WINDOWS=N`: default 1, from 0 to 64 (`:3967`). The decode windows run between two chunks of a prompt read.
- `STRATA_Q36_STATS=1`: off. Prints one line per window, 48 steps then 96: the bytes each path reads and the host's wait on the doorbell (`src/qwen35/q36_stats.hpp:24`, `:1530`).
- `STRATA_Q36_STREAM_ALL`: off. Set, the whole layer streams from host memory, the first version, PCIe-bound (`:2619`).
- `STRATA_Q36_VERIFY`: off. Set, the decode kernels are compared with the old kernel's bits (`:2077`), on the device and with CUDA graphs on; its mismatch lines print at the end of each window (decode window, MTP step or prompt chunk), not at the comparison, so a fatal CUDA error inside a window loses them. This also disables the fused router path (`:2553`).
- `STRATA_DMA_BATCH=N`: the grouping of PCIe copies, modes 0, 1 and 2 (`include/strata/core/dma_batch.hpp:32`). The q36 default is 1, one `cudaMemcpyBatchAsync` per group (`q36_engine.cu:2697`). 0 gives one copy each.

`STRATA_Q36_CPU_POOL` is a compile-time macro (`q36_engine.cu:46`), not an environment variable.

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

Other levers measured: `STRATA_DMA_BATCH=1` against 0 has no measurable effect, and neither has `--pcie-frac 0.50` against the automatic share (issue #80). The automatic share now settles at 0.45. The former default settled at 0.80 and gave 22.4 tokens/s, against 24.6. The best fixed value measured is 0.50 to 0.55. The branch also prefetches the experts of the next layer, groups the copies, and drops a single-pass CPU pool per layer, which gave no gain. The pool runs 6 threads by default (`kDefaultPoolWorkers`, `src/qwen35/q36_threads.hpp:10`), capped by the worker processors present.

Not reached: 100 tokens/s. The rate is 30.0 tokens/s at 1,024 tokens. Its open issues are #77 (CPU kernels of ik_llama: the pool stays the first cost, 20.8 ms), #79 (CPU pool), #80 (PCIe share), #81 (attention at long context), #74 (MTP), #75 (REAP) and #78 (speculative decoding). #76 (experts in 2 bits) was measured and set aside: 40.65 against 28.45 tokens/s, KL 0.4697 against 0.0423 (docs/ORNITH_122B.md).

MTP is off in the configuration. With `STRATA_Q36_ONE_TOKEN_CPU=1`, its output is identical, but 20 to 35 % slower, with no drop in missed bytes (issue #74).

## Non-finite logits

A row of logits with no finite maximum, or a NaN, draws no token. The engine prints `ERR the logits hold no finite maximum or a NaN: no token drawn` (`src/qwen35/q36_engine.cu:726`, printed at `:4531` and `:4624`) and prints no DONE line for that request. In the batch path the slot ends with `BDONE <slot> <tokens> error <ms>` (`q36_engine.cu:4090`), and the server does not reuse that slot's state (`serve/server.py:1782`).

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
