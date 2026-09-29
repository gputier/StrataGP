<h1 align="center">StrataGP</h1>

<p align="center"><sub>A fork of <a href="https://github.com/Niko1221/Strata">Strata</a> by Niko1221</sub></p>

<p align="center"><b>Run a 125-billion-parameter AI model on a normal gaming PC</b><br>
one NVIDIA card (12-24 GB) + 64 GB of RAM · Windows or Linux · one click to install</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that StrataGP's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

StrataGP runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on your own PC. It writes its answers at **60-95 tokens per second** (a token is about ¾
of a word): faster than you can read.

- **Free and open source.**

> **Jump to:** [What StrataGP adds](#what-stratagp-adds) · [How fast?](#how-fast-is-it) ·
> [Which model?](#which-model-should-i-pick) · [Install](#install) · [Using it](#using-it) ·
> [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) · [All the details](docs/DETAILS.md)

---

## What StrataGP adds

StrataGP is a fork of **[Strata](https://github.com/Niko1221/Strata) by Niko1221**. The engine, the installer, the
app, the measurements and the [paper](docs/paper/Strata-Paper.pdf) are his work; StrataGP follows it (currently engine
**0.1.21**, upstream commit `f1b1d96`, including its experimental layer split: one model across two or three GPUs, see
[docs/MULTI_GPU.md](docs/MULTI_GPU.md)) and adds a round of performance and precision work on top of it.

**How it was done.** The engine was read end to end and audited: [docs/AUDIT-PERF.md](docs/AUDIT-PERF.md) lists what
could be faster or more exact, with an estimate for each item. That gave 53 issues; 52 of them were worked on, in 16
packages (plus a build fix that upstream also made itself). [docs/PERF-CHANGES.md](docs/PERF-CHANGES.md) has the full table (issue by issue: what changed, what is
on by default, the exact switch), and [docs/perf/](docs/perf/) has one page per package.

**What kind of changes**, in plain words:

- **Fewer round trips while writing an answer.** Each "guess, then check" step used to stop several times so the
  processor and the graphics card could wait for each other; most of these waits are gone (`round-sync`), and many
  small GPU steps are batched or merged (`window-batching`, `qsa-small`).
- **SSD reads off the critical path.** The few rows of the lookup table on the SSD are now read while the graphics
  card works, instead of before it starts (`ple-io`, io_uring on Linux).
- **Faster GPU kernels for the experts** kept on the graphics card, for Q2_0 and for the IQ sizes (`grouped-experts`,
  `iq-kernels`).
- **A faster sampler** when the answer is sampled (temperature, top-k, top-p) rather than greedy (`sampler`).
- **Long context:** the attention over long conversations reads its data in larger, overlapped pieces (`qsa-longctx`).
- **Big GPUs such as the RTX 5090** (170 SMs): some work was sized for a 48-SM card and left most of a big card idle;
  it now scales with the card (`grids`).
- **CPU expert kernels:** better load balance across threads, large memory pages on Linux, memory prefetching and
  faster IQ kernels (`cpu`).
- **Reading your prompt:** fewer synchronisations and CUDA calls per layer, and several prompt kernels rewritten
  (`prefill-kernels`, `prefill-host`, `prefill-dense`).
- **The server:** the conversation is no longer re-tokenized from scratch, nor the answer re-decoded, at every
  request or token (`server-tools`), plus a few tools (`calibrate.py --spec`, `make_profile.py`, `mtp_pack.py`).
- **Correctness fixes** (`correctness` and others): a GPU barrier that could be skipped, NaN turned into -0 in a BF16
  conversion, a missing bounds check in long-context attention, fixed-size tables without a guard in the CPU expert
  pool, `--spec` without a draft layer, and missing parity tests for production kernels (now added).

**What is on by default.** A change that gives **exactly the same result**, bit for bit, is **on by default**, and
each one has a `STRATA_OLD_*` switch (an environment variable) that brings back the old code, so you can compare
both. A change that **alters the numbers**, even slightly (lower precision, another rounding), is **off by default**
and has to be turned on (`--idx-fp16`, `--prefill-dense-mmq`, `--gdn-state-bf16` and a few variables). All the
switches are listed in [docs/DETAILS.md](docs/DETAILS.md#engine-switches-stratagp).

**First GPU run: one RTX 5090, Windows, 29 September 2026.** The changes were written on a machine without a
graphics card; they have now run on one RTX 5090 (Ryzen 9 9950X3D, 128 GB, IQ3_S, 262K context, greedy, 256 tokens,
upstream `f1b1d96` built with the same compiler as the reference). Several GPUs have still not been tried. Measured,
mean of 3 runs:

| What | Upstream | StrataGP | Gain |
| --- | ---: | ---: | ---: |
| Writing answers (3 prompts: 32, 2.5K, 23K tokens) | 115 / 132 / 140 tok/s | 122 / 151 / 155 tok/s | **+5% to +15%** |
| Reading your prompt, 2.5K tokens | 1,397 tok/s | 1,982 tok/s | **+42%** |
| Reading your prompt, 23K tokens | 2,074 tok/s | 3,090 tok/s | **+49%** |

The audit had estimated +20-40% for writing and +10-20% for reading; it is the other way round. The CPU kernel
numbers (+25% to +88% for Q2_0, +14% to +137% for IQ2/IQ3) were measured on an Intel VM, for the kernel alone.

With the expert cache held at the same size in both engines, two of the three prompts give exactly upstream's
tokens; the third diverges at its 229th token, with or without the `STRATA_OLD_*` switches. The run also found and
fixed a crash of the prompt path when it has its own buffers (the multi-GPU case). Details and what is left:
[docs/PERF-CHANGES.md, section 6](docs/PERF-CHANGES.md#6-première-exécution-sur-gpu-rtx-5090-29092026).

**The ready-made engine is upstream's.** `START-HERE.bat` downloads Niko1221's release build, which has none of
these changes; only the server and tools of StrataGP apply. To run StrataGP's engine, compile it:
`START-HERE.bat --setup --build`.


## How fast is it?

These are **upstream's measurements** (Strata engine 0.1.14, before StrataGP's changes) on an RTX 5070 (12 GB), a
Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 90 tokens/s | 67 tokens/s | 1,310 tokens/s |
| **IQ2_XS** | 74 tokens/s | 60 tokens/s | 1,240 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 46 tokens/s | 1,110 tokens/s |
| **IQ3_S** | 52 tokens/s | 41 tokens/s | 1,070 tokens/s |
| **Coder** (IQ1_M) | 51 tokens/s | 44 tokens/s | 1,300 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 740-1,000 tokens/s. A 32K prompt takes about 25 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](docs/DETAILS.md#speed-measured).

Every PC is different: `START-HERE.bat --calibrate` measures a few engine settings on yours and keeps the fastest
(about 5-10 minutes; on the PC above it made the Coder 7% faster).

**Two or three NVIDIA cards?** `START-HERE.bat --setup --gpus 0,2` splits the model's layers across them (experimental):
each card keeps the experts of its own layers, and prompts flow through the cards in a pipeline. On an RTX 5080 +
RTX 3090 it read prompts 18-20% faster than the 5080 alone, with decoding on par. See [docs/MULTI_GPU.md](docs/MULTI_GPU.md).

## Which model should I pick?

**The size** (the same model, compressed more or less):

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests (original model only) |

**Will it fit?** Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**,
the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your
**RAM is at least shard 1 + about 10 GB** for Windows and your other programs. With 64 GB of RAM every size fits
(IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster, but it doesn't
lower the RAM needed.

**The version:**

- **Qwen3.8-Flash-Next** - the original.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding
  version: half of the experts removed, keeping the ones that code, tool use and images need (91% of the full model's
  SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its experts stored like IQ3_S):
  shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on 64 GB, and reads long prompts
  the fastest of all. Weaker outside coding.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI
  that thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
  token, and about the same RAM as the same size of the original (no IQ3_S). Its own license applies (see its page).

Not sure? Take **IQ2_XS** - or the **Coder** if you mainly write code, or have 32-48 GB of RAM. You can add another
one later with `SETUP.bat` (the same as `START-HERE.bat --setup`; on Linux `./setup.sh --setup`).

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](docs/ORCA.md).
It needs an explicit packing conversion and is not an installer menu option.

## Install

**You need:** an NVIDIA RTX 30, 40 or 50 card with 12 GB of VRAM or more, enough RAM for the size you pick (above),
~80 GB of free disk space (an SSD makes the first start much faster), and Windows 10/11 or Linux. The only thing you
install yourself is a current **NVIDIA driver** ([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA
App). Everything else - Python, the engine, the model - is set up for you.

**Windows**

1. [Download this project](https://github.com/gputier/StrataGP/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer a few questions - or just press Enter each time for the recommended choice:
   - **Which model and size?** The original or Swift 1.5, and Q2_0, IQ2_XS, IQ3_XXS or IQ3_S - see [above](#which-model-should-i-pick)
   - **How much context?** How much text it can keep in mind at once (it suggests one for your card)
   - **Images?** Whether it should also read pictures
   - **Experimental speed projection?** Off unless you say yes - [read what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. Your browser opens the StrataGP app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): StrataGP
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, just double-click `START-HERE.bat` again: it starts right away, nothing is downloaded twice. Close its
window to stop the model.

**Updating:** download the new version and unzip it anywhere (or `git pull`), then run `START-HERE.bat` in it. The
model files are kept in a `Strata-data` folder next to your StrataGP folder, so a new copy finds them and sets itself up
the same way - nothing big is downloaded again.

**Linux:** run `./setup.sh` - same questions, same result.

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The StrataGP app's Monitor tab next to a coding agent"><br>
<sub>The StrataGP app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8080` - the StrataGP app (it opens by itself when the model starts): **Chat**, a
  live **Monitor** of the model and your GPU/CPU/RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv\Scripts\python chat.py`
- **Your apps and coding agents:** add it as an "OpenAI-compatible" provider with base URL
  **`http://127.0.0.1:8080/v1`**, any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **off, low, medium or high** - in the chat page menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; high is best for hard questions.
- **Pictures:** in the chat page click **Picture**; in `chat.py` type `/image <path>`; in apps just attach them.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`, then open the
  address the server window prints; see the [details](docs/DETAILS.md#using-it).
- **Experimental speed projection (off by default):** an experimental control vector that setup can turn on; it
  changes how the model answers - read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first.

**Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute per
30,000 tokens); after that it keeps the conversation and reads only what is new, so follow-ups start in seconds.

## Something went wrong?

**My PC froze, or got very slow, the first time StrataGP started.**
That's normal while it starts, most of all the first time. StrataGP loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes. **Wait, and don't close the
window.** The next starts are much faster. Still frozen after 10 minutes? Restart the PC, close other programs
(browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `START-HERE.bat` again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
StrataGP is already running. Look for its window.

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: StrataGP starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `SETUP.bat` and pick more
context.

**Still stuck?** Look in the [full troubleshooting table](docs/DETAILS.md#troubleshooting), or open an issue and
attach `strata-<model>.log` from the StrataGP folder.

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. StrataGP makes it fit by **sharing the work across your whole PC** - the same idea as a kitchen, where the
things you use all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word it writes needs only 10 of them.
  So it doesn't have to have all of them on the graphics card at once.
- **Your graphics card** does the part of the work needed for every word, and keeps the few thousand experts that
  are asked most often. It keeps learning which ones those are while you use it.
- **Your RAM** holds every expert. When a word needs one the card doesn't have, **your processor** works on it -
  at the same time as the graphics card, so neither waits for the other.
- **Your SSD** holds a big lookup table; the model only reads a few small rows of it per word.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check.** A small, fast helper built into the model guesses the next few words, and the big model
  checks all the guesses in one go. It keeps the ones it agrees with and writes the next word itself - so one step
  often produces several words. The helper only guesses - the big model decides every word - so you get the same
  quality answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens - pieces of words - at a time), which is why a long
  document or code base is read at over 1,000 tokens per second.

Want the full picture? The [details](docs/DETAILS.md#how-it-works) explain every part and its numbers, and the
[paper](docs/paper/Strata-Paper.pdf) tells the whole story, with the measurements behind it.

## Credits

- Engine: [Strata](https://github.com/Niko1221/Strata) by Niko1221 - StrataGP is a fork of it; the engine, the
  installer, the app and the paper are his work.
- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI. Their licenses apply
  to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).

## License

StrataGP is open source under the [MIT License](LICENSE). A few parts carry their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml), the web app's font (SIL Open Font License 1.1) and the experimental speed projection's
vector in `data/experimental-speed-projection` (Qwen Community License 1.0, from the model's activations). The
models are not part of this repository; each model's own license applies to its files.
