# Жматель (XQuant) — custom diffusion-model quantizer (zero third-party engine)

**🇬🇧 English** · [🇷🇺 Русский](README.ru.md)

Compress diffusion models (FLUX / SDXL / SD1.5 / SD3 / Qwen-Image / Wan …) to
**2 / 3 / 4-bit GGUF** with a quantization engine written **entirely from scratch** —
our own GGML-byte-exact kernel, our own GGUF writer, our own architecture detection.
Drop a `bf16`/`fp16` model, get a compressed `.gguf` that runs in ComfyUI.

## Two modes

**🏆 Premium mode — `safetensors (bf16) → Q4_0`.** Uncompromising, magazine-grade
visuals. The model shrinks ~4× yet renders indistinguishably from the original,
because the quantizer works from the **full-precision source** and keeps the
critical layers (input embeddings, VAE output projection, norms) in `bf16`
straight from that source — nothing lossy ever touches them. This is the mode you
want for actual generation quality.

**♻️ Eco / Emergency mode — `GGUF → GGUF`.** Ultra-fast **local** re-compression of
a model you *already have* as a `.gguf` — no need to re-download the 15–24 GB
originals. Reads any source quant (`Q4_K_M`, `Q6_K`, …), repacks smaller, keeps the
tokenizer/metadata verbatim. Built for **weak PCs and slow internet**: save disk in
minutes, no giant downloads. Quality is capped by the source (re-quantizing an
already-lossy file compounds error) — so it's for *space*, not for *max fidelity*.

> Rule of thumb: **Premium** when you have the `bf16`/`fp16` original and want the
> best picture; **Eco** when you only have a `.gguf` and want it smaller, fast.

## Zero third-party engine
The standalone engine depends on **numpy only** — no `llama.cpp`, no `gguf` library,
no City96 `convert.py`, no torch:
- **`xquant.py`** — quantization kernel from scratch. Our `Q4_0` output is
  **byte-for-byte identical** to the GGUF reference; `Q3_K` / `Q2_K` verified via
  round-trip through the ComfyUI-GGUF decoder.
- **`xgguf.py`** — our own GGUF v3 writer (validated by an independent GGUF reader).
- **`xquant_standalone.py`** — own safetensors reader (manual `bf16`/`fp16`/`fp8`
  decode) + own architecture detection. **No external quantizer code.**

## LLM requantization (GGUF → GGUF, no re-download)
Drop an existing **LLM GGUF** (from LM Studio etc.) and shrink it further —
no need to download the raw 15 GB safetensors. XQuant reads the GGUF, dequantizes
weights in RAM, applies critical-layer protection, and repacks to any of
2/3/4/5/6/8-bit — **preserving all metadata and the tokenizer** verbatim (raw KV
passthrough), so the result loads in llama.cpp / LM Studio.

**Any source quant is unpacked** — our own numpy dequantizers for
`Q4_0` / `Q5_0` / `Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_0` / `F16` /
`BF16` / `F32` are **byte-for-byte identical** to the `gguf` reference
(`dequantize`), so even a `Q4_K_M` from LM Studio re-packs correctly.
```
XQuant.exe model-Q4_K_M.gguf Q3_K     # existing LM-Studio quant → smaller, tokenizer intact
```
Best source is still **F16 or Q8_0** — re-quantizing an already-lossy K-quant
(e.g. `Q4_K_M → Q2_K`) compounds error (double loss); the tool warns you when
the source is already low-bit.

## Universal critical-layer protection
Input embeddings, the output projection to the VAE (`final_layer` / `conv_out` /
`proj_out`), and norms are kept in `bf16` for **every** architecture — so the VAE
connection is never broken (the classic sub-4-bit "colour noise").

## Smart bit allocation — SMART (on by default)
Not every layer deserves the same bits. Before writing, XQuant runs a cheap
**Q4 probe** over each 2-D weight and measures the *absolute* reconstruction
error `‖W − dequant(quant(W))‖` — how much a layer actually resists quantization.
Then it **reallocates bits, size-neutrally**:

- **hard / high-impact layers → one step up** (e.g. `Q4_0 → Q5_0`) — protected;
- **easy / low-impact ("dumb") layers → one step down** (e.g. `Q4_0 → Q3_K`) — squeezed;
- the bytes spent upgrading are paid for by the bytes freed downgrading → **same file size, less total distortion.**

This is data-free (no calibration needed) — importance is read straight from the
weights, in the spirit of AWQ's "keep the salient weights, quantize the rest
harder." On real FLUX.1-dev at `Q4_0` base it upgrades ~40 critical layers and
downgrades ~90 low-impact ones at **net-zero size**; on a synthetic mix it cuts
total weight distortion **~19 % at equal size**. Works for every base
`Q2_K…Q6_K`. Disable with `XQUANT_SMART=0`.

**Modes** (pick from the GUI dropdown, or `XQUANT_SMART_MODE=`):

| mode | what it does | result |
|---|---|---|
| **⚖ Balance** (default) | protect salient / squeeze dumb, size-neutral | same size, better image |
| **🤏 Shrink** | squeeze dumb harder — the dumbest go **down two steps** (`Q4_0→Q2_K`), few upgrades | **smaller file**, salient still protected |
| **📦 Q3** | moderately press dumb layers down, keep salient — targets ~Q3 size with a living image | **compact**, keeps quality |
| **🔥 Extreme shrink** | press ~85 % of layers down, most of them two steps | **smallest file**, quality traded for size |
| **💎 Quality** | upgrade more salient layers, few downgrades | a bit bigger, max fidelity |
| **▦ Flat** | uniform quant, no reallocation | legacy behaviour |

**Real sizes (FLUX.1-dev):** `Q2_K` ≈ **4.5 GB**, `Q3_K` ≈ **5.3–5.8 GB**, `Q4_0` ≈ **6.3 GB**.
The real size win is **Q2_K (−29 %)**, not Q3 (Q3 saves only 0.5–1 GB vs Q4).

**Visual test verdict (2026-07-19):** error metrics (cosine / weighted error) **massively
overstate** low-bit damage. Weighted error screamed "Q3 = 413 % of Q4" — yet by eye Q3 is
perfectly usable, the image doesn't fall apart. imatrix gives a visible edge (cleaner skin,
no artifacts), most of all on **Q2** (where it guards the activation outliers that data-free
methods can't see). Judge quality **by eye, not by the metric**.

## imatrix — activation-aware importance (AWQ-grade)
SMART reads importance from the *weights*. **imatrix** reads it from the
*activations* — the real signal flowing through each layer — which is what makes
low-bit actually hold up. Feed an imatrix and the per-group solver (`Q2_K`/`Q3_K`)
weights each input channel by `sum(act²)` instead of `x²`:

- On synthetic Q2_K, activation-weighted error drops **~55 %** with imatrix.
- **Q2 becomes usable** (~4 GB from a 6.8 GB Q4) while keeping the picture;
  **Q3 stops graining.** This is the "much smaller *and* keeps quality" lever.

**Collecting one — two ways:**

1. **Standalone script (no ComfyUI node needed)** — run `collect_imatrix.py` (or
   double-click `Собрать-imatrix.bat`) with a CUDA torch + diffusers env (e.g. the
   `python_embeded` from ComfyUI-portable). It runs a few FLUX transformer passes on
   varied prompts, hooks every `Linear`, and writes `<model>.imatrix.npy`:
   ```
   python collect_imatrix.py --model black-forest-labs/FLUX.1-dev --out flux1-dev.imatrix.npy
   ```
   > Note: `cpu_offload` can segfault with hooks on new torch (cu13x) — the script
   > avoids it by keeping only the transformer on GPU (~24 GB, fits a 24-32 GB card).
2. **ComfyUI node** — add the **XQuant imatrix: Capture** node before your `KSampler`
   and **Save** after it, run 1–3 generations → `<model>.imatrix.npy`.

Point `XQuant.exe` at the file (🎯 imatrix field) or set `XQUANT_IMATRIX=<path>`. It composes
with **any** mode — pair it with 🤏 Shrink / 🔥 Extreme for the smallest file that
still looks right. Data-free fallback (weight `x²`) when no imatrix is given.

## LoRA-compatible
Quantizing the base model does **not** break LoRAs. A LoRA is a separate low-rank
delta applied on top of the weights at compute time:

```
W' = W_quant  +  (B·A)·scale
     ↑ Q4/Q2 base   ↑ LoRA stays fp16, added over the dequantized base
```

The quant only touches the **base** weights; the LoRA matrices stay full-precision
and their learned correction is added cleanly on the fly (ComfyUI-GGUF dequantizes
the base per-op and applies the delta). LoRAs are additive corrections, so they're
robust to the small base shift from quantization — a fp16-trained LoRA runs fine on
a Q4 (even Q2) base. Tested in production: an identity LoRA on a Q4 FLUX works as
expected. The **only** thing that breaks a LoRA is *merging* it into the weights and
then re-quantizing (double loss) — keep them separate: `Q4 gguf + fp16 LoRA`.

## Results (FLUX.1-dev, real side-by-side generation, same prompt+seed)
| bits | size | verdict |
|---|---|---|
| fp16 | 23.8 GB | reference |
| **4-bit Q4_0** | **6.4 GB** | ✅ **recommended — stable, quality, indistinguishable from fp16** |
| 3-bit Q3_K | ~5–6 GB | ❌ visible grain — worse than Q2 despite more bits; not recommended |
| 2-bit Q2_K | 4.0 GB | clean too (landscapes + close-up faces ≈ Q4), slightly softer fine detail |

**Q4_0 is the go-to: maximum quality with real stability** — it's what the
production bot runs on. Q2_K is a pleasant surprise (clean at 4.0 GB, great when
VRAM is tight or a face-swap will overwrite the base face anyway). Q3_K is the odd
one out — our symmetric 3-bit encode grains more than 2-bit, so skip it.

> ⚠️ **Weight PSNR is not a reliable quality ranking.** By weight-error Q3_K looks
> "better" than Q2_K, but real renders show the opposite. Trust the image (use the
> **🖼 Real-test** button), not the number.

**Post-hoc quantization floor ≈ 2-bit.** Below that (ternary 1.6-bit) quality
collapses without quantization-aware training.

### Showcase — 2-bit vs 4-bit (FLUX.1-dev)
![Q2_K vs Q4_0](media/q2_vs_q4.png)

Left **Q2_K (4.0 GB)**, right **Q4_0 (6.4 GB)** — same prompt, same seed. Even at
2-bit the render stays clean: straight masts, sharp planks, no sky grain; the
close-up keeps skin pores, brow/beard strands and catch-lights in the eyes. Q4_0 is
a touch crisper on the very finest hairs, for +2.4 GB.

> **Honest note:** this is a *quality* showcase, not a pixel-identical A/B. 2-bit
> perturbs the weights enough that the same seed takes a slightly different latent
> path (the man turns to a different pose), so the two frames aren't the same
> composition — they show that 2-bit holds *fidelity*, not that it reproduces the
> exact fp16 image. Q2_K here is the standard GGML Q2_K format, just encoded well
> with critical-layer protection from the bf16 source — no new magic, solid
> engineering.

## Research — how FLUX dies at 1-bit

A separate hands-on study probing the limits below Q2_K: what happens when you binarize
FLUX.1-dev to **1 bit** with this engine. Short version — pure 1-bit doesn't work, but it
dies in three distinct, structured ways (collapse-to-mean **void**, variance-blowup
**noise**, spatial-routing loss **tile**), and **attention turns out far more fragile than
MLP** (binarizing 12 % of weights as attention kills the model; 24 % as MLP leaves it
clean). Judged by rendered images, not weight-space cosine (which misleads).

Plus: attention can be **rescued training-free** with an orthogonal "box" (incoherence
rotation), while **adaLN is the hard wall** the box can't break (a magnitude problem, not
an outlier one). Full write-up, method and images: **[RESEARCH-1bit-flux.md](RESEARCH-1bit-flux.md)**.

> Honesty guard: this is an **experiment log**, not a compression win — the "alive" runs
> are effectively ~10–14 bit (fatter than Q2_K). It's a *sensitivity map*, and the
> deployable floor remains Q2_K.

## Usage
**Easiest — the standalone `XQuant.exe`** (20 MB, no Python needed, numpy bundled):
drag a `.safetensors` model onto `XQuant.exe` → get `<model>-Q2_K.gguf` next to it.
Or from a terminal: `XQuant.exe <model.safetensors> [Q4_0|Q3_K|Q2_K]`.

With Python instead:
```
python xquant_standalone.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
```
Output: `<model>-<qtype>.gguf` next to the input.

## Quality test (which bit level to pick)
The GUI has a **🧪 Quality test** button. It samples the model's largest weight
tensors and, for every bit level, quantizes → dequantizes → reports the
reconstruction error and PSNR with a plain verdict — all offline (no GPU, no
generation), in seconds. Verdicts are calibrated against real FLUX A/B renders:

```
bit    bytes/w   deviation   PSNR      verdict
Q8_0   1.06       0.6%       69 dB     flawless
Q4_0   0.56       8.9%       45 dB     clean, no visible loss
Q3_K   0.43      18.6%       39 dB     visible grain
Q2_K   0.33      33.3%       33 dB     heavy grain / mush
```
Weight PSNR isn't linear with image quality, so the thresholds are tuned so
"clean" ends where FLUX renders actually start to grain — Q4_0 is the practical
floor for indistinguishable FLUX.

### Real-generation test (🖼)
If a local ComfyUI is running (`:8000`, or `XQUANT_COMFY_URL`), the **🖼 Real-test**
button generates the *same* prompt+seed on every compressed quant of your model it
finds in ComfyUI (FLUX; auto-detects the CLIP/VAE) and shows the actual frames
side by side — a real photo A/B, not just PSNR. Compress the bit levels first, then
click it.

## Loading in ComfyUI
Copy `comfyui-node/ComfyUI-XQuant` into `ComfyUI/custom_nodes/`. It adds
**`XQuant GGUF Loader`** — pick a `.gguf` and it dequantizes on load and builds
the model with ComfyUI's own machinery (`load_diffusion_model_state_dict`), so the
architecture (FLUX / SD3 / SDXL / …) is auto-detected from the tensor keys. Our
`Q4_0` GGUF also loads fine with City96's stock `UnetLoaderGGUF`.

### Audio / music models
The quant kernel is architecture-agnostic, so it can compress audio-diffusion /
music models too. The catch is the **loader**, not the compression:
- A model **already in GGUF** (music LLMs, Whisper, MusicGen-GGUF) → the
  `.gguf → .gguf` requant path shrinks it and it runs in its native llama.cpp /
  whisper.cpp runtime **today**.
- An **audio-diffusion `.safetensors`** (Stable Audio, ACE-Step) → XQuant will
  compress it (arch tag `unknown`), and the dedicated **`XQuant Music Loader`**
  node loads it back. ComfyUI natively detects Stable Audio (by
  `transformer.rotary_pos_emb.inv_freq`) and ACE-Step (by `genre_embedder.weight`)
  from the reconstructed state-dict, so the compressed DiT loads through the same
  diffusion machinery (KSampler → `VAEDecodeAudio`). Load the VAE/conditioner
  separately, as usual. GPT-class music models (Bark, MusicGen) are **not**
  diffusion — they need their own runtime and won't load through this node.

Verified: compressing a real audio model (Bark) works end-to-end (1.8 → 0.3 GB,
valid GGUF). The Music Loader path reuses the exact dequant+load mechanism proven
on FLUX; test it against your own Stable Audio / ACE-Step checkpoint.

## License — AGPL-3.0
Licensed under **GNU AGPL-3.0**. Anyone who uses, modifies, or serves this code
over a network **must release their full source under the same license.**
Commercial closed-source use is not permitted.

*(Optional `xquant_tool.py` variant reuses City96's ComfyUI-GGUF `convert.py`,
Apache-2.0 — see `NOTICE`. The standalone engine above uses none of it.)*

## Contacts
For **commercial licensing** and custom integration inquiries, contact me on
Telegram: **[@GarrysmodMapper](https://t.me/GarrysmodMapper)**.

Try the models in action — our Telegram AI-image bot:
**[t.me/comfuibot](https://t.me/comfuibot)** (FLUX / Qwen image & video generation,
running on the very quants this tool produces).
