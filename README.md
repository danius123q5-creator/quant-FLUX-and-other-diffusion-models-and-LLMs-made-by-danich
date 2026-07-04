# XQuant — custom diffusion-model quantizer (zero third-party engine)

Compress diffusion models (FLUX / SDXL / SD1.5 / SD3 / Qwen-Image / Wan …) to
**2 / 3 / 4-bit GGUF** with a quantization engine written **entirely from scratch** —
our own GGML-byte-exact kernel, our own GGUF writer, our own architecture detection.
Drop a `bf16`/`fp16` model, get a compressed `.gguf` that runs in ComfyUI.

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

## Results (FLUX.1-dev, tested by real generation)
| bits | size | PSNR vs fp16 | verdict |
|---|---|---|---|
| fp16 | 23.8 GB | — | reference |
| 4-bit Q4_0 | 6.4 GB | ~25 dB | perfect, indistinguishable |
| 3-bit Q3_K | ~5 GB | ~18 dB | visible grain — not recommended for FLUX |
| **2-bit Q2_K** | **4.0 GB** | ~19 dB | **size/quality sweet spot** |

**Post-hoc quantization floor ≈ 2-bit.** Below that (ternary 1.6-bit) quality
collapses without quantization-aware training.

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
