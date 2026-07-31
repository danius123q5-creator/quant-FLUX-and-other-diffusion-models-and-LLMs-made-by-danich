# -*- coding: utf-8 -*-
"""PBA — Perceptual Bit Allocation probe (gears2 experiment, 2026-07-20).

Гипотеза: перцептивная чувствительность слоя (насколько ХУЖЕ ГЛАЗУ при его
2-битном квантовании) — лучший сигнал для распределения бит, чем ошибка весов
(SMART) или активаций (imatrix). Diffusion-специфично + честно перцептивно.

Метод (data-driven, image-space):
  1. Рендерим FLUX.1-dev на фикс. промптах/сидах в FP (bf16) — эталон.
  2. Для каждого Linear-слоя: round-trip квантуем ЕГО веса в ~2 бита
     (per-group asymmetric, симуляция Q2), рендерим те же промпты, меряем
     LPIPS vs эталон, восстанавливаем веса.
  3. LPIPS-дельта слоя = его перцептивная чувствительность. Ранжируем.
Сравнение с imatrix-ранком (activation p99/median) — коррелируют ли.

Запуск (python_embeded ComfyUI, torch cu13x + diffusers + lpips):
  python pba_probe.py                    # SMOKE: ~6 слоёв, 2 промпта, 8 шагов
  python pba_probe.py --full --prompts 5 --steps 20   # полный прогон (ночь)
"""
import argparse, time, json, sys, os
import numpy as np
import torch

_PENDING = r"D:\ComfyBot\bot\.pending_jobs.json"


def _bot_busy() -> bool:
    """True если у бота есть джоба в очереди/в работе — тогда уступаем GPU."""
    try:
        if os.path.exists(_PENDING) and os.path.getsize(_PENDING) > 3:
            return bool(json.load(open(_PENDING, encoding="utf-8")))
    except Exception:
        return False
    return False


def fake_quant_2bit(w: torch.Tensor, group: int = 64) -> torch.Tensor:
    """Per-group asymmetric 2-бит (4 уровня) quant->dequant — симуляция Q2-потери
    точности in-place, БЕЗ экспорта в GGUF. Мера чувствительности, не финальный кодек."""
    orig_dtype, shape = w.dtype, w.shape
    out, inn = shape
    wf = w.detach().float()
    pad = (group - inn % group) % group
    if pad:
        wf = torch.cat([wf, torch.zeros(out, pad, device=wf.device, dtype=wf.dtype)], dim=1)
    g = wf.reshape(out, -1, group)
    mn = g.amin(dim=2, keepdim=True)
    mx = g.amax(dim=2, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / 3.0            # 4 уровня: 0..3
    q = torch.round((g - mn) / scale).clamp(0, 3)
    dq = (q * scale + mn).reshape(out, -1)[:, :inn]
    return dq.reshape(shape).to(orig_dtype)


def _to_lpips(pil, dev):
    a = torch.from_numpy(np.asarray(pil).astype("float32") / 255.0)
    return (a.permute(2, 0, 1).unsqueeze(0) * 2 - 1).to(dev)   # [-1,1] NCHW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6, help="сколько слоёв в SMOKE")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--full", action="store_true", help="ВСЕ Linear (полный ночной прогон)")
    ap.add_argument("--out", default="pba_result.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import lpips
    # net='squeeze' (~5МБ) вместо 'alex' (233МБ) — сеть душит закачку, а squeeze для
    # РАНЖИРОВАНИЯ чувствительности слоёв полностью годен. 2026-07-20.
    lp = lpips.LPIPS(net="squeeze", verbose=False).to(dev)

    from diffusers import FluxPipeline
    print("[pba] загрузка FLUX.1-dev (bf16)…", flush=True)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    PROMPTS = [
        "a cinematic portrait of a woman in a rain-soaked neon city at night, sharp focus",
        "an epic fantasy castle on a cliff at sunrise, dramatic clouds, highly detailed",
        "a cozy cat on a wooden table by a window, soft daylight, photorealistic",
        "a rugged bearded man in a workshop, warm tungsten light, 50mm portrait",
        "a vast alien desert under two moons, sci-fi concept art, wide shot",
    ][: args.prompts]
    SEEDS = list(range(1, len(PROMPTS) + 1))

    # cpu_offload виснет на cu13x → идём как collect_imatrix: энкодим на GPU,
    # выгружаем текст-энкодеры, трансформер+VAE держим на GPU, рендерим через
    # готовые prompt_embeds (текст-энкодеры больше не нужны). 2026-07-20.
    print("[pba] энкожу промпты…", flush=True)
    pipe.text_encoder.to(dev); pipe.text_encoder_2.to(dev)
    ENC = []
    with torch.no_grad():
        for p in PROMPTS:
            pe, ppe, _tids = pipe.encode_prompt(prompt=p, prompt_2=p, device=dev,
                                                num_images_per_prompt=1, max_sequence_length=256)
            ENC.append((pe, ppe))
    # ОБНУЛЯЕМ текст-энкодеры (при prompt_embeds они не нужны) — иначе pipe._execution_device
    # резолвится в cpu из-за них → device-mismatch. Освобождает и VRAM/RAM.
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    import gc as _gc
    _gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    pipe.transformer.to(dev); pipe.vae.to(dev)
    print(f"[pba] трансформер+VAE на GPU (exec_device={pipe._execution_device}), текст-энкодеры сняты", flush=True)

    def render_all():
        out = []
        for (pe, ppe), s in zip(ENC, SEEDS):
            g = torch.Generator("cpu").manual_seed(s)
            im = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
                      num_inference_steps=args.steps, guidance_scale=3.5,
                      height=args.res, width=args.res, generator=g).images[0]
            out.append(im)
        return out

    t0 = time.time()
    print("[pba] эталонный рендер (FP)…", flush=True)
    base_t = [_to_lpips(im, dev) for im in render_all()]
    print(f"[pba] эталон готов +{time.time()-t0:.0f}s ({len(base_t)} шт)", flush=True)

    tr = pipe.transformer
    linears = [(n, m) for n, m in tr.named_modules() if isinstance(m, torch.nn.Linear)]
    if args.full:
        cands = linears
    else:
        want = (".0.attn.to_q", ".0.attn.to_v", ".0.ff.net", "transformer_blocks.9.attn.to_q",
                "single_transformer_blocks.0.", "single_transformer_blocks.15.")
        cands = [(n, m) for n, m in linears if any(k in n for k in want)][: args.layers]
        if not cands:
            cands = linears[: args.layers]
    # РЕЗЮМ: подхватываем уже посчитанное из --out (можно прерывать/продолжать).
    results = {}
    if os.path.exists(args.out):
        try:
            results = dict(json.load(open(args.out, encoding="utf-8")).get("lpips_per_layer", {}))
        except Exception:
            results = {}
    todo = [(n, m) for n, m in cands if n not in results]
    print(f"[pba] зондирую: всего {len(cands)}, готово {len(results)}, осталось {len(todo)}", flush=True)

    def _save(status="running"):
        ranked = sorted(results.items(), key=lambda x: -x[1])
        json.dump({"lpips_per_layer": results, "ranked": ranked,
                   "meta": {"prompts": len(PROMPTS), "steps": args.steps, "res": args.res,
                            "n_layers_done": len(results), "n_linear_total": len(linears),
                            "status": status, "elapsed_s": round(time.time() - t0)}},
                  open(args.out, "w"), indent=2)

    for i, (name, mod) in enumerate(todo, 1):
        if _bot_busy():   # юзер захотел генерить → уступаем GPU, сохраняемся, выходим
            print(f"[pba] БОТ ЗАНЯТ — уступаю GPU, сохранил {len(results)}/{len(cands)}. "
                  f"Продолжу с этого места позже.", flush=True)
            _save(status="yielded_to_bot")
            return
        orig = mod.weight.data.clone()
        mod.weight.data = fake_quant_2bit(orig)
        d = 0.0
        with torch.no_grad():
            for im, bt in zip(render_all(), base_t):
                d += lp(_to_lpips(im, dev), bt).item()
        d /= len(base_t)
        mod.weight.data = orig
        results[name] = d
        _save()   # инкрементально после КАЖДОГО слоя — прерывание не теряет прогресс
        print(f"  [{i}/{len(todo)}] {name}: LPIPS={d:.4f}  (+{time.time()-t0:.0f}s)", flush=True)

    _save(status="done")
    ranked = sorted(results.items(), key=lambda x: -x[1])
    print(f"\n[pba] ГОТОВО ({len(results)} слоёв) за {time.time()-t0:.0f}s -> {args.out}", flush=True)
    for n, d in ranked[:15]:
        print(f"  {d:.4f}  {n}", flush=True)


if __name__ == "__main__":
    main()
