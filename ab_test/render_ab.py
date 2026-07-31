# -*- coding: utf-8 -*-
"""A/B рендер квантов FLUX: один сид, один промпт, разные GGUF.

Гоняет каждый вариант модели через ComfyUI и складывает картинки так, чтобы
их можно было листать рядом. Сравнение вслепую по файлам, без опоры на метрики
в весовом пространстве — они уже один раз обманули (Q2 против Q3).

Сцены подобраны под РАЗНЫЕ типы артефактов квантования:
  portrait — кожа и глаза, тут вылезает пластик и шум
  gradient — плавное небо, тут вылезает полосатость (бэндинг)
  texture  — мелкая фактура, тут вылезает каша
  text     — надпись, тут вылезает распад букв
"""
import os, sys, time, uuid

sys.path.insert(0, r"D:\ComfyBot")

OUT = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("flat",       "flux1-dev-Q4_0.gguf"),
    ("smart-v1old", "flux1-dev-Q4SMART.gguf"),
    ("qual-v1",    "flux-Q4_0-quality-v1.gguf"),
    ("qual-v2",    "flux-Q4_0-quality-v2.gguf"),
    ("balance-v2", "flux-Q4_0-balance-v2.gguf"),   # 6.34 ГБ — легче боевого
]

SCENES = {
    "portrait": "close-up portrait of an elderly fisherman, weathered skin, sharp eyes, "
                "natural window light, photorealistic, fine skin texture",
    "gradient": "empty desert at dawn, vast smooth gradient sky from deep blue to orange, "
                "no clouds, minimal composition, photorealistic",
    "texture":  "a mossy stone wall covered in lichen and tiny cracks, macro detail, "
                "natural light, photorealistic",
    "text":     "a vintage wooden shop sign that reads 'COFFEE', hand-painted letters, "
                "sharp focus, photorealistic",
}

SEED = 424242
STEPS = 20
SIZE = 1024


def build(unet, prompt, seed):
    from bot.txt2img import workflow as W
    wf = W.build_dream_workflow(prompt, steps=STEPS, size=SIZE, amount=1)
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        if ct in ("UnetLoaderGGUF", "UNETLoader"):
            wf[nid] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}}
        if "Sampler" in ct and "seed" in (node.get("inputs") or {}):
            node["inputs"]["seed"] = seed
        if ct == "SaveImage":
            node["inputs"]["filename_prefix"] = "abq"
    return wf


def run(wf, cid):
    """Через ПУЛ бота, а не напрямую в ComfyUI.

    Пул внутри зовёт gears.acquire/release, поэтому VRAM-бюджет знает про наши
    прогоны: тест не полезет на карту мимо учёта и не подерётся с генерацией
    пользователя — при нехватке памяти он просто подождёт своей очереди.
    """
    from bot.comfyui_api import comfy
    pid = comfy.queue_prompt(wf, cid, job_type="txt2img")
    imgs, _wall = comfy.poll_results(pid, cid, on_progress=None, timeout=600, workflow=wf)
    return imgs[0] if imgs else None


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for scene, prompt in SCENES.items():
        if only and only != scene:
            continue
        for tag, unet in MODELS:
            path = os.path.join(OUT, f"{scene}__{tag}.png")
            if os.path.exists(path):
                print(f"  пропуск (есть): {os.path.basename(path)}")
                continue
            t0 = time.time()
            try:
                data = run(build(unet, prompt, SEED), f"ab_{uuid.uuid4().hex[:8]}")
            except Exception as e:
                print(f"  ОШИБКА {scene}/{tag}: {e}")
                continue
            if data:
                open(path, "wb").write(data)
                print(f"  {scene:<9} {tag:<11} {time.time()-t0:5.1f}с  -> {os.path.basename(path)}")
            else:
                print(f"  {scene:<9} {tag:<11} пусто")
