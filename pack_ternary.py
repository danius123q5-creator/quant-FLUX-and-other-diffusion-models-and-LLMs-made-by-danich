# -*- coding: utf-8 -*-
"""Упаковать fp16 FLUX в НАШ тернарный формат (−1/0/+1, base-3, 1.6 бит/вес).
Большие 2D-веса → тернар+scale (group32), остальное bf16 как есть. Грузится
нашей нодой XQuantTernaryLoader (см. custom_nodes/ComfyUI-XQuant). ~2ГБ из 22.7.

Формат .safetensors:
  <k>                — неквантованные тензоры (bf16, как есть)
  <k>||qpack (uint8) — упакованные троичные коды
  <k>||qscl  (fp16)  — масштабы per-group32
  <k>||qshp  (int64) — orig shape [rows, cols]
  __metadata__: {"xq_keys": json[...], "xq_group":"32", "xq_pad_per_key": json{...}}
"""
import sys, re, json, torch, numpy as np
sys.path.insert(0, r"D:/ComfyBot/comfyui_portable/ComfyUI_windows_portable")
from safetensors import safe_open
from safetensors.torch import save_file
import xquant as xq

SRC = sys.argv[1] if len(sys.argv)>1 else r"D:/Comfy/models/unet/flux1-dev.safetensors"
DST = sys.argv[2] if len(sys.argv)>2 else r"D:/Comfy/models/unet/flux1-dev-TERNARY.xqt.safetensors"
GROUP = 32

f = safe_open(SRC, framework="pt")
out = {}; qkeys = []; pads = {}; nq = 0
for k in f.keys():
    W = f.get_tensor(k)
    # УНИВЕРСАЛЬНАЯ защита критических слоёв (все архитектуры) — xquant.is_critical.
    if (k.endswith(".weight") and W.dim()==2 and W.numel()>=4096
            # ffn добавлен 2026-08-01: писалось под FLUX, где полносвязные зовутся
            # mlp, а в WAN они ffn — и мимо упаковки уходили САМЫЕ КРУПНЫЕ слои
            # (blocks.15.ffn.0 = 44 млн весов, вчетверо больше attention). Файл
            # выходил заметно меньше сжатым, а картинка ЛУЧШЕ настоящих 1.6 бита —
            # то есть ошибка была бы В ПОЛЬЗУ проверяемой гипотезы, самый обидный вид.
            and re.search(r"attn|mlp|ffn|linear|qkv|proj", k)
            and not xq.is_critical(k)):
        arr = W.float().numpy()
        q, scale, gpad = xq.our_quantize_ternary(arr, group=GROUP)   # −1/0/+1 + scale
        packed, ppad = xq.pack_tern5(q)                              # base-3 5/байт
        out[f"{k}||qpack"] = torch.from_numpy(packed)                # uint8
        out[f"{k}||qscl"]  = torch.from_numpy(scale.reshape(-1)).to(torch.float16)
        out[f"{k}||qshp"]  = torch.tensor(list(arr.shape), dtype=torch.int64)
        pads[k] = [int(gpad), int(ppad)]                             # pad группы, pad упаковки
        qkeys.append(k); nq += 1
    else:
        out[k] = W
meta = {"xq_keys": json.dumps(qkeys), "xq_group": str(GROUP), "xq_pads": json.dumps(pads), "xq_format":"ternary_b3"}
print(f"тернар-упаковано: {nq} тензоров")
save_file(out, DST, metadata=meta)
import os
print(f"ГОТОВО: {DST}  =  {os.path.getsize(DST)/1e9:.1f} ГБ  (из {os.path.getsize(SRC)/1e9:.1f})")
