# -*- coding: utf-8 -*-
"""XQuant STANDALONE — torch-free ужиматель для сборки в .exe (PyInstaller).
Читает safetensors напрямую numpy (bf16/f16/f32/fp8), детектит арх по ключам,
жмёт нашим ядром (xquant), пишет GGUF. Кидаешь bf16 → получаешь <модель>-Q2_K.gguf.

Deps (лёгкие, без torch): numpy, gguf, safetensors(header-only). → exe ~150МБ.
Запуск:  xquant_standalone.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
"""
import os, sys, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xquant as xq
import xgguf                                   # НАШ GGUF-писатель (без чужой gguf-либы)

QUANT_THRESHOLD = 1024
OUR = {"Q4_0": (xq.our_quantize_q4_0, xgguf.T.Q4_0, 32),
       "Q3_K": (xq.our_quantize_q3k, xgguf.T.Q3_K, 256),
       "Q2_K": (xq.our_quantize_q2k, xgguf.T.Q2_K, 256)}

# ── детект архитектуры по ключам (без torch; порт city96 keys_detect) ──
ARCHES = [
    ("flux",        [("double_blocks.0.img_attn.proj.weight",)], ["transformer_blocks.0.attn.norm_added_k.weight"]),
    ("qwen_image",  [("transformer_blocks.0.attn.add_q_proj.weight", "time_text_embed.timestep_embedder.linear_1.weight")], []),
    ("sd3",         [("joint_blocks.0.x_block.attn.qkv.weight",)], []),
    ("sdxl",        [("input_blocks.3.0.op.weight", "add_embedding.linear_1.weight")], []),
    ("sd1",         [("input_blocks.3.0.op.weight",)], []),
    ("wan",         [("blocks.0.self_attn.norm_q.weight", "head.modulation")], []),
]
def detect_arch(keys):
    ks = set(keys)
    for arch, detect, banned in ARCHES:
        for group in detect:
            if all(k in ks for k in group) and not any(b in ks for b in banned):
                return arch
    # fallback по prefix
    if any("double_blocks" in k for k in ks): return "flux"
    if any("transformer_blocks" in k for k in ks): return "sd3"
    if any("input_blocks" in k for k in ks): return "sdxl"
    raise SystemExit("Не распознал архитектуру модели")

def strip_prefix(keys):
    for p in ("model.diffusion_model.", "model."):
        if any(k.startswith(p) for k in keys): return p
    return ""

# ── ручное чтение safetensors → numpy fp32 (bf16/f16/f32/fp8) ──
def _decode(raw, dtype, shape):
    if dtype in ("F32","F32"): a = np.frombuffer(raw, np.float32)
    elif dtype == "F16": a = np.frombuffer(raw, np.float16).astype(np.float32)
    elif dtype == "BF16":
        u = np.frombuffer(raw, np.uint16).astype(np.uint32)
        a = (u << 16).view(np.float32)
    elif dtype in ("F8_E4M3","F8_E5M2"):
        a = _fp8_to_f32(np.frombuffer(raw, np.uint8), dtype)
    else:
        a = np.frombuffer(raw, np.float32)
    return a.reshape(shape)

def _fp8_to_f32(u8, dtype):
    # декод fp8 e4m3/e5m2 в fp32 таблицей (256 значений)
    e_bits = 4 if dtype=="F8_E4M3" else 5
    m_bits = 7 - e_bits; bias = (1 << (e_bits-1)) - 1
    tab = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if (b>>7)&1 else 1.0
        e = (b >> m_bits) & ((1<<e_bits)-1)
        m = b & ((1<<m_bits)-1)
        if e == 0: v = m / (1<<m_bits) * 2.0**(1-bias)
        else: v = (1 + m/(1<<m_bits)) * 2.0**(e-bias)
        tab[b] = s*v
    return tab[u8]

def load_tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, meta in hdr.items():
            if k == "__metadata__": continue
            s, e = meta["data_offsets"]
            f.seek(base + s); raw = f.read(e - s)
            yield k, meta["dtype"], meta["shape"], raw

CRITICAL = xq.is_critical  # универсальная защита слоёв

def main():
    if len(sys.argv) < 2:
        print("USAGE: xquant_standalone <model.safetensors> [Q4_0|Q3_K|Q2_K]"); return
    src = sys.argv[1].strip('"')
    qn = (sys.argv[2] if len(sys.argv) > 2 else "Q2_K").upper()
    if not os.path.isfile(src): print("НЕТ ФАЙЛА:", src); return
    keys = [k for k,_,_,_ in _iter_hdr(src)]
    pfx = strip_prefix(keys)
    arch = detect_arch([k[len(pfx):] if pfx else k for k in keys])
    print(f"XQUANT: {os.path.basename(src)}  арх={arch}  → {qn}  (движок: наш, без чужого)")
    fn, ggtype, blk = OUR.get(qn, (None,None,32))
    nq = nf = 0
    out = []   # (name, ggml_type, logical_shape, data_bytes)
    total = 0
    for k, dt, shape, raw in load_tensors(src):
        if pfx and not k.startswith(pfx):
            continue
        key = k[len(pfx):] if pfx and k.startswith(pfx) else k
        if dt not in ("F32","F16","BF16","F8_E4M3","F8_E5M2"):
            continue
        data = _decode(raw, dt, shape)
        nd = len(shape); npm = int(np.prod(shape)) if shape else 1
        if (fn and nd==2 and npm>QUANT_THRESHOLD and not CRITICAL(key) and shape[1] % blk == 0):
            try:
                packed = fn(data)                       # наше ядро → упакованные байты
                out.append((key, ggtype, tuple(shape), packed.tobytes())); nq += 1
            except Exception:
                out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data))); nf += 1
        elif nd == 1 or npm <= QUANT_THRESHOLD:
            out.append((key, xgguf.T.F32, tuple(shape), xgguf.enc_f32(data)))
        else:
            out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data)))
        total += 1
        if total % 100 == 0: print(f"  обработано {total} тензоров...", flush=True)
    dst = os.path.splitext(src)[0] + f"-{qn}.gguf"
    print(f"пишу GGUF ({len(out)} тензоров)...")
    xgguf.write_gguf(dst, arch, out)
    print(f"ужато нашим ядром: {nq} | F16: {nf}\nГОТОВО: {dst}  {os.path.getsize(src)/1e9:.1f}→{os.path.getsize(dst)/1e9:.1f} ГБ")

def _iter_hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n))
    for k,m in hdr.items():
        if k!="__metadata__": yield k, m["dtype"], m["shape"], None

if __name__ == "__main__":
    main()
