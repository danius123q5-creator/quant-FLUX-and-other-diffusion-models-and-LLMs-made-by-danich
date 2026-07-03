# -*- coding: utf-8 -*-
"""XQuant STANDALONE — torch-free ужиматель для сборки в .exe (PyInstaller).
Читает safetensors напрямую numpy (bf16/f16/f32/fp8), детектит арх по ключам,
жмёт нашим ядром (xquant), пишет GGUF. Кидаешь bf16 → получаешь <модель>-Q2_K.gguf.

Deps (лёгкие, без torch): numpy, gguf, safetensors(header-only). → exe ~150МБ.
Запуск:  xquant_standalone.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
"""
import os, sys, json, struct, numpy as np
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xquant as xq
import xgguf                                   # НАШ GGUF-писатель (без чужой gguf-либы)

QUANT_THRESHOLD = 1024
OUR = {"Q8_0": (xq.our_quantize_q8_0, xgguf.T.Q8_0, 32),
       "Q6_K": (xq.our_quantize_q6k, xgguf.T.Q6_K, 256),
       "Q5_0": (xq.our_quantize_q5_0, xgguf.T.Q5_0, 32),
       "Q4_0": (xq.our_quantize_q4_0, xgguf.T.Q4_0, 32),
       "Q3_K": (xq.our_quantize_q3k, xgguf.T.Q3_K, 256),
       "Q2_K": (xq.our_quantize_q2k, xgguf.T.Q2_K, 256)}

_BITS = [("8-бит (Q8_0) — почти без потерь","Q8_0"), ("6-бит (Q6_K) — высокое","Q6_K"),
         ("5-бит (Q5_0) — высокое","Q5_0"), ("4-бит (Q4_0) — идеал/баланс","Q4_0"),
         ("3-бит (Q3_K) — компактно","Q3_K"), ("2-бит (Q2_K) — минимум","Q2_K")]

def find_model_dirs():
    """Авто-детект папок моделей ComfyUI + LM Studio (существующие)."""
    home = os.path.expanduser("~")
    cands = []
    for ev in ("XQUANT_MODELS_DIR", "COMFYUI_MODELS", "COMFYUI_MODELS_DIR"):
        if os.environ.get(ev): cands.append(os.environ[ev])
    cands += [
        os.path.join(home, ".lmstudio", "models"),                 # LM Studio
        os.path.join(home, ".cache", "lm-studio", "models"),
        os.path.join(home, "ComfyUI", "models"),                   # ComfyUI
        os.path.join(home, "Documents", "ComfyUI", "models"),
        "D:/Comfy/models", "D:/ComfyUI/models", "C:/ComfyUI/models",
        "D:/ComfyBot/comfyui_portable/ComfyUI_windows_portable/ComfyUI/models",
    ]
    seen = set(); out = []
    for d in cands:
        d = os.path.normpath(d)
        if d not in seen and os.path.isdir(d): seen.add(d); out.append(d)
    return out

def scan_models(dirs, cap=400):
    """Найти .safetensors/.gguf в папках (с ограничением глубины/кол-ва)."""
    found = []
    for base in dirs:
        for root, dnames, files in os.walk(base):
            if root[len(base):].count(os.sep) > 4: dnames[:] = []; continue
            for f in files:
                fl = f.lower()
                if fl.endswith((".safetensors", ".gguf")) and "mmproj" not in fl:
                    try: sz = os.path.getsize(os.path.join(root, f)) / 1e9
                    except Exception: sz = 0
                    if sz >= 0.05:                                  # мелочь пропускаем
                        found.append((f"{f}  ({sz:.1f}ГБ)", os.path.join(root, f)))
                    if len(found) >= cap: return found
    return found

def run_gui(prefill=""):
    """Полноценное окно: выбор файла + битность + прогресс. Сжатие в потоке."""
    import tkinter as tk
    from tkinter import ttk, filedialog
    import threading, queue
    root = tk.Tk(); root.title("XQuant — ужиматель моделей")
    root.geometry("560x420"); root.minsize(520, 380)
    try: root.eval('tk::PlaceWindow . center')
    except Exception: pass

    tk.Label(root, text="XQuant", font=("Segoe UI", 18, "bold")).pack(pady=(14,0))
    tk.Label(root, text="diffusion model quantizer  •  own engine  •  AGPL-3.0",
             font=("Segoe UI", 9), fg="#888").pack()

    frm = tk.Frame(root); frm.pack(fill="x", padx=18, pady=(14,4))
    tk.Label(frm, text="Модель (.safetensors):", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
    path_var = tk.StringVar(value=prefill)
    ent = tk.Entry(frm, textvariable=path_var, width=48); ent.grid(row=1, column=0, sticky="we", pady=2)
    def browse():
        p = filedialog.askopenfilename(filetypes=[("Модели","*.safetensors *.gguf"),
                                                  ("Safetensors","*.safetensors"),
                                                  ("GGUF (LLM)","*.gguf"),("All","*.*")])
        if p: path_var.set(p)
    tk.Button(frm, text="Обзор…", command=browse).grid(row=1, column=1, padx=(6,0))
    frm.columnconfigure(0, weight=1)

    # авто-поиск моделей ComfyUI/LM Studio
    mf = tk.Frame(root); mf.pack(fill="x", padx=18, pady=(2,0))
    tk.Label(mf, text="Из ComfyUI/LM Studio:", font=("Segoe UI", 9), fg="#888").grid(row=0, column=0, sticky="w")
    found_map = {}
    fnd_var = tk.StringVar(value="")
    fnd_cb = ttk.Combobox(mf, textvariable=fnd_var, values=[], state="readonly", width=44)
    fnd_cb.grid(row=1, column=0, sticky="we", pady=2)
    def on_pick(_=None):
        p = found_map.get(fnd_var.get())
        if p: path_var.set(p)
    fnd_cb.bind("<<ComboboxSelected>>", on_pick)
    def do_scan():
        import threading
        def w():
            dirs = find_model_dirs()
            items = scan_models(dirs)
            found_map.clear()
            for label, p in items: found_map[label] = p
            fnd_cb["values"] = list(found_map.keys())
            fnd_cb.set(f"найдено {len(items)} моделей — выбери" if items else "не нашёл (жми Обзор)")
        threading.Thread(target=w, daemon=True).start()
    tk.Button(mf, text="🔍 Найти", command=do_scan).grid(row=1, column=1, padx=(6,0))
    mf.columnconfigure(0, weight=1)
    do_scan()   # скан при открытии

    bf = tk.Frame(root); bf.pack(fill="x", padx=18, pady=6)
    tk.Label(bf, text="Битность:", font=("Segoe UI", 10)).pack(side="left")
    bit_var = tk.StringVar(value=_BITS[3][0])
    ttk.Combobox(bf, textvariable=bit_var, values=[b[0] for b in _BITS],
                 state="readonly", width=32).pack(side="left", padx=8)

    log = tk.Text(root, height=9, width=64, font=("Consolas", 9), bg="#111", fg="#ddd")
    log.pack(fill="both", expand=True, padx=18, pady=(8,4))
    q = queue.Queue()
    def logline(m): q.put(m)
    btn = tk.Button(root, text="СЖАТЬ", font=("Segoe UI", 11, "bold"), height=1)
    btn.pack(pady=(0,12))

    def worker(src, qn):
        # сжатие в ОТДЕЛЬНОМ процессе, прогресс через файл → GUI не фризит совсем
        import subprocess, tempfile, time
        pf = tempfile.NamedTemporaryFile(prefix="xquant_", suffix=".log", delete=False)
        pf.close(); progfile = pf.name
        open(progfile, "w").close()
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, src, qn, progfile]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), src, qn, progfile]
        flags = 0x08000000 if os.name == "nt" else 0              # CREATE_NO_WINDOW
        try:
            p = subprocess.Popen(cmd, creationflags=flags)
            pos = 0
            while True:
                try:
                    with open(progfile, "r", encoding="utf-8", errors="replace") as fp:
                        fp.seek(pos); new = fp.read(); pos = fp.tell()
                    for ln in new.splitlines():
                        if ln.strip(): logline(ln)
                except Exception: pass
                if p.poll() is not None:
                    # дочитать хвост
                    try:
                        with open(progfile, "r", encoding="utf-8", errors="replace") as fp:
                            fp.seek(pos)
                            for ln in fp.read().splitlines():
                                if ln.strip(): logline(ln)
                    except Exception: pass
                    break
                time.sleep(0.2)
        except Exception as e:
            logline(f"ОШИБКА: {e}")
        finally:
            try: os.remove(progfile)
            except Exception: pass
        q.put(("__done__",))
    def start():
        src = path_var.get().strip('"')
        if not os.path.isfile(src): logline("Выбери файл модели!"); return
        qn = dict(_BITS)[bit_var.get()]
        log.delete("1.0","end"); btn.config(state="disabled", text="жму…")
        threading.Thread(target=worker, args=(src, qn), daemon=True).start()
    btn.config(command=start)

    def poll():
        try:
            while True:
                m = q.get_nowait()
                if isinstance(m, tuple): btn.config(state="normal", text="СЖАТЬ")
                else: log.insert("end", m+"\n"); log.see("end")
        except queue.Empty: pass
        root.after(120, poll)
    poll()
    root.mainloop()

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
    raise SystemExit("Unknown model architecture")

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

def compress_file(src, qn, log=print):
    """Сжать модель src в битность qn. log(msg) — колбэк прогресса. Возвращает dst."""
    keys = [k for k,_,_,_ in _iter_hdr(src)]
    pfx = strip_prefix(keys)
    arch = detect_arch([k[len(pfx):] if pfx else k for k in keys])
    log(f"{os.path.basename(src)}  arch={arch}  ->  {qn}")
    fn, ggtype, blk = OUR.get(qn, (None,None,32))
    nq = nf = 0; out = []; total = 0
    for k, dt, shape, raw in load_tensors(src):
        if pfx and not k.startswith(pfx): continue
        key = k[len(pfx):] if pfx and k.startswith(pfx) else k
        if dt not in ("F32","F16","BF16","F8_E4M3","F8_E5M2"): continue
        data = _decode(raw, dt, shape)
        nd = len(shape); npm = int(np.prod(shape)) if shape else 1
        if (fn and nd==2 and npm>QUANT_THRESHOLD and not CRITICAL(key) and shape[1] % blk == 0):
            try: out.append((key, ggtype, tuple(shape), fn(data).tobytes())); nq += 1
            except Exception: out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data))); nf += 1
        elif nd == 1 or npm <= QUANT_THRESHOLD:
            out.append((key, xgguf.T.F32, tuple(shape), xgguf.enc_f32(data)))
        else:
            out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data)))
        total += 1
        if total % 100 == 0: log(f"  обработано {total} тензоров...")
    dst = os.path.splitext(src)[0] + f"-{qn}.gguf"
    log(f"пишу GGUF ({len(out)} тензоров)...")
    xgguf.write_gguf(dst, arch, out)
    log(f"ГОТОВО: {os.path.basename(dst)}  {os.path.getsize(src)/1e9:.1f} -> {os.path.getsize(dst)/1e9:.1f} ГБ")
    return dst

def llm_critical(name):
    """Критические слои LLM → держим F16 (вход-эмбеды/выход/нормы), не крушим."""
    n = name.lower()
    return ("token_embd" in n or n == "output.weight" or "_norm" in n
            or ".norm" in n or "rope_freqs" in n)

def requantize_gguf(src, qn, log=print):
    """LLM-сжатие: реквантизация существующего GGUF (F16/BF16/Q8 → Q4/Q3/Q2...).
    Метадата+токенайзер копируются сырыми байтами (llama.cpp/LM Studio читают)."""
    f, ver, raw_meta, n_kv, tinfos, data_start, align = xgguf.read_gguf(src)
    fsize = os.path.getsize(src)
    offs = [t[3] for t in tinfos]
    fn, ggtype, blk = OUR.get(qn, (None, None, 32))
    out = []; nq = nf = npass = 0
    log(f"{os.path.basename(src)}  тензоров={len(tinfos)}  -> {qn} (LLM requant)")
    for i, (name, dims, tt, off) in enumerate(tinfos):
        start = data_start + off
        end = data_start + offs[i+1] if i+1 < len(tinfos) else fsize
        f.seek(start); raw = f.read(end - start)
        shape = tuple(reversed(dims))                    # ggml ne → numpy shape
        nelem = int(np.prod(shape)) if shape else 1
        data = xgguf.dec_source(raw, tt, nelem)
        if data is None:                                 # уже квант/неизвестный → как есть
            out.append((name, tt, dims, raw)); npass += 1
        elif fn and len(shape) == 2 and nelem > QUANT_THRESHOLD and not llm_critical(name) and shape[1] % blk == 0:
            try: out.append((name, ggtype, dims, fn(data.reshape(shape)).tobytes())); nq += 1
            except Exception: out.append((name, xgguf.T.F16, dims, xgguf.enc_f16(data))); nf += 1
        elif len(shape) == 1:
            out.append((name, xgguf.T.F32, dims, xgguf.enc_f32(data))); nf += 1
        else:
            out.append((name, xgguf.T.F16, dims, xgguf.enc_f16(data))); nf += 1
        if (i+1) % 100 == 0: log(f"  requant {i+1}/{len(tinfos)}...")
    f.close()
    dst = os.path.splitext(src)[0] + f"-{qn}.gguf"
    log(f"пишу LLM GGUF ({len(out)} тензоров, метадата+токенайзер сохранены)...")
    xgguf.write_gguf_raw(dst, raw_meta, n_kv, out)
    log(f"ГОТОВО: {os.path.basename(dst)}  {fsize/1e9:.1f} -> {os.path.getsize(dst)/1e9:.1f} ГБ  (квант {nq}, F16 {nf}, как-есть {npass})")
    return dst

def process(src, qn, log=print):
    """.gguf → LLM requant (метадата+токенайзер сохранены); .safetensors → диффузия."""
    if src.lower().endswith(".gguf"):
        return requantize_gguf(src, qn, log)
    return compress_file(src, qn, log)

def main():
    # CLI: <файл> <битность> [progfile] → жмём. Иначе → GUI.
    if len(sys.argv) > 2 and os.path.isfile(sys.argv[1].strip('"')):
        src = sys.argv[1].strip('"'); qn = sys.argv[2].upper()
        progfile = sys.argv[3] if len(sys.argv) > 3 else None
        if progfile:
            def log(m):
                try:
                    with open(progfile, "a", encoding="utf-8") as pf: pf.write(m + "\n")
                except Exception: pass
            process(src, qn, log)
        else:
            def log(m): print(m, flush=True)
            process(src, qn, log)
        return
    prefill = sys.argv[1].strip('"') if len(sys.argv) > 1 and os.path.isfile(sys.argv[1].strip('"')) else ""
    run_gui(prefill)

def _iter_hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n))
    for k,m in hdr.items():
        if k!="__metadata__": yield k, m["dtype"], m["shape"], None

if __name__ == "__main__":
    main()
