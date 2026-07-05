# -*- coding: utf-8 -*-
"""XQuant STANDALONE — torch-free ужиматель для сборки в .exe (PyInstaller).
Читает safetensors напрямую numpy (bf16/f16/f32/fp8), детектит арх по ключам,
жмёт нашим ядром (xquant), пишет GGUF. Кидаешь bf16 → получаешь <модель>-Q2_K.gguf.

Deps (лёгкие, без torch): numpy, gguf, safetensors(header-only). → exe ~150МБ.
Запуск:  xquant_standalone.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
"""
import os, sys, json, struct, time, re, numpy as np

def _fmt_dur(sec):
    """Секунды → человекочитаемо: '45.3с' / '2м 05с' / '1ч 03м'."""
    sec = max(0.0, float(sec))
    if sec < 60:   return f"{sec:.1f}с"
    m, s = divmod(int(sec), 60)
    if m < 60:     return f"{m}м {s:02d}с"
    h, m = divmod(m, 60)
    return f"{h}ч {m:02d}м"
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
         ("3-бит (Q3_K) — компактно","Q3_K"), ("2-бит (Q2_K) — минимум","Q2_K"),
         ("1-бит (Q1) — ЭКСПЕРИМЕНТ, демо-качества (не сжатие)","Q1")]

# ── СМЕШАННЫЙ квант (аналог Q3_K_M): чувствительные слои — на тип выше ──
# Слои, что кормят ОСТАТОЧНЫЙ поток (выходные проекции attn + down-проекции MLP),
# при 3/2-бит дают основной шум/«зерно». Держим их на ступень выше — зерно уходит,
# размер растёт незначительно. Выкл: XQUANT_MIXED=0. 2026-07-04.
_SENSITIVE_RE = re.compile(
    # diffusion: FLUX (img/txt_attn.proj, img/txt_mlp.2, single linear2) + generic DiT
    r"attn\.proj\.weight$|_mlp\.2\.weight$|linear2\.weight$"
    r"|\.to_out\.0\.weight$|ff\.net\.2\.weight$|mlp\.fc2\.weight$"
    # LLM (llama.cpp ggml-ключи): ffn_down / attn_output / attn_v
    r"|ffn_down\.weight$|attn_output\.weight$|attn_v\.weight$",
    re.IGNORECASE)
_UPGRADE = {"Q3_K": "Q5_0", "Q2_K": "Q4_0"}   # база → тип для чувствительных

def _eff_quant(key, qn):
    """Эффективный квант слоя: чувствительным при Q3_K/Q2_K даём ступень выше."""
    if os.environ.get("XQUANT_MIXED", "1").strip().lower() in ("0","off","no","false"):
        return qn
    up = _UPGRADE.get(qn)
    if up and _SENSITIVE_RE.search(key):
        return up
    return qn

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
    root = tk.Tk(); root.title("Жматель — ужиматель моделей")
    root.geometry("565x625"); root.minsize(565, 625)   # всегда открывать в этом размере
    try: root.eval('tk::PlaceWindow . center')
    except Exception: pass

    tk.Label(root, text="Жматель", font=("Segoe UI", 18, "bold")).pack(pady=(14,0))
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

    # папка результата (пусто = рядом с исходником)
    of = tk.Frame(root); of.pack(fill="x", padx=18, pady=(0,2))
    tk.Label(of, text="Папка результата:", font=("Segoe UI", 9), fg="#888").grid(row=0, column=0, sticky="w")
    out_var = tk.StringVar(value="")
    tk.Entry(of, textvariable=out_var).grid(row=1, column=0, sticky="we", pady=2)
    def browse_out():
        d = filedialog.askdirectory(title="Куда сохранить результат")
        if d: out_var.set(d)
    tk.Button(of, text="Обзор…", command=browse_out).grid(row=1, column=1, padx=(6,0))
    def open_out():
        d = out_var.get().strip() or (os.path.dirname(path_var.get().strip('"')) if path_var.get() else "")
        if d and os.path.isdir(d):
            try: os.startfile(d)
            except Exception as e: logline(f"открыть не удалось: {e}")
    tk.Button(of, text="📂 Открыть", command=open_out).grid(row=1, column=2, padx=(6,0))
    tk.Label(of, text="(пусто = рядом с исходником)", font=("Segoe UI", 8), fg="#666").grid(row=2, column=0, sticky="w")
    of.columnconfigure(0, weight=1)

    log = tk.Text(root, height=9, width=64, font=("Consolas", 9), bg="#111", fg="#ddd")
    log.pack(fill="both", expand=True, padx=18, pady=(8,4))
    q = queue.Queue()
    def logline(m): q.put(m)
    btnrow = tk.Frame(root); btnrow.pack(pady=(0,12))
    btn = tk.Button(btnrow, text="СЖАТЬ", font=("Segoe UI", 11, "bold"), height=1, width=14)
    btn.pack(side="left", padx=4)
    test_btn = tk.Button(btnrow, text="🧪 Тест качества", font=("Segoe UI", 10), height=1)
    test_btn.pack(side="left", padx=4)
    real_btn = tk.Button(btnrow, text="🖼 Реал-тест", font=("Segoe UI", 10), height=1)
    real_btn.pack(side="left", padx=4)
    _imgrefs = []   # держим ссылки на PhotoImage, иначе GC съест картинки

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
        od = out_var.get().strip()
        if od and os.path.isdir(od): os.environ["XQUANT_OUT_DIR"] = od       # наследуется subprocess'ом
        else: os.environ.pop("XQUANT_OUT_DIR", None)
        log.delete("1.0","end"); btn.config(state="disabled", text="жму…")
        threading.Thread(target=worker, args=(src, qn), daemon=True).start()
    btn.config(command=start)

    def run_test():
        src = path_var.get().strip('"')
        if not os.path.isfile(src): logline("Выбери файл модели!"); return
        if src.lower().endswith(".gguf"):
            logline("Тест качества — по .safetensors-исходнику (не по .gguf)."); return
        log.delete("1.0","end")
        btn.config(state="disabled"); test_btn.config(state="disabled", text="считаю…")
        def w():
            try: quality_test(src, logline)
            except Exception as e: logline(f"ОШИБКА теста: {e}")
            finally: q.put(("__test_done__",))
        threading.Thread(target=w, daemon=True).start()
    test_btn.config(command=run_test)

    def show_images(results):
        # показать сгенерированные кадры рядом в отдельном окне (главный поток)
        if not results:
            return
        win = tk.Toplevel(root); win.title("Реал-тест — сравнение кадров")
        win.configure(bg="#111")
        row = tk.Frame(win, bg="#111"); row.pack(padx=10, pady=10)
        for label, path in results:
            col = tk.Frame(row, bg="#111"); col.pack(side="left", padx=6)
            tk.Label(col, text=label, fg="#ddd", bg="#111", font=("Segoe UI", 11, "bold")).pack()
            try:
                img = tk.PhotoImage(file=path)          # Tk 8.6 читает PNG
                while img.width() > 460:                 # ужать до вменяемого превью
                    img = img.subsample(2)
                _imgrefs.append(img)                     # защита от GC
                tk.Label(col, image=img, bg="#111").pack()
            except Exception as e:
                tk.Label(col, text=f"(не показать: {e})", fg="#f88", bg="#111").pack()
            tk.Label(col, text=os.path.basename(path), fg="#666", bg="#111", font=("Consolas",7)).pack()

    def run_realtest():
        src = path_var.get().strip('"')
        if not os.path.isfile(src): logline("Выбери файл модели!"); return
        log.delete("1.0","end")
        btn.config(state="disabled"); test_btn.config(state="disabled")
        real_btn.config(state="disabled", text="генерю…")
        def w():
            res = []
            try: res = real_gen_test(src, logline)
            except Exception as e: logline(f"ОШИБКА реал-теста: {e}")
            finally: q.put(("__images__", res))
        threading.Thread(target=w, daemon=True).start()
    real_btn.config(command=run_realtest)

    last_prog = [False]   # прошлая строка была прогресс-тиком → обновляем на месте
    def poll():
        try:
            while True:
                m = q.get_nowait()
                if isinstance(m, tuple):
                    btn.config(state="normal", text="СЖАТЬ")
                    test_btn.config(state="normal", text="🧪 Тест качества")
                    real_btn.config(state="normal", text="🖼 Реал-тест")
                    last_prog[0] = False
                    if m and m[0] == "__images__":
                        try: show_images(m[1])
                        except Exception as _se: log.insert("end", f"показ не удался: {_se}\n")
                else:
                    is_prog = ("тензоров..." in m and "/" in m)   # «обработано N/M тензоров...»
                    if is_prog and last_prog[0]:
                        log.delete("end-1l", "end")   # заменить прошлый тик → счётчик тикает на месте
                    log.insert("end", m+"\n"); log.see("end")
                    last_prog[0] = is_prog
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
    return "unknown"  # незнакомая арх (напр. аудио) — жмём всё равно, метку в GGUF
                      # пишем "unknown"; загрузка зависит от поддержки в ComfyUI

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

def _out_path(src, qn):
    """Путь результата: env XQUANT_OUT_DIR если задан, иначе рядом с исходником."""
    base = os.path.splitext(os.path.basename(src))[0] + f"-{qn}.gguf"
    od = os.environ.get("XQUANT_OUT_DIR", "").strip()
    if od and os.path.isdir(od): return os.path.join(od, base)
    return os.path.splitext(src)[0] + f"-{qn}.gguf"

def compress_file(src, qn, log=print):
    """Сжать модель src в битность qn. log(msg) — колбэк прогресса. Возвращает dst."""
    t0 = time.time()
    keys = [k for k,_,_,_ in _iter_hdr(src)]
    pfx = strip_prefix(keys)
    arch = detect_arch([k[len(pfx):] if pfx else k for k in keys])
    mixed = qn in _UPGRADE and os.environ.get("XQUANT_MIXED","1").strip().lower() not in ("0","off","no","false")
    log(f"{os.path.basename(src)}  arch={arch}  ->  {qn}{' (mixed: чувств.→'+_UPGRADE[qn]+')' if mixed else ''}")
    base_fn = OUR.get(qn, (None,None,32))[0]
    n_total = len(keys)
    nq = nf = nup = 0; out = []; total = 0
    for k, dt, shape, raw in load_tensors(src):
        if pfx and not k.startswith(pfx): continue
        key = k[len(pfx):] if pfx and k.startswith(pfx) else k
        if dt not in ("F32","F16","BF16","F8_E4M3","F8_E5M2"): continue
        data = _decode(raw, dt, shape)
        nd = len(shape); npm = int(np.prod(shape)) if shape else 1
        eff = _eff_quant(key, qn) if base_fn else qn        # чувствит. слой → ступень выше
        fn, ggtype, blk = OUR.get(eff, (None,None,32))
        if (fn and nd==2 and npm>QUANT_THRESHOLD and not CRITICAL(key) and shape[1] % blk == 0):
            try:
                out.append((key, ggtype, tuple(shape), fn(data).tobytes())); nq += 1
                if eff != qn: nup += 1
            except Exception: out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data))); nf += 1
        elif nd == 1 or npm <= QUANT_THRESHOLD:
            out.append((key, xgguf.T.F32, tuple(shape), xgguf.enc_f32(data)))
        else:
            out.append((key, xgguf.T.F16, tuple(shape), xgguf.enc_f16(data)))
        total += 1
        if total % 10 == 0: log(f"  обработано {total}/{n_total} тензоров...")
    dst = _out_path(src, qn)
    log(f"пишу GGUF ({len(out)} тензоров{', '+str(nup)+' чувств.→'+_UPGRADE[qn] if nup else ''})...")
    xgguf.write_gguf(dst, arch, out)
    log(f"ГОТОВО: {os.path.basename(dst)}  {os.path.getsize(src)/1e9:.1f} -> {os.path.getsize(dst)/1e9:.1f} ГБ  за {_fmt_dur(time.time()-t0)}")
    return dst

# ── Q1 (experimental) — рецепт из нашего 1-бит исследования (RESEARCH-1bit-flux.md):
#    MLP → 1-бит (пер-канальный бинар; MLP оказался робастным), attention → Q4
#    (хрупкий — держим 4 бита), критика/остальное → fp16. Вывод — SIM-safetensors
#    (fp16, ПОЛНЫЙ размер): это ДЕМО качества экстремального кванта, НЕ сжатие
#    (в GGUF нет 1-бит типа, грузить нечем). Для оценки, не для деплоя. 2026-07-05.
def _bin_per_channel(w2d, outlier_p=0.02):
    """Пер-канальный бинар: sign(w)·mean(|w_row|); топ-2% |w| держим точными. fp32."""
    scale = np.abs(w2d).mean(axis=1, keepdims=True).astype(np.float32)
    rec = (np.sign(w2d) * scale).astype(np.float32)
    flat = w2d.reshape(-1); rf = rec.reshape(-1)
    k = max(1, int(flat.size * outlier_p))
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    rf[idx] = flat[idx]
    return rf.reshape(w2d.shape)

def _q4_roundtrip(data, shape):
    """Q4_0 квант→деквант (attention держим в 4 битах). Возвращает fp32."""
    blk = xq.our_quantize_q4_0(np.ascontiguousarray(data, np.float32).reshape(-1))
    return xgguf._deq_q4_0(blk.tobytes(), int(np.prod(shape)))

def compress_1bit(src, qn, log=print):
    """Q1 (experimental): MLP→1-бит, attn→Q4, критика→fp16. Стриминг sim-safetensors."""
    t0 = time.time()
    log("!! Q1 — ЭКСПЕРИМЕНТАЛЬНЫЙ режим (research-only).")
    log("   Рецепт из нашего 1-бит исследования: MLP -> 1-бит, attention -> Q4, критика -> fp16.")
    log("   Вывод: sim-safetensors в fp16 (ПОЛНЫЙ размер, НЕ сжатие — в GGUF нет 1-бит типа).")
    log("   Это ДЕМО КАЧЕСТВА экстремального кванта, не для продакшена. Деплой-дно = Q2_K.")
    keys = [k for k,_,_,_ in _iter_hdr(src)]
    pfx = strip_prefix(keys)
    # pass 1 — заголовок (всё F16, формы из источника; тот же порядок, что load_tensors)
    hdr = {}; off = 0
    for k, dt, shape, _ in _iter_hdr(src):
        if pfx and not k.startswith(pfx): continue
        key = k[len(pfx):] if pfx and k.startswith(pfx) else k
        nb = int(np.prod(shape)) * 2 if shape else 2
        hdr[key] = {"dtype":"F16","shape":list(shape),"data_offsets":[off, off+nb]}
        off += nb
    hj = json.dumps(hdr, separators=(",",":")).encode("utf-8"); hj += b" "*((8-len(hj)%8)%8)
    dst = os.path.splitext(_out_path(src, qn))[0] + ".safetensors"
    n_total = len(hdr); nbin = nq4 = nkeep = 0; done = 0
    with open(dst, "wb") as w:
        w.write(struct.pack("<Q", len(hj))); w.write(hj)
        for k, dt, shape, raw in load_tensors(src):
            if pfx and not k.startswith(pfx): continue
            key = k[len(pfx):] if pfx and k.startswith(pfx) else k
            data = _decode(raw, dt, shape)
            nd = len(shape); npm = int(np.prod(shape)) if shape else 1
            nl = key.lower()
            if nd == 2 and npm > QUANT_THRESHOLD and not CRITICAL(key) and shape[1] % 32 == 0:
                if "mlp" in nl:
                    rec = _bin_per_channel(np.ascontiguousarray(data, np.float32).reshape(shape)); nbin += 1
                elif "attn" in nl:
                    rec = _q4_roundtrip(data, shape); nq4 += 1
                else:
                    rec = data; nkeep += 1
            else:
                rec = data; nkeep += 1
            w.write(np.ascontiguousarray(rec, np.float32).astype(np.float16).tobytes()); del data, rec
            done += 1
            if done % 10 == 0: log(f"  обработано {done}/{n_total} (MLP-бинар {nbin}, attn-Q4 {nq4})...")
    log(f"ГОТОВО (sim): {os.path.basename(dst)}  MLP-бинар={nbin}, attn-Q4={nq4}, fp16={nkeep}  за {_fmt_dur(time.time()-t0)}")
    log("   ЗАГРУЗКА: обычный UNETLoader (это fp16-safetensors). Ждать деградацию — это 1-бит демо.")
    return dst

def llm_critical(name):
    """Критические слои LLM → держим F16 (вход-эмбеды/выход/нормы), не крушим."""
    n = name.lower()
    return ("token_embd" in n or n == "output.weight" or "_norm" in n
            or ".norm" in n or "rope_freqs" in n)

def requantize_gguf(src, qn, log=print):
    """LLM-сжатие: реквантизация существующего GGUF (F16/BF16/Q8 → Q4/Q3/Q2...).
    Метадата+токенайзер копируются сырыми байтами (llama.cpp/LM Studio читают)."""
    t0 = time.time()
    f, ver, raw_meta, n_kv, tinfos, data_start, align = xgguf.read_gguf(src)
    fsize = os.path.getsize(src)
    offs = [t[3] for t in tinfos]
    base_fn = OUR.get(qn, (None, None, 32))[0]
    mixed = qn in _UPGRADE and os.environ.get("XQUANT_MIXED","1").strip().lower() not in ("0","off","no","false")
    out = []; nq = nf = npass = nup = 0; src_types = {}
    log(f"{os.path.basename(src)}  тензоров={len(tinfos)}  -> {qn} (LLM requant){' mixed' if mixed else ''}")
    for i, (name, dims, tt, off) in enumerate(tinfos):
        start = data_start + off
        end = data_start + offs[i+1] if i+1 < len(tinfos) else fsize
        f.seek(start); raw = f.read(end - start)
        shape = tuple(reversed(dims))                    # ggml ne → numpy shape
        nelem = int(np.prod(shape)) if shape else 1
        if len(shape) == 2 and nelem > QUANT_THRESHOLD:
            src_types[tt] = src_types.get(tt, 0) + 1     # тип больших source-тензоров
        # реквантуем ТОЛЬКО большие 2D-веса; всё остальное (крит-слои, нормы,
        # мелочь) пропускаем КАК ЕСТЬ в исходном типе — без раздувания в F16.
        eff = _eff_quant(name, qn) if base_fn else qn    # чувствит. слой → ступень выше
        fn, ggtype, blk = OUR.get(eff, (None, None, 32))
        can = (fn and len(shape) == 2 and nelem > QUANT_THRESHOLD
               and not llm_critical(name) and shape[1] % blk == 0)
        if can:
            data = xgguf.dec_source(raw, tt, nelem)      # распаковка источника
            if data is None:                             # неизвестный квант → как есть
                out.append((name, tt, dims, raw)); npass += 1
            else:
                try:
                    out.append((name, ggtype, dims, fn(data.reshape(shape)).tobytes())); nq += 1
                    if eff != qn: nup += 1
                except Exception: out.append((name, tt, dims, raw)); npass += 1
        else:
            out.append((name, tt, dims, raw)); npass += 1
        if (i+1) % 10 == 0: log(f"  обработано {i+1}/{len(tinfos)} тензоров...")
    f.close()
    # оценка «битности»: тип, которым представлено БОЛЬШИНСТВО больших тензоров
    _BITS_OF = {xgguf.T.F32:32, xgguf.T.F16:16, xgguf.T.BF16:16, xgguf.T.Q8_0:8.5,
                xgguf.T.Q6_K:6.6, 13:5.5, xgguf.T.Q5_0:5.5, 12:4.5, xgguf.T.Q4_0:4.5,
                xgguf.T.Q3_K:3.4, xgguf.T.Q2_K:2.6}
    _TGT_BITS = {"Q8_0":8.5,"Q6_K":6.6,"Q5_0":5.5,"Q4_0":4.5,"Q3_K":3.4,"Q2_K":2.6}
    src_bits = _BITS_OF.get(max(src_types, key=src_types.get), 16) if src_types else 16
    tgt_bits = _TGT_BITS.get(qn, 4.5)
    if src_bits < 9:  # источник уже лоссовый (не F16/F32/Q8) → двойная потеря
        log(f"⚠️ ВНИМАНИЕ: источник УЖЕ квантован (~{src_bits:.1f} бит). Реквант в {qn} "
            f"(~{tgt_bits:.1f} бит) = ДВОЙНАЯ потеря — качество LLM просядет.")
        log("   Идеал: брать F16/BF16 или Q8_0 версию модели. Особенно не жми в Q2/Q3.")
    dst = _out_path(src, qn)
    # синхронизируем general.file_type с новым квантом (иначе LM Studio прячет модель)
    ft = xgguf.FTYPE.get(qn)
    if ft is not None:
        raw_meta = xgguf.patch_kv_u32(raw_meta, "general.file_type", ft)
    log(f"пишу LLM GGUF ({len(out)} тензоров, метадата+токенайзер сохранены)...")
    xgguf.write_gguf_raw(dst, raw_meta, n_kv, out)
    log(f"ГОТОВО: {os.path.basename(dst)}  {fsize/1e9:.1f} -> {os.path.getsize(dst)/1e9:.1f} ГБ  (реквант {nq}{', '+str(nup)+' чувств.→'+_UPGRADE[qn] if nup else ''}, как-есть {npass})  за {_fmt_dur(time.time()-t0)}")
    return dst

def requantize_gguf_1bit(src, qn, log=print):
    """LLM Q1 (experimental): ffn(MLP)→1-бит пер-канальный бинар, attn→Q4_0, критика как есть.
    В GGUF НЕТ 1-бит типа → бинарные ffn пишутся F16: это ДЕМО КАЧЕСТВА, НЕ сжатие (файл не
    меньше). Метадата+токенайзер сохранены, грузится в llama.cpp/LM Studio, но деградирует."""
    t0 = time.time()
    log("!! LLM Q1 — ЭКСПЕРИМЕНТ: ffn -> 1-бит, attn -> Q4.")
    log("   В GGUF НЕТ 1-бит типа → бинарные ffn хранятся F16: файл НЕ меньше (демо качества,")
    log("   не сжатие). Реальное дно для деплоя LLM = Q2_K/Q3_K. Это для оценки, не продакшн.")
    f, ver, raw_meta, n_kv, tinfos, data_start, align = xgguf.read_gguf(src)
    fsize = os.path.getsize(src); offs = [t[3] for t in tinfos]
    out = []; nbin = nq4 = npass = 0
    for i, (name, dims, tt, off) in enumerate(tinfos):
        start = data_start + off
        end = data_start + offs[i+1] if i+1 < len(tinfos) else fsize
        f.seek(start); raw = f.read(end - start)
        shape = tuple(reversed(dims)); nelem = int(np.prod(shape)) if shape else 1
        nl = name.lower()
        big2d = len(shape) == 2 and nelem > QUANT_THRESHOLD and not llm_critical(name) and shape[1] % 32 == 0
        if big2d and "ffn" in nl:
            data = xgguf.dec_source(raw, tt, nelem)
            if data is None: out.append((name, tt, dims, raw)); npass += 1
            else:
                rec = _bin_per_channel(np.ascontiguousarray(data, np.float32).reshape(shape))
                out.append((name, xgguf.T.F16, dims, xgguf.enc_f16(rec))); nbin += 1
        elif big2d and "attn" in nl:
            data = xgguf.dec_source(raw, tt, nelem)
            if data is None: out.append((name, tt, dims, raw)); npass += 1
            else:
                out.append((name, xgguf.T.Q4_0, dims, xq.our_quantize_q4_0(np.ascontiguousarray(data, np.float32).reshape(-1)).tobytes())); nq4 += 1
        else:
            out.append((name, tt, dims, raw)); npass += 1
        if (i+1) % 10 == 0: log(f"  обработано {i+1}/{len(tinfos)} (ffn-бинар {nbin}, attn-Q4 {nq4})...")
    f.close()
    dst = os.path.splitext(_out_path(src, qn))[0] + ".gguf"
    log("пишу LLM GGUF (Q1 демо, метадата+токенайзер сохранены)...")
    xgguf.write_gguf_raw(dst, raw_meta, n_kv, out)
    log(f"ГОТОВО (Q1 демо): {os.path.basename(dst)}  {fsize/1e9:.1f} -> {os.path.getsize(dst)/1e9:.1f} ГБ  (ffn-бинар {nbin}, attn-Q4 {nq4}, как-есть {npass})  за {_fmt_dur(time.time()-t0)}")
    return dst

def process(src, qn, log=print):
    """.gguf → LLM requant (метадата+токенайзер сохранены); .safetensors → диффузия.
    Q1 → экспериментальный 1-бит демо (diffusion: sim.safetensors; LLM: sim.gguf)."""
    if qn.upper() == "Q1":
        if src.lower().endswith(".gguf"):
            return requantize_gguf_1bit(src, qn, log)
        return compress_1bit(src, qn, log)
    if src.lower().endswith(".gguf"):
        return requantize_gguf(src, qn, log)
    return compress_file(src, qn, log)

# байт/вес для оценки размера + порядок вывода
_BPW = {"Q8_0":34/32, "Q6_K":210/256, "Q5_0":22/32, "Q4_0":18/32, "Q3_K":110/256, "Q2_K":84/256}
_TEST_ORDER = ["Q8_0","Q6_K","Q5_0","Q4_0","Q3_K","Q2_K"]

def _verdict(psnr):
    # пороги откалиброваны по ЖИВОМУ A/B на FLUX: Q4_0(~45dB)=чисто, Q3_K(~39dB)=
    # зерно, Q2_K(~33dB)=муть. Вес-PSNR не линеен к фото — не занижаем планку.
    if psnr >= 52: return "идеал — не отличить"
    if psnr >= 44: return "чисто, без потерь на глаз"
    if psnr >= 40: return "лёгкая мягкость"
    if psnr >= 35: return "заметное зерно"
    if psnr >= 30: return "сильное зерно/муть"
    return "развал"

def quality_test(src, log=print, sample=40, rows_cap=768):
    """Оффлайн-оценка ПОТЕРЬ квантизации по всем битностям (прокси качества фото).
    Берём выборку больших 2D-весов, для каждой битности квантуем→разжимаем и
    считаем отклонение + PSNR. Картинку не генерим (exe без GPU) — но PSNR весов
    напрямую коррелирует с зерном/мылом на фото."""
    t0 = time.time()
    log("=== ТЕСТ КАЧЕСТВА: потери квантизации по битам (прокси качества фото) ===")
    if src.lower().endswith(".gguf"):
        log("тест считает по .safetensors-исходнику. Для .gguf дай оригинал."); return
    # собрать кандидатов из заголовка: большие 2D-веса, кратные 256 (годятся всем типам)
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    cands = []
    for k, m in hdr.items():
        if k == "__metadata__": continue
        dt = m.get("dtype"); shape = m.get("shape") or []
        if dt not in ("F32","F16","BF16","F8_E4M3","F8_E5M2"): continue
        if len(shape) != 2 or int(np.prod(shape)) <= QUANT_THRESHOLD: continue
        if shape[1] % 256 != 0: continue
        s, e = m["data_offsets"]
        cands.append((k, dt, tuple(shape), base+s, base+e, int(np.prod(shape))))
    if not cands:
        log("нет подходящих 2D-весов для теста."); return
    cands.sort(key=lambda c: -c[5])                 # самые большие = самые значимые
    picks = cands[:sample]
    log(f"выборка {len(picks)} тензоров (из {len(cands)}), срез до {rows_cap} строк...")
    mats = []
    with open(src, "rb") as f:
        for k, dt, shape, a, b, _ in picks:
            # читаем ТОЛЬКО нужные строки (rows_cap) — не весь тензор
            rows, cols = shape
            r = min(rows, rows_cap)
            elt = {"F32":4,"F16":2,"BF16":2,"F8_E4M3":1,"F8_E5M2":1}[dt]
            f.seek(a); raw = f.read(r * cols * elt)
            d = _decode(raw, dt, (r, cols))
            mats.append(np.ascontiguousarray(d, np.float32))
    log("бит     байт/вес   отклонение   PSNR      вердикт")
    log("-" * 60)
    for qn in _TEST_ORDER:
        fn, ggtype, blk = OUR[qn]
        sq_err = 0.0; sq_sig = 0.0; peak = 0.0
        for d in mats:
            try:
                q = fn(d)
                deq = xgguf.dec_source(q.tobytes() if hasattr(q,"tobytes") else bytes(q),
                                       ggtype, d.size).reshape(d.shape)
            except Exception:
                continue
            diff = deq - d
            sq_err += float(np.sum(diff*diff)); sq_sig += float(np.sum(d*d))
            peak = max(peak, float(np.max(np.abs(d))))
        if sq_sig <= 0: continue
        rel = (sq_err/sq_sig) ** 0.5                 # относит. RMS-ошибка
        rmse = (sq_err / sum(m.size for m in mats)) ** 0.5
        psnr = 20*np.log10(peak/rmse) if rmse > 0 else 99.0
        log(f"{qn:6}  {_BPW[qn]:.2f}       {rel*100:5.1f}%       {psnr:5.1f} dB   {_verdict(psnr)}")
    log("-" * 60)
    log(f"меньше отклонение / выше PSNR = чище фото. Готово за {_fmt_dur(time.time()-t0)}")

# ═══════════ РЕАЛ-ТЕСТ: настоящая генерация через локальный ComfyUI ═══════════
# exe без GPU, но шлёт FLUX-воркфлоу в живой ComfyUI (:8000) и тащит PNG'и —
# реальное сравнение кадров на разных квантах твоей модели.
def _comfy_url():
    return os.environ.get("XQUANT_COMFY_URL", "http://127.0.0.1:8000").rstrip("/")

def _http_json(url, timeout=8):
    import urllib.request
    return json.load(urllib.request.urlopen(url, timeout=timeout))

def _comfy_alive(url):
    try: _http_json(url + "/system_stats", timeout=4); return True
    except Exception: return False

def _obj_options(url, cls, field):
    try:
        d = _http_json(f"{url}/object_info/{cls}", timeout=6)
        return d[cls]["input"]["required"][field][0] or []
    except Exception: return []

def _pick(cands, subs):
    for s in subs:
        for c in cands:
            if s.lower() in c.lower(): return c
    return None

_REAL_PROMPT = ("Photograph, a weathered old fisherman with a deeply lined face, grey "
    "stubble, sharp eyes, yellow raincoat, harbor at golden hour, fishing boats behind, "
    "fine skin pores, sharp focus, documentary photo, 8K")

def _flux_workflow(unet, t5, clipl, vae, seed, prompt, px=768):
    return {
      "1":{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":unet}},
      "2":{"class_type":"DualCLIPLoader","inputs":{"clip_name1":t5,"clip_name2":clipl,"type":"flux"}},
      "3":{"class_type":"VAELoader","inputs":{"vae_name":vae}},
      "4":{"class_type":"CLIPTextEncode","inputs":{"clip":["2",0],"text":prompt}},
      "5":{"class_type":"FluxGuidance","inputs":{"conditioning":["4",0],"guidance":3.5}},
      "6":{"class_type":"CLIPTextEncode","inputs":{"clip":["2",0],"text":""}},
      "7":{"class_type":"EmptySD3LatentImage","inputs":{"width":px,"height":px,"batch_size":1}},
      "8":{"class_type":"KSampler","inputs":{"model":["1",0],"seed":seed,"steps":20,"cfg":1.0,
            "sampler_name":"euler","scheduler":"simple","positive":["5",0],"negative":["6",0],
            "latent_image":["7",0],"denoise":1.0}},
      "9":{"class_type":"VAEDecode","inputs":{"samples":["8",0],"vae":["3",0]}},
      "10":{"class_type":"SaveImage","inputs":{"images":["9",0],"filename_prefix":"xq_realtest"}}}

def real_gen_test(src, log=print, seed=42):
    """Найти сжатые кванты модели в ComfyUI и сгенерить один сид на каждом.
    Возвращает список (label, png_path). FLUX-only (воркфлоу FLUX-специфичен)."""
    import urllib.request, urllib.parse, tempfile
    url = _comfy_url()
    if not _comfy_alive(url):
        log(f"ComfyUI не отвечает на {url}. Запусти пул (:8000) или задай XQUANT_COMFY_URL."); return []
    base = os.path.splitext(os.path.basename(src))[0]        # flux1-dev
    unets = _obj_options(url, "UnetLoaderGGUF", "unet_name")
    def _is_quant_of(u):
        b = os.path.basename(u)
        if not (b.startswith(base+"-") and b.endswith(".gguf")): return False
        core = b[len(base)+1:-5]                              # между 'base-' и '.gguf'
        if "-" in core or "_" in core.split("Q",1)[0]: return False  # -kontext- и пр. варианты — не наши
        return bool(re.match(r"(OUR)?Q\d", core, re.I))       # Q4_0 / Q3_K / OURQ2K, но не 'kontext'
    mine = sorted([u for u in unets if _is_quant_of(u)])
    if not mine:
        log(f"в ComfyUI нет сжатых квантов '{base}-Q*.gguf'. Сожми модель (СЖАТЬ) в нужные биты."); return []
    t5   = _pick(_obj_options(url,"DualCLIPLoader","clip_name1"), ["t5xxl_fp8","t5xxl","t5-xxl"])
    clipl= _pick(_obj_options(url,"DualCLIPLoader","clip_name2"), ["clip_l.","clip-l.","/clip_l"])
    vae  = _pick(_obj_options(url,"VAELoader","vae_name"), ["ae.safetensors","flux","ae."])
    if not (t5 and clipl and vae):
        log(f"не нашёл FLUX CLIP/VAE в ComfyUI (t5={t5} clip_l={clipl} vae={vae})."); return []
    log(f"ComfyUI ok. Кванты: {len(mine)} → {[os.path.basename(m) for m in mine]}")
    log(f"CLIP={os.path.basename(t5)}+{os.path.basename(clipl)} VAE={os.path.basename(vae)} seed={seed}")
    # Папка результатов на ЭТОТ прогон: <модель>_realtest_<время>, внутри —
    # по фото на каждый квант (один промпт+сид) + prompt.txt. 2026-07-05.
    _rt_root = os.environ.get("XQUANT_OUT_DIR", "").strip() or os.path.dirname(os.path.abspath(src))
    rt_dir = os.path.join(_rt_root, f"{base}_realtest_{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        os.makedirs(rt_dir, exist_ok=True)
        with open(os.path.join(rt_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
            pf.write(f"model: {base}\nseed: {seed}\nprompt:\n{_REAL_PROMPT}\n")
        log(f"папка результатов: {rt_dir}")
    except Exception as e:
        log(f"не создал папку результатов ({e}) — сохраняю во временную"); rt_dir = tempfile.gettempdir()
    out = []
    for u in mine[:4]:                                       # максимум 4, чтоб не ждать вечность
        label = os.path.basename(u).replace(base+"-","").replace(".gguf","")
        log(f"генерю {label}...")
        t0 = time.time()
        try:
            wf = _flux_workflow(u, t5, clipl, vae, seed, _REAL_PROMPT)
            d = json.dumps({"prompt":wf}).encode()
            pid = json.load(urllib.request.urlopen(urllib.request.Request(
                url+"/prompt", data=d, headers={"Content-Type":"application/json"}), timeout=30))["prompt_id"]
            fn = sub = None; tend = time.time()+300
            while time.time() < tend:
                h = _http_json(f"{url}/history/{pid}", timeout=15)
                if pid in h:
                    for o in h[pid]["outputs"].values():
                        if "images" in o: fn=o["images"][0]["filename"]; sub=o["images"][0]["subfolder"]; break
                    if fn: break
                time.sleep(2)
            if not fn: log(f"  {label}: таймаут"); continue
            q = urllib.parse.urlencode({"filename":fn,"subfolder":sub or "","type":"output"})
            png = urllib.request.urlopen(url+"/view?"+q, timeout=30).read()
            p = os.path.join(rt_dir, f"{label}.png")          # фото кванта в папку прогона
            open(p,"wb").write(png)
            out.append((label, p))
            log(f"  {label}: готово за {_fmt_dur(time.time()-t0)}")
        except Exception as e:
            log(f"  {label}: ошибка {e}")
    return out

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
