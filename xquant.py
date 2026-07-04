# -*- coding: utf-8 -*-
"""НАШЕ ЯДРО КВАНТА — своя реализация квантования с нуля (не вызов gguf.quants).
Пишем ровно формат GGML Q4_0, побайтово, чтобы существующая нода ComfyUI-GGUF
разжала наш вывод. Плюс свой БОЛЕЕ СИЛЬНЫЙ формат int3-group32 (наш, с декодером
для замера ошибки). Верификация: наш Q4_0 == эталон gguf побайтово.

Q4_0 блок = 32 веса: fp16 d + 16 байт (по 2 ниббла). Деквант: x = d*(q-8).
"""
import os
import numpy as np
import re as _re

# ═══════════ УНИВЕРСАЛЬНАЯ ЗАЩИТА КРИТИЧЕСКИХ СЛОЁВ (все дифы) ═══════════
# Входные эмбеддинги, выходная проекция в VAE, нормы, модуляция — НЕ квантуем
# ни в какой архитектуре, иначе рвётся связь с декодером = цветной шум.
# Покрывает FLUX / SDXL / SD1.5 / SD3 / Qwen-Image / Wan / Hunyuan / LTXV / PixArt.
_CRITICAL_RE = _re.compile(
    # ТОЛЬКО вход/выход/эмбеды/нормы. Модуляцию (_mod/mod.lin) НЕ трогаем —
    # она нормально квантуется (старый Q2 её жал, PSNR держался). Защита её =
    # раздутие на ~4ГБ без пользы.
    r"scale_shift|final_layer|proj_out|conv_out|\bhead\.|"       # ВЫХОД в латент/VAE
    r"conv_in|img_in|txt_in|x_embedder|context_embedder|"        # ВХОД латента/текста
    r"patch_embed|pos_embed|pos_embedder|caption_projection|"    # патч/позиц/капшн эмбеды
    r"time_embed|t_embedder|timestep_embedder|time_text_embed|"  # время
    r"y_embedder|label_emb|vector_in|guidance_in|add_embed|"     # доп. кондишн
    r"cap_embedder|context_refiner|norm_out",                    # текст-эмбеды/выход-норма
    _re.IGNORECASE,
)

def is_critical(key: str) -> bool:
    """True = слой критический (вход/выход/эмбед/норма) → держать в bf16, НЕ квантовать."""
    return bool(_CRITICAL_RE.search(key))


# ─────────────────────── НАШ Q4_0 (GGML-совместимый) ───────────────────────
QK4_0 = 32

def our_quantize_q4_0(x: np.ndarray) -> np.ndarray:
    """Наш энкодер Q4_0. Вход: float32 [.., 32-кратно]. Выход: uint8 блоки
    (18 байт/32 веса: 2 fp16 d + 16 qs). Точная калька ggml quantize_row_q4_0."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % QK4_0 == 0, "длина не кратна 32"
    g = x.reshape(-1, QK4_0)                                  # [nb,32]
    # signed max по |v| (значение с макс. модулем — знак важен!)
    absg = np.abs(g)
    idx = absg.argmax(axis=1)
    vmax = g[np.arange(g.shape[0]), idx]                      # signed max
    d = (vmax / -8.0).astype(np.float32)
    d16 = d.astype(np.float16)                                # d хранится в fp16
    d_eff = d16.astype(np.float32)                            # квант считаем от fp16-d
    id_ = np.where(d_eff != 0, 1.0 / np.where(d_eff==0,1,d_eff), 0.0).astype(np.float32)
    # q = min(15, (int8_t)(x*id + 8.5))  — усечение к нулю (для +чисел = floor)
    xi = g * id_[:, None]                                     # [nb,32]
    q = np.minimum(15, (xi + 8.5).astype(np.int8)).astype(np.uint8)  # 0..15
    lo = q[:, 0:16]                                           # первые 16
    hi = q[:, 16:32]                                          # вторые 16
    qs = (lo | (hi << 4)).astype(np.uint8)                    # [nb,16]
    # склейка блока: 2 байта d (fp16 LE) + 16 байт qs
    d_bytes = d16.view(np.uint8).reshape(-1, 2)               # [nb,2]
    blocks = np.concatenate([d_bytes, qs], axis=1)            # [nb,18]
    return blocks.reshape(-1)

def our_quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """Наш энкодер Q8_0 (8-бит, почти без потерь). Блок 34 байта/32: fp16 d + 32 int8.
    Деквант: x = d*q. Побайтово = эталон gguf."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % 32 == 0
    g = x.reshape(-1, 32)
    amax = np.abs(g).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    d16 = d.astype(np.float16)
    id_ = np.where(d != 0, 1.0/np.where(d==0,1,d), 0.0).astype(np.float32)  # ПОЛНЫЙ d
    q = np.round(g * id_[:, None]).clip(-128, 127).astype(np.int8)
    d_bytes = d16.view(np.uint8).reshape(-1, 2)
    blocks = np.concatenate([d_bytes, q.view(np.uint8)], axis=1)            # [nb,34]
    return blocks.reshape(-1)

QK5_0 = 32
def our_quantize_q5_0(x: np.ndarray) -> np.ndarray:
    """Наш энкодер Q5_0 (5-бит, GGML). Блок 22 байта/32: fp16 d + 4 qh + 16 qs.
    Деквант: x = d*(q-16), q 5-бит (низ 4 в qs, 5-й бит в qh-маске)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % QK5_0 == 0
    g = x.reshape(-1, QK5_0)
    idx = np.abs(g).argmax(axis=1)
    vmax = g[np.arange(g.shape[0]), idx]
    d = (vmax / -16.0).astype(np.float32)
    d16 = d.astype(np.float16)
    id_ = np.where(d != 0, 1.0/np.where(d==0,1,d), 0.0).astype(np.float32)  # ПОЛНЫЙ d (не fp16)
    q = np.clip((g*id_[:, None] + 16.5).astype(np.int32), 0, 31).astype(np.uint32)
    lo = q[:, 0:16]; hi = q[:, 16:32]
    qs = ((lo & 0x0F) | ((hi & 0x0F) << 4)).astype(np.uint8)          # [nb,16]
    qh = np.zeros(g.shape[0], np.uint32)
    for j in range(16):
        qh |= ((lo[:, j] >> 4) & 1) << j
        qh |= ((hi[:, j] >> 4) & 1) << (j + 16)
    d_bytes = d16.view(np.uint8).reshape(-1, 2)
    qh_bytes = qh.view(np.uint8).reshape(-1, 4)                        # LE
    blocks = np.concatenate([d_bytes, qh_bytes, qs], axis=1)          # [nb,22]
    return blocks.reshape(-1)

def our_quantize_q6k(x: np.ndarray) -> np.ndarray:
    """Наш энкодер GGML Q6_K (6-бит K-квант, 210 байт/256). Обратный к
    ComfyUI dequantize_blocks_Q6_K. Проверяется round-trip через их декодер."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % QK_K == 0
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    # индекс суб-блока (0..15) для каждой из 256 позиций (раскладка ggml Q6_K)
    p = np.arange(QK_K)
    n_ = p // 128; o = p % 128; m = o // 32; l = o % 32
    subidx = (l // 16) + 2*m + 8*n_                    # [256]
    order = np.argsort(subidx, kind="stable")          # позиции сгруппированы по суб-блоку
    # scale на суб-блок = absmax/32 (ВЕКТОРНО по всем nb)
    grouped = xb[:, order].reshape(nb, 16, 16)         # [nb,16 суб,16]
    lsc = np.abs(grouped).max(axis=2) / 32.0           # [nb,16]
    d = np.maximum(lsc.max(axis=1, keepdims=True), 1e-8) / 127.0   # [nb,1]
    d16 = d.reshape(nb).astype(np.float16); d = d16.astype(np.float32).reshape(nb, 1)
    sc = np.clip(np.round(lsc / d), -128, 127).astype(np.int8)     # [nb,16]
    eff = (d * sc.astype(np.float32))[:, subidx]        # [nb,256]
    eff[eff == 0] = 1e-8
    q = np.clip(np.round(xb / eff) + 32, 0, 63).astype(np.int32)   # [nb,256]
    # упаковка ql[128]+qh[64] — 2 группы, ВЕКТОРНО (без цикла по l)
    ql = np.zeros((nb, 128), np.uint8); qh = np.zeros((nb, 64), np.uint8)
    for grp in range(2):
        n = grp*128; qlo = grp*64; qho = grp*32
        q1 = q[:, n:n+32]; q2 = q[:, n+32:n+64]; q3 = q[:, n+64:n+96]; q4 = q[:, n+96:n+128]
        ql[:, qlo:qlo+32]      = ((q1 & 0xF) | ((q3 & 0xF) << 4)).astype(np.uint8)
        ql[:, qlo+32:qlo+64]   = ((q2 & 0xF) | ((q4 & 0xF) << 4)).astype(np.uint8)
        qh[:, qho:qho+32] = (((q1>>4)&3) | (((q2>>4)&3)<<2) | (((q3>>4)&3)<<4) | (((q4>>4)&3)<<6)).astype(np.uint8)
    d16b = d16.view(np.uint8).reshape(nb, 2)
    out = np.concatenate([ql, qh, sc.view(np.uint8), d16b], axis=1)   # [nb,210]
    return out.reshape(-1)

def our_dequantize_q4_0(blocks: np.ndarray, n: int) -> np.ndarray:
    """Обратно (для проверки/замера ошибки)."""
    b = blocks.reshape(-1, 18)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = b[:, 2:18]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    q = np.concatenate([lo, hi], axis=1)                     # [nb,32]
    x = d * (q - 8.0)
    return x.reshape(-1)[:n]


# ─────────────── НАШ СИЛЬНЕЕ: int3 group-32 (наш формат) ───────────────
def our_quantize_i3(x: np.ndarray, group: int = 32):
    """Наш 3-бит симметричный по-групповой (−4..3). Не GGML — свой формат,
    ~3.5 бит/вес (3 + fp16-масштаб/32). Возвращает (q_uint8, scale_fp16)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    pad = (group - x.size % group) % group
    if pad: x = np.concatenate([x, np.zeros(pad, np.float32)])
    g = x.reshape(-1, group)
    scale = (np.abs(g).max(axis=1, keepdims=True) / 3.0).clip(min=1e-8).astype(np.float16)
    q = np.clip(np.round(g / scale.astype(np.float32)), -4, 3).astype(np.int8)
    return q, scale

def our_dequantize_i3(q, scale, n, group=32):
    return (q.astype(np.float32) * scale.astype(np.float32)).reshape(-1)[:n]


def relerr(a, b):
    a = a.reshape(-1).astype(np.float32); b = b.reshape(-1).astype(np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))


# ═══════════ 1-БИТ (бинарный) и 1.58-БИТ (тернарный, BitNet) ═══════════
def our_quantize_bin(x, group=32):
    """1-бит: знак × per-group absmean-масштаб (как BitNet-b1). ~1.5 бит с fp16-масштабом."""
    g, pad = _groups(x, group)
    scale = np.abs(g).mean(1, keepdims=True).clip(1e-8).astype(np.float32)  # absmean
    q = np.sign(g).astype(np.int8); q[q == 0] = 1
    return q, scale, pad

def our_dequantize_bin(q, scale, n):
    return (q.astype(np.float32) * scale).reshape(-1)[:n]

def our_quantize_ternary(x, group=32):
    """1.58-бит: −1/0/+1 порогом 0.7*absmean (BitNet b1.58). Меньше ошибка чем бинар."""
    g, pad = _groups(x, group)
    scale = np.abs(g).mean(1, keepdims=True).clip(1e-8).astype(np.float32)
    thr = 0.7 * scale
    q = np.zeros_like(g, dtype=np.int8)
    q[g > thr] = 1; q[g < -thr] = -1
    return q, scale, pad

def pack_tern5(codes_pm1):
    """Упаковать троичные коды (−1/0/+1) по 5 в байт (base-3, 3^5=243<256) = 1.6 бит/вес."""
    c = (np.asarray(codes_pm1, dtype=np.int16).reshape(-1) + 1).astype(np.uint16)  # −1/0/+1 → 0/1/2
    pad = (5 - c.size % 5) % 5
    if pad: c = np.concatenate([c, np.zeros(pad, np.uint16)])
    c = c.reshape(-1, 5)
    b = (c[:,0] + 3*c[:,1] + 9*c[:,2] + 27*c[:,3] + 81*c[:,4]).astype(np.uint8)
    return b, pad

def unpack_tern5(bytes_, n):
    """Обратно: байты base-3 → троичные коды −1/0/+1, первые n штук."""
    b = np.asarray(bytes_, dtype=np.uint16)
    out = np.empty((b.size, 5), dtype=np.int8)
    for i in range(5):
        out[:, i] = (b % 3).astype(np.int8); b //= 3
    return (out.reshape(-1)[:n] - 1)  # 0/1/2 → −1/0/+1

def our_dequantize_ternary(q, scale, n):
    return (q.astype(np.float32) * scale).reshape(-1)[:n]


# ═══════════════ НОВОЕ ЯДРО: суб-3-бит, неравномерное ═══════════════
def _groups(x, group):
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    pad = (group - x.size % group) % group
    if pad: x = np.concatenate([x, np.zeros(pad, np.float32)])
    return x.reshape(-1, group), pad

def our_quantize_iN_uniform(x, bits=2, group=16):
    """Базовый равномерный int-N (для сравнения). Симметричный."""
    g, pad = _groups(x, group)
    qmax = 2**(bits-1) - 1
    scale = (np.abs(g).max(1, keepdims=True) / max(qmax,1)).clip(1e-8)
    q = np.clip(np.round(g/scale), -qmax-1, qmax)
    return q, scale, pad

def our_dequantize_iN_uniform(q, scale, n):
    return (q*scale).reshape(-1)[:n]

# --- НФ-подобные уровни: квантили нормального распределения (веса ~гауссовы) ---
def _nf_levels(bits):
    """2^bits уровней по обратной CDF нормали, нормированы в [-1,1]. Как NF4, но
    любой bit. Веса сети ближе к N(0,σ) → такие уровни точнее равномерных."""
    K = 2**bits
    # квантильные точки (середины бинов), обратная CDF через приближение
    from math import erf, sqrt
    # простое приближение обратной erf через численный поиск на сетке
    xs = np.linspace(-4, 4, 20001)
    cdf = 0.5*(1+np.vectorize(lambda t: erf(t/np.sqrt(2)))(xs))
    ps = (np.arange(K)+0.5)/K
    lv = np.interp(ps, cdf, xs)
    lv = lv/np.abs(lv).max()          # в [-1,1]
    return lv.astype(np.float32)

def our_quantize_nf(x, bits=2, group=16):
    """НАШ неравномерный квант: на группу свой absmax, значения снапаются к
    ближайшему из 2^bits НФ-уровней. Возвращает (idx uint8, scale, levels, pad)."""
    g, pad = _groups(x, group)
    levels = _nf_levels(bits)                       # [K] в [-1,1]
    scale = np.abs(g).max(1, keepdims=True).clip(1e-8)   # [ng,1]
    gn = g/scale                                    # в [-1,1]
    # ближайший уровень
    idx = np.abs(gn[...,None] - levels[None,None,:]).argmin(-1).astype(np.uint8)
    return idx, scale, levels, pad

def our_dequantize_nf(idx, scale, levels, n):
    return (levels[idx]*scale).reshape(-1)[:n]

# --- Кодбук (k-means, 2^bits центроид на группу-строк): максимум под данные ---
def our_quantize_codebook(x, bits=2, group=16, iters=8):
    """НАШ кодбук: 2^bits центроид, подобранных ПОД веса (k-means-lite на группу).
    Точнее NF там где распределение не гауссово. idx uint8 + codebook на группу."""
    g, pad = _groups(x, group)                      # [ng, group]
    K = 2**bits
    ng = g.shape[0]
    # инициализация центроид по квантилям каждой группы
    qs = np.linspace(0, 1, K)
    cb = np.quantile(g, qs, axis=1).T               # [ng, K]
    for _ in range(iters):
        d = np.abs(g[:,:,None] - cb[:,None,:])      # [ng,group,K]
        idx = d.argmin(-1)                          # [ng,group]
        for k in range(K):                          # обновляем центроиды
            mask = (idx==k)
            cnt = mask.sum(1)
            s = (g*mask).sum(1)
            cb[:,k] = np.where(cnt>0, s/np.maximum(cnt,1), cb[:,k])
    d = np.abs(g[:,:,None] - cb[:,None,:]); idx = d.argmin(-1).astype(np.uint8)
    return idx, cb, pad

def our_dequantize_codebook(idx, cb, n):
    ng, group = idx.shape
    out = np.take_along_axis(cb, idx.astype(np.int64), axis=1)
    return out.reshape(-1)[:n]


# ═══════════ НАШЕ K-ЯДРО: реальный загружаемый Q3_K (GGML) ═══════════
# Обратный энкодер к ComfyUI-GGUF dequantize_blocks_Q3_K. Блок 110 байт/256:
# hmask[32] + qs[64] + scales[12] + d(fp16). Проверяется прогоном через их декодер.
QK_K = 256

def _make_qx_sym(x, w, nmin=-4, nmax=3, niter=8):
    """Взвешенный поиск СИММЕТРИЧНОГО шага dl (q∈[nmin,nmax]) — EM-рефайн под минимум
    Σ w·(x − dl·q)². Замена наивного absmax/nmax для Q3_K: меньше зерно, тот же формат."""
    amax = np.abs(x).max(-1)                         # (N,)
    ok = amax > 1e-12
    dl = np.where(ok, amax / nmax, 1e-8)
    best_dl = dl.copy()
    q = np.clip(np.round(x / dl[:, None]), nmin, nmax)
    best_err = np.sum(w * (x - dl[:, None] * q) ** 2, axis=-1)
    for _ in range(niter):
        q = np.clip(np.round(x / dl[:, None]), nmin, nmax)
        den = np.sum(w * q * q, axis=-1); num = np.sum(w * x * q, axis=-1)
        dl = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), best_dl)
        dl = np.where(dl > 1e-12, dl, best_dl)
        q2 = np.clip(np.round(x / dl[:, None]), nmin, nmax)
        e = np.sum(w * (x - dl[:, None] * q2) ** 2, axis=-1)
        upd = e < best_err
        best_err = np.where(upd, e, best_err); best_dl = np.where(upd, dl, best_dl)
    return np.where(ok, np.maximum(best_dl, 1e-8), 1e-8)

def our_quantize_q3k(x: np.ndarray) -> np.ndarray:
    """Наш энкодер GGML Q3_K. Вход float32 (кратно 256). Выход uint8 блоки (110б).
    Шаг сабблоков — взвешенным поиском (importance=x²), не absmax. XQUANT_NAIVE_Q3=1 → старое."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % QK_K == 0, "длина не кратна 256"
    nb = x.size // QK_K
    sb = x.reshape(nb, 16, 16)                       # [nb, 16 суб-блоков, 16]
    # Наивный absmax/4 для Q3_K оказался ЛУЧШЕ взвешенного EM-поиска (симметричный
    # квант + 5-битная переквантизация dl → взвешивание только добавляло шум,
    # замер: 18.75% naive vs 19.90% weighted). Оставляем наивный. Взвешенный —
    # только опытный, за XQUANT_WEIGHTED_Q3=1.
    if os.environ.get("XQUANT_WEIGHTED_Q3", "0").strip() in ("1","on","true","yes"):
        flat = sb.reshape(nb * 16, 16)
        dl = _make_qx_sym(flat, flat * flat + 1e-8, nmin=-4, nmax=3).reshape(nb, 16)
    else:
        dl = np.abs(sb).max(axis=2) / 4.0            # дефолт: наивный (лучше)
    d = dl.max(axis=1, keepdims=True) / 31.0         # [nb,1] супер-масштаб
    d = np.where(d == 0, 1e-8, d)
    scale = np.clip(np.round(dl / d), 0, 31).astype(np.int32)   # [nb,16] 0..31
    dl_eff = (d * scale)                             # фактический dl [nb,16]
    dl_eff = np.where(dl_eff == 0, 1e-8, dl_eff)
    # квант q ∈ -4..3
    q = np.clip(np.round(sb / dl_eff[:, :, None]), -4, 3).astype(np.int32)  # [nb,16,16]
    qh = (q < 0).astype(np.uint8)                    # старший: 1 если отрицательное
    ql = (q + 4 * qh).astype(np.uint8)               # 0..3
    hbit = (1 - qh).astype(np.uint8)                 # в hmask лежит инвертированный

    # --- раскладка по байтам (инверсия декодера) ---
    qs = np.zeros((nb, 64), np.uint8)
    hmask = np.zeros((nb, 32), np.uint8)
    # для каждого (s, w): flat=s*16+w; ql→qs[g*32+b] shift 2*sft; hbit→hmask[b2] bit hsh
    for s in range(16):
        for w in range(16):
            flat = s * 16 + w
            g = flat // 128; rem = flat % 128; sft = rem // 32; b = rem % 32
            qs[:, g * 32 + b] |= (ql[:, s, w] & 3) << (2 * sft)
            hsh = flat // 32; b2 = flat % 32
            hmask[:, b2] |= (hbit[:, s, w] & 1) << hsh
    # --- scales: 16 значений 6-бит signed, stored = scale (0..31) → v = scale (т.к.
    # декодер делает (low4|high2<<4) - 32; нам нужно decoded=scale → stored6=scale+32)
    v6 = (scale + 32).astype(np.uint8)               # [nb,16] 32..63
    scales = np.zeros((nb, 12), np.uint8)
    low4 = v6 & 0x0F; high2 = (v6 >> 4) & 0x03
    for j in range(16):
        # low4: j<8 → byte j низкий ниббл; j>=8 → byte j-8 высокий ниббл
        if j < 8: scales[:, j] |= low4[:, j]
        else:     scales[:, j - 8] |= (low4[:, j] << 4)
        # high2: byte 8+(j%4), shift 2*(j//4)
        scales[:, 8 + (j % 4)] |= (high2[:, j] << (2 * (j // 4)))
    d16 = d.reshape(nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
    block = np.concatenate([hmask, qs, scales, d16], axis=1)   # [nb,110]
    return block.reshape(-1)


def _qs_pack(vals2bit):
    """Упаковать [nb,16,16] 2-битных (0..3) в qs[nb,64] по раскладке K-квантов."""
    nb = vals2bit.shape[0]
    qs = np.zeros((nb, 64), np.uint8)
    for s in range(16):
        for w in range(16):
            flat = s * 16 + w
            g = flat // 128; rem = flat % 128; sft = rem // 32; b = rem % 32
            qs[:, g * 32 + b] |= (vals2bit[:, s, w] & 3) << (2 * sft)
    return qs

def _make_qkx2(x, w, nmax, nstep=20, rmin=-1.0, rdelta=0.1):
    """Взвешенный поиск (scale, add_min) на группу — порт llama.cpp make_qkx2_quants.
    Минимизирует Σ w·(scale·q + add_min − x)², q∈[0,nmax]. Векторно: x,w=(N,g).
    Возвращает scale(N,), add_min(N,). Это и есть замена наивного min/max — оно
    даёт основной прирост качества на 2/3 битах БЕЗ обучения (importance = w)."""
    g = x.shape[-1]
    xmin = np.minimum(x.min(-1), 0.0)                 # (N,)
    xmax = np.maximum(x.max(-1), 0.0)
    rng = xmax - xmin
    ok = rng > 1e-12
    rng_s = np.where(ok, rng, 1.0)
    iscale = nmax / rng_s
    L0 = np.clip(np.round(iscale[:, None] * (x - xmin[:, None])), 0, nmax)
    def _werr(sc, mn, L):
        pred = sc[:, None] * L + mn[:, None]
        return np.sum(w * (pred - x) ** 2, axis=-1)
    best_scale = (1.0 / iscale); best_min = xmin.copy()
    best_err = _werr(best_scale, best_min, L0)
    for istep in range(nstep + 1):
        isc = (rmin + rdelta * istep + nmax) / rng_s
        l = np.clip(np.round(isc[:, None] * (x - xmin[:, None])), 0, nmax)
        sw   = np.sum(w, axis=-1)
        swl  = np.sum(w * l, axis=-1)
        swl2 = np.sum(w * l * l, axis=-1)
        swx  = np.sum(w * x, axis=-1)
        swlx = np.sum(w * l * x, axis=-1)
        D = sw * swl2 - swl * swl
        Dok = np.abs(D) > 1e-12
        Ds = np.where(Dok, D, 1.0)
        this_scale = (sw * swlx - swx * swl) / Ds
        this_min   = (swl2 * swx - swl * swlx) / Ds
        err = _werr(this_scale, this_min, l)
        upd = Dok & (err < best_err)
        best_err = np.where(upd, err, best_err)
        best_scale = np.where(upd, this_scale, best_scale)
        best_min = np.where(upd, this_min, best_min)
    best_scale = np.where(ok, best_scale, 0.0)
    best_min = np.where(ok, best_min, 0.0)
    return best_scale, best_min

def our_quantize_q2k(x: np.ndarray) -> np.ndarray:
    """Наш энкодер GGML Q2_K (загружаемый 2-бит). Блок 84 байта/256:
    scales[16] + qs[64] + d(fp16) + dmin(fp16). w = d*sc*q - dmin*m.
    Шкала сабблоков — взвешенным поиском make_qkx2 (importance=x²), не min/max:
    заметно меньше ошибка на 2 битах при том же формате. XQUANT_NAIVE_Q2=1 → старое."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    assert x.size % QK_K == 0
    nb = x.size // QK_K
    sb = x.reshape(nb, 16, 16)
    if os.environ.get("XQUANT_NAIVE_Q2", "0").strip() in ("1","on","true","yes"):
        scale = ((sb.max(2) - np.minimum(sb.min(2), 0.0)) / 3.0)      # наивный фолбэк
        add_min = np.minimum(sb.min(2), 0.0)
    else:
        flat = sb.reshape(nb * 16, 16)
        wimp = flat * flat + 1e-8                                     # importance = x²
        sc, am = _make_qkx2(flat, wimp, nmax=3)
        scale = sc.reshape(nb, 16); add_min = am.reshape(nb, 16)
    scale = np.where(scale <= 0, 1e-8, scale)
    the_min = np.maximum(-add_min, 0.0)              # dmin*m = -add_min ≥ 0
    d = scale.max(axis=1, keepdims=True) / 15.0
    dmin = the_min.max(axis=1, keepdims=True) / 15.0
    d = np.where(d <= 0, 1e-8, d); dmin = np.where(dmin <= 0, 1e-8, dmin)
    sc4 = np.clip(np.round(scale / d), 0, 15).astype(np.int32)       # [nb,16]
    m4 = np.clip(np.round(the_min / dmin), 0, 15).astype(np.int32)
    dl_eff = (d * sc4); ml_eff = (dmin * m4)
    dl_eff = np.where(dl_eff == 0, 1e-8, dl_eff)
    q = np.clip(np.round((sb + ml_eff[:, :, None]) / dl_eff[:, :, None]), 0, 3).astype(np.uint8)
    qs = _qs_pack(q)
    scales = (sc4 | (m4 << 4)).astype(np.uint8)      # [nb,16]
    d16 = d.reshape(nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
    dm16 = dmin.reshape(nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
    block = np.concatenate([scales, qs, d16, dm16], axis=1)     # [nb,84]
    return block.reshape(-1)
