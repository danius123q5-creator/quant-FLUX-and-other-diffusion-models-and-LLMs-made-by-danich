# -*- coding: utf-8 -*-
"""СИМУЛЯТОР РАСКЛАДКИ — кадры любой битности БЕЗ нового формата и без загрузчика.

Зачем. Боевой .gguf упирается в дно лестницы Q2_K = 2.625 бита: ниже собрать нечем,
такой файл некому прочитать. Но чтобы ОТВЕТИТЬ НА ВОПРОС «а что с картинкой», хранить
сжато не обязательно — достаточно СЖАТЬ И ТУТ ЖЕ РАЗЖАТЬ. Веса в файле будут ровно те,
что были бы после сжатия, до последнего веса; ComfyUI грузит его штатно как обычный
safetensors. Файл жирный (экономии нет), зато битность — любая, вплоть до 1.

Приём не новый: так снимались июльские кадры (make_ternary_sim.py, 04-05.07.2026),
на которых и нашли, что MLP держит 1 бит, а attention рвётся.

Чем этот скрипт отличается от июльского: раскладка задаётся ПО ГРУППАМ СЛОЁВ —
attention и ffn/mlp можно давить РАЗНЫМ кодером. Ровно то, чего не умеет боевой
жматель (там раздаёт автомат, и от базы Q2_K вниз он двигать не может).

⚠️ Размер файла тут НИ О ЧЁМ НЕ ГОВОРИТ — он у всех вариантов одинаковый и равен
исходному. «Сколько бы это весило» печатается отдельно, расчётом по битам.

Запуск:
    python make_layout_sim.py <src.safetensors> --preset S1 [--out-dir DIR]
    python make_layout_sim.py <src.safetensors> --attn q4 --ffn bin

Пресеты (attn / ffn):
    S1  q4   / bin    мясо в пол, нерв цел      — главная ставка
    S2  bin  / q4     ЗЕРКАЛО: кто хрупкий      — против нашей же оси
    S3  bin  / bin    дно шкалы, заведомый труп — нужен как ноль отсчёта
    S4  tern / tern   1.6 бита равномерно
Кодеры: bin (1 бит), tern (1.58), q2/q3/q4/q5/q6 (наши боевые), fp16 (не трогать).
"""
import argparse, json, os, re, sys, time
import numpy as np

# Консоль Windows под cp1251 роняет print на эмодзи и части кириллицы (UnicodeEncodeError
# уже после того, как файл записан — то есть работа сделана, а скрипт «упал»). Заворачиваем
# вывод в UTF-8 с заменой неотображаемых символов.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import xquant as xq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "comfyui-node", "ComfyUI-XQuant"))
import xgguf as _xg

# кодер -> (функция сжатия, бит/вес для расчёта «сколько бы весило»)
_CODERS = {
    "bin":  (lambda a: xq.our_quantize_bin(a, group=32),     1.5),
    "tern": (lambda a: xq.our_quantize_ternary(a, group=32), 1.6),
    "q2":   (None, 2.625), "q3": (None, 3.4375), "q4": (None, 4.5),
    "q5":   (None, 5.5),   "q6": (None, 6.5625), "fp16": (None, 16.0),
}
# кодер -> (энкодер жмателя, номер GGML-типа для декванта)
_GGUF_NAME = {
    "q2": (xq.our_quantize_q2k,  _xg.T.Q2_K),
    "q3": (xq.our_quantize_q3k,  _xg.T.Q3_K),
    "q4": (xq.our_quantize_q4_0, _xg.T.Q4_0),
    "q5": (xq.our_quantize_q5_0, _xg.T.Q5_0),
    "q6": (xq.our_quantize_q6k,  _xg.T.Q6_K),
}

# Произвольная битность: iN2..iN8 (равномерная сетка) и nf2..nf8 (уровни по квантилям
# нормали). Нужны для ПЛОТНОЙ лестницы: «остров адекватности» — это гипотеза о том, что
# качество по битности НЕмонотонно, и проверить её можно только мелким шагом, а не
# тремя точками 1 / 1.6 / 2.6. Группа 16, поэтому реальная битность = bits + 16/16 на шкалу.
for _b in range(2, 9):
    _CODERS[f"iN{_b}"] = ((lambda b: (lambda a: xq.our_quantize_iN_uniform(a, bits=b, group=16)))(_b),
                          _b + 1.0)
# nf-кодер (уровни по квантилям нормали) СЮДА НЕ ДОБАВЛЕН НАМЕРЕННО: он возвращает
# (индексы, шкала, уровни), и наивное q*scale даёт ошибку в разы хуже равномерного
# (проверено: nf2 → 1.65 против iN2 → 0.10). Нужен разжиматель через levels[idx];
# пока его тут нет — лучше не давать сломанный инструмент, чем объяснять потом кадры.

# «Коробка» (QuIP-style incoherence rotation): W·R → бинар → Rᵀ. Поворот детерминирован
# по размерности, хранить его не надо. В июле именно она вернула лицо на FLUX, когда голая
# бинаризация attention давала мёртвую ткань. Берём готовую функцию из жмателя, а не пишем свою.
def _box_bin_coder(a):
    """Бинар в коробке → возвращает (уже разжатый массив, None, 0) под общий интерфейс."""
    import importlib.util as _il
    global _BOXMOD
    try:
        _BOXMOD
    except NameError:
        _spec = _il.spec_from_file_location(
            "_xq_standalone", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "xquant_standalone.py"))
        _BOXMOD = _il.module_from_spec(_spec)
        _spec.loader.exec_module(_BOXMOD)
    return ("BOX", _BOXMOD._box_bin(np.ascontiguousarray(a, np.float32)), 0)

_CODERS["box"] = (_box_bin_coder, 1.5)


def _asym_coder(bits, group=32):
    """АСИММЕТРИЧНАЯ сетка: хранит min и scale на группу (как Q2_K), а не только шкалу.

    Зачем. Кадры 02.08 показали: при 4 уровнях симметричная сетка (iN2) даёт труп,
    а Q2_K на тех же двух битах индекса — чистую картинку. Разница ровно одна — у Q2_K
    есть СМЕЩЕНИЕ. Этот кодер повторяет Q2_K по устройству, но с произвольной битностью,
    чтобы проверить версию «виновата симметрия, а не битность».
    Накладные: два fp16 на группу 32 = 1 бит/вес, столько же, сколько у симметричного iN.
    """
    def _fn(a):
        x = np.ascontiguousarray(a, np.float32).reshape(-1)
        pad = (group - x.size % group) % group
        if pad:
            x = np.concatenate([x, np.zeros(pad, np.float32)])
        g = x.reshape(-1, group)
        qmax = (1 << bits) - 1
        lo = g.min(1, keepdims=True)
        hi = g.max(1, keepdims=True)
        if _ASYM_SEARCH:
            # Границы по min/max задаёт ОДИН выброс на группу: он растягивает сетку,
            # соседняя группа без выброса получает другую — на картинке это выступает
            # регулярной «чешуёй» с шагом в размер группы (видно на кадре ASYM80).
            # Поджимаем края и берём лучший вариант по ошибке — та же идея, что вылечила
            # Q3_K взвешенным поиском шкалы (кадр Q3FIX: рябь исчезла).
            best = None
            for k in (1.0, 0.95, 0.90, 0.85, 0.80):
                mid = (hi + lo) * 0.5
                half = (hi - lo) * 0.5 * k
                lo_k, hi_k = mid - half, mid + half
                sc = ((hi_k - lo_k) / qmax).clip(1e-8)
                qq = np.clip(np.round((g - lo_k) / sc), 0, qmax)
                dd = qq * sc + lo_k
                err = ((dd - g) ** 2).sum(1, keepdims=True)
                if best is None:
                    best = (err, dd)
                else:
                    take = err < best[0]
                    best = (np.where(take, err, best[0]), np.where(take, dd, best[1]))
            deq = best[1].reshape(-1)[:a.size]
        else:
            scale = ((hi - lo) / qmax).clip(1e-8)
            q = np.clip(np.round((g - lo) / scale), 0, qmax)
            deq = (q * scale + lo).reshape(-1)[:a.size]
        return ("BOX", deq.reshape(a.shape), 0)     # готовый массив, как у коробки
    return _fn

# ⚠️ ПОДЖАТИЕ ГРАНИЦ ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ — замер 02.08 показал, что оно ВРЕДИТ:
# на крупных весах (|w|>2σ) ошибка выросла вдвое (aN2 0.089 → 0.166, aN3 0.040 → 0.087).
# Логика понятна задним числом: поджатие обрезает ровно те выбросы, по которым мы и судим
# о качестве. Для Q3_K взвешенный поиск помогал (там подбиралась ОДНА шкала на симметричной
# сетке), здесь — нет. Включить для опытов: XQUANT_ASYM_SEARCH=1.
_ASYM_SEARCH = os.environ.get("XQUANT_ASYM_SEARCH", "0").strip().lower() in ("1", "on", "true", "yes")

for _b in (2, 3, 4):
    _CODERS[f"aN{_b}"] = (_asym_coder(_b), _b + 1.0)
    # Вариант с группой 64: накладные вдвое дешевле (0.5 бита вместо 1) — нужен, чтобы
    # раздавать биты НЕРАВНОМЕРНО и оставаться в том же бюджете. Кадр Q2_K (2.79 бита,
    # неравномерно) чище нашего ASYM80 (3.00 равномерно) — значит решает раздача, не битность.
    _CODERS[f"aN{_b}g64"] = (_asym_coder(_b, group=64), _b + 0.5)

_PRESETS = {                       # (attn, ffn)
    "S1": ("q4",   "bin"),
    "S2": ("bin",  "q4"),
    "S3": ("bin",  "bin"),
    "S4": ("tern", "tern"),
}

_ATTN_RE = re.compile(r"attn|attention|\bqkv\b|to_[qkv]\b", re.IGNORECASE)
_FFN_RE  = re.compile(r"\bffn\b|\bmlp\b|feed_forward", re.IGNORECASE)
# adaLN / модуляция / нормы. xq.is_critical их НЕ ловит (проверено: img_mod.lin → False),
# а на FLUX это 27% модели, и по отчёту 04-05.07 бинаризация adaLN даёт «зелёное поле»
# (гибнут магнитуды, поворотом не чинится). В июльских прогонах они тоже исключались —
# снимать это исключение нельзя, иначе сравниваем разное.
# time_embedding добавлен 02.08: боевой Q2_K держит его в F16 (9.4 млн весов), а мы давили.
# Слой задаёт, на каком шаге шумоподавления мы находимся, — его порча бьёт по всей цепочке.
# Цена защиты 9 млн из 5 млрд, то есть даром.
_MOD_RE  = re.compile(r"norm|mod\.lin|_mod\b|modulation|time_embedding", re.IGNORECASE)


def _quantize_dequantize(arr, coder):
    """Сжать выбранным кодером и сразу разжать. Возвращает массив исходной формы."""
    fn, _ = _CODERS[coder]
    if fn is not None:                                   # наши суб-2-битные
        q, scale, _pad = fn(arr)
        if isinstance(q, str) and q == "BOX":            # коробка возвращает готовый массив
            return np.asarray(scale, dtype=np.float32).reshape(arr.shape)
        deq = (q.astype(np.float32) * scale).reshape(-1)[:arr.size]
        return deq.reshape(arr.shape)
    # боевые K-кванты — тот же кодер, что пишет .gguf, и штатный деквант из xgguf
    enc, ggml_t = _GGUF_NAME[coder]
    packed = enc(arr)
    deq = _xg.dec_source(np.asarray(packed, np.uint8).tobytes(), ggml_t, arr.size)
    if deq is None:
        raise RuntimeError(f"нет декванта под {coder}")
    return np.asarray(deq, dtype=np.float32).reshape(arr.shape)


def _save_streaming(path, src_open, keys, make_tensor):
    """Записать safetensors ПОТОКОВО, не держа весь словарь в памяти.

    Зачем: save_file() требует готовый dict со всеми тензорами — для FLUX это 22 ГБ
    поверх читаемого исходника, и процесс падает с segfault на машине с 32 ГБ.
    Формат safetensors простой: 8 байт длины заголовка + JSON-заголовок + данные подряд,
    поэтому за два прохода (сначала посчитать размеры, потом писать) можно обойтись
    памятью в один тензор.
    """
    _DT = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
           torch.int64: "I64", torch.int32: "I32", torch.int8: "I8", torch.uint8: "U8",
           torch.bool: "BOOL"}
    header, off = {}, 0
    metas = []
    for k in keys:                                   # проход 1: только формы и dtype
        shape, dt, nbytes = src_open(k)
        header[k] = {"dtype": _DT[dt], "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
        metas.append(k)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(blob) % 8) % 8                    # выравнивание на 8 байт
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        for k in metas:                              # проход 2: по одному тензору
            t = make_tensor(k)
            fh.write(t.contiguous().view(torch.uint8).numpy().tobytes())
            del t


def _group_of(key):
    if _ATTN_RE.search(key): return "attn"
    if _FFN_RE.search(key):  return "ffn"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--preset", choices=sorted(_PRESETS))
    ap.add_argument("--attn", choices=sorted(_CODERS))
    ap.add_argument("--ffn",  choices=sorted(_CODERS))
    ap.add_argument("--other", default="fp16", choices=sorted(_CODERS),
                    help="нормы/adaLN/эмбеды: по умолчанию НЕ трогаем")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--guard-coder", default="fp16", choices=sorted(_CODERS),
                    help="Чем кодировать защищённые крайние блоки. fp16 = не трогать вовсе "
                         "(дорого: 2+2 блока WAN = 13%% модели, средняя битность 3.0 → 4.7). "
                         "aN4 = 5 бит, те же блоки стоят всего +0.26 бита к средней.")
    ap.add_argument("--guard-edges", type=int, default=0, metavar="N",
                    help="Оставить N ПЕРВЫХ и N ПОСЛЕДНИХ блоков в fp16. Крайние блоки "
                         "тянут общий тон и сборку картинки из шума; один блок WAN = 3.3%% "
                         "модели, так что защита пары стоит ~4%% веса.")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="ДОЛЯ слоёв группы, которые давим (0..1). Даёт НЕПРЕРЫВНУЮ ось "
                         "для поиска «острова»: 0.1, 0.2, 0.3 … вместо редких ступеней бит. "
                         "Давятся ПОСЛЕДНИЕ по порядку слои — глубокие блоки.")
    a = ap.parse_args()

    if a.preset:
        attn, ffn = _PRESETS[a.preset]
    elif a.attn and a.ffn:
        attn, ffn = a.attn, a.ffn
    else:
        ap.error("нужен --preset ИЛИ пара --attn/--ffn")
    tag = a.tag or (a.preset or f"{attn}-{ffn}")

    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.src))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.src))[0]
    dst = os.path.join(out_dir, f"{base}-SIM-{tag}.safetensors")
    if os.path.exists(dst):
        print(f"[sim] УЖЕ ЕСТЬ, не перезаписываю: {dst}")
        return

    coder_of = {"attn": attn, "ffn": ffn, "other": a.other}
    print(f"[sim] {tag}: attn={attn} ffn={ffn} other={a.other}")

    f = safe_open(a.src, framework="pt")
    n_done = 0
    bits = {"attn": 0.0, "ffn": 0.0, "other": 0.0}      # веса × бит/вес
    cnt  = {"attn": 0, "ffn": 0, "other": 0}
    t0 = time.time()
    keys = list(f.keys())

    # Номера блоков для защиты краёв: берём из имён вида "blocks.N." / "double_blocks.N."
    _guard = set()
    if a.guard_edges > 0:
        nums = set()
        for k in list(f.keys()):
            mm = re.search(r"blocks\.(\d+)\.", k)
            if mm:
                nums.add(int(mm.group(1)))
        if nums:
            order = sorted(nums)
            _guard = set(order[:a.guard_edges]) | set(order[-a.guard_edges:])
            print(f"[sim] защищены блоки (fp16): {sorted(_guard)}")

    def _in_guard(k):
        mm = re.search(r"blocks\.(\d+)\.", k)
        return bool(mm and int(mm.group(1)) in _guard)

    def _eligible(k):
        """Годится ли тензор под сжатие вообще (без учёта доли)."""
        if _in_guard(k) and a.guard_coder == "fp16":
            return False, (), 0
        sl = f.get_slice(k)
        shape = tuple(sl.get_shape())
        n = 1
        for d in shape:
            n *= d
        ok = (k.endswith(".weight") and len(shape) == 2 and n >= 4096
              and not xq.is_critical(k) and not _MOD_RE.search(k))
        return ok, shape, n

    # Отбор по доле: берём ПОСЛЕДНИЕ frac слоёв каждой группы (глубокие блоки).
    # Порядок в safetensors соответствует порядку слоёв, поэтому «последние» = «глубже».
    _picked = set()
    if a.frac < 1.0:
        by_grp = {"attn": [], "ffn": [], "other": []}
        for k in list(f.keys()):
            ok, _sh, _n = _eligible(k)
            if ok:
                by_grp[_group_of(k)].append(k)
        for g, lst in by_grp.items():
            take = int(round(len(lst) * a.frac))
            _picked.update(lst[len(lst) - take:] if take else [])
    else:
        _picked = None                                   # None = берём все годные

    def _plan(k):
        """Решить судьбу тензора: (жать ли, каким кодером, группа)."""
        grp = _group_of(k)
        coder = a.guard_coder if _in_guard(k) else coder_of[grp]
        ok, shape, n = _eligible(k)
        if _picked is not None and k not in _picked:
            ok = False
        return (ok and coder != "fp16"), coder, grp, shape, n

    for k in keys:                                   # проход учёта — без чтения данных
        do, coder, grp, shape, n = _plan(k)
        cnt[grp] += n
        bits[grp] += n * (_CODERS[coder][1] if do else 16.0)

    def _meta(k):
        W = f.get_tensor(k)
        return tuple(W.shape), W.dtype, W.numel() * W.element_size()

    state = {"n": 0}

    def _make(k):
        do, coder, grp, shape, n = _plan(k)
        W = f.get_tensor(k)
        if not do:
            return W
        arr = W.float().numpy()
        state["n"] += 1
        if state["n"] % 50 == 0:
            print(f"[sim]   {state['n']} тензоров, {time.time()-t0:.0f}с", flush=True)
        return torch.from_numpy(_quantize_dequantize(arr, coder)).to(W.dtype)

    # Пишем ПОТОКОВО: пик памяти — один тензор, а не вся модель. Для FLUX (22 ГБ)
    # обычный save_file() падал с segfault на машине с 32 ГБ.
    _save_streaming(dst, _meta, keys, _make)
    n_done = state["n"]
    total_w = sum(cnt.values())
    eq_gb = sum(bits.values()) / 8 / 1e9
    print(f"[sim] ГОТОВО: {dst}")
    print(f"[sim] обработано тензоров: {n_done}, весов: {total_w/1e6:.0f} млн")
    for g in ("attn", "ffn", "other"):
        if cnt[g]:
            print(f"[sim]   {g:5s} {cnt[g]/1e6:7.0f} млн  кодер {coder_of[g]:4s}"
                  f"  {bits[g]/max(cnt[g],1):5.2f} бит/вес")
    print(f"[sim] ⚠️ размер ФАЙЛА не показателен (веса разжаты). "
          f"ЭКВИВАЛЕНТНЫЙ вес: {eq_gb:.2f} ГБ по битам, "
          f"~{eq_gb*1.19:.2f} ГБ с поправкой на метаданные (x1.19).")


if __name__ == "__main__":
    main()
