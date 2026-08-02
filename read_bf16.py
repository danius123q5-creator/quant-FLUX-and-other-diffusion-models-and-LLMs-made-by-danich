"""Чтение bf16-слоёв FLUX БЕЗ torch и без safetensors.

Зачем: safetensors роняет процесс (segfault) на этом файле, а numpy не знает bf16.
Разбираем заголовок safetensors руками и читаем нужный кусок через mmap.
bf16 -> float32 = сдвиг на 16 бит влево (bf16 это просто старшая половина float32).
"""
import json, mmap, numpy as np

PATH = r"D:\Comfy\models\checkpoints\flux1-dev.safetensors"


def header(path=PATH):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n)), 8 + n


def load(key, rows=None, cols=None, path=PATH):
    """Возвращает float32-массив (кусок rows x cols) для bf16/f16/f32 тензора."""
    hdr, base = header(path)
    meta = hdr[key]
    shape = meta["shape"]
    dt = meta["dtype"]
    itemsize = {"BF16": 2, "F16": 2, "F32": 4}[dt]
    start, end = meta["data_offsets"]
    r = shape[0] if rows is None else min(rows, shape[0])
    ncol = shape[1] if len(shape) > 1 else 1
    c = ncol if cols is None else min(cols, ncol)
    need = r * ncol * itemsize                      # читаем целые строки
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm[base + start: base + start + need]
        mm.close()
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dt == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        a = np.frombuffer(raw, dtype=np.float32)
    a = a.reshape(r, ncol)[:, :c]
    return np.ascontiguousarray(a, np.float32), shape, (end - start)


def keys(path=PATH):
    hdr, _ = header(path)
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


if __name__ == "__main__":
    ks = keys()
    a, shape, nbytes = load("double_blocks.9.img_mod.lin.weight", rows=64, cols=512)
    print("tenzorov v fajle:", len(ks))
    print("adaLN sloj:", shape, "-> kusok", a.shape, "std=%.5f" % a.std())
