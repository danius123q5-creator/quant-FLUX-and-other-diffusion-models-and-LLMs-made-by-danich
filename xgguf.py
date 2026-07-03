# -*- coding: utf-8 -*-
"""НАШ GGUF-писатель — с нуля, без внешней gguf-либы. Только numpy + struct.
Пишет валидный GGUF v3, который читает ComfyUI-GGUF. Формат: magic+version+
counts → метадата KV → тензор-инфо → выравнивание → данные тензоров (align 32).

GGML-типы (что нужно): F32=0 F16=1 Q4_0=2 Q2_K=10 Q3_K=11 BF16=30.
Логика dims: ggml-порядок = numpy-shape в ОБРАТНОМ порядке (ne[0]=последняя).
Для квантованных dims = ЛОГИЧЕСКИЕ (элементы), данные = упакованные байты.
"""
import struct, numpy as np

# GGML quant types
class T:
    F32=0; F16=1; Q4_0=2; Q5_0=6; Q8_0=8; Q2_K=10; Q3_K=11; Q6_K=14; BF16=30
# метадата value-types
_U32=4; _STR=8; _U64=10
ALIGN=32
_MAGIC=0x46554747  # "GGUF" LE

def _gstr(s):
    b=s.encode("utf-8"); return struct.pack("<Q",len(b))+b

def _kv_str(k,v): return _gstr(k)+struct.pack("<I",_STR)+_gstr(v)
def _kv_u32(k,v): return _gstr(k)+struct.pack("<I",_U32)+struct.pack("<I",v)

def _pad(nbytes, align=ALIGN):
    r = nbytes % align
    return b"\x00"*((align-r)%align)

def write_gguf(path, arch, tensors):
    """tensors: list of (name, ggml_type, logical_shape_tuple, data_bytes(np.uint8/bytes)).
    logical_shape в numpy-порядке (rows, cols); мы сами развернём в ggml."""
    # --- метадата ---
    meta = b""
    kv = [ _kv_str("general.architecture", arch),
           _kv_u32("general.quantization_version", 2),
           _kv_u32("general.alignment", ALIGN),
           _kv_str("general.name", "xquant") ]
    meta = b"".join(kv)
    n_kv = len(kv)

    # --- тензор-инфо + расчёт оффсетов ---
    infos = b""
    data_blobs = []
    offset = 0
    for name, ttype, shape, data in tensors:
        db = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        ne = list(reversed([int(x) for x in shape]))  # ggml порядок
        info = _gstr(name) + struct.pack("<I", len(ne))
        for d in ne: info += struct.pack("<Q", d)
        info += struct.pack("<I", ttype) + struct.pack("<Q", offset)
        infos += info
        pad = _pad(len(db))
        data_blobs.append(db + pad)
        offset += len(db) + len(pad)

    header = struct.pack("<I", _MAGIC) + struct.pack("<I", 3) \
           + struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv)

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta)
        f.write(infos)
        # выравнивание перед секцией данных
        pos = len(header)+len(meta)+len(infos)
        f.write(_pad(pos))
        for blob in data_blobs:
            f.write(blob)


# ── кодировка простых типов (замена gguf.quants для F32/F16/BF16) ──
def enc_f32(a): return np.ascontiguousarray(a, np.float32).tobytes()
def enc_f16(a): return np.ascontiguousarray(a, np.float32).astype(np.float16).tobytes()
def enc_bf16(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u >> 16) & 0xFFFF).astype(np.uint16).tobytes()
