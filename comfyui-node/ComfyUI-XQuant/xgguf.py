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


# ═══════════ ЧТЕНИЕ GGUF (для requantize LLM) — свой ридер ═══════════
# Метадату (вкл. токенайзер) копируем СЫРЫМИ байтами → сохраняется как есть.
_VT_FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}  # value_type → размер

def _skip_kv_value(f, vt):
    if vt in _VT_FIXED: f.read(_VT_FIXED[vt])
    elif vt == _STR: n=struct.unpack("<Q",f.read(8))[0]; f.read(n)
    elif vt == 9:  # ARRAY: elem_type(u32) + count(u64) + elems
        et=struct.unpack("<I",f.read(4))[0]; cnt=struct.unpack("<Q",f.read(8))[0]
        for _ in range(cnt):
            if et==_STR: n=struct.unpack("<Q",f.read(8))[0]; f.read(n)
            elif et==9: _skip_kv_value(f, 9)
            else: f.read(_VT_FIXED.get(et,4))
    else: raise ValueError(f"unknown KV value_type {vt}")

def _val_end(buf, pos, vt):
    """Позиция сразу ПОСЛЕ значения KV в буфере (для in-place патча метадаты)."""
    if vt in _VT_FIXED: return pos + _VT_FIXED[vt]
    if vt == _STR:
        n = struct.unpack_from("<Q", buf, pos)[0]; return pos + 8 + n
    if vt == 9:
        et = struct.unpack_from("<I", buf, pos)[0]; cnt = struct.unpack_from("<Q", buf, pos+4)[0]
        p = pos + 12
        for _ in range(cnt):
            if et == _STR: n = struct.unpack_from("<Q", buf, p)[0]; p += 8 + n
            elif et == 9: p = _val_end(buf, p, 9)
            else: p += _VT_FIXED.get(et, 4)
        return p
    raise ValueError(f"vt {vt}")

def patch_kv_u32(raw_meta, key, new_val):
    """Переписать u32-значение ключа `key` в сырой метадате (длина не меняется).
    Нужно чтобы general.file_type соответствовал новому кванту (иначе LM Studio
    видит рассинхрон типа и может спрятать/отбраковать модель)."""
    buf = bytearray(raw_meta); kb = key.encode("utf-8"); pos = 0
    while pos < len(buf):
        klen = struct.unpack_from("<Q", buf, pos)[0]; p = pos + 8
        k = bytes(buf[p:p+klen]); p += klen
        vt = struct.unpack_from("<I", buf, p)[0]; p += 4
        if k == kb and vt == _U32:
            struct.pack_into("<I", buf, p, int(new_val) & 0xFFFFFFFF)
            return bytes(buf)
        pos = _val_end(buf, p, vt)
    return bytes(buf)  # ключ не найден — вернуть как есть

# LLAMA_FTYPE: наш выходной квант → значение general.file_type
FTYPE = {"Q4_0":2, "Q5_0":8, "Q8_0":7, "Q2_K":10, "Q3_K":12, "Q6_K":18}

def read_gguf(path):
    """Вернуть (f, version, raw_meta_bytes, n_kv, tensor_infos, data_start, align).
    tensor_infos: list of (name, dims_tuple_ggml, ggml_type, offset)."""
    f = open(path, "rb")
    magic, ver = struct.unpack("<II", f.read(8))
    if magic != _MAGIC: raise ValueError("не GGUF")
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    meta_start = f.tell()
    align = ALIGN
    for _ in range(n_kv):
        kn = struct.unpack("<Q", f.read(8))[0]; key = f.read(kn).decode("utf-8","replace")
        vt = struct.unpack("<I", f.read(4))[0]
        vpos = f.tell()
        if key == "general.alignment" and vt == _U32:
            align = struct.unpack("<I", f.read(4))[0];
        else:
            _skip_kv_value(f, vt)
    meta_end = f.tell()
    f.seek(meta_start); raw_meta = f.read(meta_end - meta_start); f.seek(meta_end)
    tinfos = []
    for _ in range(n_tensors):
        nn = struct.unpack("<Q", f.read(8))[0]; name = f.read(nn).decode("utf-8","replace")
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tinfos.append((name, dims, tt, off))
    after = f.tell()
    data_start = after + ((align - after % align) % align)
    return f, ver, raw_meta, n_kv, tinfos, data_start, align

def write_gguf_raw(path, raw_meta, n_kv, tensors):
    """Записать GGUF с ГОТОВОЙ метадатой (сырые байты) + новыми тензорами.
    tensors: list of (name, ggml_type, ggml_dims_tuple, data_bytes)."""
    infos = b""; blobs = []; offset = 0
    for name, tt, ne, data in tensors:
        db = bytes(data)
        info = _gstr(name) + struct.pack("<I", len(ne))
        for d in ne: info += struct.pack("<Q", int(d))
        info += struct.pack("<I", tt) + struct.pack("<Q", offset)
        infos += info; pad=_pad(len(db)); blobs.append(db+pad); offset += len(db)+len(pad)
    header = struct.pack("<I",_MAGIC)+struct.pack("<I",3)+struct.pack("<Q",len(tensors))+struct.pack("<Q",n_kv)
    with open(path,"wb") as f:
        f.write(header); f.write(raw_meta); f.write(infos)
        f.write(_pad(len(header)+len(raw_meta)+len(infos)))
        for b in blobs: f.write(b)


# ══════════ ДЕКВАНТ K-квантов (ggml-точные раскладки, векторно) ══════════
# QK_K=256. Все super-block'и деквантятся полностью numpy'ем (без python-циклов
# по блокам). Нужны, чтобы РЕКВАНТовать уже-квантованный GGUF (Q4_K_M и т.п.),
# а не только F16/Q8. Формулы = 1:1 из ggml dequantize_row_*.

def _fp16(b2):  # (nb,2) uint8 → (nb,1) float32
    return b2.copy().view(np.float16).astype(np.float32).reshape(-1,1)

def _deq_q4_0(raw, n):  # 18б/32: d(fp16)+16б(32×4бит)
    b=np.frombuffer(raw,np.uint8).reshape(-1,18)
    d=_fp16(b[:,0:2]); q=b[:,2:18].astype(np.int32)
    lo=(q & 0xF)-8; hi=(q >> 4)-8
    y=np.empty((b.shape[0],32),np.float32)
    y[:,0:16]=d*lo; y[:,16:32]=d*hi
    return y.reshape(-1)[:n]

def _deq_q5_0(raw, n):  # 22б/32: d(fp16)+qh(4б=32бит)+16б(32×4бит)
    b=np.frombuffer(raw,np.uint8).reshape(-1,22)
    d=_fp16(b[:,0:2])
    qh=b[:,2:6].copy().view(np.uint32).reshape(-1)  # (nb,)
    q=b[:,6:22].astype(np.int32)
    lo=q & 0xF; hi=q >> 4
    y=np.empty((b.shape[0],32),np.float32)
    idx=np.arange(16)
    h5_lo=((qh[:,None] >> idx) & 1) << 4          # старший 5-й бит для l<16
    h5_hi=((qh[:,None] >> (idx+16)) & 1) << 4     # для l>=16
    y[:,0:16]=d*((lo | h5_lo)-16)
    y[:,16:32]=d*((hi | h5_hi)-16)
    return y.reshape(-1)[:n]

def _scale_min_k4(s):  # s:(nb,12) int32 → sc(nb,8),mn(nb,8)
    sc=np.empty((s.shape[0],8),np.int32); mn=np.empty((s.shape[0],8),np.int32)
    for j in range(4):
        sc[:,j]=s[:,j] & 63; mn[:,j]=s[:,j+4] & 63
    for j in range(4,8):
        sc[:,j]=(s[:,j+4] & 0xF) | ((s[:,j-4] >> 6) << 4)
        mn[:,j]=(s[:,j+4] >> 4)  | ((s[:,j]   >> 6) << 4)
    return sc, mn

def _deq_q4_k(raw, n):  # 144б/256
    b=np.frombuffer(raw,np.uint8).reshape(-1,144); nb=b.shape[0]
    d=_fp16(b[:,0:2]); dmin=_fp16(b[:,2:4])
    sc,mn=_scale_min_k4(b[:,4:16].astype(np.int32))
    qs=b[:,16:144].astype(np.int32)  # (nb,128)
    y=np.empty((nb,256),np.float32)
    for p in range(4):  # 4 пары суб-блоков по 64
        blk=qs[:,p*32:p*32+32]
        lo=blk & 0xF; hi=blk >> 4
        y[:,p*64:p*64+32]   = d*sc[:,2*p:2*p+1]   * lo - dmin*mn[:,2*p:2*p+1]
        y[:,p*64+32:p*64+64]= d*sc[:,2*p+1:2*p+2] * hi - dmin*mn[:,2*p+1:2*p+2]
    return y.reshape(-1)[:n]

def _deq_q5_k(raw, n):  # 176б/256
    b=np.frombuffer(raw,np.uint8).reshape(-1,176); nb=b.shape[0]
    d=_fp16(b[:,0:2]); dmin=_fp16(b[:,2:4])
    sc,mn=_scale_min_k4(b[:,4:16].astype(np.int32))
    qh=b[:,16:48].astype(np.int32)   # (nb,32)
    ql=b[:,48:176].astype(np.int32)  # (nb,128)
    y=np.empty((nb,256),np.float32)
    for p in range(4):
        blk=ql[:,p*32:p*32+32]
        lo=blk & 0xF; hi=blk >> 4
        bit_lo=(qh >> (2*p))   & 1  # (nb,32)
        bit_hi=(qh >> (2*p+1)) & 1
        y[:,p*64:p*64+32]   = d*sc[:,2*p:2*p+1]   * (lo + (bit_lo<<4)) - dmin*mn[:,2*p:2*p+1]
        y[:,p*64+32:p*64+64]= d*sc[:,2*p+1:2*p+2] * (hi + (bit_hi<<4)) - dmin*mn[:,2*p+1:2*p+2]
    return y.reshape(-1)[:n]

def _deq_q6_k(raw, n):  # 210б/256
    b=np.frombuffer(raw,np.uint8).reshape(-1,210); nb=b.shape[0]
    ql=b[:,0:128].astype(np.int32)
    qh=b[:,128:192].astype(np.int32)
    sc=b[:,192:208].view(np.int8).astype(np.int32).reshape(nb,16)
    d=_fp16(b[:,208:210])
    y=np.empty((nb,256),np.float32)
    for h in range(2):  # два блока по 128
        qlh=ql[:,h*64:h*64+64]; qhh=qh[:,h*32:h*32+32]; sch=sc[:,h*8:h*8+8]
        l=np.arange(32)
        is_=l//16  # 0/1
        q1=((qlh[:,0:32]  & 0xF) | (((qhh >> 0) & 3) << 4)) - 32
        q2=((qlh[:,32:64] & 0xF) | (((qhh >> 2) & 3) << 4)) - 32
        q3=((qlh[:,0:32]  >> 4)  | (((qhh >> 4) & 3) << 4)) - 32
        q4=((qlh[:,32:64] >> 4)  | (((qhh >> 6) & 3) << 4)) - 32
        base=h*128
        y[:,base+0 :base+32 ] = d * sch[:,is_+0] * q1
        y[:,base+32:base+64 ] = d * sch[:,is_+2] * q2
        y[:,base+64:base+96 ] = d * sch[:,is_+4] * q3
        y[:,base+96:base+128] = d * sch[:,is_+6] * q4
    return y.reshape(-1)[:n]

def _deq_q2_k(raw, n):  # 84б/256
    b=np.frombuffer(raw,np.uint8).reshape(-1,84); nb=b.shape[0]
    scales=b[:,0:16].astype(np.int32)
    qs=b[:,16:80].astype(np.int32)  # (nb,64)
    d=_fp16(b[:,80:82]); dmin=_fp16(b[:,82:84])
    y=np.empty((nb,256),np.float32)
    op=0
    for grp in range(2):          # QK_K по 128
        q=qs[:,grp*32:grp*32+32]  # (nb,32)
        is0=grp*8
        for j in range(4):
            shift=2*j
            sca=scales[:,is0+2*j];  dl=d*(sca & 0xF)[:,None]; ml=dmin*(sca>>4)[:,None]
            y[:,op:op+16]=dl*((q[:,0:16]>>shift)&3)-ml; op+=16
            scb=scales[:,is0+2*j+1]; dl=d*(scb & 0xF)[:,None]; ml=dmin*(scb>>4)[:,None]
            y[:,op:op+16]=dl*((q[:,16:32]>>shift)&3)-ml; op+=16
    return y.reshape(-1)[:n]

def _deq_q3_k(raw, n):  # 110б/256
    b=np.frombuffer(raw,np.uint8).reshape(-1,110); nb=b.shape[0]
    hm=b[:,0:32].astype(np.int32)
    qs=b[:,32:96].astype(np.int32)  # (nb,64)
    d=_fp16(b[:,108:110])
    # распаковка 6-битных scales из 12 байт (aux трюк ggml)
    a=b[:,96:108].astype(np.uint32)
    a0=a[:,0]|(a[:,1]<<8)|(a[:,2]<<16)|(a[:,3]<<24)
    a1=a[:,4]|(a[:,5]<<8)|(a[:,6]<<16)|(a[:,7]<<24)
    a2=a[:,8]|(a[:,9]<<8)|(a[:,10]<<16)|(a[:,11]<<24)
    km1=np.uint32(0x03030303); km2=np.uint32(0x0f0f0f0f)
    tmp=a2.copy()
    na2=((a0>>4)&km2)|(((tmp>>4)&km1)<<4)
    na3=((a1>>4)&km2)|(((tmp>>6)&km1)<<4)
    na0=(a0&km2)|(((tmp>>0)&km1)<<4)
    na1=(a1&km2)|(((tmp>>2)&km1)<<4)
    # scales[16] int8 из 4×uint32
    sc=np.empty((nb,16),np.int32)
    for k,word in enumerate((na0,na1,na2,na3)):
        sc[:,k*4+0]=(word    )&0xFF; sc[:,k*4+1]=(word>>8 )&0xFF
        sc[:,k*4+2]=(word>>16)&0xFF; sc[:,k*4+3]=(word>>24)&0xFF
    sc=sc.astype(np.int8).astype(np.int32)  # знаковые
    y=np.empty((nb,256),np.float32); op=0
    for grp in range(2):
        q=qs[:,grp*32:grp*32+32]
        is0=grp*8; m_bit=grp*4  # маска hmask: биты подряд по shift-итерациям
        for j in range(4):
            shift=2*j; mask=1<<(grp*4+j)
            dl=d*(sc[:,is0+2*j]-32)[:,None]
            hbit0=np.where((hm[:,0:16]&mask)!=0,0,4)
            y[:,op:op+16]=dl*(((q[:,0:16]>>shift)&3)-hbit0); op+=16
            dl=d*(sc[:,is0+2*j+1]-32)[:,None]
            hbit1=np.where((hm[:,16:32]&mask)!=0,0,4)
            y[:,op:op+16]=dl*(((q[:,16:32]>>shift)&3)-hbit1); op+=16
    return y.reshape(-1)[:n]

_KDEQ = {T.Q4_0:_deq_q4_0, T.Q5_0:_deq_q5_0, T.Q2_K:_deq_q2_k, T.Q3_K:_deq_q3_k,
         12:_deq_q4_k, 13:_deq_q5_k, T.Q6_K:_deq_q6_k}  # 12=Q4_K 13=Q5_K

# ── дековод source-тензоров GGUF для реквантизации ──
def dec_source(raw, ggml_type, n):
    if ggml_type == T.F32: return np.frombuffer(raw, np.float32)[:n].astype(np.float32)
    if ggml_type == T.F16: return np.frombuffer(raw, np.float16)[:n].astype(np.float32)
    if ggml_type == T.BF16:
        u=np.frombuffer(raw, np.uint16).astype(np.uint32); return ((u<<16).view(np.float32))[:n]
    if ggml_type == T.Q8_0:  # блок 34б: fp16 d + 32 int8
        b=np.frombuffer(raw, np.uint8).reshape(-1,34)
        d=b[:,0:2].copy().view(np.float16).astype(np.float32).reshape(-1,1)
        q=b[:,2:34].view(np.int8).astype(np.float32)
        return (d*q).reshape(-1)[:n]
    fn=_KDEQ.get(ggml_type)
    if fn is not None:
        try: return fn(raw, n)
        except Exception: return None
    return None  # неизвестный квант — пропуск


# ── кодировка простых типов (замена gguf.quants для F32/F16/BF16) ──
def enc_f32(a): return np.ascontiguousarray(a, np.float32).tobytes()
def enc_f16(a): return np.ascontiguousarray(a, np.float32).astype(np.float16).tobytes()
def enc_bf16(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u >> 16) & 0xFFFF).astype(np.uint16).tobytes()
