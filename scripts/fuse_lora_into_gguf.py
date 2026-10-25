#!/usr/bin/env python3
"""
In-place QLoRA -> GGUF Attention Weight Fusion Tool (fuse_lora_into_gguf.py)

Fuses a GGUF v3 LoRA adapter (adapter.gguf) directly into the quantized attention
tensors (Q4_K type 12 / Q6_K type 14) of a GGUF v3 base model using Ollama's
native libggml-base.so reference dequantize/quantize routines and OpenMP multi-threading.

Why this exists:
- Ollama >= 0.40 deprecated the Modelfile `ADAPTER` directive ("Error: LoRA adapters are no longer supported").
- Fusing the rank-16 delta (W_merged = W_base + (alpha / r) * B @ A) directly into the
  192 attention matrices leaves 97.2% of the 30B MoE GGUF file (all 128 routed experts,
  embeddings, norms, metadata, and tokenizer) untouched, preserves the exact VRAM
  footprint (100% GPU across Dual RTX 2080 Ti NVLink), and eliminates runtime LoRA GEMV overhead.
"""

import argparse
import ctypes
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

GGUF_DEFAULT_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30
GGML_TYPE_MXFP4 = 39

QK_K = 256
BLOCK_Q4_K_BYTES = 144   # 256 elements -> 144 bytes
BLOCK_Q6_K_BYTES = 210   # 256 elements -> 210 bytes
BLOCK_Q8_0_BYTES = 34    # 32 elements -> 34 bytes
BLOCK_MXFP4_BYTES = 17   # 32 elements -> 17 bytes

C_KERNEL_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

extern void ggml_fp16_to_fp32_row(const void * x, float * y, int64_t n);
extern void ggml_fp32_to_fp16_row(const float * x, void * y, int64_t n);
extern void ggml_bf16_to_fp32_row(const void * x, float * y, int64_t n);
extern void ggml_fp32_to_bf16_row(const float * x, void * y, int64_t n);
extern void dequantize_row_q8_0(const void * vx, float * y, int64_t k);
extern void quantize_row_q8_0_ref(const float * x, void * vy, int64_t k);
extern void dequantize_row_q4_K(const void * vx, float * y, int64_t k);
extern void quantize_row_q4_K_ref(const float * x, void * vy, int64_t k);
extern void dequantize_row_q6_K(const void * vx, float * y, int64_t k);
extern void quantize_row_q6_K_ref(const float * x, void * vy, int64_t k);
extern void dequantize_row_mxfp4(const void * vx, float * y, int64_t k);
extern void quantize_row_mxfp4_ref(const float * x, void * vy, int64_t k);

static inline int64_t get_row_bytes(int ttype, int64_t ne0) {
    switch (ttype) {
        case 0:  return ne0 * 4;
        case 1:  return ne0 * 2;
        case 8:  return (ne0 / 32) * 34;
        case 12: return (ne0 / 256) * 144;
        case 14: return (ne0 / 256) * 210;
        case 30: return ne0 * 2;
        case 39: return (ne0 / 32) * 17;
        default: return 0;
    }
}

void fuse_lora_tensor(
    uint8_t * qdata,
    int ttype,
    int64_t ne0,
    int64_t ne1,
    int64_t r,
    float scale,
    const float * lora_a,
    const float * lora_b
) {
    const int64_t row_bytes = get_row_bytes(ttype, ne0);
    if (row_bytes <= 0) return;

    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc(sizeof(float) * ne0);
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < ne1; ++row) {
            uint8_t * qrow = qdata + row * row_bytes;
            if (ttype == 0) {
                memcpy(row_f32, qrow, sizeof(float) * ne0);
            } else if (ttype == 1) {
                ggml_fp16_to_fp32_row(qrow, row_f32, ne0);
            } else if (ttype == 8) {
                dequantize_row_q8_0(qrow, row_f32, ne0);
            } else if (ttype == 12) {
                dequantize_row_q4_K(qrow, row_f32, ne0);
            } else if (ttype == 14) {
                dequantize_row_q6_K(qrow, row_f32, ne0);
            } else if (ttype == 30) {
                ggml_bf16_to_fp32_row(qrow, row_f32, ne0);
            } else if (ttype == 39) {
                dequantize_row_mxfp4(qrow, row_f32, ne0);
            }

            const float * b_row = lora_b + row * r;
            for (int64_t k = 0; k < r; ++k) {
                const float s = scale * b_row[k];
                const float * a_row = lora_a + k * ne0;
                for (int64_t col = 0; col < ne0; ++col) {
                    row_f32[col] += s * a_row[col];
                }
            }

            if (ttype == 0) {
                memcpy(qrow, row_f32, sizeof(float) * ne0);
            } else if (ttype == 1) {
                ggml_fp32_to_fp16_row(row_f32, qrow, ne0);
            } else if (ttype == 8) {
                quantize_row_q8_0_ref(row_f32, qrow, ne0);
            } else if (ttype == 12) {
                quantize_row_q4_K_ref(row_f32, qrow, ne0);
            } else if (ttype == 14) {
                quantize_row_q6_K_ref(row_f32, qrow, ne0);
            } else if (ttype == 30) {
                ggml_fp32_to_bf16_row(row_f32, qrow, ne0);
            } else if (ttype == 39) {
                quantize_row_mxfp4_ref(row_f32, qrow, ne0);
            }
        }
        free(row_f32);
    }
}
"""


def compile_helper_so(ggml_lib_dir: str = "/usr/local/lib/ollama") -> ctypes.CDLL:
    so_path = Path("/tmp/libgguf_lora_fuse.so")
    c_path = Path("/tmp/libgguf_lora_fuse.c")
    c_path.write_text(C_KERNEL_SOURCE, encoding="utf-8")

    ggml_base = Path(ggml_lib_dir) / "libggml-base.so.0"
    if not ggml_base.exists():
        ggml_base = Path(ggml_lib_dir) / "libggml-base.so"

    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-shared",
        "-fPIC",
        str(c_path),
        str(ggml_base),
        f"-Wl,-rpath,{ggml_lib_dir}",
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    ctypes.CDLL(str(ggml_base), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(so_path))
    lib.fuse_lora_tensor.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.fuse_lora_tensor.restype = None
    return lib


def parse_gguf_index(path: Path):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a valid GGUF file (magic={magic})")

        def read_str():
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8", errors="replace")

        def read_val(vtype):
            sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
            if vtype in sizes:
                fmt = sizes[vtype]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            elif vtype == 8:
                return read_str()
            elif vtype == 9:
                atype, alen = struct.unpack("<IQ", f.read(12))
                if atype in sizes:
                    f.seek(struct.calcsize(sizes[atype]) * alen, 1)
                else:
                    for _ in range(alen):
                        read_val(atype)
                return None
            else:
                raise ValueError(f"Unknown GGUF KV type {vtype}")

        kvs = {}
        for _ in range(n_kv):
            k = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            v = read_val(vtype)
            if v is not None:
                kvs[k] = v

        tensors = {}
        for _ in range(n_tensors):
            tname = read_str()
            ndims = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(ndims))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[tname] = {"dims": dims, "type": ttype, "offset": offset}

        alignment = int(kvs.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        pos = f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        data_start = pos + pad

    return kvs, tensors, data_start


def row_byte_size(ne0: int, ttype: int) -> int:
    if ttype == GGML_TYPE_F32:
        return ne0 * 4
    elif ttype == GGML_TYPE_F16 or ttype == GGML_TYPE_BF16:
        return ne0 * 2
    elif ttype == GGML_TYPE_Q8_0:
        return (ne0 // 32) * BLOCK_Q8_0_BYTES
    elif ttype == GGML_TYPE_Q4_K:
        return (ne0 // QK_K) * BLOCK_Q4_K_BYTES
    elif ttype == GGML_TYPE_Q6_K:
        return (ne0 // QK_K) * BLOCK_Q6_K_BYTES
    elif ttype == GGML_TYPE_MXFP4:
        return (ne0 // 32) * BLOCK_MXFP4_BYTES
    raise ValueError(f"Unsupported tensor type {ttype}")


def tensor_byte_size(dims, ttype: int) -> int:
    ne0 = dims[0]
    rows = 1
    for d in dims[1:]:
        rows *= d
    return row_byte_size(ne0, ttype) * rows


def resolve_base_tensor_slice(base_name: str, base_tensors: dict, lora_tensors: dict):
    """Return (tensor_dict, row_start, ne1_slice, resolved_label) for split or concatenated attn_qkv."""
    if base_name in base_tensors:
        bt = base_tensors[base_name]
        return bt, 0, bt["dims"][1], base_name

    # Fallback for architectures (e.g. gpt-oss) that pack Q+K+V into blk.{L}.attn_qkv.weight
    for part in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
        if base_name.endswith("." + part):
            prefix = base_name[: -len(part)]  # e.g. "blk.0."
            qkv_name = prefix + "attn_qkv.weight"
            if qkv_name in base_tensors:
                bt_qkv = base_tensors[qkv_name]
                q_b = lora_tensors[prefix + "attn_q.weight.lora_b"]["dims"][1]
                k_b = lora_tensors[prefix + "attn_k.weight.lora_b"]["dims"][1]
                v_b = lora_tensors[prefix + "attn_v.weight.lora_b"]["dims"][1]
                if part == "attn_q.weight":
                    return bt_qkv, 0, q_b, f"{qkv_name}[0:{q_b}]"
                elif part == "attn_k.weight":
                    return bt_qkv, q_b, k_b, f"{qkv_name}[{q_b}:{q_b + k_b}]"
                else:
                    return bt_qkv, q_b + k_b, v_b, f"{qkv_name}[{q_b + k_b}:{q_b + k_b + v_b}]"

    raise KeyError(f"Base tensor {base_name} (or attn_qkv.weight) not found in base GGUF")


def fuse_lora_into_gguf(base_gguf: Path, adapter_gguf: Path, output_gguf: Path):
    t0 = time.monotonic()
    lib = compile_helper_so()

    print(f"[fuse] Parsing LoRA adapter index: {adapter_gguf}")
    lora_kvs, lora_tensors, lora_data_start = parse_gguf_index(adapter_gguf)
    lora_alpha = float(lora_kvs.get("adapter.lora.alpha", 32.0))

    print(f"[fuse] Parsing base GGUF index: {base_gguf}")
    _, base_tensors, base_data_start = parse_gguf_index(base_gguf)

    if base_gguf.resolve() != output_gguf.resolve():
        output_gguf.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = output_gguf.with_suffix(".gguf.tmp")
        print(f"[fuse] Copying base GGUF ({os.path.getsize(base_gguf) / (1024**3):.2f} GB) -> {tmp_out} ...")
        shutil.copyfile(base_gguf, tmp_out)
    else:
        tmp_out = output_gguf

    with open(adapter_gguf, "rb") as f_lora:
        lora_blob = f_lora.read()

    lora_pairs = []
    for tname in sorted(lora_tensors.keys()):
        if tname.endswith(".lora_a"):
            base_name = tname[: -len(".lora_a")]
            b_name = base_name + ".lora_b"
            if b_name not in lora_tensors:
                raise ValueError(f"Missing {b_name} in {adapter_gguf}")
            lora_pairs.append((base_name, tname, b_name))

    print(f"[fuse] Fusing {len(lora_pairs)} LoRA tensor pairs (alpha={lora_alpha}) using OpenMP...")
    t_fuse_start = time.monotonic()

    with open(tmp_out, "r+b") as f_out:
        for idx, (base_name, a_name, b_name) in enumerate(lora_pairs, 1):
            bt, row_start, ne1, resolved_label = resolve_base_tensor_slice(base_name, base_tensors, lora_tensors)
            at = lora_tensors[a_name]
            bt_lora = lora_tensors[b_name]

            ne0 = bt["dims"][0]
            r = at["dims"][1]
            scale = lora_alpha / float(r)
            ttype = bt["type"]

            r_bytes = row_byte_size(ne0, ttype)
            q_bytes_len = r_bytes * ne1
            a_bytes_len = tensor_byte_size(at["dims"], at["type"])
            b_bytes_len = tensor_byte_size(bt_lora["dims"], bt_lora["type"])

            abs_offset = base_data_start + bt["offset"] + (row_start * r_bytes)
            f_out.seek(abs_offset)
            q_buf = bytearray(f_out.read(q_bytes_len))

            a_start = lora_data_start + at["offset"]
            b_start = lora_data_start + bt_lora["offset"]
            a_buf = lora_blob[a_start : a_start + a_bytes_len]
            b_buf = lora_blob[b_start : b_start + b_bytes_len]

            q_ptr = (ctypes.c_uint8 * q_bytes_len).from_buffer(q_buf)
            a_ptr = ctypes.cast(ctypes.c_char_p(a_buf), ctypes.c_void_p)
            b_ptr = ctypes.cast(ctypes.c_char_p(b_buf), ctypes.c_void_p)

            lib.fuse_lora_tensor(
                ctypes.cast(q_ptr, ctypes.c_void_p),
                int(ttype),
                int(ne0),
                int(ne1),
                int(r),
                float(scale),
                a_ptr,
                b_ptr,
            )

            f_out.seek(abs_offset)
            f_out.write(q_buf)

            if idx % 32 == 0 or idx == len(lora_pairs):
                print(f"[fuse]   {idx}/{len(lora_pairs)} tensors fused ({resolved_label}, type={ttype}, shape={ne0}x{ne1}, r={r})")

    if tmp_out != output_gguf:
        os.replace(tmp_out, output_gguf)

    dt_fuse = time.monotonic() - t_fuse_start
    dt_total = time.monotonic() - t0
    print(f"[fuse] Done! Fused {len(lora_pairs)} tensors in {dt_fuse:.2f}s (total {dt_total:.2f}s) -> {output_gguf}")


def main():
    parser = argparse.ArgumentParser(description="Fuse GGUF v3 LoRA adapter into quantized GGUF v3 base model in-place")
    parser.add_argument("--base-gguf", required=True, help="Path to base GGUF file")
    parser.add_argument("--adapter-gguf", required=True, help="Path to GGUF LoRA adapter (adapter.gguf)")
    parser.add_argument("--output-gguf", required=True, help="Path to output fused GGUF file")
    args = parser.parse_args()

    fuse_lora_into_gguf(
        Path(args.base_gguf),
        Path(args.adapter_gguf),
        Path(args.output_gguf),
    )


if __name__ == "__main__":
    main()
