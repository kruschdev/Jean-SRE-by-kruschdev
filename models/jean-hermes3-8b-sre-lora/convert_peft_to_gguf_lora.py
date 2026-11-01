#!/usr/bin/env python3
"""
Convert a Hugging Face PEFT LoRA adapter (adapter_model.safetensors + adapter_config.json)
into a llama.cpp / Ollama compatible GGUF v3 LoRA adapter (adapter.gguf).

Losslessly converts BF16/F16/F32 PEFT weights into IEEE-754 F32 GGUF tensors with
32-byte alignment and architecture-matched GGUF tensor names.
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor types
GGML_TYPE_F32 = 0

LAYER_MAP = {
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.gate_proj": "ffn_gate.weight",
    "mlp.up_proj": "ffn_up.weight",
    "mlp.down_proj": "ffn_down.weight",
}


def pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_kv_string(key: str, val: str) -> bytes:
    return pack_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + pack_string(val)


def pack_kv_float32(key: str, val: float) -> bytes:
    return pack_string(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, float(val))


def pack_kv_uint32(key: str, val: int) -> bytes:
    return pack_string(key) + struct.pack("<II", GGUF_TYPE_UINT32, int(val))


def bf16_bytes_to_f32_bytes(raw: bytes) -> bytes:
    """Losslessly convert little-endian BF16 bytes to IEEE-754 F32 bytes."""
    # In little-endian, each 2-byte BF16 value becomes [0x00, 0x00, b0, b1] in F32.
    n = len(raw) // 2
    out = bytearray(n * 4)
    out[2::4] = raw[0::2]
    out[3::4] = raw[1::2]
    return bytes(out)


def map_peft_tensor_name(hf_name: str) -> str:
    m = re.match(
        r"^(?:base_model\.model\.)?model\.layers\.(\d+)\.([a-z_]+\.[a-z_]+)\.(lora_[AB])\.weight$",
        hf_name,
    )
    if not m:
        raise ValueError(f"Unsupported PEFT LoRA tensor name: {hf_name}")
    layer_idx, module_key, lora_ab = m.groups()
    if module_key not in LAYER_MAP:
        raise ValueError(f"Unsupported module key '{module_key}' in {hf_name}")
    gguf_base = LAYER_MAP[module_key]
    suffix = ".lora_a" if lora_ab == "lora_A" else ".lora_b"
    return f"blk.{layer_idx}.{gguf_base}{suffix}"


def convert_peft_to_gguf(adapter_dir: Path, output_path: Path, arch: str = "qwen3moe") -> Path:
    config_path = adapter_dir / "adapter_config.json"
    safetensors_path = adapter_dir / "adapter_model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing {safetensors_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lora_alpha = float(cfg.get("lora_alpha", 32.0))

    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        raw_blob = f.read()

    tensor_keys = sorted(k for k in header.keys() if k != "__metadata__")
    tensors = []
    current_offset = 0

    for hf_key in tensor_keys:
        info = header[hf_key]
        dtype = info["dtype"]
        shape = info["shape"]  # PyTorch [rows, cols]
        start, end = info["data_offsets"]
        tensor_bytes = raw_blob[start:end]

        if dtype == "BF16":
            f32_bytes = bf16_bytes_to_f32_bytes(tensor_bytes)
        elif dtype == "F32":
            f32_bytes = tensor_bytes
        else:
            raise ValueError(f"Unsupported dtype {dtype} in {hf_key}")

        gguf_name = map_peft_tensor_name(hf_key)
        # PyTorch [rows, cols] -> GGUF ne[0]=cols, ne[1]=rows
        ne0, ne1 = int(shape[1]), int(shape[0])

        # Align offset to 32 bytes
        pad = (GGUF_DEFAULT_ALIGNMENT - (current_offset % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        current_offset += pad

        tensors.append(
            {
                "name": gguf_name,
                "dims": (ne0, ne1),
                "type": GGML_TYPE_F32,
                "offset": current_offset,
                "pad_before": pad,
                "data": f32_bytes,
            }
        )
        current_offset += len(f32_bytes)

    kv_Section = b"".join(
        [
            pack_kv_string("general.type", "adapter"),
            pack_kv_string("general.architecture", arch),
            pack_kv_uint32("general.quantization_version", 2),
            pack_kv_uint32("general.alignment", GGUF_DEFAULT_ALIGNMENT),
            pack_kv_string("adapter.type", "lora"),
            pack_kv_float32("adapter.lora.alpha", lora_alpha),
        ]
    )
    n_kv = 6
    n_tensors = len(tensors)

    tensor_info_section = bytearray()
    for t in tensors:
        tensor_info_section += pack_string(t["name"])
        tensor_info_section += struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            tensor_info_section += struct.pack("<Q", d)
        tensor_info_section += struct.pack("<IQ", t["type"], t["offset"])

    header_bytes = (
        struct.pack("<4sIQQ", GGUF_MAGIC, GGUF_VERSION, n_tensors, n_kv)
        + kv_Section
        + bytes(tensor_info_section)
    )
    header_pad = (GGUF_DEFAULT_ALIGNMENT - (len(header_bytes) % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        out.write(header_bytes)
        if header_pad:
            out.write(b"\x00" * header_pad)
        for t in tensors:
            if t["pad_before"]:
                out.write(b"\x00" * t["pad_before"])
            out.write(t["data"])

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT safetensors LoRA to GGUF v3 LoRA")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--output", default=None, help="Output GGUF file path (defaults to <adapter-dir>/adapter.gguf)")
    parser.add_argument("--arch", default="qwen3moe", help="GGUF target architecture (default: qwen3moe)")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else (adapter_dir / "adapter.gguf")
    out = convert_peft_to_gguf(adapter_dir, output_path, arch=args.arch)
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"[convert_peft_to_gguf] Wrote {out} ({size_mb:.2f} MB, arch={args.arch})")


if __name__ == "__main__":
    main()
