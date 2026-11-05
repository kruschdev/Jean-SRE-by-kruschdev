#!/usr/bin/env python3
"""
Unified Agentic SRE + Multi-Turn Tool-Caller QLoRA + DPO Pipeline for Google Gemma 4 31B Dense (google/gemma-4-31b-it)
Target Accelerator: Colab G4 96GB RTX PRO 6000 Blackwell

Why this exists:
1. Trains the #1 Open-Weights 30.7B Dense model (`google/gemma-4-31b-it` / `gemma4:31b`)
   on the Wave-6 Unified SRE + Multi-Turn Tool-Calling curriculum (Referee CLEAN_PASS precision, P0/P1 recall,
   Lego-First `@krusch/toolkit` synthesis, partial-briefing gap search, exhaustive multi-clause tool arguments,
   and temporal state overrides).
2. Attaches LoRA adapters strictly to the 60 language_model layers (excluding vision_tower projections),
   provably anchors Stage-2 DPO reference policy (`ref`) to frozen Stage-1 SFT weights in aligned BF16 on `trl >= 1.0`,
   and emits a Gemma 4 Ollama Modelfile (`RENDERER gemma4` / `PARSER gemma4`) for in-place GGUF fusion.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import inspect
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

try:
    import datasets as hf_datasets
    import peft as hf_peft
    import torch
    import transformers as hf_transformers
    import trl as hf_trl
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
except ImportError:  # pragma: no cover - allows CPU-only unit tests to import adapter helpers
    hf_datasets = None  # type: ignore[assignment]
    hf_peft = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    hf_transformers = None  # type: ignore[assignment]
    hf_trl = None  # type: ignore[assignment]
    Dataset = Any  # type: ignore[assignment,misc]
    LoraConfig = PeftModel = get_peft_model = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None  # type: ignore[assignment]
    DPOConfig = DPOTrainer = SFTConfig = SFTTrainer = None  # type: ignore[assignment]


DEFAULT_BASE_MODEL = "google/gemma-4-31b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT + DPO fine-tune google/gemma-4-31b-it on Argon, Lego-First, and Calibrated Referee trajectories."
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "sft_dpo"],
        default="sft_dpo",
        help="Training stage: 'sft', 'dpo', or 'sft_dpo' (single-load Stage-1 SFT -> Stage-2 DPO with frozen SFT KL anchor).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model ID for Google Gemma 4 31B Instruct.",
    )
    parser.add_argument(
        "--dataset",
        default="data/distillation/gpt_oss_120b_elitist_teacher.jsonl",
        help="Path to curated ChatML JSONL dataset.",
    )
    parser.add_argument(
        "--argon-dataset",
        default="data/argon_remediation_sft_corpus.jsonl",
        help="Optional path to Gemini 4 Argon verified Golden Remediation JSONL dataset (oversampled 2x).",
    )
    parser.add_argument(
        "--calibration-dataset",
        default="data/distillation/lego_and_referee_calibration_sft.jsonl",
        help="Pinned CLEAN_PASS + P0/P1 Remediation + Lego-First ChatML SFT split (never dropped).",
    )
    parser.add_argument(
        "--calibration-oversample",
        type=int,
        default=3,
        help="Oversample factor for the pinned Lego-First + Referee calibration SFT dataset.",
    )
    parser.add_argument(
        "--agentic-dataset",
        default="data/distillation/agentic_tool_calling_sft.jsonl",
        help="Pinned multi-turn SRE + Workspace tool-calling SFT split.",
    )
    parser.add_argument(
        "--agentic-oversample",
        type=int,
        default=4,
        help="Oversample factor for the pinned multi-turn tool-calling SFT dataset (Wave-6 default: 4x = 304 rows).",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="data/distillation/referee_calibration_dpo.jsonl",
        help="Balanced Precision + Recall + Architecture Calibration DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--agentic-dpo-dataset",
        default="data/distillation/agentic_tool_calling_dpo.jsonl",
        help="Multi-turn tool-action + partial-briefing gap search + exhaustive-arg DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--agentic-dpo-oversample",
        type=int,
        default=3,
        help="Oversample factor for the agentic tool-calling DPO preference dataset (Wave-6 default: 3x).",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional existing PEFT LoRA checkpoint directory to resume/calibrate.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/jean-gemma4-31b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT epochs (or DPO epochs when --mode dpo).")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="Number of Stage-2 DPO epochs when --mode sft_dpo.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max token context during QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=460, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (1.0x scale to prevent pretrained entity-recall drift).")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Learning rate for SFT (5e-5).")
    parser.add_argument("--dpo-lr", type=float, default=5.0e-6, help="Learning rate for DPO calibration (5e-6).")
    parser.add_argument("--dpo-beta", type=float, default=0.2, help="KL penalty beta for DPOTrainer (0.2 stronger KL anchor).")
    return parser.parse_args()


def _clean_utf8(text: str) -> str:
    return str(text or "").encode("utf-8", "replace").decode("utf-8")


def _extract_inline_think(content: str) -> tuple[str, str]:
    """If content starts with <think>...</think>, split into (reasoning, clean_content)."""
    if "<think>" in content and "</think>" in content:
        start = content.find("<think>") + len("<think>")
        end = content.find("</think>")
        if start <= end:
            thought = content[start:end].strip()
            rest = (content[: content.find("<think>")] + content[end + len("</think>") :]).strip()
            return thought, rest
    return "", content


def _clean_message(m: dict, default_role: str = "user") -> dict:
    role = _clean_utf8(m.get("role", default_role))
    raw_content = _clean_utf8(m.get("content", ""))
    inline_thought, clean_content = _extract_inline_think(raw_content)
    out: dict = {"role": role, "content": clean_content}

    # Wave-5 Fix #1: Preserve `thinking` / `reasoning` / `reasoning_content` as `reasoning` for Gemma 4 chat_template.jinja
    reasoning_val = _clean_utf8(
        m.get("reasoning") or m.get("reasoning_content") or m.get("thinking") or inline_thought
    ).strip()
    if reasoning_val and role == "assistant":
        out["reasoning"] = reasoning_val

    raw_tcs = m.get("tool_calls")
    if isinstance(raw_tcs, list) and len(raw_tcs) > 0:
        norm_tcs = []
        for idx, tc in enumerate(raw_tcs):
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            args_val = fn.get("arguments", {})
            if isinstance(args_val, str):
                try:
                    args_val = json.loads(args_val)
                except Exception:
                    args_val = {}
            if not isinstance(args_val, dict):
                args_val = {}
            norm_tcs.append(
                {
                    "id": _clean_utf8(tc.get("id", f"call_{idx+1}")) if isinstance(tc, dict) else f"call_{idx+1}",
                    "type": "function",
                    "function": {
                        "name": _clean_utf8(fn.get("name", "tool")),
                        "arguments": args_val,
                    },
                }
            )
        out["tool_calls"] = norm_tcs
    if isinstance(m.get("tool_call_id"), str) and m["tool_call_id"]:
        out["tool_call_id"] = _clean_utf8(m["tool_call_id"])
    if isinstance(m.get("name"), str) and m["name"]:
        out["name"] = _clean_utf8(m["name"])
    return out


def _read_rendered_sft_records(file_paths: list[str], tokenizer, max_chars: int) -> list[dict]:
    records = []
    for fp in file_paths:
        if not fp or not Path(fp).exists():
            continue
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages")
                if not isinstance(msgs, list) or len(msgs) < 2:
                    continue
                cleaned_msgs = [_clean_message(m) for m in msgs if isinstance(m, dict)]
                tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
                has_reasoning = any(bool(m.get("reasoning")) for m in cleaned_msgs)
                rendered = tokenizer.apply_chat_template(
                    cleaned_msgs,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=has_reasoning,
                    preserve_thinking=True,
                )
                if len(rendered) <= max_chars:
                    records.append({"text": rendered})
    return records


class Gemma4AssistantOnlyCollator:
    """
    Wave-5 Fix #2: Masks SFT cross-entropy loss (`-100`) on all `<|turn>system`, `<|turn>user`,
    and `<|tool_response>...<tool_response|>` spans so 100% of gradient updates train strictly
    `<|turn>model\n` assistant reasoning (`<|channel>thought...<channel|>`), `<|tool_call>...<tool_call|>`,
    and final assistant responses.
    """

    def __init__(self, tokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @staticmethod
    def _compute_assistant_char_spans(text: str) -> list[tuple[int, int]]:
        """Return character intervals [start, end) inside `<|turn>model\\n...<turn|>` Excluding `<|tool_response>...<tool_response|>`."""
        spans: list[tuple[int, int]] = []
        marker = "<|turn>model\n"
        end_turn = "<turn|>"
        tr_open = "<|tool_response>"
        tr_close = "<tool_response|>"
        pos = 0
        n = len(text)
        while pos < n:
            idx = text.find(marker, pos)
            if idx == -1:
                break
            content_start = idx + len(marker)
            turn_end_idx = text.find(end_turn, content_start)
            if turn_end_idx == -1:
                turn_block_end = n
                next_pos = n
            else:
                turn_block_end = turn_end_idx + len(end_turn)
                next_pos = turn_block_end

            # Split around any embedded <|tool_response>...<tool_response|> blocks inside a continued model turn
            cur = content_start
            while cur < turn_block_end:
                tr_idx = text.find(tr_open, cur, turn_block_end)
                if tr_idx == -1:
                    if cur < turn_block_end:
                        spans.append((cur, turn_block_end))
                    break
                if tr_idx > cur:
                    spans.append((cur, tr_idx))
                tr_end = text.find(tr_close, tr_idx + len(tr_open), turn_block_end)
                if tr_end == -1:
                    break
                cur = tr_end + len(tr_close)
            pos = next_pos
        return spans

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feat in features:
            if "text" in feat and isinstance(feat["text"], str):
                text = feat["text"]
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                input_ids = list(enc["input_ids"])
                offsets = enc["offset_mapping"]
                spans = self._compute_assistant_char_spans(text)
                labels = [-100] * len(input_ids)
                for tok_idx, (c_start, c_end) in enumerate(offsets):
                    if c_end <= c_start:
                        continue
                    for s_start, s_end in spans:
                        if c_end > s_start and c_start < s_end:
                            labels[tok_idx] = input_ids[tok_idx]
                            break
                if all(x == -100 for x in labels):
                    labels = list(input_ids)
            else:
                input_ids = list(feat["input_ids"])[: self.max_length]
                labels = list(feat.get("labels", input_ids))[: self.max_length]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append([1] * len(input_ids))
            batch_labels.append(labels)

        max_len = max(len(x) for x in batch_input_ids)
        padded_ids = []
        padded_mask = []
        padded_labels = []
        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def load_chatml_dataset(
    data_files: list[str],
    tokenizer,
    max_chars: int = 36000,
    max_samples: int = 420,
    pinned_files: list[str] | None = None,
) -> Dataset:
    pinned_records = _read_rendered_sft_records(pinned_files or [], tokenizer=tokenizer, max_chars=max_chars)
    regular_records = _read_rendered_sft_records(data_files, tokenizer=tokenizer, max_chars=max_chars)
    if regular_records:
        reg_ds = Dataset.from_list(regular_records).shuffle(seed=42)
        if max_samples and len(pinned_records) < max_samples:
            budget = max_samples - len(pinned_records)
            if len(reg_ds) > budget:
                reg_ds = reg_ds.select(range(budget))
        elif max_samples and len(pinned_records) >= max_samples:
            reg_ds = reg_ds.select(range(min(50, len(reg_ds))))
        regular_records = reg_ds.to_list()
    combined = pinned_records + regular_records
    return Dataset.from_list(combined).shuffle(seed=42)


def _normalize_turn_list(val, default_role: str = "assistant") -> list[dict]:
    if isinstance(val, list):
        return [_clean_message(m, default_role=default_role) for m in val if isinstance(m, dict)]
    return [_clean_message({"role": default_role, "content": _clean_utf8(val)}, default_role=default_role)]


def load_dpo_dataset(dpo_paths: list[Path], tokenizer, max_chars: int = 36000) -> Dataset:
    records = []
    for dpo_path in dpo_paths:
        if not dpo_path.exists():
            continue
        with open(dpo_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = _normalize_turn_list(row.get("prompt", []), default_role="user")
                chosen = _normalize_turn_list(row.get("chosen", []), default_role="assistant")
                rejected = _normalize_turn_list(row.get("rejected", []), default_role="assistant")
                if not prompt or not chosen or not rejected:
                    continue
                tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
                has_reasoning = any(bool(m.get("reasoning")) for m in (prompt + chosen + rejected))
                prompt_str = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=has_reasoning,
                    preserve_thinking=True,
                )
                chosen_full = tokenizer.apply_chat_template(
                    prompt + chosen,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=has_reasoning,
                    preserve_thinking=True,
                )
                rejected_full = tokenizer.apply_chat_template(
                    prompt + rejected,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=has_reasoning,
                    preserve_thinking=True,
                )
                for tail in ("<|channel>thought\n<channel|>", "<|channel>thought\n"):
                    if prompt_str.endswith(tail) and (
                        not chosen_full.startswith(prompt_str) or not rejected_full.startswith(prompt_str)
                    ):
                        prompt_str = prompt_str[: -len(tail)]
                if not chosen_full.startswith(prompt_str) or not rejected_full.startswith(prompt_str):
                    raise ValueError(f"DPO template prefix mismatch on {row.get('id')}")
                chosen_suffix = chosen_full[len(prompt_str) :]
                rejected_suffix = rejected_full[len(prompt_str) :]
                if len(prompt_str) + max(len(chosen_suffix), len(rejected_suffix)) <= max_chars:
                    records.append(
                        {
                            "prompt": prompt_str,
                            "chosen": chosen_suffix,
                            "rejected": rejected_suffix,
                        }
                    )
    return Dataset.from_list(records).shuffle(seed=42)


def write_ollama_modelfile(output_dir: Path) -> Path:
    modelfile_path = output_dir / "Modelfile"
    content = """# Ollama Modelfile — Sovereign Unified Agentic SRE & Tool-Caller 31B (Wave-6 QLoRA + DPO Fused into Google Gemma 4 31B Dense)
# Base weights preserved as: gemma4:31b
FROM ./gemma4-31b-sre.gguf

TEMPLATE {{ .Prompt }}
RENDERER gemma4
PARSER gemma4
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_ctx 32768
PARAMETER num_gpu 99
PARAMETER temperature 0.2
"""
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def _unwrap_peft_model(model: Any) -> Any:
    if hasattr(model, "peft_config"):
        return model
    inner = getattr(model, "module", None)
    if inner is not None and hasattr(inner, "peft_config"):
        return inner
    raise RuntimeError("Expected PeftModel with peft_config attribute")


def snapshot_peft_default_adapter(model: Any) -> dict[str, Any]:
    """Clone Stage-1 SFT `.default.` LoRA weights to CPU FP32 before DPOTrainer mutates adapters."""
    peft_model = _unwrap_peft_model(model)
    cpu_dtype = torch.float32 if torch is not None else "float32"
    snapshot: dict[str, Any] = {}
    with (torch.no_grad() if torch is not None else contextlib.nullcontext()):
        for name, param in peft_model.named_parameters():
            if ".default." in name:
                snapshot[name] = param.detach().to(device="cpu", dtype=cpu_dtype).clone()
    if not snapshot:
        raise RuntimeError("snapshot_peft_default_adapter found 0 '.default.' LoRA parameters")
    return snapshot


def verify_and_sync_trl_ref_adapter(
    trainer_or_model: Any,
    sft_snapshot: dict[str, Any],
    tmp_ref_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Ensure Stage-2 DPO reference policy (`ref` on trl >= 1.0 or `sft_ref` on legacy trl)
    is provably anchored to the frozen Stage-1 SFT weights in aligned `default` dtype (BF16),
    never falling back to the raw base model (`adapter_name=None`) or mismatched FP32 casts.
    """
    if not sft_snapshot:
        raise RuntimeError("Empty sft_snapshot passed to verify_and_sync_trl_ref_adapter")
    model = getattr(trainer_or_model, "model", trainer_or_model)
    peft_model = _unwrap_peft_model(model)

    trainer_ref = getattr(trainer_or_model, "ref_adapter_name", None) if trainer_or_model is not model else None
    if isinstance(trainer_ref, str) and trainer_ref and trainer_ref in peft_model.peft_config:
        ref_name = trainer_ref
    elif "ref" in peft_model.peft_config:
        ref_name = "ref"
    elif "sft_ref" in peft_model.peft_config:
        ref_name = "sft_ref"
    else:
        ref_name = trainer_ref if isinstance(trainer_ref, str) and trainer_ref else "ref"
        if tmp_ref_dir is not None and hasattr(peft_model, "save_pretrained") and hasattr(peft_model, "load_adapter"):
            peft_model.save_pretrained(str(tmp_ref_dir))
            peft_model.load_adapter(str(tmp_ref_dir), adapter_name=ref_name, is_trainable=False)
        elif hasattr(peft_model, "add_adapter"):
            peft_model.add_adapter(ref_name, copy.deepcopy(peft_model.peft_config["default"]))
        else:
            raise RuntimeError(f"Cannot materialize DPO reference adapter {ref_name!r} on PeftModel")

    if (
        trainer_or_model is not model
        and hasattr(trainer_or_model, "ref_adapter_name")
        and getattr(trainer_or_model, "ref_adapter_name", None) != ref_name
    ):
        setattr(trainer_or_model, "ref_adapter_name", ref_name)

    with (torch.no_grad() if torch is not None else contextlib.nullcontext()):
        params_dict = dict(peft_model.named_parameters())
        synced_tensors = 0
        ref_dtype_str = ""
        for def_name, cpu_sft_tensor in sft_snapshot.items():
            ref_param_name = def_name.replace(".default.", f".{ref_name}.")
            if ref_param_name not in params_dict:
                raise RuntimeError(f"Missing reference adapter parameter {ref_param_name} for {def_name}")
            def_param = params_dict[def_name]
            ref_param = params_dict[ref_param_name]
            aligned_sft = cpu_sft_tensor.to(device=def_param.device, dtype=def_param.dtype)
            if ref_param.dtype != def_param.dtype or ref_param.device != def_param.device:
                ref_param.data = aligned_sft.clone()
            else:
                ref_param.copy_(aligned_sft)
            ref_dtype_str = str(ref_param.dtype)
            synced_tensors += 1

        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")

        params_dict = dict(peft_model.named_parameters())
        max_diff_vs_sft = 0.0
        ref_sq_sum = 0.0
        for def_name, cpu_sft_tensor in sft_snapshot.items():
            ref_param_name = def_name.replace(".default.", f".{ref_name}.")
            def_param = params_dict[def_name]
            ref_param = params_dict[ref_param_name]
            ref_param.requires_grad_(False)
            def_param.requires_grad_(True)
            aligned_sft = cpu_sft_tensor.to(device=def_param.device, dtype=def_param.dtype)
            diff = float((ref_param.detach().float() - aligned_sft.float()).abs().max().item())
            if diff > max_diff_vs_sft:
                max_diff_vs_sft = diff
            ref_sq_sum += float((ref_param.detach().float() ** 2).sum().item())

    if synced_tensors != len(sft_snapshot) or synced_tensors == 0:
        raise RuntimeError(f"Incomplete DPO ref adapter sync: {synced_tensors}/{len(sft_snapshot)}")
    if max_diff_vs_sft > 1e-6:
        raise RuntimeError(f"DPO ref adapter drift vs Stage-1 SFT snapshot: max_diff={max_diff_vs_sft}")

    return {
        "ref_adapter_anchored": True,
        "ref_adapter_name": ref_name,
        "synced_tensors": synced_tensors,
        "ref_dtype": ref_dtype_str,
        "ref_l2_norm": round(math.sqrt(ref_sq_sum), 6),
        "max_diff_vs_sft": max_diff_vs_sft,
    }


def cleanup_peft_ref_adapters(model: Any, output_dir: str | Path) -> list[str]:
    """Delete auxiliary `ref` / `sft_ref` PEFT adapters and stale disk folders before `save_pretrained`."""
    peft_model = _unwrap_peft_model(model)
    if hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter("default")
    cleaned: list[str] = []
    out_path = Path(output_dir)
    for aux_name in ("ref", "sft_ref"):
        if hasattr(peft_model, "peft_config") and aux_name in peft_model.peft_config:
            peft_model.delete_adapter(aux_name)
            cleaned.append(f"adapter:{aux_name}")
        aux_dir = out_path / aux_name
        if aux_dir.exists():
            shutil.rmtree(aux_dir)
            cleaned.append(f"dir:{aux_name}")
    return cleaned


def build_sft_config(output_dir: Path, args: argparse.Namespace) -> SFTConfig:
    sig_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    cfg_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": args.lr,
        "bf16": True,
        "logging_steps": 5,
        "save_strategy": "no",
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if "max_grad_norm" in sig_params:
        cfg_kwargs["max_grad_norm"] = 1.0
    if "dataset_text_field" in sig_params:
        cfg_kwargs["dataset_text_field"] = "text"
    if "dataset_kwargs" in sig_params:
        cfg_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    if "max_length" in sig_params:
        cfg_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sig_params:
        cfg_kwargs["max_seq_length"] = args.max_seq_length
    if "warmup_ratio" in sig_params:
        cfg_kwargs["warmup_ratio"] = 0.05
    return SFTConfig(**cfg_kwargs)


def build_dpo_config(output_dir: Path, args: argparse.Namespace, use_ref_adapter: bool = False) -> DPOConfig:
    sig_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    dpo_epochs = args.dpo_epochs if args.mode == "sft_dpo" else args.epochs
    cfg_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": dpo_epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": args.dpo_lr,
        "beta": args.dpo_beta,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": "none",
    }
    if "max_grad_norm" in sig_params:
        cfg_kwargs["max_grad_norm"] = 1.0
    if "max_length" in sig_params:
        cfg_kwargs["max_length"] = args.max_seq_length
    if "max_prompt_length" in sig_params:
        cfg_kwargs["max_prompt_length"] = min(6144, int(args.max_seq_length * 0.75))
    if "warmup_ratio" in sig_params:
        cfg_kwargs["warmup_ratio"] = 0.05
    if use_ref_adapter and "model_adapter_name" in sig_params and "ref_adapter_name" in sig_params:
        cfg_kwargs["model_adapter_name"] = "default"
        cfg_kwargs["ref_adapter_name"] = "sft_ref"
    return DPOConfig(**cfg_kwargs)


def _summarize_log_history(log_history: list[dict]) -> dict:
    step_logs = [entry for entry in log_history if "loss" in entry]
    summary = {
        "logged_steps": len(step_logs),
        "loss_start": step_logs[0].get("loss") if step_logs else None,
        "loss_end": step_logs[-1].get("loss") if step_logs else None,
        "history": log_history,
    }
    acc_logs = [
        entry.get("mean_token_accuracy")
        for entry in step_logs
        if entry.get("mean_token_accuracy") is not None
    ]
    if acc_logs:
        summary["token_accuracy_start"] = acc_logs[0]
        summary["token_accuracy_end"] = acc_logs[-1]
    margin_logs = [
        entry.get("rewards/margins")
        for entry in step_logs
        if entry.get("rewards/margins") is not None
    ]
    if margin_logs:
        summary["reward_margin_start"] = margin_logs[0]
        summary["reward_margin_end"] = margin_logs[-1]
    dpo_acc_logs = [
        entry.get("rewards/accuracies")
        for entry in step_logs
        if entry.get("rewards/accuracies") is not None
    ]
    if dpo_acc_logs:
        summary["reward_accuracy_start"] = dpo_acc_logs[0]
        summary["reward_accuracy_end"] = dpo_acc_logs[-1]
    return summary


def main() -> None:
    t0 = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_chars = max(36000, args.max_seq_length * 4)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_ds = None
    dpo_ds = None

    if args.mode in ("sft", "sft_dpo"):
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

        data_files = [str(dataset_path)]
        argon_path = Path(args.argon_dataset)
        if argon_path.exists():
            data_files.extend([str(argon_path)] * 2)
            print(f"[QLoRA-Gemma4-31B] Including Argon Golden Remediation dataset (2x oversampled) from {argon_path}", flush=True)

        pinned_files = []
        cal_path = Path(args.calibration_dataset)
        if cal_path.exists():
            oversample = max(1, int(args.calibration_oversample))
            pinned_files.extend([str(cal_path)] * oversample)
            print(f"[QLoRA-Gemma4-31B] Pinning {oversample}x Lego-First + Calibrated CLEAN_PASS split from {cal_path}", flush=True)

        agentic_path = Path(args.agentic_dataset)
        if agentic_path.exists():
            ag_over = max(1, int(args.agentic_oversample))
            pinned_files.extend([str(agentic_path)] * ag_over)
            print(f"[QLoRA-Gemma4-31B] Pinning {ag_over}x Wave-6 Agentic SRE + Workspace Tool-Calling split from {agentic_path}", flush=True)

        print(f"[QLoRA-Gemma4-31B] Rendering Gemma 4 chat template dataset (max_seq_length={args.max_seq_length}, max_chars={max_chars})...", flush=True)
        sft_ds = load_chatml_dataset(
            data_files,
            tokenizer=tokenizer,
            max_chars=max_chars,
            max_samples=args.max_samples,
            pinned_files=pinned_files,
        )
        print(f"[QLoRA-Gemma4-31B] Loaded {len(sft_ds)} unified Teacher + Argon + Lego-First + Tool-Calling trajectories.", flush=True)

    if args.mode in ("dpo", "sft_dpo"):
        dpo_path = Path(args.dpo_dataset)
        if not dpo_path.exists():
            raise FileNotFoundError(f"DPO preference dataset not found at {dpo_path}")
        dpo_paths = [dpo_path]
        ag_dpo_path = Path(args.agentic_dpo_dataset)
        if ag_dpo_path.exists():
            ag_dpo_over = max(1, int(args.agentic_dpo_oversample))
            dpo_paths.extend([ag_dpo_path] * ag_dpo_over)
            print(f"[DPO-Gemma4-31B] Including {ag_dpo_over}x Wave-6 Tool-Action + Exhaustive-Arg preference pairs from {ag_dpo_path}", flush=True)
        print("[DPO-Gemma4-31B] Rendering Referee + Tool-Action preference pairs...", flush=True)
        dpo_ds = load_dpo_dataset(dpo_paths, tokenizer=tokenizer, max_chars=max_chars)
        print(f"[DPO-Gemma4-31B] Loaded {len(dpo_ds)} unified Referee + Tool-Action preference pairs.", flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[QLoRA-Gemma4-31B] Loading base model {args.base_model} in 4-bit NF4 + BF16 on 96GB Blackwell G4...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    resumed_from = None
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        print(f"[QLoRA-Gemma4-31B] Resuming trainable PEFT adapter from {adapter_dir}...", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        resumed_from = str(adapter_dir)
    else:
        # Target strictly the 60 language_model attention projections (excluding vision_tower)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    metrics: dict = {
        "base_model": args.base_model,
        "mode": args.mode,
        "resumed_from": resumed_from,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "frameworks": {
            "torch": torch.__version__,
            "transformers": hf_transformers.__version__,
            "peft": hf_peft.__version__,
            "trl": hf_trl.__version__,
            "datasets": hf_datasets.__version__,
        },
    }

    if args.mode in ("sft", "sft_dpo"):
        sft_config = build_sft_config(output_dir, args)
        assistant_collator = Gemma4AssistantOnlyCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=sft_ds,
            args=sft_config,
            data_collator=assistant_collator,
        )
        print("[QLoRA-Gemma4-31B] Starting Stage-1 SFT training (Assistant-Turn-Only Loss Masking enabled)...", flush=True)
        t_sft = time.monotonic()
        sft_res = sft_trainer.train()
        sft_wall = round(time.monotonic() - t_sft, 2)
        metrics["sft"] = {
            "samples": len(sft_ds),
            "epochs": args.epochs,
            "lr": args.lr,
            "wall_seconds": sft_wall,
            "train_loss": getattr(sft_res, "training_loss", None),
            **_summarize_log_history(list(sft_trainer.state.log_history)),
        }
        model = sft_trainer.model
        del sft_trainer
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode in ("dpo", "sft_dpo"):
        dpo_sig_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
        legacy_ref_cfg = "model_adapter_name" in dpo_sig_params and "ref_adapter_name" in dpo_sig_params
        sft_snapshot = snapshot_peft_default_adapter(model)
        with tempfile.TemporaryDirectory() as tmp_ref_dir:
            if legacy_ref_cfg:
                model.save_pretrained(tmp_ref_dir)
                model.load_adapter(tmp_ref_dir, adapter_name="sft_ref", is_trainable=False)
                model.set_adapter("default")

            dpo_config = build_dpo_config(output_dir, args, use_ref_adapter=legacy_ref_cfg)
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                processing_class=tokenizer,
                train_dataset=dpo_ds,
                args=dpo_config,
            )
            ref_anchor_meta = verify_and_sync_trl_ref_adapter(
                dpo_trainer,
                sft_snapshot,
                tmp_ref_dir=tmp_ref_dir,
            )
            del sft_snapshot
            print(
                f"[DPO-Gemma4-31B] Anchored Stage-2 DPO reference policy: {json.dumps(ref_anchor_meta)}",
                flush=True,
            )
            print("[DPO-Gemma4-31B] Starting Stage-2 31B Referee + Exhaustive-Arg Preference Calibration (DPO)...", flush=True)
            t_dpo = time.monotonic()
            dpo_res = dpo_trainer.train()
            dpo_wall = round(time.monotonic() - t_dpo, 2)
            metrics["dpo"] = {
                "pairs": len(dpo_ds),
                "epochs": args.dpo_epochs if args.mode == "sft_dpo" else args.epochs,
                "lr": args.dpo_lr,
                "beta": args.dpo_beta,
                **ref_anchor_meta,
                "wall_seconds": dpo_wall,
                "train_loss": getattr(dpo_res, "training_loss", None),
                **_summarize_log_history(list(dpo_trainer.state.log_history)),
            }
            model = dpo_trainer.model
            del dpo_trainer
            cleaned_refs = cleanup_peft_ref_adapters(model, output_dir)
            metrics["dpo"]["cleaned_ref_artifacts"] = cleaned_refs
            gc.collect()
            torch.cuda.empty_cache()

    metrics["total_wall_seconds"] = round(time.monotonic() - t0, 2)
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    cleanup_peft_ref_adapters(model, output_dir)
    print(f"[QLoRA-Gemma4-31B] Saving LoRA adapter & tokenizer to {output_dir}...", flush=True)
    readme_file = output_dir / "README.md"
    prior_readme = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if prior_readme and "[More Information Needed]" not in prior_readme:
        readme_file.write_text(prior_readme, encoding="utf-8")
    modelfile = write_ollama_modelfile(output_dir)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[QLoRA-Gemma4-31B] Done! Wrote Ollama Modelfile to {modelfile} and metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
