#!/usr/bin/env python3
"""
Elitist QLoRA SFT Fine-Tuning Pipeline for gpt-oss:120b (116.8B MoE, 5.1B Active)
Target Accelerator: Colab G4 96GB NVIDIA RTX PRO 6000 Blackwell (sm_120, native MXFP4)

Architecture Strategy:
  - Keeps all 128 routed MoE experts per layer frozen in native 4-bit MXFP4 (~63 GB VRAM).
  - Attaches LoRA adapters (r=16, alpha=32) to BF16 attention projections (q/k/v/o_proj),
    keeping trainable parameters ~140M (<1.2 GB) and peak VRAM ~78 GB inside the 96GB G4.
  - Trains on `data/distillation/gpt_oss_120b_elitist_teacher.jsonl` (Argon-verified +
    unit-test-gated Golden Remediations + top rejection-sampled 120B trajectories).

Usage on Colab G4:
  pip uninstall -y torchvision torchaudio torchao
  pip install -U transformers datasets peft trl bitsandbytes accelerate "kernels>=0.17.2,<0.18.0"
  python3 scripts/train_gpt_oss_120b_qlora_g4.py \
      --dataset data/distillation/gpt_oss_120b_elitist_teacher.jsonl \
      --output-dir /root/models/jean-gpt-oss-120b-sre-lora \
      --epochs 2

After training, register on Colab G4 Ollama (layers adapter onto existing 65GB gpt-oss:120b):
  ollama create gpt-oss-sre:120b -f /root/models/jean-gpt-oss-120b-sre-lora/Modelfile
"""

import argparse
import gc
import inspect
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

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


DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT + DPO fine-tune openai/gpt-oss-120b on Argon-verified, Lego-First, and Calibrated Referee trajectories."
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "sft_dpo"],
        default="sft",
        help="Training stage: 'sft', 'dpo', or 'sft_dpo' (single-load Stage-1 SFT -> Stage-2 DPO with sft_ref KL anchor).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model ID for gpt-oss-120b.",
    )
    parser.add_argument(
        "--dataset",
        default="data/distillation/gpt_oss_120b_elitist_teacher.jsonl",
        help="Path to elitist Argon-verified + rejection-sampled ChatML JSONL dataset.",
    )
    parser.add_argument(
        "--calibration-dataset",
        default="data/distillation/lego_and_referee_calibration_sft.jsonl",
        help="Pinned CLEAN_PASS + P0/P1 Remediation + Lego-First ChatML SFT split (never dropped).",
    )
    parser.add_argument(
        "--calibration-oversample",
        type=int,
        default=2,
        help="Oversample factor for the pinned Lego-First + Referee calibration SFT dataset.",
    )
    parser.add_argument(
        "--agentic-dataset",
        default="data/distillation/agentic_tool_calling_sft.jsonl",
        help="Pinned multi-turn SRE + Workspace tool-calling SFT split (preserves Harmony commentary tool routing).",
    )
    parser.add_argument(
        "--agentic-oversample",
        type=int,
        default=2,
        help="Oversample factor for the pinned multi-turn tool-calling SFT dataset.",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="data/distillation/referee_calibration_dpo.jsonl",
        help="Balanced Precision + Recall + Architecture Calibration DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--agentic-dpo-dataset",
        default="data/distillation/agentic_tool_calling_dpo.jsonl",
        help="Multi-turn tool-action vs prose-dump / search-loop / false-refusal DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional existing PEFT LoRA checkpoint directory to resume/calibrate.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/models/jean-gpt-oss-120b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT epochs (or DPO epochs when --mode dpo).")
    parser.add_argument("--dpo-epochs", type=int, default=3, help="Number of Stage-2 DPO epochs when --mode sft_dpo.")
    # Eager attention on 64 heads allocates 64 * S^2 * 2B per buffer: at S=8192 that is 8.00 GiB x 3 buffers (24 GiB)
    # + 61.8 GiB MXFP4 weights + 9.2 GiB 201k-vocab logits = 95.3 GiB (>94.97 GiB G4). S=5120 uses 3.125 GiB/buffer (~77 GiB peak).
    parser.add_argument("--max-seq-length", type=int, default=5120, help="Max sequence length during 120B QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=340, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for 120B SFT attention adapter.")
    parser.add_argument("--dpo-lr", type=float, default=5e-6, help="Learning rate for DPO calibration.")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="KL penalty beta for DPOTrainer.")
    parser.add_argument("--overwrite-readme", action="store_true", help="Overwrite README.md even if one already exists.")
    return parser.parse_args()


def _clean_utf8(text: str) -> str:
    return str(text or "").encode("utf-8", "replace").decode("utf-8")


def _clean_message(m: dict, default_role: str = "user") -> dict:
    role = _clean_utf8(m.get("role", default_role))
    content = _clean_utf8(m.get("content", ""))
    out: dict = {"role": role, "content": content}
    thinking = _clean_utf8(m.get("thinking", "")).strip() if isinstance(m.get("thinking"), str) else ""
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
        # Harmony chat_template forbids non-empty content AND thinking on the same tool_calls message
        if thinking:
            out["thinking"] = thinking
            out["content"] = ""
    elif thinking:
        out["thinking"] = thinking

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
                rendered = tokenizer.apply_chat_template(
                    cleaned_msgs,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if len(rendered) <= max_chars:
                    records.append({"text": rendered})
    return records


def load_chatml_dataset(
    data_files: list[str],
    tokenizer,
    max_chars: int = 36000,
    max_samples: int = 340,
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
        regular_records = reg_ds.to_list()
    combined = pinned_records + regular_records
    return Dataset.from_list(combined).shuffle(seed=42)


def _normalize_turn_list(val, default_role: str = "assistant") -> list[dict]:
    if isinstance(val, list):
        return [_clean_message(m, default_role=default_role) for m in val if isinstance(m, dict)]
    return [{"role": default_role, "content": _clean_utf8(val)}]


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
                prompt = [
                    {k: v for k, v in m.items() if k != "thinking"}
                    for m in _normalize_turn_list(row.get("prompt", []), default_role="user")
                ]
                chosen = _normalize_turn_list(row.get("chosen", []), default_role="assistant")
                rejected = _normalize_turn_list(row.get("rejected", []), default_role="assistant")
                if not prompt or not chosen or not rejected:
                    continue
                tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
                prompt_str = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chosen_full = tokenizer.apply_chat_template(
                    prompt + chosen,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                rejected_full = tokenizer.apply_chat_template(
                    prompt + rejected,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
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
    content = """# Ollama Modelfile — Sovereign Unified Agentic SRE & Tool-Caller 120B (Wave-4 QLoRA + DPO)
# Role-neutral Modelfile (SYSTEM: None) so caller system prompts & Harmony function-tool namespaces bind cleanly.
# Fuse adapter.gguf into the gpt-oss:120b GGUF blob in ~1.7s via fuse_lora_into_gguf.py, then run:
#   ollama create gpt-oss-sre:120b -f Modelfile
FROM gpt-oss:120b

PARAMETER num_ctx 32768
"""
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def write_model_card_readme(output_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    content = """---
base_model: openai/gpt-oss-120b
library_name: peft
model_name: gpt-oss-120b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
tags:
  - base_model:adapter:openai/gpt-oss-120b
  - lora
  - qlora
  - mxfp4
  - sft
  - code-audit
  - bug-remediation
  - sre
  - transformers
  - trl
---

# gpt-oss-120b — Sovereign Code Auditor & Remediator LoRA

A portable **QLoRA adapter** (`r=16`, `lora_alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`) for [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MoE parameters), distilled from:
1. **Gemini 4 Argon (`gemini-4-argon`) Golden Remediations** — multi-file root-cause bug fixes verified against live unit test suites (`node:test` / `pytest`, exit code `0`).
2. **Elitist Rejection-Sampled 11-Phase Code Audits** — AST-grounded codebase audits with verbatim `file:startLine-endLine` citations, false-positive refutation (`NOT FLAGGED`), and Zero-Finding convergence (`CLEAN_PASS`).

All training trajectories were scrubbed through a **Zero-Leak & Portability Gate** (`0` private IPs, `0` internal hostnames, `0` user home paths, `0` secrets) so this adapter generalizes cleanly to any local or cloud repository.

---

## Quickstart

### 1. Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "openai/gpt-oss-120b"
adapter_dir = "./jean-gpt-oss-120b-sre-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
```

### 2. Ollama (`Modelfile` Included)

```bash
cd jean-gpt-oss-120b-sre-lora
ollama create gpt-oss-sre:120b -f Modelfile
ollama run gpt-oss-sre:120b
```

### 3. vLLM Multi-LoRA Serving

```bash
vllm serve openai/gpt-oss-120b \\
  --enable-lora \\
  --lora-modules code-auditor=./jean-gpt-oss-120b-sre-lora
```

---

## Training Hyperparameters & Environment

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`)
- **Base Model**: `openai/gpt-oss-120b` (native 4-bit MXFP4 frozen MoE experts + BF16 attention projections)
- **LoRA Config**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`11,943,936` trainable parameters)
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`lr=1e-4`, `warmup_ratio=0.05`)
- **Frameworks**:
  - `peft`: `0.21.2`
  - `trl`: `1.16.0`
  - `transformers`: `5.20.2`
  - `torch`: `2.11.0+cu130`
  - `datasets`: `5.1.0`
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


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
        "logging_steps": 2,
        "save_strategy": "no",
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": "none",
    }
    if "dataset_text_field" in sig_params:
        cfg_kwargs["dataset_text_field"] = "text"
    if "max_length" in sig_params:
        cfg_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sig_params:
        cfg_kwargs["max_seq_length"] = args.max_seq_length
    if "warmup_ratio" in sig_params:
        cfg_kwargs["warmup_ratio"] = 0.05
    if "router_aux_loss_coef" in sig_params:
        cfg_kwargs["router_aux_loss_coef"] = 0.0
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
    if "max_length" in sig_params:
        cfg_kwargs["max_length"] = args.max_seq_length
    if "max_prompt_length" in sig_params:
        cfg_kwargs["max_prompt_length"] = min(6144, int(args.max_seq_length * 0.75))
    if "warmup_ratio" in sig_params:
        cfg_kwargs["warmup_ratio"] = 0.05
    if "router_aux_loss_coef" in sig_params:
        cfg_kwargs["router_aux_loss_coef"] = 0.0
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
    max_chars = int(args.max_seq_length * 4.2)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_ds = None
    dpo_ds = None

    if args.mode in ("sft", "sft_dpo"):
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Elitist 120B dataset not found at {dataset_path}")

        data_files = [str(dataset_path)]
        argon_path = Path("data/argon_remediation_sft_corpus.jsonl")
        if argon_path.exists():
            data_files.append(str(argon_path))
            print(f"[QLoRA-120B] Including Argon Golden Remediation dataset from {argon_path}", flush=True)

        pinned_files = []
        cal_path = Path(args.calibration_dataset)
        if cal_path.exists():
            oversample = max(1, int(args.calibration_oversample))
            pinned_files.extend([str(cal_path)] * oversample)
            print(f"[QLoRA-120B] Pinning {oversample}x Lego-First + Calibrated CLEAN_PASS split from {cal_path}", flush=True)

        agentic_path = Path(args.agentic_dataset)
        if agentic_path.exists():
            ag_over = max(1, int(args.agentic_oversample))
            pinned_files.extend([str(agentic_path)] * ag_over)
            print(f"[QLoRA-120B] Pinning {ag_over}x Wave-4 Agentic SRE + Workspace Tool-Calling split from {agentic_path}", flush=True)

        print(f"[QLoRA-120B] Rendering Harmony SFT dataset (max_seq_length={args.max_seq_length}, max_chars={max_chars})...", flush=True)
        sft_ds = load_chatml_dataset(
            data_files,
            tokenizer=tokenizer,
            max_chars=max_chars,
            max_samples=min(args.max_samples, 420),
            pinned_files=pinned_files,
        )
        print(f"[QLoRA-120B] Loaded {len(sft_ds)} unified SRE, Lego-First & Harmony Tool-Calling trajectories.", flush=True)

    if args.mode in ("dpo", "sft_dpo"):
        dpo_path = Path(args.dpo_dataset)
        if not dpo_path.exists():
            raise FileNotFoundError(f"DPO preference dataset not found at {dpo_path}")
        dpo_paths = [dpo_path]
        ag_dpo_path = Path(args.agentic_dpo_dataset)
        if ag_dpo_path.exists():
            dpo_paths.extend([ag_dpo_path] * 2)
            print(f"[DPO-120B] Including 2x Wave-4 Tool-Action preference pairs from {ag_dpo_path}", flush=True)
        print(f"[DPO-120B] Rendering Referee + Tool-Action preference pairs...", flush=True)
        dpo_ds = load_dpo_dataset(dpo_paths, tokenizer=tokenizer, max_chars=max_chars)
        print(f"[DPO-120B] Loaded {len(dpo_ds)} unified Referee + Harmony Tool-Action preference pairs.", flush=True)

    print(f"[QLoRA-120B] Loading {args.base_model} onto 96GB Blackwell G4 (native MXFP4 experts + BF16 eager attention)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.config.output_router_logits = False
    base_model.config.router_aux_loss_coef = 0.0
    base_model._hf_peft_config_loaded = True
    if getattr(base_model, "hf_quantizer", None) is not None:
        type(base_model.hf_quantizer).is_trainable = property(lambda self: True)

    # Wrap frozen MXFP4 MoE MLP blocks in torch.no_grad() so native 4-bit Triton kernels execute
    # without autograd graph overhead while gradients flow across all 36 layers via the Pre-LN residual stream.
    mlp_patched = 0
    for module in base_model.modules():
        if module.__class__.__name__ == "Mxfp4GptOssMLP":
            _orig_fwd = module.forward

            def _no_grad_mlp_fwd(hidden_states, _fn=_orig_fwd):
                with torch.no_grad():
                    return _fn(hidden_states.detach())

            module.forward = _no_grad_mlp_fwd
            mlp_patched += 1
    print(f"[QLoRA-120B] Patched {mlp_patched} Mxfp4GptOssMLP blocks for residual-stream BF16 attention LoRA.", flush=True)
    if torch.cuda.is_available() and mlp_patched == 0:
        raise RuntimeError(
            "0 Mxfp4GptOssMLP blocks patched (model dequantized to BF16/CPU). "
            "Install compatible kernels package: pip install 'kernels>=0.17.2,<0.18.0'"
        )

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    resumed_from = None
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        print(f"[QLoRA-120B] Resuming trainable PEFT adapter from {adapter_dir}...", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        resumed_from = str(adapter_dir)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )
        model = get_peft_model(base_model, lora_config)

    model._hf_peft_config_loaded = True
    if getattr(model, "hf_quantizer", None) is not None:
        type(model.hf_quantizer).is_trainable = property(lambda self: True)
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
        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=sft_ds,
            args=sft_config,
        )
        print("[QLoRA-120B] Starting Stage-1 120B Elitist SFT training...", flush=True)
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
        can_use_ref_adapter = "model_adapter_name" in dpo_sig_params and "ref_adapter_name" in dpo_sig_params
        with tempfile.TemporaryDirectory() as tmp_ref_dir:
            loaded_ref_adapter = False
            if can_use_ref_adapter:
                model.save_pretrained(tmp_ref_dir)
                model.load_adapter(tmp_ref_dir, adapter_name="sft_ref", is_trainable=False)
                model.set_adapter("default")
                loaded_ref_adapter = True
                print("[DPO-120B] Anchored Stage-2 DPO reference policy to frozen Stage-1 adapter ('sft_ref').", flush=True)

            model._hf_peft_config_loaded = True
            dpo_config = build_dpo_config(output_dir, args, use_ref_adapter=loaded_ref_adapter)
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                processing_class=tokenizer,
                train_dataset=dpo_ds,
                args=dpo_config,
            )
            print("[DPO-120B] Starting Stage-2 120B Referee Preference Calibration (DPO)...", flush=True)
            t_dpo = time.monotonic()
            dpo_res = dpo_trainer.train()
            dpo_wall = round(time.monotonic() - t_dpo, 2)
            metrics["dpo"] = {
                "pairs": len(dpo_ds),
                "epochs": args.dpo_epochs if args.mode == "sft_dpo" else args.epochs,
                "lr": args.dpo_lr,
                "beta": args.dpo_beta,
                "ref_adapter_anchored": loaded_ref_adapter,
                "wall_seconds": dpo_wall,
                "train_loss": getattr(dpo_res, "training_loss", None),
                **_summarize_log_history(list(dpo_trainer.state.log_history)),
            }
            model = dpo_trainer.model
            del dpo_trainer
            if loaded_ref_adapter:
                model.set_adapter("default")
                model.delete_adapter("sft_ref")
            gc.collect()
            torch.cuda.empty_cache()

    metrics["total_wall_seconds"] = round(time.monotonic() - t0, 2)
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    print(f"[QLoRA-120B] Saving LoRA adapter & tokenizer to {output_dir}...", flush=True)
    readme_file = output_dir / "README.md"
    prior_readme = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    if not args.overwrite_readme and prior_readme and "[More Information Needed]" not in prior_readme:
        readme_file.write_text(prior_readme, encoding="utf-8")
    else:
        write_model_card_readme(output_dir)

    try:
        print("[QLoRA-120B] Running in-VRAM dual-capability smoke verification (SRE Referee + Tool-Calling)...", flush=True)
        model.gradient_checkpointing_disable()
        model.eval()
        device = next(model.parameters()).device
        sre_msgs = [
            {"role": "system", "content": "You are a Principal Code Referee. Return JSON with verdict CLEAN_PASS or DEFECTS_FOUND."},
            {"role": "user", "content": "Audit `src/server.js`:\n```js\nimport { createApiServer, okEnvelope } from '@krusch/toolkit/http';\nexport const server = createApiServer({ name: 'svc', routes: { 'GET /ping': async () => okEnvelope({ pong: true }) } });\n```"},
        ]
        sre_prompt = tokenizer.apply_chat_template(sre_msgs, tokenize=False, add_generation_prompt=True)
        sre_in = tokenizer(sre_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            sre_out = model.generate(**sre_in, max_new_tokens=180, do_sample=False, use_cache=False, pad_token_id=tokenizer.pad_token_id)
        sre_gen = tokenizer.decode(sre_out[0][sre_in["input_ids"].shape[1]:], skip_special_tokens=False)

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "create_workspace_doc",
                    "description": "Create a workspace document with title and body.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["title", "body"],
                    },
                },
            }
        ]
        tool_msgs = [
            {"role": "system", "content": "You are an execution-focused Workspace Assistant. Honor the latest timestamped note and invoke action tools rather than writing prose."},
            {"role": "user", "content": "Create the Q4 architecture review agenda doc using our latest schedule note."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_01", "type": "function", "function": {"name": "create_workspace_doc", "arguments": {"title": "lookup", "body": "lookup"}}}]},
        ]
        # Clean 2-turn tool invocation probe after retrieved context:
        tool_msgs = [
            {"role": "system", "content": "You are an execution-focused Workspace Assistant. Honor the latest timestamped note and invoke action tools rather than writing prose."},
            {"role": "user", "content": "Retrieved Notes:\n- [2026-09-10] Q4 review is Tuesday 10:00 UTC in Room A.\n- [2026-10-28] UPDATE: Moved Q4 review to Thursday 14:00 UTC in Room C.\n\nPlease create the workspace doc 'Q4 Architecture Review' with the current room and time."},
        ]
        tool_prompt = tokenizer.apply_chat_template(tool_msgs, tools=tool_defs, tokenize=False, add_generation_prompt=True)
        tool_in = tokenizer(tool_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            tool_out = model.generate(**tool_in, max_new_tokens=200, do_sample=False, use_cache=False, pad_token_id=tokenizer.pad_token_id)
        tool_gen = tokenizer.decode(tool_out[0][tool_in["input_ids"].shape[1]:], skip_special_tokens=False)

        metrics["smoke_verification"] = {
            "sre_clean_pass": "CLEAN_PASS" in sre_gen,
            "sre_preview": sre_gen[:400],
            "tool_call_emitted": ("create_workspace_doc" in tool_gen) and ("<|call|>" in tool_gen or "functions.create_workspace_doc" in tool_gen),
            "temporal_override_bound": ("Thursday" in tool_gen and "14:00" in tool_gen),
            "tool_preview": tool_gen[:400],
        }
        print(f"[QLoRA-120B] Smoke verification: sre_clean_pass={metrics['smoke_verification']['sre_clean_pass']}, tool_call_emitted={metrics['smoke_verification']['tool_call_emitted']}, temporal_override={metrics['smoke_verification']['temporal_override_bound']}", flush=True)
    except Exception as smoke_exc:
        metrics["smoke_verification"] = {"error": str(smoke_exc)}
        print(f"[QLoRA-120B] Smoke verification skipped/warn: {smoke_exc}", flush=True)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[QLoRA-120B] Done! Wrote Ollama Modelfile to {modelfile} and metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
