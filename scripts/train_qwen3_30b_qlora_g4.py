#!/usr/bin/env python3
"""
QLoRA SFT Fine-Tuning Pipeline: gpt-oss:120b (Teacher) -> Qwen3-Coder-30B-A3B (Student)
Target Accelerator: Colab G4 96GB RTX PRO 6000 Blackwell (or 80GB A100)

Usage on Colab G4:
  pip install -U torch transformers datasets peft trl bitsandbytes accelerate
  python3 scripts/train_qwen3_30b_qlora_g4.py \
      --dataset data/distillation/gpt_oss_120b_sft.jsonl \
      --output-dir /mnt/data2/models/jean-qwen3-30b-sre-lora \
      --epochs 3

After training, register in Ollama:
  ollama create qwen3-coder-30b-sre:latest -f /mnt/data2/models/jean-qwen3-30b-sre-lora/Modelfile
"""

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT + DPO fine-tune Qwen3-Coder-30B-A3B on distilled 120B, Argon, Lego-First & Calibrated Referee trajectories."
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo"],
        default="sft",
        help="Training stage: 'sft' (supervised fine-tuning) or 'dpo' (post-SFT Referee preference calibration).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model ID for Qwen3-Coder-30B-MoE (A3B).",
    )
    parser.add_argument(
        "--dataset",
        default="data/distillation/gpt_oss_120b_elitist_teacher.jsonl",
        help="Path to curated ChatML JSONL dataset captured from gpt-oss:120b.",
    )
    parser.add_argument(
        "--argon-dataset",
        default="data/argon_remediation_sft_corpus.jsonl",
        help="Optional path to Gemini 4 Argon verified Golden Remediation JSONL dataset (oversampled 3x).",
    )
    parser.add_argument(
        "--calibration-dataset",
        default="data/distillation/lego_and_referee_calibration_sft.jsonl",
        help="Pinned CLEAN_PASS + P0/P1 Remediation + Lego-First ChatML SFT split (3x oversampled, never dropped).",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="data/distillation/referee_calibration_dpo.jsonl",
        help="Balanced Precision + Recall DPO preference JSONL dataset for --mode dpo.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional existing PEFT LoRA checkpoint directory to resume/calibrate in --mode dpo.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/jean-qwen3-30b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max token context during QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=1200, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for SFT.")
    parser.add_argument("--dpo-lr", type=float, default=1e-5, help="Learning rate for --mode dpo calibration.")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="KL penalty beta for DPOTrainer.")
    return parser.parse_args()


def _clean_utf8(text: str) -> str:
    return str(text or "").encode("utf-8", "replace").decode("utf-8")


def _read_chatml_records(file_paths: list[str], max_chars: int) -> list[dict]:
    records = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages")
                if isinstance(msgs, list) and len(msgs) >= 2:
                    cleaned_msgs = [
                        {
                            "role": _clean_utf8(m.get("role", "user")),
                            "content": _clean_utf8(m.get("content", "")),
                        }
                        for m in msgs
                    ]
                    total_chars = sum(len(m["content"]) for m in cleaned_msgs)
                    if total_chars <= max_chars:
                        records.append({"messages": cleaned_msgs})
    return records


def load_chatml_dataset(
    data_files: list[str],
    max_chars: int = 36000,
    max_samples: int = 1200,
    pinned_files: list[str] | None = None,
) -> Dataset:
    pinned_records = _read_chatml_records(pinned_files or [], max_chars=max_chars)
    regular_records = _read_chatml_records(data_files, max_chars=max_chars)
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
        return [
            {
                "role": _clean_utf8(m.get("role", default_role)),
                "content": _clean_utf8(m.get("content", "")),
            }
            for m in val
            if isinstance(m, dict)
        ]
    return [{"role": default_role, "content": _clean_utf8(val)}]


def load_dpo_dataset(dpo_path: Path, max_chars: int = 36000) -> Dataset:
    records = []
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
            total_chars = (
                sum(len(m["content"]) for m in prompt)
                + sum(len(m["content"]) for m in chosen)
                + sum(len(m["content"]) for m in rejected)
            )
            if total_chars <= max_chars:
                records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(records).shuffle(seed=42)


def write_ollama_modelfile(output_dir: Path) -> Path:
    modelfile_path = output_dir / "Modelfile"
    content = """# Ollama Modelfile — Sovereign Code Auditor & Remediator 30B (Teacher: gpt-oss:120b + Gemini 4 Argon + DPO)
FROM qwen3-coder-30b-moe:latest
ADAPTER .

PARAMETER num_ctx 16384
PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM \"\"\"You are an Autonomous Principal Software Engineer, SRE, and Code Auditor.
RULES (NON-NEGOTIABLE):
1. Ground every finding strictly in the provided live source snippets, AST symbol graphs, and command outputs.
2. NEVER invent, paraphrase, or reconstruct code snippets from memory. Only quote verbatim lines that appear in the prompt with their exact line numbers.
3. If a prior Pending Issue is not supported by the Live Source Verification lines shown in the prompt, REFUTE and remove it.
4. When audited files satisfy all frozen specification invariants with zero P0/P1 defects, emit a calibrated CLEAN_PASS verdict ("findings": []) rather than inventing stylistic nitpicks.
5. Assemble application services from @krusch/toolkit and krusch_toolkit Lego bricks (INV-LEGO-01) instead of hand-rolling HTTP, SSE, MCP, DB, Webhook, or CLI boilerplate.
6. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.\"\"\"
"""
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


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
        "logging_steps": 10,
        "save_strategy": "epoch",
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": "none",
    }
    if "max_length" in sig_params:
        cfg_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sig_params:
        cfg_kwargs["max_seq_length"] = args.max_seq_length
    if "warmup_ratio" in sig_params:
        cfg_kwargs["warmup_ratio"] = 0.05
    return SFTConfig(**cfg_kwargs)


def build_dpo_config(output_dir: Path, args: argparse.Namespace) -> DPOConfig:
    sig_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    cfg_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": args.dpo_lr,
        "beta": args.dpo_beta,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "epoch",
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
    return DPOConfig(**cfg_kwargs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_chars = max(36000, args.max_seq_length * 4)

    if args.mode == "dpo":
        dpo_path = Path(args.dpo_dataset)
        if not dpo_path.exists():
            raise FileNotFoundError(f"DPO preference dataset not found at {dpo_path}")
        print(f"[DPO-30B] Loading Referee preference pairs from {dpo_path}...")
        ds = load_dpo_dataset(dpo_path, max_chars=max_chars)
        print(f"[DPO-30B] Loaded {len(ds)} balanced Precision + Recall preference pairs.")
    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

        data_files = [str(dataset_path)]
        argon_path = Path(args.argon_dataset)
        if argon_path.exists():
            # Oversample Argon Golden Remediations 3x so verified code patches carry strong weight
            data_files.extend([str(argon_path)] * 3)
            print(f"[QLoRA-30B] Including Argon Golden Remediation dataset (3x oversampled) from {argon_path}")

        pinned_files = []
        cal_path = Path(args.calibration_dataset)
        if cal_path.exists():
            pinned_files = [str(cal_path)] * 3
            print(f"[QLoRA-30B] Pinning 3x Lego-First + Calibrated CLEAN_PASS split from {cal_path}")

        print(f"[QLoRA-30B] Loading dataset files (max_seq_length={args.max_seq_length}, max_chars={max_chars})...")
        ds = load_chatml_dataset(
            data_files,
            max_chars=max_chars,
            max_samples=args.max_samples,
            pinned_files=pinned_files,
        )
        print(f"[QLoRA-30B] Loaded {len(ds)} complete combined Teacher + Argon + Lego-First trajectories.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[QLoRA-30B] Loading base model {args.base_model} in 4-bit NF4 + BF16 MoE experts...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # NOTE: Do NOT call prepare_model_for_kbit_training(model) here because in Transformers >= 5.0,
    # Qwen3-MoE stores 128 routed experts per layer as 3D BF16 Parameter tensors (~56 GB), and
    # prepare_model_for_kbit_training upcasts all non-4bit BF16 parameters to FP32 (~112 GB -> OOM).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        print(f"[QLoRA-30B] Resuming trainable PEFT adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    else:
        # Target attention projections (q/k/v/o_proj) so Qwen3-MoE fused 3D expert tensors stay intact in BF16
        # and the resulting adapter is 100% compatible with Ollama GGUF ADAPTER + PEFT/vLLM
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
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    if args.mode == "dpo":
        dpo_config = build_dpo_config(output_dir, args)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            processing_class=tokenizer,
            train_dataset=ds,
            args=dpo_config,
        )
        print("[DPO-30B] Starting 30B Referee Preference Calibration (DPO)...")
    else:
        sft_config = build_sft_config(output_dir, args)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=sft_config,
        )
        print("[QLoRA-30B] Starting SFT training...")

    trainer.train()

    print(f"[QLoRA-30B] Saving LoRA adapter & tokenizer to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    print(f"[QLoRA-30B] Done! Wrote Ollama Modelfile to {modelfile}")


if __name__ == "__main__":
    main()
