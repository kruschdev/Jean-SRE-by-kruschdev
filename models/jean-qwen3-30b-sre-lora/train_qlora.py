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
import gc
import inspect
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT + DPO fine-tune Qwen3-Coder-30B-A3B on distilled 120B, Argon, Lego-First & Calibrated Referee trajectories."
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
        help="Pinned CLEAN_PASS + P0/P1 Remediation + Lego-First ChatML SFT split (never dropped).",
    )
    parser.add_argument(
        "--calibration-oversample",
        type=int,
        default=4,
        help="Oversample factor for the pinned Lego-First + Referee calibration SFT dataset.",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="data/distillation/referee_calibration_dpo.jsonl",
        help="Balanced Precision + Recall + Architecture Calibration DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional existing PEFT LoRA checkpoint directory to resume/calibrate.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/jean-qwen3-30b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT epochs (or DPO epochs when --mode dpo).")
    parser.add_argument("--dpo-epochs", type=int, default=3, help="Number of Stage-2 DPO epochs when --mode sft_dpo.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max token context during QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=1200, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for SFT.")
    parser.add_argument("--dpo-lr", type=float, default=1e-5, help="Learning rate for DPO calibration.")
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
    content = """# Ollama Modelfile — Sovereign Code Auditor & Remediator 30B (QLoRA + DPO Fused into Qwen3-Coder-30B-A3B)
# Base weights preserved as: qwen3-coder-30b-moe:base
FROM ./qwen3-coder-30b-sre.gguf

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\"

PARAMETER num_ctx 16384
PARAMETER num_gpu 99
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>user
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
        "logging_steps": 5,
        "save_strategy": "no",
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

    sft_ds = None
    dpo_ds = None

    if args.mode in ("sft", "sft_dpo"):
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

        data_files = [str(dataset_path)]
        argon_path = Path(args.argon_dataset)
        if argon_path.exists():
            data_files.extend([str(argon_path)] * 3)
            print(f"[QLoRA-30B] Including Argon Golden Remediation dataset (3x oversampled) from {argon_path}", flush=True)

        pinned_files = []
        cal_path = Path(args.calibration_dataset)
        if cal_path.exists():
            oversample = max(1, int(args.calibration_oversample))
            pinned_files = [str(cal_path)] * oversample
            print(f"[QLoRA-30B] Pinning {oversample}x Lego-First + Calibrated CLEAN_PASS split from {cal_path}", flush=True)

        print(f"[QLoRA-30B] Loading dataset files (max_seq_length={args.max_seq_length}, max_chars={max_chars})...", flush=True)
        sft_ds = load_chatml_dataset(
            data_files,
            max_chars=max_chars,
            max_samples=args.max_samples,
            pinned_files=pinned_files,
        )
        print(f"[QLoRA-30B] Loaded {len(sft_ds)} complete combined Teacher + Argon + Lego-First trajectories.", flush=True)

    if args.mode in ("dpo", "sft_dpo"):
        dpo_path = Path(args.dpo_dataset)
        if not dpo_path.exists():
            raise FileNotFoundError(f"DPO preference dataset not found at {dpo_path}")
        print(f"[DPO-30B] Loading Referee preference pairs from {dpo_path}...", flush=True)
        dpo_ds = load_dpo_dataset(dpo_path, max_chars=max_chars)
        print(f"[DPO-30B] Loaded {len(dpo_ds)} balanced Precision + Recall + Architecture preference pairs.", flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[QLoRA-30B] Loading base model {args.base_model} in 4-bit NF4 + BF16 MoE experts...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # NOTE: Do NOT call prepare_model_for_kbit_training(base_model) here because in Transformers >= 5.0,
    # Qwen3-MoE stores 128 routed experts per layer as 3D BF16 Parameter tensors (~56 GB), and
    # prepare_model_for_kbit_training upcasts all non-4bit BF16 parameters to FP32 (~112 GB -> OOM).
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    resumed_from = None
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        print(f"[QLoRA-30B] Resuming trainable PEFT adapter from {adapter_dir}...", flush=True)
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
        print("[QLoRA-30B] Starting Stage-1 SFT training...", flush=True)
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
                print("[DPO-30B] Anchored Stage-2 DPO reference policy to frozen Stage-1 adapter ('sft_ref').", flush=True)

            dpo_config = build_dpo_config(output_dir, args, use_ref_adapter=loaded_ref_adapter)
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                processing_class=tokenizer,
                train_dataset=dpo_ds,
                args=dpo_config,
            )
            print("[DPO-30B] Starting Stage-2 30B Referee Preference Calibration (DPO)...", flush=True)
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

    print(f"[QLoRA-30B] Saving LoRA adapter & tokenizer to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[QLoRA-30B] Done! Wrote Ollama Modelfile to {modelfile} and metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
