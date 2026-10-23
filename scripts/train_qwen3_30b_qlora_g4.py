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
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3-Coder-30B-A3B on distilled gpt-oss:120b Jean SRE trajectories."
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
        "--output-dir",
        default="models/jean-qwen3-30b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT epochs.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max token context during QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=1200, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    return parser.parse_args()


def _clean_utf8(text: str) -> str:
    return str(text or "").encode("utf-8", "replace").decode("utf-8")


def load_chatml_dataset(data_files: list[str], max_chars: int = 20000, max_samples: int = 1200) -> Dataset:
    records = []
    for fp in data_files:
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
                    # Keep complete trajectories that fit cleanly inside max_seq_length without mid-response truncation
                    if total_chars <= max_chars:
                        records.append({"messages": cleaned_msgs})
    ds = Dataset.from_list(records).shuffle(seed=42)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def write_ollama_modelfile(output_dir: Path) -> Path:
    modelfile_path = output_dir / "Modelfile"
    content = """# Ollama Modelfile — Sovereign Code Auditor & Remediator 30B (Teacher: gpt-oss:120b + Gemini 4 Argon)
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
4. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers exactly.\"\"\"
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


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = [str(dataset_path)]
    argon_path = Path(args.argon_dataset)
    if argon_path.exists():
        # Oversample Argon Golden Remediations 3x so verified code patches carry strong weight
        data_files.extend([str(argon_path)] * 3)
        print(f"[QLoRA-30B] Including Argon Golden Remediation dataset (3x oversampled) from {argon_path}")

    print(f"[QLoRA-30B] Loading dataset files: {data_files}...")
    ds = load_chatml_dataset(data_files, max_chars=args.max_seq_length * 5, max_samples=args.max_samples)
    print(f"[QLoRA-30B] Loaded {len(ds)} complete combined Teacher + Argon ChatML trajectories.")

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

    sft_config = build_sft_config(output_dir, args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )

    print("[QLoRA] Starting SFT training...")
    trainer.train()

    print(f"[QLoRA] Saving LoRA adapter & tokenizer to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    print(f"[QLoRA] Done! Wrote Ollama Modelfile to {modelfile}")


if __name__ == "__main__":
    main()
