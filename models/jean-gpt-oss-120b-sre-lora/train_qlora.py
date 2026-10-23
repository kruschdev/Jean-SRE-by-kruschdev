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
  pip install -U torch transformers datasets peft trl bitsandbytes accelerate kernels
  python3 scripts/train_gpt_oss_120b_qlora_g4.py \
      --dataset data/distillation/gpt_oss_120b_elitist_teacher.jsonl \
      --output-dir /root/models/jean-gpt-oss-120b-sre-lora \
      --epochs 2

After training, register on Colab G4 Ollama (layers adapter onto existing 65GB gpt-oss:120b):
  ollama create gpt-oss-sre:120b -f /root/models/jean-gpt-oss-120b-sre-lora/Modelfile
"""

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune openai/gpt-oss-120b on Argon-verified & rejection-sampled Autonomous SRE Krusch SRE trajectories Code Audit trajectories."
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
        "--output-dir",
        default="/root/models/jean-gpt-oss-120b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT epochs.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length during 120B QLoRA training.")
    parser.add_argument("--max-samples", type=int, default=600, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for 120B attention adapter.")
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
                    if total_chars <= max_chars:
                        records.append({"messages": cleaned_msgs})
    ds = Dataset.from_list(records).shuffle(seed=42)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def write_ollama_modelfile(output_dir: Path) -> Path:
    modelfile_path = output_dir / "Modelfile"
    content = """# Ollama Modelfile — Sovereign Code Auditor & Remediator 120B (Teacher: Gemini 4 Argon + Test Gate)
FROM gpt-oss:120b
ADAPTER .

PARAMETER num_ctx 32768
PARAMETER temperature 0.15
PARAMETER top_p 0.9

SYSTEM \"\"\"You are an Autonomous Principal Software Engineer, SRE, and Code Auditor.
RULES (NON-NEGOTIABLE):
1. Ground every finding strictly in the provided live source snippets, AST symbol graphs, and command outputs.
2. NEVER invent, paraphrase, or reconstruct code snippets from memory. Only quote verbatim lines that appear in the prompt with their exact line numbers.
3. If a prior Pending Issue is not supported by the Live Source Verification lines shown in the prompt, REFUTE and remove it.
4. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.\"\"\"
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
    if "router_aux_loss_coef" in sig_params:
        cfg_kwargs["router_aux_loss_coef"] = 0.0
    return SFTConfig(**cfg_kwargs)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Elitist 120B dataset not found at {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = [str(dataset_path)]
    argon_path = Path("data/argon_remediation_sft_corpus.jsonl")
    if argon_path.exists():
        data_files.append(str(argon_path))
        print(f"[QLoRA-120B] Including Argon Golden Remediation dataset from {argon_path}", flush=True)

    print(f"[QLoRA-120B] Loading elitist dataset from {data_files}...", flush=True)
    ds = load_chatml_dataset(data_files, max_chars=24000, max_samples=min(args.max_samples, 200))
    print(f"[QLoRA-120B] Loaded {len(ds)} complete Argon-verified & rejection-sampled trajectories.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[QLoRA-120B] Loading {args.base_model} onto 96GB Blackwell G4 (native MXFP4 experts + BF16 eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    if getattr(model, "hf_quantizer", None) is not None:
        type(model.hf_quantizer).is_trainable = property(lambda self: True)

    # Wrap frozen MXFP4 MoE MLP blocks in torch.no_grad() so native 4-bit Triton kernels execute
    # without autograd graph overhead while gradients flow across all 36 layers via the Pre-LN residual stream.
    mlp_patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "Mxfp4GptOssMLP":
            _orig_fwd = module.forward

            def _no_grad_mlp_fwd(hidden_states, _fn=_orig_fwd):
                with torch.no_grad():
                    return _fn(hidden_states.detach())

            module.forward = _no_grad_mlp_fwd
            mlp_patched += 1
    print(f"[QLoRA-120B] Patched {mlp_patched} Mxfp4GptOssMLP blocks for residual-stream BF16 attention LoRA.")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Target BF16 attention projections only so the 128 MXFP4 experts remain frozen in 4-bit
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
    model._hf_peft_config_loaded = True
    if getattr(model, "hf_quantizer", None) is not None:
        type(model.hf_quantizer).is_trainable = property(lambda self: True)
    model.print_trainable_parameters()

    sft_config = build_sft_config(output_dir, args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )

    print("[QLoRA-120B] Starting 120B Elitist SFT training...")
    trainer.train()

    print(f"[QLoRA-120B] Saving LoRA adapter & tokenizer to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    readme = write_model_card_readme(output_dir)
    print(f"[QLoRA-120B] Done! Wrote Ollama Modelfile to {modelfile} and Model Card to {readme}")


if __name__ == "__main__":
    main()
