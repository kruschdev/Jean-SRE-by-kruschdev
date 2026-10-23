---
base_model: openai/gpt-oss-120b
library_name: peft
model_name: gpt-oss-120b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
tags:
  - base_model:adapter:openai/gpt-oss-120b
  - krusch-stack
  - mcp
  - agentic-harness
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

# gpt-oss-120b — KruschStack Sovereign Code Auditor & Remediator LoRA (`L2.5` Teacher)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![Base Model](https://img.shields.io/badge/Base-openai%2Fgpt--oss--120b-blue.svg)](https://huggingface.co/openai/gpt-oss-120b)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> This adapter is the official **`L2.5` High-VRAM Burst Teacher & Deep Auditor** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It fine-tunes [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MoE parameters) on trajectories generated, AST-grounded, and unit-test verified through the KruschStack 3-Tier MCP ecosystem.

---

## 🔗 The KruschStack Model & Dataset Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **📊 Training Dataset (`17,899` Trajectories)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **⚡ Companion 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`gpt-oss-120b-sre-lora` serves as the **`L2.5` 120B Burst Teacher** inside the **[`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router)** (`<25µs` routing latency), handling high-complexity multi-file root-cause analysis, architectural refactoring, and teacher distillation for the `L2` 30B local model:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        KRUSCH-STACK UNIFIED GATEWAY                        │
│               `npx krusch-stack mcp`  (39 Annotated MCP Tools)             │
├──────────────────────┬───────────────────────────┬─────────────────────────┤
│ 1. WORKING MEMORY    │ 2A. CODEBASE AST & GIT    │ 2B. OS PAGE MEMORY      │
│ krusch-context-mcp   │ krusch-git                │ krusch-fs               │
│ • retrieve / remember│ • search_symbols          │ • locate / search       │
│ • revise / nudge     │ • dependency_graph        │ • page_tree / read_page │
│ • INV-12 span drift  │ • read_tree / read_blob   │ • remember_on_page      │
├──────────────────────┼───────────────────────────┼─────────────────────────┤
│ 3. 2PC HARNESS       │ 4. FRONTEND DESIGN MCP    │ 5. WORKSPACE HUB        │
│ krusch-harness       │ krusch-frontend-mcp       │ g-krusch                │
│ • run (stages diff)  │ • generate_color_palette  │ • get_agent_steering    │
│ • apply_diff (2PC)   │ • css_to_tailwind         │ • search_drive_rag      │
│ • explain / reject   │ • canonical_templates     │ • write (.gdoc/.slides) │
├──────────────────────┴───────────────────────────┴─────────────────────────┤
│ 6. COGNITIVE CASCADE ROUTER (`krusch-cascade-router`)                      │
│ • <25µs L1 fast-path -> L2 30B Student -> [L2.5: gpt-oss-120b-sre-lora]    │
└────────────────────────────────────────────────────────────────────────────┘
```

### What This Adapter Learned Through KruschStack
1. **Gemini 4 Argon (`gemini-4-argon`) Golden Remediations (`argon_golden_remediations`):** Multi-file root-cause bug fixes staged through `krusch-harness` and verified against live unit test suites (`node:test` / `pytest`, exit code `0`).
2. **Elitist Rejection-Sampled 11-Phase Code Audits (`elitist_120b_teacher`):** AST-grounded codebase audits using `krusch-git` (`search_symbols`, `dependency_graph`, `read_blob`) with verbatim `file:startLine-endLine` citations, false-positive refutation (`NOT FLAGGED`), and Zero-Finding convergence (`CLEAN_PASS`).
3. **Zero-Leak Portability Gate:** Scrubbed of all private IPs, internal hostnames, user paths, and credentials so the adapter generalizes cleanly to any local or enterprise repository using `krusch-stack`.

---

## ⚡ Quickstart

### 1. Install KruschStack (`npx krusch-stack`)

```bash
# Auto-configure Claude Code (.mcp.json), Cursor, Windsurf & Antigravity + initialize SQLite
npx krusch-stack init

# Verify all 7 stack pillars (39 annotated MCP tools)
npx krusch-stack doctor
```

### 2. Run with Ollama (`Modelfile` Included)

```bash
git clone https://huggingface.co/krush420/gpt-oss-120b-sre-lora
cd gpt-oss-120b-sre-lora
ollama create gpt-oss-sre:120b -f Modelfile
ollama run gpt-oss-sre:120b
```

### 3. Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "openai/gpt-oss-120b"
adapter_id = "krush420/gpt-oss-120b-sre-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

### 4. vLLM Multi-LoRA Serving

```bash
vllm serve openai/gpt-oss-120b \
  --enable-lora \
  --lora-modules code-auditor=krush420/gpt-oss-120b-sre-lora
```

---

## 🔬 Training Hyperparameters & Single-GPU MXFP4 Architecture

- **Hardware**: Single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`)
- **Base Model**: `openai/gpt-oss-120b` (`116.8B` total / `5.1B` active MoE parameters; 36 layers of 128 native 4-bit MXFP4 routed experts kept frozen in `~60.8 GB` VRAM while gradients flow through the BF16 Pre-LN residual stream into `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Config**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`11,943,936` trainable parameters)
- **Training Metrics**: Loss converged `7.448` $\rightarrow$ `1.737`; token accuracy increased `20.11%` $\rightarrow$ `64.89%` (`eval_loss=1.8054`)
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`lr=1e-4`, `warmup_ratio=0.05`)
- **Frameworks**: `peft 0.21.2`, `trl 1.16.0`, `transformers 5.20.2`, `torch 2.11.0+cu130`
