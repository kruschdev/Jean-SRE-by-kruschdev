---
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
library_name: peft
model_name: qwen3-coder-30b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
tags:
  - base_model:adapter:Qwen/Qwen3-Coder-30B-A3B-Instruct
  - krusch-stack
  - mcp
  - agentic-harness
  - lora
  - qlora
  - sft
  - code-audit
  - bug-remediation
  - sre
  - transformers
  - trl
---

# Qwen3-Coder-30B-A3B — KruschStack Sovereign Code Auditor & Remediator LoRA (`L2` Student)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![Base Model](https://img.shields.io/badge/Base-Qwen3--Coder--30B--A3B-blue.svg)](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> This adapter is the official **`L2` Local MoE Student** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It was distilled from trajectories generated, AST-grounded, and unit-test verified inside the KruschStack 3-Tier MCP harness so that a fast local 30B MoE model (`3B` active parameters) can perform Principal-level 11-phase code audits and test-gated bug remediations across the KruschStack ecosystem.

---

## 🔗 The KruschStack Model & Dataset Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **📊 Training Dataset (`17,899` Trajectories)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🧠 Companion 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`qwen3-coder-30b-sre-lora` is trained specifically to reason over tool outputs from the **KruschStack Unified MCP Gateway** and act as the **`L2` Local MoE Specialist** in [`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router):

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
│ • <25µs L1 fast-path -> [L2: qwen3-coder-30b-sre-lora] -> L2.5 120B        │
└────────────────────────────────────────────────────────────────────────────┘
```

### What This Adapter Learned Through KruschStack
1. **Gemini 4 Argon (`gemini-4-argon`) Golden Remediations (`argon_golden_remediations`):** Multi-file root-cause bug fixes staged through `krusch-harness` and verified against live unit test suites (`node:test` / `pytest`, exit code `0`).
2. **`openai/gpt-oss-120b` Elitist Code Audit Trajectories (`elitist_120b_teacher`):** 11-phase AST-grounded codebase audits using `krusch-git` (`search_symbols`, `dependency_graph`, `read_blob`) with verbatim `file:startLine-endLine` citations, false-positive refutation (`NOT FLAGGED`), and Zero-Finding convergence (`CLEAN_PASS`).
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

### 2. Run Locally with Ollama (`Modelfile` Included)

```bash
git clone https://huggingface.co/krush420/qwen3-coder-30b-sre-lora
cd qwen3-coder-30b-sre-lora
ollama create qwen3-coder-30b-sre:latest -f Modelfile
ollama run qwen3-coder-30b-sre:latest
```

### 3. Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
adapter_id = "krush420/qwen3-coder-30b-sre-lora"

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
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --enable-lora \
  --lora-modules code-auditor=krush420/qwen3-coder-30b-sre-lora
```

---

## 🔬 Training Hyperparameters & Environment

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`)
- **Base Model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit NF4 attention + BF16 3D MoE experts)
- **LoRA Config**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`25,165,824` trainable parameters)
- **Training Metrics**: Loss converged `0.731` $\rightarrow$ `0.214`; token accuracy reached `93.36%` (`eval_loss=0.3358`, `eval_mean_token_accuracy=90.35%`)
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`lr=2e-4`, `warmup_ratio=0.05`)
- **Frameworks**: `peft 0.21.2`, `trl 1.16.0`, `transformers 5.20.2`, `torch 2.11.0+cu130`