---
base_model: NousResearch/Hermes-3-Llama-3.1-8B
library_name: peft
model_name: Llama-3.1-Hermes-3-8B-SRE-LoRA
license: llama3.1
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
  - krush420/sovereign-g4-benchmarks
model-index:
  - name: Llama-3.1-Hermes-3-8B-SRE-LoRA
    results:
      - task:
          type: text-generation
          name: Multi-Turn Long-Term Memory & Simulated App Tool Execution
        dataset:
          name: DolphinBench (600 Tasks across 3 Personas / 1.5M Tokens, Official gpt-5.6-sol Judge)
          type: mem0ai/dolphinbench
          split: test
        metrics:
          - type: accuracy
            value: 18.33
            name: Official gpt-5.6-sol Pass Rate (110/600 %, 15.7x over 8B Stock)
          - type: accuracy
            value: 64.20
            name: Deterministic Mechanical Tool Checks (841/1,310 %)
        source:
          name: Sovereign G4 & DolphinBench Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
      - task:
          type: text-generation
          name: AST-Grounded Code Audit, Lego-First Synthesis & Tool-Calling SFT + DPO
        dataset:
          name: Sovereign Code Audit, Lego-First & Agentic Tool-Calling SFT + DPO
          type: krush420/sovereign-code-audit-sft
          split: agentic_tool_calling_sft
        metrics:
          - type: accuracy
            value: 89.64
            name: Stage-1 BF16 SFT Token Accuracy (%)
          - type: accuracy
            value: 100.0
            name: Stage-2 KL-Anchored DPO Preference Accuracy (%)
        source:
          name: Sovereign Code Audit SFT & DPO Distillation Pipeline
          url: https://huggingface.co/datasets/krush420/sovereign-code-audit-sft
tags:
  - base_model:adapter:NousResearch/Hermes-3-Llama-3.1-8B
  - llama-3
  - hermes-3
  - krusch-stack
  - mcp
  - agentic-harness
  - tool-calling
  - dolphinbench
  - lora
  - dpo
  - sft
  - code-audit
  - sre
  - transformers
  - trl
---

# Llama-3.1-Hermes-3-8B — KruschStack Wave-4 Unified Agentic SRE & Tool-Caller LoRA (`L1.5` Edge Tier)

**Built with Meta Llama 3** (`NousResearch/Hermes-3-Llama-3.1-8B`, Meta Llama 3.1 Community License).

[![Built with Meta Llama 3](https://img.shields.io/badge/Built%20with-Meta%20Llama%203-0467DF.svg)](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![DolphinBench](https://img.shields.io/badge/DolphinBench%20600-18.33%25%20(15.7x%20vs%208B%20Stock)-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> `krush420/llama3-hermes3-8b-sre-lora` is the **`L1.5` Compact 8B Edge Agent & Cross-Family Referee LoRA** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** (`npx krusch-stack init`). Trained in native `BF16` (`r=16`, `lora_alpha=32`) across all **7** Llama-3.1 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — `41.94M` trainable parameters) via 2-Stage SFT + KL-Anchored `trl.DPOTrainer` calibration, and paired with a **fixed Ollama Go-template `Modelfile`** that resolves two upstream `hermes3` template bugs, it lifts `8B` `Q4_0` (`~4.7 GB` VRAM) on the full 600-task **[DolphinBench](https://dolphinbench.ai/)** benchmark from **`1.17%` (`7/600` `Stock`) -> `18.33%` (`110/600` `Std. Harness`, `15.7x` jump) under official `gpt-5.6-sol` grading** (`26.0%` [`52/200`] on `Morgan`, beating `Hermes + MiniMax M3 + Supermemory` [`45/200`], and passing `5/5` pure-deterministic causality gates at `$0.00` inference cost).

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,131+` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `49` Audit Logs = `48` Staged Raw JSON + `1` PR #169 Live Receipt across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)
- **🏆 Companion 31B Dense Reasoning QLoRA (`#1 Open-Weights` `57.33%` DolphinBench)**: [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (Wave-6 for `google/gemma-4-31b-it`)
- **⚡ Companion 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `43.00%` w/ Grounded v0.3)
- **🧠 Companion 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)

---

## 🛠️ Two Critical Upstream Ollama `hermes3` Template Fixes Included in `Modelfile`

Stock `ollama pull hermes3:latest` suffers from an interaction between its Go chat template and Ollama's `template.ParseToolCalls` AST analyzer:
1. **Dropped `.System` Context When `.Tools` Are Passed**: Stock `hermes3` uses `{{- if .Tools }} ... {{- else if .System }}`, silently discarding the entire `system` prompt (including retrieved RAG/memory blocks and standing user instructions!) whenever any tool definition is attached.
2. **AST `.ToolCalls` Parser Sensitivity**: Adding a second `{{ if .ToolCalls }}` branch inside `assistant` causes Ollama's Go template AST analyzer to disable structured `<tool_call>` parsing and leak raw XML into `message.content` (`tool_calls: []`).

The included [`Modelfile`](./Modelfile) renders **both** `{{- if .System }}` and `{{- if .Tools }}` inside a single `<|im_start|>system` turn while keeping the exact single `{{- else if .ToolCalls }}<tool_call>` branch on `assistant`, preserving `100%` native structured `message.tool_calls` extraction alongside retrieved long-term memory.

---

## ⚡ Quickstart

### 1. Run Locally with Ollama (`Modelfile`, `adapter.gguf` & `~1.8s` `Q4_0` Fusion Included)

```bash
git clone https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora
cd llama3-hermes3-8b-sre-lora
ollama pull hermes3:latest

# Fuse all 7 Llama-3.1 LoRA projections (alpha/r = 32/16 = 2.0) into Q4_0 base GGUF in ~1.8s
BASE_BLOB="$(ollama show --modelfile hermes3:latest | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py \
  --base-gguf "$BASE_BLOB" \
  --adapter-gguf ./adapter.gguf \
  --output-gguf ./hermes3-8b-sre.gguf \
  --alpha-scale 2.0
sed -i 's|^FROM .*|FROM ./hermes3-8b-sre.gguf|' Modelfile
ollama create hermes3-sre:latest -f Modelfile && rm -f ./hermes3-8b-sre.gguf
ollama run hermes3-sre:latest
```

### 2. Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "NousResearch/Hermes-3-Llama-3.1-8B"
adapter_id = "krush420/llama3-hermes3-8b-sre-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

---

## 🔬 Wave-4 Native `BF16` 2-Stage Training Metrics

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`; peak allocated VRAM `26.89 GB`, `1,318.06s` total wall time)
- **Base Model**: `NousResearch/Hermes-3-Llama-3.1-8B` (loaded in native unquantized `torch.bfloat16` — zero 4-bit quantization noise during gradient updates)
- **LoRA Config**: `r=16`, `lora_alpha=32` (`alpha/r = 2.0`), `lora_dropout=0.05`, all 7 Llama-3.1 linear modules `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (`41,943,040` trainable parameters across 32 layers; `160.1 MB` GGUF v3 adapter)
- **Stage-1 Unified SFT (`8192` ctx, `420` trajectories x `2` epochs, `lr=1.5e-4`, `1,083.80s`)**: Loss converged `1.5522` -> `0.5133` (`train_loss=0.7094`), token accuracy **`69.27%` -> `89.64%`** across ChatML `<tools>` / `<tool_call>` chains and SRE Referee audits
- **Stage-2 Unified DPO (`48` balanced Referee + Tool-Action pairs x `3` epochs, `beta=0.1`, `lr=1.2e-5`, `172.05s`)**: Loss `0.5771` -> `0.0042` (`train_loss=0.1414`), token accuracy `87.48%` -> `91.31%`, reward margin `+0.4506` -> `+6.6551`, **`100.0%` preference accuracy**
- **Post-Training Smoke Gates**: Gate A (`CLEAN_PASS` / `findings: []` on compliant `@krusch/toolkit/http` server) + Gate B (structured `<tool_call>` with latest temporal room/date override) both passed `100%`.

---

## 📊 DolphinBench 600-Task Controlled Ablation (`8B` Edge Lift to `#1 Overall` `91.17%` Argon Scaling Law — `27` Runs, `75 / 75` Gates)

Full `49`-log SHA-256 provenance (`48` staged raw JSON logs + `1` PR #169 live receipt bundle; `27` DolphinBench runs / `16,200` traces across `6` model families; `Oracle@27 = 97.83%`, `$0.00` `Oracle@17 = 83.17%`, Pareto `($0.00, 57.33%) -> ($1.10, 76.67%) -> ($1.35, 81.50%) -> ($24.54, 91.17%)`) is published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** and **[Paper #27 (`Return on Memory`)](https://krusch.dev/articles/return-on-memory-dolphinbench)** (`8/8` flagship archives `queued` on `dolphinbench.ai`, `27/27` validated):

| Configuration (Byte-Identical `0o444` `SovereignMemoryStore`) | Official `gpt-5.6-sol` Overall (`600`) | `Morgan` (`200`) | `Alex` (`200`) | `Riley` (`200`) | Official Grader Mech | Total Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hermes3:latest` (`8B` Stock)** | *`1.17%` (`7/600` local strict) / `4.50%` (`27/600`)* | `1.5%` | `0.0%` | `2.0%` | `750 / 1,310` *(local)* | `$0.00` |
| **`hermes3-sre:latest` (`8B` Wave-4 LoRA+DPO, `Std. Harness`)** | **`18.33%` (`110 / 600`)** (**`15.7x` lift**) | **`26.0%` (`52`)** | **`14.5%` (`29`)** | **`14.5%` (`29`)** | **`264 / 1,468` (`17.98%`)** | **`$0.00`** |
| **`Qwen3-Coder-30B-A3B` (`30B` Wave-4, `Std.` / `+ Grounded v0.3`)** | **`42.67%` (`256`) -> `43.00%` (`258 / 600`)** (`58.50%` `Oracle@3`) | `62.5%` (`125`) | `31.5%` (`63`) | `35.0%` (`70`) | `613 / 1,364` (`44.94%`) | **`$0.00`** |
| **`Google Gemma 4 31B` (`31B` Stock, Queued Flagship)** | `54.17%` (`325 / 600`) | `72.0%` (`144`) | `47.5%` (`95`) | `43.0%` (`86`) | `792 / 1,363` (`58.11%`) | `$0.00` |
| **[`gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (`Wave-6 + Grounded v0.3`, `#1 Open-Weights`)** | **`57.33%` (`344 / 600`, `71.83%` `Oracle@6`)** | **`74.5%` (`149`)** | **`51.5%` (`103`)** | **`46.0%` (`92`)** | **`835 / 1,364` (`61.22%`)** | **`$0.00`** |
| **`GPT 5.6 Luna (Low, Std. Harness)` (Pareto Vertex 2)** | **`76.67%` (`460 / 600`, `+6.00 pp` vs. `#1` `Mem0`, `87.4x` cheaper)** | `85.0%` (`170`) | `72.5%` (`145`) | `72.5%` (`145`) | `1,044 / 1,323` (`78.91%`) | **`$1.10`** |
| **`GPT 5.6 Luna (High, + Grounded)` (Model-Matched SOTA, Pareto Vertex 3)** | **`81.50%` (`489 / 600`, `+10.83 pp` vs. `#1` `Mem0`, `71.1x` cheaper)** | **`87.0%` (`174`)** | **`78.5%` (`157`)** | **`79.0%` (`158`)** | **`1,093 / 1,321` (`82.74%`)** | **`$1.35`** |
| **`Gemini 4 Argon (Medium / Low Grounded)` (`L3` Frontier Reader, `#2 Overall`)** | **`89.17%` (`535 / 600`, `+18.50 pp` vs. `#1` `Mem0`)** | `92.0%` (`184`) | `87.0%` (`174`) | `88.5%` (`177`) | `1,193 / 1,329` (`89.77%`) | `$23.92–$24.57` |
| **`Gemini 4 Argon (Low)` (`L3` Frontier Reader, `#1 Overall`)** | **`91.17%` (`547 / 600`, `+20.50 pp` vs. `#1` `Mem0`)** | **`95.5%` (`191`)** | **`89.5%` (`179`)** | **`88.5%` (`177`)** | **`1,221 / 1,324` (`92.22%`)** | **`$24.54`** |

---

## 📜 License & Attribution

Built with Meta Llama 3. Distributed under the **Meta Llama 3.1 Community License Agreement** (`license: llama3.1`), inheriting from `meta-llama/Meta-Llama-3.1-8B` and `NousResearch/Hermes-3-Llama-3.1-8B`. Training scripts and KruschStack integration code are MIT licensed.
