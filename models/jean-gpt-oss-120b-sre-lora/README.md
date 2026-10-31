---
base_model: openai/gpt-oss-120b
library_name: peft
model_name: gpt-oss-120b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
  - krush420/sovereign-g4-benchmarks
model-index:
  - name: gpt-oss-120b-sre-lora
    results:
      - task:
          type: question-answering
          name: Long-Term Conversational Memory QA
        dataset:
          name: Official ICLR 2026 BEAM (100K Full Split, n=400)
          type: Mohammadta/BEAM
          split: 100K
        metrics:
          - type: recall
            value: 92.37
            name: Strict Primary Turn Recall (9-Cat %)
          - type: accuracy
            value: 69.03
            name: E2E QA Score (Hermes-3 Cross-Family Referee %)
          - type: accuracy
            value: 60.80
            name: E2E QA Score (Qwen2.5-32B Referee %)
        source:
          name: Sovereign G4 Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
      - task:
          type: text-generation
          name: Multi-Language Repository Code Editing (2PC Verify-on-Stop)
        dataset:
          name: Aider Polyglot Benchmark (6 Languages, n=225)
          type: Aider-AI/polyglot-benchmark
          split: test
        metrics:
          - type: pass@2
            value: 51.56
            name: 2PC Verify-on-Stop Pass@2 (%)
          - type: pass@1
            value: 19.56
            name: Blind 1-Shot Pass@1 (%)
        source:
          name: Krusch 2PC Coding Harness Evaluation
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
tags:
  - base_model:adapter:openai/gpt-oss-120b
  - krusch-stack
  - mcp
  - agentic-harness
  - lora
  - qlora
  - mxfp4
  - sft
  - dpo
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
> This adapter is the official **`L2.5` High-VRAM Burst Teacher & Deep Referee Auditor** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It fine-tunes [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MoE parameters) via 2-Stage MXFP4 QLoRA SFT + `trl.DPOTrainer` preference calibration on trajectories generated, AST-grounded, and unit-test verified through the KruschStack 3-Tier MCP ecosystem.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,049` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`316,981+` Verified Records, `68/68` Gates)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
- **⚡ Companion 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`gpt-oss-120b-sre-lora` serves as the **`L2.5` 120B Burst Teacher & Blind Spec Referee** inside **[`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router)** (`<25µs` routing latency), handling high-complexity multi-file root-cause analysis, spec-first Referee convergence gates, and teacher distillation for the `L2` 30B local model:

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
3. **Lego-First Idiom Synthesis & Rule-of-3 Anti-Bloat Gate (`INV-LEGO-01..16`):** Assembling services from `@krusch/toolkit` v1.5 (24 runtime bricks + `.d.ts` declarations) and `krusch_toolkit` v1.3 (`fastapi_app`, `mcp`, `pg`, `security`), nominating `>= 3`-repo clusters via `CANDIDATE_MEETS_RULE_OF_3`, and rejecting `< 3`-repo utilities as `BELOW_THRESHOLD_DO_NOT_EXTRACT`.
4. **2-Stage SFT + Balanced DPO Referee Calibration (`trl.DPOTrainer`):** Stage-2 Direct Preference Optimization (`beta=0.1`, `12` balanced Precision + Recall + Architecture pairs) anchored to the Stage-1 SFT adapter so the model emits calibrated `CLEAN_PASS` (`"findings": []`) verdicts when production code satisfies frozen invariants while preserving `100%` P0/P1 defect recall.
5. **Zero-Leak Portability Gate:** Scrubbed of all private IPs, internal hostnames, user paths, and credentials so the adapter generalizes cleanly to any local or enterprise repository using `krusch-stack`.

---

## ⚡ Quickstart

### 1. Install KruschStack (`npx krusch-stack`)

```bash
# Auto-configure Claude Code (.mcp.json), Cursor, Windsurf & Antigravity + initialize SQLite
npx krusch-stack init

# Verify all 7 stack pillars (39 annotated MCP tools)
npx krusch-stack doctor
```

### 2. Run with Ollama (`Modelfile`, `adapter.gguf` & `~1.7s` GGUF Fusion Included)

```bash
git clone https://huggingface.co/krush420/gpt-oss-120b-sre-lora
cd gpt-oss-120b-sre-lora
ollama pull gpt-oss:120b

# Ollama >= 0.40: Fuse rank-16 attention LoRA into base GGUF in ~1.7s (preserves Harmony renderer & MXFP4 experts)
BASE_BLOB="$(ollama show --modelfile gpt-oss:120b | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf "$BASE_BLOB"
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

## 🔬 Training Hyperparameters & Single-GPU MXFP4 2-Stage Architecture

- **Hardware**: Single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`; peak allocated VRAM `74.89 GB`, `>20 GB` headroom)
- **Base Model**: `openai/gpt-oss-120b` (`116.8B` total / `5.1B` active MoE parameters; 36 layers of 128 native 4-bit MXFP4 routed experts kept frozen in `~61.8 GB` VRAM while gradients flow through `attn_implementation="eager"` BF16 Pre-LN residual stream into `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Config**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`11,943,936` trainable parameters; `45.58 MB` GGUF v3 adapter)
- **Wave-1/2 Seed SFT**: Loss converged `7.448` $\rightarrow$ `1.737`; token accuracy increased `20.11%` $\rightarrow$ `64.89%` (`eval_loss=1.8054`)
- **Wave-3 Stage-1 Calibration SFT (`5120` ctx, `224` complete trajectories, `lr=5e-5`, `560.84s`)**: Loss `1.8257` $\rightarrow$ `1.3664` (`train_loss=1.6558`); token accuracy `60.55%` $\rightarrow$ `68.02%`
- **Wave-3 Stage-2 Referee DPO (`12` balanced pairs $\times$ `3` epochs, `beta=0.1`, `lr=5e-6`, `75.83s`)**: Loss `0.6931` $\rightarrow$ `0.5824` (step 8) / `0.6631` (step 9), `train_loss=0.6634`, reward margins `0.000` $\rightarrow$ `+0.2438` (`100%` preference accuracy on calibrated Referee & Lego-First batches)
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`warmup_ratio=0.05`)
- **Frameworks**: `peft 0.21.2`, `trl 1.17.1`, `transformers 5.22.1`, `torch 2.11.0+cu130`, `datasets 5.1.0`

---

## 🏆 Hermetic External Benchmark Scorecard (`96 GB` Blackwell G4, `0` Cloud APIs)

Full SHA-256-pinned per-query logs (`68/68` integrity gates across `316,981+` verified queries) are published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** alongside the scope-qualified cross-stack write-up (**[*How an Open-Weights Sovereign Stack Stacks Up*](https://krusch.dev/articles/how-krusch-stack-stacks-up)**):

| Benchmark | Configuration Under Test | Verified Result | External Leaderboard Context |
| :--- | :--- | :--- | :--- |
| **ICLR 2026 BEAM (`100K`–`10M`)** | `gpt-oss-120b-sre-lora` & Stock `120B` + [`krusch-context-mcp`](https://github.com/kruschdev/krusch-context-mcp) `v2.0.0` | **`92.37%`–`92.58%` `100K` Recall** · **`69.03%` QLoRA E2E QA** · **`72.86%` `1M` E2E QA** | Zero E2E drop from `100K` → `1M`; beats cloud `past.dev` on Contradiction Resolution (`4/4` splits). |
| **Aider Polyglot (`225/225`)** | `openai/gpt-oss-120b` + [`krusch-coding-harness`](https://github.com/kruschdev/krusch-coding-harness) `2PC` `Verify-on-Stop` | **`Pass@1 = 19.56%` → `Pass@2 = 51.56%` (`116/225`)** (`72` tasks self-healed) | **Ties cloud `Claude 3.5 Sonnet` (`51.6%`)** & **beats `GPT-4o` (`23.1%`)** across C++, Go, Java, JS, Python & Rust at `\$0` API cost. |
| **BIRD-SQL (`500`) & AgentDojo (`1,046`)** | [`krusch-coding-harness`](https://github.com/kruschdev/krusch-coding-harness) `2PC` SQL Execution & Policy Floor + BFCL v3 (`2,251`) | **`26.00%` → `40.20%` BIRD EX** · **`0.00%` AgentDojo ASR (`0/949`)** · **`66.28%` BFCL** | **`+14.20 pp` SQL self-healing (`100%` valid SQL)**; **`19.28%` → `0.00%` prompt-injection ASR**; **`-80.67%` MCP schema tokens**. |
| **Martian RouterBench & Live SSE** | [`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router) (`RouterBench` `73K` + `LLMRouterBench` + `RouterArena`) | **`#1` RouterBench ID (`0.7488`/`0.7695`) & OOD (`0.7268`/`0.7455`)** · **`-48.0%` TTFT** | **`-72.63%` cost @ `98%` `GPT-4`** & beats `ZeroRouter` on OOD; **`#1` ACL 2026 `LLMRouterBench` Pareto (`0.4269`)**. |
| **ACL 2025 CoIR (`2.39M`) & RepoBench-R** | [`krusch-git`](https://github.com/kruschdev/krusch-git) `v1.2.3` AST + Hybrid RRF + `snowflake-arctic-embed2` | **🥇 `#1` CoIR Macro `NDCG@10` (`56.40%`)** · **`19.12%` RepoBench `Acc@1`** · **`68.58%` `Acc@5`** | Beats `Voyage-Code-002` (`56.26%`) on CoIR & `CodeBERT`/`UniXcoder` across Python & Java on RepoBench-R. |
| **LegalBench-RAG (`6,889` Qs)** | [`KruschNexus`](https://github.com/kruschdev/krusch-nexus) `v0.2.6` + Arctic `1024d` + `mxbai-rerank-large-v2` | **`26.74%` Macro F1 (`k=1`)** · **`25.81%` F1 / `44.72%` R (`k=2`)** · **`33.95%` `MAUD` Recall@2** | **`+128.5%` (`2.29x`) F1** vs. fixed dense; exceeds OpenAI `text-embedding-3-large` + Cohere Rerank `MAUD` `Recall@64` (`28.28%`) at `top_k=2`. |
