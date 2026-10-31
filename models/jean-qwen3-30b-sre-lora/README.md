---
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
library_name: peft
model_name: qwen3-coder-30b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
  - krush420/sovereign-g4-benchmarks
model-index:
  - name: qwen3-coder-30b-sre-lora
    results:
      - task:
          type: text-generation
          name: AST-Grounded Code Audit & Test-Gated Remediation SFT + DPO
        dataset:
          name: Sovereign Code Audit & Bug Remediation SFT + Referee Calibration DPO
          type: krush420/sovereign-code-audit-sft
          split: elitist_120b_teacher
        metrics:
          - type: accuracy
            value: 93.36
            name: Training Token Accuracy (%)
          - type: accuracy
            value: 90.35
            name: Held-Out Eval Mean Token Accuracy (%)
        source:
          name: Sovereign Code Audit SFT & DPO Distillation Pipeline
          url: https://huggingface.co/datasets/krush420/sovereign-code-audit-sft
      - task:
          type: text-retrieval
          name: Multi-Scale Conversational Memory Retrieval (KruschStack L2 Reader)
        dataset:
          name: Official ICLR 2026 BEAM (100K-10M Splits, n=2,000)
          type: Mohammadta/BEAM
          split: 100K
        metrics:
          - type: recall
            value: 92.58
            name: 100K Strict Primary Turn Recall (9-Cat %)
          - type: recall
            value: 93.00
            name: 10M Information Extraction Recall (%)
        source:
          name: Sovereign G4 Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
tags:
  - base_model:adapter:Qwen/Qwen3-Coder-30B-A3B-Instruct
  - krusch-stack
  - mcp
  - agentic-harness
  - lora
  - qlora
  - sft
  - dpo
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
> This adapter is the official **`L2` Local MoE Student** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It was distilled from trajectories generated, AST-grounded, and unit-test verified inside the KruschStack 3-Tier MCP harness so that a fast local 30B MoE model (`3B` active parameters) can perform Principal-level 11-phase code audits, Lego-First refactoring, and test-gated bug remediations across the KruschStack ecosystem.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,049` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`316,981+` Verified Records, `68/68` Gates)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
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
3. **Lego-First Idiom Synthesis & Rule-of-3 Anti-Bloat Gate (`INV-LEGO-01..16`):** Assembling services from `@krusch/toolkit` v1.5 and `krusch_toolkit` v1.3 bricks (`openSqliteDb`, `escapeFts5Query`, `fuseRrfResults`, `safeFetch`, `createRateLimiter`, `createApiServer`, `defineMcpTool`), nominating `>= 3`-repo clusters via `CANDIDATE_MEETS_RULE_OF_3`, and rejecting `< 3`-repo utilities as `BELOW_THRESHOLD_DO_NOT_EXTRACT`.
4. **2-Stage SFT + Balanced DPO Referee Calibration (`trl.DPOTrainer`):** Stage-2 Direct Preference Optimization (`beta=0.1`, `12` balanced Precision + Recall + Architecture pairs) anchored to the Stage-1 SFT adapter so the model emits calibrated `CLEAN_PASS` (`"findings": []`) verdicts on compliant code while maintaining `100%` P0/P1 recall on genuine security/correctness defects.
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

### 2. Run Locally with Ollama (`Modelfile`, `adapter.gguf` & `~6.8s` GGUF Fusion Included)

```bash
git clone https://huggingface.co/krush420/qwen3-coder-30b-sre-lora
cd qwen3-coder-30b-sre-lora
ollama pull qwen3-coder-30b-moe:latest

# Ollama >= 0.40: Fuse rank-16 attention LoRA into Q4_K/Q6_K base GGUF in ~6.8s (zero runtime GEMV overhead)
BASE_BLOB="$(ollama show --modelfile qwen3-coder-30b-moe:latest | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf ./qwen3-coder-30b-sre.gguf
ollama create qwen3-coder-30b-sre:latest -f Modelfile && rm -f ./qwen3-coder-30b-sre.gguf
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

## 🔬 Training Hyperparameters & 2-Stage Calibration Metrics

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`; peak allocated VRAM `59.34 GB`)
- **Base Model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit NF4 attention + frozen BF16 3D MoE experts)
- **LoRA Config**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`13,369,344` trainable attention parameters across 48 layers; `51.03 MB` GGUF v3 adapter)
- **Wave-1/2 Seed SFT**: Loss converged `0.731` $\rightarrow$ `0.214`; token accuracy `93.36%` (`eval_loss=0.3358`, `eval_mean_token_accuracy=90.35%`)
- **Wave-3 Stage-1 Calibration SFT (`8192` ctx, `320` complete trajectories, `lr=8e-5`, `300.55s`)**: Loss `1.0523` $\rightarrow$ `0.9265` (`train_loss=0.9971`); token accuracy `73.97%` $\rightarrow$ `75.42%`
- **Wave-3 Stage-2 Referee DPO (`12` balanced pairs $\times$ `3` epochs, `beta=0.1`, `lr=1e-5`, `49.57s`)**: Loss `0.7014` $\rightarrow$ `0.4457` (step 8) / `0.6850` (step 9), `train_loss=0.6456`, reward margins `0.000` $\rightarrow$ `+0.6325` (`100%` preference accuracy on calibrated Referee & Lego-First batches)
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`warmup_ratio=0.05`)
- **Frameworks**: `peft 0.21.2`, `trl 1.17.1`, `transformers 5.22.1`, `torch 2.11.0+cu130`, `datasets 5.1.0`

---

## 🏆 KruschStack Ecosystem Hermetic Benchmark Scorecard

All underlying per-query JSON audit logs and SHA-256 digests (`68/68` integrity gates across `316,981+` queries) are published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** alongside the cross-stack report (**[*How an Open-Weights Sovereign Stack Stacks Up*](https://krusch.dev/articles/how-krusch-stack-stacks-up)**):

| Pillar | Public Benchmark | Verified Sovereign Score |
| :--- | :--- | :--- |
| **Cascade Router (`L1–L3`)** | **Martian RouterBench** (`73K` rows), **ACL 2026 LLMRouterBench** (`12,566` Qs) & **RouterArena** (`8,400` Qs) | **🥇 `#1` RouterBench ID (`0.7488`/`0.7695`, `-72.63%` cost @ `98%` `GPT-4`) & OOD (`0.7268`/`0.7455` > `ZeroRouter`)** · **`#1` LLMRouterBench Pareto (`0.4269`, `-83.95%` cost)** · **RouterArena CI (`0.7883`, PR #169 open)** |
| **Working Memory (`v2.0.0`)** | **ICLR 2026 BEAM** (`2,000/2,000` Qs, `100K`–`10M`) | **`92.58%` (`100K`) & `93.00%` (`10M` IE) recall** · **`72.86%` `1M` E2E QA** (beats `past.dev` on `4/4` CR splits) |
| **Code AST & Hybrid RRF** | **ACL 2025 CoIR** (`10/10` tasks, `2.39M` docs) & **RepoBench-R** (`47,998` Py & Java Qs) | **🥇 `#1` CoIR Macro `NDCG@10` (`56.40%` > `Voyage-Code-002` `56.26%`)** · **`19.12%` RepoBench Macro `Acc@1`** · **`68.58%` `Acc@5`** |
| **2PC Coding Harness & Defense** | **Aider Polyglot** (`225/225`), **BIRD-SQL** (`500/500`), **BFCL v3 Live** (`2,251`) & **AgentDojo** (`1,046`) | **`51.56%` Polyglot `Pass@2`** · **`40.20%` BIRD-SQL EX (`+14.20 pp`)** · **`66.28%` BFCL (`-80.67%` schema toks)** · **`0.00%` AgentDojo ASR (`0/949` compromises)** |
| **Evidentiary Legal Spine** | **LegalBench-RAG** (`714` docs, `6,889` span Qs) | **`26.74%` Macro F1 (`k=1`) / `25.81%` F1 & `44.72%` R (`k=2`)** — **`33.95%` `MAUD` Recall@2** > OpenAI `3-large` + Cohere `Recall@64` (`28.28%`) |