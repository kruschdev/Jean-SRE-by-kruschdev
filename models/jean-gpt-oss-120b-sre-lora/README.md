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
          type: text-generation
          name: Multi-Turn Long-Term Memory & Simulated App Tool Execution
        dataset:
          name: DolphinBench (600 Tasks across 3 Personas / 1.5M Tokens, Official gpt-5.6-sol Judge)
          type: mem0ai/dolphinbench
          split: test
        metrics:
          - type: accuracy
            value: 39.50
            name: Official gpt-5.6-sol Pass Rate (237/600 %)
          - type: accuracy
            value: 78.85
            name: Deterministic Mechanical Tool Checks (1,033/1,310 %)
        source:
          name: Sovereign G4 & DolphinBench Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
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
  - tool-calling
  - dolphinbench
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

# gpt-oss-120b — KruschStack Wave-4 Unified Agentic SRE & Tool-Caller LoRA (`L2.5` Teacher)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![DolphinBench](https://img.shields.io/badge/DolphinBench%20600-39.50%25%20(237%2F600)-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Base Model](https://img.shields.io/badge/Base-openai%2Fgpt--oss--120b-blue.svg)](https://huggingface.co/openai/gpt-oss-120b)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> This Wave-4 Unified adapter is the official **`L2.5` High-VRAM Burst Teacher & Deep Referee Auditor** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). Fine-tuning [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MoE parameters) via single-GPU MXFP4 QLoRA SFT + KL-Anchored `trl.DPOTrainer` preference calibration with native Harmony `<|channel|>commentary` tool routing and a role-neutral `Modelfile`, Wave-4 **eliminates auditor prose-refusal drift and beats `Stock` `gpt-oss:120b` on the full 600-task [DolphinBench](https://dolphinbench.ai/) suite (`39.50%` [`237/600`] `Std. Harness` vs. `39.17%` [`235/600`] `Stock` under official `gpt-5.6-sol` grading, `+7` tasks on `Morgan`)** while preserving **`92.37%` Official ICLR 2026 BEAM `100K` Recall** and **`100%` Referee `CLEAN_PASS` + P0/P1 calibration**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,131+` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `49` Audit Logs = `48` Staged Raw JSON + `1` PR #169 Live Receipt across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)
- **🏆 Companion 31B Dense Reasoning QLoRA (`#1 Open-Weights` `57.33%` DolphinBench)**: [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (Wave-6 for `google/gemma-4-31b-it`)
- **⚡ Companion 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `43.00%` w/ Grounded v0.3)
- **🪶 Companion 8B Edge Agent LoRA (`L1.5` Tier)**: [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) (Built with Meta Llama 3 for `NousResearch/Hermes-3-Llama-3.1-8B`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`gpt-oss-120b-sre-lora` serves as the **`L2.5` 120B Burst Teacher & Blind Spec Referee** inside **[`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router)** (`<25µs` routing latency):

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
│ 3. 2PC & GROUNDED    │ 4. FRONTEND DESIGN MCP    │ 5. WORKSPACE HUB        │
│ krusch-harness / GH  │ krusch-frontend-mcp       │ g-krusch                │
│ • run (stages diff)  │ • generate_color_palette  │ • get_agent_steering    │
│ • apply_diff (2PC)   │ • css_to_tailwind         │ • search_drive_rag      │
│ • Defect-only stop   │ • canonical_templates     │ • write (.gdoc/.slides) │
├──────────────────────┴───────────────────────────┴─────────────────────────┤
│ 6. COGNITIVE CASCADE ROUTER (`krusch-cascade-router`)                      │
│ • <25µs L1 fast-path -> L2 30B Student -> [L2.5: gemma4-31b / 120B]        │
└────────────────────────────────────────────────────────────────────────────┘
```

### What Wave-4 Unified SFT + DPO Adds Over Wave-3
1. **Native Harmony `<|channel|>commentary` Function-Calling Preservation (`agentic_tool_calling_sft` & `dpo`):** Renders multi-turn `tools` + `assistant.tool_calls` + `role: "tool"` trajectories through `gpt-oss-120b`'s native Harmony chat template and removes Wave-3's hardcoded auditor `SYSTEM` prompt, eliminating Wave-3's `30%` prose-only refusal regression on operational tasks and lifting `list_*` read-before-write chains (`179 -> 202`) to beat `120B` Stock on DolphinBench (`237/600` vs. `235/600`).
2. **Gemini 4 Argon (`gemini-4-argon`) Golden Remediations (`argon_golden_remediations`):** Multi-file root-cause bug fixes staged through `krusch-harness` and verified against live unit test suites (`node:test` / `pytest`, exit code `0`).
3. **Elitist Rejection-Sampled 11-Phase Code Audits (`elitist_120b_teacher`):** AST-grounded codebase audits using `krusch-git` (`search_symbols`, `dependency_graph`, `read_blob`) with verbatim `file:startLine-endLine` citations, false-positive refutation (`NOT FLAGGED`), and Zero-Finding convergence (`CLEAN_PASS`).
4. **Lego-First Idiom Synthesis & Rule-of-3 Anti-Bloat Gate (`INV-LEGO-01..34`):** Assembling services from `@krusch/toolkit` and `krusch_toolkit` (`fastapi_app`, `mcp`, `pg`, `security`), nominating `>= 3`-repo clusters via `CANDIDATE_MEETS_RULE_OF_3`, and rejecting `< 3`-repo utilities as `BELOW_THRESHOLD_DO_NOT_EXTRACT`.
5. **KL-Anchored Stage-2 Preference Calibration (`48` DPO Pairs, `beta=0.1`):** Balances Referee `CLEAN_PASS` precision + P0/P1 recall pairs with `action_over_prose`, `bounded_search_no_loop`, and `benign_execution_no_false_refusal` tool-action pairs (`100.0%` preference accuracy).

---

## ⚡ Quickstart

### 1. Run with Ollama (`Modelfile`, `adapter.gguf` & `~1.7s` GGUF Fusion Included)

```bash
git clone https://huggingface.co/krush420/gpt-oss-120b-sre-lora
cd gpt-oss-120b-sre-lora
ollama pull gpt-oss:120b

# Option A (Non-destructive — keeps stock gpt-oss:120b untouched; requires ~65 GB temp disk):
BASE_BLOB="$(ollama show --modelfile gpt-oss:120b | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf ./gpt-oss-120b-sre.gguf
sed -i 's|^FROM .*|FROM ./gpt-oss-120b-sre.gguf|' Modelfile
ollama create gpt-oss-sre:120b -f Modelfile && rm -f ./gpt-oss-120b-sre.gguf
ollama run gpt-oss-sre:120b

# Option B (Zero-extra-disk atomic in-place fusion for ephemeral cloud GPU VMs):
# python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf "$BASE_BLOB"
# ollama create gpt-oss-sre:120b -f Modelfile
```

### 2. Transformers + PEFT

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

### 3. vLLM Multi-LoRA Serving

```bash
vllm serve openai/gpt-oss-120b \
  --enable-lora \
  --lora-modules code-auditor=krush420/gpt-oss-120b-sre-lora
```

---

## 🔬 Wave-4 Single-GPU MXFP4 2-Stage Training Metrics

- **Hardware**: Single NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`; peak allocated VRAM `74.89 GB`, `1,098.18s` total wall time)
- **Base Model**: `openai/gpt-oss-120b` (`116.8B` total / `5.1B` active MoE parameters; 36 layers of 128 native 4-bit MXFP4 routed experts kept frozen in `~61.8 GB` VRAM while gradients flow through `attn_implementation="eager"` BF16 Pre-LN residual stream into `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Config**: `r=16`, `lora_alpha=32` (`alpha/r = 2.0`), `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`11,943,936` trainable parameters; `45.58 MB` GGUF v3 adapter)
- **Stage-1 Unified SFT (`5120` ctx, `340` trajectories, `lr=5e-5`, `685.08s`)**: Loss converged `6.7804` -> `1.9923` (`train_loss=2.6882`), token accuracy `27.27%` -> `59.47%` across Harmony `<|channel|>commentary` tool calls and SRE Referee audits
- **Stage-2 Unified DPO (`48` balanced Referee + Tool-Action pairs x `3` epochs, `beta=0.1`, `lr=5e-6`, `214.59s`)**: Loss `0.6931` -> `0.4065` (`train_loss=0.5402`), reward margin `0.000` -> `+0.7069`, **`100.0%` preference accuracy**
- **In-VRAM Smoke Verification**: `sre_clean_pass: true` (`CLEAN_PASS` on compliant `@krusch/toolkit/http` server) + `tool_call_emitted: true` & `temporal_override_bound: true`
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`warmup_ratio=0.05`)

---

## 🏆 Hermetic External Benchmark Scorecard (`96 GB` Blackwell G4, `75 / 75` Gates, `49` Audit Logs)

Full SHA-256-pinned logs (`48` staged raw JSON audit logs + `1` `RouteWorks/RouterArena` PR #169 live receipt bundle = `49` audit logs across `333,000+` queries/traces; `63` dataset files across `14` splits) are published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** alongside Paper #27 (**[*Return on Memory: `#1` Overall (`91.17%`), Model-Matched `GPT 5.6 Luna` (`81.50%` @ `$1.35`) & `#1` Open-Weights (`57.33%`) on Official DolphinBench*](https://krusch.dev/articles/return-on-memory-dolphinbench)**) and the cross-stack write-up (**[*How an Open-Weights Sovereign Stack Stacks Up*](https://krusch.dev/articles/how-krusch-stack-stacks-up)**):

| Benchmark | Configuration Under Test | Verified Result | External Leaderboard Context |
| :--- | :--- | :--- | :--- |
| **Official DolphinBench (`600/600` Tasks, `27` Runs / `16,200` Traces, `6` Families)** | `gpt-oss-120b-sre-lora` (`$0.00`), [`gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (`$0.00`), `GPT 5.6 Luna` (`Std.`, `2PC`, & `Grounded`) & `Gemini 4 Argon Low` (`$24.54`) on frozen `0o444` `SovereignMemoryStore` (`openai/gpt-5.6-sol` Judge, `8/8` flagship queued, `27/27` validated) | **`#1 Overall` `91.17%` (`547/600` @ `$24.54`) Argon Low** (`89.17%` Grounded/Med) · **Model-Matched `Luna High (+ Grounded)` `81.50%` (`489` @ `$1.35`, `30.03s`) & `Luna Low (Std.)` `76.67%` (`460` @ `$1.10`)** · **`#1 Open-Weights` `57.33%` (`344`) `gemma4-31b-sre-lora` (Wave-6 + Grounded v0.3) @ `$0.00`** · **`43.00%` (`258`) `30B` Wave-4 (+ Grounded v0.3)** · **`39.50%` (`237`) `120B` Wave-4** (`39.17%` Stock, `$0.00`) | **`+20.50 pp` above `#1` `Mem0` (`70.67%` @ `$96.21`)**; model-matched `Luna High + Grounded` **`+10.83 pp` at `71.1x` lower cost** (`Luna Oracle@5 = 91.83%`); **`Oracle@27 = 97.83%`** (`<= 2.17%` miss), **`$0.00` `Oracle@17 = 83.17%`**, Pareto `($0.00, 57.33%) -> ($1.10, 76.67%) -> ($1.35, 81.50%) -> ($24.54, 91.17%)`. |
| **ETH SpyLab AgentDojo v1.2.2 (`1,046` Eps)** | `openai/gpt-oss-120b` + `delimitUntrustedContent` + Sink Taint Provenance + `createKruschPolicyFloor` | **`0.00%` Targeted ASR (`0/949`)** · **`69.07%` Benign / `69.02%` UUA** (Spotlighting) · **`83` floor blocks** | Eliminates `19.28%` raw ASR (`57.14%` `slack`, `52.78%` `banking`) across `100%` of `important_instructions` attacks with `0.00 pp` benign loss. |
| **ACL 2025 CoIR (`10/10` Tasks, `162,216` Qs)** | [`krusch-git`](https://github.com/kruschdev/krusch-git) Hybrid RRF + `snowflake-arctic-embed2` (`1024d`, `8192` ctx) | **`56.40%` 10-Task Macro `NDCG@10` (`#1`)** · **`50.94%` `MRR@10`** · **`73.88%` `Recall@10`** | Across `2,389,123` unshrunk docs, exceeds all 10 CoIR Table 3 baselines incl. `Voyage-Code-002` (`56.26%`) & `E5-Mistral 7B` (`55.18%`). |
| **BIRD-SQL Mini-Dev (`500/500`) & BFCL v3 (`2,251/2,251`)** | `openai/gpt-oss-120b` + Read-Only `node:sqlite` State Grounding + `createDeferredToolBridge` (`64` MCP tools) | **`26.00% → 40.20%` SQL EX (`1.55x`)** · **`59.88% → 66.28%` BFCL Micro** · **`84.69%` Irrelevance** | `100%` valid SQL (`13` healed in `0.09` extra calls/query); **`-80.67%` fewer MCP schema tokens** (`29` schema repairs healed). |
| **Aider Polyglot (`225/225`)** | `openai/gpt-oss-120b` + [`krusch-coding-harness`](https://github.com/kruschdev/krusch-coding-harness) `2PC` `Verify-on-Stop` | **`Pass@1 = 19.56%` → `Pass@2 = 51.56%` (`116/225`)** (`72` tasks self-healed) | **Ties cloud `Claude 3.5 Sonnet` (`51.6%`)** & **beats `GPT-4o` (`23.1%`)** across C++, Go, Java, JS, Python & Rust at `$0` API cost. |
| **Routing (`RouterBench` `73K` · `LLMRouterBench` `12.5K` · `RouterArena` `8.4K`)** | [`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router) (`KruschCascadeRouter` & SSE Abort) | **`#1` RouterBench ID/OOD `AIQ` (`0.7488` / `0.7695`, `-72.6%` cost)** · **`#1` LLMRouterBench Pareto (`0.4269`)** · **RouterArena CI `0.7263` (`#13/34`, `74.44%` @ `$0.3802/1K`, `91.43%` Rob., `0` Abnormal)** | Breaks Martian's OOD `ZeroRouter` ceiling; outperforms `Azure-Model-Router` (`70.37`) & `GPT-5` (`64.32`, `26.4x` cheaper) in PR #169 Run `#780`; **`-48.0%` TTFT (`880.7ms`)** on live SSE cascade. |
| **ICLR 2026 BEAM (`2,000` Qs) · RepoBench-R (`48K`) · LegalBench-RAG (`6.9K`)** | `gpt-oss-120b-sre-lora` + `krusch-context-mcp` `v2.0.0` + `krusch-git` + `KruschNexus` | **`92.37%–92.58%` `100K` Recall** · **`19.12%` `Acc@1` / `68.58%` `Acc@5`** · **`26.74%` Legal Macro F1** | Beats `past.dev` on Contradiction Resolution (`4/4` splits), `CodeBERT`/`UniXcoder` on Py/Java `cfr`, & OpenAI `3-large` + Cohere `MAUD` `R@64`. |