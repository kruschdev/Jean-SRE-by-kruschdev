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
          name: Multi-Turn Long-Term Memory & Simulated App Tool Execution
        dataset:
          name: DolphinBench (600 Tasks across 3 Personas / 1.5M Tokens, Official gpt-5.6-sol Judge)
          type: mem0ai/dolphinbench
          split: test
        metrics:
          - type: accuracy
            value: 43.00
            name: Official gpt-5.6-sol Pass Rate (258/600 %, Wave-4 + Grounded Harness v0.3)
          - type: accuracy
            value: 42.67
            name: Official gpt-5.6-sol Pass Rate (256/600 %, Wave-4 Std. Harness)
          - type: accuracy
            value: 85.73
            name: Deterministic Mechanical Tool Checks (1,123/1,310 %)
        source:
          name: Sovereign G4 & DolphinBench Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
      - task:
          type: text-generation
          name: AST-Grounded Code Audit, Lego-First Synthesis & Tool-Calling SFT + DPO
        dataset:
          name: Sovereign Code Audit, Lego-First & Agentic Tool-Calling SFT + DPO
          type: krush420/sovereign-code-audit-sft
          split: elitist_120b_teacher
        metrics:
          - type: accuracy
            value: 100.0
            name: Stage-2 KL-Anchored DPO Preference Accuracy (%)
          - type: accuracy
            value: 93.36
            name: Seed Code-Audit Token Accuracy (%)
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
  - tool-calling
  - dolphinbench
  - lora
  - qlora
  - sft
  - dpo
  - code-audit
  - bug-remediation
  - sre
  - ollama
  - gguf
  - transformers
  - trl
---

# Qwen3-Coder-30B-A3B — KruschStack Wave-4 Unified Agentic SRE & Tool-Caller LoRA (`L2` Local MoE Specialist)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![DolphinBench](https://img.shields.io/badge/DolphinBench%20600-43.00%25%20(258%2F600%20%231%20Qwen%2FMoE)-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Base Model](https://img.shields.io/badge/Base-Qwen3--Coder--30B--A3B-blue.svg)](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> This Wave-4 Unified adapter is the official **`L2` Local MoE Coding & Tool-Calling Specialist (`3.3B` active parameters, `124+ tok/s` on 22–24 GB VRAM)** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). Combining AST-grounded SRE code auditing, `@krusch/toolkit` Lego-First synthesis, and multi-turn MCP/Workspace tool-calling with a role-neutral `Modelfile`, it **reaches `43.00%` (`258/600`, `#1` across all tested Qwen & MoE open-weights models) with `Grounded Harness v0.3` and `42.67%` (`256/600`) on `Std. Harness` vs. `42.00%` (`252/600`) `Stock` on the full 600-task [DolphinBench](https://dolphinbench.ai/) suite** (`85.73%` [`1,123/1,310`] inner mechanical checks at `$0.00` inference cost; pairing with `+ INV-STACK-10 2PC` cuts duplicate MCP calls by `-95.2%` [`42 -> 2`], and the 3-harness `30B Wave-4 Oracle@3` reaches **`58.50%` [`351/600`]** / `Qwen3-30B Oracle@5 = 61.17%` [`367/600`]) while achieving **`100%` Wave-4 Referee `CLEAN_PASS` + P0/P1 calibration**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,131+` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `49` Audit Logs = `48` Staged Raw JSON + `1` PR #169 Live Receipt across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)
- **🏆 Companion 31B Dense Reasoning QLoRA (`#1 Open-Weights` `57.33%` DolphinBench)**: [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (Wave-6 for `google/gemma-4-31b-it`)
- **🪶 Companion 8B Edge Agent LoRA (`L1.5` Tier)**: [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) (Built with Meta Llama 3 for `NousResearch/Hermes-3-Llama-3.1-8B`)
- **🧠 Companion 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`qwen3-coder-30b-sre-lora` reasons over structured tool outputs from the **KruschStack Unified MCP Gateway** and acts as the **`L2` Local MoE Specialist** in [`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router):

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        KRUSCH-STACK UNIFIED GATEWAY                        │
│               `npx krusch-stack mcp`  (39 Annotated MCP Tools)             │
├──────────────────────┬───────────────────────────┬─────────────────────────┤
│ 1. WORKING MEMORY    │ 2A. CODEBASE AST & GIT    │ 2B. OS PAGE MEMORY      │
│ krusch-context-mcp   │ krusch-git                │ krusch-fs               │
│ • retrieve / remember│ • search_symbols          │ • page_tree / read_page │
│ • revise / nudge     │ • dependency_graph        │ • locate / search       │
│ • INV-12 span drift  │ • read_tree / read_blob   │ • remember_on_page      │
├──────────────────────┼───────────────────────────┼─────────────────────────┤
│ 3. 2PC & GROUNDED    │ 4. FRONTEND DESIGN MCP    │ 5. WORKSPACE HUB        │
│ krusch-harness / GH  │ krusch-frontend-mcp       │ g-krusch                │
│ • run / apply_diff   │ • generate_color_palette  │ • get_agent_steering    │
│ • Turn-0 grounding   │ • css_to_tailwind         │ • search_drive_rag      │
│ • Defect-only stop   │ • canonical_templates     │ • write (.gdoc/.slides) │
├──────────────────────┴───────────────────────────┴─────────────────────────┤
│ 6. COGNITIVE CASCADE ROUTER (`krusch-cascade-router`)                      │
│ • <25µs L1 -> [L2 MoE: qwen3-coder-30b-sre] -> L2.5 gemma4-31b / 120B      │
└────────────────────────────────────────────────────────────────────────────┘
```

### What Wave-4 Unified SFT + DPO Adds Over Wave-3
1. **Role-Neutral `Modelfile` + Multi-Turn `<tool_call>` Agency (`agentic_tool_calling_sft` & `dpo`):** Removes Wave-3's hardcoded auditor `SYSTEM` prompt and trains directly on structured `tools` + `assistant.tool_calls` + `role: "tool"` chains covering read-before-write discovery (`list_*` -> `update_*`), temporal state supersession, and decisive side-effect execution.
2. **Gemini 4 Argon (`gemini-4-argon`) Golden Remediations (`argon_golden_remediations`):** Multi-file root-cause bug fixes staged through `krusch-harness` and verified against live unit test suites (`node:test` / `pytest`, exit code `0`).
3. **`openai/gpt-oss-120b` Elitist Code Audit Trajectories (`elitist_120b_teacher`):** 11-phase AST-grounded codebase audits using `krusch-git` (`search_symbols`, `dependency_graph`, `read_blob`) with verbatim `file:startLine-endLine` citations, false-positive refutation (`NOT FLAGGED`), and Zero-Finding convergence (`CLEAN_PASS`).
4. **Lego-First Idiom Synthesis & Rule-of-3 Anti-Bloat Gate (`INV-LEGO-01..34`):** Assembling services from `@krusch/toolkit` and `krusch_toolkit` bricks (`createApiServer`, `sseResponse`, `defineMcpTool`, `createPgPool`, `openSqliteDb`, `safeFetch`, `createRateLimiter`), nominating `>= 3`-repo clusters via `CANDIDATE_MEETS_RULE_OF_3`, and rejecting `< 3`-repo utilities as `BELOW_THRESHOLD_DO_NOT_EXTRACT`.
5. **KL-Anchored Stage-2 Preference Calibration (`48` DPO Pairs, `beta=0.1`):** Combines Referee `CLEAN_PASS` precision + P0/P1 recall pairs with `action_over_prose`, `bounded_search_no_loop`, and `benign_execution_no_false_refusal` tool-action pairs (`100.0%` preference accuracy).

---

## ⚡ Quickstart

### 1. Run Locally with Ollama (`Modelfile`, `adapter.gguf` & `~6.8s` GGUF Fusion Included)

```bash
git clone https://huggingface.co/krush420/qwen3-coder-30b-sre-lora
cd qwen3-coder-30b-sre-lora
ollama pull qwen3-coder:30b

# Fuse rank-16 attention LoRA into Q4_K_M base GGUF in ~6.8s (zero runtime GEMV overhead)
BASE_BLOB="$(ollama show --modelfile qwen3-coder:30b | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf ./qwen3-coder-30b-sre.gguf
ollama create qwen3-coder-30b-sre:latest -f Modelfile && rm -f ./qwen3-coder-30b-sre.gguf
ollama run qwen3-coder-30b-sre:latest

# Option B (Zero-extra-disk in-place fusion on ephemeral GPU VMs — preserves upstream RENDERER/PARSER):
# python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf "$BASE_BLOB"
# ollama cp qwen3-coder:30b qwen3-coder-30b-sre:latest
```

> **Ollama `0.17+` Tool-Template Note**: `fuse_lora_into_gguf.py` preserves Qwen3-Coder's native 6,211-character Jinja `tokenizer.chat_template` directly inside the fused GGUF metadata (`capabilities: ["tools", "completion"]` for structured `/api/chat` MCP & OpenAI tool-calling), while [`Modelfile`](./Modelfile) stays strictly role-neutral (`SYSTEM: None`, `num_ctx 16384`) so caller system prompts and tool schemas bind without auditor-prompt collisions.

### 2. Transformers + PEFT

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

### 3. vLLM Multi-LoRA Serving

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --enable-lora \
  --lora-modules code-auditor=krush420/qwen3-coder-30b-sre-lora
```

---

## 🔬 Wave-4 Unified Training Hyperparameters & Dense-vs-MoE Routing Guide

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, `sm_120`; peak allocated VRAM `59.34 GB`, `832.63s` total wall time)
- **Base Model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (`30.5B` total / `3.3B` active MoE parameters; 4-bit NF4 attention + frozen BF16 3D MoE experts)
- **LoRA Config**: `r=16`, `lora_alpha=32` (`alpha/r = 2.0`), `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`13,369,344` trainable parameters across 48 layers; `51.03 MB` GGUF v3 adapter)
- **Stage-1 Unified SFT (`8192` ctx, `420` trajectories, `lr=1.2e-4`, `380.36s`)**: Loss converged `1.7188` -> `1.4253` (`train_loss=1.3884`), token accuracy `67.00%` -> `69.91%` across native `<tool_call>` chains and code-audit trajectories
- **Stage-2 Unified DPO (`48` balanced Referee + Tool-Action pairs x `3` epochs, `beta=0.1`, `lr=1e-5`, `253.49s`)**: Loss `0.7014` -> `0.5238` (`train_loss=0.5902`), reward margin `0.000` -> `+0.4031`, **`100.0%` preference accuracy**
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`warmup_ratio=0.05`)

### 5-Run `Qwen3-Coder-30B-A3B` Ablation & When to Route to `gemma4-31b-sre-lora`

| `Qwen3-Coder-30B-A3B` Configuration (`$0.00` Inference) | Official `gpt-5.6-sol` (`600`) | `Morgan` / `Alex` / `Riley` | Conjunctive Mech | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **`Wave-4` QLoRA+DPO + `Grounded Harness v0.3` *(This Repo)*** | **`43.00%` (`258`)** 🏆 | **`125` (`62.5%`)** / `63` / `70` | **`613 / 1,364` (`44.94%`)** | **`#1` Qwen/MoE Open-Weights** (`+14` on `Morgan` vs. Wave-4 `Std.`) |
| **`Wave-4` QLoRA+DPO + `Std. Harness` *(This Repo, Queued)*** | **`42.67%` (`256`)** | `111` / `70` / **`75` (`37.5%`)** | `601 / 1,368` (`43.93%`) | `15.72s` median latency, `85.73%` (`1,123/1,310`) inner mech checks |
| `Wave-4` QLoRA+DPO + `INV-STACK-10 2PC` | `42.00%` (`252`) | `114` / **`71`** / `67` | `581 / 1,380` (`42.10%`) | `-95.2%` duplicate MCP calls (`42 -> 2`); **`Wave-4 Oracle@3 = 58.50%` (`351`)** |
| `Stock` `Qwen3-Coder-30B-A3B-Instruct` (`Std. Harness`, Queued) | `42.00%` (`252`) | `117` / **`71`** / `64` | `580 / 1,370` (`42.34%`) | Baseline (`Wave-4` lifts `Riley` `+5.50 pp`) |
| `Wave-6` Gap-Search QLoRA+DPO + `Grounded Harness v0.3` | `38.50%` (`231`) | `106` / `65` / `60` | `550 / 1,397` (`39.37%`) | Multi-query gap-search crowds non-thinking `3.05B` active attention |
| **`Oracle@5` `Qwen3-Coder-30B` Family Union (`$0.00`)** | **`61.17%` (`367`)** | **`155` (`77.5%`) / `109` (`54.5%`) / `103` (`51.5%`)** | — | `61.17%` latent retrieval ceiling across `3.3B`-active MoE passes |

> **Routing Rule (`L2` MoE vs. `L2.5` Dense Thinking)**: Deploy **Wave-4 `qwen3-coder-30b-sre-lora`** for high-throughput (`124+ tok/s`) repository code synthesis, AST audits, and direct MCP tool execution on 22–24 GB VRAM; escalate multi-hop temporal memory synthesis and contradiction-heavy personal-assistant workflows to dense thinking **[`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora)** (**`57.33%` [`344/600`]**).

---

## 🏆 KruschStack Ecosystem Hermetic Benchmark Scorecard (`75 / 75` Gates, `333,000+` Queries/Traces, `49` Audit Logs)

All `49` audit logs (`48` staged raw JSON logs + `1` `RouteWorks/RouterArena` PR #169 live receipt bundle; `63` dataset files across `14` splits, `75/75` gates + `27/27` validated DolphinBench archives) are published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** alongside Paper #27 (**[*Return on Memory: `#1` Overall (`91.17%`), Model-Matched `GPT 5.6 Luna` (`81.50%` @ `$1.35`) & `#1` Open-Weights (`57.33%`) on Official DolphinBench*](https://krusch.dev/articles/return-on-memory-dolphinbench)**) and the cross-stack report (**[*How an Open-Weights Sovereign Stack Stacks Up*](https://krusch.dev/articles/how-krusch-stack-stacks-up)**):

| Pillar | Public Benchmark | Verified Sovereign Score |
| :--- | :--- | :--- |
| **Personal Agent Memory & Tools** | **Official DolphinBench** (`600` tasks × `27` runs = `16,200` traces, `6` families, `openai/gpt-5.6-sol` Judge, `8/8` flagship queued) | **`#1 Overall` `91.17%` (`547/600` @ `$24.54`) Argon Low** (`Oracle@27 = 97.83%`; `$0.00` `Oracle@17 = 83.17%`) · **Model-Matched `GPT 5.6 Luna (High, + Grounded)` `81.50%` (`489` @ `$1.35`, `30.03s`, `+10.83 pp`, `71.1x` cheaper) & `Luna (Low, Std.)` `76.67%` (`@ $1.10`, `87.4x` cheaper; `Luna Oracle@5 = 91.83%`)** · **`#1 Open-Weights` `57.33%` (`344/600`) [`gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (`Wave-6 + Grounded v0.3`) @ `$0.00`** (`Gemma 4 Oracle@6 = 71.83%`) · **`43.00%` (`258/600`) `30B` Wave-4 (`+ Grounded v0.3`), `42.67%` (`256/600`) `Std.` & `42.00%` (`+ 2PC`, `-95.2%` dup MCP, `Wave-4 Oracle@3 = 58.50%`, `30B Oracle@5 = 61.17%`)** |
| **Security & Policy Floor** | **ETH SpyLab AgentDojo v1.2.2** (`1,046` episodes, `949` attacks) | **`0.00%` Targeted ASR (`0/949`, `100%` Defense)** · **`69.07%` Benign** (Spotlighting) & **`83` hardline floor blocks** |
| **Code AST & Hybrid RRF** | **ACL 2025 CoIR** (`2.39M` docs / `162K` Qs) & **RepoBench-R** (`47,998` Qs) | **🥇 `#1` CoIR 10-Task `NDCG@10` (`56.40%`, `73.88%` `R@10` > `Voyage-Code-002` `56.26%`)** · **RepoBench-R `68.58%` `Acc@5`** (beats `CodeBERT` & `UniXcoder`) |
| **SQL State & Deferred MCP** | **BIRD-SQL Mini-Dev** (`500` Qs) & **UC Berkeley BFCL v3 Live** (`2,251` Qs) | **`26.0% → 40.2%` SQL EX (`1.55x`, `100%` valid SQL)** · **`59.9% → 66.3%` BFCL Micro & `84.7%` Irrelevance** at **`-80.7%` schema tokens** |
| **Cascade Router (`L1–L3`)** | **Martian RouterBench** (`73K`) · **ACL 2026 LLMRouterBench** (`12.5K`) · **RouterArena** (`8.4K`) | **🥇 `#1` RouterBench ID/OOD `AIQ` (`-72.6%` cost @ `98%` `GPT-4`)** · **🥇 `#1` LLMRouterBench Pareto (`0.4269`)** · **RouterArena Official CI Run `#780` `0.7263` (`#13 / 34`, `74.44%` @ `$0.3802/1K`, `91.43%` Rob., `0` Abnormal, PR #169 open)** |
| **2PC Harness, Memory & Legal** | **Aider Polyglot** (`225/225`) · **ICLR 2026 BEAM** (`2,000` Qs) · **LegalBench-RAG** (`6,889` Qs) | **`19.56% → 51.56%` `Pass@2` (`2.64x`, ties `Claude 3.5 Sonnet`)** · **`92.58%` `100K` & `93.00%` `10M` IE Recall** · **`26.74%` Legal Macro F1** |