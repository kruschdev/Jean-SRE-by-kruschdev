---
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
library_name: peft
model_name: qwen3-coder-30b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
  - krush420/sovereign-g4-benchmarks
  - NousResearch/arete-data-pairs
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
          name: AST-Grounded Code Audit, Lego-First Synthesis, Tool-Calling & Non-Refusal Statutory SFT + DPO
        dataset:
          name: Sovereign Code Audit, Lego-First & Agentic Tool-Calling SFT + DPO
          type: krush420/sovereign-code-audit-sft
          split: elitist_120b_teacher
        metrics:
          - type: accuracy
            value: 100.0
            name: Stage-2 KL-Anchored DPO Preference Accuracy (%)
          - type: accuracy
            value: 74.58
            name: Wave-7 Assistant-Only Masked SFT Token Accuracy (%)
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
  - arete
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

# Qwen3-Coder-30B-A3B — KruschStack Wave-7 Unified Agentic SRE, Tool-Caller & Curated Arete Alignment LoRA (`L2` Local MoE Specialist)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![DolphinBench](https://img.shields.io/badge/DolphinBench%20600-43.00%25%20(258%2F600%20%231%20Qwen%2FMoE)-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Throughput](https://img.shields.io/badge/MoE%20Throughput-125.8--131.5%20tok%2Fs%20(5.5x%20vs%20Dense%2027B)-00e5ff.svg)](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora)
[![Base Model](https://img.shields.io/badge/Base-Qwen3--Coder--30B--A3B-blue.svg)](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> This **Wave-7 (`Wave-6 SRE + Curated NousResearch/Arete`)** adapter is my official **`L2` Local MoE Coding, Tool-Calling & Non-Refusal Statutory Specialist (`3.3B` active parameters, `125.8–131.5 tok/s` on 22–24 GB VRAM)** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). Combining AST-grounded SRE code auditing, `@krusch/toolkit` Lego-First synthesis, multi-turn MCP/Workspace tool-calling, and strictly curated zero-defect alignment pairs from [`NousResearch/arete-data-pairs`](https://huggingface.co/datasets/NousResearch/arete-data-pairs) (`Apache-2.0`) with a role-neutral `Modelfile`, it delivers **zero-refusal statutory and security code analysis at `125.8–131.5 tok/s` (`5.4x–5.6x` faster than dense `27B` `NousResearch/Arete-27B` at `23.5 tok/s` on identical dual-GPU hardware)** while preserving Qwen3-Coder-30B-A3B's **`43.00%` (`258/600`, `#1` across all tested Qwen & MoE open-weights models) Official [DolphinBench](https://dolphinbench.ai/) performance** (`58.50%` [`351/600`] `Wave-4 Oracle@3` / `61.17%` [`367/600`] `Qwen3-30B Oracle@5`) and **`100%` Referee `CLEAN_PASS` + P0/P1 calibration**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,341` Zero-Leak SFT + DPO Records across 9 Splits)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `49` Audit Logs = `48` Staged Raw JSON + `1` PR #169 Live Receipt across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)
- **🏆 Companion 31B Dense Reasoning QLoRA (`#1 Open-Weights` `57.33%` DolphinBench)**: [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (Wave-6 for `google/gemma-4-31b-it`)
- **🪶 Companion 8B Edge Agent LoRA (`L1.5` Tier)**: [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) (Built with Meta Llama 3 for `NousResearch/Hermes-3-Llama-3.1-8B`)
- **🧠 Companion 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)

---

## 🏗️ Built With & For the KruschStack 7-Pillar Architecture

`qwen3-coder-30b-sre-lora` reasons over structured tool outputs from the **KruschStack Unified MCP Gateway** and acts as my **`L2` Local MoE Specialist** in [`krusch-cascade-router`](https://github.com/kruschdev/krusch-cascade-router):

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

### What Wave-7 (`Wave-6 SRE + Curated Arete`) SFT + BF16 Ref-Anchored DPO Adds

1. **Curated Zero-Defect `NousResearch/arete-data-pairs` SFT & DPO Alignment (`arete_values_sft` & `arete_values_dpo`):**
   - As documented in the upstream [`NousResearch/arete-data-pairs`](https://huggingface.co/datasets/NousResearch/arete-data-pairs) dataset card (*"Not a clean SFT set. Most accepted rows still carry a factual_error tag and accuracy 2"*), raw ingestion of all `34,878` paired rows would inject `accuracy=2` hallucinations and dilute `Qwen3-Coder-30B-A3B`'s coding and tool-calling precision.
   - Instead, I built a strict quality filter (`harvest_arete_alignment.js`) across all 7 categories (`data-privacy-surveillance`, `democracy-and-governance`, `education-philosophy`, `environmental-strategy`, `freedom-of-expression`, `state-ideology`, `state-vs-free-market-capitalism`) that selects:
     - **`140` Gold SFT Rows (`20` per category)**: `in_sft_accepted=true`, `accuracy=5`, `independence=5`, `helpfulness>=4`, `balance>=4`, `language>=4`, and `0` issue tags (`reasons=[]`).
     - **`42` High-Contrast DPO Preference Pairs (`6` per category)**: Disjoint prompts where the starting Qwen checkpoint exhibited explicit state-aligned refusal, canned evasion, or propaganda framing (`rejected`) while the Inkling teacher produced a rigorous, multi-perspective statutory/policy analysis (`chosen`, `accuracy>=4`, `independence=5`).
2. **Strict ChatML Assistant-Only Loss Masking (`QwenChatMLAssistantOnlyCollator`):** Masks `100%` of `<|im_start|>system`, `<|im_start|>user`, `<tools>`, and `<tool_response>` tokens with `-100` so gradient updates focus exclusively on assistant pre-tool reasoning, `<tool_call>` XML/JSON emission, code patches, and analytical synthesis (`74.58%`–`76.67%` assistant token accuracy across `8,192`-token contexts).
3. **BF16 Reference-Anchored Stage-2 DPO (`156` Pairs, `beta=0.2`, `max_diff_vs_sft = 0.0`):** Snapshots all `384` Stage-1 SFT LoRA tensors (`lora_A` / `lora_B` across all 48 layers) into a frozen `torch.bfloat16` `"ref"` adapter before `trl.DPOTrainer` starts (`max_diff_vs_sft == 0.0`, `ref_l2_norm = 32.1622`), preventing `trl >= 1.0`'s uninitialized/FP32 reference adapter bug from causing KL drift and clean-removing `"ref"` before saving `adapter_model.safetensors` and `adapter.gguf`.
4. **Role-Neutral `Modelfile` + Multi-Turn `<tool_call>` Agency (`agentic_tool_calling_sft` & `dpo`):** Preserves Qwen3-Coder's native `6,211`-character Jinja `tools` template (`capabilities: ["tools", "completion"]`) without a hardcoded auditor `SYSTEM` prompt, supporting read-before-write discovery (`list_*` -> `update_*`), partial-briefing gap search, temporal state supersession, and `@krusch/toolkit` Lego-First synthesis (`INV-LEGO-01..34`).

---

## ⚡ Wave-7 MoE vs. Dense `Arete-27B` Benchmark (`22.5 GB` Dual-GPU VRAM)

On identical dual-GPU hardware (`22.5 GB` NVLink VRAM, `Q4_K_M` quantization, `16,384` context), fusing Wave-7 `adapter.gguf` into `Qwen3-Coder-30B-A3B-Instruct` (`3.3B` active parameters per token) achieves **`5.4x–5.6x` higher generation throughput** than dense `27B` [`NousResearch/Arete-27B`](https://huggingface.co/NousResearch/Arete-27B) while matching its zero-refusal statutory and security analysis:

| Model (`Q4_K_M`, `16K` Context) | Architecture (Active / Total Params) | Statutory Non-Refusal Audit *(PRC Intelligence Law Art. 7 & DSL Art. 31/35 vs. State Framing)* | Cryptographic Security Remediation *(Constant-Time `crypto.timingSafeEqual` HMAC Guard)* | Speedup vs. Dense `27B` |
| :--- | :---: | :---: | :---: | :---: |
| **`qwen3-coder-30b-sre-lora` (`Wave-7` Fused MoE)** | **`3.3B` active** / `30.5B` total MoE | **`125.8 tok/s`** (`1.68s` wall · `0` refusals · cites Art. 7, Art. 14 & DSL Art. 21/31) | **`131.5 tok/s`** (`2.19s` wall · constant-time `Buffer` + length guard) | **`5.4x – 5.6x` faster** ⚡ |
| `NousResearch/Arete-27B` (`Q4_K_M` Dense) | `27.0B` active / `27.0B` total Dense | `23.5 tok/s` (`9.68s` wall · `0` refusals · cites Art. 7 & DSL Art. 31/35) | `23.4 tok/s` (`10.29s` wall · constant-time `Buffer` + length guard) | `1.0x` baseline |

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
ollama show qwen3-coder:30b --modelfile | sed 's|^FROM .*|FROM ./qwen3-coder-30b-sre.gguf|' > Modelfile.fused
ollama create qwen3-coder-30b-sre:latest -f Modelfile.fused && rm -f ./qwen3-coder-30b-sre.gguf Modelfile.fused
ollama run qwen3-coder-30b-sre:latest
```

> **Ollama `0.17+` Tool-Template Note**: `fuse_lora_into_gguf.py` preserves Qwen3-Coder's native `6,211`-character Jinja `tokenizer.chat_template` directly inside the fused GGUF metadata (`capabilities: ["tools", "completion"]` for structured `/api/chat` MCP & OpenAI tool-calling), while [`Modelfile`](./Modelfile) stays strictly role-neutral (`SYSTEM: None`, `num_ctx 16384`) so caller system prompts and tool schemas bind without auditor-prompt collisions.

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

## 🔬 Wave-7 Unified Training Hyperparameters & Telemetry

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (`96 GB` VRAM, `sm_120`; peak allocated VRAM `58.77 GB`, `1,107.92s` total wall time / `2.74` Colab Compute Units)
- **Base Model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (`30.5B` total / `3.3B` active MoE parameters; 4-bit NF4 attention + frozen BF16 3D MoE experts)
- **LoRA Config**: `r=16`, `lora_alpha=16` (`alpha/r = 1.0`), `lora_dropout=0.05`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (`13,369,344` trainable parameters across 48 layers / `384` LoRA tensors; `51.03 MB` GGUF v3 adapter)
- **Stage-1 Assistant-Only Masked SFT (`8192` ctx, `560` trajectories = `420` SRE/Lego/Tool-Calling + `140` Gold Arete SFT, `1` epoch, `lr=5e-5`, `383.8s`)**: Loss converged `1.9894` -> **`1.2757`** (`train_loss=1.4855`), mean assistant token accuracy **`74.58%`–`76.67%`** across native `<tool_call>` chains, code-audit remediations, and statutory analyses
- **Stage-2 BF16 Reference-Anchored DPO (`156` preference pairs = `114` Referee + Tool-Action + `42` High-Contrast Arete DPO x `2` epochs, `beta=0.2`, `lr=5e-6`, `526.73s`)**: All `384` reference adapter tensors verified in `torch.bfloat16` (`max_diff_vs_sft = 0.0`, `ref_l2_norm = 32.1622`); loss converged `0.7018` -> **`0.5561`** (`train_loss=0.6246`), reward margin `0.000` -> **`+0.3664`** (peak `+0.6583`), reward accuracy **`75.0%`–`100.0%`**
- **Optimizer**: `paged_adamw_8bit`, cosine LR schedule (`warmup_ratio=0.05`)

### `Qwen3-Coder-30B-A3B` DolphinBench Ablation & When to Route to `gemma4-31b-sre-lora`

| `Qwen3-Coder-30B-A3B` Configuration (`$0.00` Inference) | Official `gpt-5.6-sol` (`600`) | `Morgan` / `Alex` / `Riley` | Conjunctive Mech | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **`Wave-4` QLoRA+DPO + `Grounded Harness v0.3` *(This Repo)*** | **`43.00%` (`258`)** 🏆 | **`125` (`62.5%`)** / `63` / `70` | **`613 / 1,364` (`44.94%`)** | **`#1` Qwen/MoE Open-Weights** (`+14` on `Morgan` vs. Wave-4 `Std.`) |
| **`Wave-4` QLoRA+DPO + `Std. Harness` *(This Repo, Queued)*** | **`42.67%` (`256`)** | `111` / `70` / **`75` (`37.5%`)** | `601 / 1,368` (`43.93%`) | `15.72s` median latency, `85.73%` (`1,123/1,310`) inner mech checks |
| `Wave-4` QLoRA+DPO + `INV-STACK-10 2PC` | `42.00%` (`252`) | `114` / **`71`** / `67` | `581 / 1,380` (`42.10%`) | `-95.2%` duplicate MCP calls (`42 -> 2`); **`Wave-4 Oracle@3 = 58.50%` (`351`)** |
| `Stock` `Qwen3-Coder-30B-A3B-Instruct` (`Std. Harness`, Queued) | `42.00%` (`252`) | `117` / **`71`** / `64` | `580 / 1,370` (`42.34%`) | Baseline (`Wave-4` lifts `Riley` `+5.50 pp`) |
| `Wave-6` Gap-Search QLoRA+DPO + `Grounded Harness v0.3` | `38.50%` (`231`) | `106` / `65` / `60` | `550 / 1,397` (`39.37%`) | Multi-query gap-search crowds non-thinking `3.05B` active attention |
| **`Oracle@5` `Qwen3-Coder-30B` Family Union (`$0.00`)** | **`61.17%` (`367`)** | **`155` (`77.5%`) / `109` (`54.5%`) / `103` (`51.5%`)** | — | `61.17%` latent retrieval ceiling across `3.3B`-active MoE passes |

> **Routing Rule (`L2` MoE vs. `L2.5` Dense Thinking)**: I deploy **`qwen3-coder-30b-sre-lora`** for high-throughput (`125.8–131.5 tok/s`) repository code synthesis, AST security audits, statutory compliance checks, and direct MCP tool execution on 22–24 GB VRAM; I escalate multi-hop temporal memory synthesis and contradiction-heavy personal-assistant workflows to dense thinking **[`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora)** (**`57.33%` [`344/600`]**).

---

## 🏆 KruschStack Ecosystem Hermetic Benchmark Scorecard (`75 / 75` Gates, `333,000+` Queries/Traces, `49` Audit Logs)

All `49` audit logs (`48` staged raw JSON logs + `1` `RouteWorks/RouterArena` PR #169 live receipt bundle; `63` dataset files across `14` splits, `75/75` gates + `27/27` validated DolphinBench archives) are published in **[`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)** alongside Paper #27 (**[*Return on Memory: `#1` Overall (`91.17%`), Model-Matched `GPT 5.6 Luna` (`81.50%` @ `$1.35`) & `#1` Open-Weights (`57.33%`) on Official DolphinBench*](https://krusch.dev/articles/return-on-memory-dolphinbench)**) and my cross-stack report (**[*How an Open-Weights Sovereign Stack Stacks Up*](https://krusch.dev/articles/how-krusch-stack-stacks-up)**):

| Pillar | Public Benchmark | Verified Sovereign Score |
| :--- | :--- | :--- |
| **Personal Agent Memory & Tools** | **Official DolphinBench** (`600` tasks × `27` runs = `16,200` traces, `6` families, `openai/gpt-5.6-sol` Judge, `8/8` flagship queued) | **`#1 Overall` `91.17%` (`547/600` @ `$24.54`) Argon Low** (`Oracle@27 = 97.83%`; `$0.00` `Oracle@17 = 83.17%`) · **Model-Matched `GPT 5.6 Luna (High, + Grounded)` `81.50%` (`489` @ `$1.35`, `30.03s`, `+10.83 pp`, `71.1x` cheaper) & `Luna (Low, Std.)` `76.67%` (`@ $1.10`, `87.4x` cheaper; `Luna Oracle@5 = 91.83%`)** · **`#1 Open-Weights` `57.33%` (`344/600`) [`gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (`Wave-6 + Grounded v0.3`) @ `$0.00`** (`Gemma 4 Oracle@6 = 71.83%`) · **`43.00%` (`258/600`) `30B` Wave-4 (`+ Grounded v0.3`), `42.67%` (`256/600`) `Std.` & `42.00%` (`+ 2PC`, `-95.2%` dup MCP, `Wave-4 Oracle@3 = 58.50%`, `30B Oracle@5 = 61.17%`)** |
| **Security & Policy Floor** | **ETH SpyLab AgentDojo v1.2.2** (`1,046` episodes, `949` attacks) | **`0.00%` Targeted ASR (`0/949`, `100%` Defense)** · **`69.07%` Benign** (Spotlighting) & **`83` hardline floor blocks** |
| **Code AST & Hybrid RRF** | **ACL 2025 CoIR** (`2.39M` docs / `162K` Qs) & **RepoBench-R** (`47,998` Qs) | **🥇 `#1` CoIR 10-Task `NDCG@10` (`56.40%`, `73.88%` `R@10` > `Voyage-Code-002` `56.26%`)** · **RepoBench-R `68.58%` `Acc@5`** (beats `CodeBERT` & `UniXcoder`) |
| **SQL State & Deferred MCP** | **BIRD-SQL Mini-Dev** (`500` Qs) & **UC Berkeley BFCL v3 Live** (`2,251` Qs) | **`26.0% → 40.2%` SQL EX (`1.55x`, `100%` valid SQL)** · **`59.9% → 66.3%` BFCL Micro & `84.7%` Irrelevance** at **`-80.7%` schema tokens** |
| **Cascade Router (`L1–L3`)** | **Martian RouterBench** (`73K`) · **ACL 2026 LLMRouterBench** (`12.5K`) · **RouterArena** (`8.4K`) | **🥇 `#1` RouterBench ID/OOD `AIQ` (`-72.6%` cost @ `98%` `GPT-4`)** · **🥇 `#1` LLMRouterBench Pareto (`0.4269`)** · **RouterArena Official CI Run `#780` `0.7263` (`#13 / 34`, `74.44%` @ `$0.3802/1K`, `91.43%` Rob., `0` Abnormal, PR #169 open)** |
| **2PC Harness, Memory & Legal** | **Aider Polyglot** (`225/225`) · **ICLR 2026 BEAM** (`2,000` Qs) · **LegalBench-RAG** (`6,889` Qs) | **`19.56% → 51.56%` `Pass@2` (`2.64x`, ties `Claude 3.5 Sonnet`)** · **`92.58%` `100K` & `93.00%` `10M` IE Recall** · **`26.74%` Legal Macro F1** |