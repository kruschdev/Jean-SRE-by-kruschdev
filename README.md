# Jean SRE by Kruschdev (`jean-sre`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-sovereign--code--audit--sft%20(18.3K)-yellow.svg)](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
[![Hugging Face 8B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L1.5%208B-llama3--hermes3--8b--sre--lora-teal.svg)](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora)
[![Hugging Face 30B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2%2030B-qwen3--coder--30b--sre--lora%20(125.8%20tok%2Fs)-blue.svg)](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora)
[![Hugging Face 31B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20%231%20Open--Weights%2031B-gemma4--31b--sre--lora%20(57.33%25)-brightgreen.svg)](https://huggingface.co/krush420/gemma4-31b-sre-lora)
[![Hugging Face 120B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2.5%20120B-gpt--oss--120b--sre--lora-orange.svg)](https://huggingface.co/krush420/gpt-oss-120b-sre-lora)
[![Hugging Face Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmarks-75%2F75%20SHA--256%20Gates-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Arena-sovereign--router--arena-00e5ff.svg)](https://huggingface.co/spaces/krush420/sovereign-router-arena)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**The Autonomous 24/7 Site Reliability Engineer, 11-Phase Code Auditor, Test-Gated Bug Remediator & Teacher-Student QLoRA + DPO Distillation Engine for [KruschStack](https://github.com/kruschdev/krusch-stack).**

`Jean SRE` is the continuous background engineering daemon that drives my **[KruschStack 7-Pillar, 39-Tool MCP Ecosystem](https://github.com/kruschdev/krusch-stack)** (`npx krusch-stack init`). It continuously sweeps repositories through an **11-Phase Principal Engineer Code Audit**, verifies every claim against live AST symbol graphs and unit test suites (`npm test` / `pytest`, exit code `0`), refutes false positives (`NOT FLAGGED`), converges to `CLEAN_PASS`, and distills high-signal SFT + DPO preference trajectories into open-weights Wave-4 / Wave-6 / Wave-7 QLoRA adapters across `8B`, `30B`, `31B`, and `120B` parameter tiers.

---

## 🤗 Open-Weights Models, SFT+DPO Dataset & Benchmarks (`@krush420`)

All trajectories generated and verified by **Jean SRE** through **KruschStack** are published on Hugging Face Hub (`@krush420`):

| Resource | Hugging Face Repository | Role in KruschStack Cascade |
| :--- | :--- | :--- |
| **📊 SFT + DPO Dataset (`18,341` rows)** | [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft) | **9 Zero-Leak Splits**: `code_audit_11phase_sft` (`15,290`), `elitist_120b_teacher` (`2,308`), `argon_golden_remediations` (`401`), `arete_values_sft` (`140`), `agentic_tool_calling_sft` (`76`), `arete_values_dpo` (`42`), `lego_and_referee_calibration_sft` (`38`), `agentic_tool_calling_dpo` (`34`), `referee_calibration_dpo` (`12`) |
| **🥇 `#1 Open-Weights` 31B Dense QLoRA (`L2.2`)** | [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) | [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) (`30.7B` dense Apache-2.0 · Wave-6 assistant-masked `<|channel>thought` SFT + BF16 ref-anchored DPO across all `60` layers, **`57.33%` [`344/600`] `#1 Open-Weights` Official DolphinBench @ `$0.00`**, `+3.17 pp` vs. Stock, `Gemma 4 Oracle@6 = 71.83%`) |
| **🪶 8B Edge Agent & Referee LoRA (`L1.5`)** | [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) | [`NousResearch/Hermes-3-Llama-3.1-8B`](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) (Built with Meta Llama 3 · 7-proj BF16 LoRA + DPO + fixed Ollama Go-template `Modelfile`, **`15.7x` DolphinBench lift**) |
| **⚡ 30B Student QLoRA (`L2` Wave-7)** | [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) | [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (`30.5B` total / `3.3B` active MoE · **Wave-7 SRE + Curated `NousResearch/Arete` QLoRA SFT + BF16 Ref-Anchored DPO**, **`125.8–131.5 tok/s`** [`5.5x` faster than dense `Arete-27B` `23.5 tok/s`], **`43.00%` [`258/600`] `+ Grounded v0.3` / `42.67%` [`256/600`] `Std. Harness` @ `$0.00`**, **`58.50%` [`351/600`] `Wave-4 Oracle@3`**, `100%` Lego/Referee calibration) |
| **🧠 120B Teacher QLoRA (`L2.5`)** | [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MXFP4 MoE, single-GPU 96GB Blackwell QLoRA + DPO, **`39.50%` [`237/600`] Official DolphinBench `Std. Harness`**) |
| **🏆 Hermetic Benchmark Archive (`75/75` Gates)** | [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) | **`333,000+`** queries/traces across `12` staged external suites (`90/90` ecosystem gates incl. `adyen/DABstep`), **`48` SHA-256-pinned raw JSON audit logs** (`49` total incl. the `RouteWorks/RouterArena` [PR #169](https://github.com/RouteWorks/RouterArena/pull/169) live receipt bundle), and `14` interactive viewer splits (`63` dataset files) |
| **📈 Interactive Pareto & Leaderboard Arena** | [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena) | Zero-cold-start static explorer covering RouterArena, RouterBench, LLMRouterBench, Wave-2 Security/SQL/MCP/CoIR, and Tab 5 `DolphinBench 600` (`27` Sovereign runs / `16,200` `gpt-5.6-sol` traces across `6` families, `40` leaderboard rows) |
| **🛠️ KruschStack MCP Spine** | [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) | Unified 7-Pillar, 39-Tool MCP Server & CLI (`npx krusch-stack init`) |

---

## 🏗️ Architecture: How Jean SRE Drives KruschStack

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   JEAN SRE AUTONOMOUS 24/7 SWEEP & HARNESS                 │
│     • 11-Phase Code Audit Loop (`src/jean_sre_sweep.js`)                   │
│     • Thinker/Executor Split + Opportunistic 120B Burst (`src/harness.js`) │
│     • Live Test Verification + Distillation Capture (`data/public/`)       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ MCP Stdio / HTTP Bridge
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        KRUSCH-STACK UNIFIED GATEWAY                        │
│               `npx krusch-stack mcp`  (39 Annotated MCP Tools)             │
├──────────────────────┬───────────────────────────┬─────────────────────────┤
│ 1. WORKING MEMORY    │ 2A. CODEBASE AST & GIT    │ 2B. OS PAGE MEMORY      │
│ krusch-context-mcp   │ krusch-git                │ krusch-fs               │
│ • retrieve / remember│ • search_symbols          │ • dependency_graph      │
│ • revise / nudge     │ • read_tree / read_blob   │ • page_tree / read_page │
│ • INV-12 span drift  │ • semantic_search         │ • remember_on_page      │
├──────────────────────┼───────────────────────────┼─────────────────────────┤
│ 3. 2PC & GROUNDED    │ 4. FRONTEND DESIGN MCP    │ 5. WORKSPACE HUB        │
│ krusch-harness       │ krusch-frontend-mcp       │ g-krusch                │
│ • run (stages diff)  │ • generate_color_palette  │ • get_agent_steering    │
│ • apply_diff (2PC)   │ • css_to_tailwind         │ • search_drive_rag      │
│ • grounded-harness   │ • canonical_templates     │ • write (.gdoc/.slides) │
├──────────────────────┴───────────────────────────┴─────────────────────────┤
│ 6. COGNITIVE CASCADE ROUTER (`krusch-cascade-router`)                      │
│ • <25µs L1 fast-path -> L1.5 8B Edge -> L2 30B/31B -> L2.5 120B -> L3      │
└────────────────────────────────────────────────────────────────────────────┘
```

### The 11-Phase Principal Engineer Code Audit Loop
For every repository in rotation, Jean SRE executes a deterministic 11-phase investigation:
1. **`0_ORIENTATION`** — Hydrate project invariants (`krusch_context_retrieve`), map entrypoints, and reset cross-project state.
2. **`1.1_ARCHITECTURE_SEAMS`** — Trace module boundaries and AST caller/callee dependency graphs (`krusch_git_dependency_graph`).
3. **`1.2_COMPLIANCE_MATRIX`** — Cross-check prior backlog claims against live source windows (`read_file path:start-end`) and refute stale/false-positive items (`NOT FLAGGED`).
4. **`1.3_SECURITY_TRUST_BOUNDARIES`** — Audit auth, command execution, SQL parameterization, and secret handling (`src/compliance.js`).
5. **`1.4_CONCURRENCY_STATE`** — Inspect race conditions, promise deduplication, and database transaction boundaries.
6. **`1.5_ERROR_PROPAGATION`** — Verify deterministic error handling and fallback safety.
7. **`1.6_RESOURCE_BOUNDS`** — Check memory caps, timeouts, stream backpressure, and cache eviction.
8. **`1.7_SHARED_TOOLKIT_PARITY`** — Identify duplicate inline helpers that should use shared `@krusch/toolkit` primitives.
9. **`1.8_DEAD_CODE_DRIFT`** — Detect unreachable symbols, stale exports, and documentation drift.
10. **`1.9_TEST_GAP_VERIFICATION`** — Execute the live project test suite (`runProjectVerificationSuite`) and pinpoint untested edge cases.
11. **`2_MASTER_BACKLOG`** — Emit line-grounded `BUG-*`, `SEC-*`, `ARCH-*`, `TEST-*`, or `STALE-*` cards — or converge to `CLEAN_PASS` (`0` findings) when the repository is clean.

---

## 🔬 Wave-4, Wave-6 & Wave-7 2-Stage QLoRA + DPO Pipeline & DolphinBench Verification

This repository includes the exact dataset curation, 2-stage SFT + reference-anchored `trl.DPOTrainer` (`beta=0.1` / `0.2`) calibration, and GGUF fusion scripts I use on an **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)** to train all four Hugging Face open-weights adapters:

- [`scripts/build_public_sft_release.js`](scripts/build_public_sft_release.js) — Builds the **9-split (`18,341`-row)** public SFT + DPO dataset with an automated **Zero-Leak & Portability Gate** (`0` private IPs, `0` internal hostnames, `0` user home paths, `0` secrets).
- [`scripts/harvest_lego_and_referee_calibration.js`](scripts/harvest_lego_and_referee_calibration.js), [`scripts/harvest_agentic_tool_calling.js`](scripts/harvest_agentic_tool_calling.js) & [`scripts/harvest_arete_alignment.js`](scripts/harvest_arete_alignment.js) — Curate calibrated `CLEAN_PASS` (`findings: []`) Referee verdicts, `@krusch/toolkit` v1.8 / `krusch_toolkit` v1.6 Rule-of-3 Lego idioms, multi-turn MCP/workspace function-calling + Wave-6 gap-search DPO preference pairs, and Wave-7 zero-defect statutory/analytical non-refusal SFT (`140` gold rows: `accuracy=5, independence=5, 0 issue tags`) + high-contrast DPO pairs (`42` pairs) from `NousResearch/arete-data-pairs` (`Apache-2.0`).
- [`scripts/train_gemma4_31b_qlora_g4.py`](scripts/train_gemma4_31b_qlora_g4.py) / [`models/jean-gemma4-31b-sre-lora/`](models/jean-gemma4-31b-sre-lora/) — Wave-6 2-stage 4-bit NF4 QLoRA + BF16 reference-anchored DPO (`ref_adapter_name` verified `max_diff_vs_sft == 0.0`) across all 60 hybrid sliding/full attention layers (`230` GGUF LoRA tensor pairs) for `google/gemma-4-31b-it`, using `Gemma4AssistantOnlyCollator` (`<|channel>thought` assistant-only loss masking, `81.56%` token acc, `100%` DPO reward acc) to reach **`57.33%` (`344/600`, `#1 Open-Weights`)** on Official DolphinBench.
- [`scripts/train_hermes3_8b_qlora_g4.py`](scripts/train_hermes3_8b_qlora_g4.py) / [`models/jean-hermes3-8b-sre-lora/`](models/jean-hermes3-8b-sre-lora/) — 2-stage native `BF16` LoRA + DPO across all 7 Llama-3.1 linear projections (`41.94M` trainable params) for `NousResearch/Hermes-3-Llama-3.1-8B`, paired with a fixed Ollama Go-template `Modelfile`.
- [`scripts/train_qwen3_30b_qlora_g4.py`](scripts/train_qwen3_30b_qlora_g4.py) / [`models/jean-qwen3-30b-sre-lora/`](models/jean-qwen3-30b-sre-lora/) — Wave-7 2-stage QLoRA + BF16 reference-anchored DPO (`QwenChatMLAssistantOnlyCollator`, `560` SFT rows, `156` DPO pairs, `384` BF16 ref tensors with `max_diff_vs_sft == 0.0`) for `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit NF4 attention + BF16 3D MoE experts), delivering **`125.8–131.5 tok/s`** (`5.5x` faster than dense `Arete-27B` `23.5 tok/s`) on zero-refusal statutory and security code audits while reaching `43.00%` (`258/600`) on DolphinBench with `Grounded Harness v0.3`.
- [`scripts/train_gpt_oss_120b_qlora_g4.py`](scripts/train_gpt_oss_120b_qlora_g4.py) / [`models/jean-gpt-oss-120b-sre-lora/`](models/jean-gpt-oss-120b-sre-lora/) — Single-GPU native **MXFP4 QLoRA + DPO** for `openai/gpt-oss-120b` (`116.8B` parameters), keeping all 36 layers of 128 routed MoE experts frozen in native 4-bit MXFP4 while training BF16 attention adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with Harmony `<|channel|>commentary` tool routing.
- [`scripts/fuse_lora_into_gguf.py`](scripts/fuse_lora_into_gguf.py) & [`scripts/convert_peft_to_gguf_lora.py`](scripts/convert_peft_to_gguf_lora.py) — Sub-2-second in-place GGUF LoRA fuser and PEFT-to-GGUF converter supporting `BF16`, `MXFP4`, `Q4_K_M`, and `Q4_0` Ollama deployments (including Gemma 4 `blk.N.attn_*` projection mapping).

### Official `gpt-5.6-sol` DolphinBench V6 Closeout (`27` Runs, `16,200` Traces, `6` Model Families)

Paired with `SovereignMemoryStore` (`256/32` Raw-RAG + Porter-FTS5 + 4-Way RRF), **every published `@krush420` QLoRA+DPO adapter beat its `Stock` baseline** across all `600` [DolphinBench](https://dolphinbench.ai/) multi-app workspace tasks (**`8/8` flagship archives verified & queued** on `dolphinbench.ai` across **`27` official 600-task evaluations (`16,200` `openai/gpt-5.6-sol` traces across `6` model families; `27/27` validated)**; full paper: [`krusch.dev/articles/return-on-memory-dolphinbench`](https://krusch.dev/articles/return-on-memory-dolphinbench)):

| Tier & Base Model | `Stock` Checkpoint (`gpt-5.6-sol`) | **Published `@krush420` QLoRA+DPO (`gpt-5.6-sol`)** | Official Grader Mech Checks | Inference Cost |
| :--- | :---: | :---: | :---: | :---: |
| **`L2.2` `Google Gemma 4 31B` *(Dense)* ** | `54.17%` (`325/600` Std.) / `56.67%` (`340` Grounded) | **`57.33%` (`344/600`, `+19` vs. Stock, `#1 Open-Weights`)** | **`835/1,364` (`61.22%`)** | **`$0.00`** |
| **`L2` `Qwen3-Coder-30B-A3B`** *(Local 2x 2080 Ti, `125.8–131.5 tok/s`)* | `42.00%` (`252/600`) | **`42.67%` (`256/600` Std.) / `43.00%` (`258` Grounded)** | **`613/1,364` (`44.94%`)** | **`$0.00`** |
| **`L2.5` `openai/gpt-oss-120b`** *(Colab G4 96GB)* | `39.17%` (`235/600`) | **`39.50%` (`237/600`, `+2` tasks)** | `555/1,413` (`39.28%`) | **`$0.00`** |
| **`L1.5` `Hermes-3-Llama-3.1-8B`** *(`Q4_0` Edge)* | `1.17%` (`7/600` local) | **`18.33%` (`110/600`, `15.7x` lift)** | `264/1,468` (`17.98%`) | **`$0.00`** |

#### Sovereign V6 Cost–Accuracy Pareto Frontier, Grounded / 2PC Ablations & Ensembles (`27` Runs)

| Configuration (`Checkpoint` + `Tool Loop`) | Official `gpt-5.6-sol` Score | API Cost | Leaderboard & Ablation Highlights |
| :--- | :---: | :---: | :--- |
| **🥇 `#1 Overall` — `Gemini 4 Argon (Low, Std. Harness)`** | **`91.17%` (`547/600`)** | **`$24.54`** | `#1` Overall (`Argon Low + Grounded` & `Argon Medium` **`89.17%`** [`535/600`]; `Argon Low Oracle@2 = 94.00%` [`564/600`]) |
| **⚖️ Model-Matched — `GPT 5.6 Luna (High, + Grounded v0.2)`** | **`81.50%` (`489/600`)** | **`$1.35`** | **`+10.83 pp` (`+65` tasks) higher & `71.1x` cheaper** than Official `#1` `GPT 5.6 Luna + Mem0` (`70.67%` [`424/600`] @ `$96.21`), `30.03s` median (`1.57x` faster than `Luna High Std.` `81.33%`); **`Luna Oracle@5 = 91.83%`** (`551/600`) |
| **⚡ Low-Cost Frontier — `GPT 5.6 Luna (Low, Std. Harness)`** | **`76.67%` (`460/600`)** | **`$1.10`** | `+6.00 pp` above Official `#1` `Mem0` at **`87.4x` lower cost** (`+ 2PC` = `76.50%`) |
| **🥇 `#1 Open-Weights` — [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (`Wave-6 + Grounded v0.3`)** | **`57.33%` (`344/600`)** | **`$0.00`** | `#1` Open-Weights (`+3.17 pp` vs. `54.17%` Stock UUID `288671ad…`; `Gemma 4 Oracle@6 = 71.83%` [`431/600`]), beating all `8` non-Luna commercial cloud stacks (`MiniMax M3 + Mem0` `47.83%` @ `$148.02` & `Claude Sonnet 5 + Honcho` `35.83%` @ `$1,565.82`) |
| **🛡️ `Qwen3-Coder-30B-A3B Wave-4 (+ Grounded v0.3 / + 2PC)`** | **`43.00%` (`258/600`)** | **`$0.00`** | `+ Grounded v0.3` reaches `43.00%` (`Wave-4 Oracle@3 = 58.50%` [`351/600`], `30B Oracle@5 = 61.17%` [`367/600`]); `+ INV-STACK-10 2PC` (`42.00%`) cuts duplicate MCP calls by **`-95.2%` (`42 -> 2`)** |
| **🌐 Ensembles (`Oracle@27` Global / `Oracle@17` `$0.00` Open-Weights)** | **`97.83%` / `83.17%`** | **`—` / `$0.00`** | **`Oracle@27 = 97.83%` (`587/600`, `<= 2.17%` retrieval miss)** · **`$0.00` Open-Weights `Oracle@17 = 83.17%` (`499/600`, `+12.50 pp` > `Mem0`)** |

---

## ⚡ Quickstart

### 1. Install the KruschStack MCP Ecosystem
```bash
npx krusch-stack init
npx krusch-stack doctor
```

### 2. Configure & Run Jean SRE
```bash
git clone https://github.com/kruschdev/Jean-SRE-by-kruschdev.git
cd Jean-SRE-by-kruschdev
cp .env.example .env

# Run the security & compliance verification test suite
npm test

# Start the Jean SRE Harness & Autonomous Sweep via Docker Compose
docker compose up -d
curl http://127.0.0.1:3005/health
```

---

## 📜 License

MIT © Kevin Ruschman ([`kruschdev`](https://github.com/kruschdev))
