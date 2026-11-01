# Jean SRE by Kruschdev (`jean-sre`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-sovereign--code--audit--sft%20(18.1K)-yellow.svg)](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
[![Hugging Face 8B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L1.5%208B-llama3--hermes3--8b--sre--lora-teal.svg)](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora)
[![Hugging Face 30B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2%2030B-qwen3--coder--30b--sre--lora-blue.svg)](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora)
[![Hugging Face 120B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2.5%20120B-gpt--oss--120b--sre--lora-orange.svg)](https://huggingface.co/krush420/gpt-oss-120b-sre-lora)
[![Hugging Face Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmarks-73%2F73%20SHA--256%20Gates-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Arena-sovereign--router--arena-00e5ff.svg)](https://huggingface.co/spaces/krush420/sovereign-router-arena)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**The Autonomous 24/7 Site Reliability Engineer, 11-Phase Code Auditor, Test-Gated Bug Remediator & Teacher-Student QLoRA + DPO Distillation Engine for [KruschStack](https://github.com/kruschdev/krusch-stack).**

`Jean SRE` is the continuous background engineering daemon that drives the **[KruschStack 7-Pillar, 39-Tool MCP Ecosystem](https://github.com/kruschdev/krusch-stack)** (`npx krusch-stack init`). It continuously sweeps repositories through an **11-Phase Principal Engineer Code Audit**, verifies every claim against live AST symbol graphs and unit test suites (`npm test` / `pytest`, exit code `0`), refutes false positives (`NOT FLAGGED`), converges to `CLEAN_PASS`, and distills high-signal SFT + DPO preference trajectories into open-weights Wave-4 QLoRA adapters across `8B`, `30B`, and `120B` parameter tiers.

---

## 🤗 Open-Weights Models, SFT+DPO Dataset & Benchmarks (`@krush420`)

All trajectories generated and verified by **Jean SRE** through **KruschStack** are published on Hugging Face Hub (`@krush420`):

| Resource | Hugging Face Repository | Role in KruschStack Cascade |
| :--- | :--- | :--- |
| **📊 SFT + DPO Dataset (`18,131` rows)** | [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft) | **7 Zero-Leak Splits**: `code_audit_11phase_sft` (`15,290`), `elitist_120b_teacher` (`2,308`), `argon_golden_remediations` (`401`), `agentic_tool_calling_sft` (`64`), `lego_and_referee_calibration_sft` (`38`), `agentic_tool_calling_dpo` (`18`), `referee_calibration_dpo` (`12`) |
| **🪶 8B Edge Agent & Referee LoRA (`L1.5`)** | [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) | [`NousResearch/Hermes-3-Llama-3.1-8B`](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) (Built with Meta Llama 3 · 7-proj BF16 LoRA + DPO + fixed Ollama Go-template `Modelfile`, **`15.7x` DolphinBench lift**) |
| **⚡ 30B Student QLoRA (`L2`)** | [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) | [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (`30.5B` total / `3.3B` active MoE, **`42.67%` [`256/600`] Official DolphinBench @ `$0.00`**, `100%` Lego/Referee calibration) |
| **🧠 120B Teacher QLoRA (`L2.5`)** | [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MXFP4 MoE, single-GPU 96GB Blackwell QLoRA + DPO, **`39.50%` [`237/600`] Official DolphinBench**) |
| **🏆 Hermetic Benchmark Archive (`73/73` Gates)** | [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) | `317,581+` queries/episodes across `11` external suites, `26` SHA-256-pinned raw audit logs, and `14` interactive viewer splits (`350.92 MiB`) |
| **📈 Interactive Pareto & Leaderboard Arena** | [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena) | Zero-cold-start static explorer covering RouterArena, RouterBench, LLMRouterBench, Wave-2 Security/SQL/MCP/CoIR, and Tab 5 `DolphinBench 600 (gpt-5.6-sol)` |
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
│ • <25µs L1 fast-path -> L1.5 8B Edge -> L2 30B Student -> L2.5 120B -> L3  │
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

## 🔬 Wave-4 Unified 2-Stage QLoRA + DPO Pipeline & DolphinBench Verification

This repository includes the exact dataset curation, 2-stage SFT + KL-anchored `trl.DPOTrainer` (`beta=0.1`) calibration, and GGUF fusion scripts used on an **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)** to train all three Hugging Face Wave-4 adapters:

- [`scripts/build_public_sft_release.js`](scripts/build_public_sft_release.js) — Builds the **7-split (`18,131`-row)** public SFT + DPO dataset with an automated **Zero-Leak & Portability Gate** (`0` private IPs, `0` internal hostnames, `0` user home paths, `0` secrets).
- [`scripts/harvest_lego_and_referee_calibration.js`](scripts/harvest_lego_and_referee_calibration.js) & [`scripts/harvest_agentic_tool_calling.js`](scripts/harvest_agentic_tool_calling.js) — Curate calibrated `CLEAN_PASS` (`findings: []`) Referee verdicts, `@krusch/toolkit` v1.7 / `krusch_toolkit` v1.6 Rule-of-3 Lego idioms, and multi-turn MCP/workspace function-calling + DPO preference pairs (`action_over_prose`, `bounded_search_no_loop`, `benign_execution_no_false_refusal`).
- [`scripts/train_hermes3_8b_qlora_g4.py`](scripts/train_hermes3_8b_qlora_g4.py) / [`models/jean-hermes3-8b-sre-lora/`](models/jean-hermes3-8b-sre-lora/) — 2-stage native `BF16` LoRA + DPO across all 7 Llama-3.1 linear projections (`41.94M` trainable params) for `NousResearch/Hermes-3-Llama-3.1-8B`, paired with a fixed Ollama Go-template `Modelfile`.
- [`scripts/train_qwen3_30b_qlora_g4.py`](scripts/train_qwen3_30b_qlora_g4.py) / [`models/jean-qwen3-30b-sre-lora/`](models/jean-qwen3-30b-sre-lora/) — 2-stage QLoRA + DPO for `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit NF4 attention + BF16 3D MoE experts), lifting zero-tool Lego/Rule-of-3 adoption from `25%` → `100%`.
- [`scripts/train_gpt_oss_120b_qlora_g4.py`](scripts/train_gpt_oss_120b_qlora_g4.py) / [`models/jean-gpt-oss-120b-sre-lora/`](models/jean-gpt-oss-120b-sre-lora/) — Single-GPU native **MXFP4 QLoRA + DPO** for `openai/gpt-oss-120b` (`116.8B` parameters), keeping all 36 layers of 128 routed MoE experts frozen in native 4-bit MXFP4 while training BF16 attention adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with Harmony `<|channel|>commentary` tool routing.
- [`scripts/fuse_lora_into_gguf.py`](scripts/fuse_lora_into_gguf.py) & [`scripts/convert_peft_to_gguf_lora.py`](scripts/convert_peft_to_gguf_lora.py) — Sub-2-second in-place GGUF LoRA fuser supporting `BF16`, `MXFP4`, and `Q4_0` Ollama deployments.

### Official `gpt-5.6-sol` DolphinBench (`600` Tasks, `1.5M` Tokens, `$0.00` Local Inference)

Paired with `SovereignMemoryStore` (`256/32` Raw-RAG + Porter-FTS5 + 4-Way RRF), **every Wave-4 Unified QLoRA+DPO adapter beat its Stock baseline** across all `600` [DolphinBench](https://dolphinbench.ai/) multi-app workspace tasks (`5/5` archives verified & queued on `dolphinbench.ai`):

| Tier & Base Model | Stock Baseline (`gpt-5.6-sol`) | **Wave-4 Unified QLoRA+DPO (`gpt-5.6-sol`)** | Mechanical Checks (`1,310`) | Inference Cost |
| :--- | :---: | :---: | :---: | :---: |
| **`L2` `Qwen3-Coder-30B-A3B`** *(Local 2x 2080 Ti)* | `42.00%` (`252/600`) | **`42.67%` (`256/600`, `+4` tasks)** | **`1,123/1,310` (`85.73%`)** | **`$0.00`** |
| **`L2.5` `openai/gpt-oss-120b`** *(Colab G4 96GB)* | `39.17%` (`235/600`) | **`39.50%` (`237/600`, `+2` tasks)** | `1,033/1,310` (`78.85%`) | **`$0.00`** |
| **`L1.5` `Hermes-3-Llama-3.1-8B`** *(`Q4_0` Edge)* | `1.17%` (`7/600` local) | **`18.33%` (`110/600`, `15.7x` lift)** | **`841/1,310` (`64.20%`)** | **`$0.00`** |

> `30B` Wave-4 (`42.67%` @ `$0.00`) outperforms published cloud entries including `Hermes + MiniMax M3 + Hindsight` (`41.00%` @ `$158.33`), `Claude Code + Claude Sonnet 5 + Honcho` (`35.83%` @ `$1,565.82`), and `Claude Sonnet 5 + Mem0` (`32.33%` @ `$1,830.57`), while frontier cloud `Hermes + GPT 5.6 Luna + Mem0` leads overall at `70.67%` (`424/600`) @ `$96.21`.

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
