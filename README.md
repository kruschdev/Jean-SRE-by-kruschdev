# Jean SRE by Kruschdev (`jean-sre`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-sovereign--code--audit--sft-yellow.svg)](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
[![Hugging Face 30B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2%2030B-qwen3--coder--30b--sre--lora-blue.svg)](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora)
[![Hugging Face 120B LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20L2.5%20120B-gpt--oss--120b--sre--lora-orange.svg)](https://huggingface.com/krush420/gpt-oss-120b-sre-lora)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**The Autonomous 24/7 Site Reliability Engineer, 11-Phase Code Auditor, Test-Gated Bug Remediator & Teacher-Student QLoRA Distillation Engine for [KruschStack](https://github.com/kruschdev/krusch-stack).**

`Jean SRE` is the continuous background engineering daemon that drives the **[KruschStack 7-Pillar, 39-Tool MCP Ecosystem](https://github.com/kruschdev/krusch-stack)** (`npx krusch-stack init`). It continuously sweeps repositories through an **11-Phase Principal Engineer Code Audit**, verifies every claim against live AST symbol graphs and unit test suites (`npm test` / `pytest`, exit code `0`), refutes false positives (`NOT FLAGGED`), converges to `CLEAN_PASS`, and distills high-signal trajectories into open-weights QLoRA adapters.

---

## 🤗 Open-Weights Models & SFT Dataset (Trained by Jean SRE + KruschStack)

All trajectories generated and verified by **Jean SRE** through **KruschStack** are published on Hugging Face Hub (`@krush420`):

| Resource | Hugging Face Repository | Role in KruschStack Cascade |
| :--- | :--- | :--- |
| **📊 SFT Dataset (`17,899` rows)** | [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft) | 3 Splits: `argon_golden_remediations` (`401`), `elitist_120b_teacher` (`2,208`), `code_audit_11phase_sft` (`15,290`) |
| **⚡ 30B Student QLoRA (`L2`)** | [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) | [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (`30.5B` total / `3.3B` active MoE, `93.36%` token acc) |
| **🧠 120B Teacher QLoRA (`L2.5`)** | [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`116.8B` total / `5.1B` active MXFP4 MoE, single-GPU 96GB Blackwell QLoRA) |
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
│ • <25µs L1 fast-path -> L2 30B Student -> L2.5 120B Teacher -> L3 Referee  │
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

## 🔬 Single-GPU 96GB Blackwell QLoRA Training Scripts Included

This repository includes the exact training and dataset curation scripts used on an **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)** to train both Hugging Face LoRA adapters:

- [`scripts/build_public_sft_release.js`](scripts/build_public_sft_release.js) — Builds the 3-split public SFT dataset with an automated **Zero-Leak & Portability Gate** (`0` private IPs, `0` internal hostnames, `0` user home paths, `0` secrets).
- [`scripts/train_qwen3_30b_qlora_g4.py`](scripts/train_qwen3_30b_qlora_g4.py) / [`models/jean-qwen3-30b-sre-lora/train_qlora.py`](models/jean-qwen3-30b-sre-lora/train_qlora.py) — QLoRA fine-tuning for `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit NF4 attention + BF16 3D MoE experts).
- [`scripts/train_gpt_oss_120b_qlora_g4.py`](scripts/train_gpt_oss_120b_qlora_g4.py) / [`models/jean-gpt-oss-120b-sre-lora/train_qlora.py`](models/jean-gpt-oss-120b-sre-lora/train_qlora.py) — Single-GPU native **MXFP4 QLoRA** fine-tuning for `openai/gpt-oss-120b` (`116.8B` parameters), keeping all 36 layers of 128 routed MoE experts frozen in native 4-bit MXFP4 (`~60.8 GB` VRAM) while training BF16 attention LoRA adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`) through the Pre-LN residual stream.

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
