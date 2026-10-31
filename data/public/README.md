---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - code-audit
  - bug-fixing
  - sft
  - dpo
  - qlora
  - distillation
  - mcp
  - agentic-coding
  - qwen3-coder
  - gpt-oss-120b
size_categories:
  - 10K<n<100K
---

# Sovereign Code Audit & Bug Remediation SFT + DPO Dataset

A multi-stage **Teacher-to-Student Knowledge Distillation, Calibrated Referee Convergence, & Automated Code Remediation** dataset for training autonomous software engineering, code auditing, and bug-fixing agents (such as `Qwen/Qwen3-Coder-30B-A3B-Instruct` and `openai/gpt-oss-120b`).

## Dataset Splits

| Split File | Rows | Teacher Model | Description |
| :--- | :---: | :--- | :--- |
| `argon_golden_remediations.jsonl` | **401** | `gemini-4-argon` | **Test-Verified Code Patches & Unit Tests**: Pairs structured Master Audit Backlog cards (`BUG-*`, `SEC-*`, `ARCH-*`, `TEST-*`) with complete, unit-test-verified source file patches (`npm test` / `pytest`) across 85 JavaScript/ESM, TypeScript, React, and Python 3.12 repositories. |
| `lego_and_referee_calibration_sft.jsonl` | **26** | `gemini-4-argon` | **Calibrated `CLEAN_PASS`, P0/P1 Remediations & Lego-First Idioms**: Live-source `CLEAN_PASS` Referee convergence verdicts (`findings: []`), ground-truth security/concurrency bug remediations, and zero-boilerplate `@krusch/toolkit` v1.4 / `krusch_toolkit` v1.2 synthesis across Node 22 ESM, TypeScript, Cloudflare Workers, MCP, pgvector, React, and FastAPI. |
| `referee_calibration_dpo.jsonl` | **8** | `gemini-4-argon` + `gpt-oss-sre:120b` | **Balanced Referee DPO Preference Pairs (`trl.DPOTrainer`)**: 50% Precision pairs (rejecting false-positive helper return-type & package-scope misreads) and 50% Recall pairs (rejecting lazy `CLEAN_PASS` rubber-stamping on real P0/P1 defects). |
| `elitist_120b_teacher.jsonl` | **2284** | `gpt-oss:120b` + `gemini-4-argon` | **Curated High-Signal QLoRA Training Split**: Rejection-sampled trajectories requiring verbatim `file:line` grounding, concrete code blocks/diffs, calibrated `CLEAN_PASS` verdicts, Lego-First idioms, and 2x-oversampled Golden Remediations. |
| `code_audit_11phase_sft.jsonl` | **15290** | `gpt-oss:120b` | **Full 11-Phase Code Audit Corpus**: Complete phase-by-phase codebase investigations (`0_ORIENTATION` through `2_MASTER_BACKLOG`) using AST symbol search (`search_symbols`, `dependency_graph`), bounded file reads (`read_file path:start-end`), and live test verification. |

## Core Engineering Capabilities Learned

1. **Verbatim Line-Grounded Auditing (Zero Code Hallucination):** Every claim cites exact `file:startLine-endLine` coordinates verified against AST symbol graphs and bounded source windows.
2. **Ground-Truth False-Positive Refutation & DPO Calibration:** Models cross-check prior hypotheses against live source code and distinguish helper contracts (e.g., HTTP header-map builders) from JSON endpoint envelopes.
3. **Zero-Finding Convergence (`CLEAN_PASS`):** Models emit `findings: []` when production code satisfies frozen invariants rather than inventing stylistic nitpicks.
4. **Lego-First Toolkit Assembly (`INV-LEGO-01..11`):** Models compose HTTP/SSE/MCP/PG/SQLite/Edge/Webhooks/CLI/FastAPI services from verified runtime bricks instead of hand-rolling boilerplate.
5. **Test-Gated Patch & Regression Test Synthesis:** Models resolve P0–P1 bug/security cards while generating passing `node:test` and `pytest` unit tests.

## Quickstart: QLoRA SFT + DPO Fine-Tuning (`8192` Context)

```bash
# 1. Fine-tune Qwen/Qwen3-Coder-30B-A3B-Instruct (4-bit NF4 QLoRA, 8192 ctx)
python3 scripts/train_qwen3_30b_qlora_g4.py \
  --dataset data/public/elitist_120b_teacher.jsonl \
  --argon-dataset data/public/argon_golden_remediations.jsonl \
  --max-seq-length 8192 \
  --output-dir models/qwen3-coder-30b-sre-lora \
  --epochs 2

# 2. Fine-tune openai/gpt-oss-120b (Blackwell MXFP4 frozen experts + BF16 attention LoRA, 8192 ctx)
python3 scripts/train_gpt_oss_120b_qlora_g4.py \
  --dataset data/public/elitist_120b_teacher.jsonl \
  --max-seq-length 8192 \
  --output-dir models/gpt-oss-120b-sre-lora \
  --epochs 2

# 3. Optional Post-SFT Referee Preference Calibration (DPO, beta=0.1)
python3 scripts/train_gpt_oss_120b_qlora_g4.py \
  --mode dpo \
  --dpo-dataset data/public/referee_calibration_dpo.jsonl \
  --adapter-dir models/gpt-oss-120b-sre-lora \
  --output-dir models/gpt-oss-120b-sre-lora \
  --epochs 2
```
