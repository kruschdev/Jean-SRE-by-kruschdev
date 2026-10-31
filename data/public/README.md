---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - krusch-stack
  - mcp
  - code-audit
  - bug-fixing
  - sft
  - dpo
  - qlora
  - distillation
  - agentic-coding
  - agentic-harness
  - qwen3-coder
  - gpt-oss-120b
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: argon_golden_remediations
        path: argon_golden_remediations.jsonl
      - split: lego_and_referee_calibration_sft
        path: lego_and_referee_calibration_sft.jsonl
      - split: elitist_120b_teacher
        path: elitist_120b_teacher.jsonl
      - split: code_audit_11phase_sft
        path: code_audit_11phase_sft.jsonl
  - config_name: referee_calibration_dpo
    data_files:
      - split: train
        path: referee_calibration_dpo.jsonl
---

# Sovereign Code Audit & Bug Remediation SFT + DPO Dataset (`krusch-stack`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> Every trajectory and preference pair in this dataset (`18,049` total Zero-Leak records across 5 splits) was synthesized, AST-grounded, and unit-test verified through **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It powers both the **`L2` 30B Local Student** and **`L2.5` 120B Burst Teacher** inside the **KruschStack Cognitive Cascade Router**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **⚡ 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`)
- **🧠 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)
- **🏆 Hermetic Benchmark Audit Logs (`68/68` Gates, `316,981+` Verified Queries)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)

---

## 📊 Dataset Splits

| Config | Split Name | File | Rows | Teacher Model | Description |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `default` | `argon_golden_remediations` | `argon_golden_remediations.jsonl` | **401** | `gemini-4-argon` | **Test-Verified Code Patches & Unit Tests**: Pairs structured Master Audit Backlog cards (`BUG-*`, `SEC-*`, `ARCH-*`, `TEST-*`) with complete, unit-test-verified source file patches (`npm test` / `pytest`, exit code `0`) across JavaScript/ESM, TypeScript, React, and Python 3.12 repositories. |
| `default` | `lego_and_referee_calibration_sft` | `lego_and_referee_calibration_sft.jsonl` | **38** | `gemini-4-argon` | **Calibrated `CLEAN_PASS`, P0/P1 Remediations, Lego-First & Rule-of-3 Scout Idioms**: Live-source `CLEAN_PASS` Referee convergence verdicts (`findings: []`), ground-truth security/concurrency bug remediations, zero-boilerplate `@krusch/toolkit` v1.5 (24 bricks) / `krusch_toolkit` v1.3 synthesis, and cross-repository Rule-of-3 `lego_scout` triage (`INV-LEGO-01..16`). |
| `default` | `elitist_120b_teacher` | `elitist_120b_teacher.jsonl` | **2308** | `gpt-oss:120b` + `gemini-4-argon` | **Curated High-Signal QLoRA Training Split**: Rejection-sampled trajectories requiring verbatim `file:line` grounding, concrete code blocks/diffs, calibrated `CLEAN_PASS` verdicts, Lego-First / Lego-Scout idioms, and 2x-oversampled Golden Remediations. |
| `default` | `code_audit_11phase_sft` | `code_audit_11phase_sft.jsonl` | **15290** | `gpt-oss:120b` | **Full 11-Phase Code Audit Corpus**: Complete phase-by-phase codebase investigations (`0_ORIENTATION` through `2_MASTER_BACKLOG`) using AST symbol search (`search_symbols`, `dependency_graph`), bounded file reads (`read_file path:start-end`), and live test verification. |
| `referee_calibration_dpo` | `train` | `referee_calibration_dpo.jsonl` | **12** | `gemini-4-argon` + `gpt-oss-sre:120b` | **Balanced Referee & Architecture DPO Preference Pairs (`trl.DPOTrainer`)**: Precision pairs (rejecting false-positive helper return-type & package-scope misreads), Recall pairs (rejecting lazy `CLEAN_PASS` rubber-stamping on real P0/P1 defects), and Architecture Calibration pairs (Rule-of-3 anti-bloat, thin SQL vs heavy ORM, zero-build `.d.ts` vs premature Rust, connect-time SSRF vs regex). |

---

## 🎯 Core Engineering Capabilities Learned

1. **Verbatim Line-Grounded Auditing (Zero Code Hallucination):** Every claim cites exact `file:startLine-endLine` coordinates verified against `krusch-git` AST symbol graphs and bounded source windows.
2. **Ground-Truth False-Positive Refutation & DPO Calibration:** Models cross-check prior hypotheses against live source code and distinguish helper contracts (e.g., HTTP header-map builders) from JSON endpoint envelopes.
3. **Zero-Finding Convergence (`CLEAN_PASS`):** Models emit `findings: []` when production code satisfies frozen invariants rather than inventing stylistic nitpicks.
4. **Lego-First Toolkit Assembly & Rule-of-3 Scouting (`INV-LEGO-01..16`):** Models compose HTTP/SSE/MCP/PG/SQLite/Security/Edge/Webhooks/CLI/FastAPI services from verified runtime bricks while enforcing the `>= 3` repository Rule-of-3 anti-bloat gate.
5. **Test-Gated Patch & Regression Test Synthesis:** Models resolve P0–P1 bug/security cards while generating passing `node:test` and `pytest` unit tests compatible with `krusch-harness`.

---

## ⚡ Quickstart

### 1. Load SFT & DPO Splits with Hugging Face `datasets`

```python
from datasets import load_dataset

# Load SFT splits ("argon_golden_remediations", "lego_and_referee_calibration_sft", "elitist_120b_teacher", "code_audit_11phase_sft")
ds_cal = load_dataset("krush420/sovereign-code-audit-sft", "default", split="lego_and_referee_calibration_sft")
ds_elitist = load_dataset("krush420/sovereign-code-audit-sft", "default", split="elitist_120b_teacher")

# Load Balanced Referee & Architecture DPO Preference Pairs
ds_dpo = load_dataset("krush420/sovereign-code-audit-sft", "referee_calibration_dpo", split="train")
print(len(ds_cal), len(ds_elitist), len(ds_dpo))
```

### 2. 2-Stage QLoRA SFT + `trl.DPOTrainer` Calibration (`96 GB` Blackwell G4)

```bash
# Track A: Qwen/Qwen3-Coder-30B-A3B-Instruct (8192 ctx, 4-bit NF4 attention + frozen BF16 MoE)
python3 scripts/train_qwen3_30b_qlora_g4.py \
  --mode sft_dpo \
  --dataset data/public/elitist_120b_teacher.jsonl \
  --argon-dataset data/public/argon_golden_remediations.jsonl \
  --calibration-dataset data/public/lego_and_referee_calibration_sft.jsonl \
  --dpo-dataset data/public/referee_calibration_dpo.jsonl \
  --max-seq-length 8192 \
  --output-dir models/qwen3-coder-30b-sre-lora

# Track B: openai/gpt-oss-120b (5120 ctx, frozen 4-bit MXFP4 experts + eager BF16 attention LoRA)
python3 scripts/train_gpt_oss_120b_qlora_g4.py \
  --mode sft_dpo \
  --dataset data/public/elitist_120b_teacher.jsonl \
  --calibration-dataset data/public/lego_and_referee_calibration_sft.jsonl \
  --dpo-dataset data/public/referee_calibration_dpo.jsonl \
  --max-seq-length 5120 \
  --output-dir models/gpt-oss-120b-sre-lora
```
