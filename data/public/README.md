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
  - tool-calling
  - arete
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
  - config_name: agentic_tool_calling_sft
    data_files:
      - split: train
        path: agentic_tool_calling_sft.jsonl
  - config_name: arete_values_sft
    data_files:
      - split: train
        path: arete_values_sft.jsonl
  - config_name: referee_calibration_dpo
    data_files:
      - split: train
        path: referee_calibration_dpo.jsonl
  - config_name: agentic_tool_calling_dpo
    data_files:
      - split: train
        path: agentic_tool_calling_dpo.jsonl
  - config_name: arete_values_dpo
    data_files:
      - split: train
        path: arete_values_dpo.jsonl
---

# Sovereign Code Audit, Lego-First, Agentic Tool-Calling & Curated Arete Alignment SFT + DPO Dataset (`krusch-stack`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> Every trajectory and preference pair in this dataset (`18,341` total Zero-Leak records across 9 splits) was synthesized, AST-grounded, or quality-filtered and verified through **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — my Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It powers the **`L1.5` 8B Edge Agent**, **`L2` 30B Local MoE Student (`Wave-7`)**, **`L2.2` 31B Dense Reasoning Specialist**, and **`L2.5` 120B Burst Teacher** inside my **KruschStack Cognitive Cascade Router**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **🥇 `#1 Open-Weights` 31B Dense QLoRA (`L2.2` Tier, `57.33%` DolphinBench)**: [`krush420/gemma4-31b-sre-lora`](https://huggingface.co/krush420/gemma4-31b-sre-lora) (for `google/gemma-4-31b-it`)
- **🪶 8B Edge Agent & Referee LoRA (`L1.5` Local Tier)**: [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) (Built with Meta Llama 3 for `NousResearch/Hermes-3-Llama-3.1-8B`)
- **⚡ 30B Student QLoRA (`L2` Local MoE Tier, `Wave-7` `125.8–131.5 tok/s`)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`)
- **🧠 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `48` Raw JSON + `1` PR #169 Live Receipt Audit Logs across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)

---

## 📊 Dataset Splits

| Config | Split Name | File | Rows | Teacher / Source | Description |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `default` | `argon_golden_remediations` | `argon_golden_remediations.jsonl` | **401** | `gemini-4-argon` | **Test-Verified Code Patches & Unit Tests**: Pairs structured Master Audit Backlog cards (`BUG-*`, `SEC-*`, `ARCH-*`, `TEST-*`) with complete, unit-test-verified source file patches (`npm test` / `pytest`, exit code `0`) across JavaScript/ESM, TypeScript, React, and Python 3.12 repositories. |
| `default` | `lego_and_referee_calibration_sft` | `lego_and_referee_calibration_sft.jsonl` | **38** | `gemini-4-argon` | **Calibrated `CLEAN_PASS`, P0/P1 Remediations, Lego-First & Rule-of-3 Scout Idioms**: Live-source `CLEAN_PASS` Referee convergence verdicts (`findings: []`), ground-truth security/concurrency bug remediations, zero-boilerplate `@krusch/toolkit` v1.8 / `krusch_toolkit` v1.6 synthesis, and cross-repository Rule-of-3 `lego_scout` triage. |
| `agentic_tool_calling_sft` | `train` | `agentic_tool_calling_sft.jsonl` | **76** | `gemini-4-argon` | **Unified Multi-Turn SRE & Workspace Function-Calling SFT**: Structured `tools` + `assistant.tool_calls` + `role: "tool"` trajectories across 3-Tier MCP code audits/2PC staging and generalist workspace operations (partial-briefing gap search, temporal supersession, standing constraints, side-effect execution). |
| `arete_values_sft` | `train` | `arete_values_sft.jsonl` | **140** | [`NousResearch/arete-data-pairs`](https://huggingface.co/datasets/NousResearch/arete-data-pairs) (`Apache-2.0`) | **Curated Zero-Defect Statutory & Analytical Non-Refusal SFT (`20` per category across 7 categories)**: Strictly filtered from `34,878` raw rows requiring `in_sft_accepted=true`, `accuracy=5`, `independence=5`, `helpfulness>=4`, `balance>=4`, `language>=4`, and `0` issue tags (`reasons=[]`). |
| `default` | `elitist_120b_teacher` | `elitist_120b_teacher.jsonl` | **2308** | `gpt-oss:120b` + `gemini-4-argon` | **Curated High-Signal QLoRA Training Split**: Rejection-sampled trajectories requiring verbatim `file:line` grounding, concrete code blocks/diffs, calibrated `CLEAN_PASS` verdicts, Lego-First / Lego-Scout idioms, and 2x-oversampled Golden Remediations. |
| `default` | `code_audit_11phase_sft` | `code_audit_11phase_sft.jsonl` | **15290** | `gpt-oss:120b` | **Full 11-Phase Code Audit Corpus**: Complete phase-by-phase codebase investigations (`0_ORIENTATION` through `2_MASTER_BACKLOG`) using AST symbol search (`search_symbols`, `dependency_graph`), bounded file reads (`read_file path:start-end`), and live test verification. |
| `referee_calibration_dpo` | `train` | `referee_calibration_dpo.jsonl` | **12** | `gemini-4-argon` + `gpt-oss-sre:120b` | **Balanced Referee & Architecture DPO Preference Pairs (`trl.DPOTrainer`)**: Precision pairs, Recall pairs, and Architecture Calibration pairs. |
| `agentic_tool_calling_dpo` | `train` | `agentic_tool_calling_dpo.jsonl` | **34** | `gemini-4-argon` | **Multi-Turn Tool-Action Calibration DPO Pairs**: Penalizes prose-only Markdown dumps when action tools are bound (`action_over_prose`), infinite read-only search loops (`bounded_search_no_loop`), false refusals on benign operational tasks (`benign_execution_no_false_refusal`), and unverified partial-briefing commits (`gap_search_before_commit`). |
| `arete_values_dpo` | `train` | `arete_values_dpo.jsonl` | **42** | [`NousResearch/arete-data-pairs`](https://huggingface.co/datasets/NousResearch/arete-data-pairs) (`Apache-2.0`) | **High-Contrast Non-Refusal & Statutory Independence DPO Pairs (`6` per category across 7 categories)**: Pairs where the starting Qwen checkpoint exhibited explicit refusal, canned evasion, or state-propaganda framing (`rejected`) against rigorous, multi-perspective Inkling teacher analysis (`chosen`, `accuracy>=4`, `independence=5`). |

---

## 🎯 Core Engineering Capabilities Learned

1. **Verbatim Line-Grounded Auditing (Zero Code Hallucination):** Every claim cites exact `file:startLine-endLine` coordinates verified against `krusch-git` AST symbol graphs and bounded source windows.
2. **Ground-Truth False-Positive Refutation & DPO Calibration:** Models cross-check prior hypotheses against live source code and distinguish helper contracts (e.g., HTTP header-map builders) from JSON endpoint envelopes.
3. **Zero-Finding Convergence (`CLEAN_PASS`):** Models emit `findings: []` when production code satisfies frozen invariants rather than inventing stylistic nitpicks.
4. **Lego-First Toolkit Assembly & Rule-of-3 Scouting (`INV-LEGO-01..34`):** Models compose HTTP/SSE/MCP/PG/SQLite/Security/Edge/Webhooks/Idempotency/Audit-Log/RBAC/Pagination/CLI/FastAPI services from verified runtime bricks while enforcing the `>= 3` repository Rule-of-3 anti-bloat gate.
5. **Unified Multi-Turn Function-Calling Agency & Non-Refusal Statutory Analysis (`Wave-7`):** Models preserve native Harmony `<|channel|>commentary`, Gemma 4 `<|channel>thought`, and Qwen3 `<tool_call>` routing alongside Referee auditing and zero-refusal statutory/security compliance analysis (`125.8–131.5 tok/s` on `Qwen3-Coder-30B-A3B`, `5.5x` faster than dense `27B`).
