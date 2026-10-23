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
      - split: elitist_120b_teacher
        path: elitist_120b_teacher.jsonl
      - split: code_audit_11phase_sft
        path: code_audit_11phase_sft.jsonl
---

# Sovereign Code Audit & Bug Remediation SFT Dataset (`krusch-stack`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> Every trajectory in this dataset (`17,899` total SFT examples across 3 splits) was synthesized, AST-grounded, and unit-test verified through **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (`npx krusch-stack init`). It is purpose-built to train open-weights local and burst models that plug directly back into the **KruschStack Cognitive Cascade Router (`L2` & `L2.5`)**.

---

## 🔗 The KruschStack Model & Dataset Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **⚡ 30B Student QLoRA (`L2` Local MoE Tier)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for `Qwen/Qwen3-Coder-30B-A3B-Instruct`)
- **🧠 120B Teacher QLoRA (`L2.5` Burst Tier)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for `openai/gpt-oss-120b`)

---

## 🏗️ How KruschStack Generated & Verified This Dataset

Unlike synthetic datasets generated from ungrounded single-turn prompts, every example in `sovereign-code-audit-sft` was captured from real multi-step execution through the **KruschStack 3-Tier MCP Lifecycle**:

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
│ • <25µs L1 heuristic fast-path -> L2 30B Student -> L2.5 120B -> L3 Referee│
└────────────────────────────────────────────────────────────────────────────┘
```

1. **Tier 1 — Working Memory & Invariants ([`krusch-context-mcp`](https://github.com/kruschdev/krusch-context-mcp)):** Hydrated active architectural invariants (`krusch_context_retrieve`), recorded bug root-causes (`krusch_context_remember`), and enforced pre-commit invariant checks (`krusch_context_nudge`).
2. **Tier 2A — AST Symbol & Blast-Radius Graphs ([`krusch-git`](https://github.com/kruschdev/krusch-git)):** Grounded every audit claim in authoritative AST symbol definitions (`krusch_git_search_symbols`), inbound/outbound caller dependency graphs (`krusch_git_dependency_graph`), and bounded file slices (`krusch_git_read_blob`).
3. **Tier 2B — Page-Anchored OS Memory ([`krusch-fs-mcp`](https://github.com/kruschdev/krusch-fs-mcp)):** Provided deterministic page trees (`krusch_fs_page_tree`) and SHA-256 drift-verified file reads (`krusch_fs_read_page`).
4. **Tier 3 — 2-Phase Commit (2PC) Staged Verification Harness ([`krusch`](https://github.com/kruschdev/krusch)):** Every remediation patch in `argon_golden_remediations` was staged, diff-audited, applied, and verified against live unit test suites (`npm test` / `pytest`, exit code `0`) before entering the training corpus.

---

## 📊 Dataset Splits

| Split Name | File | Rows | Teacher Model | Description |
| :--- | :--- | :---: | :--- | :--- |
| `argon_golden_remediations` | `argon_golden_remediations.jsonl` | **401** | `gemini-4-argon` | **Test-Verified Code Patches & Unit Tests**: Pairs structured KruschStack Master Audit Backlog cards (`BUG-*`, `SEC-*`, `ARCH-*`, `TEST-*`) with complete, unit-test-verified source file patches (`npm test` / `pytest`, exit code `0`) across JavaScript/ESM, TypeScript, React, and Python 3.12 repositories. |
| `elitist_120b_teacher` | `elitist_120b_teacher.jsonl` | **2,208** | `gpt-oss:120b` + `gemini-4-argon` | **Curated High-Signal QLoRA Training Split**: Rejection-sampled KruschStack trajectories (`elitist_score >= 45`) requiring verbatim `file:line` grounding, concrete code blocks/diffs, false-positive refutation (`NOT FLAGGED`), and 2x-oversampled Golden Remediations. Used to train both [`qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) and [`gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora). |
| `code_audit_11phase_sft` | `code_audit_11phase_sft.jsonl` | **15,290** | `gpt-oss:120b` | **Full 11-Phase KruschStack Code Audit Corpus**: Complete phase-by-phase codebase investigations (`0_ORIENTATION` through `2_MASTER_BACKLOG`) using AST symbol search (`search_symbols`, `dependency_graph`), bounded file reads (`read_file path:start-end`), and live test verification. |

---

## 🎯 Core Engineering Capabilities Learned for KruschStack

1. **Verbatim Line-Grounded Auditing (Zero Code Hallucination):** Every claim cites exact `file:startLine-endLine` coordinates verified against `krusch-git` AST symbol graphs and bounded source windows.
2. **Ground-Truth False-Positive Refutation:** Models learn to cross-check prior hypotheses against live source code and explicitly refute invalid or already-resolved findings (`NOT FLAGGED`) instead of echoing them.
3. **Zero-Finding Convergence (`CLEAN_PASS`):** Models learn to emit `0` cards when production code is clean rather than inventing stylistic nitpicks.
4. **Test-Gated Patch & Regression Test Synthesis:** Models learn to resolve P0–P2 bug/security cards while generating passing `node:test` and `pytest` unit tests compatible with `krusch-harness`.

---

## ⚡ Quickstart

### 1. Load Dataset with Hugging Face `datasets`

```python
from datasets import load_dataset

# Load a specific split ("argon_golden_remediations", "elitist_120b_teacher", or "code_audit_11phase_sft")
ds_elitist = load_dataset("krush420/sovereign-code-audit-sft", split="elitist_120b_teacher")
ds_argon = load_dataset("krush420/sovereign-code-audit-sft", split="argon_golden_remediations")
print(ds_elitist[0]["messages"])
```

### 2. Run the Full KruschStack MCP Ecosystem Locally (`npx krusch-stack`)

```bash
# Auto-configure Claude Code (.mcp.json), Cursor, Windsurf & Antigravity + initialize SQLite
npx krusch-stack init

# Verify all 7 stack pillars (39 annotated MCP tools)
npx krusch-stack doctor
```

Learn more at **[github.com/kruschdev/krusch-stack](https://github.com/kruschdev/krusch-stack)**.
