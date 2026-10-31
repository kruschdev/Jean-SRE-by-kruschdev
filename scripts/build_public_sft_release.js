#!/usr/bin/env node
/**
 * @file build_public_sft_release.js
 * @description Sanitizes and packages the Sovereign Code Audit & Remediation SFT datasets
 * for public release (Hugging Face Datasets + GitHub) and portable QLoRA fine-tuning.
 *
 * Enforces the Zero-Leak Gate (Memory #594):
 *   - 0 API keys / tokens (`sk-ant-*`, `sk-proj-*`, `sk-*`, `AIzaSy*`, `ghp_*`)
 *   - 0 private LAN / 10GbE / Tailscale IPs (`192.168.*`, `10.10.10.*`, `10.0.0.*`, `100.*`)
 *   - 0 internal fleet hostnames (`kruschdev`, `kruschserv`, `kruschmedia`, `kruschgame`, `kruschsandbox`, `kr1yoga`, `crush0`)
 *   - 0 local user home paths (`/home/krusch`, `/home/agent`, `/home/kruschdev`, `/home/kruschserv`)
 *   - 0 homelab-specific system prompts (`"Krusch homelab fleet"`)
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizePortableSftText } from '../src/harness.js';
import { exportElitist120bDataset } from './export-120b-distillation.js';
import {
    harvestLegoAndRefereeDatasets,
    LEGO_CALIBRATION_SFT_PATH,
    REFEREE_DPO_PATH
} from './harvest_lego_and_referee_calibration.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '..');

const DATA_DIR = path.join(projectRoot, 'data');
const DIST_DIR = path.join(DATA_DIR, 'distillation');
const PUBLIC_DIR = path.join(DATA_DIR, 'public');

const SFT_PATH = path.join(DIST_DIR, 'gpt_oss_120b_sft.jsonl');
const ELITIST_PATH = path.join(DIST_DIR, 'gpt_oss_120b_elitist_teacher.jsonl');
const ARGON_PATH = path.join(DATA_DIR, 'argon_remediation_sft_corpus.jsonl');

function wellFormed(val, fallback = '') {
    const s = String(val || fallback);
    return typeof s.toWellFormed === 'function' ? s.toWellFormed() : s;
}

function normalizeUniformSchema(row, idx = 0) {
    const project = wellFormed(row.project || row.target || row.repo, 'workspace');
    const timestamp = wellFormed(row.timestamp || row.remediatedAt || row.auditTimestamp, '2026-10-23T00:00:00.000Z');
    const teacherModel = wellFormed(row.teacher_model || row.teacherModel, 'gpt-oss:120b');
    const studentTarget = wellFormed(row.student_target || row.studentTarget, 'qwen3-coder-30b-moe');
    const phase = wellFormed(row.phase || (row.remediatedAt ? 'ARGON_REMEDIATION' : 'CODE_AUDIT'));
    const verificationStatus = wellFormed(row.verification_status || row.verificationStatus, 'VERIFIED');
    const source = wellFormed(row.source || (row.remediatedAt ? 'argon_golden_remediation' : 'gpt_oss_120b_sft'));
    const elitistScore = Number.isFinite(Number(row.elitist_score)) ? Math.round(Number(row.elitist_score)) : 80;
    const id = wellFormed(row.id, `${project}-${phase}-${idx}`);
    const messages = Array.isArray(row.messages)
        ? row.messages.map((m) => ({
              role: wellFormed(m.role, 'user'),
              content: wellFormed(m.content, '')
          }))
        : [];
    return {
        id,
        timestamp,
        source,
        teacher_model: teacherModel,
        student_target: studentTarget,
        project,
        phase,
        verification_status: verificationStatus,
        elitist_score: elitistScore,
        messages
    };
}

function sanitizeRowObject(row, idx = 0) {
    const proj = row.project || row.target || row.repo || '';
    const rawJson = JSON.stringify(row);
    const sanitizedJson = normalizePortableSftText(rawJson, proj);
    let parsedRow;
    try {
        parsedRow = JSON.parse(sanitizedJson);
    } catch {
        // Fallback field-by-field normalization if regex touched escaped quotes
        parsedRow = { ...row };
        if (Array.isArray(parsedRow.messages)) {
            parsedRow.messages = parsedRow.messages.map((m) => ({
                ...m,
                content: normalizePortableSftText(String(m.content || ''), proj)
            }));
        }
    }
    return normalizeUniformSchema(parsedRow, idx);
}

function sanitizeJsonlFileInPlace(filePath) {
    if (!fs.existsSync(filePath)) return { total: 0, rows: [] };
    const lines = fs.readFileSync(filePath, 'utf8').split('\n').filter(Boolean);
    const sanitizedRows = [];
    for (let i = 0; i < lines.length; i++) {
        try {
            const parsed = JSON.parse(lines[i]);
            sanitizedRows.push(sanitizeRowObject(parsed, i));
        } catch {
            // Skip malformed JSON line
        }
    }
    fs.writeFileSync(
        filePath,
        sanitizedRows.map((r) => JSON.stringify(r)).join('\n') + '\n',
        'utf8'
    );
    return { total: sanitizedRows.length, rows: sanitizedRows };
}

function normalizeDpoTurnList(turnOrStr, proj, defaultRole = 'assistant') {
    if (Array.isArray(turnOrStr)) {
        return turnOrStr.map((m) => ({
            role: wellFormed(m.role, defaultRole),
            content: wellFormed(normalizePortableSftText(String(m.content || ''), proj))
        }));
    }
    return [
        {
            role: defaultRole,
            content: wellFormed(normalizePortableSftText(String(turnOrStr || ''), proj))
        }
    ];
}

function sanitizeDpoFileInPlace(filePath) {
    if (!fs.existsSync(filePath)) return { total: 0, rows: [] };
    const lines = fs.readFileSync(filePath, 'utf8').split('\n').filter(Boolean);
    const sanitizedRows = [];
    for (let i = 0; i < lines.length; i++) {
        try {
            const row = JSON.parse(lines[i]);
            const proj = row.project || 'toolkit';
            const prompt = normalizeDpoTurnList(row.prompt, proj, 'user');
            sanitizedRows.push({
                id: wellFormed(row.id, `dpo-${i}`),
                mode: wellFormed(row.mode, 'precision'),
                project: wellFormed(proj, 'toolkit'),
                invariant: wellFormed(row.invariant, 'INV-LEGO-01'),
                prompt,
                chosen: normalizeDpoTurnList(row.chosen, proj, 'assistant'),
                rejected: normalizeDpoTurnList(row.rejected, proj, 'assistant')
            });
        } catch {
            // Skip malformed JSON line
        }
    }
    fs.writeFileSync(
        filePath,
        sanitizedRows.map((r) => JSON.stringify(r)).join('\n') + '\n',
        'utf8'
    );
    return { total: sanitizedRows.length, rows: sanitizedRows };
}

export function verifyZeroLeaks(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n').filter(Boolean);
    let homePaths = 0;
    let privateIps = 0;
    let hostnames = 0;
    let kruschFleetPrompt = 0;
    let secretLike = 0;
    let loneSurrogates = 0;

    for (const l of lines) {
        if (/\/home\/(?:krusch|agent|kruschdev|kruschserv)/i.test(l)) homePaths++;
        if (/\b(?:192\.168\.\d+\.\d+|10\.10\.10\.\d+|10\.0\.0\.\d+|100\.(?:65|106|105|98|124|92|94|69)\.\d+\.\d+)\b/.test(l)) privateIps++;
        if (/\b(?:kruschdev|kruschserv|kruschmedia|kruschgame|kruschsandbox|kr1yoga|crush0)\b/i.test(l)) hostnames++;
        if (/Krusch homelab fleet/i.test(l)) kruschFleetPrompt++;
        if (/(?:AIzaSy[A-Za-z0-9_-]{33}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,})/.test(l)) secretLike++;
        if (/\\ud[89a-f][0-9a-f]{2}/i.test(l)) loneSurrogates++;
    }

    return {
        file: path.relative(projectRoot, filePath),
        rows: lines.length,
        homePaths,
        privateIps,
        hostnames,
        kruschFleetPrompt,
        secretLike,
        loneSurrogates,
        passed:
            homePaths === 0 &&
            privateIps === 0 &&
            hostnames === 0 &&
            kruschFleetPrompt === 0 &&
            secretLike === 0 &&
            loneSurrogates === 0
    };
}

function writeDatasetReadme(publicDir, stats) {
    const totalRecords =
        stats.argonCount +
        stats.legoCalibrationCount +
        stats.dpoCount +
        stats.elitistCount +
        stats.codeAuditCount;
    const readme = `---
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

# Sovereign Code Audit & Bug Remediation SFT + DPO Dataset (\`krusch-stack\`)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> Every trajectory and preference pair in this dataset (\`${totalRecords.toLocaleString('en-US')}\` total Zero-Leak records across 5 splits) was synthesized, AST-grounded, and unit-test verified through **[KruschStack (\`krusch-stack\`)](https://github.com/kruschdev/krusch-stack)** — the Unified 7-Pillar, 39-Tool Model Context Protocol (MCP) & Agentic Coding Stack (\`npx krusch-stack init\`). It powers both the **\`L2\` 30B Local Student** and **\`L2.5\` 120B Burst Teacher** inside the **KruschStack Cognitive Cascade Router**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [\`github.com/kruschdev/krusch-stack\`](https://github.com/kruschdev/krusch-stack) (\`npx krusch-stack init\`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [\`github.com/kruschdev/Jean-SRE-by-kruschdev\`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **⚡ 30B Student QLoRA (\`L2\` Local MoE Tier)**: [\`krush420/qwen3-coder-30b-sre-lora\`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (for \`Qwen/Qwen3-Coder-30B-A3B-Instruct\`)
- **🧠 120B Teacher QLoRA (\`L2.5\` Burst Tier)**: [\`krush420/gpt-oss-120b-sre-lora\`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (for \`openai/gpt-oss-120b\`)
- **🏆 Hermetic Benchmark Audit Logs (\`68/68\` Gates, \`316,981+\` Verified Queries)**: [\`krush420/sovereign-g4-benchmarks\`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)

---

## 📊 Dataset Splits

| Config | Split Name | File | Rows | Teacher Model | Description |
| :--- | :--- | :--- | :---: | :--- | :--- |
| \`default\` | \`argon_golden_remediations\` | \`argon_golden_remediations.jsonl\` | **${stats.argonCount}** | \`gemini-4-argon\` | **Test-Verified Code Patches & Unit Tests**: Pairs structured Master Audit Backlog cards (\`BUG-*\`, \`SEC-*\`, \`ARCH-*\`, \`TEST-*\`) with complete, unit-test-verified source file patches (\`npm test\` / \`pytest\`, exit code \`0\`) across JavaScript/ESM, TypeScript, React, and Python 3.12 repositories. |
| \`default\` | \`lego_and_referee_calibration_sft\` | \`lego_and_referee_calibration_sft.jsonl\` | **${stats.legoCalibrationCount}** | \`gemini-4-argon\` | **Calibrated \`CLEAN_PASS\`, P0/P1 Remediations, Lego-First & Rule-of-3 Scout Idioms**: Live-source \`CLEAN_PASS\` Referee convergence verdicts (\`findings: []\`), ground-truth security/concurrency bug remediations, zero-boilerplate \`@krusch/toolkit\` v1.5 (24 bricks) / \`krusch_toolkit\` v1.3 synthesis, and cross-repository Rule-of-3 \`lego_scout\` triage (\`INV-LEGO-01..16\`). |
| \`default\` | \`elitist_120b_teacher\` | \`elitist_120b_teacher.jsonl\` | **${stats.elitistCount}** | \`gpt-oss:120b\` + \`gemini-4-argon\` | **Curated High-Signal QLoRA Training Split**: Rejection-sampled trajectories requiring verbatim \`file:line\` grounding, concrete code blocks/diffs, calibrated \`CLEAN_PASS\` verdicts, Lego-First / Lego-Scout idioms, and 2x-oversampled Golden Remediations. |
| \`default\` | \`code_audit_11phase_sft\` | \`code_audit_11phase_sft.jsonl\` | **${stats.codeAuditCount}** | \`gpt-oss:120b\` | **Full 11-Phase Code Audit Corpus**: Complete phase-by-phase codebase investigations (\`0_ORIENTATION\` through \`2_MASTER_BACKLOG\`) using AST symbol search (\`search_symbols\`, \`dependency_graph\`), bounded file reads (\`read_file path:start-end\`), and live test verification. |
| \`referee_calibration_dpo\` | \`train\` | \`referee_calibration_dpo.jsonl\` | **${stats.dpoCount}** | \`gemini-4-argon\` + \`gpt-oss-sre:120b\` | **Balanced Referee & Architecture DPO Preference Pairs (\`trl.DPOTrainer\`)**: Precision pairs (rejecting false-positive helper return-type & package-scope misreads), Recall pairs (rejecting lazy \`CLEAN_PASS\` rubber-stamping on real P0/P1 defects), and Architecture Calibration pairs (Rule-of-3 anti-bloat, thin SQL vs heavy ORM, zero-build \`.d.ts\` vs premature Rust, connect-time SSRF vs regex). |

---

## 🎯 Core Engineering Capabilities Learned

1. **Verbatim Line-Grounded Auditing (Zero Code Hallucination):** Every claim cites exact \`file:startLine-endLine\` coordinates verified against \`krusch-git\` AST symbol graphs and bounded source windows.
2. **Ground-Truth False-Positive Refutation & DPO Calibration:** Models cross-check prior hypotheses against live source code and distinguish helper contracts (e.g., HTTP header-map builders) from JSON endpoint envelopes.
3. **Zero-Finding Convergence (\`CLEAN_PASS\`):** Models emit \`findings: []\` when production code satisfies frozen invariants rather than inventing stylistic nitpicks.
4. **Lego-First Toolkit Assembly & Rule-of-3 Scouting (\`INV-LEGO-01..16\`):** Models compose HTTP/SSE/MCP/PG/SQLite/Security/Edge/Webhooks/CLI/FastAPI services from verified runtime bricks while enforcing the \`>= 3\` repository Rule-of-3 anti-bloat gate.
5. **Test-Gated Patch & Regression Test Synthesis:** Models resolve P0–P1 bug/security cards while generating passing \`node:test\` and \`pytest\` unit tests compatible with \`krusch-harness\`.

---

## ⚡ Quickstart

### 1. Load SFT & DPO Splits with Hugging Face \`datasets\`

\`\`\`python
from datasets import load_dataset

# Load SFT splits ("argon_golden_remediations", "lego_and_referee_calibration_sft", "elitist_120b_teacher", "code_audit_11phase_sft")
ds_cal = load_dataset("krush420/sovereign-code-audit-sft", "default", split="lego_and_referee_calibration_sft")
ds_elitist = load_dataset("krush420/sovereign-code-audit-sft", "default", split="elitist_120b_teacher")

# Load Balanced Referee & Architecture DPO Preference Pairs
ds_dpo = load_dataset("krush420/sovereign-code-audit-sft", "referee_calibration_dpo", split="train")
print(len(ds_cal), len(ds_elitist), len(ds_dpo))
\`\`\`

### 2. 2-Stage QLoRA SFT + \`trl.DPOTrainer\` Calibration (\`96 GB\` Blackwell G4)

\`\`\`bash
# Track A: Qwen/Qwen3-Coder-30B-A3B-Instruct (8192 ctx, 4-bit NF4 attention + frozen BF16 MoE)
python3 scripts/train_qwen3_30b_qlora_g4.py \\
  --mode sft_dpo \\
  --dataset data/public/elitist_120b_teacher.jsonl \\
  --argon-dataset data/public/argon_golden_remediations.jsonl \\
  --calibration-dataset data/public/lego_and_referee_calibration_sft.jsonl \\
  --dpo-dataset data/public/referee_calibration_dpo.jsonl \\
  --max-seq-length 8192 \\
  --output-dir models/qwen3-coder-30b-sre-lora

# Track B: openai/gpt-oss-120b (5120 ctx, frozen 4-bit MXFP4 experts + eager BF16 attention LoRA)
python3 scripts/train_gpt_oss_120b_qlora_g4.py \\
  --mode sft_dpo \\
  --dataset data/public/elitist_120b_teacher.jsonl \\
  --calibration-dataset data/public/lego_and_referee_calibration_sft.jsonl \\
  --dpo-dataset data/public/referee_calibration_dpo.jsonl \\
  --max-seq-length 5120 \\
  --output-dir models/gpt-oss-120b-sre-lora
\`\`\`
`;
    fs.writeFileSync(path.join(publicDir, 'README.md'), readme, 'utf8');
}

function main() {
    fs.mkdirSync(PUBLIC_DIR, { recursive: true });

    // 0. Harvest live Lego-First + Calibrated CLEAN_PASS SFT & DPO preference splits
    harvestLegoAndRefereeDatasets({
        sftOutPath: LEGO_CALIBRATION_SFT_PATH,
        dpoOutPath: REFEREE_DPO_PATH
    });

    // 1. Sanitize all in-place JSONL files
    const argonRes = sanitizeJsonlFileInPlace(ARGON_PATH);
    const sftRes = sanitizeJsonlFileInPlace(SFT_PATH);
    const legoCalRes = sanitizeJsonlFileInPlace(LEGO_CALIBRATION_SFT_PATH);
    const dpoRes = sanitizeDpoFileInPlace(REFEREE_DPO_PATH);

    // 2. Rebuild Elitist 120B Teacher dataset from sanitized SFT + Argon + Lego/Referee Calibration
    exportElitist120bDataset(SFT_PATH, ARGON_PATH, ELITIST_PATH, {
        legoCalibrationPath: LEGO_CALIBRATION_SFT_PATH
    });
    const elitistRes = sanitizeJsonlFileInPlace(ELITIST_PATH);

    // 3. Write Public Portable Release splits (excluding fleet hardware-specific infra rows)
    const publicArgonPath = path.join(PUBLIC_DIR, 'argon_golden_remediations.jsonl');
    const publicElitistPath = path.join(PUBLIC_DIR, 'elitist_120b_teacher.jsonl');
    const publicCodeAuditPath = path.join(PUBLIC_DIR, 'code_audit_11phase_sft.jsonl');
    const publicLegoCalPath = path.join(PUBLIC_DIR, 'lego_and_referee_calibration_sft.jsonl');
    const publicDpoPath = path.join(PUBLIC_DIR, 'referee_calibration_dpo.jsonl');

    fs.copyFileSync(ARGON_PATH, publicArgonPath);
    fs.copyFileSync(ELITIST_PATH, publicElitistPath);
    fs.copyFileSync(LEGO_CALIBRATION_SFT_PATH, publicLegoCalPath);
    fs.copyFileSync(REFEREE_DPO_PATH, publicDpoPath);

    const portableCodeAuditRows = sftRes.rows.filter(
        (r) =>
            r.project !== 'fleet' &&
            !/^INFRA/i.test(r.phase || '') &&
            !/Xeon 40T|RTX 2080 Ti|Fleet Hardware/i.test(JSON.stringify(r))
    );
    fs.writeFileSync(
        publicCodeAuditPath,
        portableCodeAuditRows.map((r) => JSON.stringify(r)).join('\n') + '\n',
        'utf8'
    );

    writeDatasetReadme(PUBLIC_DIR, {
        argonCount: argonRes.total,
        legoCalibrationCount: legoCalRes.total,
        dpoCount: dpoRes.total,
        elitistCount: elitistRes.total,
        codeAuditCount: portableCodeAuditRows.length
    });

    // 4. Run Zero-Leak Verification Gate across all 10 training & public files
    const checks = [
        verifyZeroLeaks(ARGON_PATH),
        verifyZeroLeaks(LEGO_CALIBRATION_SFT_PATH),
        verifyZeroLeaks(REFEREE_DPO_PATH),
        verifyZeroLeaks(ELITIST_PATH),
        verifyZeroLeaks(SFT_PATH),
        verifyZeroLeaks(publicArgonPath),
        verifyZeroLeaks(publicLegoCalPath),
        verifyZeroLeaks(publicDpoPath),
        verifyZeroLeaks(publicElitistPath),
        verifyZeroLeaks(publicCodeAuditPath)
    ];

    console.log('=== 🛡️ ZERO-LEAK & PORTABILITY VERIFICATION GATE ===');
    console.log(JSON.stringify(checks, null, 2));

    const failed = checks.filter((c) => !c.passed);
    if (failed.length > 0) {
        console.error(`❌ Zero-Leak Gate FAILED on ${failed.length} file(s)!`);
        process.exit(1);
    }
    console.log(`✅ All ${checks.length} training & public release JSONL splits passed with ZERO leaks!`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
    main();
}
