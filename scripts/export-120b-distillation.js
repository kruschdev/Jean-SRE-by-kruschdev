#!/usr/bin/env node
/**
 * Teacher-to-Student Knowledge Distillation & SFT Dataset Exporter
 *
 * 1. Scans all audited projects under `<HOMELAB_ROOT>/projects/<proj>/.agent/audit/`
 *    and generates `teacher_120b_distilled.json` for runtime 30B Student injection.
 * 2. Extracts completed 120B audit phase turns (`phase_0_orientation.md`,
 *    `phase_1_investigation.md`, `phase_2_adversarial.md`, `master_backlog.md`)
 *    into `data/distillation/gpt_oss_120b_sft.jsonl` in ChatML format for
 *    QLoRA fine-tuning of `Qwen/Qwen3-Coder-30B-A3B-Instruct` (`qwen3-coder-30b-moe`).
 */

import fs from 'fs';
import { dirname, join } from 'path';
import os from 'os';
import { fileURLToPath } from 'url';
import {
    distillTeacherKnowledgeForStudent,
    buildProjectAuditContext
} from '../src/jean_sre_sweep.js';
import { normalizePortableSftText } from '../src/harness.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');
const HOMELAB_ROOT = (process.env.HOMELAB_PATH && fs.existsSync(process.env.HOMELAB_PATH))
    ? process.env.HOMELAB_PATH
    : join(projectRoot, '..', '..');

const SYSTEM_PROMPT = [
    'You are an Autonomous Principal Software Engineer, SRE, and Code Auditor.',
    'RULES (NON-NEGOTIABLE):',
    '1. Ground every finding strictly in the provided live source snippets, AST symbol graphs, and command outputs.',
    '2. NEVER invent, paraphrase, or reconstruct code snippets from memory. Only quote verbatim lines that appear in the prompt with their exact line numbers.',
    '3. If a prior Pending Issue is not supported by the Live Source Verification lines shown in the prompt, REFUTE and remove it.',
    '4. When audited files satisfy all frozen specification invariants with zero P0/P1 defects, emit a calibrated CLEAN_PASS verdict ("findings": []) rather than inventing stylistic nitpicks.',
    '5. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.'
].join('\n');

function extractPhaseBlocksFromCheckpoint(rawMarkdown) {
    const blocks = [];
    const regex = /## 📝 Audit Phase:\s*([0-9A-Z._-]+)\s*\(([^)]+)\)\s*\n+([\s\S]*?)(?=\n## 📝 Audit Phase:|\n---\s*$|$)/g;
    let m;
    while ((m = regex.exec(String(rawMarkdown || ''))) !== null) {
        const phase = m[1].trim();
        const timestamp = m[2].trim();
        const body = m[3].replace(/\n---\s*$/, '').trim();
        if (body.length >= 160 && !body.includes('HALLUCINATION FLAG')) {
            blocks.push({ phase, timestamp, body });
        }
    }
    return blocks;
}

export function normalizeExistingJsonlFile(jsonlPath) {
    if (!fs.existsSync(jsonlPath)) return 0;
    const rawLines = fs.readFileSync(jsonlPath, 'utf8').split('\n').filter(Boolean);
    let mutatedCount = 0;
    const normalizedLines = rawLines.map((line) => {
        try {
            const row = JSON.parse(line);
            const proj = row.project || row.target || row.repo || '';
            let changed = false;
            if (Array.isArray(row.messages)) {
                row.messages = row.messages.map((m) => {
                    const norm = normalizePortableSftText(m.content || '', proj);
                    if (norm !== m.content) changed = true;
                    return { ...m, content: norm };
                });
            }
            if (changed) mutatedCount++;
            return JSON.stringify(row);
        } catch {
            return line;
        }
    });
    if (mutatedCount > 0) {
        fs.writeFileSync(jsonlPath, `${normalizedLines.join('\n')}\n`, 'utf8');
    }
    return mutatedCount;
}

export function scoreElitistTrajectory(row) {
    if (!row || !Array.isArray(row.messages) || row.messages.length < 3) return 0;
    const assistant = String(row.messages[2]?.content || '');
    if (assistant.includes('HALLUCINATION FLAG') || assistant.includes('<think>')) return 0;

    const isCalibratedCleanPass =
        row.source === 'referee_clean_pass_calibration' ||
        (/"verdict"\s*:\s*"CLEAN_PASS"/.test(assistant) && /"findings"\s*:\s*\[\s*\]/.test(assistant));
    const isLegoFirstSft =
        row.source === 'lego_first_sft' || /@krusch\/toolkit\/|krusch_toolkit\./.test(assistant);
    const minAssistantChars = isCalibratedCleanPass || row.source === 'lego_first_sft' ? 120 : 400;
    if (assistant.length < minAssistantChars) return 0;

    let score = 0;
    // 1. Argon-verified + unit-test-gated golden remediations get maximum priority
    if (row.source === 'argon_golden_remediation' || String(row.teacher_model || '').includes('argon')) {
        score += 10;
    }
    // 2. Calibrated CLEAN_PASS verdicts & Lego-First toolkit assembly get explicit positive weight
    if (isCalibratedCleanPass) {
        score += 6;
    }
    if (isLegoFirstSft) {
        score += 3;
    }

    // 3. Verbatim file:line citations (e.g., src/foo.js:42 or #L10-L25)
    const lineCites = assistant.match(/[a-zA-Z0-9_./-]+(?::\d+|#L\d+)/g) || [];
    if (lineCites.length >= 3) score += 3;
    else if (lineCites.length >= 1) score += 1;

    // 4. Concrete fenced code blocks or unified diffs
    const codeBlocks = assistant.match(/```[\s\S]*?```/g) || [];
    if (codeBlocks.length >= 2) score += 2;
    else if (codeBlocks.length === 1) score += 1;

    // 5. Structured audit IDs / explicit refutations / invariant checks (INV-12, INV-LEGO-01, INV-STACK-04)
    if (/\[(?:BUG|SEC|PERF|ARCH|TEST|SRE)-\d+\]/.test(assistant)) score += 2;
    if (/\b(?:REFUTED|VERIFIED|CLEAN_PASS|INV-(?:[A-Z]+-)?\d+)\b/.test(assistant)) score += 2;

    // 6. Downweight legacy hand-rolled DB/SSE boilerplate when @krusch/toolkit equivalents are absent
    if (
        (/\bnew\s+(?:pg\.)?Pool\s*\(/.test(assistant) && !/@krusch\/toolkit\/pg|INV-LEGO/.test(assistant)) ||
        (/text\/event-stream/.test(assistant) && /res\.writeHead/.test(assistant) && !/@krusch\/toolkit\/streaming|INV-LEGO/.test(assistant))
    ) {
        score = Math.max(0, score - 2);
    }

    return score;
}

export function exportElitist120bDataset(sftPath, argonPath, elitistOutPath, options = {}) {
    const selected = [];
    const seenSigs = new Set();
    const legoCalibrationPath =
        options.legoCalibrationPath ??
        join(dirname(elitistOutPath), 'lego_and_referee_calibration_sft.jsonl');

    // 1. Always include all Argon Golden Remediation rows (oversampled 2x for 120B DPO/SFT grounding)
    if (fs.existsSync(argonPath)) {
        for (const line of fs.readFileSync(argonPath, 'utf8').split('\n').filter(Boolean)) {
            try {
                const row = JSON.parse(line);
                if (Array.isArray(row.messages) && row.messages.length >= 3) {
                    const tagged = {
                        ...row,
                        source: row.source || 'argon_golden_remediation',
                        elitist_score: Math.max(10, scoreElitistTrajectory(row))
                    };
                    selected.push(JSON.stringify(tagged));
                    selected.push(JSON.stringify(tagged));
                }
            } catch {}
        }
    }

    // 1b. Include calibrated CLEAN_PASS + P0/P1 Remediation + Lego-First SFT rows (oversampled 2x)
    if (legoCalibrationPath && fs.existsSync(legoCalibrationPath)) {
        for (const line of fs.readFileSync(legoCalibrationPath, 'utf8').split('\n').filter(Boolean)) {
            try {
                const row = JSON.parse(line);
                if (Array.isArray(row.messages) && row.messages.length >= 3) {
                    const score = Math.max(12, scoreElitistTrajectory(row));
                    const tagged = {
                        ...row,
                        elitist_score: score
                    };
                    selected.push(JSON.stringify(tagged));
                    selected.push(JSON.stringify(tagged));
                }
            } catch {}
        }
    }

    // 2. Rejection-sample the 120B SFT corpus (require score >= 5, cap per project:phase to top 4)
    const bucketCounts = new Map();
    if (fs.existsSync(sftPath)) {
        const candidates = [];
        for (const line of fs.readFileSync(sftPath, 'utf8').split('\n').filter(Boolean)) {
            try {
                const row = JSON.parse(line);
                if (row.project === 'fleet' || /^INFRA/i.test(row.phase || '')) continue;
                const score = scoreElitistTrajectory(row);
                if (score >= 5) {
                    candidates.push({ row, score });
                }
            } catch {}
        }
        candidates.sort((a, b) => b.score - a.score);
        for (const { row, score } of candidates) {
            const bucketKey = `${row.project || 'unknown'}:${row.phase || 'unknown'}`;
            const count = bucketCounts.get(bucketKey) || 0;
            if (count >= 4) continue;
            const sig = `${bucketKey}:${String(row.messages?.[2]?.content || '').slice(0, 100)}`;
            if (seenSigs.has(sig)) continue;
            seenSigs.add(sig);
            bucketCounts.set(bucketKey, count + 1);
            selected.push(JSON.stringify({ ...row, elitist_score: score }));
        }
    }

    fs.writeFileSync(elitistOutPath, `${selected.join('\n')}\n`, 'utf8');
    return selected.length;
}

function main() {
    const projectsDir = join(HOMELAB_ROOT, 'projects');
    if (!fs.existsSync(projectsDir)) {
        console.error(`Projects directory not found: ${projectsDir}`);
        process.exit(1);
    }

    const distDir = join(projectRoot, 'data', 'distillation');
    fs.mkdirSync(distDir, { recursive: true });
    const sftPath = join(distDir, 'gpt_oss_120b_sft.jsonl');
    const elitistPath = join(distDir, 'gpt_oss_120b_elitist_teacher.jsonl');

    // Load existing SFT keys to avoid duplicate rows on re-runs
    const existingKeys = new Set();
    if (fs.existsSync(sftPath)) {
        for (const line of fs.readFileSync(sftPath, 'utf8').split('\n').filter(Boolean)) {
            try {
                const row = JSON.parse(line);
                const sig = `${row.project}:${row.phase}:${(row.messages?.[2]?.content || '').slice(0, 80)}`;
                existingKeys.add(sig);
            } catch {}
        }
    }

    let distilledProjects = 0;
    let addedSftRows = 0;
    const entries = fs.readdirSync(projectsDir, { withFileTypes: true })
        .filter(d => d.isDirectory() && d.name !== 'archive')
        .map(d => ({ name: d.name, path: join(projectsDir, d.name) }));

    const rootTargets = ['lib', 'lib-py', 'scripts', '.agent', 'configs']
        .map(name => ({ name: `@homelab/${name}`, path: join(HOMELAB_ROOT, name) }))
        .filter(t => fs.existsSync(t.path));

    for (const target of [...entries, ...rootTargets]) {
        const projName = target.name;
        const projPath = target.path;
        const auditDir = join(projPath, '.agent', 'audit');
        if (!fs.existsSync(auditDir)) continue;

        const distilled = distillTeacherKnowledgeForStudent(projPath, { teacherModel: 'gpt-oss:120b' });
        if (distilled) {
            distilledProjects++;
            console.log(`[Distill] ✅ ${projName}: ${distilled.refutedFalsePositives.length} refuted traps, ${distilled.verifiedBacklog.length} verified cards, ${distilled.goldenToolTrajectories.length} tool trajectories`);
        }

        const checkpointFiles = [
            'phase_0_orientation.md',
            'phase_1_investigation.md',
            'phase_2_adversarial.md',
            'master_backlog.md'
        ];
        for (const cfile of checkpointFiles) {
            const cpath = join(auditDir, cfile);
            if (!fs.existsSync(cpath)) continue;
            const raw = fs.readFileSync(cpath, 'utf8');
            const blocks = extractPhaseBlocksFromCheckpoint(raw);
            for (const b of blocks) {
                const normBody = normalizePortableSftText(b.body, projName);
                const sig = `${projName}:${b.phase}:${normBody.slice(0, 80)}`;
                if (existingKeys.has(sig)) continue;
                existingKeys.add(sig);

                // Build a compact Student-budgeted user prompt so SFT examples fit inside 6K tokens
                let studentContext = '';
                try {
                    process.env.SKIP_LIVE_TEST_EXECUTION = '1';
                    studentContext = buildProjectAuditContext(projPath, b.phase, '', { isColabBurst: false }).slice(0, 9500);
                } catch {}

                const rawUserPrompt = [
                    `## 🛑 JEAN — AUTONOMOUS CODE AUDIT`,
                    `- **Target Project:** \`${projName}\``,
                    `- **Phase:** \`${b.phase}\``,
                    studentContext,
                    `Produce your grounded Phase \`${b.phase}\` synthesis strictly citing verbatim file:line evidence.`
                ].filter(Boolean).join('\n\n');

                const record = {
                    timestamp: b.timestamp || new Date().toISOString(),
                    teacher_model: 'gpt-oss:120b',
                    student_target: 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
                    project: projName,
                    phase: b.phase,
                    messages: [
                        { role: 'system', content: SYSTEM_PROMPT },
                        { role: 'user', content: normalizePortableSftText(rawUserPrompt, projName) },
                        { role: 'assistant', content: normBody }
                    ]
                };
                fs.appendFileSync(sftPath, `${JSON.stringify(record)}\n`, 'utf8');
                addedSftRows++;
            }
        }
    }

    const normalizedSft = normalizeExistingJsonlFile(sftPath);
    const argonPath = join(projectRoot, 'data', 'argon_remediation_sft_corpus.jsonl');
    const normalizedArgon = normalizeExistingJsonlFile(argonPath);
    const elitistCount = exportElitist120bDataset(sftPath, argonPath, elitistPath);

    const totalLines = fs.existsSync(sftPath)
        ? fs.readFileSync(sftPath, 'utf8').split('\n').filter(Boolean).length
        : 0;
    console.log(`\n[Summary] Distilled ${distilledProjects} targets | Added ${addedSftRows} new SFT examples | Normalized ${normalizedSft} SFT + ${normalizedArgon} Argon rows (${totalLines} total Student SFT | ${elitistCount} Elitist 120B Teacher rows)`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    main();
}

