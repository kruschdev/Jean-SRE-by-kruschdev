import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    harvestLegoAndRefereeDatasets,
    LEGO_CALIBRATION_SFT_PATH,
    REFEREE_DPO_PATH
} from '../scripts/harvest_lego_and_referee_calibration.js';
import {
    scoreElitistTrajectory,
    exportElitist120bDataset
} from '../scripts/export-120b-distillation.js';
import { verifyZeroLeaks } from '../scripts/build_public_sft_release.js';
import { getInfraChecklist } from '../src/prompts.js';
import {
    INFRA_TARGETS,
    buildLegoScoutContextBrief,
    buildProjectLegoBoilerplateBrief,
    appendLegoScoutBacklogEntry,
    parseMasterBacklogCards
} from '../src/jean_sre_sweep.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function extractFencedJson(markdown) {
    const match = /```json\s*([\s\S]*?)```/.exec(String(markdown || ''));
    assert.ok(match, `Expected fenced json block in: ${String(markdown).slice(0, 120)}`);
    return JSON.parse(match[1]);
}

describe('Lego-First & Calibrated Referee SFT + DPO Pipeline', () => {
    test('harvestLegoAndRefereeDatasets generates balanced Zero-Leak Wave-3 SFT and conversational DPO splits', () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-sft-cal-'));
        try {
            const sftOutPath = path.join(tmpDir, 'lego_cal_sft.jsonl');
            const dpoOutPath = path.join(tmpDir, 'referee_dpo.jsonl');
            const res = harvestLegoAndRefereeDatasets({
                sftOutPath,
                dpoOutPath,
                nowIso: '2026-10-30T18:00:00.000Z'
            });

            assert.equal(res.ok, true);
            assert.equal(res.data.counts.sftTotal, 38);
            assert.equal(res.data.counts.cleanPass, 10);
            assert.equal(res.data.counts.remediations, 10);
            assert.equal(res.data.counts.legoFirst, 14);
            assert.equal(res.data.counts.legoScout, 4);
            assert.equal(res.data.counts.dpoTotal, 12);
            assert.equal(res.data.counts.dpoPrecision, 4);
            assert.equal(res.data.counts.dpoRecall, 4);
            assert.equal(res.data.counts.dpoArchitecture, 4);

            const sftRows = fs
                .readFileSync(sftOutPath, 'utf8')
                .trim()
                .split('\n')
                .map((l) => JSON.parse(l));
            assert.equal(sftRows.length, 38);

            const cleanPassRows = sftRows.filter((r) => r.source === 'referee_clean_pass_calibration');
            assert.equal(cleanPassRows.length, 10);
            for (const row of cleanPassRows) {
                const parsed = extractFencedJson(row.messages[2].content);
                assert.equal(parsed.verdict, 'CLEAN_PASS');
                assert.deepEqual(parsed.findings, []);
                assert.ok(scoreElitistTrajectory(row) >= 12);
            }

            const legoFirstRows = sftRows.filter((r) => r.source === 'lego_first_sft');
            assert.equal(legoFirstRows.length, 14);
            for (const row of legoFirstRows) {
                const content = row.messages[2].content;
                assert.equal(
                    /initDb\([^)]*schema\s*:|inputType\s*:\s*['"]query['"]|(?:[{,]|\n)\s*queryVector\s*[,:}]|createMcpServer\s*\(\s*\{|stream\.streaming\b|useApiQuery\(\s*['"][^'"]+['"]\s*,\s*\[/.test(
                        content
                    ),
                    false,
                    `Unexpected drifted @krusch/toolkit API signature in ${row.id}`
                );
            }

            const scoutRows = sftRows.filter((r) => r.source === 'lego_scout_synthesis_sft');
            assert.equal(scoutRows.length, 4);
            for (const row of scoutRows) {
                assert.ok(scoreElitistTrajectory(row) >= 14, `Expected scout SFT score >= 14 for ${row.id}`);
            }

            const dpoRows = fs
                .readFileSync(dpoOutPath, 'utf8')
                .trim()
                .split('\n')
                .map((l) => JSON.parse(l));
            assert.equal(dpoRows.length, 12);
            for (const pair of dpoRows) {
                assert.ok(Array.isArray(pair.prompt) && pair.prompt.length >= 2);
                assert.ok(Array.isArray(pair.chosen) && pair.chosen.length === 1);
                assert.ok(Array.isArray(pair.rejected) && pair.rejected.length === 1);
                assert.equal(pair.chosen[0].role, 'assistant');
                assert.equal(pair.rejected[0].role, 'assistant');
                assert.notEqual(pair.chosen[0].content, pair.rejected[0].content);

                const chosenJson = extractFencedJson(pair.chosen[0].content);
                const rejectedJson = extractFencedJson(pair.rejected[0].content);
                if (pair.mode === 'precision') {
                    assert.equal(chosenJson.verdict, 'CLEAN_PASS');
                    assert.equal(rejectedJson.verdict, 'DEFECTS_FOUND');
                } else if (pair.mode === 'recall') {
                    assert.equal(chosenJson.verdict, 'DEFECTS_FOUND');
                    assert.equal(rejectedJson.verdict, 'CLEAN_PASS');
                } else {
                    assert.equal(pair.mode, 'architecture_calibration');
                    assert.notEqual(chosenJson.verdict, rejectedJson.verdict);
                    assert.ok(String(chosenJson.invariant || '').startsWith('INV-LEGO-'));
                }
            }

            assert.equal(verifyZeroLeaks(sftOutPath).passed, true);
            assert.equal(verifyZeroLeaks(dpoOutPath).passed, true);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    test('scoreElitistTrajectory rewards concise CLEAN_PASS verdicts, Lego-First bricks & LEGO-* cards while penalizing raw boilerplate', () => {
        const shortCleanPass = {
            messages: [
                { role: 'system', content: 'Auditor' },
                { role: 'user', content: 'Audit lib/http.js:126' },
                {
                    role: 'assistant',
                    content:
                        '```json\n{\n  "verdict": "CLEAN_PASS",\n  "checked_files": ["lib/http.js"],\n  "not_flagged_notes": ["Satisfies INV-LEGO-01"],\n  "findings": []\n}\n```'
                }
            ]
        };
        assert.ok(shortCleanPass.messages[2].content.length < 400);
        const cleanScore = scoreElitistTrajectory(shortCleanPass);
        assert.ok(cleanScore >= 8, `Expected short CLEAN_PASS score >= 8, got ${cleanScore}`);

        const legoSqliteTrajectory = {
            messages: [
                { role: 'system', content: 'Engineer' },
                { role: 'user', content: 'Open SQLite DB' },
                {
                    role: 'assistant',
                    content: `#### LEGO-01: Adopt @krusch/toolkit/sqlite (INV-LEGO-12) in src/db.js:1:\n\`\`\`js\nimport { openSqliteDb } from '@krusch/toolkit/sqlite';\nexport const db = openSqliteDb(':memory:');\n\`\`\`\n${'x'.repeat(300)}`
                }
            ]
        };
        const rawSqliteTrajectory = {
            messages: [
                { role: 'system', content: 'Engineer' },
                { role: 'user', content: 'Open SQLite DB' },
                {
                    role: 'assistant',
                    content: `Setup DB in src/db.js:1:\n\`\`\`js\nimport { DatabaseSync } from 'node:sqlite';\nexport const db = new DatabaseSync(':memory:');\n\`\`\`\n${'x'.repeat(300)}`
                }
            ]
        };
        assert.ok(scoreElitistTrajectory(legoSqliteTrajectory) > scoreElitistTrajectory(rawSqliteTrajectory));
        assert.equal(
            scoreElitistTrajectory({
                source: 'referee_clean_pass_calibration',
                messages: [
                    { role: 'system', content: 's' },
                    { role: 'user', content: 'u' },
                    { role: 'assistant', content: `<think>hidden</think> ${'a'.repeat(200)}` }
                ]
            }),
            0
        );
    });

    test('exportElitist120bDataset oversamples calibration split 2x alongside Argon remediations', () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-elitist-'));
        try {
            const sftPath = path.join(tmpDir, 'sft.jsonl');
            const argonPath = path.join(tmpDir, 'argon.jsonl');
            const calPath = path.join(tmpDir, 'lego_and_referee_calibration_sft.jsonl');
            const outPath = path.join(tmpDir, 'elitist.jsonl');

            fs.writeFileSync(sftPath, '', 'utf8');
            fs.writeFileSync(
                argonPath,
                `${JSON.stringify({
                    id: 'argon-1',
                    messages: [
                        { role: 'system', content: 's' },
                        { role: 'user', content: 'u' },
                        { role: 'assistant', content: `[BUG-01] VERIFIED in src/a.js:10\n\`\`\`js\nfix();\n\`\`\`\n${'a'.repeat(400)}` }
                    ]
                })}\n`,
                'utf8'
            );
            fs.writeFileSync(
                calPath,
                `${JSON.stringify({
                    id: 'cal-1',
                    source: 'referee_clean_pass_calibration',
                    teacher_model: 'gemini-4-argon',
                    messages: [
                        { role: 'system', content: 's' },
                        { role: 'user', content: 'u' },
                        {
                            role: 'assistant',
                            content: '```json\n{"verdict":"CLEAN_PASS","checked_files":["lib/http.js"],"findings":[]}\n```' + ' '.repeat(80)
                        }
                    ]
                })}\n`,
                'utf8'
            );

            const count = exportElitist120bDataset(sftPath, argonPath, outPath, {
                legoCalibrationPath: calPath
            });
            assert.equal(count, 4);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    test('Jean SRE autonomous lego_scout target, checklist, briefs, backlog writer & LEGO-* card parser work deterministically', async () => {
        assert.ok(
            INFRA_TARGETS.some((t) => t.node === 'kruschdev' && t.focus === 'lego_scout'),
            'Expected INFRA_TARGETS to include kruschdev / lego_scout'
        );

        const checklist = getInfraChecklist('kruschdev', 'lego_scout');
        assert.match(checklist, /LEGO-ADOPT/);
        assert.match(checklist, /LEGO-CANDIDATE/);
        assert.match(checklist, /BELOW_THRESHOLD_DO_NOT_EXTRACT/);
        assert.match(checklist, /\/jean-loop-gate/);

        const homelabRoot = path.resolve(projectRoot, '..', '..');
        const scoutBrief = buildLegoScoutContextBrief(homelabRoot);
        assert.match(scoutBrief, /Cross-Repository Lego Scout Ground Truth/);
        assert.match(scoutBrief, /CANDIDATE_MEETS_RULE_OF_3/);
        assert.match(scoutBrief, /BELOW_THRESHOLD_DO_NOT_EXTRACT/);

        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-lego-brief-'));
        try {
            const srcDir = path.join(tmpDir, 'src');
            fs.mkdirSync(srcDir, { recursive: true });
            fs.writeFileSync(
                path.join(srcDir, 'db.js'),
                'import { DatabaseSync } from "node:sqlite";\nexport const db = new DatabaseSync(":memory:");\n',
                'utf8'
            );
            const projBrief = buildProjectLegoBoilerplateBrief(tmpDir);
            assert.match(projBrief, /raw-node-sqlite/);
            assert.match(projBrief, /@krusch\/toolkit\/sqlite/);

            const backlogFile = path.join(tmpDir, 'cleanup.md');
            assert.equal(await appendLegoScoutBacklogEntry(backlogFile, { node: 'kruschdev' }, 'NO_FINDINGS'), false);
            assert.equal(
                await appendLegoScoutBacklogEntry(
                    backlogFile,
                    { node: 'kruschdev' },
                    '🔴 **Immediate (`LEGO-ADOPT`)**\n- Migrate `sample-proj` to `@krusch/toolkit/sqlite`.'
                ),
                true
            );
            assert.match(fs.readFileSync(backlogFile, 'utf8'), /Lego Scout/);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }

        const parsedCards = parseMasterBacklogCards(
            [
                '### MASTER BACKLOG: sample-proj',
                '#### LEGO-01: Adopt @krusch/toolkit/sqlite',
                '- **Severity:** P1',
                '- **Effort:** S',
                '- **Evidence:** `src/db.js:2`',
                '- **Fix Instructions:** Replace `new DatabaseSync` with `openSqliteDb`.',
                '### 1. SUMMARY TABLE',
                '| ID | Title | Severity | Effort | Confidence | Source Audit |',
                '| --- | --- | --- | --- | --- | --- |',
                '| **LEGO-01** | Adopt @krusch/toolkit/sqlite | P1 | S | High | 1.5 |'
            ].join('\n')
        );
        assert.equal(parsedCards.length, 1);
        assert.equal(parsedCards[0].id, 'LEGO-01');
        assert.equal(parsedCards[0].severity, 'P1');
        assert.equal(parsedCards[0].evidence, 'src/db.js:2');
    });

    test('Committed calibration & public release splits pass Zero-Leak verification', () => {
        if (fs.existsSync(LEGO_CALIBRATION_SFT_PATH)) {
            assert.equal(verifyZeroLeaks(LEGO_CALIBRATION_SFT_PATH).passed, true);
        }
        if (fs.existsSync(REFEREE_DPO_PATH)) {
            assert.equal(verifyZeroLeaks(REFEREE_DPO_PATH).passed, true);
        }
        const pubDir = path.join(projectRoot, 'data', 'public');
        for (const name of [
            'lego_and_referee_calibration_sft.jsonl',
            'referee_calibration_dpo.jsonl',
            'agentic_tool_calling_sft.jsonl',
            'agentic_tool_calling_dpo.jsonl'
        ]) {
            const p = path.join(pubDir, name);
            if (fs.existsSync(p)) {
                assert.equal(verifyZeroLeaks(p).passed, true);
            }
        }
    });

    test('Wave-4 harvestAgenticToolCallingDatasets generates clean-room multi-turn tool-calling SFT & DPO splits and role-neutral Modelfiles', async () => {
        const { harvestAgenticToolCallingDatasets } = await import(
            '../scripts/harvest_agentic_tool_calling.js'
        );
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-wave4-ag-'));
        try {
            const sftOutPath = path.join(tmpDir, 'agentic_sft.jsonl');
            const dpoOutPath = path.join(tmpDir, 'agentic_dpo.jsonl');
            const res = harvestAgenticToolCallingDatasets({
                sftOutPath,
                dpoOutPath,
                nowIso: '2026-10-31T18:00:00.000Z'
            });

            assert.equal(res.ok, true);
            assert.equal(res.data.counts.sftTotal, 64);
            assert.equal(res.data.counts.sreMcpChains, 32);
            assert.equal(res.data.counts.workspaceToolChains, 32);
            assert.equal(res.data.counts.dpoTotal, 18);
            assert.equal(res.data.counts.dpoActionVsProse, 8);
            assert.equal(res.data.counts.dpoLoopTermination, 6);
            assert.equal(res.data.counts.dpoNonRefusal, 4);

            const rawSft = fs.readFileSync(sftOutPath, 'utf8');
            const rawDpo = fs.readFileSync(dpoOutPath, 'utf8');

            // Clean-room check: 0 DolphinBench identifiers, personas, or mock-app schemas
            const cleanRoomForbiddenRe = /\b(?:dolphinbench|search_persona_memory|morgan|riley)\b/i;
            assert.equal(cleanRoomForbiddenRe.test(rawSft), false, 'Unexpected benchmark term in Wave-4 SFT');
            assert.equal(cleanRoomForbiddenRe.test(rawDpo), false, 'Unexpected benchmark term in Wave-4 DPO');

            const sftRows = rawSft.trim().split('\n').map((l) => JSON.parse(l));
            assert.equal(sftRows.length, 64);
            for (const row of sftRows) {
                assert.ok(Array.isArray(row.tools) && row.tools.length >= 2, `Expected tools array on ${row.id}`);
                let sawToolCall = false;
                for (let i = 0; i < row.messages.length; i++) {
                    const m = row.messages[i];
                    if (Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
                        sawToolCall = true;
                        assert.equal(m.tool_calls.length, 1, `Harmony requires 1 tool_call per assistant turn in ${row.id}`);
                        const fn = m.tool_calls[0].function;
                        assert.equal(typeof fn.arguments, 'object');
                        assert.notEqual(fn.arguments, null);
                        assert.equal(Array.isArray(fn.arguments), false);
                        if (m.thinking) {
                            assert.equal(m.content || '', '', `Harmony forbids content+thinking on tool_calls turn in ${row.id}`);
                        }
                        const nextMsg = row.messages[i + 1];
                        assert.ok(nextMsg && nextMsg.role === 'tool', `Expected matching role=tool turn after tool_call in ${row.id}`);
                        assert.equal(nextMsg.tool_call_id, m.tool_calls[0].id);
                    }
                }
                assert.equal(sawToolCall, true, `Expected at least 1 tool_call in ${row.id}`);
            }

            const dpoRows = rawDpo.trim().split('\n').map((l) => JSON.parse(l));
            assert.equal(dpoRows.length, 18);
            for (const pair of dpoRows) {
                assert.ok(Array.isArray(pair.tools) && pair.tools.length >= 1, `Expected tools array on ${pair.id}`);
                assert.ok(Array.isArray(pair.prompt) && pair.prompt.length >= 2);
                assert.equal(pair.chosen.length, 1);
                assert.equal(pair.rejected.length, 1);
                assert.equal(pair.chosen[0].role, 'assistant');
                assert.equal(pair.rejected[0].role, 'assistant');
                for (const turn of [...pair.prompt, ...pair.chosen, ...pair.rejected]) {
                    if (Array.isArray(turn.tool_calls) && turn.tool_calls.length > 0) {
                        assert.equal(turn.tool_calls.length, 1);
                        assert.equal(typeof turn.tool_calls[0].function.arguments, 'object');
                    }
                }
            }

            assert.equal(verifyZeroLeaks(sftOutPath).passed, true);
            assert.equal(verifyZeroLeaks(dpoOutPath).passed, true);

            for (const modelSubdir of ['jean-gpt-oss-120b-sre-lora', 'jean-qwen3-30b-sre-lora', 'jean-hermes3-8b-sre-lora']) {
                const mfPath = path.join(projectRoot, 'models', modelSubdir, 'Modelfile');
                if (fs.existsSync(mfPath)) {
                    const mfText = fs.readFileSync(mfPath, 'utf8');
                    assert.equal(/^SYSTEM\b/m.test(mfText), false, `Expected role-neutral Modelfile without SYSTEM in ${modelSubdir}`);
                }
            }
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    test('Hermes-3-8B unified trainer renders zero-skew ChatML + <tool_call>, fixes Ollama else-if-.System bug, and Q4_0 fuser row sizes match ggml', async () => {
        const { execFileSync } = await import('node:child_process');
        const out = execFileSync(
            'python3',
            [
                '-c',
                [
                    'import json, sys, tempfile',
                    'from pathlib import Path',
                    `sys.path.insert(0, ${JSON.stringify(path.join(projectRoot, 'scripts'))})`,
                    'import fuse_lora_into_gguf as fl',
                    'import train_hermes3_8b_qlora_g4 as th',
                    'assert fl.row_byte_size(4096, fl.GGML_TYPE_Q4_0) == (4096 // 32) * 18',
                    'assert fl.row_byte_size(4096, fl.GGML_TYPE_Q4_1) == (4096 // 32) * 20',
                    'assert fl.row_byte_size(4096, fl.GGML_TYPE_Q5_0) == (4096 // 32) * 22',
                    'assert fl.row_byte_size(4096, fl.GGML_TYPE_Q5_K) == (4096 // 256) * 176',
                    'rendered = th.render_hermes3_chat([{"role":"system","content":"MEMORY: Room C"},{"role":"user","content":"Book"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"book","arguments":{"room":"Room C"}}}]},{"role":"tool","tool_call_id":"c1","content":"{\\"ok\\":true}"}], tools=[{"type":"function","function":{"name":"book","parameters":{"type":"object"}}}], add_generation_prompt=True)',
                    'assert "MEMORY: Room C" in rendered and "<tools>" in rendered and "<tool_call>" in rendered and "<tool_response>" in rendered',
                    'assert rendered.endswith("<|im_start|>assistant\\n")',
                    `dpo_recs = th.read_rendered_dpo_records([Path(${JSON.stringify(path.join(projectRoot, 'data/distillation/referee_calibration_dpo.jsonl'))}), Path(${JSON.stringify(path.join(projectRoot, 'data/distillation/agentic_tool_calling_dpo.jsonl'))})])`,
                    'assert len(dpo_recs) == 30',
                    'with tempfile.TemporaryDirectory() as td:',
                    '    mf = th.write_ollama_modelfile(Path(td)).read_text()',
                    '    assert "else if .System" not in mf',
                    '    assert "{{- if .System }}" in mf and "{{- if .Tools }}" in mf',
                    '    assert "{{- else if .ToolCalls }}<tool_call>" in mf and mf.count(".ToolCalls") == 2',
                    '    assert not any(l.startswith("SYSTEM ") for l in mf.splitlines())',
                    'print(json.dumps({"ok": True, "dpo": len(dpo_recs)}))'
                ].join('\n')
            ],
            { encoding: 'utf8' }
        );
        const parsed = JSON.parse(out.trim());
        assert.equal(parsed.ok, true);
        assert.equal(parsed.dpo, 30);
    });
});
