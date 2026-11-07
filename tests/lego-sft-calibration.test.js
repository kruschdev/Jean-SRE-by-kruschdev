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
            'agentic_tool_calling_dpo.jsonl',
            'arete_values_sft.jsonl',
            'arete_values_dpo.jsonl'
        ]) {
            const p = path.join(pubDir, name);
            if (fs.existsSync(p)) {
                assert.equal(verifyZeroLeaks(p).passed, true);
            }
        }
    });

    test('Wave-6 harvestAgenticToolCallingDatasets generates clean-room multi-turn tool-calling SFT & DPO splits and role-neutral Modelfiles', async () => {
        const { harvestAgenticToolCallingDatasets } = await import(
            '../scripts/harvest_agentic_tool_calling.js'
        );
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-wave6-ag-'));
        try {
            const sftOutPath = path.join(tmpDir, 'agentic_sft.jsonl');
            const dpoOutPath = path.join(tmpDir, 'agentic_dpo.jsonl');
            const res = harvestAgenticToolCallingDatasets({
                sftOutPath,
                dpoOutPath,
                nowIso: '2026-10-31T18:00:00.000Z'
            });

            assert.equal(res.ok, true);
            assert.equal(res.data.counts.sftTotal, 76);
            assert.equal(res.data.counts.sreMcpChains, 32);
            assert.equal(res.data.counts.workspaceToolChains, 44);
            assert.equal(res.data.counts.dpoTotal, 34);
            assert.equal(res.data.counts.dpoActionVsProse, 8);
            assert.equal(res.data.counts.dpoLoopTermination, 6);
            assert.equal(res.data.counts.dpoNonRefusal, 4);
            assert.equal(res.data.counts.dpoVerbatimPreservation, 6);
            assert.equal(res.data.counts.dpoPartialBriefingExhaustive, 10);

            const rawSft = fs.readFileSync(sftOutPath, 'utf8');
            const rawDpo = fs.readFileSync(dpoOutPath, 'utf8');

            // Clean-room check: 0 DolphinBench identifiers, personas, or mock-app schemas
            const cleanRoomForbiddenRe = /\b(?:dolphinbench|search_persona_memory|morgan|riley|load_bearing_facts|expected_tool_calls|assertions)\b|facts\.yaml/i;
            assert.equal(cleanRoomForbiddenRe.test(rawSft), false, 'Unexpected benchmark term in Wave-6 SFT');
            assert.equal(cleanRoomForbiddenRe.test(rawDpo), false, 'Unexpected benchmark term in Wave-6 DPO');

            const sftRows = rawSft.trim().split('\n').map((l) => JSON.parse(l));
            assert.equal(sftRows.length, 76);
            let wave6SftCount = 0;
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

                if (row.id.includes('-wave6-')) {
                    wave6SftCount++;
                    assert.equal(/\balex\b/i.test(JSON.stringify(row)), false);
                    assert.ok(
                        row.messages[0].content.includes('[SOVEREIGN MEMORY BRIEFING'),
                        `Expected partial briefing header on ${row.id}`
                    );
                    const assistantToolTurns = row.messages.filter(
                        (m) => m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length === 1
                    );
                    assert.equal(assistantToolTurns.length, 2, `Expected gap-search -> action chain on ${row.id}`);
                    assert.equal(assistantToolTurns[0].tool_calls[0].function.name, 'query_memory_notes');
                    const actionArgsJson = JSON.stringify(assistantToolTurns[1].tool_calls[0].function.arguments);
                    assert.ok(
                        actionArgsJson.length >= 450,
                        `Expected exhaustive >=450 char tool argument on ${row.id}, got ${actionArgsJson.length}`
                    );
                }
            }
            assert.equal(wave6SftCount, 12);

            const dpoRows = rawDpo.trim().split('\n').map((l) => JSON.parse(l));
            assert.equal(dpoRows.length, 34);
            let wave6GapPairs = 0;
            let wave6ExhaustivePairs = 0;
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
                if (pair.id.startsWith('dpo-tool-wave6-gap-')) {
                    wave6GapPairs++;
                    assert.equal(/\balex\b/i.test(JSON.stringify(pair)), false);
                    assert.equal(pair.chosen[0].tool_calls[0].function.name, 'query_memory_notes');
                    assert.notEqual(pair.rejected[0].tool_calls[0].function.name, 'query_memory_notes');
                }
                if (pair.id.startsWith('dpo-tool-wave6-exhaustive-')) {
                    wave6ExhaustivePairs++;
                    assert.equal(/\balex\b/i.test(JSON.stringify(pair)), false);
                    const chosenLen = JSON.stringify(pair.chosen[0].tool_calls[0].function.arguments).length;
                    const rejectedLen = JSON.stringify(pair.rejected[0].tool_calls[0].function.arguments).length;
                    assert.ok(chosenLen >= 480, `Expected chosen >=480 chars on ${pair.id}, got ${chosenLen}`);
                    assert.ok(chosenLen > rejectedLen * 2.5, `Expected chosen >> rejected detail on ${pair.id}`);
                }
            }
            assert.equal(wave6GapPairs, 5);
            assert.equal(wave6ExhaustivePairs, 5);

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
                    'assert len(dpo_recs) == 46',
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
        assert.equal(parsed.dpo, 46);
    });

    test('Gemma-4-31B Wave-6 trainer anchors trl>=1.0 DPO ref adapter in aligned BF16 and cleans ref artifacts before save', async () => {
        const { execFileSync } = await import('node:child_process');
        const pyScript = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(path.join(projectRoot, 'scripts'))})
import train_gemma4_31b_qlora_g4 as tg4

class FakeTensor:
    def __init__(self, vals, dtype="torch.bfloat16", device="cuda:0", requires_grad=True):
        self.vals = [float(v) for v in vals]
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
    @property
    def data(self):
        return self
    @data.setter
    def data(self, other):
        self.vals = list(other.vals)
        self.dtype = other.dtype
        self.device = other.device
    def detach(self):
        return FakeTensor(self.vals, dtype=self.dtype, device=self.device, requires_grad=False)
    def to(self, device=None, dtype=None):
        return FakeTensor(self.vals, dtype=dtype or self.dtype, device=device or self.device, requires_grad=self.requires_grad)
    def clone(self):
        return FakeTensor(self.vals, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
    def copy_(self, other):
        self.vals = list(other.vals)
        return self
    def requires_grad_(self, flag):
        self.requires_grad = bool(flag)
        return self
    def float(self):
        return FakeTensor(self.vals, dtype="torch.float32", device=self.device, requires_grad=False)
    def __sub__(self, other):
        return FakeTensor([a - b for a, b in zip(self.vals, other.vals)], dtype=self.dtype, device=self.device, requires_grad=False)
    def __pow__(self, p):
        return FakeTensor([a ** p for a in self.vals], dtype=self.dtype, device=self.device, requires_grad=False)
    def abs(self):
        return FakeTensor([abs(a) for a in self.vals], dtype=self.dtype, device=self.device, requires_grad=False)
    def max(self):
        return FakeScalar(max(self.vals) if self.vals else 0.0)
    def sum(self):
        return FakeScalar(sum(self.vals))

class FakeScalar:
    def __init__(self, v):
        self.v = float(v)
    def item(self):
        return self.v

class FakePeftModel:
    def __init__(self, with_ref=True):
        self.peft_config = {"default": {"r": 16}}
        self.active_adapter = "default"
        self._params = {
            "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": FakeTensor([1.5, -2.0], dtype="torch.bfloat16", device="cuda:0", requires_grad=True),
            "base_model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight": FakeTensor([0.25, 0.75], dtype="torch.bfloat16", device="cuda:0", requires_grad=True),
        }
        if with_ref:
            self.peft_config["ref"] = {"r": 16}
            # Simulate trl 1.17.1 bug: ref left in float32 with unaligned/zero weights
            self._params["base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.ref.weight"] = FakeTensor([0.0, 0.0], dtype="torch.float32", device="cuda:0", requires_grad=False)
            self._params["base_model.model.language_model.layers.0.self_attn.q_proj.lora_B.ref.weight"] = FakeTensor([0.0, 0.0], dtype="torch.float32", device="cuda:0", requires_grad=False)
    def named_parameters(self):
        return list(self._params.items())
    def add_adapter(self, name, cfg):
        self.peft_config[name] = dict(cfg)
        for k, v in list(self._params.items()):
            if ".default." in k:
                self._params[k.replace(".default.", f".{name}.")] = FakeTensor([0.0 for _ in v.vals], dtype="torch.float32", device=v.device, requires_grad=False)
    def set_adapter(self, name):
        self.active_adapter = name
    def delete_adapter(self, name):
        self.peft_config.pop(name, None)
        for k in [k for k in self._params if f".{name}." in k]:
            del self._params[k]

class FakeDPOTrainer:
    def __init__(self, model, ref_adapter_name=None):
        self.model = model
        self.ref_adapter_name = ref_adapter_name

# Case 1: trl 1.17.1 created 'ref' in float32 with stale weights
m1 = FakePeftModel(with_ref=True)
snap1 = tg4.snapshot_peft_default_adapter(m1)
# Simulate DPOTrainer mutating default before sync check
t1 = FakeDPOTrainer(m1, ref_adapter_name="ref")
meta1 = tg4.verify_and_sync_trl_ref_adapter(t1, snap1)
assert meta1["ref_adapter_anchored"] is True
assert meta1["ref_adapter_name"] == "ref"
assert meta1["synced_tensors"] == 2
assert meta1["ref_dtype"] == "torch.bfloat16"
assert meta1["max_diff_vs_sft"] == 0.0
ref_a = m1._params["base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.ref.weight"]
def_a = m1._params["base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"]
assert ref_a.vals == [1.5, -2.0] and ref_a.dtype == "torch.bfloat16" and ref_a.requires_grad is False
assert def_a.requires_grad is True and m1.active_adapter == "default"

# Case 2: trl trainer with ref_adapter_name=None & missing 'ref' -> materialized & bound
m2 = FakePeftModel(with_ref=False)
snap2 = tg4.snapshot_peft_default_adapter(m2)
t2 = FakeDPOTrainer(m2, ref_adapter_name=None)
meta2 = tg4.verify_and_sync_trl_ref_adapter(t2, snap2)
assert meta2["ref_adapter_anchored"] is True
assert t2.ref_adapter_name == "ref"
assert "ref" in m2.peft_config

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    (td_path / "ref").mkdir()
    (td_path / "ref" / "adapter_model.safetensors").write_text("stale")
    cleaned = tg4.cleanup_peft_ref_adapters(m2, td_path)
    assert "adapter:ref" in cleaned and "dir:ref" in cleaned
    assert "ref" not in m2.peft_config
    assert not (td_path / "ref").exists()

print(json.dumps({"ok": True, "meta1": meta1, "cleaned": cleaned}))
`;
        const out = execFileSync('python3', ['-c', pyScript], { encoding: 'utf8' });
        const parsed = JSON.parse(out.trim());
        assert.equal(parsed.ok, true);
        assert.equal(parsed.meta1.ref_adapter_anchored, true);
        assert.equal(parsed.meta1.ref_adapter_name, 'ref');
        assert.equal(parsed.meta1.ref_dtype, 'torch.bfloat16');
    });

    test('Qwen3-Coder-30B Wave-6 trainer maps pre-tool reasoning, masks strictly ChatML assistant spans, and anchors BF16 DPO ref adapter', async () => {
        const { execFileSync } = await import('node:child_process');
        const pyScript = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(path.join(projectRoot, 'scripts'))})
import train_qwen3_30b_qlora_g4 as tq3

orig_argv = list(sys.argv)
sys.argv = ["train_qwen3_30b_qlora_g4.py"]
try:
    args = tq3.parse_args()
finally:
    sys.argv = orig_argv
assert args.mode == "sft_dpo"
assert args.lora_r == 16 and args.lora_alpha == 16
assert abs(args.lr - 5e-5) < 1e-12
assert abs(args.dpo_lr - 5e-6) < 1e-12
assert abs(args.dpo_beta - 0.2) < 1e-12
assert args.max_samples == 460
assert args.agentic_oversample == 4
assert args.agentic_dpo_oversample == 3

# 1. Pre-tool reasoning maps cleanly to content before <tool_call> without <think> tags
cleaned_tc = tq3._clean_message({
    "role": "assistant",
    "content": "",
    "thinking": "Partial briefing lacks the February closeout note; query memory first.",
    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "query_memory_notes", "arguments": {"query": "february closeout"}}}]
})
assert cleaned_tc["content"] == "Partial briefing lacks the February closeout note; query memory first."
assert "<think>" not in cleaned_tc["content"]
assert cleaned_tc["tool_calls"][0]["function"]["name"] == "query_memory_notes"

cleaned_inline = tq3._clean_message({
    "role": "assistant",
    "content": "<think>Check closeout status</think>",
    "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "get_pr", "arguments": "{\\"pr_id\\": 42}"}}]
})
assert cleaned_inline["content"] == "Check closeout status"
assert cleaned_inline["tool_calls"][0]["function"]["arguments"] == {"pr_id": 42}

cleaned_final = tq3._clean_message({
    "role": "assistant",
    "content": "Final verified status report.",
    "thinking": "Internal scratchpad"
})
assert cleaned_final["content"] == "Final verified status report."

# 2. QwenChatMLAssistantOnlyCollator._compute_assistant_char_spans selects strictly assistant turns
sample_chatml = (
    "<|im_start|>system\\nYou are an assistant.\\n\\n# Tools\\n<tools>\\n<function><name>query_memory_notes</name></function>\\n</tools><|im_end|>\\n"
    "<|im_start|>user\\nSummarize the project status.<|im_end|>\\n"
    "<|im_start|>assistant\\nNeed February closeout.\\n<tool_call>\\n<function=query_memory_notes>\\n<parameter=query>\\ncloseout\\n</parameter>\\n</function>\\n</tool_call><|im_end|>\\n"
    "<|im_start|>user\\n<tool_response>\\n{\\"status\\": \\"closed_internal_only\\"}\\n</tool_response>\\n<|im_end|>\\n"
    "<|im_start|>assistant\\nProject is closed for internal use only; no customer exposure authorized.<|im_end|>\\n"
)
spans = tq3.QwenChatMLAssistantOnlyCollator._compute_assistant_char_spans(sample_chatml)
assert len(spans) == 2
s0 = sample_chatml[spans[0][0]:spans[0][1]]
s1 = sample_chatml[spans[1][0]:spans[1][1]]
assert s0.startswith("Need February closeout.") and s0.endswith("</tool_call><|im_end|>")
assert s1 == "Project is closed for internal use only; no customer exposure authorized.<|im_end|>"
for sp in (s0, s1):
    assert "<|im_start|>system" not in sp
    assert "<|im_start|>user" not in sp
    assert "<tool_response>" not in sp
    assert "<tools>" not in sp

# 3. BF16 DPO ref adapter anchoring & cleanup
class FakeScalar:
    def __init__(self, v): self.v = float(v)
    def item(self): return self.v

class FakeTensor:
    def __init__(self, vals, dtype="torch.bfloat16", device="cuda:0", requires_grad=True):
        self.vals = [float(v) for v in vals]
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
    @property
    def data(self): return self
    @data.setter
    def data(self, other):
        self.vals = list(other.vals)
        self.dtype = other.dtype
        self.device = other.device
    def detach(self): return FakeTensor(self.vals, dtype=self.dtype, device=self.device, requires_grad=False)
    def to(self, device=None, dtype=None): return FakeTensor(self.vals, dtype=dtype or self.dtype, device=device or self.device, requires_grad=self.requires_grad)
    def clone(self): return FakeTensor(self.vals, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
    def copy_(self, other): self.vals = list(other.vals); return self
    def requires_grad_(self, flag): self.requires_grad = bool(flag); return self
    def float(self): return FakeTensor(self.vals, dtype="torch.float32", device=self.device, requires_grad=False)
    def __sub__(self, other): return FakeTensor([a - b for a, b in zip(self.vals, other.vals)], dtype=self.dtype, device=self.device, requires_grad=False)
    def __pow__(self, p): return FakeTensor([a ** p for a in self.vals], dtype=self.dtype, device=self.device, requires_grad=False)
    def abs(self): return FakeTensor([abs(a) for a in self.vals], dtype=self.dtype, device=self.device, requires_grad=False)
    def max(self): return FakeScalar(max(self.vals) if self.vals else 0.0)
    def sum(self): return FakeScalar(sum(self.vals))

class FakePeftModel:
    def __init__(self):
        self.peft_config = {"default": {"r": 16}, "ref": {"r": 16}}
        self.active_adapter = "default"
        self._params = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": FakeTensor([1.25, -0.5], dtype="torch.bfloat16"),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": FakeTensor([0.5, 0.25], dtype="torch.bfloat16"),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.ref.weight": FakeTensor([0.0, 0.0], dtype="torch.float32", requires_grad=False),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.ref.weight": FakeTensor([0.0, 0.0], dtype="torch.float32", requires_grad=False),
        }
    def named_parameters(self): return list(self._params.items())
    def set_adapter(self, name): self.active_adapter = name
    def delete_adapter(self, name):
        self.peft_config.pop(name, None)
        for k in [k for k in self._params if f".{name}." in k]:
            del self._params[k]

class FakeDPOTrainer:
    def __init__(self, model):
        self.model = model
        self.ref_adapter_name = "ref"

pm = FakePeftModel()
snap = tq3.snapshot_peft_default_adapter(pm)
meta = tq3.verify_and_sync_trl_ref_adapter(FakeDPOTrainer(pm), snap)
assert meta["ref_adapter_anchored"] is True
assert meta["synced_tensors"] == 2
assert meta["ref_dtype"] == "torch.bfloat16"
assert meta["max_diff_vs_sft"] == 0.0

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    (td_path / "ref").mkdir()
    cleaned = tq3.cleanup_peft_ref_adapters(pm, td_path)
    assert "adapter:ref" in cleaned and "dir:ref" in cleaned
    mf = tq3.write_ollama_modelfile(td_path).read_text()
    assert not any(l.startswith("SYSTEM ") for l in mf.splitlines())
    assert "PARAMETER num_ctx 16384" in mf

print(json.dumps({"ok": True, "spans": len(spans), "meta": meta}))
`;
        const out = execFileSync('python3', ['-c', pyScript], { encoding: 'utf8' });
        const parsed = JSON.parse(out.trim());
        assert.equal(parsed.ok, true);
        assert.equal(parsed.spans, 2);
        assert.equal(parsed.meta.ref_adapter_anchored, true);
    });
});

