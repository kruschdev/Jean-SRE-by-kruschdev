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
            'referee_calibration_dpo.jsonl'
        ]) {
            const p = path.join(pubDir, name);
            if (fs.existsSync(p)) {
                assert.equal(verifyZeroLeaks(p).passed, true);
            }
        }
    });
});
