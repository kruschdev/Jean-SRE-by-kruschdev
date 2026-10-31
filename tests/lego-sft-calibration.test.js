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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function extractFencedJson(markdown) {
    const match = /```json\s*([\s\S]*?)```/.exec(String(markdown || ''));
    assert.ok(match, `Expected fenced json block in: ${String(markdown).slice(0, 120)}`);
    return JSON.parse(match[1]);
}

describe('Lego-First & Calibrated Referee SFT + DPO Pipeline', () => {
    test('harvestLegoAndRefereeDatasets generates balanced Zero-Leak SFT and conversational DPO splits', () => {
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
            assert.equal(res.data.counts.sftTotal, 26);
            assert.equal(res.data.counts.cleanPass, 8);
            assert.equal(res.data.counts.remediations, 8);
            assert.equal(res.data.counts.legoFirst, 10);
            assert.equal(res.data.counts.dpoTotal, 8);
            assert.equal(res.data.counts.dpoPrecision, 4);
            assert.equal(res.data.counts.dpoRecall, 4);

            const sftRows = fs
                .readFileSync(sftOutPath, 'utf8')
                .trim()
                .split('\n')
                .map((l) => JSON.parse(l));
            assert.equal(sftRows.length, 26);

            const cleanPassRows = sftRows.filter((r) => r.source === 'referee_clean_pass_calibration');
            assert.equal(cleanPassRows.length, 8);
            for (const row of cleanPassRows) {
                const parsed = extractFencedJson(row.messages[2].content);
                assert.equal(parsed.verdict, 'CLEAN_PASS');
                assert.deepEqual(parsed.findings, []);
                assert.ok(scoreElitistTrajectory(row) >= 12);
            }

            const dpoRows = fs
                .readFileSync(dpoOutPath, 'utf8')
                .trim()
                .split('\n')
                .map((l) => JSON.parse(l));
            assert.equal(dpoRows.length, 8);
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
                } else {
                    assert.equal(chosenJson.verdict, 'DEFECTS_FOUND');
                    assert.equal(rejectedJson.verdict, 'CLEAN_PASS');
                }
            }

            assert.equal(verifyZeroLeaks(sftOutPath).passed, true);
            assert.equal(verifyZeroLeaks(dpoOutPath).passed, true);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    test('scoreElitistTrajectory rewards concise CLEAN_PASS verdicts & Lego-First bricks while penalizing raw boilerplate', () => {
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

        const legoTrajectory = {
            messages: [
                { role: 'system', content: 'Engineer' },
                { role: 'user', content: 'Create PostgreSQL pool' },
                {
                    role: 'assistant',
                    content: `Use @krusch/toolkit/pg (INV-LEGO-01) in src/db.js:1:\n\`\`\`js\nimport { createPgPool } from '@krusch/toolkit/pg';\nexport const pool = createPgPool();\n\`\`\`\n${'x'.repeat(300)}`
                }
            ]
        };
        const rawBoilerplateTrajectory = {
            messages: [
                { role: 'system', content: 'Engineer' },
                { role: 'user', content: 'Create PostgreSQL pool' },
                {
                    role: 'assistant',
                    content: `Setup pool in src/db.js:1:\n\`\`\`js\nimport pg from 'pg';\nexport const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });\n\`\`\`\n${'x'.repeat(300)}`
                }
            ]
        };
        assert.ok(scoreElitistTrajectory(legoTrajectory) > scoreElitistTrajectory(rawBoilerplateTrajectory));
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
