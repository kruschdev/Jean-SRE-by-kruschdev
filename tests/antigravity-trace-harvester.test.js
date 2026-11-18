import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  scrubAntigravityTraceText,
  parseAntigravityTranscriptToSftRow,
  harvestAntigravityTraces
} from '../scripts/harvest_antigravity_traces.js';
import { verifyZeroLeaks } from '../scripts/build_public_sft_release.js';

describe('Antigravity Zero-Leak Trajectory Harvester (harvest_antigravity_traces.js)', () => {
  test('scrubAntigravityTraceText removes home paths, private/CGNAT IPs, fleet hostnames, private repo names, and Bearer tokens', () => {
    const dirty = [
      'File Path: file:///home/krusch/.gemini/antigravity/brain/abc/.system_generated/logs/transcript.jsonl',
      'Edited /home/krusch/homelab/projects/krusch-stack/bin/krusch-stack.js on kruschdev (192.168.1.240 / 100.106.193.66)',
      'Private repos: acme-legacy-billing, oscar-tree-service, legacy-billing-monolith, hermes-parallax',
      'Authorization: Bearer super_secret_bearer_token_123456'
    ].join('\n');

    const clean = scrubAntigravityTraceText(dirty, 'krusch-stack');
    assert.equal(clean.includes('/home/'), false);
    assert.equal(clean.includes('192.168.'), false);
    assert.equal(clean.includes('100.106.'), false);
    assert.equal(clean.includes('kruschdev'), false);
    assert.equal(clean.includes('acme-legacy-billing'), false);
    assert.equal(clean.includes('oscar-tree-service'), false);
    assert.equal(clean.includes('legacy-billing-monolith'), false);
    assert.equal(clean.includes('hermes-parallax'), false);
    assert.equal(clean.includes('super_secret_bearer_token_123456'), false);
    assert.ok(clean.includes('Bearer REDACTED_TOKEN'));
  });

  test('harvestAntigravityTraces converts parallel subagent tool steps into strict 1-call-per-turn Zero-Leak SFT rows', () => {
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-agy-harvest-'));
    try {
      const convId = 'conv-clean-pass-01';
      const logsDir = path.join(tmpRoot, 'brain', convId, '.system_generated', 'logs');
      fs.mkdirSync(logsDir, { recursive: true });
      const transcriptPath = path.join(logsDir, 'transcript_full.jsonl');

      const syntheticSteps = [
        {
          step_index: 0,
          source: 'SYSTEM',
          type: 'SYSTEM_MESSAGE',
          status: 'DONE',
          created_at: '2026-11-18T19:00:00Z',
          content:
            '<SYSTEM_MESSAGE>content=Perform a security re-audit of /home/krusch/homelab/projects/krusch-stack/bin/krusch-stack.js and return CLEAN PASS if zero P0/P1 defects remain.</SYSTEM_MESSAGE>'
        },
        {
          step_index: 1,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          created_at: '2026-11-18T19:00:02Z',
          thinking: 'Inspect bin/krusch-stack.js and run unit tests on kruschdev (192.168.1.240).',
          tool_calls: [
            { name: 'read_file', args: { Path: '/home/krusch/homelab/projects/krusch-stack/bin/krusch-stack.js', StartLine: 1, EndLine: 50 } },
            { name: 'run_command', args: { CommandLine: 'npm test' } }
          ]
        },
        {
          step_index: 2,
          source: 'MODEL',
          type: 'GENERIC',
          status: 'DONE',
          created_at: '2026-11-18T19:00:03Z',
          content: '1: export function startHttpGateway() {}'
        },
        {
          step_index: 3,
          source: 'MODEL',
          type: 'GENERIC',
          status: 'DONE',
          created_at: '2026-11-18T19:00:05Z',
          content: 'Command exited with code 0.\npass 87\nfail 0'
        },
        {
          step_index: 4,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          created_at: '2026-11-18T19:00:08Z',
          tool_calls: [
            {
              name: 'send_message',
              args: {
                Recipient: 'parent',
                Message:
                  '## Audit Verdict: CLEAN PASS (0 P0, 0 P1, 0 P2 defects)\n\nVerified `bin/krusch-stack.js:160-245` Host allowlist, CSRF guards, constant-time SHA-256 Bearer check, and workspaceRoot symlink containment across all 87/87 passing unit tests.'
              }
            }
          ]
        }
      ];

      fs.writeFileSync(
        transcriptPath,
        syntheticSteps.map((s) => JSON.stringify(s)).join('\n') + '\n',
        'utf8'
      );

      const parsedRow = parseAntigravityTranscriptToSftRow(transcriptPath, convId);
      assert.ok(parsedRow);
      assert.equal(parsedRow.project, 'krusch-stack');
      assert.equal(parsedRow.phase, 'CALIBRATED_CLEAN_PASS_AUDIT');

      // Verify strict 1 tool_call per assistant turn
      const assistantToolTurns = parsedRow.messages.filter((m) => Array.isArray(m.tool_calls));
      assert.equal(assistantToolTurns.length, 2);
      for (const turn of assistantToolTurns) {
        assert.equal(turn.tool_calls.length, 1);
      }

      const outJsonl = path.join(tmpRoot, 'out', 'antigravity_traces_sft.jsonl');
      const summary = harvestAntigravityTraces({
        brainDir: path.join(tmpRoot, 'brain'),
        sftOutPath: outJsonl
      });
      assert.equal(summary.totalRows, 1);

      const leakCheck = verifyZeroLeaks(outJsonl);
      assert.equal(leakCheck.passed, true, `Expected 0 leaks, got: ${JSON.stringify(leakCheck)}`);
    } finally {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });
});
