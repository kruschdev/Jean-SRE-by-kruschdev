import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
    inferJeanTargetAndPhase,
    computeJeanTurnDifficulty,
    extractStructuredJeanConclusion,
    commitJeanStateTransition,
    hydrateJeanPersistentState,
    getJeanGraphStatus,
} from '../src/state-graph.js';
import {
    evaluateChairResidencyPlan,
    analyzeJeanSweepLogHealth,
    runJeanModularRsiPass,
    runDailyJeanOptimization,
    buildSafeSourceSyncArgs,
} from '../scripts/optimize_jean_daily.js';

test('computeJeanTurnDifficulty maps 11-phase sweep & interactive prompts across Easy/Medium/Hard 3-Chair bands', () => {
    const easy = computeJeanTurnDifficulty('PROJECT: **krusch-cascade-router**\nCURRENT OBJECTIVE: Phase 0_ORIENTATION');
    assert.equal(easy.difficultyBand, 'easy');
    assert.ok(easy.complexityScore <= 0.35);
    assert.equal(easy.targetProject, 'krusch-cascade-router');
    assert.equal(easy.phase, '0_ORIENTATION');

    const medium = computeJeanTurnDifficulty('PROJECT: **krusch-stack**\nCURRENT OBJECTIVE: Phase 1.3_LEGO_COMPLIANCE');
    assert.equal(medium.difficultyBand, 'medium');
    assert.ok(medium.complexityScore > 0.35 && medium.complexityScore <= 0.68);
    assert.equal(medium.targetProject, 'krusch-stack');
    assert.equal(medium.phase, '1.3_LEGO_COMPLIANCE');

    const hard = computeJeanTurnDifficulty('PROJECT: **jean-sre**\nCURRENT OBJECTIVE: Phase 2_MASTER_BACKLOG');
    assert.equal(hard.difficultyBand, 'hard');
    assert.ok(hard.complexityScore > 0.68);
    assert.equal(hard.targetProject, 'jean-sre');
    assert.equal(hard.phase, '2_MASTER_BACKLOG');

    const emergency = computeJeanTurnDifficulty('EMERGENCY DIRECTIVE: GPU thermal escalation');
    assert.equal(emergency.difficultyBand, 'hard');
    assert.ok(emergency.complexityScore >= 0.85);
});

test('extractStructuredJeanConclusion parses JEAN_INFLIGHT, LESSON, HYPOTHESIS VERDICT, and CLEAN_PASS', () => {
    const reply = [
        '### LESSON',
        'Use THREE_CHAIR_BANDIT_TIERS for zero-cost local routing across all 5 GPUs.',
        '### HYPOTHESIS VERDICT',
        'CONFIRMED: OpenCypher state transitions persist across turns.',
        'CLEAN_PASS verified.',
        '### JEAN_INFLIGHT',
        '- **Current Focus:** Auditing state-graph persistence',
        '- **Next Cycle Goal:** Rotate to deadbolt-public-board',
        '- **Pending Issues:** None',
    ].join('\n');

    const parsed = extractStructuredJeanConclusion(reply, {
        targetProject: 'jean-sre',
        phase: '2_MASTER_BACKLOG',
    });
    assert.equal(parsed.verdict, 'CLEAN_PASS');
    assert.equal(parsed.focus, 'Auditing state-graph persistence');
    assert.equal(parsed.nextGoal, 'Rotate to deadbolt-public-board');
    assert.equal(parsed.pendingIssues, 'None');
    assert.ok(parsed.lesson.includes('THREE_CHAIR_BANDIT_TIERS'));
    assert.ok(parsed.hypothesisVerdict.includes('CONFIRMED'));
});

test('commitJeanStateTransition and hydrateJeanPersistentState persist and re-hydrate state across chairs in test mode', async () => {
    process.env.NODE_ENV = 'test';

    const c1 = await commitJeanStateTransition({
        targetProject: 'demo-repo',
        phase: '0_ORIENTATION',
        routedTierId: 'L1_THIRD_CHAIR_FAST',
        routedModel: 'gemma4:e2b',
        complexityScore: 0.24,
        difficultyBand: 'easy',
        reply: '### JEAN_INFLIGHT\n- **Current Focus:** Initial repo scan\n- **Next Cycle Goal:** Run Phase 1.1\n- **Pending Issues:** None',
    });
    assert.equal(c1.ok, true);
    assert.equal(c1.data.routedModel, 'gemma4:e2b');
    assert.equal(c1.data.difficultyBand, 'easy');

    const c2 = await commitJeanStateTransition({
        targetProject: 'demo-repo',
        phase: '2_MASTER_BACKLOG',
        routedTierId: 'L2_FIRST_CHAIR_INKLING',
        routedModel: 'inkling-2-mini:ud-iq4_xs-dspark',
        complexityScore: 0.86,
        difficultyBand: 'hard',
        reply: 'CLEAN_PASS\n### JEAN_INFLIGHT\n- **Current Focus:** Final synthesis complete\n- **Next Cycle Goal:** Rotate repo\n- **Pending Issues:** None',
    });
    assert.equal(c2.ok, true);
    assert.equal(c2.data.verdict, 'CLEAN_PASS');

    const hydrated = await hydrateJeanPersistentState({ targetProject: 'demo-repo' });
    assert.equal(hydrated.ok, true);
    assert.equal(hydrated.latestState.phase, '2_MASTER_BACKLOG');
    assert.equal(hydrated.latestState.routedModel, 'inkling-2-mini:ud-iq4_xs-dspark');
    assert.ok(hydrated.promptBlock.includes('JEAN PERSISTENT GRAPH STATE'));
    assert.ok(hydrated.promptBlock.includes('inkling-2-mini:ud-iq4_xs-dspark'));

    const status = await getJeanGraphStatus({ targetProject: 'demo-repo' });
    assert.equal(status.ok, true);
    assert.ok(status.data.recentTransitions.length >= 2);
});

test('inferJeanTargetAndPhase accurately resolves Implementer, Infra, Companion, and Session Open headers without grep pollution', () => {
    const impl = inferJeanTargetAndPhase(
        '## JEAN SRE: IMPLEMENTER PHASE — @homelab/.agent [1.7_VERIFICATION_AUDIT (8/11)]\nExecuted grep output mentioning projects/krusch-coding-harness/bin/krusch.js'
    );
    assert.equal(impl.targetProject, '@homelab/.agent');
    assert.equal(impl.phase, '1.7_VERIFICATION_AUDIT');

    const implDiff = computeJeanTurnDifficulty(
        '## JEAN SRE: IMPLEMENTER PHASE — @homelab/.agent [1.7_VERIFICATION_AUDIT (8/11)]\nExecuted grep output mentioning projects/krusch-coding-harness/bin/krusch.js'
    );
    assert.equal(implDiff.targetProject, '@homelab/.agent');
    assert.equal(implDiff.phase, '1.7_VERIFICATION_AUDIT');
    assert.equal(implDiff.difficultyBand, 'hard');

    const finalBacklog = inferJeanTargetAndPhase(
        '## JEAN SRE: IMPLEMENTER PHASE (Final Master Backlog Synthesis — NO tool calls)\nThe THINKER analyzed the accumulated phase findings for **@homelab/lib** and produced the MASTER BACKLOG.\nReferences projects/pocket-lawyer'
    );
    assert.equal(finalBacklog.targetProject, '@homelab/lib');
    assert.equal(finalBacklog.phase, '2_MASTER_BACKLOG');

    const infraThinker = computeJeanTurnDifficulty('## JEAN: INFRASTRUCTURE AUDIT — KRUSCHDEV / LEGO_SCOUT');
    assert.equal(infraThinker.targetProject, 'kruschdev');
    assert.equal(infraThinker.phase, 'INFRA_LEGO_SCOUT');
    assert.equal(infraThinker.difficultyBand, 'hard');

    const infraSynth = computeJeanTurnDifficulty('## JEAN SRE: INFRA REPORT SYNTHESIS (kruschserv/docker)');
    assert.equal(infraSynth.targetProject, 'kruschserv');
    assert.equal(infraSynth.phase, 'INFRA_DOCKER');
    assert.equal(infraSynth.difficultyBand, 'medium');

    const comp = computeJeanTurnDifficulty('## JEAN SRE: COMPANION AUDITOR SYNTHESIS (deadbolt-public-board)');
    assert.equal(comp.targetProject, 'deadbolt-public-board');
    assert.equal(comp.phase, 'COMPANION_AUDIT');
    assert.equal(comp.difficultyBand, 'medium');

    const openSweep = inferJeanTargetAndPhase(
        '## JEAN SRE: SESSION OPEN\nContainer list: projects/pocket-lawyer/docker-compose.yml'
    );
    assert.equal(openSweep.targetProject, 'fleet');
    assert.equal(openSweep.phase, 'SESSION_OPEN');

    const isolated = inferJeanTargetAndPhase('[Project Isolation Context: g-krusch] Check OAuth refresh handler');
    assert.equal(isolated.targetProject, 'g-krusch');
});

test('Daily Jean SRE Optimizer evaluates 3rd-Chair VRAM residency, ModularRSI SIM attribution, and signed receipts', async () => {
    // 1. Residency planner evicts legacy 7B VRAM thief and upgrades 4K -> 8K embed context
    const driftedPlan = evaluateChairResidencyPlan([
        { name: 'qwen2.5-coder:7b', size: 5_500_000_000, size_vram: 5_500_000_000, context_length: 16384 },
        { name: 'snowflake-arctic-embed2:latest', size: 664_000_000, size_vram: 664_000_000, context_length: 4096 },
    ]);
    assert.deepEqual(driftedPlan.evictModels, ['qwen2.5-coder:7b']);
    assert.equal(driftedPlan.pinThirdChair, true);
    assert.equal(driftedPlan.pinEmbed, true);

    const optimalPlan = evaluateChairResidencyPlan([
        { name: 'gemma4:e2b', size: 4_600_000_000, size_vram: 4_600_000_000, context_length: 16384 },
        { name: 'snowflake-arctic-embed2:latest', size: 1_200_000_000, size_vram: 1_200_000_000, context_length: 8192 },
    ]);
    assert.deepEqual(optimalPlan.evictModels, []);
    assert.equal(optimalPlan.pinThirdChair, false);
    assert.equal(optimalPlan.pinEmbed, false);

    // 2. Sweep log parser + ModularRSI SIM attribution
    const logHealth = analyzeJeanSweepLogHealth(
        [
            '[Sweep] Cycle 1 complete. Next phase: 1.2_COMPLIANCE_MATRIX',
            '[Drift] BLOCKED: grep -rn "foo" ...',
            '[Compliance Gate] 🛑 Tool call run_shell_command blocked: schema violation',
            '[Persona] Pruning input message from 56000 chars to 32000 chars to stay within 16K context budget',
        ].join('\n')
    );
    assert.equal(logHealth.cyclesCompleted, 1);
    assert.equal(logHealth.driftBlockedCount, 1);
    assert.equal(logHealth.complianceBlockedCount, 1);
    assert.equal(logHealth.syntheticTrajectories.length, 2);

    const rsi = runJeanModularRsiPass({
        trajectories: [
            { taskId: 'ok-1', success: true },
            ...logHealth.syntheticTrajectories,
        ],
        round: 1,
    });
    assert.equal(rsi.failedTrajectories, 2);
    assert.equal(rsi.appliedPatches.length, 2);
    assert.equal(rsi.simSummary.problemCount, 2);

    // 3. Full mocked daily pipeline emits signed envelope & SHA-256 receipt
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-daily-opt-'));
    try {
        const receiptPath = path.join(tmpDir, 'receipt.json');
        const historyPath = path.join(tmpDir, 'history.jsonl');
        const simPath = path.join(tmpDir, 'sim.json');
        const rsiSpecsPath = path.join(tmpDir, 'specs.json');

        const envRes = await runDailyJeanOptimization({
            dryRun: false,
            enableSkillOpt: false,
            skipRemoteSync: true,
            receiptPath,
            historyPath,
            simPath,
            rsiSpecsPath,
            mockStages: {
                residency: async () => ({ ok: true, harnessReady: true }),
                quantcheck: async () => ({ ok: true, endpointCount: 4, totalLiveProbed: 4 }),
                contextCompaction: async () => ({ ok: true, sizeBeforeMb: 231.16, sizeAfterMb: 39.4, reclaimedMb: 191.76 }),
                persistentMind: async () => ({ ok: true, selfGraphConnected: true }),
                trajectories: async () => ({
                    dbTrajectories: [{ taskId: 'job-1', success: true }],
                    sweepLogHealth: logHealth,
                }),
            },
        });

        assert.equal(envRes.ok, true);
        assert.equal(envRes.tool, 'jean.daily-optimize');
        assert.equal(envRes.data.receiptSha256.length, 64);
        assert.ok(fs.existsSync(receiptPath));
        assert.ok(fs.existsSync(historyPath));
        assert.ok(fs.existsSync(simPath));
    } finally {
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }

    // 4. Safe source-sync builder enforces mandatory DB/secret exclusions and rejects unsafe paths
    const rsyncArgs = buildSafeSourceSyncArgs({
        srcDir: 'projects/krusch-companion',
        destSpec: 'remote-node:/opt/workspace',
        dryRun: true,
    });
    assert.equal(rsyncArgs.includes('--delete'), false);
    assert.ok(rsyncArgs.includes('--exclude=postgres/'));
    assert.ok(rsyncArgs.includes('--exclude=redis/'));
    assert.ok(rsyncArgs.includes('--exclude=data/'));
    assert.ok(rsyncArgs.includes('--exclude=.agent/'));
    assert.ok(rsyncArgs.includes('--exclude=.env*'));
    assert.throws(() => buildSafeSourceSyncArgs({ srcDir: '../etc', destSpec: 'x:/y' }), /Unsafe/);
    assert.throws(() => buildSafeSourceSyncArgs({ srcDir: 'benchmarks/locomo', destSpec: 'x:/y' }), /Unsafe/);
});
