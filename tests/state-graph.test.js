import assert from 'node:assert/strict';
import test from 'node:test';
import {
    computeJeanTurnDifficulty,
    extractStructuredJeanConclusion,
    commitJeanStateTransition,
    hydrateJeanPersistentState,
    getJeanGraphStatus,
} from '../src/state-graph.js';

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
