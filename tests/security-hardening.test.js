import { test, describe } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { verifyToolCompliance } from '../src/compliance.js';
import {
    parseExecutableAction,
    classifyIntentRisk,
    buildPendingIssueCodeEvidence,
    routeWorkflow,
    readInflight,
    runProjectVerificationSuite,
    checkColabBurstStatus,
    distillTeacherKnowledgeForStudent,
    loadDistilledTeacherBrief
} from '../src/jean_sre_sweep.js';

describe('CSO Security Hardening — Jean SRE & krusch-infra-mcp Verification', () => {
    test('compliance.js blocks /var/run/docker.sock, mcp_config.json, and .vscode/mcp.json access', () => {
        const dockerSockAttempt = verifyToolCompliance('run_shell_command', {
            command: 'curl --unix-socket /var/run/docker.sock http://localhost/containers/json'
        });
        assert.strictEqual(dockerSockAttempt.allowed, false);
        assert.match(dockerSockAttempt.reason, /docker\.sock/i);

        const mcpConfigAttempt = verifyToolCompliance('read_file', {
            path: '/home/krusch/.gemini/antigravity/mcp_config.json'
        });
        assert.strictEqual(mcpConfigAttempt.allowed, false);

        const vscodeMcpAttempt = verifyToolCompliance('read_file', {
            path: '/home/krusch/homelab/.vscode/mcp.json'
        });
        assert.strictEqual(vscodeMcpAttempt.allowed, false);
    });

    test('compliance.js blocks direct agent writes/patches to .env files while allowing normal code edits', () => {
        const writeEnvAttempt = verifyToolCompliance('write_file', {
            file_path: '/home/krusch/homelab/projects/jean-sre/.env',
            content: 'JEAN_API_KEY=compromised'
        });
        assert.strictEqual(writeEnvAttempt.allowed, false);
        assert.match(writeEnvAttempt.reason, /\.env/i);

        const patchEnvAttempt = verifyToolCompliance('patch_file', {
            path: '/home/krusch/homelab/.env.production',
            diff: '--- a/.env\n+++ b/.env'
        });
        assert.strictEqual(patchEnvAttempt.allowed, false);

        const normalWrite = verifyToolCompliance('write_file', {
            file_path: '/home/krusch/homelab/projects/jean-sre/src/index.js',
            content: 'console.log("ok");'
        });
        assert.strictEqual(normalWrite.allowed, true);
    });

    test('krusch-infra-mcp/server.js enforces DEFAULT_ALLOWED_PROJECT_PATHS and execFileAsync for git commit', () => {
        const infraServerCode = fs.readFileSync(
            new URL('../../krusch-infra-mcp/server.js', import.meta.url),
            'utf8'
        );
        assert.ok(
            infraServerCode.includes('DEFAULT_ALLOWED_PROJECT_PATHS'),
            'server.js must define DEFAULT_ALLOWED_PROJECT_PATHS fallback'
        );
        assert.ok(
            infraServerCode.includes('execFileAsync("git", gitArgs'),
            'server.js must use parameterized execFileAsync("git", gitArgs) for local git commit'
        );
        assert.ok(
            !infraServerCode.includes('execAsync(`git commit -m "${sanitizedMsg}"`'),
            'server.js must not interpolate commit messages into double-quoted shell strings'
        );
    });

    test('harness.js enforces requireJeanAuth on /chat, /tool, and /trigger with TRUSTED_ORIGIN support', () => {
        const harnessCode = fs.readFileSync(
            new URL('../src/harness.js', import.meta.url),
            'utf8'
        );
        assert.ok(harnessCode.includes('function requireJeanAuth('), 'harness.js must define requireJeanAuth');
        assert.ok(harnessCode.includes('process.env.TRUSTED_ORIGIN'), 'requireJeanAuth must support TRUSTED_ORIGIN');
        assert.ok(harnessCode.includes("app.post('/chat', requireJeanAuth,"), '/chat must use requireJeanAuth');
        assert.ok(harnessCode.includes("app.post('/tool', requireJeanAuth,"), '/tool must use requireJeanAuth');
        assert.ok(harnessCode.includes("app.post('/trigger', requireJeanAuth,"), '/trigger must use requireJeanAuth');
    });

    test('parseExecutableAction auto-injects project/repo for 3-Tier MCP tools and supports ranged read_file', () => {
        const ctxAction = parseExecutableAction(
            '**Action**: `krusch_context_retrieve({"query": "krusch-git", "include_state": true})` **Reason**: Hydrate project invariants.',
            '/home/krusch/homelab/projects/krusch-git'
        );
        assert.strictEqual(ctxAction.kind, 'mcp_tool');
        assert.strictEqual(ctxAction.toolName, 'krusch-context-krusch-context-retrieve');
        assert.strictEqual(ctxAction.toolArgs.project, 'krusch-git');
        assert.strictEqual(classifyIntentRisk('', ctxAction).riskClass, 'L2');

        const gitAction = parseExecutableAction(
            '**Action**: `krusch_git_semantic_search({"query": "embedding"})` **Reason**: Search embedding logic.',
            '/home/krusch/homelab/projects/krusch-git'
        );
        assert.strictEqual(gitAction.kind, 'mcp_tool');
        assert.strictEqual(gitAction.toolName, 'krusch-git-krusch-git-semantic-search');
        assert.strictEqual(gitAction.toolArgs.repo, 'krusch-git');

        const rangedRead = parseExecutableAction(
            '**Action**: `read_file server/mcp.js:120-260` **Reason**: Inspect MCP tool handlers.',
            '/home/krusch/homelab/projects/krusch-git'
        );
        assert.strictEqual(rangedRead.kind, 'read_file');
        assert.strictEqual(rangedRead.startLine, 120);
        assert.strictEqual(rangedRead.endLine, 260);
    });

    test('buildPendingIssueCodeEvidence prioritizes explicit file:line citations, refutes non-existent cross-project files, and routeWorkflow pins non-infra cycles to CODE_AUDIT', () => {
        const kruschGitDir = fileURLToPath(new URL('../../krusch-git', import.meta.url));
        const evidence = buildPendingIssueCodeEvidence(
            '- [ ] `scripts/sync_to_pg.js:100` — Verify batch embedding sync logic and `scripts/sync_to_pg.js:143` error handling.\n- [ ] `src/context-bridge.js:213` — Bleed from krusch-polygres-connector.\n- [ ] Check `searchBlobs` in `server/git-engine.js`.',
            kruschGitDir,
            '1.2_COMPLIANCE_MATRIX'
        );
        assert.ok(evidence.includes('Live Source Window around `scripts/sync_to_pg.js:100`'), 'Must include window around scripts/sync_to_pg.js:100');
        assert.ok(evidence.includes('Live Source Window around `scripts/sync_to_pg.js:143`'), 'Must include window around scripts/sync_to_pg.js:143');
        assert.ok(evidence.includes('GROUND TRUTH REFUTATION — File `src/context-bridge.js` DOES NOT EXIST in `krusch-git`'), 'Must explicitly refute cross-project file bleed');
        assert.ok(evidence.includes('Live Source for Symbol `searchBlobs`'), 'Must include full function body for searchBlobs');

        const phase0Inflight = readInflight('krusch-law', '0_ORIENTATION', false);
        assert.ok(phase0Inflight.block.includes('fresh 11-phase audit rotation for `krusch-law`'), 'Phase 0 must reset inflight to prevent cross-project bleed');

        const wf = routeWorkflow('Verify memory and load handling in krusch-git', false);
        assert.ok(wf.includes('WORKFLOW_CODE_AUDIT'), 'Non-infra cycles must remain in CODE_AUDIT workflow even when goal mentions memory/load');
    });

    test('Colab G4 120B opportunistic routing, daytime gate override, and isolated live project test verification', async () => {
        const harnessCode = fs.readFileSync(
            new URL('../src/harness.js', import.meta.url),
            'utf8'
        );
        assert.ok(harnessCode.includes("envOr('COLAB_PROBE_TIMEOUT_MS', null, '3000')"), 'harness.js must default COLAB_PROBE_TIMEOUT_MS to 3000ms for WAN SSH tunnel probes');
        assert.ok(harnessCode.includes('forceRefresh'), 'harness.js /health must support forceRefresh (?refresh=1)');

        const sweepCode = fs.readFileSync(
            new URL('../src/jean_sre_sweep.js', import.meta.url),
            'utf8'
        );
        assert.ok(sweepCode.includes('|| colabStatus.active;'), 'jean_sre_sweep.js must override daytime sleep gate when Colab G4 120B is active');

        const kruschGitDir = fileURLToPath(new URL('../../krusch-git', import.meta.url));
        const suiteRes = runProjectVerificationSuite(kruschGitDir, { force: true });
        assert.strictEqual(suiteRes.status, 'PASS', `Expected krusch-git test suite to PASS with isolated env, got ${suiteRes.status}: ${suiteRes.output.slice(0, 300)}`);
        assert.ok(suiteRes.report.includes('Live Project Test Suite Execution (`krusch-git` — GROUND TRUTH)'));
    });

    test('120B Teacher-to-30B Student runtime distillation and SFT dataset pipeline', () => {
        const harnessCode = fs.readFileSync(
            new URL('../src/harness.js', import.meta.url),
            'utf8'
        );
        assert.ok(harnessCode.includes('export function recordTeacherSftTurn'), 'harness.js must export recordTeacherSftTurn for QLoRA SFT JSONL capture');
        assert.ok(harnessCode.includes('gpt_oss_120b_sft.jsonl'), 'harness.js must write ChatML trajectories to gpt_oss_120b_sft.jsonl');
        assert.ok(harnessCode.includes("'compass-concierge'"), 'harness.js CURATED_ROTATION must include compass-concierge');
        assert.ok(harnessCode.includes('loadSafeHarnessConfig'), 'harness.js must use loadSafeHarnessConfig fallback (BUG-03)');

        const kruschGitDir = fileURLToPath(new URL('../../krusch-git', import.meta.url));
        const distilled = distillTeacherKnowledgeForStudent(kruschGitDir, {
            teacherModel: 'gpt-oss:120b'
        });
        assert.ok(distilled, 'Expected distillTeacherKnowledgeForStudent to produce distilled object for krusch-git');
        assert.strictEqual(distilled.project, 'krusch-git');
        assert.strictEqual(distilled.teacherModel, 'gpt-oss:120b');

        const brief = loadDistilledTeacherBrief(kruschGitDir, 850);
        assert.ok(brief.includes('Distilled 120B Teacher Prior'), 'Expected loadDistilledTeacherBrief to format Student prior header');
        assert.ok(brief.length <= 850, `Expected brief <= 850 chars to preserve INV-JEAN-CTX-01, got ${brief.length}`);
        assert.ok(harnessCode.includes('export function getBooleanEnv'), 'harness.js must export getBooleanEnv (SEC-01)');
        assert.ok(harnessCode.includes('_colabProbeState.inFlight'), 'harness.js must deduplicate concurrent probes via _colabProbeState.inFlight (BUG-01)');
        assert.ok(!harnessCode.includes('_colabProbeCache'), 'harness.js must have zero dangling _colabProbeCache references');
    });

    test('Whole-Homelab audit rotation (@homelab/* root targets + uncurated projects) and Staleness Triage', async () => {
        const {
            resolveTargetDir,
            CORE_PROJECTS,
            HOMELAB_ROOT_TARGETS,
            PROJECT_ROTATION
        } = await import('../src/jean_sre_sweep.js');
        const {
            buildThinkerPrompt,
            buildImplementerPrompt,
            buildInfraThinkerPrompt
        } = await import('../src/prompts.js');

        assert.strictEqual(CORE_PROJECTS.length, 27, 'CORE_PROJECTS must retain 27 active core projects');
        assert.ok(!CORE_PROJECTS.includes('krusch-pg-agent'), 'Archived krusch-pg-agent must not be in CORE_PROJECTS');
        assert.ok(!PROJECT_ROTATION.includes('krusch-pg-agent'), 'Archived krusch-pg-agent must not be in PROJECT_ROTATION');
        assert.ok(PROJECT_ROTATION.includes('krusch-dbos-mcp'), 'Archived krusch-dbos-mcp must remain in PROJECT_ROTATION for SFT training data');
        assert.ok(HOMELAB_ROOT_TARGETS.includes('@homelab/lib'), 'HOMELAB_ROOT_TARGETS must include @homelab/lib');
        assert.ok(HOMELAB_ROOT_TARGETS.includes('@homelab/scripts'), 'HOMELAB_ROOT_TARGETS must include @homelab/scripts');
        assert.ok(PROJECT_ROTATION.length > CORE_PROJECTS.length + HOMELAB_ROOT_TARGETS.length, `PROJECT_ROTATION (${PROJECT_ROTATION.length}) must include uncurated projects beyond core + root targets`);

        assert.ok(resolveTargetDir('@homelab/lib').endsWith('/homelab/lib'), 'resolveTargetDir(@homelab/lib) must resolve to root homelab/lib');
        assert.ok(resolveTargetDir('krusch-git').endsWith('/homelab/projects/krusch-git'), 'resolveTargetDir(krusch-git) must resolve to homelab/projects/krusch-git');
        assert.ok(resolveTargetDir('krusch-dbos-mcp').endsWith('/homelab/projects/archive/krusch-dbos-mcp'), 'resolveTargetDir(krusch-dbos-mcp) must resolve to homelab/projects/archive/krusch-dbos-mcp');

        const synthThinker = buildThinkerPrompt('ctx', 'inflight', 'nextGoal', 'CODE_AUDIT', '', '@homelab/lib', '/home/krusch/homelab/lib', '2_MASTER_BACKLOG', 10, 11, 'checklist');
        assert.ok(synthThinker.includes('STALE-01'), 'Synthesis Thinker prompt must instruct emitting STALE-01 cards');
        assert.ok(synthThinker.includes('### 5. STALENESS & ARCHIVAL VERDICT'), 'Synthesis Thinker prompt must require Section 5 (STALENESS & ARCHIVAL VERDICT)');

        const synthImpl = buildImplementerPrompt('MASTER BACKLOG analysis', [], 'inflight', [], {
            targetProject: '@homelab/lib',
            phase: '2_MASTER_BACKLOG',
            phaseNum: 10,
            totalPhases: 11
        });
        assert.ok(synthImpl.includes('### 5. STALENESS & ARCHIVAL VERDICT'), 'Synthesis Implementer prompt must preserve Section 5 (STALENESS & ARCHIVAL VERDICT)');

        const hygienePrompt = buildInfraThinkerPrompt('ctx', 'inflight', { node: 'krusch', focus: 'homelab_hygiene' });
        assert.ok(hygienePrompt.includes('Audit the root ~/homelab monorepo folder'), 'Infra Thinker prompt must support homelab_hygiene checklist');
    });
});
