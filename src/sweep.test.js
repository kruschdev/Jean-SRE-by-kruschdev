import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
    parseExecutableAction,
    classifyIntentRisk,
    checkIntentDrift,
    parseMasterBacklogCards,
    isCleanPassBacklog,
    extractStaleCleanupCards,
    finalizeMasterBacklogArtifacts,
    reconcileProjectBlockers,
    buildProjectAstIndex,
    buildProjectSpecAndDocsBrief,
    loadHomelabBaselineRules,
    buildPendingIssueCodeEvidence,
    readInflight,
    computeProjectSourceSignature
} from './jean_sre_sweep.js';

test('Sweep Unit Tests - parseExecutableAction', async (t) => {
    await t.test('parses read_file action (including **read_file** bold formatting) and resolves relative paths', () => {
        const res = parseExecutableAction(
            '[read_file] `src/harness.js` **Reason**: Inspect route handlers.',
            '/home/krusch/homelab/projects/jean-sre'
        );
        assert.equal(res.kind, 'read_file');
        assert.equal(res.filePath, '/home/krusch/homelab/projects/jean-sre/src/harness.js');

        const boldRes = parseExecutableAction(
            '1. **read_file** `/home/agent/homelab/projects/krusch-nexus/src/frontend/app.py` **Reason**: Inspect entrypoint.'
        );
        assert.equal(boldRes.kind, 'read_file');
        assert.ok(boldRes.filePath.endsWith('/projects/krusch-nexus/src/frontend/app.py'));
    });

    await t.test('parses Tier 2 AST search_symbols and dependency_graph actions', () => {
        const sym = parseExecutableAction(
            '[search_symbols] `classifyIntentRisk` **Reason**: Find symbol declaration.'
        );
        assert.equal(sym.kind, 'search_symbols');
        assert.equal(sym.query, 'classifyIntentRisk');

        const dep = parseExecutableAction(
            '[dependency_graph] `src/db.js` **Reason**: Trace inbound callers.'
        );
        assert.equal(dep.kind, 'dependency_graph');
        assert.equal(dep.target, 'src/db.js');
    });

    await t.test('parses run_shell_command action', () => {
        const res = parseExecutableAction(
            '[run_shell_command] `grep -rn "verifyToolCompliance" src/` **Reason**: Verify compliance calls.'
        );
        assert.equal(res.kind, 'run_shell_command');
        assert.equal(res.shellCommand, 'grep -rn "verifyToolCompliance" src/');
    });

    await t.test('parses real krusch-fs, krusch-frontend, and sentinel FRAXIS MCP tool actions as L2 safe intents', () => {
        const locateStr = '1. **Action**: [krusch_fs_locate] `krusch_fs_locate({ pattern: "*.env*", node: "all", category: "env" })` **Reason**: Inspect .env keys.';
        const fsLocate = parseExecutableAction(locateStr);
        assert.equal(fsLocate.kind, 'mcp_tool');
        assert.equal(fsLocate.toolName, 'krusch-fs-krusch-fs-locate');
        assert.equal(fsLocate.toolArgs.pattern, '*.env*');
        assert.equal(classifyIntentRisk(locateStr, fsLocate).riskClass, 'L2');

        const designStr = '2. **Action**: `query_design_rules({ queryStr: "layout spacing contrast" })` **Reason**: Check canonical layout rules.';
        const designRules = parseExecutableAction(designStr);
        assert.equal(designRules.kind, 'mcp_tool');
        assert.equal(designRules.toolName, 'krusch-frontend-query-design-rules');
        assert.equal(classifyIntentRisk(designStr, designRules).riskClass, 'L2');

        const fraxisStr = '3. **Action**: tool call `audit_agent_self_state({ node: "kruschdev" })` **Reason**: Check AGENTS.md and .env permissions.';
        const fraxisAudit = parseExecutableAction(fraxisStr);
        assert.equal(fraxisAudit.kind, 'sentinel_tool');
        assert.equal(fraxisAudit.toolName, 'sentinel-audit-agent-self-state');
        assert.equal(fraxisAudit.toolArgs.node, 'kruschdev');
        assert.equal(classifyIntentRisk(fraxisStr, fraxisAudit).riskClass, 'L2');

        const gitSemStr = '4. **Action**: `krusch_git_semantic_search({ repo: "jean-sre", query: "connection pool retry" })` **Reason**: Search code blobs semantically.';
        const gitSem = parseExecutableAction(gitSemStr, '/home/krusch/homelab/projects/jean-sre');
        assert.equal(gitSem.kind, 'mcp_tool');
        assert.equal(gitSem.toolName, 'krusch-git-krusch-git-semantic-search');
        assert.equal(gitSem.toolArgs.repo, 'jean-sre');
        assert.equal(classifyIntentRisk(gitSemStr, gitSem).riskClass, 'L2');
    });
});

test('Sweep Unit Tests - classifyIntentRisk & SEC-01 Hardening', async (t) => {
    await t.test('classifies read-only grep and AST queries as L2', () => {
        const grepTask = '[run_shell_command] `grep -rn "pool" src/` **Reason**: Check pool usage.';
        const parsed = parseExecutableAction(grepTask);
        const risk = classifyIntentRisk(grepTask, parsed);
        assert.equal(risk.riskClass, 'L2');

        const astTask = '[dependency_graph] `src/harness.js` **Reason**: Check callers.';
        const astParsed = parseExecutableAction(astTask);
        const astRisk = classifyIntentRisk(astTask, astParsed);
        assert.equal(astRisk.riskClass, 'L2');
    });

    await t.test('blocks restricted system paths via verifyToolCompliance as L0', () => {
        const task = '[read_file] `/etc/shadow` **Reason**: Attempt restricted read.';
        const parsed = parseExecutableAction(task);
        const risk = classifyIntentRisk(task, parsed);
        assert.equal(risk.riskClass, 'L0');
        assert.match(risk.reason, /restricted system path/i);
    });

    await t.test('blocks sudo and rsync --delete without database exclusions as L0', () => {
        const sudoTask = '[run_shell_command] `sudo systemctl status docker` **Reason**: Check status.';
        const sudoParsed = parseExecutableAction(sudoTask);
        assert.equal(classifyIntentRisk(sudoTask, sudoParsed).riskClass, 'L0');

        const rsyncTask = '[run_shell_command] `rsync -avz --delete ./src/ kruschdev:/tmp/out/` **Reason**: Sync.';
        const rsyncParsed = parseExecutableAction(rsyncTask);
        assert.equal(classifyIntentRisk(rsyncTask, rsyncParsed).riskClass, 'L0');
    });

    await t.test('gates subshell substitution, inline scripts, and mutating curl as L1', () => {
        const subshellTask = '[run_shell_command] `echo $(whoami)` **Reason**: Test subshell.';
        assert.equal(classifyIntentRisk(subshellTask, parseExecutableAction(subshellTask)).riskClass, 'L1');

        const inlineNodeTask = '[run_shell_command] `node -e "console.log(1)"` **Reason**: Inline node.';
        assert.equal(classifyIntentRisk(inlineNodeTask, parseExecutableAction(inlineNodeTask)).riskClass, 'L1');

        const curlPostTask = '[run_shell_command] `curl -X POST http://localhost:3005/trigger -d "{}"` **Reason**: Mutating HTTP.';
        assert.equal(classifyIntentRisk(curlPostTask, parseExecutableAction(curlPostTask)).riskClass, 'L1');
    });
});

test('Sweep Unit Tests - checkIntentDrift', async (t) => {
    await t.test('detects exact and near-identical command drift while allowing novel commands', async () => {
        const recent = [
            'read_file /home/krusch/homelab/projects/jean-sre/src/harness.js',
            'grep -rn "verifyToolCompliance" /home/krusch/homelab/projects/jean-sre/src'
        ];

        const exact = await checkIntentDrift(
            '[read_file] `/home/krusch/homelab/projects/jean-sre/src/harness.js` **Reason**: Re-read file.',
            recent
        );
        assert.equal(exact.isDrift, true);

        const novel = await checkIntentDrift(
            '[dependency_graph] `src/mcp-client.js` **Reason**: Inspect MCP client dependency graph.',
            recent
        );
        assert.equal(novel.isDrift, false);
    });
});

test('Sweep Unit Tests - parseMasterBacklogCards', async (t) => {
    await t.test('extracts structured task cards from Master Backlog markdown', () => {
        const sampleBacklog = `
### 3.1 Master Backlog

#### BUG-01
- **Title:** Emergency Trigger Active-Project Split-Brain
- **Severity:** P1
- **Effort:** S
- **Evidence:** \`src/harness.js:609\`
- **Fix Instructions:**
  1. Change active_project to 'jean-sre'.

#### SEC-01
- **Title:** Enforce Compliance Gate
- **Severity:** P1
- **Effort:** M
- **Evidence:** \`src/jean_sre_sweep.js:971\`
- **Fix Instructions:**
  1. Call verifyToolCompliance inside classifyIntentRisk.
`;
        const cards = parseMasterBacklogCards(sampleBacklog);
        assert.equal(cards.length, 2);
        assert.equal(cards[0].id, 'BUG-01');
        assert.equal(cards[0].severity, 'P1');
        assert.equal(cards[0].effort, 'S');
        assert.equal(cards[1].id, 'SEC-01');
    });

    await t.test('honors Referee Summary Table CLEAN_PASS despite Phase 1/2 recap headings and extracts STALE-* separately', () => {
        const cleanPassWithRecap = `
### MASTER BACKLOG: home-ai

#### BUG-01: Add defensive error handling to client API helper
- **Severity:** P1
- **Effort:** S
- **Evidence:** \`client/js/app.js:20-37\`
- **Fix Instructions:** Already verified in live source.

#### STALE-01: Remove unused ingestion scripts
- **Severity:** P2
- **Effort:** S
- **Evidence:** \`scripts/ingest-old.js:1\`
- **Fix Instructions:** Confirm unused and archive.

### 1. SUMMARY TABLE

| ID | Title | Severity | Effort | Confidence | Source Audit |
|----|-------|----------|--------|------------|--------------|
| **CLEAN_PASS** | Zero actionable defects verified | — | — | — | — |

### 4. NOT FLAGGED

- **[BUG-01 | P1] Add defensive error handling** (\`client/js/app.js:20-37\`) — Confirmed in test suite.
`;
        assert.equal(isCleanPassBacklog(cleanPassWithRecap), true);
        assert.deepEqual(parseMasterBacklogCards(cleanPassWithRecap), []);
        const stale = extractStaleCleanupCards(cleanPassWithRecap);
        assert.equal(stale.length, 1);
        assert.equal(stale[0].id, 'STALE-01');
    });

    await t.test('restricts parsed cards to bolded IDs approved in ### 1. SUMMARY TABLE', () => {
        const refereeFiltered = `
#### BUG-01: Real stream cancellation bug
- **Severity:** P1
- **Effort:** S
- **Evidence:** \`src/cascade.ts:1743\`
- **Fix Instructions:** Re-throw cancellation error.

#### BUG-02: Rejected recap card not in summary table
- **Severity:** P1
- **Effort:** M
- **Evidence:** \`src/cascade.ts:376\`
- **Fix Instructions:** Should be ignored because Referee omitted it from Summary Table.

### 1. SUMMARY TABLE

| ID | Title | Severity | Effort | Confidence | Source Audit |
|----|-------|----------|--------|------------|--------------|
| **BUG-01** | Real stream cancellation bug | P1 | S | High | 1.5 |
`;
        assert.equal(isCleanPassBacklog(refereeFiltered), false);
        const cards = parseMasterBacklogCards(refereeFiltered);
        assert.equal(cards.length, 1);
        assert.equal(cards[0].id, 'BUG-01');
        assert.equal(cards[0].evidence, 'src/cascade.ts:1743');
    });

    await t.test('finalizeMasterBacklogArtifacts overwrites stale task.md on CLEAN_PASS and reconcileProjectBlockers invalidates only owned stale audit blockers', async () => {
        const tmpAuditDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jean-audit-test-'));
        const taskPath = path.join(tmpAuditDir, 'task.md');
        fs.writeFileSync(taskPath, '# Stale Tasks\n- [ ] **[BUG-99] Old bug** (P1 / Effort: S)\n', 'utf8');

        const toolCalls = [];
        const mockCallTool = async (toolName, args) => {
            toolCalls.push({ toolName, args });
            if (toolName === 'krusch-context-krusch-context-list-memories') {
                if (args.project === 'unit-test-proj') {
                    return [
                        '=== 📋 Memory List (3 records) ===',
                        '* **#101** [blocker] (ACTIVE): [unit-test-proj] [BUG-99 | P1] Old resolved bug (src/old.js:10)',
                        '* **#102** [blocker] (ACTIVE): Human note: waiting on upstream vendor API key',
                        '* **#103** [blocker] (INVALIDATED): [unit-test-proj] [BUG-00 | P1] Already invalidated'
                    ].join('\n');
                }
                if (args.project === 'jean-sre') {
                    return [
                        '=== 📋 Memory List (3 records) ===',
                        '* **#501** [blocker] (ACTIVE): [unit-test-proj] [BUG-99 | P1] Old resolved bug (src/old.js:10)',
                        '* **#502** [blocker] (ACTIVE): [other-project] [SEC-01 | P0] Unrelated project blocker (src/auth.js:1)',
                        '* **#503** [blocker] (ACTIVE): [BUG-07 | P1] Jean SRE internal blocker (src/harness.js:12)'
                    ].join('\n');
                }
            }
            return 'ok';
        };

        try {
            const cleanPassReply = `
### 1. SUMMARY TABLE

| ID | Title | Severity | Effort | Confidence | Source Audit |
|----|-------|----------|--------|------------|--------------|
| **CLEAN_PASS** | Zero actionable defects verified | — | — | — | — |
`;
            await finalizeMasterBacklogArtifacts('unit-test-proj', tmpAuditDir, cleanPassReply, { callToolFn: mockCallTool });
            const updatedTaskMd = fs.readFileSync(taskPath, 'utf8');
            assert.match(updatedTaskMd, /CLEAN_PASS/);
            assert.doesNotMatch(updatedTaskMd, /BUG-99/);

            const revocations = toolCalls.filter(c => c.toolName === 'krusch_context_revise');
            assert.deepEqual(
                revocations.map(r => ({ id: r.args.target_id, project: r.args.project })),
                [
                    { id: 101, project: 'unit-test-proj' },
                    { id: 501, project: 'jean-sre' }
                ]
            );

            // Also verify jean-sre self-reconciliation does not touch [other-project] blockers
            const selfCalls = [];
            await reconcileProjectBlockers('jean-sre', [], async (toolName, args) => {
                selfCalls.push({ toolName, args });
                if (toolName === 'krusch-context-krusch-context-list-memories') {
                    return [
                        '* **#502** [blocker] (ACTIVE): [other-project] [SEC-01 | P0] Unrelated project blocker (src/auth.js:1)',
                        '* **#503** [blocker] (ACTIVE): [BUG-07 | P1] Jean SRE internal blocker (src/harness.js:12)'
                    ].join('\n');
                }
                return 'ok';
            });
            const selfRevocations = selfCalls.filter(c => c.toolName === 'krusch_context_revise');
            assert.deepEqual(selfRevocations.map(r => r.args.target_id), [503]);
        } finally {
            fs.rmSync(tmpAuditDir, { recursive: true, force: true });
        }
    });
});

test('Sweep Unit Tests - Python AST Indexing, Spec Brief & Cross-Project Refutation', async (t) => {
    const nexusPath = new URL('../../krusch-nexus', import.meta.url).pathname;
    const jeanSrePath = new URL('..', import.meta.url).pathname;

    await t.test('buildProjectAstIndex indexes Python source files in krusch-nexus without exhausting budget on docs/fixtures', () => {
        const { fileStats, allSymbols } = buildProjectAstIndex(nexusPath);
        assert.ok(fileStats.length >= 15, `Expected >= 15 code files in krusch-nexus, got ${fileStats.length}`);
        assert.ok(allSymbols.length >= 50, `Expected >= 50 symbols in krusch-nexus, got ${allSymbols.length}`);
        assert.ok(fileStats.some(f => f.relPath.includes('src/krusch_nexus/')), 'Expected src/krusch_nexus/*.py files to be indexed');
    });

    await t.test('buildProjectSpecAndDocsBrief and loadHomelabBaselineRules recognize Python pyproject.toml projects', () => {
        const brief = buildProjectSpecAndDocsBrief(nexusPath);
        assert.match(brief, /pyproject\.toml \(Python 3\.12 Project\)/);
        assert.match(brief, /Pure Python repository/);

        const rules = loadHomelabBaselineRules(nexusPath);
        assert.match(rules, /Python 3\.12 Homelab Standards/);
        assert.doesNotMatch(rules, /Baseline Rule File: `\.agent\/rules\/javascript\.md`/);
    });

    await t.test('readInflight resets Pending Issues at 0_ORIENTATION and buildPendingIssueCodeEvidence refutes non-existent files', () => {
        const inflight0 = readInflight('krusch-nexus', '0_ORIENTATION', false);
        assert.match(inflight0.block, /fresh 11-phase audit rotation/);

        const evidence = buildPendingIssueCodeEvidence('Pending issue in `src/cascade.ts:126`', nexusPath, '0_ORIENTATION');
        assert.match(evidence, /GROUND TRUTH REFUTATION/);
        assert.match(evidence, /src\/cascade\.ts/);
    });

    await t.test('computeProjectSourceSignature computes deterministic SHA-256 source tree digest', () => {
        const sig1 = computeProjectSourceSignature(jeanSrePath);
        const sig2 = computeProjectSourceSignature(jeanSrePath);
        assert.equal(sig1.length, 24);
        assert.equal(sig1, sig2);
    });
});

