import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseThinkerPlan } from './parser.js';

test('Strategy 1: Extact numbered items from INVESTIGATION PLAN', () => {
    const input = `
### ANALYSIS
Some analysis here.

### INVESTIGATION PLAN
1. **Action**: \`run_shell_command ls -la\`
   **Reason**: To see files
2. **Action**: \`read_file /path/to/file.js\`
   **Reason**: Because it is important
    `;
    const result = parseThinkerPlan(input);
    assert.equal(result.length, 2);
    assert.ok(result[0].task.includes('ls -la'));
    assert.ok(result[1].task.includes('/path/to/file.js'));
});

test('Strategy 2: Extract backtick commands', () => {
    const input = `I want to \`read_file /var/log/syslog\` and also \`grep error /var/log/auth.log\`.`;
    const result = parseThinkerPlan(input);
    assert.equal(result.length, 2);
    assert.equal(result[0].task, 'run_shell_command `read_file /var/log/syslog`');
    assert.equal(result[1].task, 'run_shell_command `grep error /var/log/auth.log`');
});

test('Strategy 3: Extract file paths that look like audit targets', () => {
    const input = `I found a bug in \`/home/kruschdev/homelab/projects/jean-sre/src/parser.js\`.`;
    const result = parseThinkerPlan(input);
    assert.equal(result.length, 1);
    assert.equal(result[0].task, 'read_file `/home/kruschdev/homelab/projects/jean-sre/src/parser.js`');
});

test('Strategy 4: Handle split-backtick format', () => {
    const input = `Let's use \`read_file\` \`/home/user/file.js\`.`;
    const result = parseThinkerPlan(input);
    assert.equal(result.length, 1);
    assert.equal(result[0].task, 'read_file `/home/user/file.js`');
});

test('Strategy 2 & 4: Extract Tier 2 AST symbol & dependency_graph commands', () => {
    const input = `Let's run \`search_symbols createServer\` and \`dependency_graph parseThinkerPlan\`.`;
    const result = parseThinkerPlan(input);
    assert.equal(result.length, 2);
    assert.equal(result[0].task, 'search_symbols createServer');
    assert.equal(result[1].task, 'dependency_graph parseThinkerPlan');
});

