import assert from 'node:assert';
import { verifyToolCompliance } from './compliance.js';

console.log('⏳ Running SRE Compliance Linter Unit Tests...\n');

// Mock tools and argument contexts
const toolName = 'krusch_execute_task';

// Test Cases for Sudo Violations
{
    console.log('Test 1: Sudo check blocks standard privilege escalation');
    const badArgs = { task: 'sudo apt update && sudo apt install nginx' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /Use of 'sudo' or 'su' is strictly prohibited/);
    console.log('✅ Test 1 Passed.');
}

{
    console.log('Test 2: Sudo check blocks "su -" root shifts');
    const badArgs = { command: 'su - postgres -c "psql"' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /Use of 'sudo' or 'su' is strictly prohibited/);
    console.log('✅ Test 2 Passed.');
}

{
    console.log('Test 3: Clean commands without sudo pass through');
    const cleanArgs = { task: 'npm run test && git diff' };
    const res = verifyToolCompliance(toolName, cleanArgs);
    assert.strictEqual(res.valid, true);
    console.log('✅ Test 3 Passed.');
}

// Test Cases for Rsync Safety
{
    console.log('Test 4: Blocks unsafe rsync --delete without exclusions');
    const badArgs = { task: 'rsync -avz --delete /src/ /dest/' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /catastrophic state loss/);
    console.log('✅ Test 4 Passed.');
}

{
    console.log('Test 5: Blocks rsync --delete with insufficient exclusions');
    const badArgs = { task: 'rsync -avz --delete --exclude=postgres /src/ /dest/' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /catastrophic state loss/);
    console.log('✅ Test 5 Passed.');
}

{
    console.log('Test 6: Allows safe rsync --delete with required exclusions');
    const safeArgs = { task: 'rsync -avz --delete --exclude=postgres --exclude=redis --exclude=data /src/ /dest/' };
    const res = verifyToolCompliance(toolName, safeArgs);
    assert.strictEqual(res.valid, true);
    console.log('✅ Test 6 Passed.');
}

// Test Cases for OS Drive Protection Policy
{
    console.log('Test 7: Blocks volume mapping directly to boot drive (/var/lib/docker)');
    const badArgs = { task: 'docker run -v /var/lib/docker:/var/lib/docker my-image' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /protected OS drive/);
    console.log('✅ Test 7 Passed.');
}

{
    console.log('Test 8: Allows mapped volumes residing under mount paths (/mnt/nvme)');
    const safeArgs = { task: 'docker run -v /mnt/nvme/docker-data:/var/lib/docker my-image' };
    const res = verifyToolCompliance(toolName, safeArgs);
    assert.strictEqual(res.valid, true);
    console.log('✅ Test 8 Passed.');
}

{
    console.log('Test 9: Blocks unrouted large model cache downloads (/.ollama)');
    const badArgs = { objective: 'mkdir -p ~/.ollama/models && curl -o ~/.ollama/models/my-weights.bin http://example.com' };
    const res = verifyToolCompliance(toolName, badArgs);
    assert.strictEqual(res.valid, false);
    assert.match(res.reason, /protected OS drive/);
    console.log('✅ Test 9 Passed.');
}

{
    console.log('Test 10: Allows routed model cache directories (/mnt/data)');
    const safeArgs = { objective: 'ln -s /mnt/data/ollama ~/.ollama && curl -o ~/.ollama/models/my-weights.bin http://example.com' };
    const res = verifyToolCompliance(toolName, safeArgs);
    assert.strictEqual(res.valid, true);
    console.log('✅ Test 10 Passed.');
}

console.log('\n🎉 All SRE Compliance Linter Unit Tests Passed Successfully!\n');
