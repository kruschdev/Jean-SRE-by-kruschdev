import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert';
import {
    MAX_MEMORY_ENTRIES,
    setBoundedMemory,
    getBoundedMemory,
    clearBoundedMemory,
    getBoundedMemorySize,
    withMemoryDbLock,
    normalizeConversationId
} from '../src/db.js';

describe('Jean SRE In-Memory Fallback Cache & Eviction (ARCH-01 / TEST-01)', () => {
    beforeEach(async () => {
        await clearBoundedMemory();
    });

    test('setBoundedMemory enforces MAX_MEMORY_ENTRIES and evicts the oldest key (TEST-01)', async () => {
        for (let i = 0; i < MAX_MEMORY_ENTRIES + 2; i++) {
            await setBoundedMemory(`conv-${i}`, [{ role: 'user', content: `msg-${i}` }]);
        }

        const size = await getBoundedMemorySize();
        assert.strictEqual(size, MAX_MEMORY_ENTRIES);
        assert.strictEqual(await getBoundedMemory('conv-0'), undefined);
        assert.strictEqual(await getBoundedMemory('conv-1'), undefined);
        assert.deepStrictEqual(await getBoundedMemory('conv-2'), [{ role: 'user', content: 'msg-2' }]);
        assert.deepStrictEqual(
            await getBoundedMemory(`conv-${MAX_MEMORY_ENTRIES + 1}`),
            [{ role: 'user', content: `msg-${MAX_MEMORY_ENTRIES + 1}` }]
        );
    });

    test('updating an existing key refreshes its LRU order so it is not evicted first', async () => {
        for (let i = 0; i < MAX_MEMORY_ENTRIES; i++) {
            await setBoundedMemory(`key-${i}`, [{ idx: i }]);
        }
        // Refresh key-0 so key-1 becomes the oldest
        await setBoundedMemory('key-0', [{ idx: 0, refreshed: true }]);
        await setBoundedMemory('key-overflow', [{ idx: 999 }]);

        assert.strictEqual(await getBoundedMemorySize(), MAX_MEMORY_ENTRIES);
        assert.deepStrictEqual(await getBoundedMemory('key-0'), [{ idx: 0, refreshed: true }]);
        assert.strictEqual(await getBoundedMemory('key-1'), undefined);
    });

    test('withMemoryDbLock serializes concurrent mutations without lost updates (ARCH-01)', async () => {
        const order = [];
        await Promise.all([
            withMemoryDbLock(async () => {
                await new Promise((r) => setTimeout(r, 10));
                order.push('first');
            }),
            withMemoryDbLock(async () => {
                order.push('second');
            })
        ]);
        assert.deepStrictEqual(order, ['first', 'second']);
    });

    test('withMemoryDbLock propagates task rejections to caller while isolating subsequent MemoryCache operations (BUG-01 / BUG-02)', async () => {
        await assert.rejects(
            () =>
                withMemoryDbLock(async () => {
                    throw new Error('simulated lock failure');
                }),
            /simulated lock failure/
        );

        await setBoundedMemory('post-failure-key', [{ role: 'assistant', content: 'isolated' }]);
        const recovered = await getBoundedMemory('post-failure-key');
        assert.deepStrictEqual(recovered, [{ role: 'assistant', content: 'isolated' }]);
    });

    test('normalizeConversationId produces deterministic UUID v4 format for arbitrary strings', () => {
        const id1 = normalizeConversationId('session-alpha');
        const id2 = normalizeConversationId('session-alpha');
        assert.strictEqual(id1, id2);
        assert.match(id1, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-a[0-9a-f]{3}-[0-9a-f]{12}$/);
    });

    test('normalizePortableSftText strips host-specific paths into portable relative and $WORKSPACE_ROOT paths', async () => {
        const { normalizePortableSftText } = await import('../src/harness.js');
        const raw = 'Inspect /home/agent/homelab/projects/krusch-git/src/server.js and /home/krusch/homelab/lib/llm.js in /home/agent/homelab/projects/krusch-git';
        const normalized = normalizePortableSftText(raw, 'krusch-git');
        assert.strictEqual(
            normalized,
            'Inspect src/server.js and $WORKSPACE_ROOT/lib/llm.js in .'
        );
    });

    test('scoreElitistTrajectory ranks Argon-verified and verbatim-grounded trajectories above ungrounded text', async () => {
        const { scoreElitistTrajectory } = await import('../scripts/export-120b-distillation.js');
        const high = scoreElitistTrajectory({
            source: 'argon_golden_remediation',
            messages: [
                { role: 'system', content: 'sys' },
                { role: 'user', content: 'usr' },
                {
                    role: 'assistant',
                    content: `[BUG-01] VERIFIED in src/server.js:42 and src/db.js:18 and src/auth.js:90.\n\`\`\`js\nconst ok = true;\n\`\`\`\n\`\`\`diff\n+fix\n\`\`\`\n${'A'.repeat(400)}`
                }
            ]
        });
        const hallucinated = scoreElitistTrajectory({
            messages: [
                { role: 'system', content: 'sys' },
                { role: 'user', content: 'usr' },
                { role: 'assistant', content: `HALLUCINATION FLAG ${'B'.repeat(450)}` }
            ]
        });
        assert.ok(high >= 15, `Expected high elitist score >= 15, got ${high}`);
        assert.strictEqual(hallucinated, 0);
    });

    test('loadSafeHarnessConfig fails fast on malformed config errors and returns defaults when config file is absent (BUG-01)', async () => {
        const { loadSafeHarnessConfig, DEFAULT_HARNESS_CONFIG } = await import('../src/harness.js');

        await assert.rejects(
            () =>
                loadSafeHarnessConfig('/tmp/bad', async () => {
                    throw new SyntaxError('Unexpected token } in JSON');
                }),
            /Unexpected token \} in JSON/
        );

        const fallback = await loadSafeHarnessConfig('/tmp/empty', async () => ({}));
        assert.deepStrictEqual(fallback, DEFAULT_HARNESS_CONFIG);
    });
});

