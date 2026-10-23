import pg from 'pg';
import { createHash } from 'crypto';
import { envOr } from '@krusch/toolkit/config';

const DATABASE_URL = envOr('DBOS_DATABASE_URL', envOr('DATABASE_URL', null, null), null);
export const pool = DATABASE_URL ? new pg.Pool({ connectionString: DATABASE_URL, max: 6, idleTimeoutMillis: 30000 }) : null;

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const MAX_MEMORY_ENTRIES = 100;

// Bounded in-memory fallback cache encapsulated with failure isolation and async mutex (ARCH-01 / BUG-01 / BUG-02)
export class MemoryCache {
    constructor(maxEntries = MAX_MEMORY_ENTRIES) {
        this.maxEntries = maxEntries;
        this._store = new Map();
        this._lock = Promise.resolve();
    }

    async withLock(fn) {
        const next = this._lock.then(fn, fn);
        this._lock = next.catch((err) => {
            console.error('[MemoryDbLock] Lock task error:', err?.message || err);
            this._lock = Promise.resolve();
        });
        return next;
    }

    setSync(key, value) {
        if (this._store.has(key)) this._store.delete(key);
        this._store.set(key, value);
        if (this._store.size > this.maxEntries) {
            const oldestKey = this._store.keys().next().value;
            this._store.delete(oldestKey);
        }
    }

    async set(key, value) {
        return this.withLock(() => this.setSync(key, value));
    }

    async get(key) {
        return this.withLock(() => this._store.get(key));
    }

    async clear() {
        return this.withLock(() => this._store.clear());
    }

    async size() {
        return this.withLock(() => this._store.size);
    }
}

export const memoryCache = new MemoryCache(MAX_MEMORY_ENTRIES);

export async function withMemoryDbLock(fn) {
    return memoryCache.withLock(fn);
}

export function setBoundedMemorySync(key, value) {
    return memoryCache.setSync(key, value);
}

export async function setBoundedMemory(key, value) {
    return memoryCache.set(key, value);
}

export async function getBoundedMemory(key) {
    return memoryCache.get(key);
}

export async function clearBoundedMemory() {
    return memoryCache.clear();
}

export async function getBoundedMemorySize() {
    return memoryCache.size();
}

export function normalizeConversationId(conversationId) {
    const raw = String(conversationId || '').trim();
    if (UUID_REGEX.test(raw)) return raw.toLowerCase();
    const hex = createHash('sha256').update(raw || 'default').digest('hex');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}

export async function getHistory(conversationId) {
    const normId = normalizeConversationId(conversationId);
    if (!pool) return (await getBoundedMemory(normId)) || [];
    try {
        const res = await pool.query(
            `SELECT content FROM ide_agent_memory WHERE category = 'activity' AND id = $1`,
            [normId]
        );
        if (res.rows.length > 0) {
            try {
                return JSON.parse(res.rows[0].content);
            } catch {
                return [];
            }
        }
    } catch (e) {
        console.warn(`[DB Fallback] PostgreSQL query failed, using in-memory store: ${e.message}`);
        return (await getBoundedMemory(normId)) || [];
    }
    return [];
}

export async function saveHistory(conversationId, messages) {
    const normId = normalizeConversationId(conversationId);
    if (!pool) {
        await setBoundedMemory(normId, messages);
        return;
    }
    try {
        await pool.query(
            `INSERT INTO ide_agent_memory (id, category, content) 
             VALUES ($1, 'activity', $2) 
             ON CONFLICT (id) DO UPDATE SET content = $2`,
            [normId, JSON.stringify(messages)]
        );
    } catch (e) {
        console.warn(`[DB Fallback] PostgreSQL insert failed, saving to in-memory store: ${e.message}`);
        await setBoundedMemory(normId, messages);
    }
}

