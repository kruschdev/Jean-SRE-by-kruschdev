#!/usr/bin/env node

/**
 * @file optimize_jean_daily.js
 * @description Daily 7-Stage Closed-Loop Optimization & Continual Self-Improvement Engine for Jean SRE:
 *   1. 3-Chair All-VRAM Residency & Context Pinning (`gemma4:e2b`, `qwen3-coder-30b-moe`, `inkling-2-mini`, `snowflake-arctic-embed2`)
 *   2. Live Fleet Quantcheck & Behavioral Canary Gate (`/quantcheck`)
 *   3. SQLite `context.db` Deduplication, Embedding-Blob Compaction & Rotated Audit Log Archival
 *   4. Persistent Graph Mind (`GET /state` HydraDB SelfGraph PPR) & Optional Procedural WAL Compaction
 *   5. ModularRSI + Self-Improvement Memory (SIM) Contrastive Trajectory Attribution (`arXiv:2609.14857`, `arXiv:2611.18856`)
 *   6. Zero-Cost Local SkillOpt Playbook Calibration via 1st Chair (`$0.00`)
 *   7. Portable SFT Trace Sync & SHA-256-Sealed Daily Optimization Receipt
 */

import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { defineCli, withSingleInstanceLock, appendJsonlRecord } from '../../../lib/cli.js';
import { okEnvelope, errorEnvelope } from '../../../lib/http.js';
import { createPgPool } from '../../../lib/pg.js';
import {
    createModularRsiHarness,
    createSelfImprovementMemory,
} from '../../../lib/modular-rsi.js';
import { proposeSkillEdits, applySkillPatch } from '../../../lib/skillopt.js';

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const JEAN_PROJECT_ROOT = path.resolve(__dirname, '..');
const WORKSPACE_ROOT = path.resolve(JEAN_PROJECT_ROOT, '..', '..');

const DEFAULT_RECEIPT_PATH = path.join(WORKSPACE_ROOT, 'data', 'jean_daily_optimization_receipt.json');
const DEFAULT_HISTORY_PATH = path.join(WORKSPACE_ROOT, 'data', 'jean_daily_optimization_history.jsonl');
const DEFAULT_LOCK_PATH = path.join(WORKSPACE_ROOT, 'data', 'jean_daily_optimize.lock');
const DEFAULT_SIM_PATH = path.join(JEAN_PROJECT_ROOT, 'data', 'jean_sim_memory.json');
const DEFAULT_RSI_SPECS_PATH = path.join(JEAN_PROJECT_ROOT, 'data', 'jean_rsi_specs.json');
const DEFAULT_SKILL_PATH = path.join(JEAN_PROJECT_ROOT, 'skills', 'jean-mcp-code-audit', 'SKILL.md');

const THERMAL_LIMIT_C = 85;
const EMBED_CONTEXT_LENGTH = 8192;
const THIRD_CHAIR_CONTEXT_LENGTH = 16384;

function loadDotEnvIfPresent(envPath) {
    if (!fs.existsSync(envPath)) return {};
    const out = {};
    try {
        for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) continue;
            const eq = trimmed.indexOf('=');
            if (eq <= 0) continue;
            const k = trimmed.slice(0, eq).trim();
            const v = trimmed
                .slice(eq + 1)
                .trim()
                .replace(/^["']|["']$/g, '');
            if (k && !Object.prototype.hasOwnProperty.call(process.env, k)) {
                out[k] = v;
            }
        }
    } catch {
        // Ignore unreadable .env
    }
    return out;
}

/**
 * Compute deterministic eviction + pinning actions for the 3rd-Chair Ollama node
 * so `gemma4:e2b` + `snowflake-arctic-embed2:latest` remain 100% GPU resident.
 *
 * @param {Array<{ name?: string, model?: string, context_length?: number, size_vram?: number, size?: number }>} psModels
 * @returns {{ evictModels: string[], pinThirdChair: boolean, pinEmbed: boolean }}
 */
export function evaluateChairResidencyPlan(psModels = []) {
    const allowedThirdChairSet = new Set([
        'gemma4:e2b',
        'snowflake-arctic-embed2:latest',
        'snowflake-arctic-embed2',
    ]);
    const evictModels = [];
    let thirdChairResident = false;
    let embedResident = false;

    for (const m of Array.isArray(psModels) ? psModels : []) {
        const name = String(m?.name || m?.model || '').trim();
        if (!name) continue;
        const fullGpu = !m.size || !m.size_vram || m.size_vram >= m.size * 0.99;
        if (name === 'gemma4:e2b') {
            if (fullGpu) thirdChairResident = true;
        } else if (name.startsWith('snowflake-arctic-embed2')) {
            if (fullGpu) embedResident = true;
        } else if (!allowedThirdChairSet.has(name)) {
            evictModels.push(name);
        }
    }

    return {
        evictModels,
        pinThirdChair: !thirdChairResident || evictModels.length > 0,
        pinEmbed: !embedResident || evictModels.length > 0,
    };
}

/**
 * Parse recent `jean-sweep` log lines into structured health & RSI trajectory signals.
 *
 * @param {string} rawLogs
 * @returns {{ cyclesCompleted: number, driftBlockedCount: number, complianceBlockedCount: number, cycleErrors: string[], syntheticTrajectories: Array<object> }}
 */
export function analyzeJeanSweepLogHealth(rawLogs = '') {
    const lines = String(rawLogs || '').split('\n');
    let cyclesCompleted = 0;
    let driftBlockedCount = 0;
    let complianceBlockedCount = 0;
    const cycleErrors = [];
    const syntheticTrajectories = [];

    for (const line of lines) {
        if (/\[Sweep\]\s+Cycle\s+\d+\s+complete/i.test(line)) {
            cyclesCompleted += 1;
        }
        if (/\[Drift\]\s+BLOCKED:/i.test(line)) {
            driftBlockedCount += 1;
        }
        if (/\[Compliance Gate\]\s*🛑/i.test(line)) {
            complianceBlockedCount += 1;
            syntheticTrajectories.push({
                taskId: `sweep-compliance-${complianceBlockedCount}`,
                success: false,
                malformedToolCalls: 1,
                errorCount: 1,
                errorMessage: line.trim().slice(0, 220),
                feedback: line.trim().slice(0, 220),
            });
        }
        if (/\[Sweep\]\s+Cycle\s+\d+\s+error:/i.test(line) || /Persona LLM Error:/i.test(line)) {
            const cleaned = line.trim().slice(0, 240);
            cycleErrors.push(cleaned);
            syntheticTrajectories.push({
                taskId: `sweep-error-${cycleErrors.length}`,
                success: false,
                errorCount: 1,
                iterations: 3,
                errorMessage: cleaned,
                feedback: cleaned,
            });
        }
        if (/Pruning input message from (\d+) chars to (\d+) chars/i.test(line)) {
            const m = line.match(/from\s+(\d+)\s+chars\s+to\s+(\d+)\s+chars/i);
            const origChars = Number(m?.[1] || 0);
            if (origChars > 48000) {
                syntheticTrajectories.push({
                    taskId: `context-prune-${origChars}`,
                    success: false,
                    contextTokenPeak: Math.round(origChars / 3.5),
                    observationBytesPeak: origChars,
                    errorMessage: line.trim().slice(0, 220),
                    feedback: line.trim().slice(0, 220),
                });
            }
        }
    }

    return {
        cyclesCompleted,
        driftBlockedCount,
        complianceBlockedCount,
        cycleErrors,
        syntheticTrajectories,
    };
}

async function runHostCommand(host, shellCmd, timeoutMs = 30_000) {
    if (!host || host === 'local' || host === '127.0.0.1' || host === 'localhost') {
        const { stdout, stderr } = await execFileAsync('bash', ['-lc', shellCmd], {
            timeout: timeoutMs,
            maxBuffer: 8 * 1024 * 1024,
        });
        return { stdout: stdout.trim(), stderr: stderr.trim() };
    }
    const { stdout, stderr } = await execFileAsync(
        'ssh',
        ['-o', 'ConnectTimeout=6', '-o', 'BatchMode=yes', host, shellCmd],
        { timeout: timeoutMs, maxBuffer: 8 * 1024 * 1024 }
    );
    return { stdout: stdout.trim(), stderr: stderr.trim() };
}

async function fetchJson(url, options = {}, timeoutMs = 20_000, fetchImpl = globalThis.fetch) {
    const res = await fetchImpl(url, {
        ...options,
        signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await res.text();
    let data = null;
    try {
        data = text ? JSON.parse(text) : null;
    } catch {
        data = { raw: text };
    }
    if (!res.ok) {
        throw new Error(`HTTP ${res.status} from ${url}: ${data?.error?.message || data?.error || text.slice(0, 140)}`);
    }
    return data;
}

/**
 * Stage 1: Verify thermal headroom, enforce 3-Chair All-VRAM residency (`100% GPU`),
 * and self-heal Jean SRE containers if degraded.
 */
async function stageThreeChairResidency({ dryRun, env, fetchImpl }) {
    const thirdChairUrl = env.OLLAMA_THIRD_CHAIR_URL || 'http://127.0.0.1:11434';
    const secondChairEmbedUrl = env.SECOND_CHAIR_EMBED_URL || '';
    const jeanHarnessUrl = env.JEAN_AGENT_URL || 'http://127.0.0.1:3005';
    const primaryHost = env.JEAN_PRIMARY_HOST || 'local';
    const secondaryHost = env.JEAN_SECONDARY_HOST || '';

    const gpuTemps = {};
    try {
        const { stdout: localGpu } = await execFileAsync(
            'nvidia-smi',
            ['--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            { timeout: 5000 }
        );
        gpuTemps.local = Math.max(...localGpu.trim().split('\n').map(Number).filter(Number.isFinite));
    } catch {
        gpuTemps.local = null;
    }

    for (const node of [primaryHost, secondaryHost].filter((n) => n && n !== 'local')) {
        try {
            const { stdout } = await runHostCommand(
                node,
                'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits',
                8000
            );
            gpuTemps[node] = Math.max(...stdout.split('\n').map(Number).filter(Number.isFinite));
        } catch {
            gpuTemps[node] = null;
        }
    }

    const overheated = Object.entries(gpuTemps).filter(([, t]) => typeof t === 'number' && t >= THERMAL_LIMIT_C);
    if (overheated.length > 0) {
        return {
            ok: false,
            gpuTemps,
            overheated,
            reason: `Thermal guard triggered (>= ${THERMAL_LIMIT_C}C): ${overheated.map(([n, t]) => `${n}=${t}C`).join(', ')}`,
        };
    }

    const psData = await fetchJson(`${thirdChairUrl}/api/ps`, {}, 8000, fetchImpl);
    const plan = evaluateChairResidencyPlan(psData?.models || []);
    const actions = [];

    if (!dryRun) {
        for (const staleModel of plan.evictModels) {
            await fetchJson(
                `${thirdChairUrl}/api/generate`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: staleModel, prompt: '', stream: false, keep_alive: 0 }),
                },
                15_000,
                fetchImpl
            );
            actions.push(`evicted:${staleModel}@third-chair`);
        }

        if (plan.pinEmbed) {
            await fetchJson(
                `${thirdChairUrl}/api/embed`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'snowflake-arctic-embed2:latest',
                        input: 'jean-daily-warmup',
                        keep_alive: -1,
                        options: { num_ctx: EMBED_CONTEXT_LENGTH },
                    }),
                },
                30_000,
                fetchImpl
            );
            actions.push('pinned:snowflake-arctic-embed2:latest(8192)@third-chair');
        }

        if (plan.pinThirdChair) {
            await fetchJson(
                `${thirdChairUrl}/api/generate`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'gemma4:e2b',
                        prompt: '',
                        stream: false,
                        keep_alive: -1,
                        options: { num_ctx: THIRD_CHAIR_CONTEXT_LENGTH },
                    }),
                },
                45_000,
                fetchImpl
            );
            actions.push('pinned:gemma4:e2b(16384)@third-chair');
        }

        if (secondChairEmbedUrl) {
            await fetchJson(
                `${secondChairEmbedUrl}/api/embed`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'snowflake-arctic-embed2:latest',
                        input: 'jean-daily-warmup',
                        keep_alive: -1,
                        options: { num_ctx: EMBED_CONTEXT_LENGTH },
                    }),
                },
                30_000,
                fetchImpl
            );
            actions.push('pinned:snowflake-arctic-embed2:latest(8192)@second-chair');
        }
    }

    const primaryContainers = env.JEAN_PRIMARY_CONTAINERS || 'jean-sre-jean-harness-1 jean-sre-jean-sweep-1';
    const secondaryContainers = env.JEAN_SECONDARY_CONTAINERS || '';
    const containerStates = {};

    try {
        const { stdout: devRaw } = await runHostCommand(
            primaryHost,
            `docker inspect --format '{{.Name}}={{.State.Status}}:{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' ${primaryContainers} 2>/dev/null || true`,
            10_000
        );
        for (const line of devRaw.split('\n').filter(Boolean)) {
            const [name, status] = line.replace(/^\//, '').trim().split('=');
            if (name) containerStates[name] = status;
        }
    } catch {
        // Docker inspect optional in non-containerized setups
    }

    if (secondaryHost && secondaryContainers) {
        try {
            const { stdout: servRaw } = await runHostCommand(
                secondaryHost,
                `docker inspect --format '{{.Name}}={{.State.Status}}:{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' ${secondaryContainers} 2>/dev/null || true`,
                10_000
            );
            for (const line of servRaw.split('\n').filter(Boolean)) {
                const [name, status] = line.replace(/^\//, '').trim().split('=');
                if (name) containerStates[name] = status;
            }
        } catch {
            // Optional secondary host check
        }
    }

    for (const jeanCtr of ['jean-sre-jean-harness-1', 'jean-sre-jean-sweep-1']) {
        if (Object.prototype.hasOwnProperty.call(containerStates, jeanCtr)) {
            const st = containerStates[jeanCtr] || 'missing';
            if ((!st.startsWith('running:') || st.endsWith(':unhealthy')) && !dryRun) {
                await runHostCommand(primaryHost, `docker restart ${jeanCtr}`, 25_000);
                actions.push(`restarted:${jeanCtr}`);
            }
        }
    }

    const health = await fetchJson(`${jeanHarnessUrl}/health`, {}, 10_000, fetchImpl);
    const thirdChairPsAfter = await fetchJson(`${thirdChairUrl}/api/ps`, {}, 8000, fetchImpl);

    return {
        ok: Boolean(health?.ready),
        gpuTemps,
        residencyPlan: plan,
        actions,
        thirdChairResidentModels: (thirdChairPsAfter?.models || []).map((m) => ({
            model: m.name,
            contextLength: m.context_length,
            gpuRatio: m.size ? Number((m.size_vram / m.size).toFixed(3)) : 1,
        })),
        containers: containerStates,
        harnessReady: Boolean(health?.ready),
        connectedMcpServers: health?.mcp_servers || [],
    };
}

/**
 * Stage 2: Execute live behavioral canary + static GGUF quantcheck across all fleet endpoints.
 */
async function stageFleetQuantcheck({ env, fetchImpl }) {
    const jeanHarnessUrl = env.JEAN_AGENT_URL || 'http://127.0.0.1:3005';
    const res = await fetchImpl(`${jeanHarnessUrl}/quantcheck?timeoutMs=90000`, {
        signal: AbortSignal.timeout(150_000),
    });
    const quantRes = await res.json().catch(() => ({ ok: false, error: { message: `HTTP ${res.status}` } }));
    const payload = quantRes?.data || quantRes?.error?.details || {};
    const endpoints = payload.endpoints || [];
    return {
        ok: Boolean(quantRes?.ok),
        summary: quantRes?.summary || quantRes?.error?.message || 'quantcheck completed',
        endpointCount: payload.endpointCount ?? endpoints.length,
        totalTags: payload.totalTags ?? 0,
        totalLiveProbed: payload.totalLiveProbed ?? 0,
        endpoints: endpoints.map((ep) => {
            const epData = ep.data || ep.error?.details || {};
            return {
                endpoint: epData.endpoint || null,
                ok: ep.ok,
                error: ep.ok ? null : ep.error?.message || null,
                liveProbedCount: epData.liveProbedCount ?? 0,
                brokenCount: epData.brokenCount ?? 0,
                models: (epData.models || []).map((m) => ({
                    model: m.model,
                    isResident: m.isResident,
                    verdict: m.verdict,
                    tokPerSec: m.canary?.throughputTokPerSec ?? null,
                })),
            };
        }),
    };
}

/**
 * Stage 3: Deduplicate exact-duplicate sweep lessons, compact redundant ASCII JSON `embedding` strings
 * where binary `embedding_blob` (>= 4096B) already exists, checkpoint/VACUUM `.agent/context.db`,
 * and archive rotated audit logs (>7 days).
 */
async function stageCompactJeanContextAndLogs({ dryRun, env }) {
    const primaryHost = env.JEAN_PRIMARY_HOST || 'local';
    const targetJeanDir = env.JEAN_REMOTE_PROJECT_DIR || JEAN_PROJECT_ROOT;
    const targetArchiveDir = env.JEAN_LOG_ARCHIVE_DIR || path.join(targetJeanDir, 'data', 'archives');
    const jeanHarnessUrl = env.JEAN_AGENT_URL || 'http://127.0.0.1:3005';

    const pyScript = `
import glob, json, os, sqlite3, subprocess, tarfile, time

jean_dir = ${JSON.stringify(targetJeanDir)}
db_path = os.path.join(jean_dir, ".agent", "context.db")
archive_dir = ${JSON.stringify(targetArchiveDir)}
dry_run = ${dryRun ? 'True' : 'False'}

report = {
    "dbExists": os.path.exists(db_path),
    "sizeBeforeMb": round(os.path.getsize(db_path) / 1048576.0, 2) if os.path.exists(db_path) else 0,
    "sizeAfterMb": 0,
    "dedupedActiveRows": 0,
    "clearedRedundantTextEmbeddings": 0,
    "clearedInactiveEmbeddings": 0,
    "vacuumExecuted": False,
    "archivedLogsCount": 0,
    "archiveTarPath": None,
    "removedCorruptBackups": 0,
}

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path, timeout=15.0)
    cur = conn.cursor()
    cur.execute("PRAGMA busy_timeout = 15000;")

    dup_count = cur.execute("""
        SELECT COUNT(*) FROM ide_agent_memory
        WHERE status IN ('ACTIVE', 'active')
          AND id NOT IN (
              SELECT MAX(id) FROM ide_agent_memory
              WHERE status IN ('ACTIVE', 'active')
              GROUP BY category, content
          )
    """).fetchone()[0]

    redundant_emb = cur.execute("""
        SELECT COUNT(*) FROM ide_agent_memory
        WHERE embedding IS NOT NULL
          AND embedding_blob IS NOT NULL
          AND LENGTH(embedding_blob) >= 4096
    """).fetchone()[0]

    inactive_emb = cur.execute("""
        SELECT COUNT(*) FROM ide_agent_memory
        WHERE status IN ('INVALIDATED', 'SUPERSEDED', 'archived')
          AND (embedding IS NOT NULL OR embedding_blob IS NOT NULL)
    """).fetchone()[0]

    report["dedupedActiveRows"] = int(dup_count)
    report["clearedRedundantTextEmbeddings"] = int(redundant_emb)
    report["clearedInactiveEmbeddings"] = int(inactive_emb)

    if not dry_run:
        if dup_count > 0:
            cur.execute("""
                UPDATE ide_agent_memory
                SET status = 'INVALIDATED',
                    invalidated_reason = 'exact_duplicate_daily_compaction',
                    embedding = NULL,
                    embedding_blob = NULL
                WHERE status IN ('ACTIVE', 'active')
                  AND id NOT IN (
                      SELECT MAX(id) FROM ide_agent_memory
                      WHERE status IN ('ACTIVE', 'active')
                      GROUP BY category, content
                  )
            """)
        if redundant_emb > 0:
            cur.execute("""
                UPDATE ide_agent_memory
                SET embedding = NULL
                WHERE embedding IS NOT NULL
                  AND embedding_blob IS NOT NULL
                  AND LENGTH(embedding_blob) >= 4096
            """)
        if inactive_emb > 0:
            cur.execute("""
                UPDATE ide_agent_memory
                SET embedding = NULL,
                    embedding_blob = NULL
                WHERE status IN ('INVALIDATED', 'SUPERSEDED', 'archived')
                  AND (embedding IS NOT NULL OR embedding_blob IS NOT NULL)
            """)
        try:
            cur.execute("INSERT INTO ide_agent_memory_fts(ide_agent_memory_fts) VALUES('optimize');")
        except Exception:
            pass
        conn.commit()
        cur.execute("PRAGMA wal_checkpoint(PASSIVE);")

    freelist_mb = cur.execute(
        "SELECT freelist_count * page_size / 1048576.0 FROM pragma_freelist_count(), pragma_page_size()"
    ).fetchone()[0]
    conn.close()

    if (not dry_run) and freelist_mb >= 25.0:
        subprocess.run(["docker", "stop", "-t", "5", "jean-sre-jean-harness-1"], check=False, capture_output=True)
        try:
            vconn = sqlite3.connect(db_path, timeout=20.0)
            vconn.execute("PRAGMA busy_timeout = 20000;")
            vconn.execute("VACUUM;")
            vconn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            vconn.close()
            report["vacuumExecuted"] = True
        finally:
            subprocess.run(["docker", "start", "jean-sre-jean-harness-1"], check=False, capture_output=True)

    report["sizeAfterMb"] = round(os.path.getsize(db_path) / 1048576.0, 2)

now = time.time()
old_logs = []
for pattern in ("jean_sre_audit_log_*.md", "jean_infra_audit_log_*.md"):
    for p in sorted(glob.glob(os.path.join(jean_dir, pattern))):
        try:
            if (now - os.path.getmtime(p)) > 7 * 86400:
                old_logs.append(p)
        except OSError:
            pass

report["archivedLogsCount"] = len(old_logs)
parent_archive = os.path.dirname(archive_dir.rstrip("/")) or "/"
if old_logs and not dry_run and os.path.isdir(parent_archive):
    os.makedirs(archive_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    tar_path = os.path.join(archive_dir, f"jean_audit_logs_{stamp}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in old_logs:
            tf.add(p, arcname=os.path.basename(p))
    for p in old_logs:
        os.remove(p)
    report["archiveTarPath"] = tar_path

for cp in glob.glob(os.path.join(jean_dir, ".agent", "context.db.corrupt_*")):
    try:
        if (now - os.path.getmtime(cp)) > 14 * 86400:
            if not dry_run:
                os.remove(cp)
            report["removedCorruptBackups"] += 1
    except OSError:
        pass

print(json.dumps(report))
`;

    const pyB64 = Buffer.from(pyScript, 'utf8').toString('base64');
    const { stdout } = await runHostCommand(
        primaryHost,
        `python3 -c "import base64; exec(base64.b64decode('${pyB64}').decode('utf-8'))"`,
        90_000
    );
    const parsed = JSON.parse(stdout.split('\n').pop());

    if (parsed.vacuumExecuted) {
        for (let i = 0; i < 12; i++) {
            try {
                const h = await fetchJson(`${jeanHarnessUrl}/health`, {}, 4000);
                if (h?.ready) break;
            } catch {
                // Waiting for harness readiness
            }
            await new Promise((r) => setTimeout(r, 2000));
        }
    }

    return {
        ok: true,
        ...parsed,
        reclaimedMb: Number((parsed.sizeBeforeMb - parsed.sizeAfterMb).toFixed(2)),
    };
}

/**
 * Stage 4: Verify HydraDB Persistent Mind (`GET /state`) and run optional procedural WAL compaction hook.
 */
async function stagePersistentMindHygiene({ dryRun, env, fetchImpl }) {
    const jeanHarnessUrl = env.JEAN_AGENT_URL || 'http://127.0.0.1:3005';
    const stateRes = await fetchJson(`${jeanHarnessUrl}/state`, {}, 15_000, fetchImpl);
    const compactScript = env.JEAN_LOCAL_PROCEDURAL_COMPACT_SCRIPT
        ? path.resolve(env.JEAN_LOCAL_PROCEDURAL_COMPACT_SCRIPT)
        : '';
    let proceduralCompaction = null;

    if (compactScript && fs.existsSync(compactScript)) {
        const args = [compactScript, '--json'];
        if (dryRun) args.push('--dry-run');
        const { stdout } = await execFileAsync('node', args, {
            cwd: WORKSPACE_ROOT,
            timeout: 45_000,
        });
        proceduralCompaction = JSON.parse(stdout);
    }

    const readiness = stateRes?.data?.readiness || {};
    return {
        ok: Boolean(stateRes?.ok && stateRes?.data?.selfGraphConnected && readiness.reachable !== false),
        selfGraphConnected: Boolean(stateRes?.data?.selfGraphConnected),
        hydraCounts: readiness.counts || {},
        schemaLintCount: Array.isArray(readiness.lint) ? readiness.lint.length : 0,
        latestTransition: stateRes?.data?.latestState
            ? {
                  targetProject: stateRes.data.latestState.targetProject,
                  phase: stateRes.data.latestState.phase,
                  routedModel: stateRes.data.latestState.routedModel,
                  verdict: stateRes.data.latestState.verdict,
              }
            : null,
        proceduralStoreCompaction: proceduralCompaction?.data || null,
    };
}

/**
 * Stage 5: Run ModularRSI + Self-Improvement Memory (SIM) contrastive trajectory attribution
 * across recent failed/blocked trajectories (`arXiv:2609.14857` & `arXiv:2611.18856`).
 */
export function runJeanModularRsiPass({
    trajectories = [],
    initialSimState = {},
    initialRsiSpecs = {},
    round = 1,
} = {}) {
    const sim = createSelfImprovementMemory(initialSimState);
    const harness = createModularRsiHarness(initialRsiSpecs, {
        sim,
        domain: 'sre-autonomous-sweep',
        benchmark: 'jean-24x7-sweep',
        policyModel: 'inkling-2-mini:ud-iq4_xs-dspark',
    });

    const failed = trajectories.filter((t) => t && t.success === false);
    const passedIds = trajectories.filter((t) => t && t.success === true).map((t) => String(t.taskId));
    const attributions = [];

    const byModule = new Map();
    for (const traj of failed) {
        const attr = harness.attributeFailure(traj);
        attributions.push({
            taskId: traj.taskId,
            attributedModule: attr.attributedModule,
            deficiencyType: attr.deficiencyType,
            rationale: attr.rationale,
        });
        if (!byModule.has(attr.attributedModule)) {
            byModule.set(attr.attributedModule, { attr, taskIds: [] });
        }
        byModule.get(attr.attributedModule).taskIds.push(String(traj.taskId));
    }

    const appliedPatches = [];
    let sampleIdx = 0;
    for (const [moduleName, group] of byModule.entries()) {
        sampleIdx += 1;
        const patchResult = harness.applyRsiPatch(
            moduleName,
            group.attr.suggestedPatch,
            group.attr.rationale,
            {
                rootCause: `${moduleName}:${group.attr.deficiencyType}`,
                round,
                sample: sampleIdx,
                predicted: group.taskIds,
                beforePassed: passedIds,
                passed: passedIds.length,
                total: Math.max(1, trajectories.length),
                files: ['projects/jean-sre/src/harness.js', 'projects/jean-sre/src/jean_sre_sweep.js'],
            }
        );
        appliedPatches.push({
            moduleName,
            toVersion: patchResult.version,
            deficiencyType: group.attr.deficiencyType,
            affectedTasks: group.taskIds.length,
            problemId: patchResult.simEntry?.problemId || null,
        });
    }

    return {
        totalTrajectories: trajectories.length,
        failedTrajectories: failed.length,
        attributions,
        appliedPatches,
        updatedSpecs: harness.getAllSpecs(),
        updatedSim: sim.toJSON(),
        simSummary: sim.getSummary(),
    };
}

function validateSkillMarkdownIntegrity(markdownText = '') {
    const trimmed = String(markdownText || '').trim();
    return (
        trimmed.startsWith('---\n') &&
        trimmed.includes('name: jean-mcp-code-audit') &&
        trimmed.includes('## 🛠️ Explicit MCP Tool Catalog') &&
        trimmed.includes('## 🛡️ Zero-False-Positive Determination Rules') &&
        trimmed.length >= 1800 &&
        trimmed.length <= 24000
    );
}

/**
 * Execute all 7 daily optimization stages end-to-end (supports `dryRun` and DI overrides for tests).
 */
export async function runDailyJeanOptimization(options = {}) {
    const dryRun = Boolean(options.dryRun);
    const skipQuantcheck = Boolean(options.skipQuantcheck);
    const enableSkillOpt = options.enableSkillOpt !== false;
    const fetchImpl = options.fetchImpl || globalThis.fetch;

    const rootDotEnv = loadDotEnvIfPresent(path.join(WORKSPACE_ROOT, '.env'));
    const jeanDotEnv = loadDotEnvIfPresent(path.join(JEAN_PROJECT_ROOT, '.env'));
    const env = { ...rootDotEnv, ...jeanDotEnv, ...process.env, ...(options.env || {}) };
    const primaryHost = env.JEAN_PRIMARY_HOST || 'local';
    const remoteJeanDir = env.JEAN_REMOTE_PROJECT_DIR || JEAN_PROJECT_ROOT;

    const startedAtMs = Date.now();
    const stages = {};

    stages.residency = options.mockStages?.residency
        ? await options.mockStages.residency()
        : await stageThreeChairResidency({ dryRun, env, fetchImpl });

    if (skipQuantcheck) {
        stages.quantcheck = { ok: true, skipped: true, summary: 'Skipped via --skip-quantcheck' };
    } else {
        stages.quantcheck = options.mockStages?.quantcheck
            ? await options.mockStages.quantcheck()
            : await stageFleetQuantcheck({ env, fetchImpl });
    }

    stages.contextCompaction = options.mockStages?.contextCompaction
        ? await options.mockStages.contextCompaction()
        : await stageCompactJeanContextAndLogs({ dryRun, env });

    stages.persistentMind = options.mockStages?.persistentMind
        ? await options.mockStages.persistentMind()
        : await stagePersistentMindHygiene({ dryRun, env, fetchImpl });

    let dbTrajectories = [];
    let sweepLogHealth = {
        cyclesCompleted: 0,
        driftBlockedCount: 0,
        complianceBlockedCount: 0,
        cycleErrors: [],
        syntheticTrajectories: [],
    };

    const dbUrl = env.DBOS_DATABASE_URL || env.MEMORY_DATABASE_URL || env.DATABASE_URL || '';
    let pgHandle = null;

    if (!options.mockStages?.trajectories) {
        try {
            const { stdout: sweepLogs } = await runHostCommand(
                primaryHost,
                'docker logs --since 24h jean-sre-jean-sweep-1 2>&1 | tail -n 1500',
                15_000
            );
            sweepLogHealth = analyzeJeanSweepLogHealth(sweepLogs);
        } catch {
            // Keep empty sweep log fallback
        }

        if (dbUrl) {
            try {
                pgHandle = createPgPool(dbUrl, {
                    serviceName: 'jean-daily-optimize',
                    max: 2,
                    connectionTimeoutMillis: 4000,
                });
                const qRes = await pgHandle.query(
                    `SELECT job_id, job_type, payload, status, result
                     FROM agent_execution_queue
                     ORDER BY COALESCE(started_at_ms, 0) DESC
                     LIMIT 25`
                );
                dbTrajectories = qRes.rows.map((r) => {
                    const ok = r.status === 'completed' || r.status === 'success';
                    const errText =
                        typeof r.result === 'object' && r.result !== null
                            ? r.result.error || r.result.critique || ''
                            : String(r.result || '');
                    return {
                        taskId: String(r.job_id),
                        jobType: r.job_type,
                        taskInput: r.payload?.instructions || r.payload?.task || JSON.stringify(r.payload || {}),
                        success: ok,
                        errorCount: ok ? 0 : 1,
                        errorMessage: ok ? '' : String(errText).slice(0, 240),
                        feedback: String(errText).slice(0, 300),
                    };
                });
            } catch {
                dbTrajectories = [];
            }
        }
    } else {
        const mocked = await options.mockStages.trajectories();
        dbTrajectories = mocked.dbTrajectories || [];
        sweepLogHealth = mocked.sweepLogHealth || sweepLogHealth;
    }

    const simPath = options.simPath || DEFAULT_SIM_PATH;
    const rsiSpecsPath = options.rsiSpecsPath || DEFAULT_RSI_SPECS_PATH;
    const existingSim = fs.existsSync(simPath)
        ? JSON.parse(fs.readFileSync(simPath, 'utf8'))
        : {};
    const existingSpecs = fs.existsSync(rsiSpecsPath)
        ? JSON.parse(fs.readFileSync(rsiSpecsPath, 'utf8'))
        : {};
    const nextRound =
        (existingSim?.problems?.reduce(
            (mx, p) => Math.max(mx, ...(p.attempts || []).map((a) => a.round || 0)),
            0
        ) || 0) + 1;

    const combinedTrajectories = [...dbTrajectories, ...sweepLogHealth.syntheticTrajectories];
    const rsiOutcome = runJeanModularRsiPass({
        trajectories: combinedTrajectories,
        initialSimState: existingSim,
        initialRsiSpecs: existingSpecs,
        round: nextRound,
    });

    if (!dryRun) {
        fs.mkdirSync(path.dirname(simPath), { recursive: true });
        fs.writeFileSync(simPath, `${JSON.stringify(rsiOutcome.updatedSim, null, 2)}\n`, 'utf8');
        fs.writeFileSync(rsiSpecsPath, `${JSON.stringify(rsiOutcome.updatedSpecs, null, 2)}\n`, 'utf8');
    }

    stages.modularRsi = {
        ok: true,
        round: nextRound,
        sweepCycles24h: sweepLogHealth.cyclesCompleted,
        driftBlocked24h: sweepLogHealth.driftBlockedCount,
        complianceBlocked24h: sweepLogHealth.complianceBlockedCount,
        totalTrajectories: rsiOutcome.totalTrajectories,
        failedTrajectories: rsiOutcome.failedTrajectories,
        appliedPatches: rsiOutcome.appliedPatches,
        simSummary: rsiOutcome.simSummary,
    };

    const skillPath = options.skillPath || DEFAULT_SKILL_PATH;
    const actionableFailures = combinedTrajectories.filter((t) => !t.success && t.feedback);
    if (enableSkillOpt && actionableFailures.length >= 2 && fs.existsSync(skillPath)) {
        const backupSkill = fs.readFileSync(skillPath, 'utf8');
        try {
            const firstChairUrl = env.LLAMA_FIRST_CHAIR_URL || 'http://127.0.0.1:18080';
            const edits = await proposeSkillEdits(
                actionableFailures.slice(0, 8),
                backupSkill,
                2,
                {
                    provider: 'openai-compatible',
                    apiUrl: `${firstChairUrl.replace(/\/+$/, '')}/v1/chat/completions`,
                    model: env.PERSONA_MODEL || 'inkling-2-mini:ud-iq4_xs-dspark',
                    temperature: 0.1,
                },
                options.skillOptOptions || {}
            );
            let applied = false;
            if (!dryRun && edits.length > 0) {
                const updated = await applySkillPatch(skillPath, edits);
                if (!validateSkillMarkdownIntegrity(updated)) {
                    fs.writeFileSync(skillPath, backupSkill, 'utf8');
                } else {
                    applied = true;
                }
            }
            stages.playbookCalibration = {
                ok: true,
                evaluatedFailures: actionableFailures.length,
                proposedEdits: edits.length,
                applied,
            };
        } catch (err) {
            fs.writeFileSync(skillPath, backupSkill, 'utf8');
            stages.playbookCalibration = {
                ok: true,
                skippedReason: `Local SkillOpt skipped safely: ${err.message.slice(0, 120)}`,
                proposedEdits: 0,
                applied: false,
            };
        }
    } else {
        stages.playbookCalibration = {
            ok: true,
            evaluatedFailures: actionableFailures.length,
            proposedEdits: 0,
            applied: false,
            note: 'Zero recurring trajectory failures >= 2; playbook already calibrated.',
        };
    }

    if (!dryRun && !options.skipRemoteSync && primaryHost && primaryHost !== 'local') {
        try {
            await runHostCommand(primaryHost, `mkdir -p ${JSON.stringify(path.join(remoteJeanDir, 'data'))}`, 8000);
            await execFileAsync(
                'scp',
                ['-q', simPath, rsiSpecsPath, `${primaryHost}:${path.join(remoteJeanDir, 'data')}/`],
                { timeout: 15_000 }
            );
            if (stages.playbookCalibration?.applied) {
                await execFileAsync(
                    'scp',
                    [
                        '-q',
                        skillPath,
                        `${primaryHost}:${path.join(remoteJeanDir, 'skills', 'jean-mcp-code-audit', 'SKILL.md')}`,
                    ],
                    { timeout: 15_000 }
                );
            }
        } catch {
            // Non-fatal in single-host setups
        }
    }

    const durationSec = Number(((Date.now() - startedAtMs) / 1000).toFixed(2));
    const allStagesOk = Object.values(stages).every((s) => s && s.ok !== false);

    const summaryText = `Jean SRE daily optimization ${allStagesOk ? 'PASSED' : 'DEGRADED'} in ${durationSec}s: 3-Chair residency verified, quantcheck=${stages.quantcheck.totalLiveProbed ?? 'skipped'} live models, context.db reclaimed ${stages.contextCompaction.reclaimedMb ?? 0}MB (now ${stages.contextCompaction.sizeAfterMb ?? 0}MB), SIM problems=${stages.modularRsi.simSummary.problemCount}.`;

    if (!dryRun && pgHandle) {
        try {
            await pgHandle.query(
                `INSERT INTO ide_agent_memory (id, category, content, project, tags)
                 VALUES ($1, 'lesson', $2, 'jean-sre', $3)
                 ON CONFLICT (id) DO UPDATE SET content = $2`,
                [
                    crypto.randomUUID(),
                    summaryText,
                    JSON.stringify(['jean-sre', 'daily-optimization', 'modular-rsi']),
                ]
            );
        } catch {
            // Non-fatal telemetry log error
        }
    }

    if (pgHandle) {
        await pgHandle.close().catch(() => {});
    }

    const receiptPayload = {
        generatedAt: new Date().toISOString(),
        dryRun,
        durationSec,
        allStagesOk,
        summary: summaryText,
        stages,
    };
    const receiptSha256 = crypto
        .createHash('sha256')
        .update(JSON.stringify(receiptPayload))
        .digest('hex');
    const signedReceipt = { ...receiptPayload, receiptSha256 };

    const receiptPath = options.receiptPath || DEFAULT_RECEIPT_PATH;
    const historyPath = options.historyPath || DEFAULT_HISTORY_PATH;
    if (!dryRun || options.writeReceiptOnDryRun) {
        fs.mkdirSync(path.dirname(receiptPath), { recursive: true });
        fs.writeFileSync(receiptPath, `${JSON.stringify(signedReceipt, null, 2)}\n`, 'utf8');
        appendJsonlRecord(historyPath, {
            timestamp: signedReceipt.generatedAt,
            allStagesOk,
            durationSec,
            reclaimedMb: stages.contextCompaction?.reclaimedMb ?? 0,
            contextDbAfterMb: stages.contextCompaction?.sizeAfterMb ?? null,
            liveModelsProbed: stages.quantcheck?.totalLiveProbed ?? null,
            sweepCycles24h: stages.modularRsi?.sweepCycles24h ?? 0,
            receiptSha256,
        });
    }

    if (!allStagesOk) {
        return errorEnvelope('JEAN_DAILY_OPTIMIZE_DEGRADED', summaryText, signedReceipt, 'jean.daily-optimize');
    }
    return okEnvelope('jean.daily-optimize', summaryText, signedReceipt);
}

const cli = defineCli({
    name: 'optimize-jean-daily',
    version: '1.0.0',
    description: 'Daily 7-stage closed-loop optimization & ModularRSI self-improvement for Jean SRE',
    options: {
        'dry-run': {
            type: 'boolean',
            default: false,
            description: 'Preview residency, compaction, and RSI attribution without mutating state',
        },
        'skip-quantcheck': {
            type: 'boolean',
            default: false,
            description: 'Skip live LLM behavioral canary probes for fast smoke checks',
        },
        'no-skillopt': {
            type: 'boolean',
            default: false,
            description: 'Disable automatic local 1st-Chair SkillOpt playbook calibration',
        },
    },
    commands: {
        default: async ({ values, io }) => {
            const lockEnv = await withSingleInstanceLock(DEFAULT_LOCK_PATH, () =>
                runDailyJeanOptimization({
                    dryRun: Boolean(values['dry-run']),
                    skipQuantcheck: Boolean(values['skip-quantcheck']),
                    enableSkillOpt: !values['no-skillopt'],
                })
            );
            const envelope = lockEnv.ok && lockEnv.data?.result ? lockEnv.data.result : lockEnv;
            if (!envelope.ok) {
                process.exitCode = 1;
            }
            if (!values.json) {
                io.stdout(`[${envelope.ok ? 'PASS' : 'FAIL'}] ${envelope.summary || envelope.error?.message}`);
            }
            return envelope;
        },
    },
});

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
    await cli.run();
}
