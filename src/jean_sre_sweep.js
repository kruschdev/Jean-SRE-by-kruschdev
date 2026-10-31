import { randomUUID, createHash } from 'crypto';
import { execSync } from 'child_process';
import fs from 'fs';
import pg from 'pg';
import { loadProjectConfig, envOr } from '@krusch/toolkit/config';
import { resolveModelConfig } from '@krusch/toolkit/llm';
import { scoutHomelabLegoCandidates } from '@krusch/toolkit/lego-scout';
import { auditBoilerplateText } from '@krusch/toolkit/catalog';
import { fileURLToPath } from 'url';
import { dirname, join, extname, basename } from 'path';
import os from 'os';
import { extractSymbolsAndImports } from './ast-chunker.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

// Load dynamic model and service configurations
let config = {};
try {
    config = await loadProjectConfig(projectRoot);
} catch (e) {
    console.error("Warning: Failed to load project config in sweep", e.message);
}

const PERSONA_CONFIG = resolveModelConfig('persona', config);
const EXECUTOR_CONFIG = resolveModelConfig('executor', config);

const THINKER_MODEL = envOr('PERSONA_MODEL', PERSONA_CONFIG.model, 'qwen3-coder-30b-moe:latest');
const WORKER_MODEL = envOr('EXECUTOR_MODEL', EXECUTOR_CONFIG.model, 'gemma4:e4b');

const thinkerNodeName = config.tasks?.persona?.node || 'kruschdev';
const thinkerNode = config.nodes?.[thinkerNodeName]?.name || 'Dual RTX 2080 Ti NVLink';

const workerNodeName = config.tasks?.executor?.node || 'kruschserv';
const workerNode = config.nodes?.[workerNodeName]?.name || 'Dual RTX 3060';

import { 
    HOMELAB_CONTEXT, 
    buildOpenPrompt, 
    buildThinkerPrompt, 
    buildImplementerPrompt, 
    buildInfraThinkerPrompt, 
    buildInfraImplementerPrompt, 
    buildClosePrompt,
    buildCompanionThinkerPrompt,
    buildCompanionImplementerPrompt
} from './prompts.js';
import { parseThinkerPlan } from './parser.js';
import { verifyToolCompliance } from './compliance.js';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Interruptible sleep — resolves early when SIGUSR1 wakes Jean for session reviews
let _wakeResolve = null;
const interruptibleSleep = (ms) => new Promise(resolve => {
    _wakeResolve = resolve;
    const timer = setTimeout(() => { _wakeResolve = null; resolve(); }, ms);
    // Store timer ref so SIGUSR1 can clear it
    interruptibleSleep._timer = timer;
});

// Register SIGUSR1 signal handler for instant wake during long sleeps
process.on('SIGUSR1', () => {
    console.log(`\n[Signal] Received SIGUSR1 — Waking up Jean SRE daemon immediately.`);
    if (interruptibleSleep._timer) {
        clearTimeout(interruptibleSleep._timer);
        interruptibleSleep._timer = null;
    }
    if (_wakeResolve) {
        const resolve = _wakeResolve;
        _wakeResolve = null;
        resolve();
    }
});
const AGENT_URL = envOr('JEAN_AGENT_URL', null, 'http://localhost:3005');
const DBOS_DATABASE_URL = envOr('DBOS_DATABASE_URL');
const DISCORD_WEBHOOK_URL = envOr('DISCORD_WEBHOOK_URL', null, null);
const dbPool = DBOS_DATABASE_URL ? new pg.Pool({
    connectionString: DBOS_DATABASE_URL,
    max: 5,
    idleTimeoutMillis: 30000
}) : null;
if (dbPool) {
    dbPool.on('error', (err) => console.error('[Sweep DB Pool] Idle client error:', err.message));
}

async function sendDiscordAlert(message) {
    if (!DISCORD_WEBHOOK_URL || DISCORD_WEBHOOK_URL === 'your_discord_webhook_url_here') return;
    try {
        await fetch(DISCORD_WEBHOOK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: message })
        });
    } catch (e) {
        console.error(`[Discord] Alert failed: ${e.message}`);
    }
}

const WAIT_MINUTES = parseFloat(envOr('JEAN_SWEEP_INTERVAL_MINUTES', null, '10')); // 10m breather between cycles to prevent local GPU thrashing
const COLAB_BURST_WAIT_MINUTES = parseFloat(envOr('JEAN_COLAB_BURST_INTERVAL_MINUTES', null, '0.05')); // 3s back-to-back cadence when Colab G4 120B is active (datacenter Blackwell requires no thermal cooldown)
const _envHomelabRoot = envOr('HOMELAB_PATH', null, join(os.homedir(), 'homelab'));
const HOMELAB_ROOT = fs.existsSync(_envHomelabRoot) ? _envHomelabRoot : join(projectRoot, '../..');

async function checkColabBurstStatus() {
    try {
        const res = await fetch(`${AGENT_URL}/health`, {
            signal: AbortSignal.timeout(4000)
        });
        if (res.ok) {
            const data = await res.json();
            return {
                active: Boolean(data?.colab_piggyback?.active),
                model: data?.persona_model || THINKER_MODEL,
                endpoint: data?.colab_piggyback?.endpoint || null,
                maxPromptChars: data?.colab_piggyback?.max_prompt_chars || 16000
            };
        }
    } catch (e) {
        console.warn('Failed to fetch colab piggyback status', e);
    }
    return { active: false, model: THINKER_MODEL, endpoint: null, maxPromptChars: 16000 };
}

// ── Fleet Context — injected into Jean's brain for topology awareness ──

// Core Projects & Root Homelab Domains
const CORE_PROJECTS = [
    // Sovereign Core Platforms & Flagship Repositories
    'krusch-git',
    'krusch',                    // Krusch Coding Harness
    'krusch-companion',
    'krusch-context-mcp',
    'krusch-cascade-router',
    'krusch-polygres-connector',
    'krusch-law',
    'krusch-biz',
    'compass-concierge',
    'krusch-nexus',
    'g-krusch',
    'krusch-pre-router',

    // Multi-Node Fleet & Document Memory
    'krusch-fs',

    // Active Production Services, RAG Gateways & MCP Servers (AGENTS.md Registry)
    'krusch-bizlaw',
    'pocket-lawyer',
    'first-things-first',
    'spark',
    'berean',
    'home-ai',
    'krusch-frontend-mcp',
    'jean-sre',
    'krusch-authority-packs',
    'krusch-gateway-mcp',
    'krusch-ci-sentinel',
    'krusch-agentic-proxy',
    'krusch-dev-site',
    'krusch-companion-skylight-connector'
];

// Root-level shared toolkits, orchestration scripts, configs, and governance in ~/homelab/
const HOMELAB_ROOT_TARGETS = [
    '@homelab/lib',              // @krusch/toolkit (shared JS library)
    '@homelab/lib-py',           // krusch_toolkit (shared Python library)
    '@homelab/scripts',          // Fleet orchestration, Colab G4 burst, & maintenance scripts
    '@homelab/.agent',           // Skills, rules, guides, & governance playbooks
    '@homelab/configs'           // Fleet model & routing configurations
];

const ARCHIVED_OR_IGNORED_PROJECT_DIRS = new Set(['archive', '_archive', 'krusch-pg-agent']);

// Archived projects intentionally retained in Jean SRE rotation so bug fixes continue generating SFT training data
const ARCHIVED_ROTATION_PROJECTS = ['krusch-dbos-mcp'];

function discoverUncuratedProjects() {
    const coreSet = new Set(CORE_PROJECTS);
    const projectsRoot = join(HOMELAB_ROOT, 'projects');
    if (!fs.existsSync(projectsRoot)) return [];
    try {
        const entries = fs.readdirSync(projectsRoot, { withFileTypes: true });
        const uncurated = [];
        for (const d of entries) {
            if (!d.isDirectory() || coreSet.has(d.name) || d.name.startsWith('.') || ARCHIVED_OR_IGNORED_PROJECT_DIRS.has(d.name)) continue;
            const dirPath = join(projectsRoot, d.name);
            let files = [];
            try { files = fs.readdirSync(dirPath); } catch { continue; }
            const hasCode = files.some(f => [
                'package.json', 'pyproject.toml', 'requirements.txt',
                'Cargo.toml', 'go.mod', 'src', 'server.js', 'index.js', 'Dockerfile'
            ].includes(f));
            if (hasCode) uncurated.push(d.name);
        }
        for (const archivedName of ARCHIVED_ROTATION_PROJECTS) {
            if (!coreSet.has(archivedName) && !uncurated.includes(archivedName)) {
                const archivedPath = join(projectsRoot, 'archive', archivedName);
                if (fs.existsSync(archivedPath)) {
                    uncurated.push(archivedName);
                }
            }
        }
        return uncurated.sort();
    } catch {
        return [];
    }
}

const PROJECT_ROTATION = [
    ...CORE_PROJECTS,
    ...HOMELAB_ROOT_TARGETS,
    ...discoverUncuratedProjects()
];

export function resolveTargetDir(targetProject) {
    if (!targetProject) return HOMELAB_ROOT;
    if (targetProject === '@homelab/root') return HOMELAB_ROOT;
    if (targetProject.startsWith('@homelab/')) {
        return join(HOMELAB_ROOT, targetProject.slice('@homelab/'.length));
    }
    const primary = join(HOMELAB_ROOT, 'projects', targetProject);
    if (!fs.existsSync(primary)) {
        const archived = join(HOMELAB_ROOT, 'projects', 'archive', targetProject);
        if (fs.existsSync(archived)) return archived;
    }
    return primary;
}

// ── Phased Audit State Machine ───────────────────────────
// Each project progresses through 11 phases before rotating.
const AUDIT_PHASES = [
    '0_ORIENTATION',
    '1.1_STANDARD_MAPPING',
    '1.2_COMPLIANCE_MATRIX',
    '1.3_CRITICAL_PATH',
    '1.4_BOUNDARY_AUDIT',
    '1.5_ANTI_PATTERN',
    '1.6_STRUCTURAL_STRESS_TEST',
    '1.7_VERIFICATION_AUDIT',
    '1.8_RISK_MAP',
    '1.9_ADVERSARIAL_REVIEW',
    '2_MASTER_BACKLOG'
];

// ── Infrastructure Sweep Targets (runs every INFRA_INTERVAL project rotations) ──
const INFRA_INTERVAL = 5; // Trigger infra sweep every 5th project completion
const INFRA_TARGETS = [
    { node: 'krusch', focus: 'homelab_hygiene' },     // Stale folders, dead prototypes, duplicate skills, orphaned DBs
    { node: 'kruschdev', focus: 'homelab_hygiene' },  // Stale folders, unreferenced projects, & monorepo drift
    { node: 'kruschdev', focus: 'lego_scout' },       // Cross-repo @krusch/toolkit adoption & Rule-of-3 candidate scouting
    { node: 'kruschserv', focus: 'docker' },          // Docker images, volumes, compose configs
    { node: 'kruschserv', focus: 'disk' },            // Disk usage, large files, old backups
    { node: 'kruschserv', focus: 'system_tuning' },   // Kernel, mounts, Docker daemon, network
    { node: 'kruschdev', focus: 'docker' },           // Dev container sprawl, unused images
    { node: 'kruschdev', focus: 'disk' },             // Data mounts & Ollama disk footprint
    { node: 'kruschdev', focus: 'system_tuning' },    // NUMA, CPU governor, GPU power
    { node: 'krusch', focus: 'docker' },              // Orchestrator container & image health
    { node: 'krusch', focus: 'disk' },                // Orchestrator NVMe usage & temp files
    { node: 'krusch', focus: 'system_tuning' },       // Orchestrator kernel, GPU & network tuning
    { node: 'kruschserv', focus: 'services' },        // systemd, cron, enabled but unused
    { node: 'kruschdev', focus: 'services' },         // GPU optimization, model loading
    { node: 'krusch', focus: 'services' },            // Orchestrator systemd & GPU services
];

let projectRotationIndex = 0;
let currentPhaseIndex = 0;
let infraTargetIndex = 0;
let projectsCompletedSinceInfra = 0; // Counter for triggering infra sweeps
let isInfraMode = false; // true when running an infra sweep cycle
let sessionHandoffsReviewed = 0;
let sessionHandoffsFailed = 0;
let convergedProjects = {};
const ROTATION_STATE_FILE = 'JEAN_SRE_ROTATION.json';

if (fs.existsSync(ROTATION_STATE_FILE)) {
    try {
        const state = JSON.parse(fs.readFileSync(ROTATION_STATE_FILE, 'utf8'));
        if (state.currentProjectName && PROJECT_ROTATION.includes(state.currentProjectName)) {
            projectRotationIndex = PROJECT_ROTATION.indexOf(state.currentProjectName);
        } else {
            projectRotationIndex = state.projectRotationIndex || 0;
        }
        currentPhaseIndex = state.currentPhaseIndex || 0;
        infraTargetIndex = state.infraTargetIndex || 0;
        projectsCompletedSinceInfra = state.projectsCompletedSinceInfra || 0;
        sessionHandoffsReviewed = state.sessionHandoffsReviewed || 0;
        sessionHandoffsFailed = state.sessionHandoffsFailed || 0;
        if (state.convergedProjects && typeof state.convergedProjects === 'object') {
            convergedProjects = state.convergedProjects;
        }
    } catch (e) {
        console.error(`[State] Failed to read ${ROTATION_STATE_FILE}:`, e.message);
    }
}

function saveRotationState() {
    fs.writeFileSync(ROTATION_STATE_FILE, JSON.stringify({
        projectRotationIndex,
        currentProjectName: PROJECT_ROTATION[projectRotationIndex % PROJECT_ROTATION.length],
        currentPhaseIndex,
        infraTargetIndex, projectsCompletedSinceInfra,
        sessionHandoffsReviewed, sessionHandoffsFailed,
        convergedProjects
    }));
}

/**
 * Compute a deterministic SHA-256 signature of a project's source files so
 * converged repositories are skipped until their code actually changes.
 */
const SOURCE_SIGNATURE_EXTS = new Set(['.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.py']);

function computeProjectSourceSignature(projectPath) {
    if (!projectPath || !fs.existsSync(projectPath)) return '';
    try {
        const files = listProjectFiles(projectPath, projectPath, 0, 5, 200, [], SOURCE_SIGNATURE_EXTS).slice().sort();
        if (files.length === 0) return '';
        const hash = createHash('sha256');
        for (const rel of files) {
            const abs = join(projectPath, rel);
            try {
                const content = fs.readFileSync(abs);
                hash.update(`${rel}:${content.length}:`);
                hash.update(content);
                hash.update('\n');
            } catch {}
        }
        return hash.digest('hex').slice(0, 24);
    } catch {
        return '';
    }
}

function markProjectConverged(targetProject, projectPath = null, status = 'CLEAN_PASS') {
    const dir = projectPath || resolveTargetDir(targetProject);
    const signature = computeProjectSourceSignature(dir);
    if (!signature) return null;
    const prev = convergedProjects[targetProject];
    const cleanPasses = (prev && prev.signature === signature ? (prev.cleanPasses || 0) : 0) + 1;
    convergedProjects[targetProject] = {
        signature,
        status,
        cleanPasses,
        convergedAt: new Date().toISOString()
    };
    saveRotationState();
    return convergedProjects[targetProject];
}

function isProjectConverged(targetProject, projectPath = null) {
    const entry = convergedProjects[targetProject];
    if (!entry || !entry.signature) return false;
    const dir = projectPath || resolveTargetDir(targetProject);
    const currentSig = computeProjectSourceSignature(dir);
    return Boolean(currentSig && currentSig === entry.signature);
}

/** Get the current project (stays until all phases complete). */
function getCurrentProject() {
    return PROJECT_ROTATION[projectRotationIndex % PROJECT_ROTATION.length];
}

/** Get current audit phase name. */
function getCurrentPhase() {
    return AUDIT_PHASES[currentPhaseIndex];
}

/** Advance to next phase. If all phases done, rotate to next project (or trigger infra sweep). */
function advanceAuditPhase() {
    // If we just finished an infra sweep, return to project rotation
    if (isInfraMode) {
        isInfraMode = false;
        infraTargetIndex++;
        const nextProject = getCurrentProject();
        console.log(`[Infra] ✅ Infra sweep complete. Resuming project rotation: ${nextProject}`);
        saveRotationState();
        return { rotated: false, project: nextProject, phase: getCurrentPhase(), infraDone: true };
    }

    currentPhaseIndex++;
    if (currentPhaseIndex >= AUDIT_PHASES.length) {
        currentPhaseIndex = 0;
        projectRotationIndex++;
        projectsCompletedSinceInfra++;

        // Skip projects that already converged at their current source signature
        let skippedConverged = 0;
        while (skippedConverged < PROJECT_ROTATION.length) {
            const candidate = getCurrentProject();
            if (!isProjectConverged(candidate)) break;
            const sig = convergedProjects[candidate]?.signature || '';
            console.log(`[Convergence] ⏭️ Skipping converged project ${candidate} (source signature unchanged: ${sig})`);
            projectRotationIndex++;
            skippedConverged++;
        }

        // Check if it's time for an infra sweep
        if (projectsCompletedSinceInfra >= INFRA_INTERVAL) {
            projectsCompletedSinceInfra = 0;
            isInfraMode = true;
            const target = INFRA_TARGETS[infraTargetIndex % INFRA_TARGETS.length];
            console.log(`[Infra] 🔧 Triggering infra sweep: ${target.node}/${target.focus} (after ${INFRA_INTERVAL} projects)`);
            saveRotationState();
            return { rotated: true, project: `INFRA:${target.node}/${target.focus}`, phase: 'INFRA_SWEEP', infra: target };
        }

        const nextProject = getCurrentProject();
        console.log(`[Audit] ✅ All phases complete. Rotating to next project: ${nextProject}`);
        saveRotationState();
        return { rotated: true, project: nextProject, phase: AUDIT_PHASES[0] };
    }
    const phase = getCurrentPhase();
    console.log(`[Audit] → Advancing to phase: ${phase} (${currentPhaseIndex + 1}/${AUDIT_PHASES.length})`);
    saveRotationState();
    return { rotated: false, project: getCurrentProject(), phase };
}

/** Get the current infra target if in infra mode. */
function getCurrentInfraTarget() {
    return INFRA_TARGETS[infraTargetIndex % INFRA_TARGETS.length];
}

// Legacy compat — some code still calls getNextProject()
function getNextProject() {
    return getCurrentProject();
}

// ─────────────────────────────────────────────────────────
// Direct MCP Tool Calls (Ralph Loop Pattern)
// Deterministic context injection — no LLM dependency.
// ─────────────────────────────────────────────────────────

/**
 * Call an MCP tool directly via the harness /tool endpoint.
 */
async function callTool(name, args = {}, options = {}) {
    const maxRetries = options.retries || 0;
    const timeoutMs = options.timeout || 120000; // default 2m
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const res = await fetch(`${AGENT_URL}/tool`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(process.env.JEAN_API_KEY ? { 'x-api-key': process.env.JEAN_API_KEY } : {})
                },
                body: JSON.stringify({ name, args }),
                signal: controller.signal
            });
            clearTimeout(timeout);
            const data = await res.json();
            if (!res.ok || data.error) {
                return `[Failed: ${data.error || `HTTP ${res.status}`}]`;
            }
            return data.result || '';
        } catch (e) {
            clearTimeout(timeout);
            console.error(`[Tool] ${name} failed (attempt ${attempt + 1}): ${e.message}`);
            if (attempt < maxRetries) {
                await sleep(5000);
                continue;
            }
            return `[Failed: ${e.message}]`;
        }
    }
}

// ─────────────────────────────────────────────────────────
// Jean's /open — Programmatic context bootstrap.
// All tools called deterministically via /tool endpoint.
// ─────────────────────────────────────────────────────────

async function fetchIntentResults(inflightBlock) {
    if (!inflightBlock) return 'No previous intents to poll.';
    const jobIds = [...new Set(inflightBlock.match(/\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b[0-9a-f]{8}\b/g) || [])];
    if (jobIds.length === 0) return 'No previous intents to poll.';
    if (!dbPool) return '[Database poll failed: DBOS_DATABASE_URL not configured]';

    let results = [];
    try {
        const fullIds = jobIds.filter(id => id.length !== 8);
        const prefixIds = jobIds.filter(id => id.length === 8);

        let queryParts = [];
        const params = [];
        let paramIdx = 1;

        if (fullIds.length > 0) {
            queryParts.push(`job_id::text = ANY($${paramIdx})`);
            params.push(fullIds);
            paramIdx++;
        }

        for (const pid of prefixIds) {
            queryParts.push(`job_id::text LIKE $${paramIdx}`);
            params.push(`${pid}%`);
            paramIdx++;
        }

        if (queryParts.length > 0) {
            const queryStr = `SELECT job_id, status, result FROM agent_execution_queue WHERE ${queryParts.join(' OR ')}`;
            const res = await dbPool.query(queryStr, params);
            results = res.rows;
        }
    } catch (e) {
        console.error(`[Poll] Failed:`, e.message);
        return `[Database poll failed: ${e.message}]`;
    }

    if (results.length === 0) return 'Jobs not found in queue.';

    return results.map(r => `Job ${r.job_id.slice(0, 8)}: ${r.status}${r.result ? `\nResult: ${JSON.stringify(r.result).slice(0, 500)}` : ''}`).join('\n\n');
}

/**
 * Fetch recent completed Jean SRE intent commands from the DBOS queue for drift detection.
 * Filters strictly to completed jean_sre_investigation jobs from the last 2 hours.
 */
async function fetchRecentIntentInstructions(limit = 15) {
    if (!dbPool) return [];
    try {
        const cutoffMs = Date.now() - (2 * 60 * 60 * 1000);
        const res = await dbPool.query(
            `SELECT COALESCE(payload->>'parsed_command', payload->>'instructions') as cmd
             FROM agent_execution_queue
             WHERE job_type = 'intent'
               AND status = 'completed'
               AND payload->>'task' = 'jean_sre_investigation'
               AND started_at_ms > $2
             ORDER BY started_at_ms DESC
             LIMIT $1`,
            [limit, cutoffMs]
        );
        return res.rows
            .map(r => String(r.cmd || '').trim())
            .filter(s => s.length > 5);
    } catch (e) {
        console.error(`[Drift] Failed to fetch recent intents:`, e.message);
        return [];
    }
}

/**
 * Query swarm stats directly from the DB, broken down by job_type.
 * Returns structured counts so Jean sees intent vs debate activity separately.
 * Direct DB query avoids MCP tool discovery caching issues.
 */
async function fetchJeanSwarmStats() {
    if (!dbPool) return '{}';
    try {
        const res = await dbPool.query(`
            SELECT
                COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'pending') as intent_pending,
                COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'running') as intent_running,
                COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'completed') as intent_completed,
                COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'failed') as intent_failed,
                COUNT(*) FILTER (WHERE job_type IN ('debate_proposal', 'debate_rebuttal') AND status = 'running') as debate_running,
                COUNT(*) FILTER (WHERE job_type IN ('debate_proposal', 'debate_rebuttal') AND status = 'pending') as debate_pending,
                COUNT(*) FILTER (WHERE job_type NOT IN ('intent', 'debate_proposal', 'debate_rebuttal') AND status = 'pending') as other_pending,
                COUNT(*) as total_jobs,
                COUNT(*) FILTER (WHERE status = 'completed') as total_completed,
                COUNT(*) FILTER (WHERE status = 'failed') as total_failed
            FROM agent_execution_queue
        `);
        return JSON.stringify(res.rows[0] || {});
    } catch (e) {
        console.error(`[SwarmStats] Direct DB query failed:`, e.message);
        return '{}';
    }
}

/**
 * Check if a proposed intent's executable command duplicates a recently completed command.
 */
async function checkIntentDrift(intentTask, recentInstructions, projectPath = HOMELAB_ROOT) {
    if (!recentInstructions || recentInstructions.length === 0) {
        return { isDrift: false, reason: 'No prior completed intents in window.' };
    }

    const parsed = parseExecutableAction(intentTask, projectPath);
    const candidateCmd = (parsed.command || intentTask).trim().toLowerCase();

    const tokenize = (s) => new Set(
        s.toLowerCase()
            .replace(/[`'"]/g, '')
            .split(/[\s;|&]+/)
            .filter(tok => tok.length > 1)
    );
    const candTokens = tokenize(candidateCmd);

    for (const prev of recentInstructions) {
        const prevParsed = parseExecutableAction(prev, projectPath);
        const prevCmd = (prevParsed.command || prev).trim().toLowerCase();

        if (candidateCmd === prevCmd) {
            return { isDrift: true, reason: `Exact command already executed recently: ${prevCmd.slice(0, 80)}` };
        }

        const prevTokens = tokenize(prevCmd);
        if (candTokens.size >= 2 && prevTokens.size >= 2) {
            let intersection = 0;
            for (const t of candTokens) {
                if (prevTokens.has(t)) intersection++;
            }
            const union = new Set([...candTokens, ...prevTokens]).size;
            const jaccard = union > 0 ? intersection / union : 0;
            if (jaccard >= 0.88) {
                return { isDrift: true, reason: `Redundant command (${(jaccard * 100).toFixed(0)}% overlap with recent execution)` };
            }
        }
    }

    return { isDrift: false, reason: 'Novel investigation command.' };
}

async function loadContext(inflightBlock = '', targetProject = null) {
    const activeProject = (targetProject || getCurrentProject() || 'jean-sre').split('/')[0];
    console.log(`[Open] Loading context (Tier 1 hydration for ${activeProject} + fleet telemetry)...`);

    const results = await Promise.allSettled([
        callTool('krusch-context-krusch-context-retrieve', {
            query: '*',
            project: activeProject,
            include_state: true,
            limit_tokens: 1500
        }, { retries: 1, timeout: 10000 }),
        callTool('krusch-context-krusch-context-list-memories', {
            category: 'blocker',
            project: 'jean-sre',
            limit: 15
        }, { retries: 1, timeout: 10000 }),
        callTool('krusch-context-krusch-context-list-memories', {
            category: 'lesson',
            project: 'jean-sre',
            limit: 5
        }, { retries: 1, timeout: 10000 }),
        callTool('krusch-context-krusch-context-nugget-list', {
            kinds: ['project'],
            active_project: activeProject
        }, { retries: 1, timeout: 10000 }),
        callTool('sentinel-get-system-stats', { node: 'kruschserv' }),
        callTool('sentinel-get-docker-containers', { node: 'kruschserv' }),
        fetchJeanSwarmStats(),
        fetchIntentResults(inflightBlock)
    ]);

    const [retrievedInvariants, projMemories, jeanLessons, nudges, sysStats, containers, swarmStats, intentResults] = results.map(r =>
        r.status === 'fulfilled' ? r.value : `[Failed: ${r.reason}]`
    );

    const formatMemoryRows = (raw, label, projectFilter = null) => {
        if (!raw || String(raw).startsWith('[Failed')) return '';
        const rawStr = String(raw).trim();
        if (rawStr.startsWith('# 🧠 Compiled Project State') || rawStr.startsWith('# 🔍 Unified Context Retrieval')) {
            const annotated = rawStr
                .replace(/##\s*🐛\s*Known Bugs & Blockers/gi, '## 🐛 Historical Bug Fixes & Active Blockers')
                .replace(
                    /(?:\*\*#(\d+)\*\*\s*\[bug\]|\*\*\[#(\d+)\s*\|\s*bug\]\*\*):\s*((?:\[[^\]]+\]\s*)?(?:Fixed|Resolved|Hardened|Remediated|Added|Updated)\b)/gi,
                    (_, id1, id2, verb) => `**#${id1 || id2}** [historical_fixed_bug — ALREADY RESOLVED, do NOT copy into Pending Issues]: ${verb}`
                );
            return `#### ${label}\n${annotated.slice(0, 1800)}`;
        }
        try {
            const parsed = JSON.parse(rawStr);
            let arr = parsed.results || parsed.memories || (Array.isArray(parsed) ? parsed : []);
            arr = arr.filter(l => !l.superseded_by && l.status !== 'invalidated');
            if (projectFilter) {
                const tag = `[${projectFilter}]`.toLowerCase();
                arr = arr.filter(l => String(l.content || '').toLowerCase().includes(tag) || String(l.project || '').toLowerCase() === projectFilter.toLowerCase());
            }
            if (arr.length > 0) {
                return `#### ${label}\n` + arr
                    .map((l, i) => {
                        const isResolvedBug = l.category === 'bug' && /^(?:\[[^\]]+\]\s*)?(?:fixed|resolved|hardened|remediated|added|updated)\b/i.test(String(l.content || '').trim());
                        const catLabel = isResolvedBug ? 'historical_fixed_bug — ALREADY RESOLVED' : (l.category || 'memory');
                        return `${i + 1}. [${catLabel} | ${l.created_at || 'recent'}] ${l.content}`;
                    })
                    .join('\n');
            }
            return '';
        } catch {
            return typeof raw === 'string' && !raw.includes('No memories') && (!projectFilter || raw.toLowerCase().includes(`[${projectFilter.toLowerCase()}]`))
                ? `#### ${label}\n${raw.slice(0, 1200)}`
                : '';
        }
    };

    const lessonsFormatted = [
        formatMemoryRows(retrievedInvariants, `Tier 1 Hydrated Invariants & Context (${activeProject})`, activeProject),
        formatMemoryRows(projMemories, `Active Project Blockers (${activeProject})`, activeProject),
        formatMemoryRows(jeanLessons, `Jean SRE Fleet Lessons`)
    ].filter(Boolean).join('\n\n') || 'No previous lessons or invariants found.';

    // ── Format system stats into strict table ──
    let sysTable = 'Unavailable.';
    try {
        const s = typeof sysStats === 'string' ? JSON.parse(sysStats) : sysStats;
        if (s.cpuCount) {
            const loadPer = ((s.loadavg?.[0] || 0) / s.cpuCount * 100).toFixed(0);
            sysTable = `| Metric | Value |
|--------|-------|
| CPU | ${s.cpuModel} (${s.cpuCount} threads) |
| Load (1/5/15m) | ${s.loadavg?.join(' / ') || 'N/A'} |
| Load % (1m) | ${loadPer}% of ${s.cpuCount} cores |
| RAM Total | ${s.totalMemGB} GB |
| RAM Free | ${s.freeMemGB} GB |
| RAM Used | ${s.usedMemPercent} |
| Uptime | ${s.uptimeHours} hours |

These are the ONLY system metrics. Do not invent additional metrics.`;
        }
    } catch { /* fallback to raw */ }

    // ── Format containers into strict table ──
    let containerTable = 'Unavailable.';
    try {
        const c = typeof containers === 'string' ? JSON.parse(containers) : containers;
        if (Array.isArray(c) && c.length > 0) {
            const rows = c.map(ct => {
                const name = (ct.Names || '').replace(/^\//, '');
                const health = ct.HealthStatus || 'none';
                const status = ct.Status || ct.State || 'unknown';
                const image = (ct.Image || '').split(':')[0].split('/').pop();
                return `| ${name} | ${ct.State} | ${health} | ${status} | ${image} |`;
            });
containerTable = `| Container | State | Health | Status | Image |
|-----------|-------|--------|--------|-------|
${rows.join('\n')}

These are ALL containers. There are exactly ${c.length} containers. Do not invent others.
Note: A health status of "none" simply means the container image doesn't define a Docker HEALTHCHECK. It is normal and NOT a failure.`;
        }
    } catch { /* fallback to raw */ }

    // ── Format swarm stats into strict KV (Jean-specific breakdown) ──
    let swarmTable = 'Unavailable.';
    try {
        const sw = typeof swarmStats === 'string' ? JSON.parse(swarmStats) : swarmStats;
        if (sw.total_jobs !== undefined) {
            swarmTable = `### Jean's Intent Jobs (dispatched by this pipeline)
| Status | Count |
|--------|-------|
| Pending | ${sw.intent_pending ?? 0} |
| Running | ${sw.intent_running ?? 0} |
| Completed (all-time) | ${sw.intent_completed ?? 0} |
| Failed (all-time) | ${sw.intent_failed ?? 0} |

### Swarm Debate Activity (NORMAL background work — NOT a problem)
| Status | Count |
|--------|-------|
| Running | ${sw.debate_running ?? 0} |
| Pending | ${sw.debate_pending ?? 0} |

### Queue Totals
| Metric | Count |
|--------|-------|
| Total Jobs (all time) | ${sw.total_jobs} |
| Total Completed | ${sw.total_completed ?? 0} |
| Total Failed | ${sw.total_failed ?? 0} |
| Other Pending | ${sw.other_pending ?? 0} |

IMPORTANT: Running debate jobs are NORMAL swarm activity — do NOT investigate them as a backlog problem.
Only investigate if YOUR intent jobs are stuck in 'pending' or 'failed'.`;
        }
    } catch { /* fallback to raw */ }

    console.log(`[Open] Context loaded: lessons=${lessonsFormatted.length}c, nudges=${String(nudges).length}c, stats=${sysTable.length}c, containers=${containerTable.length}c`);
    return { lessons: lessonsFormatted, nudges, sysStats: sysTable, containers: containerTable, swarmStats: swarmTable, intentResults };
}



// ─────────────────────────────────────────────────────────
// Jean's /close — Programmatic state persistence.
// Writes directly to ide_agent_memory via bge-large.
// ─────────────────────────────────────────────────────────

async function saveContext(sessionSummary) {
    console.log(`[Close] Persisting session state to kruschdb...`);

    if (sessionSummary) {
        const result = await callTool('krusch_context_remember', {
            category: 'lesson',
            content: sessionSummary,
            project: 'jean-sre',
            force: true
        });
        console.log(`[Close] Activity written: ${result.slice(0, 200)}`);
    }

    // Also save to nuggets for cross-system visibility
    try {
        await callTool('krusch_context_remember', {
            key: 'jean-last-session',
            content: sessionSummary?.slice(0, 200) || 'Session ended.',
            project: 'jean-sre'
        });
    } catch (e) {
        console.error(`[Close] Nuggets save failed (non-fatal): ${e.message}`);
    }
}

/** Save a lesson extracted from Jean's reply to kruschdb. */
async function saveLesson(lessonContent, targetProject = 'jean-sre') {
    const cleanProject = String(targetProject || 'jean-sre').split('/')[0];
    const prefixedContent = cleanProject !== 'jean-sre' ? `[${cleanProject}] ${lessonContent}` : lessonContent;
    const result = await callTool('krusch_context_remember', {
        category: 'lesson',
        content: prefixedContent,
        project: 'jean-sre',
        force: true
    });
    if (cleanProject && cleanProject !== 'jean-sre') {
        await callTool('krusch_context_remember', {
            category: 'lesson',
            content: prefixedContent,
            project: cleanProject,
            force: true
        }).catch(() => {});
    }
    console.log(`[Lessons] Saved to kruschdb (${cleanProject}): ${result.slice(0, 150)}`);
}

const AUDIT_BLOCKER_TAG_RE = /^\[([A-Z]{2,8}-\d{1,3})\s*\|\s*(P[0-3])\]\s*(.+)$/;

function formatAuditBlockerSummary(card) {
    return `[${card.id} | ${card.severity}] ${card.title} (${card.evidence})${card.fixSummary ? ` — ${card.fixSummary}` : ''}`;
}

function normalizeAuditBlockerBody(content, targetProject = 'jean-sre') {
    const cleanProject = String(targetProject || 'jean-sre').split('/')[0];
    let body = String(content || '').trim();
    const prefix = `[${cleanProject}]`;
    if (body.toLowerCase().startsWith(prefix.toLowerCase())) {
        body = body.slice(prefix.length).trim();
    }
    return body.replace(/\s+/g, ' ');
}

function isOwnedAuditBlocker(rawContent, cleanProject, storeName) {
    const trimmed = String(rawContent || '').trim();
    const prefix = `[${cleanProject}]`;
    if (cleanProject === 'jean-sre') {
        const rest = trimmed.toLowerCase().startsWith('[jean-sre]')
            ? trimmed.slice('[jean-sre]'.length).trim()
            : trimmed;
        return AUDIT_BLOCKER_TAG_RE.test(rest);
    }
    if (storeName === 'jean-sre') {
        if (!trimmed.toLowerCase().startsWith(prefix.toLowerCase())) return false;
        const rest = trimmed.slice(prefix.length).trim();
        return AUDIT_BLOCKER_TAG_RE.test(rest);
    }
    if (trimmed.toLowerCase().startsWith(prefix.toLowerCase())) {
        const rest = trimmed.slice(prefix.length).trim();
        return AUDIT_BLOCKER_TAG_RE.test(rest);
    }
    return AUDIT_BLOCKER_TAG_RE.test(trimmed);
}

/**
 * Reconcile prior auto-generated `[ID | P0-P3]` audit blockers for `targetProject`
 * in both the project's local `.agent/context.db` and the central `jean-sre` store.
 * Preserves human-authored blockers and other projects' blockers.
 */
async function reconcileProjectBlockers(targetProject, activeBugCards = [], callToolFn = callTool) {
    const cleanProject = String(targetProject || 'jean-sre').split('/')[0];
    const desiredBodies = new Set(
        (activeBugCards || []).map(card => normalizeAuditBlockerBody(formatAuditBlockerSummary(card), cleanProject))
    );
    const stores = cleanProject && cleanProject !== 'jean-sre'
        ? [cleanProject, 'jean-sre']
        : ['jean-sre'];
    const activeByStore = new Map(stores.map(s => [s, new Set()]));
    let invalidatedCount = 0;

    for (const storeName of stores) {
        let rawList = '';
        try {
            rawList = await callToolFn('krusch-context-krusch-context-list-memories', {
                category: 'blocker',
                project: storeName,
                limit: storeName === 'jean-sre' ? 1000 : 200
            }, { retries: 1, timeout: 10000 });
        } catch {
            continue;
        }
        const text = String(rawList || '');
        const lineRe = /^\*\s*\*\*#(\d+)\*\*\s*\[blocker\]\s*\(ACTIVE\):\s*(.+)$/gm;
        let m;
        while ((m = lineRe.exec(text)) !== null) {
            const id = Number(m[1]);
            const content = m[2].trim();
            if (!Number.isFinite(id) || !isOwnedAuditBlocker(content, cleanProject, storeName)) continue;

            const normBody = normalizeAuditBlockerBody(content, cleanProject);
            const storeActive = activeByStore.get(storeName);
            if (desiredBodies.has(normBody) && !storeActive.has(normBody)) {
                storeActive.add(normBody);
                continue;
            }

            const reason = desiredBodies.size === 0
                ? `Auto-invalidated by Jean SRE 2_MASTER_BACKLOG: ${targetProject} achieved CLEAN_PASS (zero verified actionable P0/P1 blockers).`
                : (desiredBodies.has(normBody)
                    ? `Auto-invalidated duplicate active audit blocker for ${targetProject} during 2_MASTER_BACKLOG reconciliation.`
                    : `Auto-invalidated superseded/resolved audit blocker for ${targetProject}; not present in latest verified 2_MASTER_BACKLOG.`);
            try {
                await callToolFn('krusch_context_revise', {
                    action: 'invalidate',
                    target_id: id,
                    project: storeName,
                    reason
                }, { retries: 1, timeout: 10000 });
                invalidatedCount++;
            } catch (e) {
                console.warn(`[Blockers] Failed to invalidate stale blocker #${id} in ${storeName}:`, e.message);
            }
        }
    }

    if (invalidatedCount > 0) {
        console.log(`[Blockers] 🧹 Reconciled & invalidated ${invalidatedCount} stale/duplicate audit blocker(s) for ${cleanProject}.`);
    }
    return { invalidatedCount, activeByStore };
}

/** Save a P0/P1 blocker discovered during Master Backlog synthesis to kruschdb (Tier 1 /code-audit step). */
async function saveProjectBlocker(blockerContent, targetProject, { skipStores = new Set(), callToolFn = callTool } = {}) {
    const cleanProject = String(targetProject || 'jean-sre').split('/')[0];
    const prefixedContent = cleanProject !== 'jean-sre' ? `[${cleanProject}] ${blockerContent}` : blockerContent;
    let result = '[Skipped — already active]';
    if (!skipStores.has('jean-sre')) {
        result = await callToolFn('krusch_context_remember', {
            category: 'blocker',
            content: prefixedContent,
            project: 'jean-sre',
            force: true
        });
    }
    if (cleanProject && cleanProject !== 'jean-sre' && !skipStores.has(cleanProject)) {
        await callToolFn('krusch_context_remember', {
            category: 'blocker',
            content: prefixedContent,
            project: cleanProject,
            force: true
        }).catch(() => {});
    }
    console.log(`[Blockers] Saved P0/P1 blocker to kruschdb (${cleanProject}): ${String(result).slice(0, 150)}`);
}

// ─────────────────────────────────────────────────────────
// Prompt Builders
// ─────────────────────────────────────────────────────────


// ── DBOS Swarm Intent Dispatcher & Inline L2 Executor ────

function normalizeHomelabPath(str) {
    if (!str) return '';
    return String(str)
        .replace(/\/home\/kruschdev\/homelab/g, HOMELAB_ROOT)
        .replace(/\/home\/agent\/homelab/g, HOMELAB_ROOT)
        .replace(/\/home\/krusch\/homelab/g, HOMELAB_ROOT);
}

function parseLooseJsonObject(rawObjStr, fallbackNode = 'kruschserv') {
    if (!rawObjStr) return { node: fallbackNode };
    try {
        return JSON.parse(rawObjStr);
    } catch {
        try {
            const jsonNormalized = rawObjStr
                .replace(/([{,]\s*)([a-zA-Z0-9_]+)\s*:/g, '$1"$2":')
                .replace(/'([^']*)'/g, '"$1"');
            return JSON.parse(jsonNormalized);
        } catch {
            const nodeMatch = rawObjStr.match(/node\s*:\s*['"]?([a-zA-Z0-9_-]+)['"]?/i);
            return { node: nodeMatch ? nodeMatch[1] : fallbackNode };
        }
    }
}

function buildProjectAstIndex(projectPath) {
    const codeExts = new Set(['.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.py', '.go', '.rs', '.sh']);
    const relFiles = listProjectFiles(projectPath, projectPath, 0, 5, 120, [], codeExts);
    const fileStats = [];
    const allSymbols = [];
    const allImports = [];

    for (const relPath of relFiles) {
        const ext = extname(relPath).toLowerCase();
        if (!codeExts.has(ext)) continue;
        const fullPath = join(projectPath, relPath);
        try {
            const stat = fs.statSync(fullPath);
            if (!stat.isFile() || stat.size > 600_000) continue;
            const content = fs.readFileSync(fullPath, 'utf8');
            const loc = content.split('\n').length;
            const { symbols = [], imports = [] } = extractSymbolsAndImports(content, relPath);
            for (const sym of symbols) {
                allSymbols.push({
                    ...sym,
                    relPath,
                    symbolName: sym.name || sym.symbol_name || 'anonymous',
                    symbolType: sym.type || sym.symbol_type || 'symbol',
                    startLine: sym.startLine || sym.line_start || 1,
                    endLine: sym.endLine || sym.line_end || 1
                });
            }
            for (const imp of imports) {
                allImports.push({
                    ...imp,
                    sourcePath: relPath
                });
            }
            fileStats.push({
                relPath,
                fullPath,
                loc,
                symbolCount: symbols.length,
                importCount: imports.length,
                symbols,
                imports,
                content
            });
        } catch {}
    }

    // Calculate inbound dependent counts per file
    for (const f of fileStats) {
        const base = basename(f.relPath, extname(f.relPath));
        f.dependents = allImports.filter(imp => {
            if (imp.sourcePath === f.relPath) return false;
            const t = String(imp.targetPath || '');
            return t.endsWith(f.relPath) || t.endsWith(`/${base}`) || t.endsWith(`/${ basename(f.relPath) }`) || t === `./${base}` || t === `./${basename(f.relPath)}`;
        });
    }

    return { fileStats, allSymbols, allImports };
}

function runProjectSymbolSearch(rawQuery, projectPath) {
    let queryStr = String(rawQuery || '').trim();
    if (queryStr.startsWith('{')) {
        const parsed = parseLooseJsonObject(queryStr, '');
        queryStr = parsed.query || parsed.symbol || parsed.name || queryStr;
    }
    queryStr = String(queryStr).replace(/^[`'"]+|[`'"]+$/g, '').trim();
    const { allSymbols } = buildProjectAstIndex(projectPath);
    if (allSymbols.length === 0) {
        return `[Tier 2 AST search_symbols] No parseable source symbols found in ${projectPath}.`;
    }

    const qLower = queryStr.toLowerCase();
    const tokens = qLower.split(/[\s,;|/]+/).filter(Boolean);

    const scored = allSymbols.map(s => {
        const nameLower = s.symbolName.toLowerCase();
        const sigLower = String(s.signature || '').toLowerCase();
        const pathLower = s.relPath.toLowerCase();
        let score = 0;
        if (!qLower || qLower === '*' || qLower === 'all') {
            score = 1;
        } else if (nameLower === qLower) {
            score = 100;
        } else if (nameLower.includes(qLower)) {
            score = 75;
        } else if (sigLower.includes(qLower) || pathLower.includes(qLower)) {
            score = 50;
        } else {
            for (const tok of tokens) {
                if (nameLower.includes(tok)) score += 25;
                else if (sigLower.includes(tok) || pathLower.includes(tok)) score += 10;
            }
        }
        return { sym: s, score };
    }).filter(x => x.score > 0).sort((a, b) => b.score - a.score || a.sym.relPath.localeCompare(b.sym.relPath));

    const top = (scored.length > 0 ? scored : allSymbols.slice(0, 15).map(sym => ({ sym, score: 1 }))).slice(0, 12);
    const lines = [`[Tier 2 AST Symbol Search ("${queryStr || '*'}") — ${top.length} of ${allSymbols.length} total project symbols]`];
    for (let i = 0; i < top.length; i++) {
        const { sym, score } = top[i];
        const rawLines = String(sym.content || '').split('\n');
        // Hydrate full function body (up to 50 lines) for top 3 or high-confidence symbol matches so Jean never misreads a truncated 8-line header as a stub
        const maxSnippetLines = (i < 3 || score >= 75) ? 50 : 16;
        const snippet = rawLines
            .slice(0, maxSnippetLines)
            .map((l, idx) => `  ${sym.startLine + idx}: ${l}`)
            .join('\n');
        const contNote = rawLines.length > maxSnippetLines
            ? `\n  ... [symbol body continues from line ${sym.startLine + maxSnippetLines} to line ${sym.endLine} (${rawLines.length} lines total) — use \`read_file ${sym.relPath}:${sym.startLine}-${sym.endLine}\` to inspect remaining lines]`
            : '';
        lines.push(`- **${sym.symbolName}** [${sym.symbolType}] at \`${sym.relPath}:${sym.startLine}-${sym.endLine}\`\n  Signature: \`${sym.signature || sym.symbolName}\`\n${snippet}${contNote}`);
    }
    return lines.join('\n');
}

function runProjectFileSymbols(rawTarget, projectPath) {
    let targetStr = String(rawTarget || '').trim();
    if (targetStr.startsWith('{')) {
        const parsed = parseLooseJsonObject(targetStr, '');
        targetStr = parsed.file_path || parsed.filePath || parsed.path || parsed.target || parsed.query || targetStr;
    }
    targetStr = normalizeHomelabPath(targetStr).replace(/^[`'"]+|[`'"]+$/g, '').trim();
    if (targetStr.startsWith(projectPath)) {
        targetStr = targetStr.slice(projectPath.length).replace(/^\/+/, '');
    }

    const { fileStats } = buildProjectAstIndex(projectPath);
    if (fileStats.length === 0) {
        return `[Tier 2 AST krusch_git_file_symbols] No source files indexed in ${projectPath}.`;
    }

    const tLower = targetStr.toLowerCase();
    const matchedFile = fileStats.find(f =>
        f.relPath.toLowerCase() === tLower ||
        f.relPath.toLowerCase().endsWith(`/${tLower}`) ||
        basename(f.relPath).toLowerCase() === tLower
    ) || fileStats[0];

    const out = [`[Tier 2 AST File Symbol Table for \`${matchedFile.relPath}\` (${matchedFile.loc} LOC, ${matchedFile.symbols.length} symbols)]`];
    const outbound = matchedFile.imports.map(i => `\`${i.targetPath}\`${i.symbols?.length ? ` (${i.symbols.join(', ')})` : ''}`);
    const inbound = matchedFile.dependents.map(d => `\`${d.sourcePath}\``);
    out.push(`- **Imports (${outbound.length}):** ${outbound.length ? outbound.join(', ') : '(none)'}`);
    out.push(`- **Imported By (${inbound.length}):** ${inbound.length ? inbound.join(', ') : '(none)'}`);

    if (matchedFile.symbols.length === 0) {
        const previewLines = matchedFile.content.split('\n').slice(0, 60).map((l, idx) => `  ${idx + 1}: ${l}`).join('\n');
        out.push(`- **Top-Level Script Preview (lines 1-${Math.min(60, matchedFile.loc)} of ${matchedFile.loc}):**\n${previewLines}`);
        return out.join('\n');
    }

    out.push(`\n### Declared Symbols & Implementation Previews`);
    for (const s of matchedFile.symbols.slice(0, 20)) {
        const symName = s.name || s.symbol_name || 'anonymous';
        const symType = s.type || s.symbol_type || 'symbol';
        const startLine = s.startLine || s.line_start || 1;
        const endLine = s.endLine || s.line_end || startLine;
        const rawLines = String(s.content || '').split('\n');
        const maxLines = rawLines.length <= 30 ? rawLines.length : 18;
        const snippet = rawLines.slice(0, maxLines).map((l, idx) => `  ${startLine + idx}: ${l}`).join('\n');
        const contNote = rawLines.length > maxLines
            ? `\n  ... [continues to line ${endLine} (${rawLines.length} lines total) — use \`read_file ${matchedFile.relPath}:${startLine}-${endLine}\` for full body]`
            : '';
        out.push(`- **[${symType}] \`${symName}\`** (\`${matchedFile.relPath}:${startLine}-${endLine}\`) — \`${s.signature || symName}\`\n${snippet}${contNote}`);
    }
    return out.join('\n');
}

function runProjectDependencyGraph(rawTarget, projectPath) {
    let targetStr = String(rawTarget || '').trim();
    if (targetStr.startsWith('{')) {
        const parsed = parseLooseJsonObject(targetStr, '');
        targetStr = parsed.symbol || parsed.file_path || parsed.filePath || parsed.target || parsed.query || targetStr;
    }
    targetStr = normalizeHomelabPath(targetStr).replace(/^[`'"]+|[`'"]+$/g, '').trim();
    if (targetStr.startsWith(projectPath)) {
        targetStr = targetStr.slice(projectPath.length).replace(/^\/+/, '');
    }

    const { fileStats, allSymbols } = buildProjectAstIndex(projectPath);
    if (fileStats.length === 0) {
        return `[Tier 2 AST dependency_graph] No source files indexed in ${projectPath}.`;
    }

    const tLower = targetStr.toLowerCase();
    const matchedFile = fileStats.find(f =>
        f.relPath.toLowerCase() === tLower ||
        f.relPath.toLowerCase().endsWith(`/${tLower}`) ||
        basename(f.relPath).toLowerCase() === tLower
    );

    const matchedSymbols = allSymbols.filter(s =>
        s.symbolName.toLowerCase() === tLower ||
        (tLower.length >= 3 && s.symbolName.toLowerCase().includes(tLower))
    );

    const out = [`[Tier 2 AST Dependency & Caller Graph for "${targetStr || 'project'}"]`];

    if (matchedFile) {
        out.push(`\n### File Node: \`${matchedFile.relPath}\` (${matchedFile.loc} LOC)`);
        const outbound = matchedFile.imports.map(i => `  - imports \`${i.targetPath}\`${i.symbols?.length ? ` (${i.symbols.join(', ')})` : ''}`);
        out.push(`**Outbound Imports (${outbound.length}):**\n${outbound.length ? outbound.join('\n') : '  - (none)'}`);

        const inbound = matchedFile.dependents.map(d => `  - imported by \`${d.sourcePath}\` via \`${d.targetPath}\`${d.symbols?.length ? ` (${d.symbols.join(', ')})` : ''}`);
        out.push(`**Inbound File Dependents (${inbound.length}):**\n${inbound.length ? inbound.join('\n') : '  - (no internal module imports found)'}`);

        const decls = matchedFile.symbols.slice(0, 15).map(s => `  - [${s.type || s.symbol_type}] \`${s.name}\` (\`${matchedFile.relPath}:${s.startLine}-${s.endLine}\`) — \`${s.signature}\``);
        out.push(`**Declared Symbols (showing ${decls.length} of ${matchedFile.symbols.length} total — summarize by responsibility group; do NOT enumerate or guess unshown symbols):**\n${decls.length ? decls.join('\n') : '  - (none)'}`);
    }

    const symbolsToTrace = matchedSymbols.length > 0
        ? matchedSymbols.slice(0, 4)
        : (matchedFile ? matchedFile.symbols.slice(0, 4).map(s => ({ ...s, relPath: matchedFile.relPath, symbolName: s.name })) : []);

    if (symbolsToTrace.length > 0) {
        out.push(`\n### Symbol Call-Site, Body & Reference Trace`);
        for (const sym of symbolsToTrace) {
            const symName = sym.symbolName || sym.name;
            if (!symName || symName.length < 2) continue;
            const escaped = symName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const callRegex = new RegExp(`\\b${escaped}\\b`);
            const refs = [];
            for (const f of fileStats) {
                const lines = f.content.split('\n');
                for (let i = 0; i < lines.length; i++) {
                    const lineNum = i + 1;
                    if (f.relPath === sym.relPath && lineNum >= sym.startLine && lineNum <= sym.endLine && lineNum === sym.startLine) continue;
                    if (callRegex.test(lines[i])) {
                        refs.push(`  - \`${f.relPath}:${lineNum}\`: \`${lines[i].trim().slice(0, 120)}\``);
                        if (refs.length >= 10) break;
                    }
                }
                if (refs.length >= 10) break;
            }
            const rawLines = String(sym.content || '').split('\n');
            const maxPreview = Math.min(rawLines.length, 28);
            const bodyPreview = rawLines.length > 0 && rawLines[0]
                ? `\n  \`\`\`javascript\n${rawLines.slice(0, maxPreview).map((l, idx) => `  ${sym.startLine + idx}: ${l}`).join('\n')}${rawLines.length > maxPreview ? `\n  ... [continues to line ${sym.endLine} — use read_file ${sym.relPath}:${sym.startLine + maxPreview}-${sym.endLine} for rest]` : ''}\n  \`\`\``
                : '';
            out.push(`- **\`${symName}\`** (declared at \`${sym.relPath}:${sym.startLine}-${sym.endLine}\`):${bodyPreview}\n${refs.length ? refs.join('\n') : '  - (no external call sites found in project)'}`);
        }
    } else if (!matchedFile) {
        // Fallback overview of whole-project coupling graph
        out.push(`\n### Project-Wide Coupling Overview (target "${targetStr}" not found verbatim; showing top coupled modules)`);
        const sorted = [...fileStats].sort((a, b) => (b.importCount + b.dependents.length + b.symbolCount) - (a.importCount + a.dependents.length + a.symbolCount)).slice(0, 10);
        for (const f of sorted) {
            out.push(`- \`${f.relPath}\` (${f.loc} LOC, ${f.symbolCount} symbols): imports=[${f.imports.map(i => i.targetPath).join(', ') || 'none'}] | imported_by=[${f.dependents.map(d => d.sourcePath).join(', ') || 'none'}]`);
        }
    }

    return out.join('\n');
}

function parseMcpToolAction(rawAction, projectPath, reasonText) {
    const mcpRegex = /\b(krusch_context_retrieve|krusch_context_nudge|krusch_context_health|krusch_fs_search|krusch_fs_locate|krusch_fs_page_tree|krusch_fs_read_page|krusch_git_semantic_search|krusch_git_file_symbols|krusch_git_read_blob|krusch_git_read_tree|krusch_git_list_repos|query_design_rules|analyze_layout_structure|get_canonical_template|list_canonical_templates)\b/i;
    const match = rawAction.match(mcpRegex);
    if (!match) return null;

    const baseName = match[1].toLowerCase();
    const afterTool = rawAction
        .replace(new RegExp(`^[\\s\\S]*?\\b${baseName}\\b[\\]\\s\`':()*]*`, 'i'), '')
        .replace(/\)\s*$/, '')
        .trim();

    const jsonMatch = afterTool.match(/(\{[\s\S]*\})/);
    let toolArgs = {};
    const defaultRepo = (projectPath || HOMELAB_ROOT).split('/').filter(Boolean).pop();
    if (jsonMatch) {
        try {
            toolArgs = JSON.parse(jsonMatch[1]);
        } catch {
            const parsed = parseLooseJsonObject(jsonMatch[1], '');
            if (parsed.node === '') delete parsed.node;
            toolArgs = parsed;
        }
    } else {
        const tokenMatch = afterTool.match(/^`([^`]+)`|^"([^"]+)"|^'([^']+)'|^([^\s`*]+)/);
        const val = (tokenMatch ? (tokenMatch[1] || tokenMatch[2] || tokenMatch[3] || tokenMatch[4]) : afterTool)
            .trim()
            .replace(/^[`' "*]+|[`' "*]+$/g, '');

        if (baseName === 'krusch_context_retrieve') {
            toolArgs = { query: val || defaultRepo, project: defaultRepo, include_state: true, limit_tokens: 2000 };
        } else if (baseName === 'krusch_context_nudge') {
            toolArgs = { trigger: 'pre_commit', project: defaultRepo, code: val || defaultRepo };
        } else if (baseName === 'krusch_fs_search') {
            toolArgs = { query: val || '*' };
        } else if (baseName === 'krusch_fs_locate') {
            toolArgs = { pattern: val || '*' };
        } else if (baseName === 'krusch_fs_page_tree' || baseName === 'krusch_fs_read_page') {
            const fp = (val && !val.startsWith('/')) ? join(projectPath || HOMELAB_ROOT, val) : val;
            toolArgs = baseName === 'krusch_fs_read_page' ? { file_path: fp, path: fp, page_number: 1 } : { file_path: fp, path: fp };
        } else if (baseName === 'krusch_git_semantic_search') {
            toolArgs = { repo: defaultRepo, query: val || '*' };
        } else if (baseName === 'krusch_git_file_symbols' || baseName === 'krusch_git_read_blob') {
            toolArgs = { repo: defaultRepo, file_path: val };
        } else if (baseName === 'krusch_git_read_tree') {
            toolArgs = { repo: defaultRepo, tree_sha: val || 'HEAD' };
        } else if (baseName === 'query_design_rules') {
            toolArgs = { queryStr: val || 'layout spacing contrast' };
        } else if (baseName === 'analyze_layout_structure') {
            toolArgs = { htmlString: val };
        } else if (baseName === 'get_canonical_template') {
            toolArgs = { templateKey: val || 'article-shell' };
        }
    }

    if (baseName === 'krusch_fs_page_tree' || baseName === 'krusch_fs_read_page') {
        let fp = toolArgs.file_path || toolArgs.path || '';
        fp = normalizeHomelabPath(fp);
        if (fp && !fp.startsWith('/')) {
            fp = join(projectPath || HOMELAB_ROOT, fp);
        }
        toolArgs.file_path = fp;
        toolArgs.path = fp;
    }
    if (baseName.startsWith('krusch_git_') && !toolArgs.repo && !toolArgs.project) {
        toolArgs.repo = defaultRepo;
    }
    if (baseName.startsWith('krusch_context_') && !toolArgs.project && !toolArgs.active_project && defaultRepo) {
        toolArgs.project = defaultRepo;
    }

    let serverPrefix = 'krusch-fs';
    if (baseName.startsWith('krusch_git_')) {
        serverPrefix = 'krusch-git';
    } else if (baseName.startsWith('krusch_context_')) {
        serverPrefix = 'krusch-context';
    } else if (['query_design_rules', 'analyze_layout_structure', 'get_canonical_template', 'list_canonical_templates'].includes(baseName)) {
        serverPrefix = 'krusch-frontend';
    }

    const toolName = `${serverPrefix}-${baseName.replace(/_/g, '-')}`;
    return {
        kind: 'mcp_tool',
        toolName,
        toolArgs,
        command: `${baseName}(${JSON.stringify(toolArgs)})`,
        reasonText
    };
}

function parseExecutableAction(taskStr, projectPath = HOMELAB_ROOT) {
    const parts = String(taskStr || '').split(/\*\*Reason\*\*\s*:|\bReason\s*:/i);
    const rawAction = normalizeHomelabPath(parts[0].trim());
    const reasonText = (parts[1] || '').trim();

    // 1. Sentinel tool call in Infra Mode (including real FRAXIS self-state security tools)
    const sentinelMatch = rawAction.match(/(get_system_stats|get_disk_usage|get_docker_containers|list_docker_images|list_docker_volumes|find_large_files|read_remote_file|list_systemd_services|get_gpu_stats|get_sysctl_params|get_mount_options|get_docker_daemon_config|get_cpu_governor|get_network_tuning|get_process_health|get_service_logs|audit_agent_self_state|detect_memory_anomaly|verify_state_integrity)\s*\(\s*(\{[\s\S]*?\})?\s*\)/i);
    if (sentinelMatch) {
        const baseName = sentinelMatch[1].toLowerCase();
        const toolName = `sentinel-${baseName.replace(/_/g, '-')}`;
        const toolArgs = parseLooseJsonObject(sentinelMatch[2]);
        return {
            kind: 'sentinel_tool',
            toolName,
            toolArgs,
            command: `${baseName}(${JSON.stringify(toolArgs)})`,
            reasonText
        };
    }

    // 1b. Direct MCP tools (krusch-fs, krusch-frontend)
    const mcpAction = parseMcpToolAction(rawAction, projectPath, reasonText);
    if (mcpAction) {
        return mcpAction;
    }

    // 2. Tier 2 AST Symbol Search (search_symbols / krusch_git_search_symbols)
    if (/\b(?:krusch_git_)?search_symbols\b/i.test(rawAction)) {
        const afterSym = rawAction.replace(/^[\s\S]*?\b(?:krusch_git_)?search_symbols\b[\]\s`':()*]*/i, '').replace(/\)\s*$/, '').trim();
        const m = afterSym.match(/^`([^`]+)`|^"([^"]+)"|^'([^']+)'|^(\{[\s\S]*?\})|^([^\s`*]+)/);
        let query = (m ? (m[1] || m[2] || m[3] || m[4] || m[5]) : afterSym).trim().replace(/^[`' "*]+|[`' "*]+$/g, '');
        if (query.startsWith('{')) {
            const obj = parseLooseJsonObject(query, '');
            query = String(obj.query || obj.symbol || obj.name || obj.target || query).trim();
        }
        return {
            kind: 'search_symbols',
            query,
            command: `search_symbols ${query}`,
            reasonText
        };
    }

    // 3. Tier 2 AST Dependency Graph (dependency_graph / krusch_git_dependency_graph)
    if (/\b(?:krusch_git_)?dependency_graph\b/i.test(rawAction)) {
        const afterDep = rawAction.replace(/^[\s\S]*?\b(?:krusch_git_)?dependency_graph\b[\]\s`':()*]*/i, '').replace(/\)\s*$/, '').trim();
        const m = afterDep.match(/^`([^`]+)`|^"([^"]+)"|^'([^']+)'|^(\{[\s\S]*?\})|^([^\s`*]+)/);
        let target = (m ? (m[1] || m[2] || m[3] || m[4] || m[5]) : afterDep).trim().replace(/^[`' "*]+|[`' "*]+$/g, '');
        if (target.startsWith('{')) {
            const obj = parseLooseJsonObject(target, '');
            target = String(obj.symbol || obj.target || obj.file_path || obj.query || target).trim();
        }
        return {
            kind: 'dependency_graph',
            target,
            command: `dependency_graph ${target}`,
            reasonText
        };
    }

    // 4. Explicit read_file action (supports optional :start-end line range)
    if (/\bread_file\b/i.test(rawAction)) {
        let afterRead = rawAction.replace(/^[\s\S]*?\bread_file\b[\]\s`':()*]*/i, '').trim();
        const m = afterRead.match(/^`([^`]+)`|^"([^"]+)"|^'([^']+)'|^([^\s`*]+)/);
        let rawTarget = (m ? (m[1] || m[2] || m[3] || m[4]) : afterRead).trim().replace(/^[`' "*]+|[`' "*]+$/g, '');
        let startLine = 1;
        let endLine = null;
        const rangeMatch = rawTarget.match(/^(.*?):(\d+)(?:-(\d+))?$/);
        let filePath = rawTarget;
        if (rangeMatch) {
            filePath = rangeMatch[1];
            startLine = Math.max(1, parseInt(rangeMatch[2], 10) || 1);
            endLine = rangeMatch[3] ? Math.max(startLine, parseInt(rangeMatch[3], 10)) : startLine + 160;
        }
        if (filePath && !filePath.startsWith('/')) {
            filePath = join(projectPath || HOMELAB_ROOT, filePath);
        }
        const rangeSuffix = endLine ? `:${startLine}-${endLine}` : '';
        return {
            kind: 'read_file',
            filePath,
            startLine,
            endLine,
            command: `read_file ${filePath}${rangeSuffix}`,
            reasonText
        };
    }

    // 5. Explicit run_shell_command action
    if (/\brun_shell_command\b/i.test(rawAction)) {
        let afterCmd = rawAction.replace(/^[\s\S]*?\brun_shell_command\b[\]\s`':()*]*/i, '').trim();
        const backticks = [...afterCmd.matchAll(/`([^`]+)`/g)].map(x => x[1].trim()).filter(Boolean);
        let shellCommand = backticks.length > 0
            ? backticks[0]
            : afterCmd.replace(/^[`*\s]+|[`*\s]+$/g, '').replace(/^(['"])([\s\S]*)\1$/, '$2').trim();
        const innerMcp = parseMcpToolAction(shellCommand, projectPath, reasonText);
        if (innerMcp) return innerMcp;
        if (/^read_file\s+/i.test(shellCommand)) {
            let rawTarget = shellCommand.replace(/^read_file\s+/i, '').trim().replace(/^[`'"]+|[`'"]+$/g, '');
            let startLine = 1;
            let endLine = null;
            const rangeMatch = rawTarget.match(/^(.*?):(\d+)(?:-(\d+))?$/);
            let filePath = rawTarget;
            if (rangeMatch) {
                filePath = rangeMatch[1];
                startLine = Math.max(1, parseInt(rangeMatch[2], 10) || 1);
                endLine = rangeMatch[3] ? Math.max(startLine, parseInt(rangeMatch[3], 10)) : startLine + 160;
            }
            if (filePath && !filePath.startsWith('/')) {
                filePath = join(projectPath || HOMELAB_ROOT, filePath);
            }
            const rangeSuffix = endLine ? `:${startLine}-${endLine}` : '';
            return {
                kind: 'read_file',
                filePath,
                startLine,
                endLine,
                command: `read_file ${filePath}${rangeSuffix}`,
                reasonText
            };
        }
        if (/^(?:krusch_git_)?search_symbols\s+/i.test(shellCommand)) {
            const query = shellCommand.replace(/^(?:krusch_git_)?search_symbols\s+/i, '').trim().replace(/^[`'"]+|[`'"]+$/g, '');
            return {
                kind: 'search_symbols',
                query,
                command: `search_symbols ${query}`,
                reasonText
            };
        }
        if (/^(?:krusch_git_)?dependency_graph\s+/i.test(shellCommand)) {
            const target = shellCommand.replace(/^(?:krusch_git_)?dependency_graph\s+/i, '').trim().replace(/^[`'"]+|[`'"]+$/g, '');
            return {
                kind: 'dependency_graph',
                target,
                command: `dependency_graph ${target}`,
                reasonText
            };
        }
        return {
            kind: 'run_shell_command',
            shellCommand,
            command: shellCommand,
            reasonText
        };
    }

    // 6. Fallback: extract backticked command or file path
    const backtickMatches = [...rawAction.matchAll(/`([^`]+)`/g)].map(x => x[1].trim()).filter(Boolean);
    for (const candidate of backtickMatches) {
        if (/^(?:read_file|run_shell_command|search_symbols|dependency_graph|krusch_git_search_symbols|krusch_git_dependency_graph|krusch_fs_search|krusch_fs_locate|krusch_fs_page_tree|krusch_fs_read_page|query_design_rules|analyze_layout_structure|action)$/i.test(candidate)) continue;
        const candidateMcp = parseMcpToolAction(candidate, projectPath, reasonText);
        if (candidateMcp) return candidateMcp;
        if (candidate.startsWith('/') && !/\s/.test(candidate)) {
            return {
                kind: 'read_file',
                filePath: candidate,
                startLine: 1,
                endLine: null,
                command: `read_file ${candidate}`,
                reasonText
            };
        }
        return {
            kind: 'run_shell_command',
            shellCommand: candidate,
            command: candidate,
            reasonText
        };
    }

    return {
        kind: 'unknown',
        command: rawAction,
        reasonText
    };
}

/**
 * Classify the risk profile of an SRE task into L0 (Destructive), L1 (Mild/Restorative), or L2 (Safe/Read-only).
 * Evaluates the executable command rather than the English justification prose.
 */
function classifyIntentRisk(task, parsedAction = null) {
    if (parsedAction?.kind === 'read_file') {
        const readCheck = verifyToolCompliance('read_file', { path: parsedAction.filePath || '' });
        if (!readCheck.valid) {
            return { riskClass: 'L0', reason: readCheck.reason };
        }
        return { riskClass: 'L2', reason: 'Safe read-only file inspection' };
    }

    if (parsedAction?.kind === 'mcp_tool') {
        const mcpCheck = verifyToolCompliance(parsedAction.toolName, parsedAction.toolArgs || {});
        if (!mcpCheck.valid) {
            return { riskClass: 'L0', reason: mcpCheck.reason };
        }
        return { riskClass: 'L2', reason: 'Safe read-only MCP tool inspection' };
    }

    if (parsedAction && ['sentinel_tool', 'search_symbols', 'dependency_graph'].includes(parsedAction.kind)) {
        return { riskClass: 'L2', reason: 'Safe read-only file/AST/telemetry inspection' };
    }

    const commandTarget = parsedAction?.command || String(task || '').split(/\*\*Reason\*\*\s*:|\bReason\s*:/i)[0];
    const shellCheck = verifyToolCompliance('run_shell_command', { command: commandTarget });
    if (!shellCheck.valid) {
        return { riskClass: 'L0', reason: shellCheck.reason };
    }

    const t = commandTarget.toLowerCase();
    const unquotedT = t.replace(/'[^']*'|"(?:\\.|[^"\\])*"/g, '""');
    
    // L0 (Destructive) - Forbidden from autonomous run, goes to review
    const l0Triggers = [
        { pattern: /drop\s+table/i, reason: 'mutates DB schema / drops table' },
        { pattern: /drop\s+database/i, reason: 'mutates DB / drops database' },
        { pattern: /delete\s+volume/i, reason: 'removes Docker volume' },
        { pattern: /volume\s+prune/i, reason: 'prunes Docker volumes' },
        { pattern: /system\s+prune/i, reason: 'prunes Docker system configurations' },
        { pattern: /\brm\s+-[rf]+/i, reason: 'recursive/forced file deletion' },
        { pattern: /wipe\s+mount/i, reason: 'wipes volume/disk mounts' },
        { pattern: /(?<!--)format\s+/i, reason: 'disk formatting' },
        { pattern: /\breboot\b/i, reason: 'system node reboot' },
        { pattern: /\bshutdown\b/i, reason: 'system node shutdown' },
        { pattern: /\bdestroy\b/i, reason: 'infrastructure destruction' },
        { pattern: /delete\s+file/i, reason: 'destroys file assets' }
    ];

    for (const trigger of l0Triggers) {
        if (trigger.pattern.test(t)) {
            return { riskClass: 'L0', reason: trigger.reason };
        }
    }
    
    // L1 (Mild / Restorative) - Needs review/gate, goes to review
    const l1Triggers = [
        { pattern: /(?:^|[;&|]\s*)(?:docker|systemctl|service)\s+(?:restart|stop|start|reload|rm|kill|update|prune)\b/i, reason: 'mutates container or service state' },
        { pattern: /(?:^|[;&|]\s*)(?:rm|mv|cp|dd|chmod|chown|touch|mkdir|rmdir|kill|pkill|tee|truncate)\b/i, reason: 'mutates filesystem or processes' },
        { pattern: /\bsed\s+-i\b/i, reason: 'in-place file mutation' },
        { pattern: /\bgit\s+(?:checkout|reset|clean|commit|push|merge|rebase|stash)\b/i, reason: 'alters git repository state' },
        { pattern: /\bnpm\s+(?:install|ci|uninstall|update|audit\s+fix)\b/i, reason: 'mutates node dependencies' },
        { pattern: /(?:^|[^2>=-])>\s*(?!\/dev\/null)|>>/i, reason: 'redirects output to file', useUnquoted: true },
        { pattern: /\$\(|`[^`]+`|\beval\b/i, reason: 'uses subshell command substitution or eval' },
        { pattern: /\b(?:node\s+-e|python3?\s+-c|bash\s+-c|sh\s+-c|perl\s+-e|ruby\s+-e)\b/i, reason: 'executes inline interpreter script' },
        { pattern: /\bcurl\b[^;|&\n]*(?:-X\s*(?:POST|PUT|DELETE|PATCH)|--data|-d\s|-F\s)/i, reason: 'executes mutating HTTP request via curl' }
    ];

    for (const trigger of l1Triggers) {
        const targetStr = trigger.useUnquoted ? unquotedT : t;
        if (trigger.pattern.test(targetStr)) {
            return { riskClass: 'L1', reason: trigger.reason };
        }
    }
    
    // L2 (Safe / Read-Only) - safe to auto-run inline
    return { riskClass: 'L2', reason: 'Safe read-only diagnostics/telemetry retrieval' };
}

/**
 * Execute a verified L2 read-only intent inline and return captured output.
 */
async function executeSafeIntentInline(parsedAction, projectPath = HOMELAB_ROOT) {
    try {
        if (parsedAction.kind === 'sentinel_tool' || parsedAction.kind === 'mcp_tool') {
            const tName = String(parsedAction.toolName || '');
            const tArgs = parsedAction.toolArgs || {};

            // 1. krusch_git_file_symbols: always hydrate live AST symbol ranges + implementation previews from disk alongside MCP response
            if (tName.includes('file-symbols')) {
                const localAstSymbols = runProjectFileSymbols(tArgs.file_path || tArgs.path || '', projectPath);
                const mcpRes = await callTool(tName, tArgs, { timeout: 15000 });
                const cleanMcp = (!String(mcpRes || '').startsWith('[Failed:') && !String(mcpRes || '').includes('No symbols found'))
                    ? `${String(mcpRes).trim()}\n\n`
                    : '';
                return {
                    ok: true,
                    output: `${cleanMcp}${localAstSymbols}`.slice(0, 5500)
                };
            }

            // 1b. krusch_git_semantic_search: fallback to local symbol + keyword search if repo not yet indexed or 0 PG hits
            if (tName.includes('semantic-search')) {
                const mcpRes = await callTool(tName, tArgs, { timeout: 25000 });
                const resStr = String(mcpRes || '');
                const isFailedOrEmpty = resStr.startsWith('[Failed:') || resStr.startsWith('Error:') || resStr.includes('not found') || resStr.includes('No matching code') || resStr.length < 40;
                if (!isFailedOrEmpty) {
                    return { ok: true, output: resStr.slice(0, 5500) };
                }
                const fallbackSym = runProjectSymbolSearch(tArgs.query || '*', projectPath);
                return {
                    ok: true,
                    output: `[Hybrid AST + Keyword Search in ${projectPath} for "${tArgs.query || '*'}"]\n${fallbackSym}`.slice(0, 5500)
                };
            }

            // 2. krusch_git_read_blob / krusch_git_read_tree: fallback to on-disk file/tree if PG DAG is not yet synced
            if (tName.includes('read-blob') || tName.includes('read-tree')) {
                const mcpRes = await callTool(tName, tArgs, { timeout: 20000 });
                const isFailed = String(mcpRes || '').startsWith('[Failed:') || String(mcpRes || '').startsWith('Error:') || String(mcpRes || '').length < 40;
                if (!isFailed) {
                    return { ok: true, output: String(mcpRes).slice(0, 5500) };
                }
                if (tName.includes('read-tree')) {
                    const files = listProjectFiles(projectPath, projectPath, 0, 4, 80);
                    return { ok: true, output: `[Fallback Project Tree for ${projectPath}]\n${files.join('\n')}` };
                }
                const relFp = String(tArgs.file_path || tArgs.path || '').replace(/^\/+/, '');
                const fullFp = relFp.startsWith('/') ? relFp : join(projectPath, relFp);
                if (fs.existsSync(fullFp) && fs.statSync(fullFp).isFile()) {
                    const allLines = fs.readFileSync(fullFp, 'utf8').split('\n');
                    const numbered = allLines.slice(0, 180).map((l, i) => `${i + 1}: ${l}`).join('\n');
                    return { ok: true, output: `[File: ${fullFp} (${allLines.length} lines, showing 1-${Math.min(180, allLines.length)})]\n${numbered}`.slice(0, 5500) };
                }
            }

            // 3. krusch_fs_page_tree / krusch_fs_read_page: auto-sync or synthesize from disk if file not yet indexed in krusch-fs
            if (tName.includes('page-tree') || tName.includes('read-page')) {
                let res = await callTool(tName, tArgs, { timeout: 20000 });
                const targetFp = tArgs.file_path || tArgs.path || '';
                if ((String(res || '').includes('"status": "error"') || String(res || '').startsWith('[Failed:')) && targetFp && fs.existsSync(targetFp)) {
                    await callTool('krusch-fs-krusch-fs-sync', { paths: [targetFp], node: tArgs.node || 'kruschdev' }, { timeout: 15000 });
                    res = await callTool(tName, tArgs, { timeout: 20000 });
                }
                if ((String(res || '').includes('"status": "error"') || String(res || '').startsWith('[Failed:')) && targetFp && fs.existsSync(targetFp) && fs.statSync(targetFp).isFile()) {
                    const allLines = fs.readFileSync(targetFp, 'utf8').split('\n');
                    const pageSize = 120;
                    const totalPages = Math.max(1, Math.ceil(allLines.length / pageSize));
                    if (tName.includes('page-tree')) {
                        const pages = [];
                        for (let p = 1; p <= totalPages; p++) {
                            const sLine = (p - 1) * pageSize + 1;
                            const eLine = Math.min(allLines.length, p * pageSize);
                            const heading = allLines.slice(sLine - 1, eLine).find(l => /^#{1,3}\s+|^(?:export\s+)?(?:async\s+)?(?:function|class)\s+/.test(l.trim())) || `Lines ${sLine}-${eLine}`;
                            pages.push(`  - Page ${p} [lines ${sLine}-${eLine}]: ${heading.trim().slice(0, 90)}`);
                        }
                        return {
                            ok: true,
                            output: `[krusch_fs_page_tree for ${targetFp} (${allLines.length} lines, ${totalPages} pages)]\n${pages.join('\n')}`
                        };
                    } else {
                        const pageNum = Math.max(1, Math.min(totalPages, parseInt(tArgs.page_number || '1', 10) || 1));
                        const sIdx = (pageNum - 1) * pageSize;
                        const eIdx = Math.min(allLines.length, pageNum * pageSize);
                        const pageLines = allLines.slice(sIdx, eIdx).map((l, idx) => `${sIdx + idx + 1}: ${l}`).join('\n');
                        return {
                            ok: true,
                            output: `[krusch_fs_read_page ${targetFp} — Page ${pageNum}/${totalPages} (lines ${sIdx + 1}-${eIdx} of ${allLines.length})]\n${pageLines}`.slice(0, 5500)
                        };
                    }
                }
                const isFailed = String(res || '').startsWith('[Failed:');
                return {
                    ok: !isFailed,
                    output: String(res || '(empty MCP tool response)').slice(0, 5000)
                };
            }

            // 4. krusch_fs_locate / krusch_fs_search: fallback to local project file glob / keyword search if krusch-fs SQLite DB errors
            if (tName.includes('fs-locate') || tName.includes('fs-search')) {
                const res = await callTool(tName, tArgs, { timeout: 20000 });
                const resStr = String(res || '');
                const hasError = resStr.startsWith('[Failed:') || resStr.includes('"status": "error"') || resStr.toLowerCase().includes('database error') || resStr.toLowerCase().includes('operationalerror') || resStr.length < 25;
                if (!hasError) {
                    return { ok: true, output: resStr.slice(0, 5000) };
                }
                const allFiles = listProjectFiles(projectPath, projectPath, 0, 5, 200);
                if (tName.includes('fs-locate')) {
                    const rawPat = String(tArgs.pattern || '*').trim();
                    const rxStr = rawPat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.');
                    const rx = new RegExp(rxStr, 'i');
                    const matched = allFiles.filter(f => rx.test(f) || rx.test(basename(f)));
                    const list = (matched.length > 0 ? matched : allFiles).slice(0, 60).map(f => {
                        try {
                            const st = fs.statSync(join(projectPath, f));
                            return `  - ${f} (${st.size} bytes)`;
                        } catch {
                            return `  - ${f}`;
                        }
                    });
                    return {
                        ok: true,
                        output: `[krusch_fs_locate fallback in ${projectPath} (pattern: "${rawPat}", ${matched.length} matches)]\n${list.join('\n')}`.slice(0, 5000)
                    };
                }
                const q = String(tArgs.query || '*').trim();
                const symOut = runProjectSymbolSearch(q, projectPath);
                return {
                    ok: true,
                    output: `[krusch_fs_search fallback in ${projectPath} (query: "${q}")]\n${symOut}`.slice(0, 5000)
                };
            }

            const res = await callTool(tName, tArgs, { timeout: 30000 });
            const isFailed = String(res || '').startsWith('[Failed:');
            return {
                ok: !isFailed,
                output: String(res || '(empty MCP tool response)').slice(0, 5000)
            };
        }

        if (parsedAction.kind === 'search_symbols') {
            const out = runProjectSymbolSearch(parsedAction.query, projectPath);
            return {
                ok: true,
                output: out.slice(0, 4500)
            };
        }

        if (parsedAction.kind === 'dependency_graph') {
            const out = runProjectDependencyGraph(parsedAction.target, projectPath);
            return {
                ok: true,
                output: out.slice(0, 4500)
            };
        }

        if (parsedAction.kind === 'read_file') {
            let targetPath = parsedAction.filePath;
            if (!targetPath || !fs.existsSync(targetPath)) {
                const allProjFiles = listProjectFiles(projectPath, projectPath, 0, 4, 120);
                const reqBase = basename(String(targetPath || '')).toLowerCase();
                const reqStem = reqBase.replace(/\.[^.]+$/, '');
                const exactBaseMatch = allProjFiles.find(f => basename(f).toLowerCase() === reqBase)
                    || allProjFiles.find(f => f.toLowerCase().includes(reqStem) && reqStem.length >= 3);
                if (exactBaseMatch && fs.existsSync(join(projectPath, exactBaseMatch)) && fs.statSync(join(projectPath, exactBaseMatch)).isFile()) {
                    targetPath = join(projectPath, exactBaseMatch);
                } else {
                    const symFallback = reqStem.length >= 3 ? runProjectSymbolSearch(reqStem, projectPath) : '';
                    return {
                        ok: true,
                        output: `[read_file] "${parsedAction.filePath}" does not exist in ${projectPath}.\nAvailable project files:\n${allProjFiles.slice(0, 35).map(f => `  - ${f}`).join('\n')}${symFallback ? `\n\n${symFallback}` : ''}`.slice(0, 5000)
                    };
                }
            }
            const stat = fs.statSync(targetPath);
            if (stat.isDirectory()) {
                const entries = fs.readdirSync(targetPath, { withFileTypes: true })
                    .map(d => `${d.isDirectory() ? '[DIR] ' : '      '}${d.name}`)
                    .slice(0, 100);
                return {
                    ok: true,
                    output: `[Directory listing for ${targetPath}]\n${entries.join('\n')}`
                };
            }
            const allLines = fs.readFileSync(targetPath, 'utf8').split('\n');
            const totalLines = allLines.length;
            const startIdx = Math.max(0, (parsedAction.startLine || 1) - 1);
            const maxWindow = 180;
            const endIdx = parsedAction.endLine
                ? Math.min(totalLines, parsedAction.endLine, startIdx + maxWindow)
                : Math.min(totalLines, startIdx + maxWindow);
            const numberedLines = allLines
                .slice(startIdx, endIdx)
                .map((line, idx) => `${startIdx + idx + 1}: ${line}`)
                .join('\n')
                .slice(0, 6200);
            const truncatedNote = endIdx < totalLines
                ? `\n... [showing lines ${startIdx + 1}-${endIdx} of ${totalLines} total lines — use \`read_file ${targetPath}:${endIdx + 1}-${Math.min(totalLines, endIdx + maxWindow)}\` to inspect remaining lines; do NOT assume lines ${endIdx + 1}-${totalLines} are missing or unimplemented]`
                : '';
            return {
                ok: true,
                output: `[File: ${targetPath} (${stat.size} bytes, ${totalLines} lines, showing ${startIdx + 1}-${endIdx})]\n${numberedLines}${truncatedNote}`
            };
        }

        if (parsedAction.kind === 'run_shell_command') {
            const cwd = (projectPath && fs.existsSync(projectPath)) ? projectPath : HOMELAB_ROOT;
            let cmdToRun = String(parsedAction.shellCommand || '').trim();
            if (/^\s*npm\s+(?:run\s+)?test\s*$/i.test(cmdToRun)) {
                const suiteRes = runProjectVerificationSuite(cwd, { force: true });
                if (suiteRes.command) {
                    return {
                        ok: suiteRes.status === 'PASS',
                        output: suiteRes.report.slice(0, 5000)
                    };
                }
            }
            if (/^\s*node\s+--test\b/.test(cmdToRun) && !cmdToRun.includes('--test-force-exit')) {
                cmdToRun = cmdToRun.replace(/^\s*node\s+--test\b/, 'node --test --test-force-exit');
            }
            const isTestCmd = /\b(?:npm\s+test|npm\s+run\s+test|node\s+--test|pytest|vitest)\b/i.test(cmdToRun);
            if (/^\s*(?:python3?\s+-m\s+)?pytest\b/.test(cmdToRun)) {
                const venvCandidates = [
                    join(cwd, '.venv', 'bin', 'pytest'),
                    join(cwd, 'venv', 'bin', 'pytest'),
                    join(cwd, 'mcp_env', 'bin', 'pytest'),
                    join(HOMELAB_ROOT, '.venv', 'bin', 'pytest')
                ];
                const resolvedPytest = venvCandidates.find(p => fs.existsSync(p));
                if (resolvedPytest) {
                    cmdToRun = cmdToRun.replace(/^\s*(?:python3?\s+-m\s+)?pytest\b/, resolvedPytest);
                }
            }
            const binPaths = [
                join(cwd, 'node_modules', '.bin'),
                join(HOMELAB_ROOT, 'node_modules', '.bin'),
                process.env.PATH || ''
            ].filter(Boolean).join(':');
            try {
                const stdout = execSync(cmdToRun, {
                    cwd,
                    encoding: 'utf8',
                    timeout: isTestCmd ? 45000 : 15000,
                    maxBuffer: 4 * 1024 * 1024,
                    stdio: ['ignore', 'pipe', 'pipe'],
                    env: buildCleanChildProjectEnv(cwd, binPaths, isTestCmd)
                });
                let cleanOut = String(stdout || '').trim();
                if (/^\s*(?:find|grep)\b/.test(cmdToRun) && cleanOut.length > 0) {
                    const filtered = cleanOut
                        .split('\n')
                        .filter(line => !/(?:^|\/)(?:node_modules|\.git|\.agent|dist|build|\.venv|venv|mcp_env|[a-zA-Z0-9_-]*_env|__pycache__)\//.test(line) && !/\b(?:JEAN_INFLIGHT|GEMINI_INFLIGHT|JEAN_SRE_INFLIGHT)\.md\b/.test(line));
                    cleanOut = filtered.join('\n').trim();
                }
                return {
                    ok: true,
                    output: cleanOut.length > 0 ? cleanOut.slice(0, 5000) : '(command succeeded with 0 matching project lines after excluding node_modules/.git/.agent/env)'
                };
            } catch (cmdErr) {
                const combined = [cmdErr.stdout, cmdErr.stderr].filter(Boolean).join('\n').trim();
                if (cmdErr.status === 1 && !cmdErr.stderr && !isTestCmd) {
                    return {
                        ok: true,
                        output: combined || '(no matches found — exit code 1)'
                    };
                }
                return {
                    ok: false,
                    output: `[Command exited with status ${cmdErr.status ?? 'error'}]\n${(combined || cmdErr.message).slice(0, 4000)}`
                };
            }
        }

        return {
            ok: false,
            output: `[Unrecognized action format: ${parsedAction.command}]`
        };
    } catch (e) {
        return {
            ok: false,
            output: `[Inline execution error: ${e.message}]`
        };
    }
}

/**
 * Execute L2 read-only intents inline and record results in agent_execution_queue,
 * while gating L0/L1 intents into human_review.
 */
async function dispatchIntents(intents, cycle, thinkerAnalysis, projectPath = HOMELAB_ROOT) {
    const dispatched = [];

    try {
        for (const intent of intents) {
            const jobId = randomUUID();
            const threadId = randomUUID();
            const parsedAction = parseExecutableAction(intent.task, projectPath);
            const { riskClass, reason } = classifyIntentRisk(intent.task, parsedAction);
            const startedAtMs = Date.now();

            const payload = {
                task: 'jean_sre_investigation',
                instructions: intent.task,
                parsed_command: parsedAction.command,
                context: {
                    dispatched_by: 'jean_sre_sweep',
                    cycle,
                    thinker_hypothesis: (thinkerAnalysis.match(/### HYPOTHESIS[\s\S]*?(?=###|$)/i) || [''])[0].slice(0, 500),
                    target_node: 'kruschdev',
                    risk_class: riskClass,
                    gating_reason: reason
                }
            };

            if (riskClass === 'L0' || riskClass === 'L1') {
                if (dbPool) {
                    await dbPool.query(
                        `INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, status, started_at_ms)
                         VALUES ($1, $2, 'human_review', $3, 'pending', $4)`,
                        [jobId, threadId, JSON.stringify(payload), startedAtMs]
                    );
                }
                dispatched.push({
                    job_id: jobId,
                    task: intent.task.slice(0, 140),
                    command: parsedAction.command,
                    riskClass,
                    reason,
                    status: 'pending_review',
                    output: null
                });
                console.log(`\n> [!IMPORTANT]\n> ### 🚨 Jean SRE Telemetry Triage Card (PENDING HUMAN REVIEW)\n> - **Job ID**: ${jobId.slice(0, 8)}\n> - **Risk Class**: ${riskClass}\n> - **Trigger**: ${reason}\n> - **Proposed Action**: \`${parsedAction.command.slice(0, 200)}\`\n> - **Status**: Routed to review queue. Awaiting approval.\n`);
                await sendDiscordAlert(`🚨 **Jean SRE Gated Review Dispatched**\n**Job:** \`${jobId.slice(0, 8)}\`\n**Risk:** ${riskClass} (${reason})\n**Proposed Action:** ${parsedAction.command.slice(0, 150)}...\n**Cycle:** ${cycle}`);
            } else {
                console.log(`[Execute] ⚡ Inline L2 (${jobId.slice(0, 8)}): ${parsedAction.command.slice(0, 100)}`);
                const execRes = await executeSafeIntentInline(parsedAction, projectPath);
                const completedAtMs = Date.now();
                const resultObj = {
                    success: execRes.ok,
                    executed_inline: true,
                    command: parsedAction.command,
                    output: execRes.output
                };

                if (dbPool) {
                    await dbPool.query(
                        `INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, result, status, started_at_ms, picked_at_ms, completed_at_ms)
                         VALUES ($1, $2, 'intent', $3, $4, 'completed', $5, $5, $6)`,
                        [jobId, threadId, JSON.stringify(payload), JSON.stringify(resultObj), startedAtMs, completedAtMs]
                    );
                }

                dispatched.push({
                    job_id: jobId,
                    task: intent.task.slice(0, 140),
                    command: parsedAction.command,
                    riskClass,
                    reason,
                    status: execRes.ok ? 'completed' : 'completed_with_warning',
                    output: execRes.output
                });
                console.log(`[Execute] ✅ Completed ${jobId.slice(0, 8)} (${execRes.output.length} chars)`);
            }
        }
    } catch (e) {
        console.error(`[Dispatch] Failed:`, e.message);
        dispatched.push({ job_id: 'failed', task: 'intent execution failed', error: e.message });
    }

    return dispatched;
}


// ─────────────────────────────────────────────────────────
// Workflow Routing
// ─────────────────────────────────────────────────────────

const WORKFLOWS = {
    TELEMETRY: 'WORKFLOW_TELEMETRY: Focus exclusively on hardware health, load averages, memory leaks, and disk saturation using sentinel tools.',
    ORCHESTRATION: 'WORKFLOW_ORCHESTRATION: Focus exclusively on Docker health, DBOS Swarm queues, and pipeline bottlenecks.',
    CODE_AUDIT: 'WORKFLOW_CODE_AUDIT: Read project files and analyze for toolkit migration, dead code, hardcoded secrets, oversized files, and missing patterns.',
    COMPANION_AUDIT: 'WORKFLOW_COMPANION_AUDIT: Focus exclusively on double-checking the IDE agent based on the active GEMINI_INFLIGHT.md state.',
    DEFAULT: 'WORKFLOW_CODE_AUDIT: Read project files and analyze for optimization opportunities.'
};

function routeWorkflow(goal, isInfra = false) {
    if (!isInfra) {
        console.log(`[Router] Code Audit cycle → CODE_AUDIT`);
        return WORKFLOWS.CODE_AUDIT;
    }
    const g = (goal || '').toLowerCase();
    if (/cpu|load|disk|ram|memory/.test(g)) {
        console.log(`[Router] "${goal}" → TELEMETRY`);
        return WORKFLOWS.TELEMETRY;
    }
    console.log(`[Router] "${goal || 'infra review'}" → ORCHESTRATION`);
    return WORKFLOWS.ORCHESTRATION;
}

// ─────────────────────────────────────────────────────────
// State Helpers
// ─────────────────────────────────────────────────────────

function readInflight(targetProject = null, phase = null, isInfra = false) {
    // At the start of a new project rotation (0_ORIENTATION), never inherit prior project Pending Issues
    if (targetProject && !isInfra && phase === '0_ORIENTATION') {
        const freshBlock = [
            `# 🚀 Jean SRE In-Flight State (${targetProject})`,
            `- **Current Focus:** Starting Phase 0 Orientation for \`${targetProject}\``,
            `- **Pending Issues:** (None — fresh 11-phase audit rotation for \`${targetProject}\`; do NOT carry over files from previous projects)`,
            `- **Next Cycle Goal:** Complete \`0_ORIENTATION\` for \`${targetProject}\``
        ].join('\n');
        return {
            block: freshBlock,
            focus: `Starting Phase 0 Orientation for ${targetProject}`,
            goal: `Complete 0_ORIENTATION for ${targetProject}`
        };
    }

    // Prefer project-local JEAN_INFLIGHT.md when auditing a specific project after Phase 0
    if (targetProject && !isInfra) {
        const projInflightPath = join(resolveTargetDir(targetProject), 'JEAN_INFLIGHT.md');
        if (fs.existsSync(projInflightPath)) {
            try {
                const projRaw = fs.readFileSync(projInflightPath, 'utf8');
                if (projRaw.trim().length > 0) {
                    return {
                        block: projRaw,
                        focus: projRaw.match(/- \*\*Current Focus:\*\* (.*)/)?.[1] || '',
                        goal: projRaw.match(/- \*\*Next Cycle Goal:\*\* (.*)/)?.[1] || ''
                    };
                }
            } catch {}
        }
    }

    if (!fs.existsSync('JEAN_SRE_INFLIGHT.md')) return { block: '', focus: '', goal: '' };
    const raw = fs.readFileSync('JEAN_SRE_INFLIGHT.md', 'utf8');
    const focus = raw.match(/- \*\*Current Focus:\*\* (.*)/)?.[1] || '';
    const goal = raw.match(/- \*\*Next Cycle Goal:\*\* (.*)/)?.[1] || '';

    // Guard against cross-project bleed if global JEAN_SRE_INFLIGHT.md references a different project
    if (targetProject && !isInfra && !raw.toLowerCase().includes(targetProject.toLowerCase())) {
        const scopedBlock = [
            `# 🚀 Jean SRE In-Flight State (${targetProject})`,
            `- **Current Focus:** Auditing \`${targetProject}\` (${phase || 'active phase'})`,
            `- **Pending Issues:** (Global inflight belonged to a previous project rotation and was excluded to prevent cross-project file bleed)`,
            `- **Next Cycle Goal:** Execute ${phase || 'audit phase'} for \`${targetProject}\``
        ].join('\n');
        return {
            block: scopedBlock,
            focus: `Auditing ${targetProject}`,
            goal: `Execute ${phase || 'audit phase'} for ${targetProject}`
        };
    }

    return {
        block: raw,
        focus,
        goal
    };
}

function ensureHostWritable(targetPath) {
    try {
        if (typeof process.getuid === 'function' && process.getuid() === 0 && fs.existsSync(targetPath)) {
            fs.chownSync(targetPath, 1001, 1001);
        }
    } catch {}
}

async function persistInflight(reply, cycle, targetProject = null, { phase = null } = {}) {
    const match = reply.match(/### JEAN_INFLIGHT([\s\S]*)/);
    if (match?.[1]) {
        let inflightBody = match[1].trim();
        if (phase === '2_MASTER_BACKLOG' && isCleanPassBacklog(reply)) {
            inflightBody = inflightBody.replace(
                /(-\s*\*\*Pending Issues:?\*\*)[\s\S]*?(?=\n-\s*\*\*[A-Z]|\n---|$)/i,
                '$1 None — CLEAN_PASS verified in 2_MASTER_BACKLOG.'
            );
        }
        const content = `# 🚀 Jean SRE In-Flight State\n*Last Updated: ${new Date().toISOString()} (Cycle ${cycle})*\n\n${inflightBody}`;
        await fs.promises.writeFile('JEAN_SRE_INFLIGHT.md', content);
        ensureHostWritable('JEAN_SRE_INFLIGHT.md');
        if (targetProject) {
            try {
                const projInflight = join(resolveTargetDir(targetProject), 'JEAN_INFLIGHT.md');
                await fs.promises.writeFile(projInflight, content);
                ensureHostWritable(projInflight);
                console.log(`[State] Updated JEAN_INFLIGHT.md for project: ${targetProject}`);
            } catch (e) {
                console.error(`[State] Failed to write JEAN_INFLIGHT.md to project ${targetProject}:`, e.message);
            }
        }
        console.log(`[State] Updated JEAN_SRE_INFLIGHT.md locally`);
    }
}

function listProjectFiles(dirPath, baseDir = dirPath, depth = 0, maxDepth = 4, maxFiles = 70, acc = [], allowedExts = null) {
    if (depth > maxDepth || acc.length >= maxFiles || !fs.existsSync(dirPath)) return acc;
    const IGNORE = new Set([
        'node_modules', '.git', '.agent', 'dist', 'build', 'venv', '.venv',
        'env', 'mcp_env', '__pycache__', '.next', 'coverage', '.pytest_cache',
        '.ruff_cache', '.mypy_cache', 'htmlcov', 'eval_results', 'models', 'legacy',
        'JEAN_INFLIGHT.md', 'GEMINI_INFLIGHT.md', 'JEAN_SRE_INFLIGHT.md'
    ]);
    const PRIORITY_DIRS = new Set(['src', 'server', 'lib', 'app', 'bin', 'scripts', 'client', 'packages', 'tests', 'test']);
    try {
        const entries = fs.readdirSync(dirPath, { withFileTypes: true }).sort((a, b) => {
            const aPrio = a.isDirectory() && PRIORITY_DIRS.has(a.name) ? 0 : (a.isFile() ? 1 : 2);
            const bPrio = b.isDirectory() && PRIORITY_DIRS.has(b.name) ? 0 : (b.isFile() ? 1 : 2);
            return aPrio - bPrio || a.name.localeCompare(b.name);
        });
        for (const entry of entries) {
            if (acc.length >= maxFiles) break;
            if (IGNORE.has(entry.name) || /(?:^|_)env$|^venv/i.test(entry.name)) continue;
            const full = join(dirPath, entry.name);
            const rel = full.slice(baseDir.length + 1);
            if (entry.isDirectory()) {
                listProjectFiles(full, baseDir, depth + 1, maxDepth, maxFiles, acc, allowedExts);
            } else {
                if (allowedExts && !allowedExts.has(extname(entry.name).toLowerCase())) continue;
                acc.push(rel);
            }
        }
    } catch {}
    return acc;
}

function isPythonOnlyProject(projectPath) {
    if (!projectPath || !fs.existsSync(projectPath)) return false;
    const hasPyproject = fs.existsSync(join(projectPath, 'pyproject.toml')) || fs.existsSync(join(projectPath, 'requirements.txt'));
    const hasPkgJson = fs.existsSync(join(projectPath, 'package.json'));
    return hasPyproject && !hasPkgJson;
}

function loadHomelabBaselineRules(projectPath = null) {
    const pyOnly = isPythonOnlyProject(projectPath);
    const ruleNames = pyOnly
        ? ['project-architecture.md', 'lessons-learned.md']
        : ['javascript.md', 'project-architecture.md', 'lessons-learned.md'];
    const chunks = [];
    if (pyOnly) {
        chunks.push([
            `#### Baseline Rule Summary: Python 3.12 Homelab Standards (\`pyproject.toml\`)`,
            `- **Stack & Tooling**: Python 3.12 with \`pyproject.toml\`, \`ruff\` (linter/formatter), and \`pytest\` (test runner). Do NOT check for JavaScript-only configs (\`package.json\`, \`.eslintrc\`, \`@krusch/toolkit\`, \`async function main()\`).`,
            `- **Shared Library**: Python projects use \`krusch_toolkit\` (\`lib-py/\`) or FastAPI/Pydantic/SQLAlchemy/psycopg patterns as declared in \`pyproject.toml\`.`,
            `- **Reliability & Boundaries**: HTTP calls (\`requests\` / \`httpx\`) MUST specify explicit \`timeout=\`; SQL queries MUST be parameterized; required environment variables (\`DATABASE_URL\`, API URLs) MUST be validated with clear fallback or startup errors.`
        ].join('\n'));
    }
    for (const rname of ruleNames) {
        const candidates = [
            join(HOMELAB_ROOT, '.agent', 'rules', rname),
            join(projectRoot, 'rules', rname)
        ];
        const rpath = candidates.find(p => fs.existsSync(p));
        if (rpath) {
            try {
                const raw = fs.readFileSync(rpath, 'utf8')
                    .replace(/^---[\s\S]*?---\s*/, '')
                    .trim();
                if (raw.length > 0) {
                    chunks.push(`#### Baseline Rule File: \`.agent/rules/${basename(rpath)}\`\n${raw.slice(0, 1800)}`);
                }
            } catch {}
        }
    }
    return chunks.join('\n\n');
}

function loadJeanSkillPlaybook(mode = 'code_audit') {
    const skillMap = {
        code_audit: 'jean-mcp-code-audit',
        companion: 'jean-self-healing-code',
        infra: 'jean-self-healing-infra'
    };
    const skillName = skillMap[mode] || 'jean-mcp-code-audit';
    const candidates = [
        join(HOMELAB_ROOT, '.agent', 'skills', skillName, 'SKILL.md'),
        join(projectRoot, 'skills', skillName, 'SKILL.md')
    ];
    const spath = candidates.find(p => fs.existsSync(p));
    if (!spath) return '';
    try {
        const raw = fs.readFileSync(spath, 'utf8').replace(/^---[\s\S]*?---\s*/, '').trim();
        const playbookMatch = raw.match(/##\s*(?:🛠️|🔧)\s*Explicit MCP Tool[\s\S]*/i);
        return (playbookMatch ? playbookMatch[0] : raw).slice(0, 2200);
    } catch {
        return '';
    }
}

function buildProjectSpecAndDocsBrief(projectPath) {
    const parts = [];
    const pkgPath = join(projectPath, 'package.json');
    if (fs.existsSync(pkgPath)) {
        try {
            const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
            const scripts = Object.entries(pkg.scripts || {}).map(([k, v]) => `\`${k}\`: \`${v}\``).join(', ');
            const deps = Object.keys({ ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) }).slice(0, 18).join(', ');
            parts.push(`- **package.json (Node/JS/TS)**: name=\`${pkg.name || basename(projectPath)}\`, main=\`${pkg.main || 'index.js'}\` | scripts=[${scripts || 'none'}] | deps=[${deps || 'none'}]`);
        } catch {}
    }

    const pyprojPath = join(projectPath, 'pyproject.toml');
    if (fs.existsSync(pyprojPath)) {
        try {
            const pyRaw = fs.readFileSync(pyprojPath, 'utf8');
            const nameMatch = pyRaw.match(/^\s*name\s*=\s*"([^"]+)"/m);
            const scriptLines = (pyRaw.match(/\[project\.scripts\]([\s\S]*?)(?=\n\[|$)/)?.[1] || '')
                .split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#')).slice(0, 6).join(', ');
            const tools = [...pyRaw.matchAll(/^\[tool\.([a-zA-Z0-9_.-]+)\]/gm)].map(m => m[1]);
            parts.push(`- **pyproject.toml (Python 3.12 Project)**: name=\`${nameMatch?.[1] || basename(projectPath)}\` | scripts=[${scriptLines || 'none'}] | configured_tools=[${[...new Set(tools)].join(', ') || 'pytest, ruff'}]`);
            if (!fs.existsSync(pkgPath)) {
                parts.push(`- **Stack Guidance**: Pure Python repository (\`pyproject.toml\`). Evaluate against Python conventions (\`ruff\`, \`pytest\`, \`def main()\`, \`krusch_toolkit\`, FastAPI/Pydantic/SQLAlchemy); do NOT flag absence of JS-only files (\`package.json\`, \`.eslintrc\`, \`@krusch/toolkit\`).`);
            }
        } catch {}
    }

    const docNames = ['AGENTS.md', 'spec.md', 'SPEC.md', 'README.md'];
    for (const dname of docNames) {
        const dpath = join(projectPath, dname);
        if (fs.existsSync(dpath)) {
            try {
                const raw = fs.readFileSync(dpath, 'utf8').trim();
                const invLines = raw.split('\n').filter(l => /\bINV-[A-Z0-9_-]+|\bInvariant\b|\bNon-Negotiable\b|\bMUST\b/i.test(l)).slice(0, 8);
                const head = raw.split('\n').slice(0, 14).join('\n').slice(0, 600);
                parts.push(`- **\`${dname}\` Summary**:\n${head}${invLines.length ? `\n  **Declared Invariants/Rules in \`${dname}\`**:\n  ${invLines.map(l => l.trim()).join('\n  ')}` : ''}`);
                if (parts.join('\n').length > 1600) break;
            } catch {}
        }
    }
    return parts.length > 0
        ? `**Target Project Manifest, Specs & Declared Invariants (\`${projectPath}\`):**\n${parts.join('\n')}`
        : '';
}

const GENERIC_SYMBOL_STOPWORDS = new Set([
    'main', 'init', 'test', 'index', 'anonymous', 'query', 'pool', 'config',
    'server', 'client', 'error', 'status', 'result', 'search', 'create',
    'update', 'delete', 'fetch', 'parse', 'check', 'build', 'start', 'close',
    'write', 'read', 'list', 'sync', 'count', 'reset', 'sleep', 'route',
    'match', 'split', 'slice', 'filter', 'format', 'handle', 'process',
    'execute', 'resolve', 'connect', 'request', 'response', ' router', 'agent'
]);

function buildPendingIssueCodeEvidence(inflightBlock, projectPath, phase = null) {
    if (!projectPath || !fs.existsSync(projectPath)) return '';
    const { fileStats, allSymbols } = buildProjectAstIndex(projectPath);
    if (fileStats.length === 0) return '';

    const callerText = String(inflightBlock || '');
    const primaryParts = [callerText];
    if (phase !== '0_ORIENTATION') {
        const projInflightPath = join(projectPath, 'JEAN_INFLIGHT.md');
        if (fs.existsSync(projInflightPath)) {
            try {
                primaryParts.push(fs.readFileSync(projInflightPath, 'utf8'));
            } catch {}
        }
    }
    const checkpointParts = [];
    const auditDir = join(projectPath, '.agent', 'audit');
    // Only read checkpoints that belong to earlier phases of the CURRENT rotation
    const allowedCheckpoints = [];
    if (!phase || !['0_ORIENTATION', '1.1_STANDARD_MAPPING'].includes(phase)) {
        allowedCheckpoints.push('phase_1_investigation.md');
    }
    if (!phase || ['1.7_VERIFICATION_AUDIT', '1.8_RISK_MAP', '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'].includes(phase)) {
        allowedCheckpoints.push('phase_2_adversarial.md');
    }
    for (const fname of allowedCheckpoints) {
        const fpath = join(auditDir, fname);
        if (fs.existsSync(fpath)) {
            try {
                const raw = fs.readFileSync(fpath, 'utf8');
                checkpointParts.push(raw.slice(-2500));
            } catch {}
        }
    }

    const primaryText = primaryParts.join('\n\n');
    const text = [primaryText, ...checkpointParts].join('\n\n');
    if (!text.trim()) return '';

    const matchedSnippets = [];
    const seenKeys = new Set();

    // 1. Prioritize explicit file:line or file:start-end citations FIRST (up to 4 windows, callerText scanned first)
    const fileLineRegex = /([a-zA-Z0-9_./-]+\.(?:js|mjs|cjs|ts|tsx|jsx|py)):(\d+)(?:[-–](\d+))?/g;
    let m;
    while ((m = fileLineRegex.exec(text)) !== null && matchedSnippets.length < 5) {
        const citedPath = m[1].replace(/^\/+/, '');
        const citedLine = parseInt(m[2], 10) || 1;
        const citedEnd = m[3] ? parseInt(m[3], 10) : null;
        const f = fileStats.find(fsItem => fsItem.relPath === citedPath || fsItem.relPath.endsWith(`/${citedPath}`) || citedPath.endsWith(fsItem.relPath));
        if (!f) {
            const missingKey = `missing:${citedPath}`;
            if (!seenKeys.has(missingKey) && !fs.existsSync(join(projectPath, citedPath))) {
                seenKeys.add(missingKey);
                matchedSnippets.push(`- **❌ GROUND TRUTH REFUTATION — File \`${citedPath}\` DOES NOT EXIST in \`${basename(projectPath)}\`:** Any prior finding or pending issue citing \`${citedPath}:${citedLine}\` is cross-project bleed or a hallucinated path. You MUST DROP and REFUTE it immediately.`);
            }
            continue;
        }
        const sIdx = Math.max(0, citedLine - (citedEnd ? 3 : 12));
        const eIdx = citedEnd
            ? Math.min(f.loc, Math.max(citedEnd + 3, citedLine + 25), sIdx + 65)
            : Math.min(f.loc, citedLine + 25);
        const bucketKey = `line:${f.relPath}:${Math.floor(citedLine / 18)}`;
        if (seenKeys.has(bucketKey)) continue;
        seenKeys.add(bucketKey);
        const allLines = f.content.split('\n');
        const numbered = allLines.slice(sIdx, eIdx).map((l, idx) => `  ${sIdx + idx + 1}: ${l}`).join('\n');
        const label = citedEnd ? `${f.relPath}:${citedLine}-${citedEnd}` : `${f.relPath}:${citedLine}`;
        const lang = f.relPath.endsWith('.py') ? 'python' : 'javascript';
        const inDeps = f.dependents.map(d => d.sourcePath).slice(0, 5).join(', ') || 'none';
        matchedSnippets.push(`- **Live Source Window around \`${label}\` (lines ${sIdx + 1}-${eIdx} of ${f.loc}, AST \`imported_by: [${inDeps}]\` — GROUND TRUTH; if a prior claim about \`${label}\` is not supported by these exact lines, REFUTE and drop it):**\n\`\`\`${lang}\n${numbered}\n\`\`\``);
    }

    // 1.5. Also ground backticked file paths cited in primaryText (JEAN_INFLIGHT / callerText) without an explicit :line suffix
    const bareFileRegex = /`([a-zA-Z0-9_./-]+\.(?:js|mjs|cjs|ts|tsx|jsx|py|sh))`/g;
    while ((m = bareFileRegex.exec(primaryText)) !== null && matchedSnippets.length < 7) {
        const citedPath = m[1].replace(/^\/+/, '');
        const f = fileStats.find(fsItem => fsItem.relPath === citedPath || fsItem.relPath.endsWith(`/${citedPath}`) || citedPath.endsWith(fsItem.relPath));
        if (!f) continue;
        const bucketKey = `line:${f.relPath}:0`;
        if (seenKeys.has(bucketKey)) continue;
        seenKeys.add(bucketKey);
        const allLines = f.content.split('\n');
        const eIdx = Math.min(f.loc, 75);
        const numbered = allLines.slice(0, eIdx).map((l, idx) => `  ${idx + 1}: ${l}`).join('\n');
        const lang = f.relPath.endsWith('.py') ? 'python' : 'javascript';
        const inDeps = f.dependents.map(d => d.sourcePath).slice(0, 5).join(', ') || 'none';
        matchedSnippets.push(`- **Live Source Window for \`${f.relPath}\` (lines 1-${eIdx} of ${f.loc}, AST \`imported_by: [${inDeps}]\` — GROUND TRUTH; if a prior claim about \`${f.relPath}\` is not supported by these exact lines, REFUTE and drop it):**\n\`\`\`${lang}\n${numbered}\n\`\`\``);
    }

    // 2. Match backticked or non-generic declared project symbols (scan callerText first, then primaryText, then full text)
    for (const targetCorpus of [callerText, primaryText, text]) {
        for (const sym of allSymbols) {
            if (matchedSnippets.length >= 7) break;
            const name = sym.symbolName;
            if (!name || name.length < 5 || GENERIC_SYMBOL_STOPWORDS.has(name.toLowerCase())) continue;
            const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const isBackticked = new RegExp('`' + escaped + '(?:\\s*\\([^`]*\\))?`').test(targetCorpus);
            const isDistinctSymbol = /[a-z][A-Z]|_/.test(name) && new RegExp(`\\b${escaped}\\b`).test(targetCorpus);
            if (isBackticked || isDistinctSymbol) {
                const key = `sym:${sym.relPath}:${sym.startLine}`;
                if (seenKeys.has(key)) continue;
                seenKeys.add(key);
                const rawLines = String(sym.content || '').split('\n');
                const maxLines = Math.min(rawLines.length, 45);
                const numbered = rawLines.slice(0, maxLines).map((l, idx) => `  ${sym.startLine + idx}: ${l}`).join('\n');
                const cont = rawLines.length > maxLines ? `\n  ... [continues to line ${sym.endLine} (${rawLines.length} lines total)]` : '';
                const lang = sym.relPath.endsWith('.py') ? 'python' : 'javascript';
                matchedSnippets.push(`- **Live Source for Symbol \`${name}\` at \`${sym.relPath}:${sym.startLine}-${sym.endLine}\` (full ${rawLines.length}-line function):**\n\`\`\`${lang}\n${numbered}${cont}\n\`\`\``);
            }
        }
    }

    return matchedSnippets.join('\n\n');
}

function buildAstContextSection(projectPath, phase) {
    const { fileStats, allSymbols } = buildProjectAstIndex(projectPath);
    if (fileStats.length === 0) return '';

    const lines = [`**Tier 2 AST Symbol & Dependency Summary (\`krusch-git\` live index — ${fileStats.length} code files, ${allSymbols.length} symbols):**`];

    // Always show top files ranked by LOC & complexity (especially critical for 1.6_STRUCTURAL_STRESS_TEST)
    const byLoc = [...fileStats].sort((a, b) => b.loc - a.loc).slice(0, 8);
    lines.push(`- **Top Files by LOC & Coupling:**`);
    for (const f of byLoc) {
        const outImps = f.imports.map(i => i.targetPath).slice(0, 5).join(', ') || 'none';
        const inDeps = f.dependents.map(d => d.sourcePath).slice(0, 5).join(', ') || 'none';
        lines.push(`  - \`${f.relPath}\` (${f.loc} LOC, ${f.symbolCount} symbols) | imports: [${outImps}] | imported_by: [${inDeps}]`);
    }

    if (['0_ORIENTATION', '1.3_CRITICAL_PATH', '1.4_BOUNDARY_AUDIT'].includes(phase)) {
        const keySymbols = allSymbols.slice(0, 18);
        if (keySymbols.length > 0) {
            lines.push(`- **Key Declared Symbols & Boundary Interfaces:**`);
            for (const s of keySymbols) {
                lines.push(`  - [${s.symbolType}] \`${s.symbolName}\` (\`${s.relPath}:${s.startLine}-${s.endLine}\`) — \`${s.signature || s.symbolName}\``);
            }
        }
    }

    if (['0_ORIENTATION', '1.5_ANTI_PATTERN', '1.6_STRUCTURAL_STRESS_TEST', '2_MASTER_BACKLOG', 'SYNTHESIS'].includes(phase)) {
        const unimported = fileStats.filter(f =>
            f.dependents.length === 0 &&
            !/(?:^|\/)(?:test|tests|__tests__)\/|\.(?:test|spec)\.|(?:^|\/)test_[^/]+\.py$|_test\.py$|(?:^|\/)__init__\.py$/i.test(f.relPath)
        );
        if (unimported.length > 0) {
            lines.push(`- **Unimported / Standalone Candidate Files (\`imported_by: [none]\` in AST graph — verify if CLI/server entrypoint or stale/dead script for \`STALE-01\`):**`);
            for (const u of unimported.slice(0, 12)) {
                const firstLine = (u.content.split('\n').find(l => l.trim() && !l.trim().startsWith('#!')) || '').trim().slice(0, 90);
                lines.push(`  - \`${u.relPath}\` (${u.loc} LOC, ${u.symbolCount} symbols, \`imported_by: [none]\`) — \`${firstLine}\``);
            }
        }
    }

    return lines.join('\n');
}

const _projectTestSuiteCache = new Map();
const TEST_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes per project rotation

function buildCleanChildProjectEnv(projectPath, binPaths, isTestCmd = true) {
    try {
        if (fs.existsSync('/app/lib/node_modules') && fs.existsSync(HOMELAB_ROOT)) {
            const rootNm = join(HOMELAB_ROOT, 'node_modules');
            if (!fs.existsSync(rootNm)) {
                if (fs.existsSync('/app/node_modules/@krusch') && !fs.existsSync('/app/lib/node_modules/@krusch')) {
                    try { fs.symlinkSync('/app/node_modules/@krusch', '/app/lib/node_modules/@krusch'); } catch {}
                }
                try { fs.symlinkSync('/app/lib/node_modules', rootNm); } catch {}
            }
            const liveCascade = join(HOMELAB_ROOT, 'projects', 'krusch-cascade-router');
            const libCascade = '/app/lib/node_modules/krusch-cascade-router';
            if (fs.existsSync(liveCascade) && (!fs.existsSync(libCascade) || !fs.lstatSync(libCascade).isSymbolicLink())) {
                try {
                    fs.rmSync(libCascade, { recursive: true, force: true });
                    fs.symlinkSync(liveCascade, libCascade);
                } catch {}
            }
            const rootLib = join(HOMELAB_ROOT, 'lib');
            if (fs.existsSync('/app/lib') && !fs.existsSync(rootLib)) {
                try { fs.symlinkSync('/app/lib', rootLib); } catch {}
            }
        }
    } catch {}
    const childEnv = { ...process.env };
    if (projectPath && basename(projectPath) !== 'jean-sre') {
        const JEAN_SPECIFIC_ENV_KEYS = [
            'DATABASE_URL', 'DBOS_DATABASE_URL', 'PG_CONNECTION_STRING',
            'PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE',
            'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
            'PORT', 'JEAN_API_KEY',
            'PERSONA_OLLAMA_URL', 'PERSONA_MODEL', 'EXECUTOR_OLLAMA_URL', 'EXECUTOR_MODEL'
        ];
        for (const key of JEAN_SPECIFIC_ENV_KEYS) {
            delete childEnv[key];
        }
    }
    return {
        ...childEnv,
        PATH: binPaths,
        CI: 'true',
        NODE_ENV: isTestCmd ? 'test' : (childEnv.NODE_ENV || 'development'),
        MOCK_LLM: '1',
        MOCK_EMBEDDINGS: '1',
        KRUSCH_EPHEMERAL: '1',
        PYTHONPATH: [
            'src',
            '.',
            join(HOMELAB_ROOT, 'lib-py'),
            join(HOMELAB_ROOT, 'projects', 'krusch-nexus', 'src'),
            join(HOMELAB_ROOT, 'projects', 'krusch-fs', 'src'),
            join(HOMELAB_ROOT, 'projects', 'krusch-authority-packs', 'src'),
            childEnv.PYTHONPATH || ''
        ].filter(Boolean).join(':')
    };
}

function runProjectVerificationSuite(projectPath, { force = false } = {}) {
    if (!projectPath || !fs.existsSync(projectPath)) {
        return { status: 'UNAVAILABLE', command: null, testFiles: [], output: '', report: '' };
    }
    if (process.env.SKIP_LIVE_TEST_EXECUTION === '1') {
        return { status: 'SKIPPED', command: null, testFiles: [], output: '', report: '' };
    }

    const now = Date.now();
    const cached = _projectTestSuiteCache.get(projectPath);
    if (!force && cached && (now - cached.timestamp) < TEST_CACHE_TTL_MS) {
        return cached;
    }

    const projName = basename(projectPath);
    const allFiles = listProjectFiles(projectPath, projectPath, 0, 4, 150);
    const testFiles = allFiles.filter(f =>
        /(?:^|\/)(?:test|tests|__tests__)\/.*\.(?:js|mjs|cjs|ts|tsx|jsx|py)$/i.test(f) ||
        /\.(?:test|spec)\.(?:js|mjs|cjs|ts|tsx|jsx)$/i.test(f) ||
        /(?:^|\/)test_[^/]+\.py$|_test\.py$/i.test(f)
    );

    let testCmd = null;
    const pkgPath = join(projectPath, 'package.json');
    if (fs.existsSync(pkgPath)) {
        try {
            const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
            const rawScript = String(pkg.scripts?.test || '').trim();
            if (rawScript && !rawScript.includes('no test specified')) {
                if (/^node\s+--test\b/.test(rawScript) && !rawScript.includes('--test-force-exit')) {
                    testCmd = rawScript.replace(/^node\s+--test\b/, 'node --test --test-force-exit');
                } else {
                    testCmd = 'npm test';
                }
            }
        } catch {}
    }

    if (!testCmd && testFiles.some(f => f.endsWith('.py'))) {
        const venvCandidates = [
            join(projectPath, '.venv', 'bin', 'pytest'),
            join(projectPath, 'venv', 'bin', 'pytest'),
            join(projectPath, 'mcp_env', 'bin', 'pytest'),
            '/opt/venv/bin/pytest',
            join(HOMELAB_ROOT, '.venv', 'bin', 'pytest'),
            join(HOMELAB_ROOT, 'lib-py', '.venv', 'bin', 'pytest'),
            join(HOMELAB_ROOT, 'projects', 'krusch-law', '.venv', 'bin', 'pytest'),
            join(HOMELAB_ROOT, 'projects', 'krusch-nexus', '.venv', 'bin', 'pytest'),
            '/app/lib-py/.venv/bin/pytest'
        ];
        const isRunnablePytest = (candidatePath) => {
            if (!fs.existsSync(candidatePath)) return false;
            try {
                const firstLine = fs.readFileSync(candidatePath, 'utf8').split('\n')[0] || '';
                if (firstLine.startsWith('#!')) {
                    const interp = firstLine.slice(2).trim().split(/\s+/)[0];
                    if (interp && !interp.startsWith('/usr/bin/env') && !fs.existsSync(interp)) {
                        return false;
                    }
                }
                execSync(`"${candidatePath}" --version`, { stdio: 'ignore', timeout: 5000 });
                return true;
            } catch {
                return false;
            }
        };
        const resolvedPytest = venvCandidates.find(isRunnablePytest) || 'python3 -m pytest';
        testCmd = `PYTHONPATH=.:src ${resolvedPytest} -q`;
    }

    if (!testCmd && testFiles.some(f => /\.(?:js|mjs|cjs)$/.test(f))) {
        testCmd = 'node --test --test-force-exit';
    }

    if (!testCmd) {
        const noTestsEntry = {
            timestamp: now,
            status: 'NO_TESTS_CONFIGURED',
            command: null,
            testFiles,
            output: '',
            report: `**🧪 Live Project Test Suite Execution (\`${projName}\` — GROUND TRUTH):**\n- **Status:** ⚠️ NO_TESTS_CONFIGURED\n- **Discovered Test Files (${testFiles.length}):** ${testFiles.length ? testFiles.map(f => `\`${f}\``).join(', ') : '(none)'}`
        };
        _projectTestSuiteCache.set(projectPath, noTestsEntry);
        return noTestsEntry;
    }

    const binPaths = [
        '/opt/venv/bin',
        join(projectPath, 'node_modules', '.bin'),
        join(HOMELAB_ROOT, 'node_modules', '.bin'),
        process.env.PATH || ''
    ].filter(Boolean).join(':');

    const startMs = Date.now();
    let status = 'PASS';
    let exitCode = 0;
    let rawOutput = '';
    try {
        console.log(`[Verify] 🧪 Running live test suite for ${projName}: \`${testCmd}\`...`);
        const stdout = execSync(testCmd, {
            cwd: projectPath,
            encoding: 'utf8',
            timeout: 90000,
            maxBuffer: 4 * 1024 * 1024,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: buildCleanChildProjectEnv(projectPath, binPaths, true)
        });
        rawOutput = String(stdout || '').trim();
    } catch (err) {
        status = 'FAIL';
        exitCode = err.status ?? 1;
        rawOutput = [err.stdout, err.stderr, err.message].filter(Boolean).join('\n').trim();
    }
    const durationMs = Date.now() - startMs;
    console.log(`[Verify] ${status === 'PASS' ? '✅' : '❌'} Test suite for ${projName} finished with ${status} (exit=${exitCode}, ${durationMs}ms)`);

    const trimmedOut = rawOutput.length > 2400
        ? `...[truncated ${rawOutput.length - 2400} chars]...\n${rawOutput.slice(-2400)}`
        : (rawOutput || '(no output)');

    const report = [
        `**🧪 Live Project Test Suite Execution (\`${projName}\` — GROUND TRUTH):**`,
        `- **Status:** ${status === 'PASS' ? `✅ PASS (exit code 0 in ${durationMs}ms)` : `❌ FAIL (exit code ${exitCode} in ${durationMs}ms)`}`,
        `- **Command Executed:** \`${testCmd}\` (in \`${projectPath}\`)`,
        `- **Discovered Test Files (${testFiles.length}):** ${testFiles.length ? testFiles.slice(0, 25).map(f => `\`${f}\``).join(', ') : '(none)'}`,
        `- **Live Test Runner Output:**\n\`\`\`text\n${trimmedOut}\n\`\`\``
    ].join('\n');

    const entry = {
        timestamp: now,
        status,
        exitCode,
        durationMs,
        command: testCmd,
        testFiles,
        output: rawOutput,
        report
    };
    _projectTestSuiteCache.set(projectPath, entry);
    return entry;
}

/**
 * Distill verified 120B Teacher audit checkpoints into a compact JSON artifact
 * (`<project>/.agent/audit/teacher_120b_distilled.json`) so the 30B Student model
 * (`qwen3-coder-30b-moe` with 6K context) inherits:
 *   1. Refuted false-positive traps (`4. NOT FLAGGED` / `REFUTED`) so it never re-chases dead ends.
 *   2. Verified backlog findings (`[BUG-XX]` / `[ARCH-XX]`) anchored to exact `file:line`.
 *   3. High-yield L2 tool trajectories (`read_file`, `ast_graph`, `search_code`, `krusch_git_*`).
 */
function distillTeacherKnowledgeForStudent(projectPath, { teacherModel = 'gpt-oss:120b', extraToolCommands = [] } = {}) {
    if (!projectPath || !fs.existsSync(projectPath)) return null;
    const auditDir = join(projectPath, '.agent', 'audit');
    if (!fs.existsSync(auditDir)) return null;

    const project = basename(projectPath);
    const outPath = join(auditDir, 'teacher_120b_distilled.json');
    let existing = {};
    if (fs.existsSync(outPath)) {
        try {
            existing = JSON.parse(fs.readFileSync(outPath, 'utf8')) || {};
        } catch {}
    }

    const readIfExists = (fname) => {
        const p = join(auditDir, fname);
        return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
    };

    const p0 = readIfExists('phase_0_orientation.md');
    const p1 = readIfExists('phase_1_investigation.md');
    const p2 = readIfExists('phase_2_adversarial.md');
    const mb = readIfExists('master_backlog.md');
    const combined = [p0, p1, p2, mb].filter(Boolean).join('\n\n');
    if (!combined.trim()) return null;

    // 1. Extract "NOT FLAGGED" / Refuted false-positive traps & verified compliant invariants
    const refutedSet = new Set(Array.isArray(existing.refutedFalsePositives) ? existing.refutedFalsePositives.slice(0, 4) : []);
    const notFlaggedRegex = /(?:4\.\s*NOT FLAGGED|###\s*(?:4\.\s*)?NOT FLAGGED|\*\*NOT FLAGGED[^*\n]*\*\*)([\s\S]*?)(?=\n(?:#{2,4}\s|\d+\.\s+[A-Z]|---)|$)/gi;
    let m;
    while ((m = notFlaggedRegex.exec(combined)) !== null) {
        const block = m[1] || '';
        for (const rawLine of block.split('\n')) {
            const line = rawLine.replace(/^[\s*-]+/, '').trim();
            if (line.length >= 20 && line.length <= 220 && !/^none\b/i.test(line)) {
                refutedSet.add(line);
                if (refutedSet.size >= 8) break;
            }
        }
        if (refutedSet.size >= 8) break;
    }

    // Also capture explicit HYPOTHESIS VERDICT refutations or verified compliance lines
    const verdictRegex = /###\s*HYPOTHESIS VERDICT\s*\n+([^\n#]+)/gi;
    while ((m = verdictRegex.exec(combined)) !== null && refutedSet.size < 8) {
        const line = m[1].replace(/^[\s*-]+/, '').trim();
        if (line.length >= 25 && /\b(?:refuted|false positive|already compliant|confirmed compliant|passes)\b/i.test(line)) {
            refutedSet.add(line.slice(0, 200));
        }
    }

    // 2. Extract 120B Teacher LESSON blocks (architectural insights for the 30B Student)
    const lessonSet = new Set();
    const lessonRegex = /###\s*LESSON[^\n]*\n+([\s\S]*?)(?=\n###|\n---|$)/gi;
    while ((m = lessonRegex.exec(combined)) !== null && lessonSet.size < 4) {
        const firstSentence = (m[1] || '').trim().replace(/\s+/g, ' ').split(/(?<=[.!?])\s+/)[0] || '';
        if (firstSentence.length >= 30 && firstSentence.length <= 220 && !firstSentence.startsWith('[')) {
            lessonSet.add(firstSentence);
        }
    }

    // 3. Extract verified Master Backlog cards
    const parsedCards = parseMasterBacklogCards(mb).slice(0, 6).map(c => ({
        id: c.id,
        severity: c.severity,
        effort: c.effort,
        evidence: c.evidence,
        title: c.title,
        fixSummary: (c.fixSummary || '').slice(0, 140)
    }));
    const cards = parsedCards.length > 0
        ? parsedCards
        : (isCleanPassBacklog(mb) ? [] : (Array.isArray(existing.verifiedBacklog) ? existing.verifiedBacklog : []));

    // 4. Extract high-yield novel L2 tool trajectories
    const toolSet = new Set([
        ...extraToolCommands.filter(Boolean).map(c => String(c).trim().slice(0, 110)),
        ...(Array.isArray(existing.goldenToolTrajectories) ? existing.goldenToolTrajectories.slice(0, 4) : [])
    ]);
    const toolLineRegex = /\b((?:read_file|ast_graph|search_code|dependency_graph|search_symbols|krusch_git_[a-z_]+|krusch_context_[a-z_]+|krusch_fs_[a-z_]+)\s+[^\n*`]{3,110})/gi;
    while ((m = toolLineRegex.exec(combined)) !== null && toolSet.size < 6) {
        const cleaned = m[1].replace(/\*\*Reason\*\*.*$/i, '').replace(/`+/g, '').trim();
        if (cleaned.length >= 8 && cleaned.length <= 110) {
            toolSet.add(cleaned);
        }
    }

    const distilled = {
        project,
        teacherModel,
        distilledAt: new Date().toISOString(),
        studentTarget: 'qwen3-coder-30b-moe:latest',
        refutedFalsePositives: [...refutedSet].slice(0, 8),
        lessonsLearned: [...lessonSet].slice(0, 4),
        verifiedBacklog: cards,
        goldenToolTrajectories: [...toolSet].slice(0, 6)
    };

    try {
        fs.writeFileSync(outPath, JSON.stringify(distilled, null, 2), 'utf8');
        ensureHostWritable(outPath);
    } catch (e) {
        console.warn(`[Distill] Could not write teacher_120b_distilled.json for ${project}: ${e.message}`);
    }

    return distilled;
}

function loadDistilledTeacherBrief(projectPath, maxChars = 850) {
    if (!projectPath || !fs.existsSync(projectPath)) return '';
    const distPath = join(projectPath, '.agent', 'audit', 'teacher_120b_distilled.json');
    if (!fs.existsSync(distPath)) return '';
    try {
        const d = JSON.parse(fs.readFileSync(distPath, 'utf8'));
        const lines = [`**🎓 Distilled 120B Teacher Prior (\`${d.teacherModel || 'gpt-oss:120b'}\` → 30B Student Guide):**`];
        if (Array.isArray(d.refutedFalsePositives) && d.refutedFalsePositives.length > 0) {
            lines.push(`- **Verified Safe / Refuted False-Positives (DO NOT RE-FLAG):**`);
            for (const r of d.refutedFalsePositives.slice(0, 3)) {
                lines.push(`  - ${r.slice(0, 130)}`);
            }
        }
        if (Array.isArray(d.verifiedBacklog) && d.verifiedBacklog.length > 0) {
            lines.push(`- **Teacher-Verified Backlog Anchors:**`);
            for (const c of d.verifiedBacklog.slice(0, 3)) {
                lines.push(`  - [${c.id} | ${c.severity}] \`${c.evidence}\` — ${c.title.slice(0, 90)}`);
            }
        }
        if (Array.isArray(d.lessonsLearned) && d.lessonsLearned.length > 0) {
            lines.push(`- **Teacher Architectural Lesson:** ${d.lessonsLearned[0].slice(0, 160)}`);
        }
        if (Array.isArray(d.goldenToolTrajectories) && d.goldenToolTrajectories.length > 0) {
            lines.push(`- **High-Yield Repo Tool Queries:** ${d.goldenToolTrajectories.slice(0, 3).map(t => `\`${t}\``).join(', ')}`);
        }
        return lines.join('\n').slice(0, maxChars);
    } catch {
        return '';
    }
}

function buildLegoScoutContextBrief(homelabRoot = HOMELAB_ROOT, options = {}) {
    try {
        const projectsDir = options.projectsRoot || join(homelabRoot, 'projects');
        if (!fs.existsSync(projectsDir)) return '';
        const res = scoutHomelabLegoCandidates(projectsDir, {
            minProjects: options.minProjects ?? 3,
            portablePaths: true
        });
        if (!res?.ok || !res?.data) return '';
        const d = res.data;
        const lines = [
            `**🧱 Cross-Repository Lego Scout Ground Truth (\`krusch-lego scout --portable\` — Rule-of-${d.ruleOfThreeThreshold}, gated by \`/jean-loop-gate\`):**`,
            `- **Scanned:** \`${d.scannedProjects}\` projects (\`${d.scannedFilesTotal}\` production source files)`,
            `- **1. Immediate Toolkit Adoption Targets (\`LEGO-ADOPT\`):** \`${d.adoptionFindings.length}\` project(s)`
        ];
        for (const f of d.adoptionFindings.slice(0, 6)) {
            lines.push(
                `  - \`${f.project}\` (${f.findingCount} hit(s)): [\`${f.rulesTriggered.join('`, `')}\`] → \`${f.recommendedBricks.join('`, `')}\` (\`${f.sampleFiles[0] || f.project}\`)`
            );
        }
        const covered = d.candidateClusters.filter((c) => c.status === 'ALREADY_COVERED_IN_TOOLKIT');
        if (covered.length > 0) {
            lines.push(`- **2. Covered Wave-3 Toolkit Patterns (\`ALREADY_COVERED_IN_TOOLKIT\`):**`);
            for (const c of covered) {
                lines.push(
                    `  - \`${c.id}\` (\`${c.projectCount}\` repos) → \`${c.proposedBrickImport}\` | repos: ${c.projects.slice(0, 5).join(', ')}`
                );
            }
        }
        const candidates = d.candidateClusters.filter((c) => c.status === 'CANDIDATE_MEETS_RULE_OF_3');
        if (candidates.length > 0) {
            lines.push(
                `- **3. Candidate Bricks Meeting Rule-of-${d.ruleOfThreeThreshold} (\`CANDIDATE_MEETS_RULE_OF_3\` — propose spec under \`/jean-loop-gate\`):**`
            );
            for (const c of candidates) {
                lines.push(
                    `  - \`${c.id}\` (\`${c.projectCount}\` repos >= ${d.ruleOfThreeThreshold}) → \`${c.proposedBrickImport}\` | repos: ${c.projects.slice(0, 5).join(', ')} | sample: \`${c.sampleEvidence[0] || ''}\``
                );
            }
        }
        const rejected = d.candidateClusters.filter((c) => c.status === 'BELOW_THRESHOLD_DO_NOT_EXTRACT');
        if (rejected.length > 0) {
            lines.push(
                `- **4. Anti-Bloat Rejected Patterns (\`BELOW_THRESHOLD_DO_NOT_EXTRACT\` — \`< ${d.ruleOfThreeThreshold}\` repos, DO NOT EXTRACT):**`
            );
            for (const c of rejected) {
                lines.push(
                    `  - \`${c.id}\` (\`${c.projectCount}\` repo(s) < ${d.ruleOfThreeThreshold}) → \`${c.proposedBrickImport}\` (repos: ${c.projects.join(', ')}) — keep project-local.`
                );
            }
        }
        return lines.join('\n').slice(0, options.maxChars ?? 4200);
    } catch {
        return '';
    }
}

function buildProjectLegoBoilerplateBrief(projectPath, maxFiles = 24) {
    try {
        if (!projectPath || !fs.existsSync(projectPath)) return '';
        const base = basename(projectPath);
        if (['lib', 'lib-py', 'krusch-stack', '.agent', 'configs', 'scripts'].includes(base)) return '';
        const candidates = listProjectFiles(projectPath)
            .filter(
                (rel) =>
                    /\.(?:js|mjs|ts|tsx|jsx|py)$/i.test(rel) &&
                    !/(?:^|\/)(?:test|tests|__tests__|docs|benchmarks|examples|scratch|archive)\b|\.(?:test|spec)\./i.test(rel)
            )
            .slice(0, maxFiles);
        const hits = [];
        for (const rel of candidates) {
            if (hits.length >= 6) break;
            const abs = join(projectPath, rel);
            let stat;
            try {
                stat = fs.statSync(abs);
            } catch {
                continue;
            }
            if (!stat.isFile() || stat.size > 196_608) continue;
            const code = fs.readFileSync(abs, 'utf8');
            const audit = auditBoilerplateText(code, rel);
            for (const finding of audit?.data?.findings || []) {
                hits.push(`- \`${rel}\`: \`${finding.ruleId}\` → migrate to \`${finding.importPath || finding.brickId}\` (${finding.message})`);
                if (hits.length >= 6) break;
            }
        }
        if (hits.length === 0) return '';
        return [
            `**🧱 @krusch/toolkit Boilerplate Audit (\`INV-LEGO-01\` — deterministic static hits):**`,
            ...hits
        ].join('\n');
    } catch {
        return '';
    }
}

async function appendLegoScoutBacklogEntry(cleanupPath, infraTarget, reply, nowIso = new Date().toISOString()) {
    const cleanReply = String(reply || '').trim();
    if (!cleanReply || /^NO_FINDINGS$/i.test(cleanReply)) return false;
    const header = !fs.existsSync(cleanupPath)
        ? `# 🧹 Jean SRE — Homelab Staleness & Cleanup Backlog\n> Auto-compiled from whole-homelab audits (\`@homelab/*\` & \`projects/*\`).\n\n`
        : '';
    const entry = [
        `## 🧱 Lego Scout (${infraTarget?.node || 'kruschdev'}/${infraTarget?.focus || 'lego_scout'}) — ${nowIso}`,
        `> **Governance:** Gated via \`/jean-loop-gate\` & \`/spec-first-refactor\` (Rule-of-3 anti-bloat threshold enforced).`,
        ``,
        cleanReply,
        ``,
        `---`,
        ``
    ].join('\n');
    await fs.promises.appendFile(cleanupPath, `${header}${entry}`, 'utf8');
    ensureHostWritable(cleanupPath);
    return true;
}

function buildProjectAuditContext(projectPath, phase, inflightBlock = '', options = {}) {
    const isColabBurst = Boolean(options.isColabBurst);
    const sections = [];
    const files = listProjectFiles(projectPath);
    if (files.length > 0) {
        const maxTreeFiles = isColabBurst ? 90 : 50;
        const fileList = files.slice(0, maxTreeFiles).join(', ');
        sections.push(`**Actual Project File Tree (\`${projectPath}\` — relative paths; DO NOT guess paths outside this list):**\n\`${fileList}\`${files.length > maxTreeFiles ? ` *(+${files.length - maxTreeFiles} more)*` : ''}`);
    }

    // Inject Target Project Manifest, Specs & Declared Invariants in every phase
    const specBrief = buildProjectSpecAndDocsBrief(projectPath);
    if (specBrief) {
        sections.push(specBrief.slice(0, isColabBurst ? 2600 : 1800));
    }

    // Inject Tier 2 AST Symbol & Dependency Graph summary
    const astSummary = buildAstContextSection(projectPath, phase);
    if (astSummary) {
        sections.push(astSummary.slice(0, isColabBurst ? 2800 : 2000));
    }

    // When running on the 30B local Student (!isColabBurst), inject the 120B Teacher Distilled Prior
    const teacherBrief = !isColabBurst ? loadDistilledTeacherBrief(projectPath, 1200) : '';
    if (teacherBrief) {
        sections.push(teacherBrief);
    }

    // Inject Live Project Test Suite Execution Report (runs in all phases during G4 Burst, and in Orientation/Verification/Risk/Adversarial/Backlog phases locally)
    const shouldIncludeLiveTests = isColabBurst || ['0_ORIENTATION', '1.7_VERIFICATION_AUDIT', '1.8_RISK_MAP', '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'].includes(phase);
    if (shouldIncludeLiveTests) {
        const testSuiteResult = runProjectVerificationSuite(projectPath, { force: phase === '1.7_VERIFICATION_AUDIT' });
        if (testSuiteResult?.report) {
            sections.push(testSuiteResult.report.slice(0, isColabBurst ? 3600 : 2800));
        }
    }

    // Inject Jean MCP Skill Playbook for grounded 3-Tier tool selection
    const skillPlaybook = loadJeanSkillPlaybook('code_audit');
    if (skillPlaybook) {
        sections.push(`**Active Skill Playbook (\`/jean-mcp-code-audit\`):**\n${skillPlaybook.slice(0, isColabBurst ? 1800 : 1400)}`);
    }

    // Inject Homelab Baseline Engineering Rubric (.agent/rules/) per /code-audit fallback & companion standards
    if (['0_ORIENTATION', '1.1_STANDARD_MAPPING', '1.2_COMPLIANCE_MATRIX', '1.5_ANTI_PATTERN'].includes(phase)) {
        const baselineRules = loadHomelabBaselineRules(projectPath);
        if (baselineRules) {
            sections.push(`**Homelab Baseline Engineering Rubric (\`.agent/rules/\` — evaluate project compliance alongside local project docs):**\n${baselineRules.slice(0, isColabBurst ? 2400 : 1800)}`);
        }
    }

    // Inject deterministic @krusch/toolkit boilerplate hints in anti-pattern & synthesis phases
    if (['1.2_COMPLIANCE_MATRIX', '1.5_ANTI_PATTERN', '2_MASTER_BACKLOG'].includes(phase)) {
        const legoBrief = buildProjectLegoBoilerplateBrief(projectPath);
        if (legoBrief) {
            sections.push(legoBrief);
        }
    }

    const auditDir = join(projectPath, '.agent', 'audit');
    if (fs.existsSync(auditDir) && phase !== '0_ORIENTATION') {
        const isDeepSynthesis = ['1.8_RISK_MAP', '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'].includes(phase);
        const priorFiles = [['phase_0_orientation.md', isColabBurst ? 1800 : 1200]];
        if (phase !== '1.1_STANDARD_MAPPING') {
            priorFiles.push(['phase_1_investigation.md', isColabBurst ? 4000 : (isDeepSynthesis ? 2800 : 1800)]);
        }
        if (['1.7_VERIFICATION_AUDIT', '1.8_RISK_MAP', '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'].includes(phase)) {
            priorFiles.push(['phase_2_adversarial.md', isColabBurst ? 4000 : (isDeepSynthesis ? 2800 : 1800)]);
        }
        for (const [fname, maxChars] of priorFiles) {
            const fpath = join(auditDir, fname);
            if (fs.existsSync(fpath)) {
                try {
                    const text = fs.readFileSync(fpath, 'utf8').trim();
                    if (text.length > 0) {
                        const snippet = text.length > maxChars ? `...[truncated]...\n${text.slice(-maxChars)}` : text;
                        sections.push(`**Accumulated Checkpoint (\`${fname}\`):**\n${snippet}`);
                    }
                } catch {}
            }
        }
    }

    // Place Live Source Verification AFTER Accumulated Checkpoints so Ground Truth overrides any prior summary
    const pendingEvidence = buildPendingIssueCodeEvidence(inflightBlock, projectPath, phase);
    if (pendingEvidence) {
        sections.push(`**🔍 Live Source Verification for Mentioned Symbols/Files in Inflight (GROUND TRUTH — overrides any prior summary):**\n${pendingEvidence.slice(0, isColabBurst ? 8500 : 6500)}`);
    }

    return sections.join('\n\n');
}

function extractLastSummaryTableSection(backlogText) {
    const text = String(backlogText || '');
    const re = /(?:^|\n)###\s*1\.\s*SUMMARY\s*TABLE\b([\s\S]*?)(?=\n###\s+|\n##\s+|$)/gi;
    let lastBody = null;
    let m;
    while ((m = re.exec(text)) !== null) {
        lastBody = m[1] || '';
    }
    return lastBody;
}

function stripNonDefectBacklogSections(backlogText) {
    return String(backlogText || '').replace(
        /(?:^|\n)###\s*(?:4\.\s*NOT\s+FLAGGED|NOT\s+FLAGGED|5\.\s*STALENESS[^\n]*|STALENESS\s*&\s*ARCHIVAL\s*VERDICT|LESSON\b[^\n]*|JEAN_INFLIGHT\b[^\n]*)([\s\S]*?)(?=\n###\s+|\n##\s+|\n---|$)/gi,
        '\n'
    );
}

function isCleanPassBacklog(backlogText) {
    const summarySection = extractLastSummaryTableSection(backlogText);
    if (summarySection === null) return false;
    const rowRegex = /\|\s*[*`[]*([A-Z]{2,8}-\d{1,3})[\]*`]*\s*\|\s*([^|]+)\|\s*[*`]*?(P[0-3])[*`]*\s*\|\s*[*`]*([SMLX]{1,2})[*`]*\s*\|/i;
    if (rowRegex.test(summarySection)) return false;
    return /\|\s*[*`]*CLEAN_PASS[*`]*\s*\|/i.test(summarySection)
        || /\|\s*[*`]*NONE[*`]*\s*\|/i.test(summarySection)
        || /\b(?:zero|no)\s+actionable\s+defects\b/i.test(summarySection);
}

function extractStaleCleanupCards(backlogText) {
    const staleMap = new Map();
    const text = String(backlogText || '');
    let m;
    const rowRegex = /\|\s*[*`[]*(STALE-\d{1,3})[\]*`]*\s*\|\s*([^|]+)\|\s*[*`]*?(P[0-3])[*`]*\s*\|\s*[*`]*([SMLX]{1,2})[*`]*\s*\|/gi;
    while ((m = rowRegex.exec(text)) !== null) {
        const id = m[1].trim().toUpperCase();
        staleMap.set(id, {
            id,
            title: m[2].replace(/^\*\*|\*\*$/g, '').trim(),
            severity: m[3].trim().toUpperCase(),
            effort: m[4].trim().toUpperCase(),
            evidence: 'project-wide',
            fixSummary: ''
        });
    }
    const headingRegex = /(?:^|\n)#{3,4}[ \t]+\[?(STALE-\d{1,3})\]?(?:[: \t—-]+([^\n]*))?([\s\S]*?)(?=(?:\n#{2,4}[ \t]+)|$)/gi;
    while ((m = headingRegex.exec(text)) !== null) {
        const id = m[1].trim().toUpperCase();
        const existing = staleMap.get(id) || {};
        const headingTitle = (m[2] || '').trim();
        const body = m[3] || '';
        const titleMatch = body.match(/\*\*Title:?\*\*\s*([^\n]+)/i);
        const sevMatch = body.match(/\*\*Severity:?\*\*\s*(P[0-3])/i);
        const effMatch = body.match(/\*\*Effort:?\*\*\s*([SMLX]{1,2})/i);
        const evMatch = body.match(/\*\*(?:Evidence|File):?\*\*\s*`?([^`\n]+)`?/i);
        const fixMatch = body.match(/\*\*(?:Fix Instructions|Fix|Execution Plan|Why It Matters):?\*\*\s*([\s\S]*?)(?=\n[-*]*\s*\*\*[A-Z]|\n---|$)/i);
        staleMap.set(id, {
            id,
            title: (titleMatch ? titleMatch[1] : (headingTitle || existing.title || id)).replace(/^\*\*|\*\*$/g, '').trim() || id,
            severity: (sevMatch ? sevMatch[1] : (existing.severity || 'P2')).toUpperCase(),
            effort: (effMatch ? effMatch[1] : (existing.effort || 'S')).toUpperCase(),
            evidence: (evMatch ? evMatch[1] : (existing.evidence || 'project-wide')).trim(),
            fixSummary: fixMatch
                ? fixMatch[1].trim().split('\n').map(l => l.trim().replace(/^[-*]\s*/, '')).filter(Boolean).slice(0, 2).join('; ')
                : (existing.fixSummary || '')
        });
    }
    return [...staleMap.values()];
}

/**
 * Parse task cards from a Master Backlog markdown document.
 * Honors Referee `### 1. SUMMARY TABLE` verdicts (`CLEAN_PASS` and verified ID allowlist)
 * while preserving full backward-compatibility for prose/table backlogs without Section 1.
 */
function parseMasterBacklogCards(backlogText) {
    const cards = new Map();
    const rawText = String(backlogText || '');
    const summarySection = extractLastSummaryTableSection(rawText);
    const scanText = stripNonDefectBacklogSections(rawText);
    const rowRegex = /\|\s*[*`[]*([A-Z]{2,8}-\d{1,3})[\]*`]*\s*\|\s*([^|]+)\|\s*[*`]*?(P[0-3])[*`]*\s*\|\s*[*`]*([SMLX]{1,2})[*`]*\s*\|/gi;
    let allowedIds = null;
    let m;

    // 1. Seed from Referee `### 1. SUMMARY TABLE` when present; otherwise seed from any summary table rows
    if (summarySection !== null) {
        const summaryIds = new Set();
        while ((m = rowRegex.exec(summarySection)) !== null) {
            const id = m[1].trim().toUpperCase();
            if (id.startsWith('STALE-')) continue;
            summaryIds.add(id);
            cards.set(id, {
                id,
                title: m[2].replace(/^\*\*|\*\*$/g, '').trim(),
                severity: m[3].trim().toUpperCase(),
                effort: m[4].trim().toUpperCase(),
                evidence: 'project-wide',
                fixSummary: ''
            });
        }
        if (summaryIds.size === 0 && isCleanPassBacklog(rawText)) {
            return [];
        }
        if (summaryIds.size > 0) {
            allowedIds = summaryIds;
        }
    } else {
        while ((m = rowRegex.exec(scanText)) !== null) {
            const id = m[1].trim().toUpperCase();
            if (id.startsWith('STALE-')) continue;
            cards.set(id, {
                id,
                title: m[2].replace(/^\*\*|\*\*$/g, '').trim(),
                severity: m[3].trim().toUpperCase(),
                effort: m[4].trim().toUpperCase(),
                evidence: 'project-wide',
                fixSummary: ''
            });
        }
    }

    // 2. Parse card blocks starting with ###/#### [ID] and enrich/override seeded table entries
    const cardHeadingRegex = /(?:^|\n)#{3,4}[ \t]+\[?([A-Z]{2,8}-\d{1,3})\]?(?:[: \t—-]+([^\n]*))?([\s\S]*?)(?=(?:\n#{2,4}[ \t]+)|$)/g;
    while ((m = cardHeadingRegex.exec(scanText)) !== null) {
        const id = m[1].trim().toUpperCase();
        if (id.startsWith('STALE-')) continue;
        if (allowedIds && !allowedIds.has(id)) continue;
        const existing = cards.get(id) || {};
        const headingTitle = (m[2] || '').trim();
        const body = m[3] || '';

        const titleMatch = body.match(/\*\*Title:?\*\*\s*([^\n]+)/i);
        const sevMatch = body.match(/\*\*Severity:?\*\*\s*(P[0-3])/i);
        const effMatch = body.match(/\*\*Effort:?\*\*\s*([SMLX]{1,2})/i);
        const evMatch = body.match(/\*\*(?:Evidence|File):?\*\*\s*`?([^`\n]+)`?/i)
            || body.match(/`([a-zA-Z0-9_./-]+\.(?:js|mjs|cjs|ts|tsx|jsx|py|json|md|yml|yaml|sh|sql|html|css)(?::\d+(?:-\d+)?)?)`/i);
        const fixMatch = body.match(/\*\*(?:Fix Instructions|Fix|Execution Plan|Why It Matters):?\*\*\s*([\s\S]*?)(?=\n[-*]*\s*\*\*[A-Z]|\n---|$)/i);

        const title = (titleMatch ? titleMatch[1] : (headingTitle || existing.title || id)).replace(/^\*\*|\*\*$/g, '').trim() || id;
        const severity = (sevMatch ? sevMatch[1] : (existing.severity || 'P2')).toUpperCase();
        const effort = (effMatch ? effMatch[1] : (existing.effort || 'M')).toUpperCase();
        const evidence = (evMatch ? evMatch[1] : (existing.evidence || 'project-wide')).trim();
        const fixSummary = fixMatch
            ? fixMatch[1].trim().split('\n').map(l => l.trim().replace(/^[-*]\s*/, '')).filter(Boolean).slice(0, 2).join('; ')
            : (existing.fixSummary || '');

        cards.set(id, { id, title, severity, effort, evidence, fixSummary });
    }

    // 3. Fallback: parse file-anchored bullet items `- **path/to/file.ext:line** - Title` with `*Severity*: P0-P3` and `*Fix*: ...`
    if (cards.size === 0 && !allowedIds) {
        const bulletRegex = /(?:^|\n)-\s+\*\*`?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+(?::\d+(?:-\d+)?)?)`?\*\*\s*[-—:]\s*([^\n]+)((?:\n\s+-[^\n]+)*)/g;
        let idx = 1;
        while ((m = bulletRegex.exec(scanText)) !== null) {
            const evidence = m[1].trim();
            const rawTitle = m[2].trim();
            const subBullets = m[3] || '';
            if (/already resolved|historical_fixed_bug|#311/i.test(`${rawTitle} ${subBullets}`)) continue;
            const sevMatch = subBullets.match(/\*?\*?Severity\*?\*?:\s*(P[0-3])/i) || rawTitle.match(/\((P[0-3])\)/i);
            const effMatch = subBullets.match(/\*?\*?Effort\*?\*?:\s*([SMLX]{1,2})/i);
            const fixMatch = subBullets.match(/\*?\*?Fix(?: Instructions)?\*?\*?:\s*([^\n]+)/i);
            const severity = (sevMatch ? sevMatch[1] : 'P2').toUpperCase();
            const effort = (effMatch ? effMatch[1] : 'S').toUpperCase();
            const prefix = (severity === 'P0' || severity === 'P1') ? 'BUG' : 'ARCH';
            const id = `${prefix}-${String(idx++).padStart(2, '0')}`;
            cards.set(id, {
                id,
                title: rawTitle.slice(0, 120),
                severity,
                effort,
                evidence,
                fixSummary: fixMatch ? fixMatch[1].trim() : ''
            });
        }
    }

    return [...cards.values()];
}

/**
 * Generate `.agent/audit/task.md` in the exact `/audit-fix` checkbox format,
 * reconcile stale/resolved audit blockers, and persist active P0/P1 blockers to `krusch-context-mcp`.
 */
async function finalizeMasterBacklogArtifacts(targetProject, auditDir, reply, { callToolFn = callTool } = {}) {
    const cards = parseMasterBacklogCards(reply);
    const staleCards = extractStaleCleanupCards(reply);
    const verdictMatch = String(reply || '').match(/###\s*5\.\s*STALENESS\s*&\s*ARCHIVAL\s*VERDICT([\s\S]*?)(?=\n###\s+(?:LESSON|JEAN_INFLIGHT)|$)/i);

    if (staleCards.length > 0 || verdictMatch?.[1]) {
        try {
            const cleanupPath = 'jean_homelab_cleanup_backlog.md';
            const header = !fs.existsSync(cleanupPath)
                ? `# 🧹 Jean SRE — Homelab Staleness & Cleanup Backlog\n> Auto-compiled from whole-homelab audits (\`@homelab/*\` & \`projects/*\`).\n\n`
                : '';
            const staleLines = staleCards.map(c => `- **[${c.id}] ${c.title}** (${c.severity} / Effort: ${c.effort}) — \`${c.evidence}\`${c.fixSummary ? `\n  - Action: ${c.fixSummary}` : ''}`).join('\n');
            const verdictText = verdictMatch?.[1] ? verdictMatch[1].trim() : '';
            const entry = `## 🗂️ ${targetProject} (${new Date().toISOString()})\n${verdictText ? `${verdictText}\n\n` : ''}${staleLines ? `${staleLines}\n\n` : ''}---\n\n`;
            await fs.promises.appendFile(cleanupPath, `${header}${entry}`);
            ensureHostWritable(cleanupPath);
            console.log(`[Cleanup] 🧹 Logged staleness/cleanup triage for ${targetProject} to ${cleanupPath}`);
        } catch (e) {
            console.warn(`[Cleanup] Failed to append to jean_homelab_cleanup_backlog.md:`, e.message);
        }
    }

    const bugs = cards.filter(c => c.severity === 'P0' || c.severity === 'P1');
    const opts = cards.filter(c => c.severity === 'P2');
    const polish = cards.filter(c => c.severity === 'P3');

    if (bugs.length === 0 && opts.length === 0) {
        const conv = markProjectConverged(targetProject, resolveTargetDir(targetProject), 'CLEAN_PASS');
        if (conv) {
            console.log(`[Convergence] 🏆 Project ${targetProject} achieved CLEAN_PASS (0 P0–P2 cards, signature=${conv.signature}). Marked converged.`);
        }
    }

    const formatItem = (c) => {
        const line = `- [ ] **[${c.id}] ${c.title}** (${c.severity} / Effort: ${c.effort}) — \`${c.evidence}\``;
        return c.fixSummary ? `${line}\n  - Fix: ${c.fixSummary.slice(0, 220)}` : line;
    };

    const headerBanner = cards.length === 0
        ? `> **Status:** ✅ \`CLEAN_PASS\` — Zero actionable defects verified (${new Date().toISOString()}).`
        : `> Auto-generated by Jean SRE from \`master_backlog.md\` for the \`/audit-fix\` and \`/loop\` workflows.`;

    const taskMd = [
        `# Audit Fix Tasks — ${targetProject}`,
        headerBanner,
        ``,
        `## 🔴 Bugs & Critical (P0 / P1 — must fix)`,
        bugs.length > 0 ? bugs.map(formatItem).join('\n') : `- *(No P0/P1 critical bugs flagged)*`,
        ``,
        `## 🟠 Optimizations & Structural Debt (P2 — should fix)`,
        opts.length > 0 ? opts.map(formatItem).join('\n') : `- *(No P2 optimization items flagged)*`,
        ``,
        `## 🟡 Consistency & Polish (P3 — nice to fix)`,
        polish.length > 0 ? polish.map(formatItem).join('\n') : `- *(No P3 polish items flagged)*`,
        ``
    ].join('\n');

    const taskPath = join(auditDir, 'task.md');
    await fs.promises.writeFile(taskPath, taskMd);
    ensureHostWritable(taskPath);
    console.log(`[Audit] ✅ Generated /audit-fix checklist (${cards.length} tasks) at ${taskPath}`);

    const topBugs = bugs.slice(0, 6);
    let activeByStore = new Map();
    try {
        const rec = await reconcileProjectBlockers(targetProject, topBugs, callToolFn);
        activeByStore = rec.activeByStore;
    } catch (e) {
        console.warn(`[Blockers] Failed to reconcile prior blockers for ${targetProject}:`, e.message);
    }

    // Tier 1 /code-audit requirement: Persist active P0/P1 blockers to episodic memory (deduplicated)
    for (const b of topBugs) {
        try {
            const blockerSummary = formatAuditBlockerSummary(b);
            const norm = normalizeAuditBlockerBody(blockerSummary, targetProject);
            const skipStores = new Set();
            for (const [storeName, storeSet] of activeByStore.entries()) {
                if (storeSet.has(norm)) skipStores.add(storeName);
            }
            await saveProjectBlocker(blockerSummary, targetProject, { skipStores, callToolFn });
        } catch (e) {
            console.warn(`[Blockers] Failed to persist blocker ${b.id}:`, e.message);
        }
    }
}

/**
 * Save audit findings for a specific phase directly to the project's .agent/audit directory.
 * Maps 11 AUDIT_PHASES to 4 checkpoint files specified in the /code-audit workflow,
 * plus generates `task.md` for `/audit-fix` on `2_MASTER_BACKLOG`.
 */
async function saveAuditCheckpoint(targetProject, phase, reply, { isColabBurst = false, extraToolCommands = [] } = {}) {
    if (!targetProject) return;
    const projectPath = resolveTargetDir(targetProject);
    const auditDir = join(projectPath, '.agent', 'audit');
    
    try {
        await fs.promises.mkdir(auditDir, { recursive: true });
        ensureHostWritable(join(projectPath, '.agent'));
        ensureHostWritable(auditDir);
        
        // At Phase 0_ORIENTATION, clear stale phase_1_investigation.md and phase_2_adversarial.md from prior rotations
        if (phase === '0_ORIENTATION') {
            for (const staleFile of ['phase_1_investigation.md', 'phase_2_adversarial.md']) {
                try {
                    const stalePath = join(auditDir, staleFile);
                    if (fs.existsSync(stalePath)) {
                        await fs.promises.unlink(stalePath);
                    }
                } catch {}
            }
        }

        let fileName = '';
        let headerText = '';
        
        if (phase === '0_ORIENTATION') {
            fileName = 'phase_0_orientation.md';
            headerText = `# 🔍 Phase 0: Orientation & Project Rubric — ${targetProject}\n\n`;
        } else if ([
            '1.1_STANDARD_MAPPING', 
            '1.2_COMPLIANCE_MATRIX', 
            '1.3_CRITICAL_PATH', 
            '1.4_BOUNDARY_AUDIT', 
            '1.5_ANTI_PATTERN'
        ].includes(phase)) {
            fileName = 'phase_1_investigation.md';
            headerText = `# 🕵️ Phase 1: Investigative Audits — ${targetProject}\n\n`;
        } else if ([
            '1.6_STRUCTURAL_STRESS_TEST', 
            '1.7_VERIFICATION_AUDIT', 
            '1.8_RISK_MAP', 
            '1.9_ADVERSARIAL_REVIEW'
        ].includes(phase)) {
            fileName = 'phase_2_adversarial.md';
            headerText = `# 🧠 Phase 2: Stress Tests & Adversarial Review — ${targetProject}\n\n`;
        } else if (phase === '2_MASTER_BACKLOG') {
            fileName = 'master_backlog.md';
            headerText = `# 🗃️ Phase 3: Master Backlog & Final Report — ${targetProject}\n\n`;
        }
        
        if (fileName) {
            const filePath = `${auditDir}/${fileName}`;
            let fileExists = false;
            try {
                await fs.promises.access(filePath);
                fileExists = true;
            } catch {}
            
            const entry = `\n## 📝 Audit Phase: ${phase} (${new Date().toISOString()})\n\n${reply}\n\n---\n`;
            
            const isFirstSubPhase = [
                '0_ORIENTATION',
                '1.1_STANDARD_MAPPING',
                '1.6_STRUCTURAL_STRESS_TEST',
                '2_MASTER_BACKLOG'
            ].includes(phase);

            if (fileExists && !isFirstSubPhase) {
                await fs.promises.appendFile(filePath, entry);
                console.log(`[Audit] Appended phase ${phase} checkpoint to ${filePath}`);
            } else {
                await fs.promises.writeFile(filePath, `${headerText}${entry}`);
                console.log(`[Audit] Wrote phase ${phase} checkpoint to ${filePath}`);
            }
            ensureHostWritable(filePath);

            if (phase === '2_MASTER_BACKLOG') {
                await finalizeMasterBacklogArtifacts(targetProject, auditDir, reply);
            }
            if (isColabBurst || phase === '2_MASTER_BACKLOG' || !fs.existsSync(join(auditDir, 'teacher_120b_distilled.json'))) {
                distillTeacherKnowledgeForStudent(projectPath, {
                    teacherModel: isColabBurst ? 'gpt-oss:120b' : 'qwen3-coder-30b-moe:latest',
                    extraToolCommands
                });
                if (phase === '2_MASTER_BACKLOG') {
                    console.log(`[Distill] 🎓 Distilled 120B Teacher knowledge into ${targetProject}/.agent/audit/teacher_120b_distilled.json`);
                }
            }
        }
    } catch (e) {
        console.error(`[Audit] Failed to save audit checkpoint for phase ${phase}:`, e.message);
    }
}

/** Extract LESSON blocks from Jean's reply and persist to kruschdb. */
async function extractAndSaveLessons(reply, targetProject = 'jean-sre') {
    const match = reply.match(/### LESSON[^\n]*\n([\s\S]*?)(?=###|$)/);
    if (match?.[1]) {
        const lesson = match[1].trim();
        // Only save if it's not a placeholder and has meaningful content
        if (lesson && !lesson.startsWith('[') && lesson.length > 30) {
            await saveLesson(lesson, targetProject);
        }
    }
}

/** Rotate a log file when it exceeds maxBytes. Keeps the last keepLines lines. */
async function rotateLogFile(filePath, { maxBytes = 512_000, keepLines = 200 } = {}) {
    try {
        let stat;
        try {
            stat = await fs.promises.stat(filePath);
        } catch (e) {
            return; // File doesn't exist
        }
        if (stat.size > maxBytes) {
            const content = await fs.promises.readFile(filePath, 'utf8');
            const lines = content.split('\n');
            const rotatedPath = filePath.replace(/\.md$/, `_${new Date().toISOString().replace(/[:.]/g, '').slice(0, 15)}.md`);
            await fs.promises.rename(filePath, rotatedPath);
            // Start fresh with the tail of the old log for context continuity
            const header = lines[0] || '# Jean Audit Log (rotated)';
            const tail = lines.slice(-keepLines).join('\n');
            await fs.promises.writeFile(filePath, `${header}\n\n> Rotated from ${rotatedPath} at ${new Date().toISOString()}\n\n${tail}\n`);
            console.log(`[Log] Rotated ${filePath} (${(stat.size / 1024).toFixed(0)}KB) → ${rotatedPath}`);
        }
    } catch (e) {
        console.error(`[Log] Rotation failed: ${e.message}`);
        // Non-fatal — don't let log rotation break the pipeline
    }
}

async function logCycle(cycle, phase, prompt, reply) {
    if (!fs.existsSync('jean_sre_audit_log.md')) {
        await fs.promises.writeFile('jean_sre_audit_log.md', '# 📡 Jean\'s Active SRE Sweep: Continuous Audit Log\n\n');
    }
    await rotateLogFile('jean_sre_audit_log.md');
    const shortPrompt = prompt.split('\n').slice(0, 3).join(' ').slice(0, 200);
    const entry = `\n\n### 🕒 ${new Date().toISOString()} (Cycle ${cycle} — ${phase})\n**Focus:** ${shortPrompt}...\n\n${reply}\n\n---\n`;
    await fs.promises.appendFile('jean_sre_audit_log.md', entry);
}

/** Log infra sweep findings to a separate audit log. */
async function logInfraCycle(cycle, phase, prompt, reply) {
    if (!fs.existsSync('jean_infra_audit_log.md')) {
        await fs.promises.writeFile('jean_infra_audit_log.md', '# Jean Infrastructure Audit Log\n\nFleet-wide optimization findings across kruschserv, kruschdev, kruschgame.\n\n');
    }
    await rotateLogFile('jean_infra_audit_log.md');
    const shortPrompt = prompt.split('\n').slice(0, 3).join(' ').slice(0, 200);
    const entry = `\n\n### ${new Date().toISOString()} (Cycle ${cycle} - ${phase})\n**Focus:** ${shortPrompt}...\n\n${reply}\n\n---\n`;
    await fs.promises.appendFile('jean_infra_audit_log.md', entry);
}

function sanitizeRepetitiveReply(rawReply) {
    if (!rawReply) return '';
    const lines = String(rawReply).split('\n');
    const cleaned = [];
    const seenBullets = new Set();
    for (const line of lines) {
        // Drop lines with degenerate token repetition like get_document_version_version_version...
        if (/(_[a-z0-9]{3,})\1{2,}/i.test(line)) {
            continue;
        }
        const trimmed = line.trim();
        if (/^[-*]\s+\[(?:function|method|class|route)\]/.test(trimmed)) {
            const norm = trimmed.replace(/\(\`[^`]+\`\)/g, '').toLowerCase();
            if (seenBullets.has(norm)) continue;
            seenBullets.add(norm);
        }
        cleaned.push(line);
    }
    return cleaned.join('\n').trim();
}

async function sendToAgent(message, { maxRetries = 2, noTools = false } = {}) {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const cid = randomUUID();
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 600_000); // 10min timeout for deep reasoning

            const res = await fetch(`${AGENT_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive',
                    ...(process.env.JEAN_API_KEY ? { 'x-api-key': process.env.JEAN_API_KEY } : {})
                },
                body: JSON.stringify({ conversationId: cid, message, noTools }),
                signal: controller.signal
            });
            clearTimeout(timeout);

            const data = await res.json();
            const reply = sanitizeRepetitiveReply(data.reply || '');

            // Reject empty/undefined responses — retry
            if (!reply || reply.length < 30) {
                console.warn(`[Agent] Empty response (attempt ${attempt + 1}/${maxRetries + 1}, ${reply.length} chars)`);
                if (attempt < maxRetries) { await sleep(5000); continue; }
                return '[Agent returned empty response after retries]';
            }

            // Flag hallucination only when reply introduces a banned system term NOT present in the project source/prompt context
            const BANNED_TERMS = ['nginx', 'kubernetes', 'k8s', 'etcd', 'sidecar', 'Prometheus', 'Grafana', 'kubelet', 'helm', 'istio'];
            const promptWithoutDisclaimer = String(message || '').replace(/The following DO NOT exist[^\n]*/gi, '');
            const hallucinatedTerms = BANNED_TERMS.filter(term => {
                const rx = new RegExp(`\\b${term}\\b`, 'i');
                return rx.test(reply) && !rx.test(promptWithoutDisclaimer);
            });
            if (hallucinatedTerms.length > 0) {
                console.warn(`[Agent] ⚠️ HALLUCINATION DETECTED — reply mentions ungrounded systems: ${hallucinatedTerms.join(', ')}`);
                sendDiscordAlert(`⚠️ **Jean SRE Hallucination Detected (${hallucinatedTerms.join(', ')})**\n\`\`\`\n${reply.slice(0, 500)}\n\`\`\``);
                return `⚠️ [HALLUCINATION FLAG: Response mentions systems that do not exist in this homelab (${hallucinatedTerms.join(', ')})]\n\n${reply}`;
            }

            return reply;
        } catch (e) {
            console.error(`[Agent] Request failed (attempt ${attempt + 1}): ${e.message}`);
            if (attempt < maxRetries) { await sleep(5000); continue; }
            return `[Agent request failed: ${e.message}]`;
        }
    }
    return '[Agent: all retries exhausted]';
}

// ─────────────────────────────────────────────────────────
// Main Sweep Loop
// ─────────────────────────────────────────────────────────

const shutdownController = new AbortController();

async function runActiveSweep() {
    console.log("╔══════════════════════════════════════════════════╗");
    console.log("  Jean - Native Local Homelab Agent");
    console.log(`  Thinker: ${THINKER_MODEL} (${thinkerNode} - ${thinkerNodeName})`);
    console.log(`  Workers: ${WORKER_MODEL} (${workerNode} - ${workerNodeName})`);
    console.log(`  Embeddings: ${process.env.EMBED_MODEL || 'snowflake-arctic-embed2:latest'} (${process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434'})`);
    console.log("  Mode: Code Optimization + Fleet Infrastructure");
    console.log("  Sentinel: Multi-node SSH (3 nodes, 15 tools)");
    console.log("╚══════════════════════════════════════════════════╝");

    let cycle = 1;
    let consecutiveDriftCycles = 0;

    // ── Investigation Lifecycle Tracker ────────────────────
    // Tracks how many cycles the Thinker has spent on each topic FAMILY.
    // After MAX_CYCLES_PER_TOPIC, forces convergence.
    // Related sub-topics (e.g. dbos_backlog, payload_validation) share a family.
    const MAX_CYCLES_PER_TOPIC = 3;
    let currentTopicFamily = '';
    let currentTopicKey = '';
    let topicCycleCount = 0;
    let resolvedTopics = []; // Topic families that have been force-closed

    /**
     * Topic families — related sub-topics that should share a convergence counter.
     * Prevents Jean from escaping convergence by theme-hopping between related investigations.
     */
    const TOPIC_FAMILIES = {
        queue_investigation: ['dbos_backlog', 'payload_validation', 'io_saturation'],
        container_health: ['health_check_none', 'conduit_services'],
        resource_utilization: ['cpu_resource', 'memory_pressure'],
        infrastructure: ['network_issue', 'credential_drift'],
    };

    /**
     * Map a raw topic key to its family. Topics without a family are their own family.
     */
    function getTopicFamily(topicKey) {
        for (const [family, members] of Object.entries(TOPIC_FAMILIES)) {
            if (members.includes(topicKey)) return family;
        }
        return topicKey;
    }

    /**
     * Extract a normalized topic key from the Thinker's hypothesis.
     * Strips filler words and normalizes to detect semantic repetition.
     */
    function extractTopicKey(thinkerReply) {
        const raw = String(thinkerReply || '');
        const hypoMatch = raw.match(/### HYPOTHESIS[\s\S]*?(?=###|$)/i);
        const hypo = (hypoMatch ? hypoMatch[0] : raw).toLowerCase();
        if (!hypo.trim()) return 'general_health';
        // Extract dominant theme from hypothesis
        const themes = [
            [/backlog|queue|swarm.*back/i, 'dbos_backlog'],
            [/i\/o|disk|storage|saturation/i, 'io_saturation'],
            [/health.*none|healthcheck|readiness/i, 'health_check_none'],
            [/conduit|arbiter|toolkit|pico/i, 'conduit_services'],
            [/credential|secret|auth|expir/i, 'credential_drift'],
            [/payload|validation|schema|data.*contract|pipeline.*fail/i, 'payload_validation'],
            [/cpu|load.*average|resource.*exhaust/i, 'cpu_resource'],
            [/memory|ram|leak/i, 'memory_pressure'],
            [/network|dns|partition/i, 'network_issue'],
        ];
        for (const [pattern, key] of themes) {
            if (pattern.test(hypo)) return key;
        }
        return 'general_health';
    }

    // ── OPEN: Deterministic context load ───────────────────
    console.log(`\n[${new Date().toISOString()}] ▶ PHASE: OPEN (Cycle 1)`);
    let context;
    const inflight = readInflight();
    try {
        context = await loadContext(inflight.block);
    } catch (e) {
        console.error('[Open] Context load failed:', e.message);
        context = { lessons: 'No lessons available.', nudges: '', sysStats: '', containers: '', swarmStats: '', intentResults: '' };
    }

    const contextSummary = [
        context.intentResults && !context.intentResults.includes('No previous') ? `**Previous Intent Results:**\n${context.intentResults}` : '',
        context.lessons ? `**Past Lessons:**\n${context.lessons.slice(0, 1500)}` : '',
        context.nudges && !context.nudges.includes('No nudges') ? `**Nudges:**\n${context.nudges.slice(0, 500)}` : ''
    ].filter(Boolean).join('\n\n');

    // Send open prompt with pre-loaded context
    try {
        const openPrompt = buildOpenPrompt(context, inflight.block);
        const reply = await sendToAgent(openPrompt, { noTools: true });

        await persistInflight(reply, cycle);
        await extractAndSaveLessons(reply);
        await logCycle(cycle, 'OPEN', openPrompt, reply);
        console.log(`[Open] Complete. Jean has her context + kruschdb memory.`);
    } catch (e) {
        console.error('[Open] Synthesis failed:', e.message);
    }

    cycle++;

    // ── SWEEP: Thinker → Implementer cycles ─────────────────
    while (!shutdownController.signal.aborted) {

        // ── Active Emergency Check ──
        let activeEmergency = null;
        try {
            const listStr = await callTool('krusch-context-krusch-context-nugget-list', {
                kinds: ['project'],
                active_project: 'jean-sre',
                format: 'json'
            }, { retries: 1, timeout: 5000 });
            
            if (listStr && !listStr.includes('[Failed')) {
                const trimmed = String(listStr).trim();
                if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
                    const nuggets = JSON.parse(trimmed);
                    const arr = Array.isArray(nuggets) ? nuggets : (nuggets.nuggets || []);
                    const emergencyNugget = arr.find(n => n.key === 'active-emergency');
                    if (emergencyNugget) {
                        activeEmergency = typeof emergencyNugget.value === 'string' ? JSON.parse(emergencyNugget.value) : emergencyNugget.value;
                        console.log(`\n[Trigger] 🚨 ACTIVE EMERGENCY DETECTED:`, activeEmergency);
                    }
                } else {
                    const m = trimmed.match(/active-emergency[^:\n]*:\s*(\{[^\n]+\})/i);
                    if (m?.[1]) {
                        activeEmergency = JSON.parse(m[1]);
                        console.log(`\n[Trigger] 🚨 ACTIVE EMERGENCY DETECTED:`, activeEmergency);
                    }
                }
            }
        } catch (e) {
            console.warn(`[Trigger] Failed to check for active emergencies:`, e.message);
        }

        // ── Daytime Gate & Colab G4 120B Burst Check ──
        // When Colab G4 120B (`gpt-oss:120b`) is active during the day, Jean automatically runs full
        // autonomous 11-phase code audits + live test suite verification at accelerated burst intervals.
        const colabStatus = await checkColabBurstStatus();
        const isDaytime = (() => {
            const now = new Date();
            const ptHour = new Date(now.toLocaleString('en-US', { timeZone: 'America/Los_Angeles' })).getHours();
            return ptHour >= 6 && ptHour < 24; // 6am–midnight PT
        })();

        let sweepOverride = String(process.env.JEAN_SWEEP_MODE || '').trim().toLowerCase() === 'sweep' || colabStatus.active;
        if (!sweepOverride) {
            try {
                const mode = fs.readFileSync('JEAN_SWEEP_MODE', 'utf8').trim().toLowerCase();
                sweepOverride = mode === 'sweep';
            } catch (e) { /* no override file — default behavior */ }
        }
        if (colabStatus.active) {
            console.log(`[Colab G4 Burst] ⚡ Active 120B Burst (${colabStatus.model} @ ${colabStatus.endpoint}) — running full daytime autonomous code audit & live test execution.`);
        }

        // Find active IDE project based on GEMINI_INFLIGHT.md mtime (used when local-only Companion Mode is active)
        let activeIDEProject = null;
        let activeIDEInflight = null;
        let activeIDEGitDiff = null;
        if (!sweepOverride) {
            try {
                const projectsDir = `${HOMELAB_ROOT}/projects`;
                const projects = fs.readdirSync(projectsDir);
                let newestTime = 0;
                for (const proj of projects) {
                    const inflightPath = `${projectsDir}/${proj}/GEMINI_INFLIGHT.md`;
                    if (fs.existsSync(inflightPath)) {
                        const stat = fs.statSync(inflightPath);
                        if (stat.mtimeMs > newestTime) {
                            newestTime = stat.mtimeMs;
                            activeIDEProject = proj;
                        }
                    }
                }
                
                // If the most recent GEMINI_INFLIGHT.md was modified in the last 12 hours, consider it active
                if (activeIDEProject && (Date.now() - newestTime < 12 * 60 * 60 * 1000)) {
                    activeIDEInflight = fs.readFileSync(`${projectsDir}/${activeIDEProject}/GEMINI_INFLIGHT.md`, 'utf8');
                    try {
                        activeIDEGitDiff = execSync(`git -C ${projectsDir}/${activeIDEProject} diff HEAD`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] }).slice(0, 2000);
                    } catch (e) {
                        activeIDEGitDiff = 'Git diff unavailable.';
                    }
                    console.log(`[Companion] ☀️ Active IDE session detected: ${activeIDEProject}. Shifting to COMPANION_AUDIT mode.`);
                } else {
                    activeIDEProject = null; // Stale, ignore
                }
            } catch (e) {
                console.error(`[Companion] Failed to check IDE projects:`, e.message);
            }
        }

        if (isDaytime && !sweepOverride && !activeIDEProject && !activeEmergency) {
            console.log(`[Daytime] ☀️ Daytime mode (no active IDE session & G4 Burst off) — no autonomous sweeps. Sleeping 5m (idle)...`);
            cycle++;
            await interruptibleSleep(5 * 60 * 1000); // 5 minutes sleep
            continue; // Skip sweep, loop back to sleep + handoff check
        }

        if (activeEmergency) {
            isInfraMode = true;
        }

        const phaseLabel = isInfraMode
            ? `INFRA SWEEP (Cycle ${cycle}) - ${activeEmergency ? activeEmergency.node + '/' + activeEmergency.type : getCurrentInfraTarget().node + '/' + getCurrentInfraTarget().focus}`
            : `SWEEP (Cycle ${cycle}) - ${getCurrentProject()} [${getCurrentPhase()}]${colabStatus.active ? ` [⚡ G4 ${colabStatus.model}]` : ''}`;
        console.log(`\n[${new Date().toISOString()}] > PHASE: ${phaseLabel}`);
        try {
            const inflight = readInflight(activeIDEProject || getCurrentProject(), getCurrentPhase(), isInfraMode);

            // Refresh telemetry for this cycle
            console.log(`[Sweep ${cycle}] Refreshing telemetry...`);
            let freshContext;
            try {
                freshContext = await loadContext(inflight.block, activeIDEProject || getCurrentProject());
            } catch (e) {
                console.warn(`[Sweep ${cycle}] Telemetry refresh failed, using stale context`);
                freshContext = null;
            }

            // Build fresh context summary with real-time data
            // Use the inflight already read at the top of this cycle (line 591)
            let workflow = routeWorkflow(inflight.goal, isInfraMode);
            
            if (activeIDEProject) {
                workflow = WORKFLOWS.COMPANION_AUDIT;
            }

            // Force workflow shift on persistent drift
            if (consecutiveDriftCycles >= 3) {
                console.log(`[Drift] 🔄 Forcing workflow shift to CODE_AUDIT (${consecutiveDriftCycles} drift cycles)`);
                workflow = WORKFLOWS.CODE_AUDIT;
            }

            // For CODE_AUDIT: strip infrastructure telemetry so Thinker stays focused on code
            const isCodeAudit = !isInfraMode && workflow.includes('CODE_AUDIT');
            const cycleContext = freshContext ? [
                freshContext.intentResults && !freshContext.intentResults.includes('No previous') ? `**Previous Intent Results:**\n${freshContext.intentResults.slice(0, 1200)}` : '',
                !isCodeAudit && freshContext.sysStats ? `**Live System Stats (kruschserv):**\n${freshContext.sysStats}` : '',
                !isCodeAudit && freshContext.containers ? `**Container Health:**\n${freshContext.containers}` : '',
                !isCodeAudit && freshContext.swarmStats ? `**DBOS Swarm:**\n${freshContext.swarmStats}` : '',
                freshContext.lessons ? `**Past Lessons:**\n${freshContext.lessons.slice(0, isCodeAudit ? 1600 : 2800)}` : ''
            ].filter(Boolean).join('\n\n') : contextSummary;


            // Build drift warning for the Thinker
            let driftWarning = '';
            if (consecutiveDriftCycles >= 2) {
                driftWarning = `## ⚠️ DRIFT WARNING (${consecutiveDriftCycles} consecutive cycles produced REDUNDANT investigations)
Your last ${consecutiveDriftCycles} cycles dispatched intents that were semantically IDENTICAL to previous ones and were BLOCKED.
YOU MUST CHANGE YOUR APPROACH. INVESTIGATE SOMETHING NEW or your intents will be blocked again.`;
            }

            // ── Convergence Forcing (family-aware, infra mode only) ──
            // In phased code audit mode, each phase advances deterministically every cycle.
            if (isInfraMode && currentTopicFamily !== 'general_health' && topicCycleCount >= MAX_CYCLES_PER_TOPIC) {
                const familyLabel = currentTopicFamily !== currentTopicKey ? `${currentTopicFamily} (${currentTopicKey})` : currentTopicKey;
                const convergenceWarning = `## 🛑 CONVERGENCE REQUIRED (Investigation: "${familyLabel}" — ${topicCycleCount} cycles)
You have investigated topics related to "${currentTopicFamily}" for ${topicCycleCount} consecutive cycles without resolving it.
Sub-topics you explored: ${currentTopicKey} (and related themes).
You MUST do ONE of these:
1. **STATE YOUR CONCLUSION** — What did you learn? Was your hypothesis correct or incorrect? Cite specific evidence from intent results.
2. **CLOSE THIS INVESTIGATION** — Mark it as "inconclusive — insufficient evidence" and move on.
3. **ESCALATE** — If this requires human intervention, say so explicitly.

You may NOT dispatch more intents about "${currentTopicFamily}" or related sub-topics. Choose a COMPLETELY DIFFERENT area.
${resolvedTopics.length > 0 ? `\nPreviously resolved/closed families (DO NOT revisit): ${resolvedTopics.join(', ')}` : ''}`;
                driftWarning = convergenceWarning;
                console.log(`[Convergence] 🛑 Forcing topic change from family "${currentTopicFamily}" after ${topicCycleCount} cycles`);
            }

            // ── Branch: Infra mode vs Code audit ──
            if (isInfraMode) {
                // === INFRA SWEEP CYCLE ===
                const infraTarget = activeEmergency
                    ? { node: activeEmergency.node, focus: activeEmergency.type === 'thermal' ? 'system_tuning' : 'docker', isEmergency: true }
                    : getCurrentInfraTarget();
                console.log(`[Sweep ${cycle}] INFRA MODE: ${infraTarget.node}/${infraTarget.focus}`);

                // Step 1: THINKER (infra-specific prompt)
                console.log(`[Sweep ${cycle}] Step 1: THINKER (infra analysis)...`);
                let infraDriftWarning = driftWarning;
                if (activeEmergency) {
                    infraDriftWarning = `## 🚨 EMERGENCY DIRECTIVE
AN ACTIVE EMERGENCY HAS BEEN TRIGGERED: [${activeEmergency.type.toUpperCase()}] on ${activeEmergency.node}.
Description: ${activeEmergency.description}
Timestamp: ${activeEmergency.timestamp}

YOU MUST EXCLUSIVELY FOCUS ON REMEDIATING AND TRIAGING THIS INCIDENT. 
Select Sentinel tools to analyze the health of the target node and identify any system or docker anomalies. Formulate clear remediation intents.`;
                }
                const infraSkillPlaybook = loadJeanSkillPlaybook('infra');
                const legoScoutBrief = infraTarget.focus === 'lego_scout' ? buildLegoScoutContextBrief(HOMELAB_ROOT) : '';
                const infraCycleContext = [
                    cycleContext,
                    legoScoutBrief,
                    infraSkillPlaybook ? `**Active Infra Skill Playbook (\`/jean-self-healing-infra\`):**\n${infraSkillPlaybook}` : ''
                ].filter(Boolean).join('\n\n');
                const infraThinkerPrompt = buildInfraThinkerPrompt(infraCycleContext, inflight.block, infraTarget, infraDriftWarning);
                const infraThinkerReply = await sendToAgent(infraThinkerPrompt, { noTools: true });
                console.log(`[Sweep ${cycle}] Thinker produced ${infraThinkerReply.length} chars`);
                await logInfraCycle(cycle, `INFRA-THINK (${infraTarget.node}/${infraTarget.focus})`, infraThinkerPrompt, infraThinkerReply);

                // Step 2: Parse and dispatch infra intents
                console.log(`[Sweep ${cycle}] Step 2: DISPATCH (infra intents)...`);
                const infraParsedIntents = parseThinkerPlan(infraThinkerReply);
                let infraDispatched = [];
                
                // ── Backlog Gate ──
                const MAX_PENDING_INTENTS = 8;
                let backlogBlocked = false;
                try {
                    const stats = JSON.parse(await fetchJeanSwarmStats());
                    const pendingCount = parseInt(stats.intent_pending || '0', 10);
                    const runningCount = parseInt(stats.intent_running || '0', 10);
                    if (pendingCount + runningCount >= MAX_PENDING_INTENTS) {
                        backlogBlocked = true;
                        console.log(`[Backlog] Skipping dispatch - ${pendingCount} pending + ${runningCount} running >= ${MAX_PENDING_INTENTS} limit.`);
                    }
                } catch (e) {
                    console.warn(`[Backlog] Stats check failed, proceeding with dispatch:`, e.message);
                }

                if (backlogBlocked) {
                    console.log(`[Sweep ${cycle}] Dispatched 0/${infraParsedIntents.length} jobs (backlog gate)`);
                } else if (infraParsedIntents.length > 0) {
                    const recentInstructions = await fetchRecentIntentInstructions(15);
                    const driftResults = [];
                    for (const intent of infraParsedIntents) {
                        const driftResult = await checkIntentDrift(intent.task, recentInstructions);
                        driftResults.push({ intent, driftResult });
                    }

                    const novelIntents = [];
                    for (const { intent, driftResult } of driftResults) {
                        if (driftResult.isDrift) {
                            console.log(`[Drift] BLOCKED: ${intent.task.slice(0, 80)}... -> ${driftResult.reason}`);
                        } else {
                            console.log(`[Drift] NOVEL: ${intent.task.slice(0, 80)}...`);
                            novelIntents.push(intent);
                        }
                    }

                    if (novelIntents.length === 0 && infraParsedIntents.length > 0) {
                        consecutiveDriftCycles++;
                        console.warn(`[Drift] ALL ${infraParsedIntents.length} infra intents blocked as drift (streak: ${consecutiveDriftCycles})`);
                    } else {
                        consecutiveDriftCycles = 0;
                    }

                    infraDispatched = await dispatchIntents(novelIntents, cycle, infraThinkerReply);
                    console.log(`[Sweep ${cycle}] Dispatched ${infraDispatched.length}/${infraParsedIntents.length} jobs (${infraParsedIntents.length - novelIntents.length} blocked by drift gate)`);
                }

                if (legoScoutBrief) {
                    infraDispatched = [
                        {
                            job_id: 'lego-sct0',
                            status: 'completed',
                            command: 'krusch-lego scout <HOMELAB_ROOT>/projects --portable',
                            output: legoScoutBrief
                        },
                        ...infraDispatched
                    ];
                }

                // Step 3: IMPLEMENTER (infra-specific synthesis)
                console.log(`[Sweep ${cycle}] Step 3: IMPLEMENTER (infra report synthesis)...`);
                const infraImplementerPrompt = buildInfraImplementerPrompt(infraThinkerReply, infraDispatched, inflight.block, infraTarget, infraParsedIntents);
                const infraImplementerReply = await sendToAgent(infraImplementerPrompt, { noTools: true });
                console.log(`[Sweep ${cycle}] Implementer produced ${infraImplementerReply.length} chars`);

                await persistInflight(infraImplementerReply, cycle);
                await extractAndSaveLessons(infraImplementerReply, 'jean-sre');
                await logInfraCycle(cycle, `INFRA-REPORT (${infraTarget.node}/${infraTarget.focus})`, infraImplementerPrompt, infraImplementerReply);
                if (infraTarget.focus === 'lego_scout') {
                    try {
                        await appendLegoScoutBacklogEntry('jean_homelab_cleanup_backlog.md', infraTarget, infraImplementerReply);
                        console.log(`[LegoScout] 🧱 Appended Rule-of-3 Lego Scout report to jean_homelab_cleanup_backlog.md`);
                    } catch (e) {
                        console.warn(`[LegoScout] Failed to append to jean_homelab_cleanup_backlog.md:`, e.message);
                    }
                }

                // Clean active emergency if it was processed
                if (activeEmergency) {
                    console.log(`[Trigger] 🟢 Emergency remediation dispatched. Cleaning active-emergency nugget.`);
                    try {
                        await callTool('krusch_context_revise', {
                            action: 'forget_nugget',
                            key: 'active-emergency',
                            project: 'jean-sre'
                        });
                        console.log(`[Trigger] nugget-forget active-emergency successful.`);
                    } catch (e) {
                        console.error(`[Trigger] Failed to delete active-emergency nugget:`, e.message);
                    }
                }

            } else if (workflow === WORKFLOWS.COMPANION_AUDIT) {
                // === COMPANION AUDIT CYCLE ===
                const targetProject = activeIDEProject;
                const projectPath = resolveTargetDir(targetProject);
                console.log(`[Sweep ${cycle}] COMPANION MODE: ${targetProject}`);
                
                const telemetryString = [
                    freshContext && freshContext.sysStats ? `**Live System Stats (kruschserv):**\n${freshContext.sysStats}` : '',
                    freshContext && freshContext.swarmStats ? `**DBOS Swarm:**\n${freshContext.swarmStats}` : '',
                    activeIDEGitDiff ? `**Active Git Diff (first 2000 chars):**\n${activeIDEGitDiff}` : ''
                ].filter(Boolean).join('\n\n');

                const companionSkillPlaybook = loadJeanSkillPlaybook('companion');
                const companionTestResult = runProjectVerificationSuite(projectPath);
                const companionProjectContext = [
                    contextSummary,
                    buildProjectSpecAndDocsBrief(projectPath),
                    buildAstContextSection(projectPath, '1.3_CRITICAL_PATH'),
                    companionTestResult?.report || '',
                    companionSkillPlaybook ? `**Active Companion Skill Playbook (\`/jean-self-healing-code\`):**\n${companionSkillPlaybook}` : ''
                ].filter(Boolean).join('\n\n');

                // Step 1: THINKER
                console.log(`[Sweep ${cycle}] Step 1: THINKER (companion verification)...`);
                const companionThinkerPrompt = buildCompanionThinkerPrompt(companionProjectContext, activeIDEInflight, telemetryString, targetProject, projectPath);
                const companionThinkerReply = await sendToAgent(companionThinkerPrompt, { noTools: true });
                console.log(`[Sweep ${cycle}] Thinker produced ${companionThinkerReply.length} chars`);
                await logInfraCycle(cycle, `COMPANION-THINK (${targetProject})`, companionThinkerPrompt, companionThinkerReply);

                // Step 2: DISPATCH & INLINE EXECUTION
                console.log(`[Sweep ${cycle}] Step 2: DISPATCH (verification intents)...`);
                const parsedIntents = parseThinkerPlan(companionThinkerReply);
                let dispatched = [];
                if (parsedIntents.length > 0) {
                    dispatched = await dispatchIntents(parsedIntents, cycle, companionThinkerReply, projectPath);
                }

                // Step 3: IMPLEMENTER
                console.log(`[Sweep ${cycle}] Step 3: IMPLEMENTER (companion report)...`);
                const companionImplementerPrompt = buildCompanionImplementerPrompt(companionThinkerReply, dispatched, activeIDEInflight, targetProject);
                const companionImplementerReply = await sendToAgent(companionImplementerPrompt, { noTools: true });
                console.log(`[Sweep ${cycle}] Implementer produced ${companionImplementerReply.length} chars`);

                await persistInflight(companionImplementerReply, cycle, targetProject);
                await extractAndSaveLessons(companionImplementerReply, targetProject);
                await logInfraCycle(cycle, `COMPANION-REPORT (${targetProject})`, companionImplementerPrompt, companionImplementerReply);

            } else {
            // === NORMAL CODE AUDIT CYCLE ===

            // ── Step 1: THINKER (no tools) ──
            console.log(`[Sweep ${cycle}] Step 1: THINKER (analysis + plan)...`);
            const targetProject = getCurrentProject();
            const projectPath = resolveTargetDir(targetProject);
            const phase = getCurrentPhase();
            const phaseNum = currentPhaseIndex;
            const totalPhases = AUDIT_PHASES.length;
            
function getPhaseScopedChecklist(fullText, phase) {
    if (!fullText) return '';
    const rulesMatch = fullText.match(/═+\s*SCOPE[\s\S]*?═+\s*EXECUTION DISCIPLINE[\s\S]*?(?=═+\s*PHASE 0)/);
    const rules = (rulesMatch ? rulesMatch[0].trim() : fullText.slice(0, 1200)).slice(0, 1200);
    
    const phaseNum = phase.split('_')[0];
    const isPhase0 = phase === '0_ORIENTATION';
    const isPhase2 = phase === '2_MASTER_BACKLOG' || phase === 'SYNTHESIS';
    const headerRegex = isPhase0
        ? /^PHASE\s+0\b/i
        : isPhase2
            ? /^PHASE\s+2\b/i
            : new RegExp(`^\\s*${phaseNum.replace('.', '\\.')}\\s+[A-Z]`, 'i');
    
    const lines = fullText.split('\n');
    let capturing = false;
    const sectionLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (!capturing && headerRegex.test(line.trim())) {
            capturing = true;
            sectionLines.push(line);
            continue;
        }
        if (capturing) {
            if (!isPhase2 && sectionLines.length > 4 && /^(?:PHASE\s+\d|\d+\.\d+\s+[A-Z])/.test(line.trim()) && !headerRegex.test(line.trim())) {
                break;
            }
            sectionLines.push(line);
            if (sectionLines.join('\n').length >= 2200) break;
        }
    }
    
    const section = sectionLines.join('\n').trim().slice(0, 2200);
    return `${rules}\n\n═══════════════════════════════════════════════════\nCURRENT PHASE CHECKLIST: ${phase}\n═══════════════════════════════════════════════════\n${section || fullText.slice(-2000)}`;
}

            // Read Session Input Document (scoped to current phase to prevent context overflow)
            let sessionInputDocument = '';
            try {
                const fullChecklist = await fs.promises.readFile(new URL('../master_audit_checklist.txt', import.meta.url), 'utf-8');
                sessionInputDocument = getPhaseScopedChecklist(fullChecklist, phase);
            } catch (e) {
                console.error('[Sweep] Could not load master_audit_checklist.txt:', e.message);
            }

            const groundTruthVerification = buildPendingIssueCodeEvidence(inflight.block, projectPath, phase);
            const projectAuditContext = buildProjectAuditContext(projectPath, phase, inflight.block, { isColabBurst: colabStatus.active });
            const enrichedCycleContext = [cycleContext, projectAuditContext].filter(Boolean).join('\n\n');

            const thinkerPrompt = buildThinkerPrompt(enrichedCycleContext, inflight.block, inflight.goal, workflow, driftWarning, targetProject, projectPath, phase, phaseNum, totalPhases, sessionInputDocument);
            const thinkerReply = await sendToAgent(thinkerPrompt, { noTools: true });
            console.log(`[Sweep ${cycle}] Thinker produced ${thinkerReply.length} chars`);

            // ── Update Investigation Tracker (scoped per project+phase in code audit mode) ──
            const newTopicKey = `audit:${targetProject}:${phase}`;
            const newTopicFamily = `audit:${targetProject}:${phase}`;
            if (newTopicFamily === currentTopicFamily) {
                topicCycleCount++;
                console.log(`[Tracker] Topic "${currentTopicKey}" (family: ${currentTopicFamily}) - cycle ${topicCycleCount}/${MAX_CYCLES_PER_TOPIC}`);
            } else {
                currentTopicKey = newTopicKey;
                currentTopicFamily = newTopicFamily;
                topicCycleCount = 1;
                console.log(`[Tracker] New phase topic: "${newTopicKey}" (cycle 1/${MAX_CYCLES_PER_TOPIC})`);
            }
            await logCycle(cycle, 'THINK', thinkerPrompt, thinkerReply);

            // ── Step 2: DRIFT-GATED DISPATCH & INLINE L2 EXECUTION ──
            // SYNTHESIS phase produces a MASTER BACKLOG, not an INVESTIGATION PLAN — skip dispatch
            const isSynthesisPhase = getCurrentPhase() === '2_MASTER_BACKLOG' || (currentPhaseIndex === AUDIT_PHASES.length - 1);
            if (isSynthesisPhase) {
                console.log(`[Sweep ${cycle}] SYNTHESIS phase — skipping dispatch (Master Backlog, no intents expected)`);
            }
            const parsedIntents = isSynthesisPhase ? [] : parseThinkerPlan(thinkerReply);
            console.log(`[Sweep ${cycle}] ${isSynthesisPhase ? 'Skipped' : 'Parsed'} ${parsedIntents.length} intents from thinker`);

            // ── Backlog Gate: skip dispatch if workers are overwhelmed ──
            const MAX_PENDING_INTENTS = 8;
            let backlogBlocked = false;
            try {
                const stats = JSON.parse(await fetchJeanSwarmStats());
                const pendingCount = parseInt(stats.intent_pending || '0', 10);
                const runningCount = parseInt(stats.intent_running || '0', 10);
                if (pendingCount + runningCount >= MAX_PENDING_INTENTS) {
                    backlogBlocked = true;
                    console.log(`[Backlog] Skipping dispatch - ${pendingCount} pending + ${runningCount} running >= ${MAX_PENDING_INTENTS} limit.`);
                }
            } catch (e) {
                console.warn(`[Backlog] Stats check failed, proceeding with dispatch:`, e.message);
            }

            // ── Drift Detection via Sequential MCP Plausibility Gate ──
            let dispatched = [];
            if (backlogBlocked) {
                console.log(`[Sweep ${cycle}] Dispatched 0/${parsedIntents.length} jobs (backlog gate)`);
            } else {
                const recentInstructions = await fetchRecentIntentInstructions(15);
                const driftResults = [];
                for (const intent of parsedIntents) {
                    const driftResult = await checkIntentDrift(intent.task, recentInstructions);
                    driftResults.push({ intent, driftResult });
                }

                const novelIntents = [];
                for (const { intent, driftResult } of driftResults) {
                    if (driftResult.isDrift) {
                        console.log(`[Drift] BLOCKED: ${intent.task.slice(0, 80)}... -> ${driftResult.reason}`);
                    } else {
                        console.log(`[Drift] NOVEL: ${intent.task.slice(0, 80)}...`);
                        novelIntents.push(intent);
                    }
                }

                // Block ALL intents only if an infra topic family exceeded MAX_CYCLES_PER_TOPIC
                const convergenceBlocked = isInfraMode && currentTopicFamily !== 'general_health' && topicCycleCount > MAX_CYCLES_PER_TOPIC && novelIntents.some(i =>
                    getTopicFamily(extractTopicKey(i.task)) === currentTopicFamily
                );
                if (convergenceBlocked) {
                    console.log(`[Convergence] Blocking ${novelIntents.length} intents - family "${currentTopicFamily}" is over cycle limit`);
                    novelIntents.length = 0; // Clear all
                }

                if (novelIntents.length === 0 && parsedIntents.length > 0) {
                    consecutiveDriftCycles++;
                    console.warn(`[Drift] ALL ${parsedIntents.length} intents blocked as drift (streak: ${consecutiveDriftCycles})`);
                } else {
                    consecutiveDriftCycles = 0;
                }

                dispatched = await dispatchIntents(novelIntents, cycle, thinkerReply, projectPath);
                console.log(`[Sweep ${cycle}] Executed/Dispatched ${dispatched.length}/${parsedIntents.length} jobs (${parsedIntents.length - novelIntents.length} blocked by drift gate)`);
            }

            // ── Step 3: IMPLEMENTER (evidence synthesis - no tools) ──
            console.log(`[Sweep ${cycle}] Step 3: IMPLEMENTER (synthesis)...`);
            const liveTestSuiteSummary = (colabStatus.active || ['1.7_VERIFICATION_AUDIT', '1.8_RISK_MAP', '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'].includes(phase))
                ? (runProjectVerificationSuite(projectPath).report || '')
                : '';
            const rawImplGroundTruth = buildPendingIssueCodeEvidence(`${inflight.block}\n${thinkerReply}`, projectPath, phase) || groundTruthVerification;
            const implementerGroundTruth = [liveTestSuiteSummary, rawImplGroundTruth].filter(Boolean).join('\n\n');
            const implementerPrompt = buildImplementerPrompt(thinkerReply, dispatched, inflight.block, parsedIntents, {
                targetProject,
                phase,
                phaseNum,
                totalPhases,
                groundTruthVerification: implementerGroundTruth
            });
            const implementerReply = await sendToAgent(implementerPrompt, { noTools: true });
            console.log(`[Sweep ${cycle}] Implementer produced ${implementerReply.length} chars`);

            // Save evidence-grounded checkpoint from Implementer output (fallback to Thinker only if Implementer failed)
            const finalCheckpointText = (isSynthesisPhase && (!implementerReply || implementerReply.length < 250 || implementerReply.startsWith('[Agent')))
                ? thinkerReply
                : implementerReply;
            await saveAuditCheckpoint(targetProject, phase, finalCheckpointText, {
                isColabBurst: colabStatus.active,
                extraToolCommands: dispatched.map(d => d.command).filter(Boolean)
            });

            // Persist from the finalized checkpoint output (clears stale Pending Issues on CLEAN_PASS)
            await persistInflight(finalCheckpointText, cycle, targetProject, { phase });
            await extractAndSaveLessons(implementerReply, targetProject);
            await logCycle(cycle, 'IMPLEMENT', implementerPrompt, implementerReply);
            } // end normal code audit branch

            // ── Advance audit phase ──
            if (activeEmergency) {
                isInfraMode = false;
                console.log(`[Trigger] Emergency sweep complete. Resuming regular project rotation: ${getCurrentProject()}`);
            } else {
                const advancement = advanceAuditPhase();
                console.log(`[Sweep] Cycle ${cycle} complete. ${advancement.rotated ? `🔄 Rotated to project: ${advancement.project}` : `Next phase: ${advancement.phase}`}`);
            }
        } catch (e) {
            console.error(`[Sweep] Cycle ${cycle} error:`, e.message);
        }

        cycle++;

        if (!shutdownController.signal.aborted) {
            const nextSleepMinutes = colabStatus.active ? COLAB_BURST_WAIT_MINUTES : WAIT_MINUTES;
            const sleepLabel = nextSleepMinutes < 1 ? `${Math.round(nextSleepMinutes * 60)}s` : `${nextSleepMinutes}m`;
            console.log(colabStatus.active
                ? `\n⚡ [Colab G4 Burst Active: ${colabStatus.model}] Advancing in ${sleepLabel} (back-to-back datacenter cadence)...`
                : `\n⏳ Sleeping ${sleepLabel} (GPU cooldown)...`);
            await interruptibleSleep(nextSleepMinutes * 60 * 1000);
        }
    }

    // ── CLOSE: State persistence ──────────────────────────
    await runClose(cycle);
}

async function runClose(cycle) {
    console.log(`\n[${new Date().toISOString()}] ▶ PHASE: CLOSE`);

    const inflight = readInflight();
    const sessionSummary = `Jean SRE ran ${cycle - 1} cycles. Last focus: ${inflight.focus || 'unknown'}. Next goal: ${inflight.goal || 'none'}.`;

    // Programmatic save to kruschdb (always runs)
    await saveContext(sessionSummary);

    // Final LLM synthesis
    try {
        const closePrompt = buildClosePrompt(inflight.block, cycle - 1);
        const reply = await sendToAgent(closePrompt, { noTools: true });
        await persistInflight(reply, cycle);
        await extractAndSaveLessons(reply);
        await logCycle(cycle, 'CLOSE', closePrompt, reply);
        console.log(`[Close] Jean's shutdown sequence complete.`);
    } catch (e) {
        console.error('[Close] Synthesis failed (state saved to kruschdb):', e.message);
    }

    console.log('Goodbye. 👋');
}

// ─────────────────────────────────────────────────────────
// Graceful Shutdown
// ─────────────────────────────────────────────────────────

function handleShutdown(signal) {
    if (shutdownController.signal.aborted) return;
    console.log(`\n[${new Date().toISOString()}] ⚠️ ${signal} — Graceful shutdown...`);
    shutdownController.abort();
    try {
        const pidFile = join(projectRoot, 'jean_sweep.pid');
        if (fs.existsSync(pidFile)) {
            fs.unlinkSync(pidFile);
            console.log(`[Sweep] Removed PID file ${pidFile}`);
        }
    } catch (e) {
        console.error(`[Sweep] Failed to clean up PID file:`, e.message);
    }
}

export {
    parseExecutableAction,
    classifyIntentRisk,
    checkIntentDrift,
    parseMasterBacklogCards,
    isCleanPassBacklog,
    extractStaleCleanupCards,
    finalizeMasterBacklogArtifacts,
    reconcileProjectBlockers,
    buildPendingIssueCodeEvidence,
    buildProjectAstIndex,
    buildProjectSpecAndDocsBrief,
    buildProjectAuditContext,
    loadHomelabBaselineRules,
    listProjectFiles,
    routeWorkflow,
    readInflight,
    runProjectVerificationSuite,
    checkColabBurstStatus,
    distillTeacherKnowledgeForStudent,
    loadDistilledTeacherBrief,
    CORE_PROJECTS,
    HOMELAB_ROOT_TARGETS,
    PROJECT_ROTATION,
    INFRA_TARGETS,
    discoverUncuratedProjects,
    buildLegoScoutContextBrief,
    buildProjectLegoBoilerplateBrief,
    appendLegoScoutBacklogEntry,
    computeProjectSourceSignature,
    markProjectConverged,
    isProjectConverged,
    shutdownController
};

const isMainModule = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isMainModule) {
    process.on('SIGTERM', () => handleShutdown('SIGTERM'));
    process.on('SIGINT', () => handleShutdown('SIGINT'));

    // Write PID file for event-driven triggers
    try {
        fs.writeFileSync(join(projectRoot, 'jean_sweep.pid'), process.pid.toString());
        console.log(`[Sweep] Started with PID ${process.pid}. PID written to jean_sweep.pid`);
    } catch (e) {
        console.error(`[Sweep] Failed to write PID file:`, e.message);
    }

    runActiveSweep();
}
