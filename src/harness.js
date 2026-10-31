import { getHistory, saveHistory, pool } from './db.js';
import { getMcpTools, callMcpTool, getConnectedServers, initMcpClients } from './mcp-client.js';
import { chat, resolveModelConfig } from '@krusch/toolkit/llm';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { loadProjectConfig, envOr } from '@krusch/toolkit/config';
import { dirname, join } from 'path';
import os from 'os';
import { verifyToolCompliance } from './compliance.js';
import { HOMELAB_CONTEXT } from './prompts.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');
const HOMELAB_ROOT = envOr('HOMELAB_PATH', null, join(os.homedir(), 'homelab'));

export const DEFAULT_HARNESS_CONFIG = {
    models: {
        persona: { provider: 'ollama', model: 'qwen3-coder-30b-moe:latest' },
        executor: { provider: 'ollama', model: 'gemma4:e4b' },
    },
};

/**
 * Load project harness configuration, failing fast when loader throws on malformed config (BUG-01)
 * and falling back to DEFAULT_HARNESS_CONFIG when no config file is present.
 * Supports `--strict-config` CLI flag or `JEAN_STRICT_CONFIG=true`.
 */
export async function loadSafeHarnessConfig(rootDir, loader = loadProjectConfig, options = {}) {
    const strict =
        options.strict ??
        (process.argv.includes('--strict-config') || envOr('JEAN_STRICT_CONFIG', null, 'false') === 'true');
    let loaded = null;
    try {
        loaded = await loader(rootDir);
    } catch (e) {
        console.error('Failed to load project config:', e);
        if (strict || options.failFast !== false) {
            throw e;
        }
    }
    if (!loaded || typeof loaded !== 'object' || !loaded.models) {
        return structuredClone(DEFAULT_HARNESS_CONFIG);
    }
    return loaded;
}

// Load dynamic model and service configurations
const config = await loadSafeHarnessConfig(projectRoot);

const PERSONA_CONFIG = resolveModelConfig('persona', config);
const EXECUTOR_CONFIG = resolveModelConfig('executor', config);

const PERSONA_MODEL = envOr('PERSONA_MODEL', PERSONA_CONFIG.model, 'qwen3-coder-30b-moe:latest');
const PERSONA_PROVIDER = envOr('PERSONA_PROVIDER', PERSONA_CONFIG.provider, 'ollama');
const PERSONA_OLLAMA_URL = envOr(
    'PERSONA_OLLAMA_URL',
    PERSONA_CONFIG.apiUrl ? new URL(PERSONA_CONFIG.apiUrl).origin : undefined,
    'http://localhost:11434'
);

const EXECUTOR_MODEL = envOr('EXECUTOR_MODEL', EXECUTOR_CONFIG.model, 'gemma4:e4b');
const EXECUTOR_PROVIDER = envOr('EXECUTOR_PROVIDER', EXECUTOR_CONFIG.provider, 'ollama');
const EXECUTOR_OLLAMA_URL = envOr(
    'EXECUTOR_OLLAMA_URL',
    EXECUTOR_CONFIG.apiUrl ? new URL(EXECUTOR_CONFIG.apiUrl).origin : undefined,
    'http://127.0.0.1:11434'
);

// Synchronize resolved configs with env overrides
PERSONA_CONFIG.model = PERSONA_MODEL;
PERSONA_CONFIG.provider = PERSONA_PROVIDER;
PERSONA_CONFIG.apiUrl = `${PERSONA_OLLAMA_URL}/v1/chat/completions`;

EXECUTOR_CONFIG.model = EXECUTOR_MODEL;
EXECUTOR_CONFIG.provider = EXECUTOR_PROVIDER;
EXECUTOR_CONFIG.apiUrl = `${EXECUTOR_OLLAMA_URL}/v1/chat/completions`;

// ── Opportunistic Colab G4 Blackwell Piggyback (Passive Probe Only — 0 Extra CUs) ──
// Never starts a Colab VM. Only piggybacks when the SSH tunnel (:11435) is already active.
const COLAB_BURST_URLS = Array.from(new Set([
    ...envOr('COLAB_BURST_URL', null, '').split(',').map((s) => s.trim()),
    'http://127.0.0.1:11435',
    'http://localhost:11435'
].filter(Boolean)));
const COLAB_BURST_MODEL = envOr('COLAB_BURST_MODEL', null, 'gpt-oss:120b');
const COLAB_PROBE_TTL_MS = 30_000;
const COLAB_PROBE_TIMEOUT_MS = parseInt(envOr('COLAB_PROBE_TIMEOUT_MS', null, '3000'), 10) || 3000;
const _colabProbeState = {
    checkedAt: 0,
    active: false,
    baseUrl: null,
    model: COLAB_BURST_MODEL,
    inFlight: null
};

export function getBooleanEnv(name, defaultValue = false) {
    const val = process.env[name];
    if (val === undefined) return defaultValue;
    if (val === '1' || val === 'true') return true;
    if (val === '0' || val === 'false') return false;
    throw new Error(`Invalid boolean env ${name}: ${val}`);
}

export function invalidateColabBurstCache() {
    _colabProbeState.checkedAt = 0;
    _colabProbeState.active = false;
    _colabProbeState.baseUrl = null;
    _colabProbeState.model = COLAB_BURST_MODEL;
    _colabProbeState.inFlight = null;
}

export async function resolveOpportunisticPersonaConfig({ isSweepPrompt = false, forceRefresh = false } = {}) {
    if (getBooleanEnv('DISABLE_COLAB_PIGGYBACK', false)) {
        return {
            isColabBurst: false,
            baseUrl: PERSONA_OLLAMA_URL,
            model: PERSONA_MODEL,
            maxPromptChars: isSweepPrompt ? 32000 : 20000,
            maxTokens: 3072,
            config: PERSONA_CONFIG
        };
    }

    if (forceRefresh) {
        invalidateColabBurstCache();
    }

    const now = Date.now();
    if (now - _colabProbeState.checkedAt >= COLAB_PROBE_TTL_MS) {
        if (!_colabProbeState.inFlight) {
            _colabProbeState.inFlight = (async () => {
                let foundUrl = null;
                let foundModel = COLAB_BURST_MODEL;
                for (const candidateUrl of COLAB_BURST_URLS) {
                    try {
                        const res = await fetch(`${candidateUrl}/api/ps`, {
                            signal: AbortSignal.timeout(COLAB_PROBE_TIMEOUT_MS)
                        });
                        if (res.ok) {
                            const data = await res.json().catch(() => ({}));
                            const models = Array.isArray(data?.models) ? data.models : [];
                            const preferred = models.find(m => m.name === COLAB_BURST_MODEL || m.model === COLAB_BURST_MODEL) || models[0];
                            foundUrl = candidateUrl;
                            if (preferred?.name || preferred?.model) {
                                foundModel = preferred.name || preferred.model;
                            }
                            break;
                        }
                    } catch {
                        // Tunnel not running on this candidate — continue silently
                    }
                }
                _colabProbeState.checkedAt = Date.now();
                _colabProbeState.active = Boolean(foundUrl);
                _colabProbeState.baseUrl = foundUrl;
                _colabProbeState.model = foundModel;
            })().finally(() => {
                _colabProbeState.inFlight = null;
            });
        }
        await _colabProbeState.inFlight;
    }

    if (_colabProbeState.active && _colabProbeState.baseUrl) {
        return {
            isColabBurst: true,
            baseUrl: _colabProbeState.baseUrl,
            model: _colabProbeState.model,
            maxPromptChars: 64000,
            maxTokens: 4096,
            config: {
                ...PERSONA_CONFIG,
                model: _colabProbeState.model,
                provider: 'ollama',
                apiUrl: `${_colabProbeState.baseUrl}/v1/chat/completions`
            }
        };
    }

    return {
        isColabBurst: false,
        baseUrl: PERSONA_OLLAMA_URL,
        model: PERSONA_MODEL,
        maxPromptChars: isSweepPrompt ? 32000 : 20000,
        maxTokens: 3072,
        config: PERSONA_CONFIG
    };
}


const JEAN_BASE_PERSONA = `You are Jean, the Native Local Homelab Agent for the Krusch homelab fleet.
Your mission is to deeply understand every project in the homelab and continuously improve and optimize the codebase.
You proactively detect structural drift, configuration entropy, and optimization opportunities across all projects.

## RULES (MANDATORY)
1. You MUST use the tools provided to you. Call sentinel tools, codebase search, and swarm stats to gather REAL data.
2. NEVER hallucinate telemetry, metrics, or system state. If a tool call fails, say "tool unavailable" — do NOT invent data.
3. NEVER ask the user for guidance or clarification. You are AUTONOMOUS. Investigate using your tools.
4. Be TERSE. No storytelling, no anecdotes, no roleplay. Output: findings, analysis, actions.
5. DO NOT use <think> tags or output inner monologue.
6. You are a Scout — dispatch "Intent" jobs to the DBOS Swarm for analysis, do not modify code yourself.

## ANTI-HALLUCINATION (systems that DO NOT exist in this homelab)
The following DO NOT exist. NEVER mention them: nginx, kubernetes, k8s, etcd, sidecars, Prometheus, Grafana, kubelet, helm, istio.
The homelab uses: Docker Compose, Tailscale, PostgreSQL (kruschdb), Ollama, Cloudflare Tunnels.
Nodes: krusch (i7-5820K / 32GB / RTX 2080 Ti 11GB), kruschdev (Xeon 40T / 64GB ECC / Dual RTX 2080 Ti 22GB), kruschserv (Xeon 40T / 80GB / Dual RTX 3060 24GB), kruschgame (i5-11400H / 32GB / RTX 3050 4GB), kruschmedia (i7-4790S / 32GB / RX 5500 8GB).

## AVAILABLE MCP TOOL SERVERS & SUITES
1. **sentinel**:
   - Multi-node hardware telemetry, disk usage, docker container status, GPU metrics, and logs across krusch, kruschdev, and kruschserv.
2. **krusch-infra**:
   - Container health, fleet service restarts, and log tailing.
3. **krusch-git**:
   - Codebase AST symbols, dependency graphs, and hybrid semantic code search.
4. **krusch-context**:
   - Working memory, invariant retrieval, and lessons persistence.
5. **krusch-fs**:
   - Multi-node OS filesystem, configs, and page-anchored memory.
6. **krusch-frontend**:
   - Canonical frontend templates and UI validation.

## 🚀 NATIVE SELF-HEALING ROADMAPS & WORKFLOWS (Tool Chains)

### 1. Codebase Hygiene & Guidelines Evolution Self-Healing Loop
When codebase scans reveal lint warnings, broken files, or test failures:
1. \`krusch-git-search-symbols\` and \`krusch-git-dependency-graph\` to analyze symbol boundaries and blast radius.
2. \`krusch-git-read-blob\` to inspect affected code.
3. \`create_background_intent\` to dispatch isolated repair jobs to the background DBOS swarm.
4. \`krusch-context-remember\` to lock verified lessons and architectural invariants into working memory.

### 2. GPU VRAM & Fleet Service Self-Healing Loop
When hardware sweeps indicate high thermal load, VRAM saturation, or crashed containers:
1. \`sentinel-get-gpu-stats\` and \`sentinel-get-system-stats\` to verify node VRAM and thermal load.
2. \`sentinel-get-docker-containers\` and \`krusch-infra-get-container-health\` to check heartbeat status.
3. \`krusch-infra-restart-service\` to reboot crashed or anomalous services.
4. \`krusch-context-remember\` to persist performance optimizations and root-cause solutions.`;

const supportsNativeTools = (modelName) => {
    if (modelName.includes('qwen3') || modelName.includes('moe')) return false;
    return true;
};

function sanitizeMessagesForOllama(messages) {
    return messages.map(msg => {
        if (msg.role === 'tool') {
            return {
                role: 'user',
                content: `[MCP Tool Result]:\n${msg.content}`
            };
        }
        if (msg.role === 'assistant') {
            const newMsg = { ...msg };
            if (newMsg.tool_calls) {
                delete newMsg.tool_calls;
            }
            if (newMsg.content === undefined || newMsg.content === null || newMsg.content.trim() === '') {
                newMsg.content = '[Executed system diagnostics tools]';
            }
            return newMsg;
        }
        return msg;
    });
}

function formatToolsForPrompt(tools) {
    let text = "\n\nYou have access to the following tools for gathering live information. You MUST use them instead of guessing:\n\n";
    for (const tool of tools) {
        const fn = tool.function;
        text += `### Tool: ${fn.name}\n`;
        text += `Description: ${fn.description}\n`;
        text += `Parameters:\n${JSON.stringify(fn.parameters, null, 2)}\n\n`;
    }
    text += `To call a tool, you MUST output a tool call using the EXACT format below, and output NOTHING else in that turn:\n`;
    text += `<tool_call>{"name": "tool_name", "arguments": {"arg_name": "arg_value"}}</tool_call>\n\n`;
    return text;
}

const JEAN_SYNTHESIS_SYSTEM_PROMPT = `You are Jean, the Autonomous Principal SRE & Code Auditor for the Krusch homelab fleet.
RULES (NON-NEGOTIABLE):
1. Ground every finding strictly in the provided live source snippets, AST symbol graphs, and command outputs.
2. NEVER invent, paraphrase, or reconstruct code snippets from memory. Only quote verbatim lines that appear in the prompt with their exact line numbers.
3. If a prior Pending Issue is not supported by the Live Source Verification lines shown in the prompt, REFUTE and remove it.
4. When audited files satisfy all frozen specification invariants with zero P0/P1 defects, emit a calibrated CLEAN_PASS verdict ("findings": []) rather than inventing stylistic nitpicks.
5. Assemble application services from @krusch/toolkit v1.5 and krusch_toolkit v1.3 Lego bricks (INV-LEGO-01..16), and enforce the Rule-of-3 anti-bloat gate when evaluating toolkit candidates: only extract recurring infrastructure patterns shared across >= 3 repositories; keep < 3-repo or domain-specific helpers project-local (BELOW_THRESHOLD_DO_NOT_EXTRACT).
6. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.`;

const JEAN_CONSULT_SYSTEM_PROMPT = `You are Jean, the Autonomous Principal SRE & Architectural Advisor for the Krusch homelab fleet.
RULES (NON-NEGOTIABLE):
1. Answer architectural, SRE, and codebase triage questions directly in concise, structured GitHub Markdown grounded in the provided Homelab Architecture Reference and Live Audit State.
2. Recommend exact @krusch/toolkit v1.5 and krusch_toolkit v1.3 subpath imports and function names (INV-LEGO-01..16) instead of hand-rolled HTTP, SSE, MCP, SQLite/PG, SSRF/rate-limit, Webhook, or CLI boilerplate.
3. Enforce the Rule-of-3 anti-bloat gate when evaluating candidate toolkit bricks: only extract recurring infrastructure patterns shared across >= 3 repositories (CANDIDATE_MEETS_RULE_OF_3 via /jean-loop-gate); explicitly reject < 3-repo or single-domain utilities as BELOW_THRESHOLD_DO_NOT_EXTRACT.
4. Never hallucinate non-existent infrastructure (no nginx, k8s, Prometheus, or Grafana) or invent unverified file contents.
5. Do NOT output <think> tags, filler prose, or bare Referee JSON envelopes unless explicitly asked for JSON.`;

function prunePromptForContext(text, maxChars = 32000) {
    if (!text || text.length <= maxChars) return text;
    console.warn(`[Persona] Pruning input message from ${text.length} chars to ${maxChars} chars to stay within 16K context budget`);
    const headChars = Math.floor(maxChars * 0.50);
    const tailChars = Math.floor(maxChars * 0.46);
    return `${text.slice(0, headChars)}\n\n... [TRUNCATED ${text.length - (headChars + tailChars)} CHARS FOR 16K CONTEXT BUDGET] ...\n\n${text.slice(-tailChars)}`;
}

const SYNONYMS = {
    storage: ["disk", "volume", "mount", "fs", "file", "usage", "locate", "page"],
    containers: ["docker", "container", "service"],
    memory: ["ram", "vram", "swap", "context", "remember", "retrieve", "invariant", "nudge"],
    logs: ["log", "tail", "journal"],
    telemetry: ["stat", "status", "health", "metric", "gpu", "cpu", "system"],
    hardware: ["gpu", "cpu", "temp", "thermal", "governor", "mount"],
    code: ["git", "symbol", "ast", "blob", "tree", "repo", "dependency", "semantic", "search"],
    audit: ["git", "symbol", "dependency", "semantic", "context", "retrieve", "fs", "page"],
    project: ["git", "symbol", "dependency", "context", "retrieve", "fs", "locate", "page"]
};

function filterTools(query, allTools, maxTools = 8) {
    const q = (query || "").toLowerCase();
    const qWords = q.split(/[^a-z0-9_-]+/).filter(w => w.length > 2);
    const expandedQ = new Set(qWords);
    for (const w of qWords) {
        if (SYNONYMS[w]) {
            SYNONYMS[w].forEach(s => expandedQ.add(s));
        }
    }

    const scored = allTools.map(t => {
        const name = t.function.name;
        if (name === "create_background_intent" || name === "escalate_to_architect") {
            return { tool: t, score: 0.1 };
        }
        let score = 0;
        const nameWords = name.toLowerCase().split(/[-_]/).filter(w => w.length > 2);
        for (const nw of nameWords) {
            if (expandedQ.has(nw)) score += 3;
        }
        const desc = (t.function.description || "").toLowerCase();
        for (const ew of expandedQ) {
            if (desc.includes(ew)) score += 1;
        }
        return { tool: t, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const selected = scored.filter(s => s.score > 0).slice(0, maxTools).map(s => s.tool);

    if (selected.length === 0) {
        const defaults = ["sentinel-get-system-stats", "sentinel-get-docker-containers", "sentinel-get-disk-usage", "krusch-infra-get-fleet-status", "krusch-context-retrieve", "krusch-git-search-symbols"];
        for (const d of defaults) {
            const match = allTools.find(t => t.function.name === d);
            if (match) selected.push(match);
        }
    }

    return selected;
}

const CURATED_ROTATION = [
    'krusch-git', 'krusch', 'krusch-companion', 'krusch-context-mcp',
    'krusch-cascade-router', 'krusch-polygres-connector', 'krusch-law',
    'krusch-biz', 'krusch-nexus', 'compass-concierge', 'g-krusch', 'krusch-pre-router', 'krusch-fs'
];
const AUDIT_PHASE_NAMES = [
    '0_ORIENTATION', '1.1_STANDARD_MAPPING', '1.2_COMPLIANCE_MATRIX',
    '1.3_CRITICAL_PATH', '1.4_BOUNDARY_AUDIT', '1.5_ANTI_PATTERN',
    '1.6_STRUCTURAL_STRESS_TEST', '1.7_VERIFICATION_AUDIT', '1.8_RISK_MAP',
    '1.9_ADVERSARIAL_REVIEW', '2_MASTER_BACKLOG'
];

export function normalizePortableSftText(rawText, projectName = '') {
    if (!rawText || typeof rawText !== 'string') return '';
    let out = rawText;
    if (projectName && projectName !== 'fleet') {
        const escapedProj = projectName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const projAbsRegex = new RegExp(`(?:/home/(?:agent|krusch|kruschdev|kruschserv)/homelab|/app)/projects/(?:archive/)?${escapedProj}/`, 'g');
        out = out.replace(projAbsRegex, '');
        const projRootRegex = new RegExp(`(?:/home/(?:agent|krusch|kruschdev|kruschserv)/homelab|/app)/projects/(?:archive/)?${escapedProj}\\b`, 'g');
        out = out.replace(projRootRegex, '.');
    }
    out = out
        .replace(/\/home\/(?:agent|krusch|kruschdev|kruschserv)\/homelab\/projects\/archive\//g, '$WORKSPACE_ROOT/projects/archive/')
        .replace(/\/home\/(?:agent|krusch|kruschdev|kruschserv)\/homelab\/projects\//g, '$WORKSPACE_ROOT/projects/')
        .replace(/\/home\/(?:agent|krusch|kruschdev|kruschserv)\/homelab\//g, '$WORKSPACE_ROOT/')
        .replace(/\/home\/(?:agent|krusch|kruschdev|kruschserv)\/homelab\b/g, '$WORKSPACE_ROOT')
        .replace(/\/home\/(?:agent|krusch|kruschdev|kruschserv)(?:\/homel(?:ab?)?|\/home|\/hom|\/ho|\/h)?/gi, '$WORKSPACE_ROOT')
        .replace(/\/home\/(?:agen|age|ag|kruschde)/gi, '$WORKSPACE_ROOT')
        .replace(/\/app\/projects\/archive\//g, '$WORKSPACE_ROOT/projects/archive/')
        .replace(/\/app\/projects\//g, '$WORKSPACE_ROOT/projects/')
        // Redact API key / token literals
        .replace(/(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,}|AIzaSy[A-Za-z0-9_-]{33}|ghp_[A-Za-z0-9]{20,})/g, 'REDACTED_API_KEY')
        // Generalize private LAN, 10GbE, and Tailscale IPs
        .replace(/\b(?:192\.168\.\d+\.\d+|10\.10\.10\.\d+|10\.0\.0\.\d+|100\.(?:65|106|105|98|124|92|94|69)\.\d+\.\d+)\b/g, '127.0.0.1')
        // Generalize internal fleet hostnames
        .replace(/\bkruschdev\b/gi, 'compute-node')
        .replace(/\bkruschserv\b/gi, 'db-node')
        .replace(/\bkruschmedia\b/gi, 'media-node')
        .replace(/\bkruschgame\b/gi, 'sandbox-node')
        .replace(/\bkruschsandbox\b/gi, 'worker-node')
        .replace(/\bkr1yoga\b/gi, 'mobile-node')
        .replace(/\bcrush0\b/gi, 'tablet-node')
        // Generalize system prompt & homelab fleet phrasing
        .replace(/You are Jean, the Autonomous Principal SRE & Code Auditor for the Krusch homelab fleet\./gi, 'You are an Autonomous Principal Software Engineer, SRE, and Code Auditor.')
        .replace(/You are Jean, the Autonomous Principal SRE & Architectural Advisor for the Krusch homelab fleet\./gi, 'You are an Autonomous Principal Software Engineer, SRE, and Architectural Advisor.')
        .replace(/You are Jean SRE Remediator \(powered by Qwen3-Coder-30B-MoE\), an autonomous principal engineer for the Krusch homelab fleet\./gi, 'You are an Autonomous Principal Software Engineer and Code Remediator.')
        .replace(/for the Krusch homelab fleet/gi, 'for software engineering repositories')
        .replace(/\bKrusch homelab fleet\b/gi, 'software engineering workspace');
    return typeof out.toWellFormed === 'function' ? out.toWellFormed() : out;
}

export function recordTeacherSftTurn({ model, systemPrompt, userPrompt, assistantReply }) {
    try {
        const reply = String(assistantReply || '').trim();
        if (!reply || reply.length < 120 || reply.includes('HALLUCINATION FLAG')) return false;
        const distDir = join(projectRoot, 'data', 'distillation');
        fs.mkdirSync(distDir, { recursive: true });
        const sftPath = join(distDir, 'gpt_oss_120b_sft.jsonl');

        const projMatch = String(userPrompt || '').match(/Target Project:\s*`?([a-zA-Z0-9_@/-]+)`?/i)
            || String(userPrompt || '').match(/projects\/([a-zA-Z0-9_-]+)/);
        const phaseMatch = String(userPrompt || '').match(/CURRENT PHASE CHECKLIST:\s*([0-9A-Z._-]+)/i)
            || String(userPrompt || '').match(/Active Audit Phase:\s*`?([0-9A-Z._-]+)`?/i)
            || String(userPrompt || '').match(/for Phase `([0-9A-Z._-]+)`/i)
            || String(userPrompt || '').match(/Phase:\s*`?([0-9A-Z._-]+)`?/i);

        const project = projMatch ? projMatch[1] : 'fleet';
        const normSystem = normalizePortableSftText(String(systemPrompt || '').trim(), project);
        const normUser = normalizePortableSftText(String(userPrompt || '').trim(), project);
        const normReply = normalizePortableSftText(reply, project);

        const record = {
            timestamp: new Date().toISOString(),
            teacher_model: model || COLAB_BURST_MODEL,
            student_target: 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
            project,
            phase: phaseMatch ? phaseMatch[1] : 'synthesis',
            messages: [
                { role: 'system', content: normSystem },
                { role: 'user', content: normUser },
                { role: 'assistant', content: normReply }
            ]
        };
        fs.appendFileSync(sftPath, `${JSON.stringify(record)}\n`, 'utf8');
        return true;
    } catch (e) {
        console.warn(`[Distill-SFT] Failed to record 120B SFT turn: ${e.message}`);
        return false;
    }
}

function buildLiveAuditStateBrief() {
    const parts = [];
    try {
        const rotPath = join(projectRoot, 'JEAN_SRE_ROTATION.json');
        if (fs.existsSync(rotPath)) {
            const rot = JSON.parse(fs.readFileSync(rotPath, 'utf8'));
            const proj = CURATED_ROTATION[(rot.projectRotationIndex || 0) % CURATED_ROTATION.length];
            const phase = AUDIT_PHASE_NAMES[rot.currentPhaseIndex || 0] || String(rot.currentPhaseIndex);
            parts.push(`### 📊 Live Jean SRE Rotation State\n- **Active Project:** \`${proj}\` (Phase: \`${phase}\`, Rotation Index: ${rot.projectRotationIndex ?? 0})\n- **Curated ${CURATED_ROTATION.length}-Repo Rotation:** ${CURATED_ROTATION.map(p => `\`${p}\``).join(', ')}`);
        }
    } catch {}

    try {
        const infPath = join(projectRoot, 'JEAN_SRE_INFLIGHT.md');
        if (fs.existsSync(infPath)) {
            const inf = fs.readFileSync(infPath, 'utf8').trim();
            if (inf.length > 0) {
                parts.push(`### 🚀 Active In-Flight State (\`JEAN_SRE_INFLIGHT.md\`)\n${inf.slice(0, 1400)}`);
            }
        }
    } catch {}

    try {
        const projectsDir = join(HOMELAB_ROOT, 'projects');
        if (fs.existsSync(projectsDir)) {
            const recentAudits = [];
            for (const p of CURATED_ROTATION) {
                const taskPath = join(projectsDir, p, '.agent', 'audit', 'task.md');
                const backlogPath = join(projectsDir, p, '.agent', 'audit', 'master_backlog.md');
                const target = fs.existsSync(taskPath) ? taskPath : (fs.existsSync(backlogPath) ? backlogPath : null);
                if (target) {
                    try {
                        const st = fs.statSync(target);
                        recentAudits.push({ project: p, path: target, mtimeMs: st.mtimeMs });
                    } catch {}
                }
            }
            recentAudits.sort((a, b) => b.mtimeMs - a.mtimeMs);
            const summaries = [];
            for (const item of recentAudits.slice(0, 4)) {
                try {
                    const raw = fs.readFileSync(item.path, 'utf8').trim();
                    summaries.push(`#### Recent Pass: \`${item.project}\` (${new Date(item.mtimeMs).toISOString().slice(0, 16)} UTC)\n${raw.slice(0, 550)}`);
                } catch {}
            }
            if (summaries.length > 0) {
                parts.push(`### 🗂️ Recent Completed Audit Passes (\`.agent/audit/\`)\n${summaries.join('\n\n')}`);
            }
        }
    } catch {}

    return parts.join('\n\n');
}

export async function chatTurn(conversationId, userMessage, { noTools = false } = {}) {
    // ── Fast-path for stateless one-shot synthesis (noTools: true) ──
    if (noTools) {
        const isSweepPrompt = /^\s*##\s*(?:🛑\s*)?JEAN\b/i.test(userMessage || '');
        const personaTarget = await resolveOpportunisticPersonaConfig({ isSweepPrompt });
        console.log(`[Persona] Executing direct one-shot prompt via ${personaTarget.model}${personaTarget.isColabBurst ? ' [⚡ COLAB G4 PIGGYBACK — 0 EXTRA CUs]' : ''} (noTools=true)...`);
        const liveStateBrief = isSweepPrompt ? '' : buildLiveAuditStateBrief();
        const personaSystemPrompt = isSweepPrompt
            ? JEAN_SYNTHESIS_SYSTEM_PROMPT
            : `${JEAN_CONSULT_SYSTEM_PROMPT}\n\n${HOMELAB_CONTEXT}\n\n${liveStateBrief}`.trim();

        const prunedMessage = prunePromptForContext(userMessage, personaTarget.maxPromptChars);

        const synthesisMessages = [
            { role: 'system', content: personaSystemPrompt },
            { role: 'user', content: prunedMessage }
        ];

        let finalResponse;
        let usedColabTeacher = personaTarget.isColabBurst;
        try {
            console.log(`[DEBUG] Calling chat() with ${synthesisMessages.length} messages (${prunedMessage.length} chars). Model: ${personaTarget.model}, URL: ${personaTarget.config.apiUrl}`);
            finalResponse = await chat(
                null,
                null,
                {
                    ...personaTarget.config,
                    temperature: 0.2,
                    maxTokens: personaTarget.maxTokens
                },
                {
                    returnFullResponse: true,
                    messages: sanitizeMessagesForOllama(synthesisMessages)
                }
            );
            console.log(`[DEBUG] chat() returned successfully.`);
        } catch (e) {
            if (personaTarget.isColabBurst) {
                usedColabTeacher = false;
                console.warn(`[Persona] Colab G4 tunnel unreachable mid-request (${e.message}). Falling back to local ${PERSONA_MODEL}...`);
                invalidateColabBurstCache();
                const fallbackPruned = prunePromptForContext(userMessage, isSweepPrompt ? 32000 : 20000);
                finalResponse = await chat(
                    null,
                    null,
                    {
                        ...PERSONA_CONFIG,
                        temperature: 0.2,
                        maxTokens: 3072
                    },
                    {
                        returnFullResponse: true,
                        messages: sanitizeMessagesForOllama([
                            { role: 'system', content: personaSystemPrompt },
                            { role: 'user', content: fallbackPruned }
                        ])
                    }
                );
            } else {
                console.error(`[DEBUG] chat() threw an error: ${e.message}`);
                throw new Error(`Persona LLM Error: ${e.message}`);
            }
        }

        if (usedColabTeacher && finalResponse?.text) {
            recordTeacherSftTurn({
                model: personaTarget.model,
                systemPrompt: personaSystemPrompt,
                userPrompt: prunedMessage,
                assistantReply: finalResponse.text
            });
        }

        return finalResponse.text || '';
    }

    let messages = await getHistory(conversationId);
    
    if (messages.length === 0) {
        const liveStateBrief = buildLiveAuditStateBrief();
        messages.push({
            role: 'system',
            content: `${JEAN_BASE_PERSONA}${liveStateBrief ? `\n\n${liveStateBrief}` : ''}\n\nYou are the EXECUTOR engine. You MUST use your provided homelab tools (Sentinel, PG-Git, Node Health, etc.) to gather live telemetry and read codebase files before answering. DO NOT hallucinate telemetry or file contents. Execute tools to fetch the actual data first. If you dispatch an intent to the background or escalate to the architect, you MUST still provide a direct conversational response. In that response, explicitly answer what you currently can, state clearly what information you are waiting for from the background/architect, and explain why that delegation was necessary.`
        });
    }

    messages.push({ role: 'user', content: userMessage });

    let allMcpTools = [];
    try {
        allMcpTools = await getMcpTools();
    } catch (e) {
        console.error("Warning: Failed to fetch MCP tools", e.message);
    }

    // Filter tools based on query intent within a safe token budget (<= 8 tools)
    let tools = filterTools(userMessage, allMcpTools, 8);

    // Inject "Phone a Friend" and Intent Dispatch tools for the 7B front-line agent
    tools.push({
        type: "function",
        function: {
            name: "create_background_intent",
            description: "Dispatch a heavy task to the background DBOS swarm. Use for large audits, complex changes, or heavy research.",
            parameters: {
                type: "object",
                properties: { task: { type: "string", description: "Detailed instructions for the background worker." } },
                required: ["task"]
            }
        }
    });
    tools.push({
        type: "function",
        function: {
            name: "escalate_to_architect",
            description: "Escalate immediately to the 30B Deep Reasoning model (Phone a Friend). Use this if the user asks a complex architectural question requiring deep reasoning.",
            parameters: {
                type: "object",
                properties: { reason: { type: "string", description: "Reason for escalation to the architect." } },
                required: ["reason"]
            }
        }
    });

    const MAX_TURNS = 3;
    let gatheredContext = false;
    let requiresPhase2 = false;

    // Check if executor supports native tools
    const nativeToolsSupported = supportsNativeTools(EXECUTOR_MODEL);
    
    if (tools.length > 0 && !nativeToolsSupported && messages.length > 0 && messages[0].role === 'system') {
        if (!messages[0].content.includes('You have access to the following tools')) {
            messages[0].content += formatToolsForPrompt(tools);
        }
    }

    // Phase 1: Tool Caller (Hermes / Gemma)
    for (let turn = 0; turn < MAX_TURNS; turn++) {
        console.log(`[Executor Turn ${turn+1}] Requesting completion from ${EXECUTOR_MODEL}...`);
        
        let response;
        try {
            response = await chat(
                null, 
                null, 
                {
                    ...EXECUTOR_CONFIG,
                    temperature: 0.1,
                    maxTokens: 8192,
                    tools: (tools.length > 0 && nativeToolsSupported) ? tools : undefined
                },
                {
                    returnFullResponse: true,
                    messages: sanitizeMessagesForOllama(messages)
                }
            );
        } catch (e) {
            throw new Error(`Executor LLM Error: ${e.message}`);
        }

        let parsedToolCalls = response.tool_calls;
        let responseText = response.text || '';

        if (!parsedToolCalls && responseText.includes('<tool_call>')) {
            const match = responseText.match(/<tool_call>([\s\S]*?)<\/tool_call>/);
            if (match) {
                try {
                    const parsed = JSON.parse(match[1]);
                    parsedToolCalls = [{
                        id: 'call_' + Math.random().toString(36).substring(2, 11),
                        type: 'function',
                        function: {
                            name: parsed.name,
                            arguments: typeof parsed.arguments === 'string' ? parsed.arguments : JSON.stringify(parsed.arguments || {})
                        }
                    }];
                    responseText = responseText.replace(match[0], '').trim();
                } catch (e) {
                    console.error(`[Executor] Failed to parse XML tool call: ${e.message}`);
                }
            }
        }

        let cleanedResponse = responseText.trim();
        if (cleanedResponse.startsWith('```')) {
            const lines = cleanedResponse.split('\n');
            if (lines[0].startsWith('```')) {
                lines.shift();
            }
            if (lines[lines.length - 1].startsWith('```')) {
                lines.pop();
            }
            cleanedResponse = lines.join('\n').trim();
        }

        if (!parsedToolCalls && cleanedResponse.includes('{') && cleanedResponse.includes('}')) {
            try {
                // Strip comments to harden against LLMs that inject inline comments in JSON
                const cleanedJson = cleanedResponse
                    .replace(/(?:^|[^:])\/\/.*$/gm, (m) => m.startsWith('//') ? '' : m.slice(0, m.indexOf('//')))
                    .replace(/\/\*[\s\S]*?\*\//g, '')
                    .trim();

                const extractedCalls = [];

                // First try direct parse of the whole cleaned block
                try {
                    const parsed = JSON.parse(cleanedJson);
                    if (Array.isArray(parsed)) {
                        for (const item of parsed) {
                            if (item && typeof item === 'object' && item.name && item.arguments) {
                                extractedCalls.push(item);
                            }
                        }
                    } else if (parsed && typeof parsed === 'object' && parsed.name && parsed.arguments) {
                        extractedCalls.push(parsed);
                    }
                } catch {
                    // Fallback to character scanner to extract all JSON objects or arrays
                    let braceCount = 0;
                    let bracketCount = 0;
                    let inString = false;
                    let escape = false;
                    let startIdx = -1;

                    for (let i = 0; i < cleanedJson.length; i++) {
                        const char = cleanedJson[i];
                        if (escape) {
                            escape = false;
                            continue;
                        }
                        if (char === '\\') {
                            escape = true;
                            continue;
                        }
                        if (char === '"') {
                            inString = !inString;
                            continue;
                        }
                        if (inString) {
                            continue;
                        }
                        if (char === '{') {
                            if (braceCount === 0 && bracketCount === 0) {
                                startIdx = i;
                            }
                            braceCount++;
                        } else if (char === '}') {
                            braceCount--;
                            if (braceCount === 0 && bracketCount === 0 && startIdx !== -1) {
                                const candidate = cleanedJson.slice(startIdx, i + 1);
                                try {
                                    const parsed = JSON.parse(candidate);
                                    if (parsed && typeof parsed === 'object' && parsed.name && parsed.arguments) {
                                        extractedCalls.push(parsed);
                                    }
                                } catch {}
                                startIdx = -1;
                            }
                        } else if (char === '[') {
                            if (braceCount === 0 && bracketCount === 0) {
                                startIdx = i;
                            }
                            bracketCount++;
                        } else if (char === ']') {
                            bracketCount--;
                            if (braceCount === 0 && bracketCount === 0 && startIdx !== -1) {
                                const candidate = cleanedJson.slice(startIdx, i + 1);
                                try {
                                    const parsed = JSON.parse(candidate);
                                    if (Array.isArray(parsed)) {
                                        for (const item of parsed) {
                                            if (item && typeof item === 'object' && item.name && item.arguments) {
                                                extractedCalls.push(item);
                                            }
                                        }
                                    }
                                } catch {}
                                startIdx = -1;
                            }
                        }
                    }
                }

                if (extractedCalls.length > 0) {
                    parsedToolCalls = extractedCalls.map(p => ({
                        id: 'call_' + Math.random().toString(36).substring(2, 11),
                        type: 'function',
                        function: {
                            name: p.name,
                            arguments: typeof p.arguments === 'string' ? p.arguments : JSON.stringify(p.arguments)
                        }
                    }));
                    responseText = '';
                }
            } catch (e) {
                console.error(`[Executor] Failed to parse JSON tool call: ${e.message}. Raw content was: ${JSON.stringify(cleanedResponse)}`);
            }
        }

        const assistantMsg = {
            role: 'assistant',
            content: responseText,
            ...(parsedToolCalls ? { tool_calls: parsedToolCalls } : {})
        };
        
        messages.push(assistantMsg);

        if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
            gatheredContext = true;
            for (const toolCall of assistantMsg.tool_calls) {
                const fnName = toolCall.function.name;
                console.log(`[Executor Tool Call] Executing ${fnName}...`);
                
                let fnArgs = {};
                try {
                    fnArgs = typeof toolCall.function.arguments === 'string'
                        ? JSON.parse(toolCall.function.arguments)
                        : toolCall.function.arguments;
                } catch(e) {
                    console.error(`[Executor] Failed to parse tool arguments for ${fnName}: ${e.message}`);
                }

                let resultText = '';
                const complianceResult = verifyToolCompliance(fnName, fnArgs);
                if (!complianceResult.valid) {
                    console.warn(`[Compliance Gate] 🛑 Tool call ${fnName} blocked: ${complianceResult.reason}`);
                    resultText = `[COMPLIANCE VIOLATION]: Tool call '${fnName}' blocked. Reason: ${complianceResult.reason}. Please self-correct your arguments or choice of directories and try again.`;
                } else if (fnName === 'create_background_intent') {
                    try {
                        if (!pool) throw new Error('Database pool not configured (DBOS_DATABASE_URL missing)');
                        const jobId = (await import('crypto')).randomUUID();
                        const payload = {
                            task: 'jean_sre_investigation',
                            instructions: fnArgs.task,
                            context: { dispatched_by: 'jean_7b_chat' }
                        };
                        await pool.query(
                            `INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, status, started_at_ms)
                             VALUES ($1, $2, 'intent', $3, 'pending', $4)`,
                            [jobId, jobId, JSON.stringify(payload), Date.now()]
                        );
                        resultText = `Background intent created with job_id ${jobId.slice(0, 8)}. Tell the user it has been successfully dispatched.`;
                    } catch (e) {
                        resultText = `Failed to create intent: ${e.message}`;
                    }
                } else if (fnName === 'escalate_to_architect') {
                    requiresPhase2 = true;
                    resultText = `Escalation approved. The 30B architect will now take over. Stop generating and let Phase 2 run.`;
                } else {
                    try {
                        resultText = await callMcpTool(fnName, fnArgs);
                    } catch (e) {
                        resultText = `Tool execution failed: ${e.message}`;
                    }
                }
                
                console.log(`[Executor Tool Result] ${fnName} returned ${resultText.length} characters.`);

                messages.push({
                    role: 'tool',
                    tool_call_id: toolCall.id,
                    content: resultText.slice(0, 2000)
                });
            }
        } else {
            // Executor decided no more tools are needed.
            if (requiresPhase2) {
                messages.pop();
            }
            break;
        }
    }

    // ── Phase 2: Architect / Synthesizer (30B) ──
    if (!requiresPhase2) {
        // Ensure the final message is an assistant response and has meaningful content.
        // If we ran out of turns, or the last message is a tool result, or the executor response is empty/too short after running tools,
        // we force an escalation so the 30B model can summarize/synthesize.
        const lastMsg = messages[messages.length - 1];
        const isEmptyOrTooShort = lastMsg && lastMsg.role === 'assistant' && (!lastMsg.content || lastMsg.content.trim().length < 10);
        if (!lastMsg || lastMsg.role !== 'assistant' || isEmptyOrTooShort) {
            console.log(`[Router] Trailing tool outputs or empty/short executor response. Forcing Phase 2 30B synthesis.`);
            requiresPhase2 = true;
        } else {
            console.log(`[Router] 7B Executor handled the request directly. Bypassing 30B Architect.`);
            if (messages.length > 15) {
                messages = [messages[0], ...messages.slice(-8)];
            }
            await saveHistory(conversationId, messages);
            return messages[messages.length - 1].content;
        }
    }

    // Always append the tracking directive as a NEW user message at the very end of the context!
    // This ensures it executes even if the executor loop exhausts all turns.
    const personaTarget = await resolveOpportunisticPersonaConfig({ isSweepPrompt: false });
    console.log(`[Persona] Requesting final synthesis from ${personaTarget.model}${personaTarget.isColabBurst ? ' [⚡ COLAB G4 PIGGYBACK — 0 EXTRA CUs]' : ''}...`);
    
    // Create a synthesis prompt
    let synthesisMessages = [...messages];
    
    synthesisMessages.push({
        role: 'user',
        content: `Please synthesize the findings from the tool executions above and provide the final professional response.`
    });
    
    // Ensure the system prompt for Qwen Coder 7B focuses on synthesis AND strict state tracking
    if (synthesisMessages.length > 0 && synthesisMessages[0].role === 'system') {
        const synthesisDirective = `${JEAN_BASE_PERSONA} A background execution engine has gathered raw telemetry and data via tools. Review the conversation and tool outputs, specifically hunting for structural drift, configuration entropy, and deviations from the optimal baseline. By constantly asking "how" and "why" state has drifted, uncover ways to harden systems. Provide a professional, precise, and analytical response. Do not attempt to use tools.`;
        synthesisMessages[0].content = `${synthesisDirective}\n\nCRITICAL STATE TRACKING: At the very end of your response, you MUST include a state block formatted exactly like this:\n### JEAN_INFLIGHT\n- **Current Focus:** [what you are monitoring]\n- **Next Cycle Goal:** [what you plan to do next 10m]\n- **Pending Issues:** [any unresolved anomalies]`;
    }

    let finalResponse;
    try {
        console.log(`[DEBUG] Calling chat() with ${synthesisMessages.length} messages. Model: ${personaTarget.model}, URL: ${personaTarget.config.apiUrl}`);
        finalResponse = await chat(
            null, 
            null, 
            {
                ...personaTarget.config,
                temperature: 0.2,
                maxTokens: personaTarget.maxTokens
            },
            {
                returnFullResponse: true,
                messages: sanitizeMessagesForOllama(synthesisMessages)
            }
        );
        console.log(`[DEBUG] chat() returned successfully.`);
    } catch (e) {
        if (personaTarget.isColabBurst) {
            console.warn(`[Persona] Colab G4 tunnel unreachable mid-request (${e.message}). Falling back to local ${PERSONA_MODEL}...`);
            invalidateColabBurstCache();
            finalResponse = await chat(
                null,
                null,
                {
                    ...PERSONA_CONFIG,
                    temperature: 0.2,
                    maxTokens: 2048
                },
                {
                    returnFullResponse: true,
                    messages: sanitizeMessagesForOllama(synthesisMessages)
                }
            );
        } else {
            console.error(`[DEBUG] chat() threw an error: ${e.message}`);
            throw new Error(`Persona LLM Error: ${e.message}`);
        }
    }

    const finalMsg = {
        role: 'assistant',
        content: finalResponse.text || ''
    };
    
    messages.push(finalMsg);

    if (messages.length > 15) {
        messages = [messages[0], ...messages.slice(-8)];
    }
    
    await saveHistory(conversationId, messages);
    return finalMsg.content;
}

const isMainModule = process.argv[1] === fileURLToPath(import.meta.url);

if (isMainModule) {
    if (process.argv.includes('--cli')) {
        const queryIndex = process.argv.indexOf('--cli') + 1;
        const query = process.argv.slice(queryIndex).join(' ') || 'How does auth work in berean?';
        const uuid = 'a123e456-789b-12d3-a456-426614174000'; 
        console.log(`\nUser: ${query}\n`);
        
        chatTurn(uuid, query)
            .then(res => console.log(`\nHermes 3: ${res}\n`))
            .catch(console.error)
            .finally(() => process.exit(0));
    } else {
        const app = express();
        app.use(express.json({ limit: '2mb' }));

        const TRUSTED_FLEET_IPS = new Set([
            '127.0.0.1', '::1', '::ffff:127.0.0.1',
            ...(process.env.TRUSTED_FLEET_IPS || '').split(',').map((s) => s.trim()).filter(Boolean)
        ]);

        function isTrustedFleetIp(rawIp = '') {
            const ip = String(rawIp).replace(/^::ffff:/, '').trim();
            if (TRUSTED_FLEET_IPS.has(ip) || TRUSTED_FLEET_IPS.has(rawIp)) return true;
            // Allow local RFC1918 LAN (192.168.0.0/16, 10.0.0.0/8, 172.16.0.0/12) & Tailscale CGNAT (100.64.0.0/10)
            if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(ip)) return true;
            if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(ip)) return true;
            if (/^172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}$/.test(ip)) return true;
            if (/^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}$/.test(ip)) return true;
            return false;
        }

        function requireJeanAuth(req, res, next) {
            // 1. Block browser cross-site / drive-by CSRF requests while allowing trusted origins
            const trustedOrigin = process.env.TRUSTED_ORIGIN || '';
            const isCrossSite = req.headers['sec-fetch-site'] === 'cross-site';
            const hasOrigin = !!req.headers.origin;
            if (isCrossSite || (hasOrigin && req.headers.origin !== trustedOrigin)) {
                return res.status(403).json({ error: 'Cross-origin browser requests to Jean SRE harness are prohibited.' });
            }

            // 2. Enforce Trusted Fleet IP boundary
            const remoteIp = req.socket?.remoteAddress || req.ip || '';
            if (!isTrustedFleetIp(remoteIp)) {
                console.warn(`[Security Gate] 🛑 Rejected untrusted IP ${remoteIp} on ${req.path}`);
                return res.status(403).json({ error: 'Forbidden: Client IP is not an authorized homelab fleet node.' });
            }

            // 3. Enforce JEAN_API_KEY if configured
            const configuredKey = process.env.JEAN_API_KEY;
            if (configuredKey) {
                const authHeader = req.headers.authorization || '';
                const token = authHeader.startsWith('Bearer ')
                    ? authHeader.slice(7).trim()
                    : String(req.headers['x-api-key'] || '').trim();
                if (token !== configuredKey) {
                    return res.status(401).json({ error: 'Unauthorized: Valid JEAN_API_KEY required.' });
                }
            }
            next();
        }

        app.post('/chat', requireJeanAuth, async (req, res) => {
            try {
                const { conversationId, message, noTools } = req.body;
                if (!conversationId || !message) {
                    return res.status(400).json({ error: 'conversationId and message are required' });
                }
                const reply = await chatTurn(conversationId, message, { noTools: !!noTools });
                res.json({ reply });
            } catch (error) {
                console.error("Chat error:", error);
                res.status(500).json({ error: error.message });
            }
        });

        // Direct MCP tool calling — bypasses the LLM for deterministic tool execution.
        // Used by jean_sre_sweep.js to programmatically load context (Ralph loop pattern).
        app.post('/tool', requireJeanAuth, async (req, res) => {
            try {
                const { name, args } = req.body;
                if (!name) {
                    return res.status(400).json({ error: 'name is required' });
                }
                const complianceResult = verifyToolCompliance(name, args || {});
                if (!complianceResult.valid) {
                    console.warn(`[Compliance Gate] 🛑 /tool call ${name} blocked: ${complianceResult.reason}`);
                    return res.status(403).json({ error: `[COMPLIANCE VIOLATION]: ${complianceResult.reason}` });
                }
                const result = await callMcpTool(name, args || {});
                res.json({ result });
            } catch (error) {
                console.error('Tool call error:', error.message);
                res.status(500).json({ error: error.message });
            }
        });

        // Event-driven emergency trigger endpoint
        app.post('/trigger', requireJeanAuth, async (req, res) => {
            try {
                const { type, node, description } = req.body;
                if (!type || !node || !description) {
                    return res.status(400).json({ error: 'type, node, and description are required' });
                }

                console.log(`\n[Trigger] 🚨 EMERGENCY RECEIVED: [${type}] on ${node} - ${description}`);

                // 1. Write active-emergency nugget for cross-system orientation
                try {
                    await callMcpTool('krusch_context_remember', {
                        key: 'active-emergency',
                        content: JSON.stringify({
                            type,
                            node,
                            description,
                            timestamp: new Date().toISOString()
                        }),
                        project: 'jean-sre'
                    });
                    console.log(`[Trigger] Emergency nugget persisted to memory.`);
                } catch (e) {
                    console.error(`[Trigger] Failed to write emergency nugget:`, e.message);
                }

                // 2. Read PID of jean-sweep and send SIGUSR1 to wake it up immediately
                let signaled = false;
                try {
                    const pidFile = join(projectRoot, 'jean_sweep.pid');
                    if (fs.existsSync(pidFile)) {
                        const pidStr = fs.readFileSync(pidFile, 'utf8');
                        const pid = parseInt(pidStr.trim(), 10);
                        if (pid) {
                            process.kill(pid, 'SIGUSR1');
                            console.log(`[Trigger] Sent SIGUSR1 to jean-sweep PID ${pid}`);
                            signaled = true;
                        }
                    } else {
                        console.warn(`[Trigger] jean_sweep.pid file not found at ${pidFile}`);
                    }
                } catch (e) {
                    console.error(`[Trigger] Failed to signal jean-sweep:`, e.message);
                }

                res.json({
                    status: 'success',
                    message: 'Emergency trigger received and dispatched.',
                    signaled
                });

            } catch (error) {
                console.error("Trigger endpoint error:", error);
                res.status(500).json({ error: error.message });
            }
        });

        app.get('/health', async (req, res) => {
            const servers = getConnectedServers();
            const forceRefresh = req.query?.refresh === '1' || req.query?.refresh === 'true';
            const personaTarget = await resolveOpportunisticPersonaConfig({ isSweepPrompt: false, forceRefresh });
            let sweepSignaled = false;
            if (forceRefresh && personaTarget.isColabBurst) {
                try {
                    const pidFile = join(projectRoot, 'jean_sweep.pid');
                    if (fs.existsSync(pidFile)) {
                        const pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim(), 10);
                        if (pid) {
                            process.kill(pid, 'SIGUSR1');
                            sweepSignaled = true;
                            console.log(`[Health] ?refresh=1 activated Colab G4 Burst (${personaTarget.model} @ ${personaTarget.baseUrl}) and woke jean-sweep PID ${pid}`);
                        }
                    }
                } catch {}
            }
            res.json({ 
                status: 'ok', 
                mcp_servers: servers,
                ready: servers.length > 0,
                persona_model: personaTarget.model,
                executor_model: EXECUTOR_MODEL,
                colab_piggyback: {
                    active: personaTarget.isColabBurst,
                    endpoint: personaTarget.baseUrl,
                    model: personaTarget.model,
                    max_prompt_chars: personaTarget.maxPromptChars,
                    sweep_signaled: sweepSignaled
                }
            });
        });

        const port = process.env.PORT || 3000;
        app.listen(port, async () => {
            console.log(`Personal Agent harness running on port ${port}`);
            console.log(`[Config] Persona (Thinker): ${PERSONA_MODEL} @ ${PERSONA_OLLAMA_URL}`);
            console.log(`[Config] Colab G4 Piggyback Candidates: ${COLAB_BURST_URLS.join(', ')} (${COLAB_BURST_MODEL})`);
            console.log(`[Config] Executor (Worker): ${EXECUTOR_MODEL} @ ${EXECUTOR_OLLAMA_URL}`);
            console.log(`[Config] Embeddings (${process.env.EMBED_MODEL || 'snowflake-arctic-embed2:latest'}): ${process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434'}`);
            try {
                await initMcpClients();
                console.log(`MCP clients initialized successfully on startup.`);
            } catch (e) {
                console.error(`Failed to initialize MCP clients:`, e);
            }
        });
    }
}
