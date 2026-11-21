/**
 * @module state-graph
 * @description Database-First Persistent Mind for Jean SRE backed by HydraDB v0.4.0 +
 * `@hydradb/selfgraph` Personalized PageRank (`strategy: "ppr"`, `contradiction: "supersede"`)
 * and `kruschdb` PostgreSQL (`ide_agent_memory`).
 *
 * Jean IS the persistent graph state; the 3 GPU chairs (`gemma4:e2b`, `qwen3-coder-30b-moe`,
 * `inkling-2-mini`) are stateless compute co-processors leased by difficulty via the Cascade Router
 * to read conclusions from — and commit new verified state transitions into — HydraDB + kruschdb.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { envOr } from '@krusch/toolkit/config';
import { okEnvelope, errorEnvelope } from '@krusch/toolkit/http';
import { pool, normalizeConversationId, setBoundedMemory, getBoundedMemory } from './db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MONOREPO_HYDRASCOPE_DIR = path.resolve(__dirname, '..', '..', 'hydrascope');

let _selfGraphSdk = undefined;
let _activeAdapterPromise = null;
const _inMemoryTransitions = [];
const MAX_LOCAL_TRANSITIONS = 64;

function loadHydraEnvFallback() {
    const envPath = path.join(MONOREPO_HYDRASCOPE_DIR, '.env');
    const out = {};
    if (!fs.existsSync(envPath)) return out;
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
            if (k) out[k] = v;
        }
    } catch {
        // Ignore unreadable .env fallback
    }
    return out;
}

async function loadSelfGraphSdk() {
    if (_selfGraphSdk !== undefined) return _selfGraphSdk;
    try {
        _selfGraphSdk = await import('@hydradb/selfgraph');
        return _selfGraphSdk;
    } catch {
        const siblingDist = path.join(
            MONOREPO_HYDRASCOPE_DIR,
            'node_modules',
            '@hydradb',
            'selfgraph',
            'dist',
            'index.js'
        );
        if (fs.existsSync(siblingDist)) {
            try {
                _selfGraphSdk = await import(pathToFileURL(siblingDist).href);
                return _selfGraphSdk;
            } catch {
                _selfGraphSdk = null;
            }
        } else {
            _selfGraphSdk = null;
        }
    }
    return _selfGraphSdk;
}

function slugifyId(raw, fallback = 'global') {
    const clean = String(raw || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return clean.slice(0, 64) || fallback;
}

/**
 * Extract target project and phase from a sweep/chat prompt when not passed explicitly.
 *
 * @param {string} promptText
 * @param {object} [hints={}]
 * @returns {{ targetProject: string, phase: string }}
 */
export function inferJeanTargetAndPhase(promptText = '', hints = {}) {
    const text = String(promptText || '');
    const explicitProj = hints.targetProject ? String(hints.targetProject).trim() : '';
    const explicitPhase = hints.phase ? String(hints.phase).trim() : '';

    const projMatch =
        explicitProj ||
        text.match(/PROJECT:\s*\*\*([^*]+)\*\*/i)?.[1] ||
        text.match(/Target Project:\s*`?([A-Za-z0-9_@/-]+)`?/i)?.[1] ||
        text.match(/INFRA MODE[^:]*:\s*([A-Za-z0-9_-]+)/i)?.[1] ||
        text.match(/\bprojects\/([A-Za-z0-9_-]+)/)?.[1] ||
        'fleet';

    const phaseMatch =
        explicitPhase ||
        text.match(/CURRENT OBJECTIVE:\s*Phase\s+([0-9A-Z._-]+)/i)?.[1] ||
        text.match(/CURRENT PHASE CHECKLIST:\s*([0-9A-Z._-]+)/i)?.[1] ||
        text.match(/PHASE:\s*([0-9A-Z._-]+)/i)?.[1] ||
        text.match(/##\s*JEAN SRE:\s*([A-Z_ ]+)/i)?.[1]?.trim()?.replace(/\s+/g, '_') ||
        'INTERACTIVE_CHAT';

    return {
        targetProject: String(projMatch).trim(),
        phase: String(phaseMatch).trim(),
    };
}

/**
 * Compute deterministic difficulty `[0, 1]` for routing Jean's state transition across the 3 Chairs:
 *   - Easy (`<= 0.35` -> 3rd Chair `gemma4:e2b`): `0_ORIENTATION`, `SESSION_OPEN`, quick status/triage
 *   - Medium (`0.36 - 0.68` -> 2nd Chair `qwen3-coder-30b-moe`): Phases `1.1..1.5`, infra & companion checks, code queries
 *   - Hard (`> 0.68` -> 1st Chair `inkling-2-mini`): Phases `1.6..1.9`, `2_MASTER_BACKLOG`, emergencies, escalated synthesis
 *
 * @param {string} promptText
 * @param {object} [options={}]
 * @returns {{ complexityScore: number, difficultyBand: 'easy'|'medium'|'hard', targetProject: string, phase: string }}
 */
export function computeJeanTurnDifficulty(promptText = '', options = {}) {
    if (typeof options.complexityScore === 'number' && Number.isFinite(options.complexityScore)) {
        const clamped = Math.min(1, Math.max(0, options.complexityScore));
        const { targetProject, phase } = inferJeanTargetAndPhase(promptText, options);
        return {
            complexityScore: clamped,
            difficultyBand: clamped > 0.68 ? 'hard' : clamped > 0.35 ? 'medium' : 'easy',
            targetProject,
            phase,
        };
    }

    const text = String(promptText || '');
    const { targetProject, phase } = inferJeanTargetAndPhase(text, options);
    const upperPhase = phase.toUpperCase();

    if (options.forceArchitect === true || /EMERGENCY DIRECTIVE/i.test(text)) {
        return { complexityScore: 0.9, difficultyBand: 'hard', targetProject, phase };
    }

    // Explicit 11-phase sweep & infra difficulty bands
    if (upperPhase === '0_ORIENTATION' || upperPhase === 'SESSION_OPEN') {
        return { complexityScore: 0.24, difficultyBand: 'easy', targetProject, phase };
    }
    if (/^1\.[1-5]_/.test(upperPhase) || /INFRA|COMPANION/i.test(upperPhase)) {
        if (/lego_scout/i.test(text)) {
            return { complexityScore: 0.78, difficultyBand: 'hard', targetProject, phase };
        }
        return { complexityScore: 0.52, difficultyBand: 'medium', targetProject, phase };
    }
    if (
        /^1\.[6-9]_/.test(upperPhase) ||
        upperPhase === '2_MASTER_BACKLOG' ||
        upperPhase === 'SYNTHESIS'
    ) {
        return { complexityScore: 0.86, difficultyBand: 'hard', targetProject, phase };
    }

    // Interactive / consult prompt difficulty scoring
    if (
        /\b(?:architectur(?:e|al)|root\s*cause|multi-file|blast\s*radius|race\s*condition|deadlock|compose|pipeline|security\s*audit|adversarial|master\s*backlog)\b/i.test(
            text
        ) ||
        text.length > 5000
    ) {
        return { complexityScore: 0.84, difficultyBand: 'hard', targetProject, phase };
    }
    if (
        /\b(?:function|class|module|sql|query|docker|container|refactor|adapter|binding|schema|test|bug|error|grep|symbol)\b/i.test(
            text
        ) ||
        text.length > 320
    ) {
        return { complexityScore: 0.52, difficultyBand: 'medium', targetProject, phase };
    }
    return { complexityScore: 0.22, difficultyBand: 'easy', targetProject, phase };
}

/**
 * Parse structured conclusions from any Chair's response (`JEAN_INFLIGHT`, `LESSON`, `HYPOTHESIS VERDICT`, `CLEAN_PASS`).
 *
 * @param {string} replyText
 * @param {object} [meta={}]
 */
export function extractStructuredJeanConclusion(replyText = '', meta = {}) {
    const raw = String(replyText || '').trim();
    const inflightMatch = raw.match(/###\s*JEAN_INFLIGHT([\s\S]*)/i);
    const inflightBody = inflightMatch ? inflightMatch[1].trim() : '';

    const focus =
        inflightBody.match(/-\s*\*\*Current Focus:?\*\*\s*([^\n]+)/i)?.[1]?.trim() ||
        meta.defaultFocus ||
        `Monitoring ${meta.targetProject || 'fleet'} (${meta.phase || 'active'})`;
    const nextGoal =
        inflightBody.match(/-\s*\*\*Next Cycle Goal:?\*\*\s*([^\n]+)/i)?.[1]?.trim() ||
        meta.defaultGoal ||
        `Advance ${meta.targetProject || 'fleet'} state`;
    const pendingIssues =
        inflightBody.match(/-\s*\*\*Pending Issues:?\*\*\s*([^\n]+)/i)?.[1]?.trim() || 'None';

    const lessonMatch = raw.match(/###\s*LESSON[^\n]*\n+([\s\S]*?)(?=\n###|\n---|$)/i);
    const lesson = lessonMatch ? lessonMatch[1].trim().replace(/\s+/g, ' ').slice(0, 320) : '';

    const verdictMatch = raw.match(/###\s*HYPOTHESIS VERDICT[^\n]*\n+([^\n#]+)/i);
    const hypothesisVerdict = verdictMatch ? verdictMatch[1].trim().slice(0, 220) : '';

    const isCleanPass = /\bCLEAN_PASS\b/i.test(raw);
    const summarySentence =
        lesson ||
        hypothesisVerdict ||
        raw
            .replace(/###\s*JEAN_INFLIGHT[\s\S]*/i, '')
            .replace(/[#*`>-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, 280) ||
        focus;

    return {
        focus: focus.slice(0, 240),
        nextGoal: nextGoal.slice(0, 240),
        pendingIssues: pendingIssues.slice(0, 320),
        lesson,
        hypothesisVerdict,
        verdict: isCleanPass ? 'CLEAN_PASS' : 'UPDATED',
        summary: summarySentence,
    };
}

/**
 * Lazily initialize the shared `@hydradb/selfgraph` instance connected to HydraDB over Bolt.
 */
export async function getJeanSelfGraphAdapter() {
    if (process.env.NODE_ENV === 'test' && process.env.ENABLE_LIVE_SELFGRAPH_IN_TEST !== '1') {
        return null;
    }
    if (_activeAdapterPromise) return _activeAdapterPromise;

    _activeAdapterPromise = (async () => {
        const sdk = await loadSelfGraphSdk();
        if (!sdk) return null;

        const fallbackEnv = loadHydraEnvFallback();
        const boltUrl = envOr(
            'HYDRADB_BOLT_URL',
            fallbackEnv.HYDRADB_BOLT_URL,
            'bolt://127.0.0.1:17688'
        );
        const boltToken = envOr(
            'HYDRADB_GRAPH_TOKEN',
            fallbackEnv.HYDRADB_GRAPH_TOKEN,
            ''
        );
        const namespace = envOr(
            'HYDRADB_NAMESPACE',
            fallbackEnv.HYDRADB_NAMESPACE,
            'default'
        );

        if (!boltUrl || !boltToken) return null;

        try {
            const { HydraDBStore, Schema, SelfGraph, toJsonable } = sdk;
            const schema = Schema.parse({
                entities: [
                    { name: 'JeanMind', aliases: ['AgentMind', 'PersistentMind'] },
                    { name: 'ProjectTarget', aliases: ['Project', 'Repo', 'Node'] },
                    { name: 'StateConclusion', aliases: ['Conclusion', 'StateTransition', 'Finding'] },
                    { name: 'ComputeChair', aliases: ['Model', 'Chair'] },
                    { name: 'GoldenBrick', aliases: ['Brick', 'SealedBrick'] },
                ],
                relations: [
                    {
                        type: 'CURRENT_FOCUS_OF',
                        from: { types: ['JeanMind'], max_count: 1 },
                        to: { types: ['StateConclusion'] },
                    },
                    {
                        type: 'ACTIVE_CONCLUSION_FOR',
                        from: { types: ['ProjectTarget'], max_count: 1 },
                        to: { types: ['StateConclusion'] },
                    },
                    {
                        type: 'DERIVED_BY_CHAIR',
                        from: { types: ['StateConclusion'] },
                        to: { types: ['ComputeChair'] },
                    },
                    {
                        type: 'EVALUATES_TARGET',
                        from: { types: ['StateConclusion'] },
                        to: { types: ['ProjectTarget'] },
                    },
                    {
                        type: 'USES_GOLDEN_BRICK',
                        from: { types: ['StateConclusion'] },
                        to: { types: ['GoldenBrick'] },
                    },
                ],
            });

            const store = new HydraDBStore(boltUrl, {
                auth: ['neo4j', boltToken],
            });
            const graph = new SelfGraph(store, {
                namespace,
                schema,
                schema_mode: 'warn',
                contradiction: 'supersede',
                resolver: 'default',
            });
            await graph.initialize();
            return { graph, toJsonable, boltUrl, namespace };
        } catch (err) {
            console.warn(`[JeanStateGraph] HydraDB SelfGraph unavailable (${err.message}); using PG/memory fallback.`);
            _activeAdapterPromise = null;
            return null;
        }
    })();

    return _activeAdapterPromise;
}

async function readPgStateRecord(key) {
    const normId = normalizeConversationId(`jean_state:${key}`);
    if (!pool) return (await getBoundedMemory(normId)) || null;
    try {
        const res = await pool.query(
            `SELECT content FROM ide_agent_memory WHERE id = $1 LIMIT 1`,
            [normId]
        );
        if (res.rows.length > 0) {
            return JSON.parse(res.rows[0].content);
        }
    } catch {
        return (await getBoundedMemory(normId)) || null;
    }
    return null;
}

async function writePgStateRecord(key, record) {
    const normId = normalizeConversationId(`jean_state:${key}`);
    await setBoundedMemory(normId, record);
    if (!pool) return;
    try {
        await pool.query(
            `INSERT INTO ide_agent_memory (id, category, content)
             VALUES ($1, 'decision', $2)
             ON CONFLICT (id) DO UPDATE SET content = $2`,
            [normId, JSON.stringify(record)]
        );
    } catch (err) {
        console.warn(`[JeanStateGraph] PG state upsert fallback: ${err.message}`);
    }
}

/**
 * Hydrate Jean's persistent memory state and PPR conclusions from HydraDB + SelfGraph + kruschdb
 * BEFORE routing a turn to any GPU compute chair.
 *
 * @param {object} [options={}]
 * @returns {Promise<{ ok: boolean, targetProject: string, pprEntities: Array<object>, pprRelations: Array<object>, latestState: object|null, promptBlock: string }>}
 */
export async function hydrateJeanPersistentState(options = {}) {
    const targetProject = String(options.targetProject || 'fleet').trim() || 'fleet';
    const query = String(options.query || targetProject).trim() || targetProject;
    const projSlug = slugifyId(targetProject, 'fleet');

    let pprEntities = [];
    let pprRelations = [];
    const adapter = await getJeanSelfGraphAdapter();
    if (adapter) {
        try {
            const retrieved = adapter.toJsonable(
                await adapter.graph.retrieve(`${targetProject} ${query}`.slice(0, 200), {
                    strategy: 'ppr',
                    budget_tokens: 650,
                    depth: 3,
                })
            );
            pprEntities = Array.isArray(retrieved?.subgraph?.entities)
                ? retrieved.subgraph.entities
                : [];
            pprRelations = Array.isArray(retrieved?.subgraph?.relations)
                ? retrieved.subgraph.relations
                : [];
        } catch (err) {
            console.warn(`[JeanStateGraph] SelfGraph PPR retrieve warning: ${err.message}`);
        }
    }

    const latestState =
        (await readPgStateRecord(`project:${projSlug}`)) ||
        (await readPgStateRecord('global')) ||
        null;

    const lines = [];
    if (latestState || pprEntities.length > 0) {
        lines.push('### 🧠 JEAN PERSISTENT GRAPH STATE (HydraDB + SelfGraph PPR)');
        if (latestState) {
            lines.push(
                `- **Active Project State (\`${latestState.targetProject}\` @ \`${latestState.phase}\`):** Focus="${latestState.focus}" | NextGoal="${latestState.nextGoal}" | Pending="${latestState.pendingIssues}" | LastChair=\`${latestState.routedModel}\` (difficulty=${latestState.complexityScore})`
            );
            if (latestState.summary) {
                lines.push(`- **Prior Verified Conclusion:** ${latestState.summary}`);
            }
        }
        if (pprEntities.length > 0) {
            const topNodes = pprEntities
                .slice(0, 5)
                .map((e) => {
                    const sum = e.properties?.summary ? ` — ${String(e.properties.summary).slice(0, 120)}` : '';
                    return `\`${e.id}\` (${e.name}${sum})`;
                })
                .join('; ');
            lines.push(`- **SelfGraph PPR Multi-Hop Conclusions (${pprEntities.length} nodes, ${pprRelations.length} edges):** ${topNodes}`);
        }
    }

    return {
        ok: true,
        targetProject,
        pprEntities,
        pprRelations,
        latestState,
        promptBlock: lines.join('\n'),
    };
}

/**
 * Mutate Jean's persistent memory state inside HydraDB + SelfGraph (`contradiction: "supersede"`,
 * `max_count: 1`) AND `kruschdb` PostgreSQL after any Chair completes a turn.
 *
 * @param {object} params
 */
export async function commitJeanStateTransition(params = {}) {
    const targetProject = String(params.targetProject || 'fleet').trim() || 'fleet';
    const phase = String(params.phase || 'INTERACTIVE_CHAT').trim();
    const routedTierId = String(params.routedTierId || 'L2_FIRST_CHAIR_INKLING').trim();
    const routedModel = String(params.routedModel || 'inkling-2-mini:ud-iq4_xs-dspark').trim();
    const complexityScore = Number.isFinite(params.complexityScore)
        ? Number(params.complexityScore.toFixed(2))
        : 0.5;
    const difficultyBand =
        params.difficultyBand ||
        (complexityScore > 0.68 ? 'hard' : complexityScore > 0.35 ? 'medium' : 'easy');
    const cycle = Number.isFinite(params.cycle) ? params.cycle : null;
    const timestamp = new Date().toISOString();

    const extracted = extractStructuredJeanConclusion(params.reply || '', {
        targetProject,
        phase,
    });

    const projSlug = slugifyId(targetProject, 'fleet');
    const chairSlug = slugifyId(routedModel, 'chair');
    const transitionHash = crypto
        .createHash('sha256')
        .update(`${targetProject}:${phase}:${routedModel}:${timestamp}:${extracted.summary}`)
        .digest('hex')
        .slice(0, 12);
    const conclusionEntityId = `stateconclusion:${projSlug}-${transitionHash}`;
    const projectEntityId = `projecttarget:${projSlug}`;
    const chairEntityId = `computechair:${chairSlug}`;

    const stateRecord = {
        transitionId: conclusionEntityId,
        timestamp,
        cycle,
        targetProject,
        phase,
        difficultyBand,
        complexityScore,
        routedTierId,
        routedModel,
        ...extracted,
    };

    _inMemoryTransitions.push(stateRecord);
    if (_inMemoryTransitions.length > MAX_LOCAL_TRANSITIONS) {
        _inMemoryTransitions.shift();
    }

    await Promise.all([
        writePgStateRecord(`project:${projSlug}`, stateRecord),
        writePgStateRecord('global', stateRecord),
    ]);

    let selfGraphSynced = false;
    let supersededCount = 0;
    const adapter = await getJeanSelfGraphAdapter();
    if (adapter) {
        try {
            const brickMatches = Array.from(
                new Set(
                    (String(params.reply || '').match(
                        /\b(?:stripe-webhook-normalizer|sox-integer-fee-allocator|ledger-audit-sealer|telemetry-cadence-window)\b/g
                    ) || [])
                )
            );
            const entities = [
                {
                    id: 'jeanmind:core',
                    type: 'JeanMind',
                    name: 'Jean Persistent Mind',
                    properties: {
                        active_project: targetProject,
                        active_phase: phase,
                        updated_at: timestamp,
                    },
                },
                {
                    id: projectEntityId,
                    type: 'ProjectTarget',
                    name: targetProject,
                    properties: {
                        last_phase: phase,
                        last_verdict: extracted.verdict,
                        updated_at: timestamp,
                    },
                },
                {
                    id: conclusionEntityId,
                    type: 'StateConclusion',
                    name: `${targetProject} [${phase}]: ${extracted.focus}`.slice(0, 140),
                    properties: {
                        summary: extracted.summary,
                        focus: extracted.focus,
                        next_goal: extracted.nextGoal,
                        pending_issues: extracted.pendingIssues,
                        phase,
                        verdict: extracted.verdict,
                        difficulty_band: difficultyBand,
                        complexity_score: complexityScore,
                        tier_id: routedTierId,
                        model: routedModel,
                        updated_at: timestamp,
                    },
                },
                {
                    id: chairEntityId,
                    type: 'ComputeChair',
                    name: routedModel,
                    properties: {
                        tier_id: routedTierId,
                        difficulty_band: difficultyBand,
                    },
                },
                ...brickMatches.map((bId) => ({
                    id: `goldenbrick:${slugifyId(bId)}`,
                    type: 'GoldenBrick',
                    name: bId,
                })),
            ];

            const relations = [
                {
                    from_entity: 'jeanmind:core',
                    type: 'CURRENT_FOCUS_OF',
                    to_entity: conclusionEntityId,
                    event_time: timestamp,
                },
                {
                    from_entity: projectEntityId,
                    type: 'ACTIVE_CONCLUSION_FOR',
                    to_entity: conclusionEntityId,
                    event_time: timestamp,
                },
                {
                    from_entity: conclusionEntityId,
                    type: 'DERIVED_BY_CHAIR',
                    to_entity: chairEntityId,
                    event_time: timestamp,
                },
                {
                    from_entity: conclusionEntityId,
                    type: 'EVALUATES_TARGET',
                    to_entity: projectEntityId,
                    event_time: timestamp,
                },
                ...brickMatches.map((bId) => ({
                    from_entity: conclusionEntityId,
                    type: 'USES_GOLDEN_BRICK',
                    to_entity: `goldenbrick:${slugifyId(bId)}`,
                    event_time: timestamp,
                })),
            ];

            const inserted = adapter.toJsonable(
                await adapter.graph.insert.facts(
                    { entities, relations },
                    { source_id: `jean:${projSlug}:${transitionHash}` }
                )
            );
            selfGraphSynced = true;
            supersededCount = Array.isArray(inserted?.warnings)
                ? inserted.warnings.filter((w) => w.code === 'relation_superseded').length
                : 0;
        } catch (err) {
            console.warn(`[JeanStateGraph] SelfGraph insert warning: ${err.message}`);
        }
    }

    return okEnvelope(
        'jean.state.commit',
        `Committed Jean state transition ${conclusionEntityId} via ${routedModel} (${difficultyBand}, superseded=${supersededCount})`,
        {
            ...stateRecord,
            selfGraphSynced,
            supersededRelations: supersededCount,
        }
    );
}

/**
 * Inspect Jean's live SelfGraph + HydraDB readiness, PPR conclusions, and recent transitions.
 */
export async function getJeanGraphStatus(options = {}) {
    const targetProject = String(options.targetProject || 'fleet').trim() || 'fleet';
    const hydrated = await hydrateJeanPersistentState({
        targetProject,
        query: options.query || targetProject,
    });
    const adapter = await getJeanSelfGraphAdapter();
    let readiness = null;
    if (adapter) {
        try {
            readiness = adapter.toJsonable(await adapter.graph.readiness());
        } catch (err) {
            readiness = { ready: false, error: err.message };
        }
    }
    return okEnvelope('jean.state.status', `Jean Persistent State active (${_inMemoryTransitions.length} recent transitions)`, {
        targetProject,
        selfGraphConnected: Boolean(adapter),
        readiness,
        latestState: hydrated.latestState,
        pprSubgraph: {
            entityCount: hydrated.pprEntities.length,
            relationCount: hydrated.pprRelations.length,
            entities: hydrated.pprEntities.slice(0, 12),
            relations: hydrated.pprRelations.slice(0, 12),
        },
        recentTransitions: _inMemoryTransitions.slice(-12),
    });
}

/**
 * Attach `/state` inspection route onto Jean's Express app without shifting `harness.js` line anchors.
 */
export function registerJeanStateRoutes(app, authMiddleware) {
    app.get('/state', authMiddleware, async (req, res) => {
        try {
            const targetProject =
                typeof req.query?.project === 'string' && req.query.project.trim()
                    ? req.query.project.trim()
                    : 'fleet';
            const query =
                typeof req.query?.query === 'string' && req.query.query.trim()
                    ? req.query.query.trim()
                    : targetProject;
            const envelope = await getJeanGraphStatus({ targetProject, query });
            res.status(envelope.ok ? 200 : 503).json(envelope);
        } catch (err) {
            res.status(500).json(errorEnvelope('JEAN_STATE_ERROR', err.message, null, 'jean.state.status'));
        }
    });
}
