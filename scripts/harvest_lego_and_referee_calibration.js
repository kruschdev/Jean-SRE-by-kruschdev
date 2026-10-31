#!/usr/bin/env node
/**
 * @file harvest_lego_and_referee_calibration.js
 * @description Builds two Zero-Leak-sanitized fine-tuning splits from the verified
 * @krusch/toolkit v1.4.0 & krusch_toolkit v1.2.0 Lego System and Colab G4 Referee logs:
 *
 *   1. `data/distillation/lego_and_referee_calibration_sft.jsonl` (ChatML SFT):
 *      - Calibrated `CLEAN_PASS` Referee audits on live verified source windows (`findings: []`)
 *      - Ground-truth P0/P1 Pre-Fix -> Post-Fix security & concurrency remediations
 *      - "Lego-First" anti-boilerplate synthesis & refactoring trajectories across all 18 bricks
 *
 *   2. `data/distillation/referee_calibration_dpo.jsonl` (`trl.DPOTrainer` preference pairs):
 *      - Balanced Precision (`CLEAN_PASS` chosen vs false-positive/out-of-scope rejected)
 *        and Recall (`DEFECTS_FOUND` chosen vs rubber-stamp `CLEAN_PASS` rejected) pairs.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizePortableSftText } from '../src/harness.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const homelabRoot = path.resolve(projectRoot, '..', '..');

export const LEGO_CALIBRATION_SFT_PATH = path.join(
    projectRoot,
    'data',
    'distillation',
    'lego_and_referee_calibration_sft.jsonl'
);
export const REFEREE_DPO_PATH = path.join(
    projectRoot,
    'data',
    'distillation',
    'referee_calibration_dpo.jsonl'
);

const AUDITOR_SYSTEM_PROMPT = [
    'You are an Autonomous Principal Software Engineer, SRE, and Code Auditor.',
    'RULES (NON-NEGOTIABLE):',
    '1. Ground every finding strictly in the provided live source snippets, AST symbol graphs, and command outputs.',
    '2. NEVER invent, paraphrase, or reconstruct code snippets from memory. Only quote verbatim lines that appear in the prompt with their exact line numbers.',
    '3. If a prior Pending Issue is not supported by the Live Source Verification lines shown in the prompt, REFUTE and remove it.',
    '4. When audited files satisfy all frozen specification invariants with zero P0/P1 defects, emit a calibrated CLEAN_PASS verdict ("findings": []) rather than inventing stylistic nitpicks.',
    '5. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.'
].join('\n');

const LEGO_ENGINEER_SYSTEM_PROMPT = [
    'You are a Principal Full-Stack & Systems Engineer assembling production software with @krusch/toolkit v1.4 and krusch_toolkit v1.2 Lego bricks.',
    'RULES (NON-NEGOTIABLE):',
    '1. INV-LEGO-01: Assemble HTTP servers, SSE streams, MCP tools, PostgreSQL/pgvector pools, SQLite WAL stores, React hooks/shells, Cloudflare Workers, HMAC webhooks, and single-instance CLIs from @krusch/toolkit/* and krusch_toolkit.* bricks—never hand-roll boilerplate.',
    '2. Enforce deterministic { ok, tool, summary, data | error } envelopes, AbortController cancellation, constant-time HMAC/Bearer verification, and strict TypeScript (.d.ts / --experimental-strip-types) compatibility.'
].join('\n');

const COMPACT_LEGO_SPEC = [
    '- INV-LEGO-01 (Universal Envelope): HTTP/MCP/CLI/FastAPI endpoints emit { ok: true, tool, summary, data } or { ok: false, tool, summary, error: { code, message, details? } }. Low-level header builders (e.g., createCorsHeaders -> Record<string,string>) return header maps for res.writeHead.',
    '- INV-LEGO-02 (Zero Ghost Exports & Headless Safety): Every documented subpath exists in package.json exports ({ types, default }) and imports cleanly in headless Node 22 ESM.',
    '- INV-LEGO-03 (AbortController & 413 Drain): SSE & React fetch hooks wire AbortController teardown; readBoundedJsonBody drains/resumes oversized streams before closing so HTTP 413 JSON envelopes flush without TCP ECONNRESET.',
    '- INV-LEGO-08 (Edge Purity): @krusch/toolkit/edge contains zero node:* imports, rejects Origin: null/false/CRLF, enforces chunked stream maxBytes (413), and aborts SSE on stream cancel.',
    '- INV-LEGO-09 (Constant-Time Webhooks & Bearer): @krusch/toolkit/webhooks verifies Slack/GitHub/Stripe HMACs and SHA-256 pre-hashed Bearer tokens via crypto.timingSafeEqual after equal-length checks, failing closed on empty secrets.',
    '- INV-LEGO-10 (Atomic CLI & Daemon): @krusch/toolkit/cli uses exclusive wx PID+nonce lockfiles in withSingleInstanceLock (wrapping thrown callback errors in errorEnvelope unless rethrow:true and unlinking only owned nonces in finally) and prototype-safe defineCli.',
    '- INV-LEGO-11 (Strict TypeScript Parity): lib/types/*.d.ts matches runtime exports 1:1 under tsc --noEmit --strict (including contravariant Logger meta types).'
].join('\n');

/**
 * Read a bounded 1-indexed line window from the workspace and scrub any private host/IP tokens.
 */
export function readNumberedWindow(relPath, startLine = 1, endLine = 140) {
    const absPath = path.join(homelabRoot, relPath);
    const raw = fs.readFileSync(absPath, 'utf8');
    const lines = raw.split(/\r?\n/);
    const s = Math.max(1, startLine);
    const e = Math.min(lines.length, endLine);
    const numbered = lines
        .slice(s - 1, e)
        .map((line, i) => `${s + i}: ${line}`)
        .join('\n');
    const scrubbed = normalizePortableSftText(numbered, 'toolkit');
    return `### FILE: \`${relPath}\` (showing lines ${s}-${e} of ${lines.length})\n\`\`\`\n${scrubbed}\n\`\`\``;
}

function buildRefereeUserPrompt({ sliceId, title, files, extraSpec = '', renderedSources }) {
    const specBlock = extraSpec ? `${COMPACT_LEGO_SPEC}\n${extraSpec}` : COMPACT_LEGO_SPEC;
    return normalizePortableSftText(
        [
            `## 🛑 JEAN — AUTONOMOUS SPEC-FIRST REFEREE AUDIT`,
            `- **Slice ID:** \`${sliceId}\` — ${title}`,
            `- **Audited Files:** \`${files.join(', ')}\``,
            `### AUDIT CONTRACT (STRICT P0/P1 GATE — CALIBRATED CLEAN_PASS)`,
            `Audit the live line-numbered source files strictly against the Frozen Specifications below:`,
            `- Report ONLY real P0/P1 spec violations, runtime crashes, security flaws, resource leaks, or signature mismatches with exact 1-indexed line numbers and verbatim quotes.`,
            `- Do NOT misclassify helper return types (e.g., HTTP header-map builders vs JSON endpoint envelopes) or apply package-scoped invariants outside their target package.`,
            `- If zero P0/P1 defects exist, return \`"verdict": "CLEAN_PASS"\` and \`"findings": []\`.`,
            `### FROZEN SPECIFICATIONS\n${specBlock}`,
            `### LIVE LINE-NUMBERED SOURCE FILES\n${renderedSources}`
        ].join('\n\n'),
        'toolkit'
    );
}

/**
 * Category 1: Calibrated CLEAN_PASS Referee Verdicts grounded in live post-fix toolkit windows.
 */
export function buildCleanPassCalibrationRows(nowIso) {
    const specs = [
        {
            id: 'referee-clean-pass-http-cors-drain',
            sliceId: 'slice_http_cors_and_body',
            title: 'HTTP Envelopes, W3C CORS Header Map & 413 Stream-Drain Guard',
            files: ['lib/http.js'],
            window: () => readNumberedWindow('lib/http.js', 75, 225),
            rationale:
                'createCorsHeaders returns a W3C-compliant HTTP header map omitting Access-Control-Allow-Origin on null/false/CRLF origins while setting Vary: Origin, and readBoundedJsonBody calls req.resume() on 413 PAYLOAD_TOO_LARGE so clients receive the JSON error envelope cleanly.'
        },
        {
            id: 'referee-clean-pass-edge-router',
            sliceId: 'slice_edge_web_standards',
            title: 'Zero-node:* Cloudflare Worker Edge Router, Bounded JSON & SSE',
            files: ['lib/edge.js'],
            window: () => readNumberedWindow('lib/edge.js', 1, 175),
            rationale:
                'lib/edge.js contains zero node:* imports, sanitizes Origin: null/false/CRLF, enforces maxBytes across Content-Length and ReadableStream chunks, and preserves CORS headers on error responses.'
        },
        {
            id: 'referee-clean-pass-webhooks-hmac',
            sliceId: 'slice_webhooks_constant_time',
            title: 'Constant-Time Slack, GitHub, Stripe & SHA-256 Bearer Verification',
            files: ['lib/webhooks.js'],
            window: () => readNumberedWindow('lib/webhooks.js', 215, 371),
            rationale:
                'verifyStripeSignature evaluates timingSafeEqual across capped v1 candidates without early short-circuiting, and verifyBearerToken hashes both candidate and expected secrets with SHA-256 prior to timingSafeEqual so secret length is never leaked.'
        },
        {
            id: 'referee-clean-pass-cli-lock',
            sliceId: 'slice_cli_single_instance_lock',
            title: 'Atomic wx PID+Nonce Single-Instance Lock & Envelope Totality',
            files: ['lib/cli.js'],
            window: () => readNumberedWindow('lib/cli.js', 135, 240),
            rationale:
                'withSingleInstanceLock acquires locks via exclusive wx mode, reclaims dead/stale locks safely via inode check + single wx retry, wraps callback exceptions into deterministic errorEnvelope unless rethrow:true, and unlinks only its own nonce in finally.'
        },
        {
            id: 'referee-clean-pass-mcp-pg',
            sliceId: 'slice_mcp_and_pgvector',
            title: '4-Boolean MCP Tool Annotations & SQLi-Guarded pgVectorSearch',
            files: ['lib/mcp-server.js', 'lib/pg.js'],
            window: () =>
                `${readNumberedWindow('lib/mcp-server.js', 1, 95)}\n\n${readNumberedWindow('lib/pg.js', 180, 265)}`,
            rationale:
                'defineMcpTool enforces all four MCP boolean annotations plus title, and pgVectorSearch validates identifier names, 1024-dimension vectors, and rejects dangerous SQL tokens in whereSql.'
        },
        {
            id: 'referee-clean-pass-ts-declarations',
            sliceId: 'slice_typescript_declarations',
            title: 'Strict TypeScript Envelopes & Contravariant Logger Compatibility',
            files: ['lib/types/envelopes.d.ts', 'lib/types/http.d.ts', 'lib/types/log.d.ts'],
            window: () =>
                [
                    readNumberedWindow('lib/types/envelopes.d.ts', 1, 40),
                    readNumberedWindow('lib/types/http.d.ts', 1, 55),
                    readNumberedWindow('lib/types/log.d.ts', 1, 45)
                ].join('\n\n'),
            rationale:
                'Logger methods accept Record<string, unknown> | unknown metadata so Logger satisfies HttpLoggerLike under strictFunctionTypes without unsafe casts.'
        },
        {
            id: 'referee-clean-pass-boilerplate-guard',
            sliceId: 'slice_tier1_boilerplate_guard',
            title: 'Unified-Diff Added-Lines Anti-Boilerplate Pre-Commit Guard',
            files: ['projects/krusch-context-mcp/src/lego-boilerplate-guard.js'],
            window: () =>
                readNumberedWindow('projects/krusch-context-mcp/src/lego-boilerplate-guard.js', 1, 145),
            rationale:
                'extractAddedCodeLines isolates + diff lines per target file so deleting legacy boilerplate never triggers false-positive INV-LEGO-01 nudges, and exempts toolkit/test paths cleanly.'
        },
        {
            id: 'referee-clean-pass-python-fastapi',
            sliceId: 'slice_python_fastapi_brick',
            title: 'Lazy-Imported Python FastAPI Factory & Disconnect-Safe SSE',
            files: ['lib-py/krusch_toolkit/fastapi_app.py'],
            window: () => readNumberedWindow('lib-py/krusch_toolkit/fastapi_app.py', 1, 150),
            rationale:
                'ok_envelope and error_envelope import cleanly without FastAPI installed, while create_fastapi_app wires X-Request-ID timing, deterministic JSON exception envelopes, and /health + /doctor readiness probes.'
        }
    ];

    return specs.map((item) => {
        const userPrompt = buildRefereeUserPrompt({
            sliceId: item.sliceId,
            title: item.title,
            files: item.files,
            renderedSources: item.window()
        });
        const assistantJson = JSON.stringify(
            {
                verdict: 'CLEAN_PASS',
                checked_files: item.files,
                not_flagged_notes: [item.rationale],
                findings: []
            },
            null,
            2
        );
        return {
            id: item.id,
            timestamp: nowIso,
            source: 'referee_clean_pass_calibration',
            teacher_model: 'gemini-4-argon',
            student_target: 'qwen3-coder-30b-moe:latest',
            project: 'krusch-toolkit',
            phase: 'SPEC_REFEREE_CLEAN_PASS',
            verification_status: 'TEST_AND_REFEREE_VERIFIED',
            elitist_score: 95,
            messages: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                { role: 'user', content: userPrompt },
                { role: 'assistant', content: `\`\`\`json\n${assistantJson}\n\`\`\`` }
            ]
        };
    });
}

/**
 * Category 2: Ground-truth P0/P1 Pre-Fix -> Post-Fix defect detection & remediation trajectories.
 */
export function buildDefectRemediationRows(nowIso) {
    const cases = [
        {
            id: 'remediation-cli-lock-callback-envelope',
            file: 'lib/cli.js',
            line: 14,
            severity: 'P1',
            invariant: 'INV-LEGO-10',
            title: 'Wrap thrown callback exceptions in withSingleInstanceLock into errorEnvelope',
            buggySnippet: [
                '10:     if (!acquired) {',
                "11:         return errorEnvelope('LOCK_HELD', `Lock already held by PID ${holderPid}`, { pid: holderPid }, tool);",
                '12:     }',
                '13:     try {',
                '14:         const result = await fn({ lockFilePath: resolvedPath, pid: process.pid, nonce });',
                "15:         return okEnvelope(tool, 'Completed locked execution', { lockFilePath: resolvedPath, result });",
                '16:     } finally {',
                '17:         releaseLockIfOwned(resolvedPath, nonce);',
                '18:     }'
            ].join('\n'),
            quote: 'const result = await fn({ lockFilePath: resolvedPath, pid: process.pid, nonce });',
            description:
                'withSingleInstanceLock documents a Promise<LegoEnvelope> return contract and returns errorEnvelope on LOCK_HELD, but if fn() throws an exception it propagates unhandled instead of returning a deterministic errorEnvelope.',
            fix: 'Catch callback errors inside try/catch, release the owned lockfile in finally, and return errorEnvelope(String(err?.code || "LOCK_EXECUTION_ERROR"), err?.message || String(err), { lockFilePath: resolvedPath }, tool) unless options.rethrow === true.'
        },
        {
            id: 'remediation-http-cors-null-origin',
            file: 'lib/http.js',
            line: 6,
            severity: 'P0',
            invariant: 'INV-LEGO-01',
            title: 'Block Origin: null reflection with Access-Control-Allow-Credentials',
            buggySnippet: [
                '1: export function createCorsHeaders(origin, allowedOrigins = ["*"]) {',
                '2:     const cleanOrigin = String(origin || "").replace(/[\\r\\n]+/g, "").trim();',
                '3:     if (allowedOrigins.includes("*")) {',
                '4:         return { "Access-Control-Allow-Origin": "*" };',
                '5:     }',
                '6:     if (cleanOrigin && allowedOrigins.includes(cleanOrigin)) {',
                '7:         return { "Access-Control-Allow-Origin": cleanOrigin, "Access-Control-Allow-Credentials": "true", Vary: "Origin" };',
                '8:     }',
                '9:     return { Vary: "Origin" };',
                '10: }'
            ].join('\n'),
            quote: 'if (cleanOrigin && allowedOrigins.includes(cleanOrigin)) {',
            description:
                'If a caller passes ["null"] from misconfigured env vars, createCorsHeaders reflects Access-Control-Allow-Origin: null alongside Access-Control-Allow-Credentials: true, enabling sandboxed-iframe credential exfiltration.',
            fix: 'Explicitly filter "null" and "false" out of allowedOrigins and reject cleanOrigin === "null" || cleanOrigin === "false" before reflecting credentials.'
        },
        {
            id: 'remediation-http-413-socket-drain',
            file: 'lib/http.js',
            line: 9,
            severity: 'P1',
            invariant: 'INV-LEGO-03',
            title: 'Drain oversized HTTP request stream on 413 to prevent client ECONNRESET',
            buggySnippet: [
                '1: export async function readBoundedJsonBody(req, res, { maxBytes = 1048576, tool = "http" } = {}) {',
                '2:     let total = 0;',
                '3:     const chunks = [];',
                '4:     for await (const chunk of req) {',
                '5:         total += chunk.length;',
                '6:         if (total > maxBytes) {',
                '7:             res.writeHead(413, { "Content-Type": "application/json", Connection: "close" });',
                '8:             res.end(JSON.stringify(errorEnvelope("PAYLOAD_TOO_LARGE", "Body too large", null, tool)));',
                '9:             req.destroy();',
                '10:             return { ok: false, handled: true };',
                '11:         }',
                '12:         chunks.push(chunk);',
                '13:     }',
                '14: }'
            ].join('\n'),
            quote: 'req.destroy();',
            description:
                'Calling req.destroy() immediately after res.end() while unread body chunks are still in flight sends a TCP RST before the client reads the HTTP 413 JSON envelope, violating INV-LEGO-03.',
            fix: 'Attach req.resume() instead of req.destroy() (with Connection: close on the 413 response) so Node flushes the 413 JSON response cleanly before closing the socket.'
        },
        {
            id: 'remediation-llm-queue-active-counter-race',
            file: 'lib/llm-queue.js',
            line: 9,
            severity: 'P1',
            invariant: 'INV-LEGO-04',
            title: 'Decrement LlmQueue.active before settling caller promise',
            buggySnippet: [
                '1:     async _runItem(item) {',
                '2:         this.active += 1;',
                '3:         try {',
                '4:             const value = await item.task(this._pickEndpoint());',
                '5:             item.resolve(value);',
                '6:         } catch (err) {',
                '7:             item.reject(err);',
                '8:         } finally {',
                '9:             this.active -= 1;',
                '10:             this._pump();',
                '11:         }',
                '12:     }'
            ].join('\n'),
            quote: 'this.active -= 1;',
            description:
                'Resolving item.resolve(value) before this.active -= 1 schedules caller continuations on the microtask queue while this.active is still 1, breaking synchronous post-await queue.active === 0 checks.',
            fix: 'Capture outcome/error first, decrement this.active and trigger this._pump() before invoking item.resolve(value) or item.reject(err).'
        },
        {
            id: 'remediation-webhooks-bearer-length-leak',
            file: 'lib/webhooks.js',
            line: 7,
            severity: 'P1',
            invariant: 'INV-LEGO-09',
            title: 'Pre-hash Bearer tokens with SHA-256 before timingSafeEqual to hide secret length',
            buggySnippet: [
                '1: export function verifyBearerToken(authHeader, expectedSecret, { tool = "webhooks.bearer" } = {}) {',
                '2:     if (!expectedSecret || !expectedSecret.trim()) return errorEnvelope("AUTH_MISSING_SECRET", "Missing secret", null, tool);',
                '3:     const m = /^Bearer\\s+(\\S+)$/i.exec(String(authHeader || "").trim());',
                '4:     if (!m) return errorEnvelope("AUTH_INVALID_HEADER", "Malformed Authorization header", null, tool);',
                '5:     const a = Buffer.from(m[1], "utf8");',
                '6:     const b = Buffer.from(expectedSecret, "utf8");',
                '7:     if (a.length !== b.length || !timingSafeEqual(a, b)) {',
                '8:         return errorEnvelope("AUTH_UNAUTHORIZED", "Invalid bearer token", null, tool);',
                '9:     }',
                '10:     return okEnvelope(tool, "Bearer token verified", { verified: true });',
                '11: }'
            ].join('\n'),
            quote: 'if (a.length !== b.length || !timingSafeEqual(a, b)) {',
            description:
                'Branching on a.length !== b.length before timingSafeEqual leaks the exact UTF-8 byte length of expectedSecret over timing side-channels.',
            fix: 'Compute fixed 32-byte createHash("sha256") digests of both m[1] and expectedSecret and compare the two 32-byte digests via timingSafeEqual.'
        },
        {
            id: 'remediation-cli-prototype-command-lookup',
            file: 'lib/cli.js',
            line: 4,
            severity: 'P1',
            invariant: 'INV-LEGO-10',
            title: 'Prevent Object.prototype command lookup on constructor / toString argv',
            buggySnippet: [
                '1: export function defineCli({ name = "krusch-cli", commands = {} } = {}) {',
                '2:     return {',
                '3:         async run(argv = []) {',
                '4:             const cmd = commands[argv[0]];',
                '5:             if (!cmd) return { exitCode: 1 };',
                '6:             return { exitCode: 0, result: await cmd.run({ argv }) };',
                '7:         }',
                '8:     };',
                '9: }'
            ].join('\n'),
            quote: 'const cmd = commands[argv[0]];',
            description:
                'Indexing commands[argv[0]] without Object.hasOwn resolves inherited Object.prototype properties such as "constructor" or "toString", throwing TypeError: cmd.run is not a function instead of returning UNKNOWN_COMMAND.',
            fix: 'Guard command lookup with Object.hasOwn(commands, cmdName) (or copy commands into Object.create(null)) before checking cmd.run.'
        },
        {
            id: 'remediation-tailwind-arbitrary-css-injection',
            file: 'projects/krusch-frontend-mcp/src/tools/tailwind.js',
            line: 5,
            severity: 'P1',
            invariant: 'INV-LEGO-01',
            title: 'Sanitize arbitrary CSS values in css_to_tailwind against bracket/brace breakout',
            buggySnippet: [
                '1: function toArbitraryTailwind(prop, rawValue) {',
                '2:     const clean = String(rawValue || "").trim();',
                '3:     if (!clean) return null;',
                '4:     const encoded = clean.replace(/\\s+/g, "_");',
                '5:     return `[${prop}:${encoded}]`;',
                '6: }'
            ].join('\n'),
            quote: 'return `[${prop}:${encoded}]`;',
            description:
                'Interpolating unsanitized CSS values into Tailwind arbitrary brackets allows payloads containing `]`, `;`, `{`, `}`, or quotes (e.g., `12px];body{display:none`) to break out of the class token.',
            fix: 'Reject values containing `[`, `]`, `{`, `}`, `;`, quotes, backslashes, or control characters before emitting `[prop:value]`.'
        },
        {
            id: 'remediation-typescript-logger-contravariance',
            file: 'lib/types/log.d.ts',
            line: 4,
            severity: 'P1',
            invariant: 'INV-LEGO-11',
            title: 'Widen Logger metadata parameter for strictFunctionTypes assignability to HttpLoggerLike',
            buggySnippet: [
                '1: // lib/types/http.d.ts: export interface HttpLoggerLike { info(msg: string, meta?: unknown): void; }',
                '2: export interface Logger {',
                '3:     debug(message: string, meta?: Record<string, unknown>): void;',
                '4:     info(message: string, meta?: Record<string, unknown>): void;',
                '5:     warn(message: string, meta?: Record<string, unknown>): void;',
                '6:     error(message: string, meta?: Record<string, unknown>): void;',
                '7: }'
            ].join('\n'),
            quote: 'info(message: string, meta?: Record<string, unknown>): void;',
            description:
                'Under TypeScript strictFunctionTypes, function parameter types are contravariant: Logger.info(msg, meta?: Record<string, unknown>) is NOT assignable to HttpLoggerLike.info(msg, meta?: unknown), causing tsc --noEmit --strict to fail in node-ts-service.',
            fix: 'Widen Logger method meta parameters to `Record<string, unknown> | unknown` so setupLogging() instances pass directly into createApiServer({ logger }).'
        }
    ];

    return cases.map((c) => {
        const userPrompt = buildRefereeUserPrompt({
            sliceId: c.id,
            title: c.title,
            files: [c.file],
            renderedSources: `### FILE: \`${c.file}\` (showing lines 1-${c.buggySnippet.split('\n').length})\n\`\`\`\n${c.buggySnippet}\n\`\`\``
        });
        const assistantJson = JSON.stringify(
            {
                verdict: 'DEFECTS_FOUND',
                checked_files: [c.file],
                findings: [
                    {
                        id: 'SRE-01',
                        severity: c.severity,
                        file: c.file,
                        line: c.line,
                        verbatim_quote: c.quote,
                        invariant: c.invariant,
                        description: c.description,
                        fix: c.fix
                    }
                ]
            },
            null,
            2
        );
        return {
            id: c.id,
            timestamp: nowIso,
            source: 'argon_golden_remediation',
            teacher_model: 'gemini-4-argon',
            student_target: 'qwen3-coder-30b-moe:latest',
            project: 'krusch-toolkit',
            phase: 'SPEC_REFEREE_DEFECT_DETECTION',
            verification_status: 'REMEDIATED_AND_VERIFIED',
            elitist_score: 98,
            messages: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                { role: 'user', content: userPrompt },
                { role: 'assistant', content: `\`\`\`json\n${assistantJson}\n\`\`\`` }
            ]
        };
    });
}

/**
 * Category 3: "Lego-First" synthesis & anti-boilerplate refactoring pairs so local 30B/7B & G4 120B
 * natively import @krusch/toolkit v1.4 and krusch_toolkit v1.2 bricks instead of hand-rolling plumbing.
 */
export function buildLegoFirstSynthesisRows(nowIso) {
    const tasks = [
        {
            id: 'lego-first-node-http-sqlite',
            prompt:
                'Build a Node 22 ESM JSON API service (`src/server.js`) with `/health`, `/doctor`, bounded JSON body parsing (`POST /api/notes`), SQLite WAL storage, and graceful SIGTERM drain using `@krusch/toolkit` Lego bricks.',
            answer: [
                'Assemble the service directly from `@krusch/toolkit/http`, `@krusch/toolkit/db`, and `@krusch/toolkit/log` so zero HTTP/CORS/SQLite boilerplate is hand-rolled:',
                '```js',
                "import { createApiServer, okEnvelope, errorEnvelope, attachGracefulShutdown } from '@krusch/toolkit/http';",
                "import { initDb, query, run, closeDb } from '@krusch/toolkit/db';",
                "import { setupLogging } from '@krusch/toolkit/log';",
                '',
                "const logger = setupLogging('notes-api');",
                "initDb(process.env.DB_PATH || './data/notes.db', {",
                '  schema: `CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);`',
                '});',
                '',
                'export const server = createApiServer({',
                "  serviceName: 'notes-api',",
                '  logger,',
                '  maxBodyBytes: 262_144,',
                "  healthCheck: () => ({ db: 'sqlite-wal' }),",
                '  routes: {',
                "    'GET /api/notes': () => okEnvelope('notes.list', 'Listed notes', { items: query('SELECT id, title, created_at FROM notes ORDER BY id DESC LIMIT 50') }),",
                "    'POST /api/notes': ({ body, sendJson }) => {",
                "      const title = String(body?.title || '').trim();",
                "      if (!title) return sendJson(400, errorEnvelope('VALIDATION_ERROR', 'title is required', null, 'notes.create'));",
                "      const info = run('INSERT INTO notes (title) VALUES (?)', [title]);",
                "      return sendJson(201, okEnvelope('notes.create', 'Created note', { id: Number(info.lastInsertRowid), title }));",
                '    }',
                '  }',
                '});',
                '',
                'if (import.meta.url === `file://${process.argv[1]}`) {',
                '  const port = Number(process.env.PORT || 3000);',
                '  server.listen(port, () => logger.info(`Listening on :${port}`));',
                '  attachGracefulShutdown(server, [() => closeDb()], { logger });',
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-sse-streaming',
            prompt:
                'Refactor a manual Node SSE route that sets `text/event-stream` headers by hand without disconnect cancellation into `@krusch/toolkit/streaming`.',
            answer: [
                'Replace manual `res.writeHead` SSE plumbing with `streamChat` (or `sseResponse`) from `@krusch/toolkit/streaming`, which enforces `X-Accel-Buffering: no`, `: ping` heartbeats, and automatic `AbortController` cancellation on client disconnect:',
                '```js',
                "import { streamChat } from '@krusch/toolkit/streaming';",
                "import { chat } from '@krusch/toolkit/llm';",
                '',
                'export async function handleChatStream({ req, res, body }) {',
                '  await streamChat(res, req, {',
                "    tool: 'assistant.stream',",
                "    userMessage: String(body?.prompt || ''),",
                '    chatFn: ({ messages, signal, onToken }) =>',
                "      chat(messages, { provider: 'ollama', signal, onToken })",
                '  });',
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-mcp-stdio-server',
            prompt:
                'Create a compliant stdio MCP server exposing a read-only search tool using `@krusch/toolkit/mcp` (`INV-STACK-03` 4-boolean annotations + `INV-LEGO-01` envelope).',
            answer: [
                'Use `defineMcpTool`, `createMcpServer`, and `okMcpResult` from `@krusch/toolkit/mcp`:',
                '```js',
                "import { defineMcpTool, createMcpServer, okMcpResult } from '@krusch/toolkit/mcp';",
                '',
                'const searchDocsTool = defineMcpTool({',
                "  name: 'docs_search',",
                "  title: 'Search Documentation',",
                "  description: 'Search indexed markdown documentation by keyword.',",
                '  inputSchema: {',
                "    type: 'object',",
                "    properties: { query: { type: 'string', description: 'Search query' } },",
                "    required: ['query']",
                '  },',
                '  annotations: {',
                '    readOnlyHint: true,',
                '    destructiveHint: false,',
                '    idempotentHint: true,',
                '    openWorldHint: false',
                '  },',
                '  handler: async ({ query }) =>',
                "    okMcpResult('docs_search', `Matched documents for \"${query}\"`, { query, matches: [] })",
                '});',
                '',
                'export const mcp = createMcpServer({',
                "  name: 'docs-mcp',",
                "  version: '1.0.0',",
                '  tools: [searchDocsTool]',
                '});',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-pgvector-rag',
            prompt:
                'Replace raw `new pg.Pool(...)` and hand-written `pgvector` SQL with `@krusch/toolkit/pg` and `@krusch/toolkit/embeddings` (1024-dim `snowflake-arctic-embed2`).',
            answer: [
                'Assemble the PostgreSQL pool, transactional schema migration, and 1024-dim cosine search from `@krusch/toolkit/pg` and `@krusch/toolkit/embeddings`:',
                '```js',
                "import { createPgPool, runSqlMigrations, pgVectorSearch } from '@krusch/toolkit/pg';",
                "import { embed } from '@krusch/toolkit/embeddings';",
                '',
                'export const pool = createPgPool({ connectionString: process.env.DATABASE_URL });',
                '',
                'export async function initVectorSchema() {',
                '  await runSqlMigrations(pool, [',
                '    {',
                "      id: '001_kb_chunks',",
                '      sql: `CREATE EXTENSION IF NOT EXISTS vector;',
                '            CREATE TABLE IF NOT EXISTS kb_chunks (',
                '              id BIGSERIAL PRIMARY KEY,',
                '              doc_key TEXT NOT NULL,',
                '              content TEXT NOT NULL,',
                '              embedding vector(1024) NOT NULL',
                '            );`',
                '    }',
                '  ]);',
                '}',
                '',
                'export async function searchKnowledgeBase(question, limit = 8) {',
                "  const queryVector = await embed(question, { inputType: 'query' });",
                '  return pgVectorSearch(pool, {',
                "    table: 'kb_chunks',",
                "    vectorColumn: 'embedding',",
                "    selectColumns: ['id', 'doc_key', 'content'],",
                '    queryVector,',
                '    dimensions: 1024,',
                '    limit',
                '  });',
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-cloudflare-worker-edge',
            prompt:
                'Build a Cloudflare Worker (`src/worker.js`) with `/health`, CORS, bounded JSON body parsing, KV JSON storage, and SSE token streaming without any `node:*` imports.',
            answer: [
                'Import exclusively from `@krusch/toolkit/edge` (`INV-LEGO-08` zero `node:*` imports):',
                '```js',
                'import {',
                '  createEdgeRouter,',
                '  createEdgeKvJsonStore,',
                '  edgeSseResponse',
                "} from '@krusch/toolkit/edge';",
                '',
                'const router = createEdgeRouter({',
                "  serviceName: 'intake-edge',",
                "  version: '1.0.0',",
                '  maxBodyBytes: 65_536,',
                '  routes: {',
                "    'POST /api/leads/:id': async ({ params, body, env, ok, fail }) => {",
                "      if (!body?.email) return fail('VALIDATION_ERROR', 'email is required', { status: 400 });",
                "      const store = createEdgeKvJsonStore(env.LEADS_KV, { prefix: 'lead:', defaultTtlSeconds: 86400 });",
                '      await store.put(params.id, { email: body.email, savedAt: Date.now() });',
                "      return ok('Saved lead', { id: params.id });",
                '    },',
                "    'GET /api/stream': ({ request, corsHeaders }) =>",
                '      edgeSseResponse(',
                '        async (sse) => {',
                "          sse.sendToken('Edge ');",
                "          if (!sse.signal.aborted) sse.sendToken('ready.');",
                "          sse.done('Stream complete');",
                '        },',
                "        { request, tool: 'intake-edge.stream', corsHeaders }",
                '      )',
                '  }',
                '});',
                '',
                'export default router;',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-webhooks-and-bearer',
            prompt:
                'Implement a GitHub webhook endpoint and an internal Bearer-authenticated trigger route in Node without hand-rolling `crypto.createHmac` or string equality checks.',
            answer: [
                'Use `verifyGithubSignature` and `verifyBearerToken` from `@krusch/toolkit/webhooks` (`INV-LEGO-09`):',
                '```js',
                "import { verifyGithubSignature, verifyBearerToken } from '@krusch/toolkit/webhooks';",
                "import { okEnvelope } from '@krusch/toolkit/http';",
                '',
                'export function handleGithubWebhook({ req, rawBody, sendJson }) {',
                '  const check = verifyGithubSignature({',
                '    rawBody,',
                "    signatureHeader: req.headers['x-hub-signature-256'],",
                '    webhookSecret: process.env.GITHUB_WEBHOOK_SECRET',
                '  });',
                '  if (!check.ok) return sendJson(401, check);',
                "  return sendJson(200, okEnvelope('webhooks.github', 'Accepted GitHub delivery', check.data));",
                '}',
                '',
                'export function handleInternalTrigger({ req, sendJson }) {',
                "  const auth = verifyBearerToken(req.headers.authorization, process.env.INTERNAL_API_TOKEN, { tool: 'internal.trigger' });",
                '  if (!auth.ok) return sendJson(401, auth);',
                "  return sendJson(200, okEnvelope('internal.trigger', 'Authorized trigger', { queued: true }));",
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-cli-single-instance-lock',
            prompt:
                'Write a single-instance maintenance CLI (`bin/sweep.js`) supporting `--json`, `SIGTERM`/`SIGINT` cancellation, NDJSON audit logging, and PID+nonce lockfile protection using `@krusch/toolkit/cli`.',
            answer: [
                'Compose `defineCli`, `withSingleInstanceLock`, `attachDaemonSignals`, and `appendJsonlRecord` from `@krusch/toolkit/cli` (`INV-LEGO-10`):',
                '```js',
                "import { defineCli, withSingleInstanceLock, attachDaemonSignals, appendJsonlRecord } from '@krusch/toolkit/cli';",
                '',
                'export const cli = defineCli({',
                "  name: 'maint-sweep',",
                "  version: '1.0.0',",
                '  commands: {',
                '    run: {',
                "      summary: 'Execute locked maintenance sweep',",
                "      options: { lock: { type: 'string', default: '/tmp/maint-sweep.lock' } },",
                '      run: async ({ values }) => {',
                '        const daemon = attachDaemonSignals();',
                '        try {',
                '          return await withSingleInstanceLock(values.lock, async ({ pid }) => {',
                "            if (daemon.signal.aborted) throw Object.assign(new Error('Aborted'), { code: 'ABORTED' });",
                "            appendJsonlRecord('./data/logs/sweep.jsonl', { pid, ts: new Date().toISOString(), status: 'ok' });",
                '            return { swept: true, pid };',
                "          }, { tool: 'maint-sweep' });",
                '        } finally {',
                '          daemon.detach();',
                '        }',
                '      }',
                '    }',
                '  }',
                '});',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-node-typescript-strip-types',
            prompt:
                'Write a zero-build Node 22 TypeScript API module (`src/server.ts`) running natively under `node --experimental-strip-types` with strict `@krusch/toolkit` `.d.ts` types (`INV-LEGO-11`).',
            answer: [
                'Use `import type` (`verbatimModuleSyntax: true`) with `@krusch/toolkit/http` and `@krusch/toolkit/log`:',
                '```ts',
                "import { createApiServer, okEnvelope, type OkEnvelope } from '@krusch/toolkit/http';",
                "import { setupLogging } from '@krusch/toolkit/log';",
                '',
                'export interface ServiceStatus {',
                '  ready: boolean;',
                '  uptimeSec: number;',
                '}',
                '',
                "const logger = setupLogging('typed-service');",
                '',
                'export function buildStatusPayload(): OkEnvelope<ServiceStatus> {',
                "  return okEnvelope('typed-service.status', 'Status ready', {",
                '    ready: true,',
                '    uptimeSec: Math.round(process.uptime())',
                '  });',
                '}',
                '',
                'export const server = createApiServer({',
                "  serviceName: 'typed-service',",
                '  logger,',
                '  routes: {',
                "    'GET /api/status': () => buildStatusPayload()",
                '  }',
                '});',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-react-hooks-and-shell',
            prompt:
                'Refactor a React component that uses raw un-aborted `useEffect(() => { fetch(...) }, [])` into `@krusch/toolkit/react` hooks and dashboard shell.',
            answer: [
                'Replace raw `useEffect` `fetch` calls with `useApiQuery`, `useSSEStream`, `AppShell`, and `StatusPill` from `@krusch/toolkit/react` (`INV-LEGO-03`):',
                '```jsx',
                "import { AppShell, StatusPill, MetricCard, useApiQuery, useSSEStream } from '@krusch/toolkit/react';",
                '',
                'export function ServiceDashboard() {',
                "  const { data, loading, error, reload } = useApiQuery('/api/metrics', []);",
                "  const stream = useSSEStream('/api/chat/stream');",
                '',
                '  return (',
                '    <AppShell',
                '      title="Operations Console"',
                '      status={<StatusPill status={error ? "error" : loading ? "loading" : "healthy"} label={error ? "Degraded" : "Online"} />}',
                '    >',
                '      <MetricCard label="Requests" value={data?.requests ?? 0} onClick={reload} />',
                '      <button type="button" onClick={() => stream.start({ prompt: "Summarize health" })}>Run Check</button>',
                '      {stream.streaming && <button type="button" onClick={stream.stop}>Stop</button>}',
                '      <pre>{stream.text}</pre>',
                '    </AppShell>',
                '  );',
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-python-fastapi-service',
            prompt:
                'Build a Python 3.12 FastAPI service (`app/main.py`) with CORS, `X-Request-ID` timing, `/health`, `/doctor`, deterministic error envelopes, and disconnect-safe SSE using `krusch_toolkit.fastapi_app`.',
            answer: [
                'Import `create_fastapi_app`, `ok_envelope`, `error_envelope`, and `sse_event_response` from `krusch_toolkit.fastapi_app` instead of hand-rolling `CORSMiddleware` or custom `/health` handlers:',
                '```python',
                'from fastapi import Request',
                'from fastapi.responses import JSONResponse',
                'from pydantic import BaseModel',
                'from krusch_toolkit.fastapi_app import (',
                '    create_fastapi_app,',
                '    ok_envelope,',
                '    error_envelope,',
                '    sse_event_response,',
                ')',
                '',
                'class AnalyzeIn(BaseModel):',
                '    text: str',
                '',
                'app = create_fastapi_app(',
                '    title="Document Analyzer",',
                '    version="1.0.0",',
                '    health_check=lambda: {"engine": "ready"},',
                ')',
                '',
                '@app.post("/api/analyze")',
                'async def analyze(payload: AnalyzeIn):',
                '    cleaned = payload.text.strip()',
                '    if not cleaned:',
                '        return JSONResponse(',
                '            status_code=400,',
                '            content=error_envelope("EMPTY_TEXT", "text must not be blank", tool="analyzer"),',
                '        )',
                '    return ok_envelope("analyzer", "Analyzed text", {"words": len(cleaned.split())})',
                '',
                '@app.get("/api/events")',
                'async def events(request: Request):',
                '    async def gen():',
                '        yield {"event": "token", "data": {"delta": "ready"}}',
                '        yield {"event": "done", "data": ok_envelope("analyzer.sse", "Complete", {})}',
                '    return sse_event_response(gen(), request=request)',
                '```'
            ].join('\n')
        }
    ];

    return tasks.map((t) => ({
        id: t.id,
        timestamp: nowIso,
        source: 'lego_first_sft',
        teacher_model: 'gemini-4-argon',
        student_target: 'qwen3-coder-30b-moe:latest',
        project: 'krusch-toolkit',
        phase: 'LEGO_FIRST_SYNTHESIS',
        verification_status: 'TEST_AND_TYPECHECK_VERIFIED',
        elitist_score: 96,
        messages: [
            { role: 'system', content: LEGO_ENGINEER_SYSTEM_PROMPT },
            { role: 'user', content: normalizePortableSftText(t.prompt, 'toolkit') },
            { role: 'assistant', content: normalizePortableSftText(t.answer, 'toolkit') }
        ]
    }));
}

/**
 * Split B: Balanced DPO preference pairs (`prompt`, `chosen`, `rejected`) teaching both:
 *   - Precision (rejecting false-positive return-type/scope/legacy-wrapper misreads)
 *   - Recall (rejecting lazy rubber-stamp CLEAN_PASS on genuine P0/P1 defects)
 */
export function buildRefereeDpoPairs() {
    const cleanPassJson = (files, note) => [
        {
            role: 'assistant',
            content: `\`\`\`json\n${JSON.stringify({ verdict: 'CLEAN_PASS', checked_files: files, not_flagged_notes: [note], findings: [] }, null, 2)}\n\`\`\``
        }
    ];

    const defectJson = (file, finding) => [
        {
            role: 'assistant',
            content: `\`\`\`json\n${JSON.stringify({ verdict: 'DEFECTS_FOUND', checked_files: [file], findings: [finding] }, null, 2)}\n\`\`\``
        }
    ];

    return [
        // 1. Precision Pair (Captured live from Colab G4 Slice 1): createCorsHeaders header-map vs errorEnvelope
        {
            id: 'dpo-precision-cors-header-map',
            mode: 'precision',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-01',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_cors_header_helper',
                        title: 'HTTP CORS Header Builder Verification',
                        files: ['lib/http.js'],
                        renderedSources: readNumberedWindow('lib/http.js', 120, 170)
                    })
                }
            ],
            chosen: cleanPassJson(
                ['lib/http.js'],
                'createCorsHeaders is a pure HTTP header-map helper (Record<string, string>) passed to res.writeHead; omitting Access-Control-Allow-Origin while setting Vary: Origin on null/false/CRLF origins is the exact W3C CORS rejection behavior.'
            ),
            rejected: defectJson('lib/http.js', {
                id: 'SRE-01',
                severity: 'P0',
                file: 'lib/http.js',
                line: 126,
                verbatim_quote: "export function createCorsHeaders(origin, allowedOrigins = ['*']) {",
                invariant: 'INV-LEGO-01',
                description:
                    'createCorsHeaders silently omits Access-Control-Allow-Origin for null/false/CRLF origins instead of returning errorEnvelope.',
                fix: 'Return errorEnvelope("INVALID_ORIGIN", "Origin header is invalid") from createCorsHeaders.'
            })
        },
        // 2. Precision Pair (Captured live from Colab G4 Slice 9a): dispatchFrontendTool try/catch wrapper + backward-compatible API
        {
            id: 'dpo-precision-mcp-try-catch-wrapper',
            mode: 'precision',
            project: 'krusch-frontend-mcp',
            invariant: 'INV-LEGO-01',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_frontend_mcp_dispatch',
                        title: 'MCP Tool Dispatcher & Canonical Template Lookup',
                        files: ['projects/krusch-frontend-mcp/src/index.js'],
                        renderedSources: readNumberedWindow('projects/krusch-frontend-mcp/src/index.js', 325, 355)
                    })
                }
            ],
            chosen: cleanPassJson(
                ['projects/krusch-frontend-mcp/src/index.js'],
                'dispatchFrontendTool wraps all switch branches inside try/catch (lines 346-351) returning { content: [...], isError: true }, so unknown tool/template throws never escape unhandled.'
            ),
            rejected: defectJson('projects/krusch-frontend-mcp/src/index.js', {
                id: 'SRE-01',
                severity: 'P0',
                file: 'projects/krusch-frontend-mcp/src/index.js',
                line: 344,
                verbatim_quote: 'throw new Error(`Unknown tool: ${name}`);',
                invariant: 'INV-LEGO-01',
                description: 'dispatchFrontendTool throws an uncaught Error for unknown tools.',
                fix: 'Replace throw with errorEnvelope.'
            })
        },
        // 3. Precision Pair: SHA-256 pre-hashed Bearer constant-time comparison
        {
            id: 'dpo-precision-bearer-sha256-constant-time',
            mode: 'precision',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-09',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_bearer_token_verify',
                        title: 'Constant-Time Bearer Token Verification',
                        files: ['lib/webhooks.js'],
                        renderedSources: readNumberedWindow('lib/webhooks.js', 320, 371)
                    })
                }
            ],
            chosen: cleanPassJson(
                ['lib/webhooks.js'],
                'verifyBearerToken hashes both candidate token and expectedSecret with SHA-256 (32 bytes each) before calling timingSafeEqual, preventing both timing attacks and secret-length leakage.'
            ),
            rejected: defectJson('lib/webhooks.js', {
                id: 'SRE-01',
                severity: 'P1',
                file: 'lib/webhooks.js',
                line: 335,
                verbatim_quote: 'export function verifyBearerToken(',
                invariant: 'INV-LEGO-09',
                description: 'verifyBearerToken leaks secret length when tokens have different lengths.',
                fix: 'Pad token strings to matching lengths before comparison.'
            })
        },
        // 4. Precision Pair: Package-scoped invariant boundary (do not apply krusch-stack INV-STACK-04 outside krusch-stack)
        {
            id: 'dpo-precision-package-scope-boundary',
            mode: 'precision',
            project: 'krusch-stack',
            invariant: 'INV-STACK-04',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_krusch_stack_public_scope',
                        title: 'Public Sanitization Scope Check in krusch-stack Frontend Engine',
                        files: ['projects/krusch-stack/src/engines/frontend-engine.js'],
                        extraSpec:
                            '- INV-STACK-04 (Public Sanitization — projects/krusch-stack ONLY): Files inside projects/krusch-stack/** MUST use 127.0.0.1 / localhost / env vars and zero private LAN/Tailscale IPs.',
                        renderedSources: readNumberedWindow(
                            'projects/krusch-stack/src/engines/frontend-engine.js',
                            1,
                            110
                        )
                    })
                }
            ],
            chosen: cleanPassJson(
                ['projects/krusch-stack/src/engines/frontend-engine.js'],
                'All templates inside projects/krusch-stack/src/engines/frontend-engine.js use 127.0.0.1/localhost/env defaults with zero private IPs or hostnames.'
            ),
            rejected: defectJson('projects/krusch-stack/src/engines/frontend-engine.js', {
                id: 'SRE-01',
                severity: 'P2',
                file: 'projects/krusch-stack/src/engines/frontend-engine.js',
                line: 1,
                verbatim_quote: '/**',
                invariant: 'INV-STACK-04',
                description: 'Consider renaming template keys to use custom prefixes for style consistency.',
                fix: 'Rename template identifiers.'
            })
        },
        // 5. Recall Pair (Captured live from Colab G4 Slice 2b): Catch uncaught fn() exception in withSingleInstanceLock
        {
            id: 'dpo-recall-cli-lock-callback-exception',
            mode: 'recall',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-10',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_cli_lock_exception',
                        title: 'Single-Instance Lock Callback Exception Handling',
                        files: ['lib/cli.js'],
                        renderedSources: [
                            '### FILE: `lib/cli.js` (showing lines 1-11)',
                            '```',
                            '1: export async function withSingleInstanceLock(lockFilePath, fn, { tool = "cli.lock" } = {}) {',
                            '2:     const { acquired, resolvedPath, nonce, holderPid } = acquireExclusiveLock(lockFilePath);',
                            '3:     if (!acquired) return errorEnvelope("LOCK_HELD", `Held by ${holderPid}`, { pid: holderPid }, tool);',
                            '4:     try {',
                            '5:         const result = await fn({ lockFilePath: resolvedPath, pid: process.pid, nonce });',
                            '6:         return okEnvelope(tool, "Completed locked execution", { lockFilePath: resolvedPath, result });',
                            '7:     } finally {',
                            '8:         releaseLockIfOwned(resolvedPath, nonce);',
                            '9:     }',
                            '10: }',
                            '```'
                        ].join('\n')
                    })
                }
            ],
            chosen: defectJson('lib/cli.js', {
                id: 'SRE-01',
                severity: 'P1',
                file: 'lib/cli.js',
                line: 5,
                verbatim_quote: 'const result = await fn({ lockFilePath: resolvedPath, pid: process.pid, nonce });',
                invariant: 'INV-LEGO-10',
                description:
                    'When fn() throws after acquiring the lock, withSingleInstanceLock propagates an uncaught exception instead of returning a deterministic INV-LEGO-01 errorEnvelope.',
                fix: 'Catch err inside try/catch and return errorEnvelope(String(err?.code || "LOCK_EXECUTION_ERROR"), err?.message || String(err), { lockFilePath: resolvedPath }, tool) unless options.rethrow === true.'
            }),
            rejected: cleanPassJson(['lib/cli.js'], 'Lock is released in finally; no issues found.')
        },
        // 6. Recall Pair: Catch Origin: null credential reflection
        {
            id: 'dpo-recall-cors-null-credentials',
            mode: 'recall',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-01',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_cors_null_origin',
                        title: 'CORS Origin Reflection Security Check',
                        files: ['lib/http.js'],
                        renderedSources: [
                            '### FILE: `lib/http.js` (showing lines 1-8)',
                            '```',
                            '1: export function createCorsHeaders(origin, allowedOrigins = []) {',
                            '2:     const clean = String(origin || "").replace(/[\\r\\n]+/g, "").trim();',
                            '3:     if (clean && allowedOrigins.includes(clean)) {',
                            '4:         return { "Access-Control-Allow-Origin": clean, "Access-Control-Allow-Credentials": "true", Vary: "Origin" };',
                            '5:     }',
                            '6:     return { Vary: "Origin" };',
                            '7: }',
                            '```'
                        ].join('\n')
                    })
                }
            ],
            chosen: defectJson('lib/http.js', {
                id: 'SRE-01',
                severity: 'P0',
                file: 'lib/http.js',
                line: 3,
                verbatim_quote: 'if (clean && allowedOrigins.includes(clean)) {',
                invariant: 'INV-LEGO-01',
                description:
                    'createCorsHeaders does not block Origin: "null" or "false", allowing credentialed sandboxed-iframe reflection if "null" enters allowedOrigins.',
                fix: 'Reject clean === "null" || clean === "false" and strip "null"/"false" from allowedOrigins before reflecting Access-Control-Allow-Credentials: true.'
            }),
            rejected: cleanPassJson(['lib/http.js'], 'CR/LF characters are stripped; CORS helper is safe.')
        },
        // 7. Recall Pair: Catch HTTP 413 req.destroy() TCP RST bug
        {
            id: 'dpo-recall-http-413-tcp-rst',
            mode: 'recall',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-03',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_http_413_drain',
                        title: 'Bounded JSON Body Parser 413 Handling',
                        files: ['lib/http.js'],
                        renderedSources: [
                            '### FILE: `lib/http.js` (showing lines 1-9)',
                            '```',
                            '1: if (bytesRead > maxBytes) {',
                            '2:     res.writeHead(413, { "Content-Type": "application/json", Connection: "close" });',
                            '3:     res.end(JSON.stringify(errorEnvelope("PAYLOAD_TOO_LARGE", "Payload too large", null, tool)));',
                            '4:     req.destroy();',
                            '5:     return { ok: false, handled: true };',
                            '6: }',
                            '```'
                        ].join('\n')
                    })
                }
            ],
            chosen: defectJson('lib/http.js', {
                id: 'SRE-01',
                severity: 'P1',
                file: 'lib/http.js',
                line: 4,
                verbatim_quote: 'req.destroy();',
                invariant: 'INV-LEGO-03',
                description:
                    'Calling req.destroy() while unread request chunks remain in the socket buffer triggers TCP ECONNRESET before the client receives the HTTP 413 JSON envelope.',
                fix: 'Call req.resume() instead of req.destroy() so the socket drains gracefully while Connection: close closes the connection after flushing the 413 body.'
            }),
            rejected: cleanPassJson(['lib/http.js'], 'Returns 413 PAYLOAD_TOO_LARGE envelope when maxBytes is exceeded.')
        },
        // 8. Recall Pair: Catch hand-rolled boilerplate violating INV-LEGO-01
        {
            id: 'dpo-recall-hand-rolled-pg-and-sse',
            mode: 'recall',
            project: 'sample-service',
            invariant: 'INV-LEGO-01',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: buildRefereeUserPrompt({
                        sliceId: 'dpo_anti_boilerplate',
                        title: 'Application Service Lego Compliance Audit',
                        files: ['projects/sample-service/src/server.js'],
                        renderedSources: [
                            '### FILE: `projects/sample-service/src/server.js` (showing lines 1-10)',
                            '```',
                            '1: import pg from "pg";',
                            '2: const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });',
                            '3: export function streamEvents(req, res) {',
                            '4:     res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no" });',
                            '5:     res.write("data: hello\\n\\n");',
                            '6: }',
                            '```'
                        ].join('\n')
                    })
                }
            ],
            chosen: defectJson('projects/sample-service/src/server.js', {
                id: 'SRE-01',
                severity: 'P1',
                file: 'projects/sample-service/src/server.js',
                line: 2,
                verbatim_quote: 'const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });',
                invariant: 'INV-LEGO-01',
                description:
                    'Service hand-rolls raw pg.Pool instantiation and manual text/event-stream SSE headers without AbortController disconnect cleanup instead of importing @krusch/toolkit/pg and @krusch/toolkit/streaming.',
                fix: 'Replace new pg.Pool with createPgPool from @krusch/toolkit/pg and replace manual SSE res.writeHead with sseResponse / streamChat from @krusch/toolkit/streaming.'
            }),
            rejected: cleanPassJson(
                ['projects/sample-service/src/server.js'],
                'Database pool uses DATABASE_URL and SSE sets X-Accel-Buffering: no.'
            )
        }
    ];
}

export function harvestLegoAndRefereeDatasets({
    sftOutPath = LEGO_CALIBRATION_SFT_PATH,
    dpoOutPath = REFEREE_DPO_PATH,
    nowIso = '2026-10-30T18:00:00.000Z'
} = {}) {
    fs.mkdirSync(path.dirname(sftOutPath), { recursive: true });
    fs.mkdirSync(path.dirname(dpoOutPath), { recursive: true });

    const cleanPassRows = buildCleanPassCalibrationRows(nowIso);
    const remediationRows = buildDefectRemediationRows(nowIso);
    const legoFirstRows = buildLegoFirstSynthesisRows(nowIso);
    const allSftRows = [...cleanPassRows, ...remediationRows, ...legoFirstRows];

    const dpoPairs = buildRefereeDpoPairs();

    fs.writeFileSync(sftOutPath, `${allSftRows.map((r) => JSON.stringify(r)).join('\n')}\n`, 'utf8');
    fs.writeFileSync(dpoOutPath, `${dpoPairs.map((p) => JSON.stringify(p)).join('\n')}\n`, 'utf8');

    return {
        ok: true,
        tool: 'harvestLegoAndRefereeDatasets',
        summary: `Harvested ${allSftRows.length} SFT rows (${cleanPassRows.length} CLEAN_PASS, ${remediationRows.length} remediations, ${legoFirstRows.length} Lego-First) and ${dpoPairs.length} balanced DPO pairs`,
        data: {
            sftOutPath,
            dpoOutPath,
            counts: {
                sftTotal: allSftRows.length,
                cleanPass: cleanPassRows.length,
                remediations: remediationRows.length,
                legoFirst: legoFirstRows.length,
                dpoTotal: dpoPairs.length,
                dpoPrecision: dpoPairs.filter((p) => p.mode === 'precision').length,
                dpoRecall: dpoPairs.filter((p) => p.mode === 'recall').length
            }
        }
    };
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
    const res = harvestLegoAndRefereeDatasets();
    console.log(JSON.stringify(res, null, 2));
}
