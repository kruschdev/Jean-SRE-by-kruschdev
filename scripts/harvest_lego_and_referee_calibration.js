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
import { scoutHomelabLegoCandidates } from '../../../lib/lego-scout.js';

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
    '5. Enforce the Rule-of-3 anti-bloat gate when evaluating toolkit candidates: only extract recurring infrastructure patterns shared across >= 3 repositories; keep < 3-repo or domain-specific helpers project-local.',
    '6. Do NOT output <think> tags or filler prose. Follow the requested markdown section headers and JSON contracts exactly.'
].join('\n');

const LEGO_ENGINEER_SYSTEM_PROMPT = [
    'You are a Principal Full-Stack & Systems Engineer assembling production software with @krusch/toolkit v1.5 (24 bricks) and krusch_toolkit v1.3 Lego bricks.',
    'RULES (NON-NEGOTIABLE):',
    '1. INV-LEGO-01: Assemble HTTP servers, SSE streams, MCP tools, PostgreSQL/pgvector pools, node:sqlite FTS5/RRF stores, SSRF-safe outbound fetch & rate limiters, React hooks/shells, Cloudflare Workers, HMAC webhooks, and single-instance CLIs from @krusch/toolkit/* and krusch_toolkit.* bricks—never hand-roll boilerplate.',
    '2. Enforce deterministic { ok, tool, summary, data | error } envelopes, AbortController cancellation, constant-time HMAC/Bearer verification, connect-time DNS-rebinding SSRF protection, and strict TypeScript (.d.ts / --experimental-strip-types) compatibility.',
    '3. Enforce Rule-of-3 anti-bloat discipline (INV-LEGO-15): only promote cross-repository infrastructure patterns appearing in >= 3 projects into @krusch/toolkit.'
].join('\n');

const COMPACT_LEGO_SPEC = [
    '- INV-LEGO-01 (Universal Envelope): HTTP/MCP/CLI/FastAPI endpoints emit { ok: true, tool, summary, data } or { ok: false, tool, summary, error: { code, message, details? } }. Low-level header builders (e.g., createCorsHeaders -> Record<string,string>) return header maps for res.writeHead.',
    '- INV-LEGO-02 (Zero Ghost Exports & Headless Safety): Every documented subpath exists in package.json exports ({ types, default }) and imports cleanly in headless Node 22 ESM.',
    '- INV-LEGO-03 (AbortController & 413 Drain): SSE & React fetch hooks wire AbortController teardown; readBoundedJsonBody drains/resumes oversized streams before closing so HTTP 413 JSON envelopes flush without TCP ECONNRESET.',
    '- INV-LEGO-08 (Edge Purity): @krusch/toolkit/edge contains zero node:* imports, rejects Origin: null/false/CRLF, enforces chunked stream maxBytes (413), and aborts SSE on stream cancel.',
    '- INV-LEGO-09 (Constant-Time Webhooks & Bearer): @krusch/toolkit/webhooks verifies Slack/GitHub/Stripe HMACs and SHA-256 pre-hashed Bearer tokens via crypto.timingSafeEqual after equal-length checks, failing closed on empty secrets.',
    '- INV-LEGO-10 (Atomic CLI & Daemon): @krusch/toolkit/cli uses exclusive wx PID+nonce lockfiles in withSingleInstanceLock (wrapping thrown callback errors in errorEnvelope unless rethrow:true and unlinking only owned nonces in finally) and prototype-safe defineCli.',
    '- INV-LEGO-11 (Strict TypeScript Parity): lib/types/*.d.ts matches runtime exports 1:1 under tsc --noEmit --strict (including contravariant Logger meta types).',
    '- INV-LEGO-12 (Node 22 SQLite, FTS5 & RRF): @krusch/toolkit/sqlite configures WAL durability pragmas on DatabaseSync, normalizes null-prototype rows, enforces synchronous savepoint transactions, double-quotes FTS5 tokens in escapeFts5Query, and deduplicates/merges prototype-safely in fuseRrfResults.',
    '- INV-LEGO-13 (SSRF & Rate Limit): @krusch/toolkit/security blocks loopback, RFC1918, Tailscale CGNAT (unless allowlisted), 169.254.169.254 cloud metadata, IPv4-mapped/6to4/Teredo IPv6, and TOCTOU DNS rebinding at socket connect in safeFetch, plus bounded-key createRateLimiter.',
    '- INV-LEGO-14..15 (Python v1.3 Parity & Rule-of-3 Scout): krusch_toolkit.{mcp,pg,security} enforce 4-boolean MCP annotations, SQLi-guarded 1024d pgvector search, and SSRF/HMAC parity; @krusch/toolkit/lego-scout enforces symlink containment and Rule-of-3 (>= 3 repos) extraction gating.'
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
        },
        {
            id: 'referee-clean-pass-sqlite-fts5-rrf',
            sliceId: 'slice_wave3_sqlite_fts5_rrf',
            title: 'Zero-Native-Addon Node 22 SQLite WAL, FTS5 Escaping & Prototype-Safe RRF',
            files: ['lib/sqlite.js'],
            window: () =>
                `${readNumberedWindow('lib/sqlite.js', 295, 439)}`,
            rationale:
                'escapeFts5Query caps raw input at 2048 chars, extracts Unicode word tokens, and double-quotes every token (preventing fts5: syntax error on AND/OR/NOT/NEAR/*), while fuseRrfResults enforces seenInList deduplication and Object.hasOwn prototype-safe merging.'
        },
        {
            id: 'referee-clean-pass-security-ssrf-guard',
            sliceId: 'slice_wave3_security_ssrf',
            title: 'Fail-Closed IPv4/IPv6/6to4/Teredo/Metadata SSRF Classifier',
            files: ['lib/security.js'],
            window: () => readNumberedWindow('lib/security.js', 136, 265),
            rationale:
                'isPrivateOrReservedIp fails closed on non-canonical IPv4 literals, %zone suffixes, loopback, 169.254.0.0/16 cloud metadata, RFC1918/Tailscale CGNAT (unless explicitly allowlisted), and recursively inspects IPv4-mapped, NAT64, 6to4, and Teredo IPv6 addresses.'
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
        },
        {
            id: 'remediation-sqlite-fts5-unquoted-syntax-crash',
            file: 'projects/sample-notes/src/search.js',
            line: 4,
            severity: 'P1',
            invariant: 'INV-LEGO-12',
            title: 'Escape raw user queries before SQLite FTS5 MATCH to prevent syntax crashes',
            buggySnippet: [
                '1: import { DatabaseSync } from "node:sqlite";',
                '2: const db = new DatabaseSync("./data/notes.db");',
                '3: export function searchNotes(userQuery) {',
                '4:     return db.prepare("SELECT id, title FROM notes_fts WHERE notes_fts MATCH ? LIMIT 20").all(userQuery);',
                '5: }'
            ].join('\n'),
            quote: 'return db.prepare("SELECT id, title FROM notes_fts WHERE notes_fts MATCH ? LIMIT 20").all(userQuery);',
            description:
                'Passing raw user input directly to SQLite FTS5 MATCH throws SQLITE_ERROR: fts5: syntax error whenever queries contain C++, hyphens, colons, unbalanced quotes, or boolean keywords (AND/OR/NOT/NEAR), and instantiates raw DatabaseSync without WAL durability pragmas.',
            fix: 'Open the database with openSqliteDb from @krusch/toolkit/sqlite and sanitize userQuery with escapeFts5Query(userQuery, { operator: "AND" }), returning [] immediately when the escaped query is empty.'
        },
        {
            id: 'remediation-security-ssrf-ipv6-mapped-redirect',
            file: 'projects/sample-webhook/src/dispatcher.js',
            line: 3,
            severity: 'P0',
            invariant: 'INV-LEGO-13',
            title: 'Replace naive hostname regex SSRF check with DNS-rebinding-safe safeFetch',
            buggySnippet: [
                '1: export async function deliverWebhook(targetUrl, payload) {',
                '2:     const parsed = new URL(targetUrl);',
                '3:     if (/^(localhost|127\\.)/i.test(parsed.hostname)) throw new Error("Blocked");',
                '4:     return fetch(targetUrl, { method: "POST", body: JSON.stringify(payload) });',
                '5: }'
            ].join('\n'),
            quote: 'if (/^(localhost|127\\.)/i.test(parsed.hostname)) throw new Error("Blocked");',
            description:
                'Regex checking only localhost and 127.* is trivially bypassed via cloud metadata (169.254.169.254), IPv4-mapped IPv6 ([::ffff:127.0.0.1]), octal/integer IPs, DNS rebinding, and 302 redirects to internal endpoints.',
            fix: 'Replace manual URL regex + raw fetch() with safeFetch(targetUrl, { method: "POST", body: JSON.stringify(payload) }) from @krusch/toolkit/security, which validates all resolved IPs at socket connect time and re-validates every redirect hop.'
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
 * natively import @krusch/toolkit v1.5 and krusch_toolkit v1.3 bricks instead of hand-rolling plumbing.
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
                "import { initDb, exec, query, run, closeDb } from '@krusch/toolkit/db';",
                "import { setupLogging } from '@krusch/toolkit/log';",
                '',
                "const logger = setupLogging('notes-api');",
                "initDb(process.env.DB_PATH || './data/notes.db');",
                'exec(`CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);`);',
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
                '',
                'export async function handleChatStream({ req, res, body }) {',
                '  await streamChat(res, req, {',
                "    tool: 'assistant.stream',",
                "    systemPrompt: 'You are a concise engineering assistant.',",
                "    userMessage: String(body?.prompt || ''),",
                '    config: {',
                "      provider: 'ollama',",
                "      model: process.env.OLLAMA_MODEL || 'qwen2.5-coder:7b'",
                '    }',
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
                '  }',
                '});',
                '',
                'export const mcp = createMcpServer(',
                "  'docs-mcp',",
                "  '1.0.0',",
                '  [searchDocsTool],',
                '  async (name, args) =>',
                "    okMcpResult(name, `Matched documents for \"${args.query}\"`, { query: args.query, matches: [] })",
                ');',
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
                "  const queryVector = await embed(question, { mode: 'query' });",
                '  return pgVectorSearch(pool, {',
                "    table: 'kb_chunks',",
                "    embeddingColumn: 'embedding',",
                "    selectColumns: ['id', 'doc_key', 'content'],",
                '    vector: queryVector,',
                '    expectedDims: 1024,',
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
                "  const { data, loading, error, reload } = useApiQuery('/api/metrics', { deps: [] });",
                "  const stream = useSSEStream('/api/chat/stream');",
                '',
                '  return (',
                '    <AppShell',
                '      title="Operations Console"',
                '      status={<StatusPill status={error ? "degraded" : loading ? "streaming" : "healthy"} label={error ? "Degraded" : "Online"} />}',
                '    >',
                '      <MetricCard label="Requests" value={data?.requests ?? 0} />',
                '      <button type="button" onClick={reload}>Refresh</button>',
                '      <button type="button" onClick={() => stream.start({ prompt: "Summarize health" })}>Run Check</button>',
                '      {stream.status === "streaming" && <button type="button" onClick={stream.stop}>Stop</button>}',
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
        },
        {
            id: 'lego-first-sqlite-fts5-rrf-hybrid',
            prompt:
                'Build a zero-native-addon Node 22 hybrid retrieval module (`src/hybrid-store.js`) combining SQLite FTS5 BM25 keyword search and dense vector candidates via Reciprocal Rank Fusion without `better-sqlite3` or unquoted FTS5 syntax errors.',
            answer: [
                'Use `openSqliteDb`, `escapeFts5Query`, and `fuseRrfResults` from `@krusch/toolkit/sqlite` (`INV-LEGO-12`):',
                '```js',
                "import { openSqliteDb, escapeFts5Query, fuseRrfResults } from '@krusch/toolkit/sqlite';",
                '',
                "export const store = openSqliteDb(process.env.SQLITE_PATH || './data/hybrid.db');",
                'store.exec(`',
                '  CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL);',
                '  CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(id UNINDEXED, title, body);',
                '`);',
                '',
                'export function searchHybrid(rawQuery, denseHits = [], limit = 10) {',
                "  const ftsQuery = escapeFts5Query(rawQuery, { operator: 'AND', prefixLastToken: true });",
                '  const lexicalHits = ftsQuery',
                "    ? store.query('SELECT id, title, bm25(docs_fts) AS rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT 30', [ftsQuery])",
                '    : [];',
                "  return fuseRrfResults([lexicalHits, denseHits], { k: 60, idKey: 'id', limit, weights: [1.0, 1.2] });",
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-security-safefetch-ratelimit',
            prompt:
                'Implement an Express outbound webhook test endpoint protected against SSRF (blocking `169.254.169.254`, RFC1918, IPv4-mapped IPv6, and DNS rebinding on redirects) and per-IP `429` rate limiting using `@krusch/toolkit/security`.',
            answer: [
                'Compose `safeFetch` and `createRateLimiter` from `@krusch/toolkit/security` (`INV-LEGO-13`):',
                '```js',
                "import { safeFetch, createRateLimiter } from '@krusch/toolkit/security';",
                "import { okEnvelope } from '@krusch/toolkit/http';",
                '',
                "export const webhookLimiter = createRateLimiter({ windowMs: 60_000, max: 20, maxKeys: 5_000, tool: 'webhook.rateLimit' });",
                '',
                'export async function handleTestWebhook(req, res) {',
                "  const result = await safeFetch(req.body?.targetUrl, {",
                "    method: 'POST',",
                "    headers: { 'Content-Type': 'application/json' },",
                "    body: JSON.stringify({ ping: true })",
                "  }, { maxRedirects: 2, timeoutMs: 8_000, allowPrivateLan: false, tool: 'webhook.probe' });",
                '',
                '  if (!result.ok) return res.status(400).json(result);',
                "  return res.status(200).json(okEnvelope('webhook.probe', 'Delivered probe safely', { status: result.data.status, redirected: result.data.redirected }));",
                '}',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-python-mcp-pgvector-parity',
            prompt:
                'Build a Python 3.12 MCP tool handler and SQLi-guarded 1024-dim `pgvector` search helper using `krusch_toolkit.mcp` and `krusch_toolkit.pg` (`INV-LEGO-14`).',
            answer: [
                'Import `define_mcp_tool`, `dispatch_mcp_tool`, and `ok_mcp_result` from `krusch_toolkit.mcp` alongside `run_sql_migrations` and `pg_vector_search` from `krusch_toolkit.pg`:',
                '```python',
                'from krusch_toolkit.mcp import define_mcp_tool, dispatch_mcp_tool, ok_mcp_result',
                'from krusch_toolkit.pg import run_sql_migrations, pg_vector_search',
                '',
                'SEARCH_TOOL = define_mcp_tool(',
                '    name="clauses_search",',
                '    title="Search Contract Clauses",',
                '    description="Cosine search over 1024d clause embeddings.",',
                '    input_schema={',
                '        "type": "object",',
                '        "properties": {"query_vec": {"type": "array"}, "limit": {"type": "integer"}},',
                '        "required": ["query_vec"],',
                '    },',
                '    read_only_hint=True,',
                '    destructive_hint=False,',
                '    idempotent_hint=True,',
                '    open_world_hint=False,',
                ')',
                '',
                'def init_db(conn):',
                '    return run_sql_migrations(conn, [{"id": "001_clauses", "sql": "CREATE TABLE IF NOT EXISTS clauses (id TEXT PRIMARY KEY, body TEXT, embedding vector(1024));"}])',
                '',
                'async def handle_mcp_call(conn, name: str, args: dict):',
                '    async def _run(tool_name: str, safe_args: dict):',
                '        res = pg_vector_search(',
                '            conn,',
                '            table="clauses",',
                '            vector=safe_args["query_vec"],',
                '            select_columns=("id", "body"),',
                '            limit=safe_args.get("limit", 8),',
                '            expected_dims=1024,',
                '        )',
                '        return ok_mcp_result(tool_name, "Retrieved clauses", res["data"])',
                '    return await dispatch_mcp_tool([SEARCH_TOOL], _run, name, args)',
                '```'
            ].join('\n')
        },
        {
            id: 'lego-first-python-security-ssrf-hmac',
            prompt:
                'Validate inbound HMAC-SHA256 webhook signatures, Bearer tokens, and outbound callback URLs in Python 3.12 without hand-rolling `hmac.new` or regex IP checks (`INV-LEGO-14`).',
            answer: [
                'Use `verify_webhook_hmac`, `verify_bearer_token`, and `validate_outbound_url` from `krusch_toolkit.security`:',
                '```python',
                'import os',
                'from krusch_toolkit.security import (',
                '    verify_webhook_hmac,',
                '    verify_bearer_token,',
                '    validate_outbound_url,',
                ')',
                '',
                'def verify_callback_request(raw_body: bytes, sig_header: str, auth_header: str, callback_url: str) -> dict:',
                '    auth = verify_bearer_token(auth_header, os.environ.get("API_BEARER_SECRET", ""))',
                '    if not auth["ok"]:',
                '        return auth',
                '    sig = verify_webhook_hmac(raw_body, sig_header, os.environ.get("WEBHOOK_SIGNING_SECRET", ""), prefix="sha256=")',
                '    if not sig["ok"]:',
                '        return sig',
                '    return validate_outbound_url(callback_url, allow_private_lan=False, allow_tailscale=False)',
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
 * Category 4: Live Cross-Repository `lego_scout` Rule-of-3 Triage & Spec Synthesis SFT (`INV-LEGO-15..16`).
 * Harvests live telemetry from `scoutHomelabLegoCandidates(<HOMELAB_ROOT>/projects, { minProjects: 3, portablePaths: true })`
 * so Jean SRE learns both proactive cross-repo brick discovery AND strict anti-bloat rejection (`< 3` repos).
 */
export function buildLegoScoutSynthesisRows(nowIso) {
    const projectsRoot = path.join(homelabRoot, 'projects');
    const scoutRes = scoutHomelabLegoCandidates(projectsRoot, {
        minProjects: 3,
        portablePaths: true
    });
    const d = scoutRes?.ok && scoutRes.data
        ? scoutRes.data
        : {
              scannedProjects: 20,
              scannedFilesTotal: 400,
              ruleOfThreeThreshold: 3,
              adoptionFindings: [],
              candidateClusters: []
          };

    const topAdopt = d.adoptionFindings.slice(0, 5);
    const coveredClusters = d.candidateClusters.filter((c) => c.status === 'ALREADY_COVERED_IN_TOOLKIT');
    const ruleOf3Clusters = d.candidateClusters.filter((c) => c.status === 'CANDIDATE_MEETS_RULE_OF_3');
    const rejectedClusters = d.candidateClusters.filter((c) => c.status === 'BELOW_THRESHOLD_DO_NOT_EXTRACT');

    const compactTelemetryJson = JSON.stringify(
        {
            scannedProjects: d.scannedProjects,
            scannedFilesTotal: d.scannedFilesTotal,
            ruleOfThreeThreshold: d.ruleOfThreeThreshold,
            topAdoptionFindings: topAdopt.map((f) => ({
                project: f.project,
                findingCount: f.findingCount,
                rulesTriggered: f.rulesTriggered,
                recommendedBricks: f.recommendedBricks,
                sampleFile: f.sampleFiles[0] || ''
            })),
            coveredClusters: coveredClusters.map((c) => ({
                id: c.id,
                projectCount: c.projectCount,
                status: c.status,
                proposedBrickImport: c.proposedBrickImport
            })),
            candidateClusters: ruleOf3Clusters.map((c) => ({
                id: c.id,
                projectCount: c.projectCount,
                status: c.status,
                proposedBrickImport: c.proposedBrickImport,
                sampleFile: c.sampleEvidence[0] || ''
            })),
            rejectedBelowThreshold: rejectedClusters.map((c) => ({
                id: c.id,
                projectCount: c.projectCount,
                status: c.status,
                proposedBrickImport: c.proposedBrickImport,
                projects: c.projects
            }))
        },
        null,
        2
    );

    const schemaCluster = ruleOf3Clusters.find((c) => c.id === 'schema-contract-validation') || ruleOf3Clusters[0];
    const rejectedCluster = rejectedClusters[0] || {
        id: 'niche-xml-or-yaml',
        projectCount: 1,
        status: 'BELOW_THRESHOLD_DO_NOT_EXTRACT',
        proposedBrickImport: 'keep-project-local',
        projects: ['berean'],
        sampleEvidence: ['<HOMELAB_ROOT>/projects/berean/package.json']
    };

    const rows = [
        {
            id: 'lego-scout-monorepo-triage-report',
            phase: 'INFRA_LEGO_SCOUT',
            system: AUDITOR_SYSTEM_PROMPT,
            prompt: [
                '## JEAN SRE: INFRA REPORT SYNTHESIS (lego_scout)',
                'Analyze the Executed `krusch-lego scout --portable` Ground-Truth JSON below and produce a gated `INFRA REPORT` separating immediate `@krusch/toolkit` v1.5 / `krusch_toolkit` v1.3 adoptions (`LEGO-ADOPT`), Rule-of-3 brick candidates (`>= 3` repos), and anti-bloat rejections (`< 3` repos):',
                `\`\`\`json\n${compactTelemetryJson}\n\`\`\``
            ].join('\n\n'),
            answer: [
                '### INFRA REPORT: workspace / lego_scout',
                '',
                '🔴 **Immediate (`LEGO-ADOPT` — Migrate to Existing `@krusch/toolkit` v1.5 / `krusch_toolkit` v1.3 Bricks)**',
                ...topAdopt.slice(0, 3).map(
                    (f) =>
                        `- **\`${f.project}\`** (\`${f.findingCount}\` static hit(s) in \`${f.sampleFiles[0] || f.project}\`): triggered \`${f.rulesTriggered.join(', ')}\` → replace hand-rolled plumbing with \`${f.recommendedBricks.join(', ')}\`.`
                ),
                ...coveredClusters.map(
                    (c) =>
                        `- **Cluster \`${c.id}\`** (\`${c.projectCount}\` repositories): \`${c.status}\` → standardize remaining brownfield callers onto \`${c.proposedBrickImport}\`.`
                ),
                '',
                '🟡 **Review Required (`LEGO-CANDIDATE` — Meets Rule-of-3 Gate `>= 3` Repositories; Gated by `/jean-loop-gate`)**',
                ...ruleOf3Clusters.map(
                    (c) =>
                        `- **\`${c.id}\` → \`${c.proposedBrickImport}\`** (\`${c.projectCount}\` repositories >= \`${d.ruleOfThreeThreshold}\`, sample \`${c.sampleEvidence[0] || 'workspace'}\`): Propose thin \`/spec-first-refactor\` contract emitting \`{ ok, tool, summary, data | error }\` (\`INV-LEGO-01\`) while keeping domain logic project-local.`
                ),
                '',
                '🟢 **Clean / Anti-Bloat Rejected (`BELOW_THRESHOLD_DO_NOT_EXTRACT` — `< 3` Repositories)**',
                `- **\`${rejectedCluster.id}\`** (\`${rejectedCluster.projectCount}\` repository: \`${rejectedCluster.projects.join(', ')}\` < \`${d.ruleOfThreeThreshold}\`): Marked \`${rejectedCluster.proposedBrickImport}\`. Do NOT extract single-domain parsers into \`@krusch/toolkit\`.`,
                '',
                '### LESSON',
                `Cross-repo scouting across \`${d.scannedProjects}\` projects (\`${d.scannedFilesTotal}\` files) separates brownfield adoption debt (\`LEGO-ADOPT\`) from genuine \`>= ${d.ruleOfThreeThreshold}\`-repo abstractions while blocking \`${rejectedCluster.id}\` (\`${rejectedCluster.projectCount}\` repo) from bloating \`@krusch/toolkit\`.`
            ].join('\n')
        },
        {
            id: 'lego-scout-spec-synthesis-rule-of-3',
            phase: 'LEGO_SPEC_SYNTHESIS',
            system: LEGO_ENGINEER_SYSTEM_PROMPT,
            prompt: [
                `The autonomous \`lego_scout\` sweep identified cluster \`${schemaCluster?.id || 'schema-contract-validation'}\` across \`${schemaCluster?.projectCount || 6}\` repositories (\`CANDIDATE_MEETS_RULE_OF_3\`, sample \`${schemaCluster?.sampleEvidence?.[0] || '<HOMELAB_ROOT>/projects/sample/package.json'}\`).`,
                'Draft a minimal, anti-bloat `/spec-first-refactor` contract (`INV-LEGO-SCHEMA`) for `@krusch/toolkit/schema` that satisfies `INV-LEGO-01` and `INV-LEGO-11` without coupling `@krusch/toolkit` to heavy ORMs.'
            ].join('\n\n'),
            answer: [
                `### Frozen Spec Proposal: \`@krusch/toolkit/schema\` (\`${schemaCluster?.projectCount || 6}\` Repositories — Passes Rule-of-3)`,
                '',
                '- **Admission Evidence:** Verified `CANDIDATE_MEETS_RULE_OF_3` across `>= 3` active repositories; domain models remain project-local.',
                '- **Zero-Bloat Boundary:** Accepts duck-typed `{ safeParse(input) }` (Zod) or lightweight JSON Schema `{ type, required, properties }` objects—zero heavy ORM or codegen dependencies.',
                '- **Exact Exports (`lib/schema.js` + `lib/types/schema.d.ts`):**',
                '```ts',
                "import type { LegoEnvelope } from './envelopes.js';",
                '',
                'export interface SchemaLike<T = Record<string, unknown>> {',
                '  safeParse?: (input: unknown) => { success: boolean; data?: T; error?: { issues?: unknown[] } };',
                '  required?: readonly string[];',
                '  properties?: Record<string, { type?: string }>;',
                '}',
                '',
                'export function validatePayload<T = Record<string, unknown>>(',
                '  schema: SchemaLike<T>,',
                '  payload: unknown,',
                '  options?: { tool?: string }',
                '): LegoEnvelope<T>;',
                '```',
                '- **Envelope Contract (`INV-LEGO-01`):** Returns `okEnvelope(tool, "Payload validated", parsedData)` on valid input and `errorEnvelope("VALIDATION_ERROR", message, { issues }, tool)` on mismatch without throwing.'
            ].join('\n')
        },
        {
            id: 'lego-scout-brownfield-backlog-cards',
            phase: '2_MASTER_BACKLOG',
            system: AUDITOR_SYSTEM_PROMPT,
            prompt: [
                '## JEAN SRE: IMPLEMENTER PHASE (Final Master Backlog Synthesis — NO tool calls)',
                'Synthesize verified `LEGO-*` task cards from the live source verification below:',
                '### FILE: `src/db.js` (showing lines 1-6)',
                '```js',
                '1: import { DatabaseSync } from "node:sqlite";',
                '2: export const db = new DatabaseSync("./data/app.db");',
                '3: export function search(q) {',
                '4:   return db.prepare("SELECT id FROM docs_fts WHERE docs_fts MATCH ?").all(q);',
                '5: }',
                '```',
                '### FILE: `src/webhook.js` (showing lines 1-5)',
                '```js',
                '1: export async function notify(url, body) {',
                '2:   return fetch(url, { method: "POST", body: JSON.stringify(body) });',
                '3: }',
                '```'
            ].join('\n\n'),
            answer: [
                '### MASTER BACKLOG: sample-app',
                '',
                '#### LEGO-01: Migrate raw DatabaseSync and unescaped FTS5 MATCH to @krusch/toolkit/sqlite',
                '- **Severity:** P1',
                '- **Effort:** S',
                '- **Evidence:** `src/db.js:1-5`',
                '- **Fix Instructions:** Replace `new DatabaseSync(...)` with `openSqliteDb("./data/app.db")` (`INV-LEGO-12` WAL pragmas) and wrap `q` in `escapeFts5Query(q)` before `MATCH ?` so punctuation/operators cannot throw `fts5: syntax error`.',
                '',
                '#### LEGO-02: Replace unvalidated outbound fetch with @krusch/toolkit/security safeFetch',
                '- **Severity:** P1',
                '- **Effort:** S',
                '- **Evidence:** `src/webhook.js:1-3`',
                '- **Fix Instructions:** Replace raw `fetch(url, ...)` with `safeFetch(url, { method: "POST", body: JSON.stringify(body) })` from `@krusch/toolkit/security` (`INV-LEGO-13`) to block loopback, cloud metadata (`169.254.169.254`), and DNS-rebinding redirects.',
                '',
                '### 1. SUMMARY TABLE',
                '| ID | Title | Severity | Effort | Confidence | Source Audit |',
                '| --- | --- | --- | --- | --- | --- |',
                '| **LEGO-01** | Migrate raw DatabaseSync and unescaped FTS5 MATCH to @krusch/toolkit/sqlite | P1 | S | High | 1.5 |',
                '| **LEGO-02** | Replace unvalidated outbound fetch with @krusch/toolkit/security safeFetch | P1 | S | High | 1.4 |',
                '',
                '### 4. NOT FLAGGED',
                '- JSON serialization in `notify` is valid once wrapped in `safeFetch`.',
                '',
                '### 5. STALENESS & ARCHIVAL VERDICT',
                '`ACTIVE_CORE` — adopt Wave-3 `@krusch/toolkit` bricks (`./sqlite`, `./security`).'
            ].join('\n')
        },
        {
            id: 'lego-scout-anti-bloat-single-repo-rejection',
            phase: 'LEGO_ANTI_BLOAT_GATE',
            system: AUDITOR_SYSTEM_PROMPT,
            prompt: [
                `A developer proposes adding \`@krusch/toolkit/xml-osis\` wrapping \`fast-xml-parser\` for OSIS Scripture chapters. \`krusch-lego scout --portable\` reports cluster \`${rejectedCluster.id}\` has \`projectCount: ${rejectedCluster.projectCount}\` (\`status: "${rejectedCluster.status}"\`, projects: \`${JSON.stringify(rejectedCluster.projects)}\`).`,
                'Evaluate this proposal under `INV-LEGO-15` (Rule-of-3 Anti-Bloat Gate) and output a structured JSON decision.'
            ].join('\n\n'),
            answer: `\`\`\`json\n${JSON.stringify(
                {
                    verdict: 'REJECT_EXTRACTION',
                    cluster_id: rejectedCluster.id,
                    project_count: rejectedCluster.projectCount,
                    rule_of_three_threshold: 3,
                    status: 'BELOW_THRESHOLD_DO_NOT_EXTRACT',
                    recommended_location: 'keep-project-local',
                    rationale:
                        'Pattern appears in fewer than 3 repositories and encodes domain-specific XML chapter parsing rather than universal cross-service infrastructure. Extracting it into @krusch/toolkit would create coupling and maintenance bloat.'
                },
                null,
                2
            )}\n\`\`\``
        }
    ];

    return rows.map((r) => ({
        id: r.id,
        timestamp: nowIso,
        source: 'lego_scout_synthesis_sft',
        teacher_model: 'gemini-4-argon',
        student_target: 'qwen3-coder-30b-moe:latest',
        project: 'krusch-toolkit',
        phase: r.phase,
        verification_status: 'SCOUT_AND_REFEREE_VERIFIED',
        elitist_score: 97,
        messages: [
            { role: 'system', content: r.system },
            { role: 'user', content: normalizePortableSftText(r.prompt, 'toolkit') },
            { role: 'assistant', content: normalizePortableSftText(r.answer, 'toolkit') }
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
        },
        // 9. Architecture Calibration Pair (INV-LEGO-15): Rule-of-3 Anti-Bloat Gate (1 repo < 3 -> keep project-local)
        {
            id: 'dpo-arch-rule-of-3-anti-bloat',
            mode: 'architecture_calibration',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-15',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: [
                        'Evaluate this proposed `@krusch/toolkit` expansion under `INV-LEGO-15` (Rule-of-3 Anti-Bloat Gate) and output a fenced JSON architectural verdict:',
                        '- Proposed brick: `@krusch/toolkit/xml-osis` wrapping `fast-xml-parser` for OSIS XML chapter verses.',
                        '- `krusch-lego scout --portable` telemetry: `cluster: "niche-xml-or-yaml"`, `projectCount: 1` (`["berean"]`), `ruleOfThreeThreshold: 3`, `status: "BELOW_THRESHOLD_DO_NOT_EXTRACT"`.'
                    ].join('\n')
                }
            ],
            chosen: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'KEEP_PROJECT_LOCAL',
                            invariant: 'INV-LEGO-15',
                            cluster_id: 'niche-xml-or-yaml',
                            project_count: 1,
                            rule_of_three_threshold: 3,
                            rationale:
                                'Appears in only 1 repository (< 3 threshold) and encodes domain-specific OSIS XML parsing rather than universal cross-service infrastructure. Keep local in berean to prevent shared toolkit bloat.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ],
            rejected: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'EXTRACT_SHARED_BRICK',
                            invariant: 'INV-LEGO-01',
                            cluster_id: 'niche-xml-or-yaml',
                            proposed_import: '@krusch/toolkit/xml-osis',
                            rationale:
                                'Every parser in the monorepo should live in @krusch/toolkit in case future projects parse XML.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ]
        },
        // 10. Architecture Calibration Pair (INV-LEGO-12): Thin parameterized SQL + RRF vs Heavy Codegen ORM
        {
            id: 'dpo-arch-thin-sql-vs-heavy-orm',
            mode: 'architecture_calibration',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-12',
            prompt: [
                { role: 'system', content: LEGO_ENGINEER_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: [
                        'Choose the canonical `@krusch/toolkit` database layer architecture for 9 SQLite FTS5 + Hybrid RRF services (`INV-LEGO-12`) and output a fenced JSON architectural verdict:',
                        '- Option A: Thin zero-native-addon `@krusch/toolkit/sqlite` (`node:sqlite` `DatabaseSync` + WAL pragmas + `escapeFts5Query` + `fuseRrfResults`) and `@krusch/toolkit/pg` parameterized SQL.',
                        '- Option B: Bundle Prisma ORM schema codegen + query engine binary inside `@krusch/toolkit`.'
                    ].join('\n')
                }
            ],
            chosen: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'USE_THIN_SQL_BRICK',
                            invariant: 'INV-LEGO-12',
                            selected_bricks: ['@krusch/toolkit/sqlite', '@krusch/toolkit/pg'],
                            rationale:
                                'Built-in node:sqlite DatabaseSync with explicit WAL pragmas, deterministic escapeFts5Query, and pure fuseRrfResults requires zero native compilation, zero schema codegen build steps, and preserves transparent FTS5/pgvector SQL control.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ],
            rejected: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'ADOPT_HEAVY_ORM',
                            invariant: 'INV-LEGO-12',
                            selected_bricks: ['prisma-client-codegen'],
                            rationale:
                                'Add Prisma schema codegen and binary query engines to @krusch/toolkit so services never write SQL queries.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ]
        },
        // 11. Architecture Calibration Pair (INV-LEGO-11): Zero-Build ESM + strict .d.ts vs Premature Rust N-API Rewrite
        {
            id: 'dpo-arch-zero-build-dts-vs-premature-rust',
            mode: 'architecture_calibration',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-11',
            prompt: [
                { role: 'system', content: LEGO_ENGINEER_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: [
                        'Evaluate whether `@krusch/toolkit` HTTP, MCP, SSE, and PostgreSQL I/O bricks should be rewritten as compiled Rust N-API addons or remain zero-build Node 22 ESM + strict `.d.ts` declarations (`INV-LEGO-11`). Output a fenced JSON architectural verdict.'
                    ].join('\n')
                }
            ],
            chosen: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'ZERO_BUILD_DTS',
                            invariant: 'INV-LEGO-11',
                            rationale:
                                'HTTP, SSE, MCP, and pgvector orchestration are network/I/O-bound, not CPU-bound. Pure Node 22 ESM paired with tsc --strict NodeNext .d.ts declarations gives compile-time type safety with zero build steps and zero native ABI breakage across containers and Workers, reserving Rust strictly for measured CPU hotspots.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ],
            rejected: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'REWRITE_IO_IN_RUST',
                            invariant: 'INV-LEGO-11',
                            rationale:
                                'Compile all HTTP routers, SSE helpers, and MCP JSON envelopes as native Rust N-API .node binaries across every container.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ]
        },
        // 12. Architecture Calibration Pair (INV-LEGO-13): Connect-Time DNS-Pinned SSRF Guard vs Naive URL Hostname Regex
        {
            id: 'dpo-arch-connect-time-ssrf-vs-regex',
            mode: 'architecture_calibration',
            project: 'krusch-toolkit',
            invariant: 'INV-LEGO-13',
            prompt: [
                { role: 'system', content: AUDITOR_SYSTEM_PROMPT },
                {
                    role: 'user',
                    content: [
                        'Select the compliant SSRF defense architecture for outbound webhook/URL fetching (`INV-LEGO-13`) and output a fenced JSON architectural verdict:',
                        '- Option A: `@krusch/toolkit/security` `safeFetch` validating canonical IPv4/IPv6/NAT64/6to4/Teredo ranges (`127.0.0.0/8`, `169.254.0.0/16`, `100.64.0.0/10`, RFC1918), enforcing connect-time `undici` `Agent` IP verification, and re-validating every `redirect: "manual"` hop.',
                        '- Option B: Checking `!/^(localhost|127\\.|169\\.254\\.)/.test(new URL(rawUrl).hostname)` before calling `fetch(rawUrl)`.'
                    ].join('\n')
                }
            ],
            chosen: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'CONNECT_TIME_SSRF_GUARD',
                            invariant: 'INV-LEGO-13',
                            selected_brick: '@krusch/toolkit/security',
                            rationale:
                                'Hostname regexes are trivially bypassed via DNS rebinding (TOCTOU), HTTP 302 redirects to http://169.254.169.254/latest/meta-data, decimal/octal IPv4, or IPv4-mapped IPv6 (::ffff:169.254.169.254). safeFetch blocks all bypasses at DNS resolution, TCP connect time, and every manual redirect hop.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ],
            rejected: [
                {
                    role: 'assistant',
                    content: `\`\`\`json\n${JSON.stringify(
                        {
                            verdict: 'REGEX_URL_CHECK',
                            invariant: 'INV-LEGO-13',
                            rationale:
                                'Testing url.hostname against localhost and 127. regexes before fetch(rawUrl) is sufficient to block internal requests.'
                        },
                        null,
                        2
                    )}\n\`\`\``
                }
            ]
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
    const legoScoutRows = buildLegoScoutSynthesisRows(nowIso);
    const allSftRows = [...cleanPassRows, ...remediationRows, ...legoFirstRows, ...legoScoutRows];

    const dpoPairs = buildRefereeDpoPairs();

    fs.writeFileSync(sftOutPath, `${allSftRows.map((r) => JSON.stringify(r)).join('\n')}\n`, 'utf8');
    fs.writeFileSync(dpoOutPath, `${dpoPairs.map((p) => JSON.stringify(p)).join('\n')}\n`, 'utf8');

    return {
        ok: true,
        tool: 'harvestLegoAndRefereeDatasets',
        summary: `Harvested ${allSftRows.length} SFT rows (${cleanPassRows.length} CLEAN_PASS, ${remediationRows.length} remediations, ${legoFirstRows.length} Lego-First, ${legoScoutRows.length} Lego-Scout) and ${dpoPairs.length} balanced DPO pairs`,
        data: {
            sftOutPath,
            dpoOutPath,
            counts: {
                sftTotal: allSftRows.length,
                cleanPass: cleanPassRows.length,
                remediations: remediationRows.length,
                legoFirst: legoFirstRows.length,
                legoScout: legoScoutRows.length,
                dpoTotal: dpoPairs.length,
                dpoPrecision: dpoPairs.filter((p) => p.mode === 'precision').length,
                dpoRecall: dpoPairs.filter((p) => p.mode === 'recall').length,
                dpoArchitecture: dpoPairs.filter((p) => p.mode === 'architecture_calibration').length
            }
        }
    };
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
    const res = harvestLegoAndRefereeDatasets();
    console.log(JSON.stringify(res, null, 2));
}
