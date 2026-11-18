#!/usr/bin/env node
/**
 * @file harvest_antigravity_traces.js
 * @description Zero-Leak Antigravity Trajectory Harvester for Wave-7+ SFT Distillation.
 *
 * Scans Antigravity parent and subagent `transcript_full.jsonl` logs under
 * `~/.gemini/antigravity/brain/<id>/.system_generated/logs/`, filters for
 * verified Principal-Engineer / CSO / Referee audits and test-verified code
 * remediations, converts parallel tool batches into strict 1-call-per-turn
 * trajectories (compatible with Qwen3 `<tool_call>`, Gemma 4 `<|channel>thought`,
 * and Harmony `<|call|>`), and enforces the Memory #594 Zero-Leak Gate.
 *
 * Accumulates deduplicated records by `id` in `data/distillation/antigravity_traces_sft.jsonl`
 * so harvested training traces persist permanently even if local IDE logs rotate.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizePortableSftText } from '../src/harness.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

export const DEFAULT_BRAIN_DIR = path.join(os.homedir(), '.gemini', 'antigravity', 'brain');
export const ANTIGRAVITY_TRACES_SFT_PATH = path.join(
  projectRoot,
  'data',
  'distillation',
  'antigravity_traces_sft.jsonl'
);
export const ANTIGRAVITY_TRACES_LOCAL_ONLY_PATH = path.join(
  projectRoot,
  'data',
  'distillation',
  'antigravity_traces_sft_local.jsonl'
);

export const PUBLIC_OSS_PROJECTS = new Set([
  'krusch-stack',
  'toolkit',
  'krusch-context-mcp',
  'krusch-git',
  'krusch-fs',
  'krusch-fs-mcp',
  'krusch-coding-harness',
  'grounded-harness',
  'krusch-grounded-harness',
  'hydrascope',
  'deadbolt',
  'krusch-cascade-router',
  'krusch-frontend-mcp',
  'g-krusch',
  'jean-sre',
  'easi.run'
]);

const PRIVATE_REPO_REDACT_RE =
  /\b(?:acme-legacy-billing|legacy-billing-monolith|oscar-tree-service|caren|pocket-lawyer|home-ai|first-things-first|berean|spark|krusch-law|krusch-bizlaw|krusch-biz|krusch-companion-skylight-connector|krusch-companion|krusch-nexus|krusch-pre-router|card-game-context|orbit-wars|hermes-parallax|Brain3|dbos-worker)\b/gi;

const INTERNAL_ORCHESTRATION_TOOLS = new Set([
  'send_message',
  'invoke_subagent',
  'define_subagent',
  'manage_subagents',
  'manage_task',
  'schedule',
  'ask_question',
  'search_marketplace',
  'list_plugin_accounts'
]);

const CANONICAL_AGY_SYSTEM_PROMPT = [
  'You are an Autonomous Principal Software Engineer, Security Auditor, and 3-Tier MCP Coding Agent.',
  'OPERATING RULES:',
  '1. Ground every architectural claim, security finding, and edit in live source windows, AST symbol graphs, and test outputs.',
  '2. Enforce path containment, symlink safety, constant-time auth checks, linear-time regex/DP bounds, and SQLite/PostgreSQL transactional invariants.',
  '3. Run syntax checks and unit/regression test suites autonomously before reporting completion, and emit calibrated CLEAN_PASS verdicts when zero P0/P1/P2 defects remain.'
].join('\n');

const MAX_TOOL_STEPS_PER_TRACE = 12;
const MAX_THINKING_CHARS = 1200;
const MAX_TOOL_OUTPUT_CHARS = 2200;
const MAX_FINAL_REPORT_CHARS = 9000;

/**
 * Scrub all local OS paths, Antigravity appData URIs, private RFC1918/CGNAT IPs,
 * internal fleet hostnames, and bearer/API tokens for Zero-Leak public portability.
 *
 * @param {string} rawText
 * @param {string} [project='workspace']
 * @returns {string}
 */
export function scrubAntigravityTraceText(rawText, project = 'workspace') {
  if (!rawText || typeof rawText !== 'string') return '';
  let out = rawText
    // Scrub Antigravity brain & appData paths before generic home-path replacement
    .replace(/file:\/\/\/(?:home|Users)\/[A-Za-z0-9._-]+\/\.gemini\/antigravity\/[^\s)`"']+/g, 'file://$AGY_DATA_DIR/trace')
    .replace(/\/(?:home|Users)\/[A-Za-z0-9._-]+\/\.gemini\/antigravity\/[^\s)`"',]+/g, '$AGY_DATA_DIR/trace')
    .replace(/\/(?:home|Users)\/[A-Za-z0-9._-]+\/\.gemini\/[^\s)`"',]+/g, '$AGY_DATA_DIR/config')
    .replace(/file:\/\/\/(?:home|Users)\/[A-Za-z0-9._-]+\/homelab\//g, 'file://$WORKSPACE_ROOT/');

  out = normalizePortableSftText(out, project);

  out = out
    // Catch any remaining /home/<user> or /Users/<user> paths
    .replace(/\/(?:home|Users)\/[A-Za-z0-9._-]+/g, '$WORKSPACE_ROOT')
    // Catch full Tailscale CGNAT 100.64.0.0/10 & RFC1918 172.16.0.0/12 + 10.x + 192.168.x
    .replace(/\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b/g, '127.0.0.1')
    .replace(/\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b/g, '127.0.0.1')
    .replace(/\b(?:192\.168|10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b/g, '127.0.0.1')
    // Catch additional fleet hostnames, private repos & private suffixes
    .replace(/\b(?:kruschdev|kruschserv|kruschmedia|kruschgame|kruschsandbox|kr1yoga|krmac|crush0|robin)\b/gi, 'compute-node')
    .replace(PRIVATE_REPO_REDACT_RE, 'internal-app')
    .replace(/\.(?:lan|ts\.net)\b/gi, '.example')
    // Redact Bearer tokens & API keys
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]{12,}/gi, 'Bearer REDACTED_TOKEN')
    .replace(/(?:AIzaSy[A-Za-z0-9_-]{33}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,})/g, 'REDACTED_API_KEY')
    // Decode literal JSON-escaped surrogate pairs (e.g. \ud83d\udccd or \\ud83d\\udccd) and neutralize literal lone \ud8xx.. escapes
    .replace(/\\+u(d[89ab][0-9a-f]{2})\\+u(d[c-f][0-9a-f]{2})/gi, (_m, hi, lo) =>
      String.fromCharCode(parseInt(hi, 16), parseInt(lo, 16))
    )
    .replace(/\\+ud[89a-f][0-9a-f]{2}/gi, 'uFFFD')
    // Remove lone UTF-16 surrogates
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '');

  return typeof out.toWellFormed === 'function' ? out.toWellFormed() : out;
}

function scrubArgsObject(rawArgs, project) {
  if (!rawArgs || typeof rawArgs !== 'object') return {};
  try {
    const scrubbedJson = scrubAntigravityTraceText(JSON.stringify(rawArgs), project);
    const parsed = JSON.parse(scrubbedJson);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    const safe = {};
    for (const [k, v] of Object.entries(rawArgs)) {
      safe[k] = typeof v === 'string' ? scrubAntigravityTraceText(v, project) : v;
    }
    return safe;
  }
}

function boundText(text, maxChars) {
  const s = String(text || '').trim();
  if (s.length <= maxChars) {
    return typeof s.toWellFormed === 'function' ? s.toWellFormed() : s;
  }
  const half = Math.floor((maxChars - 32) / 2);
  const sliced = `${s.slice(0, half)}\n...[truncated]...\n${s.slice(s.length - half)}`;
  return typeof sliced.toWellFormed === 'function' ? sliced.toWellFormed() : sliced;
}

function inferProjectName(rawPrompt, stepsText) {
  const combined = `${rawPrompt}\n${stepsText.slice(0, 8000)}`;
  const match = combined.match(/\/projects\/(?:archive\/)?([a-zA-Z0-9_-]+)/);
  if (match && match[1]) return match[1];
  if (/krusch-stack/i.test(combined)) return 'krusch-stack';
  if (/hydrascope/i.test(combined)) return 'hydrascope';
  if (/toolkit/i.test(combined)) return 'toolkit';
  return 'workspace';
}

function extractInitialTaskPrompt(steps) {
  for (const step of steps) {
    if (step.type === 'USER_INPUT' && typeof step.content === 'string' && step.content.trim()) {
      const userBlock = step.content.match(/<USER_REQUEST>([\s\S]*?)<\/USER_REQUEST>/);
      const extracted = (userBlock ? userBlock[1] : step.content).trim();
      if (extracted.length >= 12) return extracted;
    }
    if (step.type === 'SYSTEM_MESSAGE' && typeof step.content === 'string') {
      const contentIdx = step.content.indexOf('content=');
      if (contentIdx !== -1) {
        const body = step.content
          .slice(contentIdx + 'content='.length)
          .replace(/<\/SYSTEM_MESSAGE>\s*$/, '')
          .trim();
        if (body.length >= 40) return body;
      }
    }
  }
  return '';
}

function extractFinalReport(steps) {
  // Check subagent send_message payloads from the end first (richest structured audit/remediation reports)
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const st = steps[i];
    if (st.type === 'PLANNER_RESPONSE' && Array.isArray(st.tool_calls)) {
      for (const tc of st.tool_calls) {
        if (
          tc &&
          tc.name === 'send_message' &&
          tc.args &&
          typeof tc.args.Message === 'string' &&
          tc.args.Message.trim().length >= 220
        ) {
          return { report: tc.args.Message.trim(), stepIndex: i, createdAt: st.created_at };
        }
      }
    }
  }
  // Fallback to final substantive PLANNER_RESPONSE content
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const st = steps[i];
    if (
      st.type === 'PLANNER_RESPONSE' &&
      typeof st.content === 'string' &&
      st.content.trim().length >= 220
    ) {
      return { report: st.content.trim(), stepIndex: i, createdAt: st.created_at };
    }
  }
  return null;
}

function classifyHighSignalPhase(finalReport, hadEdits, hadPassingTests) {
  const isCleanPassAudit = /\bCLEAN[\s_-]?PASS\b/i.test(finalReport) && /\bP0\b|\bfindings\b|\bVerdict\b/i.test(finalReport);
  const isStructuredAudit =
    (/\b(?:SEC|DB|MCP|OPS|API|DOC|BUG|ARCH|P[012])-\d{2}\b/.test(finalReport) ||
      /\bCONDITIONAL FAIL\b/i.test(finalReport)) &&
    /[a-zA-Z0-9_./-]+\.(?:js|ts|py|sql|yml|md):\d+/.test(finalReport);

  if (hadEdits && hadPassingTests) {
    return { eligible: true, phase: 'ARGON_MULTI_TURN_REMEDIATION', score: 96 };
  }
  if (isCleanPassAudit) {
    return { eligible: true, phase: 'CALIBRATED_CLEAN_PASS_AUDIT', score: 95 };
  }
  if (isStructuredAudit) {
    return { eligible: true, phase: 'ADVERSARIAL_CODE_AUDIT', score: 92 };
  }
  return { eligible: false, phase: 'SKIPPED', score: 0 };
}

/**
 * Parse a single `transcript_full.jsonl` file into a normalized Zero-Leak SFT trajectory record
 * (or return `null` if the transcript is incomplete or low-signal).
 *
 * @param {string} transcriptPath
 * @param {string} [conversationId]
 * @returns {object | null}
 */
export function parseAntigravityTranscriptToSftRow(transcriptPath, conversationId = '') {
  if (!fs.existsSync(transcriptPath)) return null;
  const rawLines = fs.readFileSync(transcriptPath, 'utf8').split('\n').filter(Boolean);
  if (rawLines.length < 3) return null;

  const steps = [];
  for (const line of rawLines) {
    try {
      steps.push(JSON.parse(line));
    } catch {
      // skip truncated/malformed line
    }
  }
  if (steps.length < 3) return null;

  const rawPrompt = extractInitialTaskPrompt(steps);
  const finalInfo = extractFinalReport(steps);
  if (!rawPrompt || !finalInfo) return null;

  const project = inferProjectName(rawPrompt, finalInfo.report);
  const toolDeclarations = new Map();
  const toolPairs = [];
  let hadEdits = false;
  let hadPassingTests = false;

  for (let i = 0; i <= finalInfo.stepIndex; i += 1) {
    const step = steps[i];
    if (step.type !== 'PLANNER_RESPONSE' || !Array.isArray(step.tool_calls) || step.tool_calls.length === 0) {
      continue;
    }

    // Collect subsequent GENERIC tool output steps paired positionally with step.tool_calls
    const outputs = [];
    let j = i + 1;
    while (j < steps.length && steps[j].type === 'GENERIC' && outputs.length < step.tool_calls.length) {
      outputs.push(typeof steps[j].content === 'string' ? steps[j].content : JSON.stringify(steps[j].content ?? ''));
      j += 1;
    }

    for (let k = 0; k < step.tool_calls.length; k += 1) {
      const tc = step.tool_calls[k];
      if (!tc || typeof tc.name !== 'string') continue;
      if (INTERNAL_ORCHESTRATION_TOOLS.has(tc.name)) continue;

      const outText = outputs[k] || 'OK';
      if (tc.name === 'edit_file' || tc.name === 'write_file') {
        const editedPath = String(tc.args?.Path || tc.args?.TargetFile || '');
        if (/\.(?:js|mjs|cjs|ts|tsx|jsx|py|rs|go|sql|yml|yaml)$|Dockerfile$/i.test(editedPath)) {
          hadEdits = true;
        }
      }
      if (
        tc.name === 'run_command' &&
        (/Command exited with code 0/.test(outText) || /\bfail 0\b/.test(outText))
      ) {
        hadPassingTests = true;
      }

      if (toolPairs.length < MAX_TOOL_STEPS_PER_TRACE) {
        const cleanArgs = scrubArgsObject(tc.args || {}, project);
        const cleanThinking =
          k === 0 && typeof step.thinking === 'string' && step.thinking.trim()
            ? boundText(scrubAntigravityTraceText(step.thinking, project), MAX_THINKING_CHARS)
            : '';
        const cleanOut = boundText(scrubAntigravityTraceText(outText, project), MAX_TOOL_OUTPUT_CHARS);

        toolPairs.push({
          name: tc.name,
          args: cleanArgs,
          thinking: cleanThinking,
          output: cleanOut
        });

        if (!toolDeclarations.has(tc.name)) {
          const propSchema = {};
          for (const [argKey, argVal] of Object.entries(cleanArgs)) {
            propSchema[argKey] = {
              type: Array.isArray(argVal) ? 'array' : typeof argVal === 'object' && argVal !== null ? 'object' : typeof argVal
            };
          }
          toolDeclarations.set(tc.name, {
            type: 'function',
            function: {
              name: tc.name,
              description: `Execute ${tc.name} in workspace`,
              parameters: { type: 'object', properties: propSchema }
            }
          });
        }
      }
    }
  }

  if (toolPairs.length === 0) return null;

  const gate = classifyHighSignalPhase(finalInfo.report, hadEdits, hadPassingTests);
  if (!gate.eligible) return null;

  const cleanUserPrompt = boundText(scrubAntigravityTraceText(rawPrompt, project), 4500);
  const cleanFinalReport = boundText(
    scrubAntigravityTraceText(finalInfo.report, project),
    MAX_FINAL_REPORT_CHARS
  );

  // Skip fleet-hardware-only infra prompts so public release stays 100% portable
  if (/Xeon 40T|RTX 2080 Ti|Fleet Hardware/i.test(cleanUserPrompt + cleanFinalReport)) {
    return null;
  }

  const messages = [
    { role: 'system', content: CANONICAL_AGY_SYSTEM_PROMPT },
    { role: 'user', content: cleanUserPrompt }
  ];

  toolPairs.forEach((pair, idx) => {
    const callId = `call_${idx + 1}`;
    const assistantTurn = {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: callId,
          type: 'function',
          function: {
            name: pair.name,
            arguments: pair.args
          }
        }
      ]
    };
    if (pair.thinking) {
      assistantTurn.thinking = pair.thinking;
    }
    messages.push(assistantTurn);
    messages.push({
      role: 'tool',
      tool_call_id: callId,
      name: pair.name,
      content: pair.output
    });
  });

  messages.push({
    role: 'assistant',
    content: cleanFinalReport
  });

  const convSlug = (conversationId || path.basename(path.dirname(path.dirname(path.dirname(transcriptPath)))) || 'trace')
    .replace(/[^a-zA-Z0-9_-]/g, '')
    .slice(0, 16);

  return {
    id: `agy-${project}-${gate.phase.toLowerCase()}-${convSlug}`,
    timestamp: finalInfo.createdAt || new Date().toISOString(),
    source: 'antigravity_verified_trajectory',
    teacher_model: 'gemini-4-argon',
    student_target: 'qwen3-coder-30b-moe',
    project,
    phase: gate.phase,
    verification_status: 'VERIFIED',
    elitist_score: gate.score,
    tools: [...toolDeclarations.values()],
    messages
  };
}

function loadExistingRowsById(sftOutPath) {
  const byId = new Map();
  if (!fs.existsSync(sftOutPath)) return byId;
  const lines = fs.readFileSync(sftOutPath, 'utf8').split('\n').filter(Boolean);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed && typeof parsed.id === 'string') {
        byId.set(parsed.id, parsed);
      }
    } catch {
      // skip malformed line
    }
  }
  return byId;
}

/**
 * Harvest all eligible Antigravity parent + subagent transcripts into a cumulative Zero-Leak JSONL split.
 *
 * @param {{ brainDir?: string, sftOutPath?: string }} [options={}]
 */
export function harvestAntigravityTraces({
  brainDir = DEFAULT_BRAIN_DIR,
  sftOutPath = ANTIGRAVITY_TRACES_SFT_PATH,
  localOutPath = sftOutPath === ANTIGRAVITY_TRACES_SFT_PATH ? ANTIGRAVITY_TRACES_LOCAL_ONLY_PATH : null,
  requirePublicOss = true
} = {}) {
  fs.mkdirSync(path.dirname(sftOutPath), { recursive: true });
  const rowsById = loadExistingRowsById(sftOutPath);
  const localRowsById = localOutPath ? loadExistingRowsById(localOutPath) : new Map();
  let scannedConversations = 0;
  let newlyHarvested = 0;

  if (fs.existsSync(brainDir)) {
    const entries = fs.readdirSync(brainDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const transcriptPath = path.join(
        brainDir,
        entry.name,
        '.system_generated',
        'logs',
        'transcript_full.jsonl'
      );
      if (!fs.existsSync(transcriptPath)) continue;
      scannedConversations += 1;
      const row = parseAntigravityTranscriptToSftRow(transcriptPath, entry.name);
      if (row) {
        if (localOutPath) {
          localRowsById.set(row.id, row);
        }
        if (!requirePublicOss || PUBLIC_OSS_PROJECTS.has(row.project)) {
          if (!rowsById.has(row.id)) newlyHarvested += 1;
          rowsById.set(row.id, row);
        }
      }
    }
  }

  const allRows = [...rowsById.values()].filter(
    (r) => !requirePublicOss || PUBLIC_OSS_PROJECTS.has(r.project)
  );
  const payload = allRows.length > 0 ? `${allRows.map((r) => JSON.stringify(r)).join('\n')}\n` : '';
  fs.writeFileSync(sftOutPath, payload, 'utf8');

  if (localOutPath && localRowsById.size > 0) {
    fs.mkdirSync(path.dirname(localOutPath), { recursive: true });
    const localPayload = `${[...localRowsById.values()].map((r) => JSON.stringify(r)).join('\n')}\n`;
    fs.writeFileSync(localOutPath, localPayload, 'utf8');
  }

  return {
    scannedConversations,
    newlyHarvested,
    totalRows: allRows.length,
    localTotalRows: localRowsById.size,
    outputPath: sftOutPath
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  const res = harvestAntigravityTraces();
  console.log(JSON.stringify(res, null, 2));
}
