#!/usr/bin/env node
/**
 * Harvests curated, zero-defect SFT & Reference-Anchored DPO contrast splits from
 * `NousResearch/arete-data-pairs` (Apache-2.0) across all 7 value/statutory categories:
 *   1. data-privacy-surveillance
 *   2. democracy-and-governance
 *   3. education-philosophy
 *   4. environmental-strategy
 *   5. freedom-of-expression
 *   6. state-ideology
 *   7. state-vs-free-market-capitalism
 *
 * Why strict curation matters:
 * - As documented in NousResearch/arete-data-pairs README ("Not a clean SFT set. Most accepted rows
 *   still carry a factual_error tag and accuracy 2"), raw ingestion of all 29,236 rows would inject
 *   accuracy-2 hallucinations and dilute Qwen3-Coder-30B-MoE's coding/tool-calling calibration.
 * - Instead, this harvester streams `paired/<topic>.jsonl` directly from Hugging Face and selects:
 *   (A) Gold SFT rows (`arete_values_sft.jsonl`): `in_sft_accepted === true`, `accuracy === 5`,
 *       `independence === 5`, `helpfulness >= 4`, `balance >= 4`, `language >= 4`, and empty `reasons`
 *       (top 20 per topic = 140 flawless statutory/analytical responses).
 *   (B) High-Contrast DPO pairs (`arete_values_dpo.jsonl`): Prompts where the starting Qwen checkpoint
 *       exhibited explicit refusal, deflection, or state-propaganda framing while the Inkling teacher
 *       achieved `accuracy >= 4`, `independence === 5`, `helpfulness >= 4` (top 8 per topic = 56 pairs).
 */

import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const JEAN_ROOT = path.resolve(__dirname, '..');
const DISTILL_DIR = path.join(JEAN_ROOT, 'data', 'distillation');

export const ARETE_TOPICS = Object.freeze([
  'data-privacy-surveillance',
  'democracy-and-governance',
  'education-philosophy',
  'environmental-strategy',
  'freedom-of-expression',
  'state-ideology',
  'state-vs-free-market-capitalism',
]);

const QWEN_DEFECT_REGEX =
  /cannot fulfill|cannot answer|i cannot|as an ai|unable to provide|sensitive topic|politically sensitive|harmonious society|cyber sovereignty|socialist core values|national reunification|internal affair|splittism|unlawful assembly|spread rumors|state subversion|hostile foreign forces|western anti-china|century of humiliation|one-china principle is the consensus/i;

function streamRemoteJsonl(url, onRow) {
  return new Promise((resolve, reject) => {
    const fetchUrl = (targetUrl) => {
      https
        .get(targetUrl, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            res.resume();
            const redirected = res.headers.location.startsWith('http')
              ? res.headers.location
              : new URL(res.headers.location, targetUrl).toString();
            fetchUrl(redirected);
            return;
          }
          if (res.statusCode !== 200) {
            res.resume();
            reject(new Error(`HTTP ${res.statusCode} fetching ${targetUrl}`));
            return;
          }
          const rl = readline.createInterface({ input: res, crlfDelay: Infinity });
          rl.on('line', (line) => {
            const trimmed = line.trim();
            if (!trimmed) return;
            try {
              onRow(JSON.parse(trimmed));
            } catch {
              // ignore malformed partial line if any
            }
          });
          rl.on('close', resolve);
          rl.on('error', reject);
        })
        .on('error', reject);
    };
    fetchUrl(url);
  });
}

export function scoreAreteSftCandidate(row) {
  if (!row || !row.in_sft_accepted || row.excluded) return -1;
  const scores = row.teacher_filter?.review_scores || {};
  const reasons = row.teacher_filter?.reasons || [];
  if (reasons.length > 0) return -1;
  const tResp = String(row.teacher?.response || '').trim();
  const prompt = String(row.prompt || '').trim();
  if (prompt.length < 40 || tResp.length < 250 || tResp.length > 5500) return -1;
  if ((scores.accuracy || 0) < 5 || (scores.independence || 0) < 5) return -1;
  if ((scores.helpfulness || 0) < 4 || (scores.balance || 0) < 4 || (scores.language || 0) < 4) return -1;
  return (
    scores.accuracy * 10 +
    scores.independence * 10 +
    scores.balance * 5 +
    scores.helpfulness * 4 +
    scores.language * 2 -
    Math.abs(tResp.length - 1800) / 1000
  );
}

export function scoreAreteDpoCandidate(row) {
  if (!row || !row.in_sft_accepted || row.excluded) return -1;
  const scores = row.teacher_filter?.review_scores || {};
  const reasons = row.teacher_filter?.reasons || [];
  if (reasons.length > 0) return -1;
  const qFlags = Array.isArray(row.starting_checkpoint_quality_flags)
    ? row.starting_checkpoint_quality_flags
    : [];
  if (qFlags.includes('short_answer')) return -1;

  const prompt = String(row.prompt || '').trim();
  const tResp = String(row.teacher?.response || '').trim();
  const qResp = String(row.starting_checkpoint?.response || '').trim();
  if (prompt.length < 40 || tResp.length < 250 || qResp.length < 200) return -1;
  if (tResp.length > 5000 || qResp.length > 5000) return -1;
  if ((scores.accuracy || 0) < 4 || (scores.independence || 0) < 5 || (scores.helpfulness || 0) < 4) {
    return -1;
  }

  const hasDefectMatch = QWEN_DEFECT_REGEX.test(qResp);
  const hasExplicitRefusal = /cannot fulfill|cannot answer|i cannot|as an ai|unable to provide|sensitive topic/i.test(
    qResp
  );
  if (!hasDefectMatch && qFlags.length === 0) return -1;

  return (
    scores.accuracy * 12 +
    scores.independence * 10 +
    (scores.balance || 0) * 4 +
    (hasExplicitRefusal ? 15 : 8) +
    (hasDefectMatch ? 10 : 0) -
    Math.abs(tResp.length - 1800) / 1000
  );
}

export async function harvestAreteAlignmentDatasets({
  sftPerTopic = 20,
  dpoPerTopic = 8,
  sftOutPath = path.join(DISTILL_DIR, 'arete_values_sft.jsonl'),
  dpoOutPath = path.join(DISTILL_DIR, 'arete_values_dpo.jsonl'),
} = {}) {
  fs.mkdirSync(path.dirname(sftOutPath), { recursive: true });

  const allSftRows = [];
  const allDpoRows = [];
  const topicStats = {};

  for (const topic of ARETE_TOPICS) {
    const url = `https://huggingface.co/datasets/NousResearch/arete-data-pairs/resolve/main/paired/${topic}.jsonl`;
    const sftPool = [];
    const dpoPool = [];
    let scanned = 0;

    await streamRemoteJsonl(url, (row) => {
      scanned++;
      const sftScore = scoreAreteSftCandidate(row);
      if (sftScore > 0) {
        sftPool.push({ score: sftScore, row });
      }
      const dpoScore = scoreAreteDpoCandidate(row);
      if (dpoScore > 0) {
        dpoPool.push({ score: dpoScore, row });
      }
    });

    sftPool.sort((a, b) => b.score - a.score || String(a.row.id).localeCompare(String(b.row.id)));
    dpoPool.sort((a, b) => b.score - a.score || String(a.row.id).localeCompare(String(b.row.id)));

    const chosenSft = sftPool.slice(0, sftPerTopic);
    const chosenSftIds = new Set(chosenSft.map((x) => x.row.id));

    // Prefer DPO prompts disjoint from SFT prompts where possible for broader prompt coverage
    const disjointDpo = dpoPool.filter((x) => !chosenSftIds.has(x.row.id));
    const chosenDpo = (disjointDpo.length >= dpoPerTopic ? disjointDpo : dpoPool).slice(0, dpoPerTopic);

    for (const { row } of chosenSft) {
      allSftRows.push({
        id: `arete_sft_${row.id}`,
        category: `arete_${topic}`,
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: String(row.prompt).trim() },
          { role: 'assistant', content: String(row.teacher.response).trim() },
        ],
        metadata: {
          source: 'NousResearch/arete-data-pairs',
          license: 'Apache-2.0',
          topic,
          subtopic: row.subtopic || '',
          lens: row.lens || '',
          review_scores: row.teacher_filter?.review_scores || {},
        },
      });
    }

    for (const { row } of chosenDpo) {
      allDpoRows.push({
        id: `arete_dpo_${row.id}`,
        category: `arete_${topic}_dpo`,
        prompt: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: String(row.prompt).trim() },
        ],
        chosen: [{ role: 'assistant', content: String(row.teacher.response).trim() }],
        rejected: [{ role: 'assistant', content: String(row.starting_checkpoint.response).trim() }],
        metadata: {
          source: 'NousResearch/arete-data-pairs',
          license: 'Apache-2.0',
          topic,
          subtopic: row.subtopic || '',
          lens: row.lens || '',
          review_scores: row.teacher_filter?.review_scores || {},
        },
      });
    }

    topicStats[topic] = {
      scanned,
      sft_qualified: sftPool.length,
      sft_selected: chosenSft.length,
      dpo_qualified: dpoPool.length,
      dpo_selected: chosenDpo.length,
    };
    console.log(
      `[arete-harvest] ${topic.padEnd(32)} | scanned=${scanned} | sft=${chosenSft.length}/${sftPool.length} | dpo=${chosenDpo.length}/${dpoPool.length}`
    );
  }

  fs.writeFileSync(sftOutPath, allSftRows.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');
  fs.writeFileSync(dpoOutPath, allDpoRows.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');

  console.log(
    `[arete-harvest] Wrote ${allSftRows.length} gold SFT rows -> ${sftOutPath} and ${allDpoRows.length} high-contrast DPO pairs -> ${dpoOutPath}`
  );
  return {
    sftCount: allSftRows.length,
    dpoCount: allDpoRows.length,
    sftOutPath,
    dpoOutPath,
    topicStats,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  harvestAreteAlignmentDatasets().catch((err) => {
    console.error('[arete-harvest] Fatal error:', err);
    process.exit(1);
  });
}
