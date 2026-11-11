import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
    GGUF_MAGIC,
    parseParamCountBillions,
    parseGgufHeaderBuffer,
    inspectGgufFileHeader,
    evaluateArtifactRiskProfile,
    computeTokenRepetitionMetrics,
    executeSandboxedClampCanary,
    runModelCanarySuite,
    auditOllamaEndpointQuantization,
    auditFleetQuantization
} from '../src/quantcheck.js';
import { createQuantcheckCli } from '../scripts/quantcheck.js';

function ggufStr(str) {
    const b = Buffer.from(str, 'utf8');
    const len = Buffer.alloc(8);
    len.writeBigUInt64LE(BigInt(b.length), 0);
    return Buffer.concat([len, b]);
}

function kvUint32(key, val) {
    const v = Buffer.alloc(8);
    v.writeUInt32LE(4, 0); // GGUF_TYPE_UINT32
    v.writeUInt32LE(val >>> 0, 4);
    return Buffer.concat([ggufStr(key), v]);
}

function kvString(key, val) {
    const t = Buffer.alloc(4);
    t.writeUInt32LE(8, 0); // GGUF_TYPE_STRING
    return Buffer.concat([ggufStr(key), t, ggufStr(val)]);
}

function kvUint32Array(key, values) {
    const hdr = Buffer.alloc(16);
    hdr.writeUInt32LE(9, 0); // GGUF_TYPE_ARRAY
    hdr.writeUInt32LE(4, 4); // elemType UINT32
    hdr.writeBigUInt64LE(BigInt(values.length), 8);
    const body = Buffer.alloc(values.length * 4);
    values.forEach((n, idx) => body.writeUInt32LE(n >>> 0, idx * 4));
    return Buffer.concat([ggufStr(key), hdr, body]);
}

function tensorDesc(name, dims, ggmlType, relOffset) {
    const meta = Buffer.alloc(4 + dims.length * 8 + 4 + 8);
    let p = 0;
    meta.writeUInt32LE(dims.length, p);
    p += 4;
    for (const d of dims) {
        meta.writeBigUInt64LE(BigInt(d), p);
        p += 8;
    }
    meta.writeUInt32LE(ggmlType, p);
    p += 4;
    meta.writeBigUInt64LE(BigInt(relOffset), p);
    return Buffer.concat([ggufStr(name), meta]);
}

function buildSyntheticGgufBuffer({
    version = 3,
    fileType = 15, // Q4_K_M
    sizeLabel = '30.5B',
    includeImatrix = true,
    includeTemplate = true,
    firstTensorOffset = 0,
    truncateBytes = 0
} = {}) {
    const kvs = [
        kvString('general.architecture', 'qwen3moe'),
        kvString('general.name', 'Qwen3-Coder-30B-A3B'),
        kvUint32('general.file_type', fileType),
        kvString('general.size_label', sizeLabel),
        kvUint32('tokenizer.ggml.eos_token_id', 151645),
        kvUint32Array('tokenizer.ggml.scores_sample', [1, 2, 3, 4])
    ];
    if (includeImatrix) {
        kvs.push(kvString('quantize.imatrix.file', 'calibration_imatrix.dat'));
        kvs.push(kvUint32('quantize.imatrix.entries_count', 240));
    }
    if (includeTemplate) {
        kvs.push(kvString('tokenizer.chat_template', '{% for message in messages %}{{ message.content }}{% endfor %}'));
    }

    // Tensor 0: Q4_K (type 12, blockElems=256, blockBytes=144), dims [256, 4] => 1024 elems => 4 * 144 = 576B (divisible by 32!)
    // Tensor 1: F32 (type 0, blockElems=1, blockBytes=4), dims [256] => 1024B at relOffset 576
    const tensors = [
        tensorDesc('blk.0.attn_q.weight', [256, 4], 12, firstTensorOffset),
        tensorDesc('output_norm.weight', [256], 0, 576)
    ];

    const hdr = Buffer.alloc(24);
    hdr.writeUInt32LE(GGUF_MAGIC, 0);
    hdr.writeUInt32LE(version, 4);
    hdr.writeBigUInt64LE(BigInt(tensors.length), 8);
    hdr.writeBigUInt64LE(BigInt(kvs.length), 16);

    const preData = Buffer.concat([hdr, ...kvs, ...tensors]);
    const rem = preData.length % 32;
    const pad = rem === 0 ? Buffer.alloc(0) : Buffer.alloc(32 - rem);
    const dataPayload = Buffer.alloc(576 + 1024, 0xaa);
    const full = Buffer.concat([preData, pad, dataPayload]);
    return truncateBytes > 0 ? full.subarray(0, full.length - truncateBytes) : full;
}

describe('quantcheck GGUF & Broken-on-Arrival Sanity Gate (INV-JEAN-QC-01..04)', () => {
    it('parses valid GGUF v3 binary headers, imatrix provenance, and tensor alignment bounds', () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'quantcheck-gguf-'));
        const ggufPath = path.join(tmpDir, 'valid-wave7.gguf');
        try {
            const buf = buildSyntheticGgufBuffer();
            fs.writeFileSync(ggufPath, buf);

            const res = inspectGgufFileHeader(ggufPath);
            assert.equal(res.ok, true);
            assert.equal(res.tool, 'jean.quantcheck');
            assert.equal(res.data.version, 3);
            assert.equal(res.data.architecture, 'qwen3moe');
            assert.equal(res.data.quantizationLabel, 'Q4_K_M');
            assert.equal(res.data.hasImatrix, true);
            assert.equal(res.data.hasChatTemplate, true);
            assert.equal(res.data.tensorCount, 2);
            assert.deepEqual(res.data.tensorTypeCounts, { Q4_K: 1, F32: 1 });
            assert.equal(res.data.warnings.length, 0);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it('rejects truncated GGUF payloads and misaligned tensor offsets', () => {
        const truncated = buildSyntheticGgufBuffer({ truncateBytes: 64 });
        const parsedTrunc = parseGgufHeaderBuffer(truncated, { fileSizeBytes: truncated.length });
        assert.equal(parsedTrunc.valid, false);
        assert.ok(parsedTrunc.errors.some((e) => e.includes('Truncated GGUF file')));

        const misaligned = buildSyntheticGgufBuffer({ firstTensorOffset: 12 });
        const parsedMis = parseGgufHeaderBuffer(misaligned, { fileSizeBytes: misaligned.length });
        assert.equal(parsedMis.valid, false);
        assert.ok(parsedMis.errors.some((e) => e.includes('violates 32-byte alignment')));
    });

    it('flags uncalibrated sub-4.5B extreme low-bit GGUFs per arXiv:2609.05881', () => {
        const riskyBuf = buildSyntheticGgufBuffer({
            fileType: 10, // Q2_K
            sizeLabel: '3.0B',
            includeImatrix: false
        });
        const parsed = parseGgufHeaderBuffer(riskyBuf, { fileSizeBytes: riskyBuf.length });
        assert.equal(parsed.valid, true);
        assert.equal(parsed.quantizationLabel, 'Q2_K');
        assert.equal(parsed.hasImatrix, false);
        assert.ok(parsed.warnings.some((w) => w.includes('arXiv:2609.05881')));
    });

    it('evaluates static risk profiles and sandboxed clampRange + repetition guards', () => {
        assert.equal(parseParamCountBillions('566.70M'), 0.5667);
        assert.equal(parseParamCountBillions('30.5B'), 30.5);

        const highRisk = evaluateArtifactRiskProfile({
            model: 'tiny-coder:3b-q2_K',
            quantizationLevel: 'Q2_K',
            parameterSize: '3.2B',
            capabilities: ['completion'],
            hasImatrix: false
        });
        assert.equal(highRisk.riskTier, 'HIGH_RISK_LOW_BIT_SMALL_MODEL');

        const spillRisk = evaluateArtifactRiskProfile({
            model: 'qwen3-coder-30b-moe:latest',
            quantizationLevel: 'Q4_K_M',
            parameterSize: '30.5B',
            capabilities: ['completion', 'tools'],
            isResident: true,
            sizeBytes: 18_000_000_000,
            sizeVramBytes: 12_000_000_000
        });
        assert.equal(spillRisk.riskTier, 'UNPINNED_CPU_SPILL');

        assert.equal(
            computeTokenRepetitionMetrics('the the the the the the the the the the the').collapsed,
            true
        );
        assert.equal(
            computeTokenRepetitionMetrics('00000000000000000000000000000000000000').collapsed,
            true
        );

        const goodExec = executeSandboxedClampCanary(
            '<think>clamping</think>\n```js\nfunction clampRange(val, lo, hi) { return val < lo ? lo : (val > hi ? hi : val); }\n```'
        );
        assert.equal(goodExec.ok, true);
        assert.equal(goodExec.passedVectors, 5);

        const badLogic = executeSandboxedClampCanary(
            'function clampRange(val, lo, hi) { return Math.max(hi, Math.min(lo, val)); }'
        );
        assert.equal(badLogic.ok, false);

        const forbidden = executeSandboxedClampCanary(
            'function clampRange(val, lo, hi) { process.exit(1); return val; }'
        );
        assert.equal(forbidden.ok, false);
        assert.match(forbidden.error, /Forbidden identifier/);
    });

    it('detects Broken-on-Arrival generative and collapsed embedding models in runModelCanarySuite', async () => {
        // 1. Broken generative model (repetition collapse + wrong JSON sum)
        const brokenGenFetch = async (url, init) => {
            const body = JSON.parse(init.body);
            if (body.format === 'json') {
                return new Response(JSON.stringify({ message: { content: '{"sum": 51, "valid": false}' } }), {
                    status: 200
                });
            }
            return new Response(
                JSON.stringify({
                    message: { content: 'function clampRange clampRange clampRange clampRange clampRange clampRange clampRange clampRange clampRange clampRange clampRange' },
                    eval_count: 20,
                    eval_duration: 200_000_000
                }),
                { status: 200 }
            );
        };
        const brokenGen = await runModelCanarySuite({
            baseUrl: 'http://127.0.0.1:11434',
            model: 'broken-quant:3b-q2_K',
            modality: 'generative',
            supportsTools: false,
            fetchImpl: brokenGenFetch
        });
        assert.equal(brokenGen.verdict, 'BROKEN_ON_ARRIVAL');

        // 2. Collapsed isotropic embedding model (zero semantic separation)
        const collapsedEmbedFetch = async () =>
            new Response(
                JSON.stringify({
                    embeddings: [
                        new Array(64).fill(0.5),
                        new Array(64).fill(0.5),
                        new Array(64).fill(0.5)
                    ]
                }),
                { status: 200 }
            );
        const brokenEmbed = await runModelCanarySuite({
            baseUrl: 'http://127.0.0.1:11434',
            model: 'snowflake-arctic-embed2:broken',
            modality: 'embedding',
            fetchImpl: collapsedEmbedFetch
        });
        assert.equal(brokenEmbed.verdict, 'BROKEN_ON_ARRIVAL');

        // 3. Healthy embedding model
        const healthyEmbedFetch = async () => {
            const v0 = new Array(64).fill(0).map((_, i) => (i < 32 ? 1 : 0));
            const v1 = new Array(64).fill(0).map((_, i) => (i < 30 ? 1 : 0.1));
            const v2 = new Array(64).fill(0).map((_, i) => (i >= 32 ? 1 : 0));
            return new Response(JSON.stringify({ embeddings: [v0, v1, v2] }), { status: 200 });
        };
        const passEmbed = await runModelCanarySuite({
            baseUrl: 'http://127.0.0.1:11434',
            model: 'snowflake-arctic-embed2:latest',
            modality: 'embedding',
            fetchImpl: healthyEmbedFetch
        });
        assert.equal(passEmbed.verdict, 'PASS');
    });

    it('enforces INV-JEAN-QC-02 zero-VRAM-thrash by probing only /api/ps resident models by default', async () => {
        const chatModelsCalled = [];
        const mockFetch = async (url, init = {}) => {
            if (url.endsWith('/api/tags')) {
                return new Response(
                    JSON.stringify({
                        models: [
                            {
                                name: 'qwen3-coder-30b-moe:latest',
                                size: 18_500_000_000,
                                details: { parameter_size: '30.5B', quantization_level: 'Q4_K_M', family: 'qwen3moe' }
                            },
                            {
                                name: 'unloaded-experimental:3b-q2_K',
                                size: 1_400_000_000,
                                details: { parameter_size: '3.0B', quantization_level: 'Q2_K', family: 'qwen2' }
                            }
                        ]
                    }),
                    { status: 200 }
                );
            }
            if (url.endsWith('/api/ps')) {
                return new Response(
                    JSON.stringify({
                        models: [
                            {
                                name: 'qwen3-coder-30b-moe:latest',
                                size: 18_500_000_000,
                                size_vram: 18_500_000_000,
                                details: { parameter_size: '30.5B', quantization_level: 'Q4_K_M' }
                            }
                        ]
                    }),
                    { status: 200 }
                );
            }
            if (url.endsWith('/api/show')) {
                const { model } = JSON.parse(init.body);
                return new Response(
                    JSON.stringify({
                        capabilities: model.startsWith('qwen3') ? ['completion', 'tools'] : ['completion'],
                        template: '{{ .Prompt }}'
                    }),
                    { status: 200 }
                );
            }
            if (url.endsWith('/api/chat')) {
                const body = JSON.parse(init.body);
                chatModelsCalled.push({ model: body.model, keepAlive: body.keep_alive });
                if (body.tools) {
                    return new Response(
                        JSON.stringify({
                            message: {
                                content: '',
                                tool_calls: [
                                    { function: { name: 'check_disk_space', arguments: { mount: '/var/lib/app' } } }
                                ]
                            }
                        }),
                        { status: 200 }
                    );
                }
                if (body.format === 'json') {
                    return new Response(
                        JSON.stringify({ message: { content: '{"sum": 49, "valid": true}' } }),
                        { status: 200 }
                    );
                }
                return new Response(
                    JSON.stringify({
                        message: {
                            content:
                                'function clampRange(val, lo, hi) {\n  if (val < lo) return lo;\n  if (val > hi) return hi;\n  return val;\n}'
                        },
                        eval_count: 42,
                        eval_duration: 330_000_000
                    }),
                    { status: 200 }
                );
            }
            throw new Error(`Unexpected URL ${url}`);
        };

        const fleetRes = await auditFleetQuantization(['http://127.0.0.1:11434'], { fetchImpl: mockFetch });
        assert.equal(fleetRes.ok, true);
        assert.equal(fleetRes.tool, 'jean.quantcheck');
        assert.equal(fleetRes.data.totalTags, 2);
        assert.equal(fleetRes.data.totalLiveProbed, 1);

        // Unloaded tag MUST NEVER trigger /api/chat (0 GPU eviction!)
        assert.equal(chatModelsCalled.every((c) => c.model === 'qwen3-coder-30b-moe:latest'), true);
        assert.equal(chatModelsCalled.every((c) => c.keepAlive === -1), true);

        const epModels = fleetRes.data.endpoints[0].data.models;
        const residentReport = epModels.find((m) => m.model === 'qwen3-coder-30b-moe:latest');
        const unloadedReport = epModels.find((m) => m.model === 'unloaded-experimental:3b-q2_K');
        assert.equal(residentReport.verdict, 'PASS');
        assert.equal(unloadedReport.verdict, 'STATIC_WARN');
        assert.equal(unloadedReport.canary, null);
    });

    it('runs CLI subcommands deterministically with INV-LEGO-01 --json envelopes', async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'quantcheck-cli-'));
        const ggufPath = path.join(tmpDir, 'sample.gguf');
        try {
            fs.writeFileSync(ggufPath, buildSyntheticGgufBuffer());
            const lines = [];
            const cli = createQuantcheckCli();
            const { exitCode, result } = await cli.run(['gguf', '--file', ggufPath, '--json'], {
                stdout: (l) => lines.push(l),
                stderr: () => {}
            });
            assert.equal(exitCode, 0);
            assert.equal(result.ok, true);
            const parsedJson = JSON.parse(lines[0]);
            assert.equal(parsedJson.ok, true);
            assert.equal(parsedJson.data.quantizationLabel, 'Q4_K_M');
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });
});
