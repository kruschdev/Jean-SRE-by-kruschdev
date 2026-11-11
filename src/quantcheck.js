/**
 * @module quantcheck
 * GGUF Quantization & "Broken-on-Arrival" Artifact Sanity Gate (`INV-JEAN-QC-01..04`).
 *
 * Grounded in "Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model
 * Registries and How to Catch Them" (`arXiv:2609.05881`):
 *   1. Binary GGUF v2/v3 header, `imatrix` calibration metadata, and 32-byte tensor-offset
 *      bounds verification (`parseGgufHeaderBuffer`, `inspectGgufFileHeader`) reading <4MB
 *      via bounded file descriptors without loading multi-GB weights into V8 heap.
 *   2. Static Ollama `/api/show` + `/api/ps` artifact risk profiling (`evaluateArtifactRiskProfile`),
 *      catching uncalibrated sub-4.5B extreme low-bit quants (`Q2_K`, `Q3_K_*`, `IQ1_*`..`IQ3_*`),
 *      missing chat templates, and partial GPU->CPU VRAM spills (`size_vram < size * 0.99`).
 *   3. Deterministic behavioral canary probes (`runModelCanarySuite`) across embedding geometry
 *      (finite L2 norm + semantic margin >= 0.12) and generative/code/tool fidelity (repetition
 *      entropy collapse detection, zero-privilege `node:vm` `clampRange` execution across 5 vectors,
 *      JSON arithmetic invariant, and tool-call schema validation).
 *   4. Zero-thrash fleet screening (`auditOllamaEndpointQuantization`, `auditFleetQuantization`):
 *      statically profiles all installed `/api/tags` via `/api/show` while running live behavioral
 *      probes strictly on VRAM-resident `/api/ps` models by default (`INV-JEAN-QC-02`).
 */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { performance } from 'node:perf_hooks';
import { okEnvelope, errorEnvelope } from '@krusch/toolkit/http';
import { defineCli } from '@krusch/toolkit/cli';

export const GGUF_MAGIC = 0x46554747; // "GGUF" little-endian ('G','G','U','F')
const DEFAULT_ALIGNMENT = 32;
const INITIAL_HEADER_READ_BYTES = 8 * 1024 * 1024; // 8 MiB fast path
const MAX_HEADER_READ_BYTES = 32 * 1024 * 1024; // 32 MiB ceiling for 256K+ BPE vocab headers (Gemma 4 / Qwen3.8)

// GGUF value type IDs (GGUF v2/v3 specification)
const GGUF_TYPE_UINT8 = 0;
const GGUF_TYPE_INT8 = 1;
const GGUF_TYPE_UINT16 = 2;
const GGUF_TYPE_INT16 = 3;
const GGUF_TYPE_UINT32 = 4;
const GGUF_TYPE_INT32 = 5;
const GGUF_TYPE_FLOAT32 = 6;
const GGUF_TYPE_BOOL = 7;
const GGUF_TYPE_STRING = 8;
const GGUF_TYPE_ARRAY = 9;
const GGUF_TYPE_UINT64 = 10;
const GGUF_TYPE_INT64 = 11;
const GGUF_TYPE_FLOAT64 = 12;

/**
 * Known GGML tensor element block sizes (`[blockElements, blockBytes, label]`).
 */
const GGML_TENSOR_TYPES = new Map([
    [0, [1, 4, 'F32']],
    [1, [1, 2, 'F16']],
    [2, [32, 18, 'Q4_0']],
    [3, [32, 20, 'Q4_1']],
    [6, [32, 22, 'Q5_0']],
    [7, [32, 24, 'Q5_1']],
    [8, [32, 34, 'Q8_0']],
    [9, [32, 36, 'Q8_1']],
    [10, [256, 84, 'Q2_K']],
    [11, [256, 110, 'Q3_K']],
    [12, [256, 144, 'Q4_K']],
    [13, [256, 176, 'Q5_K']],
    [14, [256, 210, 'Q6_K']],
    [15, [256, 292, 'Q8_K']],
    [16, [256, 66, 'IQ2_XXS']],
    [17, [256, 74, 'IQ2_XS']],
    [18, [256, 98, 'IQ3_XXS']],
    [19, [256, 50, 'IQ1_S']],
    [20, [32, 18, 'IQ4_NL']],
    [21, [256, 110, 'IQ3_S']],
    [22, [256, 82, 'IQ2_S']],
    [23, [256, 136, 'IQ4_XS']],
    [24, [1, 1, 'I8']],
    [25, [1, 2, 'I16']],
    [26, [1, 4, 'I32']],
    [27, [1, 8, 'I64']],
    [28, [1, 8, 'F64']],
    [29, [256, 56, 'IQ1_M']],
    [30, [1, 2, 'BF16']],
    [39, [32, 17, 'MXFP4']]
]);

/**
 * GGUF `general.file_type` enum codes mapped to canonical quantization names.
 */
const GGUF_FILE_TYPE_LABELS = new Map([
    [0, 'F32'],
    [1, 'F16'],
    [2, 'Q4_0'],
    [3, 'Q4_1'],
    [7, 'Q8_0'],
    [8, 'Q5_0'],
    [9, 'Q5_1'],
    [10, 'Q2_K'],
    [11, 'Q3_K_S'],
    [12, 'Q3_K_M'],
    [13, 'Q3_K_L'],
    [14, 'Q4_K_S'],
    [15, 'Q4_K_M'],
    [16, 'Q5_K_S'],
    [17, 'Q5_K_M'],
    [18, 'Q6_K'],
    [19, 'IQ2_XXS'],
    [20, 'IQ2_XS'],
    [21, 'Q2_K_S'],
    [22, 'IQ3_XS'],
    [23, 'IQ3_XXS'],
    [24, 'IQ1_S'],
    [25, 'IQ4_NL'],
    [26, 'IQ3_S'],
    [27, 'IQ3_M'],
    [28, 'IQ2_S'],
    [29, 'IQ2_M'],
    [30, 'IQ4_XS'],
    [31, 'IQ1_M'],
    [32, 'BF16'],
    [38, 'MXFP4']
]);

const EXTREME_LOW_BIT_QUANT_RE = /^(?:Q2_K(?:_[A-Z]+)?|Q3_K(?:_[A-Z]+)?|IQ1_[A-Z0-9]+|IQ2_[A-Z0-9]+|IQ3_[A-Z0-9]+|Q2_0|Q3_0)$/i;
const EMBED_FAMILY_HINT_RE = /(?:embed|arctic-embed|nomic-embed|bge-|minilm|e5-|gte-)/i;

/**
 * Parses human-readable parameter size strings (`"566.70M"`, `"3.1B"`, `"30.5B"`, `"7B"`) into billions.
 *
 * @param {string|number|null|undefined} rawSize
 * @returns {number|null}
 */
export function parseParamCountBillions(rawSize) {
    if (typeof rawSize === 'number' && Number.isFinite(rawSize) && rawSize > 0) {
        return Number((rawSize > 1e6 ? rawSize / 1e9 : rawSize).toFixed(6));
    }
    if (typeof rawSize !== 'string') return null;
    const m = rawSize.trim().match(/^([0-9]+(?:\.[0-9]+)?)\s*([TBMK])?B?$/i);
    if (!m) return null;
    const val = Number(m[1]);
    if (!Number.isFinite(val) || val <= 0) return null;
    const unit = (m[2] || 'B').toUpperCase();
    if (unit === 'T') return Number((val * 1000).toFixed(6));
    if (unit === 'B') return Number(val.toFixed(6));
    if (unit === 'M') return Number((val / 1000).toFixed(6));
    if (unit === 'K') return Number((val / 1_000_000).toFixed(6));
    return Number(val.toFixed(6));
}

function alignOffset(offset, alignment = DEFAULT_ALIGNMENT) {
    const safeAlign = Number.isInteger(alignment) && alignment > 0 ? alignment : DEFAULT_ALIGNMENT;
    const rem = offset % safeAlign;
    return rem === 0 ? offset : offset + (safeAlign - rem);
}

/**
 * Parses a GGUF v2/v3 header and tensor descriptor table from a Buffer.
 * Validates magic, version, `imatrix` provenance keys, 32-byte alignment, and file truncation bounds.
 *
 * @param {Buffer|Uint8Array} rawBuffer
 * @param {{ fileSizeBytes?: number|null, maxCapturedArrayLen?: number }} [options={}]
 * @returns {{
 *   valid: boolean,
 *   errors: string[],
 *   warnings: string[],
 *   version: number|null,
 *   tensorCount: number,
 *   metadataKvCount: number,
 *   alignment: number,
 *   architecture: string|null,
 *   modelName: string|null,
 *   fileTypeCode: number|null,
 *   quantizationLabel: string|null,
 *   parameterCountBillions: number|null,
 *   hasImatrix: boolean,
 *   imatrixSource: string|null,
 *   hasChatTemplate: boolean,
 *   eosTokenId: number|null,
 *   dataSectionOffset: number|null,
 *   expectedMinFileBytes: number|null,
 *   tensorTypeCounts: Record<string, number>
 * }}
 */
export function parseGgufHeaderBuffer(rawBuffer, options = {}) {
    const buf = Buffer.isBuffer(rawBuffer) ? rawBuffer : Buffer.from(rawBuffer || []);
    const fileSizeBytes = Number.isFinite(options.fileSizeBytes) ? Number(options.fileSizeBytes) : buf.length;
    const maxCapturedArrayLen = options.maxCapturedArrayLen ?? 16;

    const errors = [];
    const warnings = [];
    const baseOut = {
        valid: false,
        errors,
        warnings,
        version: null,
        tensorCount: 0,
        metadataKvCount: 0,
        alignment: DEFAULT_ALIGNMENT,
        architecture: null,
        modelName: null,
        fileTypeCode: null,
        quantizationLabel: null,
        parameterCountBillions: null,
        hasImatrix: false,
        imatrixSource: null,
        hasChatTemplate: false,
        eosTokenId: null,
        dataSectionOffset: null,
        expectedMinFileBytes: null,
        tensorTypeCounts: {}
    };

    if (buf.length < 24) {
        errors.push(`GGUF header truncated (${buf.length} bytes < 24-byte minimum)`);
        return baseOut;
    }

    const magic = buf.readUInt32LE(0);
    if (magic !== GGUF_MAGIC) {
        errors.push(`Invalid GGUF magic 0x${magic.toString(16)} (expected 0x46554747 "GGUF")`);
        return baseOut;
    }

    const version = buf.readUInt32LE(4);
    baseOut.version = version;
    if (version !== 2 && version !== 3) {
        errors.push(`Unsupported GGUF version ${version} (expected v2 or v3)`);
        return baseOut;
    }

    const tensorCountBig = buf.readBigUInt64LE(8);
    const kvCountBig = buf.readBigUInt64LE(16);
    if (tensorCountBig > 2_000_000n || kvCountBig > 500_000n) {
        errors.push(`Implausible GGUF counts (tensors=${tensorCountBig}, kv=${kvCountBig})`);
        return baseOut;
    }

    const tensorCount = Number(tensorCountBig);
    const metadataKvCount = Number(kvCountBig);
    baseOut.tensorCount = tensorCount;
    baseOut.metadataKvCount = metadataKvCount;

    let pos = 24;

    const ensureBytes = (need, label) => {
        if (pos + need > buf.length) {
            throw new Error(`Unexpected EOF reading ${label} at offset ${pos} (needed ${need}B, buffer ${buf.length}B)`);
        }
    };

    const readGgufString = (label, maxKeepChars = 256) => {
        ensureBytes(8, `${label}.len`);
        const strLenBig = buf.readBigUInt64LE(pos);
        pos += 8;
        if (strLenBig > 64n * 1024n * 1024n) {
            throw new Error(`Excessive GGUF string length ${strLenBig} for ${label}`);
        }
        const strLen = Number(strLenBig);
        ensureBytes(strLen, `${label}.bytes`);
        const keptLen = Math.min(strLen, maxKeepChars);
        const val = buf.subarray(pos, pos + keptLen).toString('utf8');
        pos += strLen;
        return { text: val, byteLength: strLen };
    };

    const skipOrReadValue = (vtype, keyName, captureScalar = true) => {
        switch (vtype) {
            case GGUF_TYPE_UINT8: {
                ensureBytes(1, keyName);
                const v = buf.readUInt8(pos);
                pos += 1;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_INT8: {
                ensureBytes(1, keyName);
                const v = buf.readInt8(pos);
                pos += 1;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_BOOL: {
                ensureBytes(1, keyName);
                const v = buf.readUInt8(pos) !== 0;
                pos += 1;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_UINT16: {
                ensureBytes(2, keyName);
                const v = buf.readUInt16LE(pos);
                pos += 2;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_INT16: {
                ensureBytes(2, keyName);
                const v = buf.readInt16LE(pos);
                pos += 2;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_UINT32: {
                ensureBytes(4, keyName);
                const v = buf.readUInt32LE(pos);
                pos += 4;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_INT32: {
                ensureBytes(4, keyName);
                const v = buf.readInt32LE(pos);
                pos += 4;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_FLOAT32: {
                ensureBytes(4, keyName);
                const v = buf.readFloatLE(pos);
                pos += 4;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_UINT64: {
                ensureBytes(8, keyName);
                const v = Number(buf.readBigUInt64LE(pos));
                pos += 8;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_INT64: {
                ensureBytes(8, keyName);
                const v = Number(buf.readBigInt64LE(pos));
                pos += 8;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_FLOAT64: {
                ensureBytes(8, keyName);
                const v = buf.readDoubleLE(pos);
                pos += 8;
                return captureScalar ? v : null;
            }
            case GGUF_TYPE_STRING: {
                const s = readGgufString(keyName, captureScalar ? 512 : 0);
                return captureScalar ? s.text : null;
            }
            case GGUF_TYPE_ARRAY: {
                ensureBytes(12, `${keyName}.arrayHeader`);
                const elemType = buf.readUInt32LE(pos);
                const elemCountBig = buf.readBigUInt64LE(pos + 4);
                pos += 12;
                if (elemCountBig > 10_000_000n) {
                    throw new Error(`Excessive GGUF array length ${elemCountBig} for ${keyName}`);
                }
                const elemCount = Number(elemCountBig);
                const fixedBytesByType = {
                    [GGUF_TYPE_UINT8]: 1,
                    [GGUF_TYPE_INT8]: 1,
                    [GGUF_TYPE_BOOL]: 1,
                    [GGUF_TYPE_UINT16]: 2,
                    [GGUF_TYPE_INT16]: 2,
                    [GGUF_TYPE_UINT32]: 4,
                    [GGUF_TYPE_INT32]: 4,
                    [GGUF_TYPE_FLOAT32]: 4,
                    [GGUF_TYPE_UINT64]: 8,
                    [GGUF_TYPE_INT64]: 8,
                    [GGUF_TYPE_FLOAT64]: 8
                };
                if (!captureScalar && fixedBytesByType[elemType]) {
                    const skipTotal = elemCount * fixedBytesByType[elemType];
                    ensureBytes(skipTotal, `${keyName}.arrayData`);
                    pos += skipTotal;
                    return null;
                }
                if (!captureScalar && elemType === GGUF_TYPE_STRING) {
                    for (let idx = 0; idx < elemCount; idx++) {
                        if (pos + 8 > buf.length) {
                            throw new Error(`Unexpected EOF reading ${keyName}[${idx}].len at offset ${pos}`);
                        }
                        const sLen = Number(buf.readBigUInt64LE(pos));
                        pos += 8;
                        if (sLen < 0 || sLen > 64 * 1024 * 1024 || pos + sLen > buf.length) {
                            throw new Error(`Unexpected EOF reading ${keyName}[${idx}].bytes at offset ${pos}`);
                        }
                        pos += sLen;
                    }
                    return null;
                }
                const out = [];
                for (let idx = 0; idx < elemCount; idx++) {
                    const keepElem = captureScalar && idx < maxCapturedArrayLen;
                    const v = skipOrReadValue(elemType, `${keyName}[${idx}]`, keepElem);
                    if (keepElem) out.push(v);
                }
                return captureScalar ? out : null;
            }
            default:
                throw new Error(`Unknown GGUF metadata value type ${vtype} for key "${keyName}"`);
        }
    };

    const INTERESTING_KV_KEYS = new Set([
        'general.architecture',
        'general.name',
        'general.file_type',
        'general.quantization_version',
        'general.alignment',
        'general.size_label',
        'quantize.imatrix.file',
        'quantize.imatrix.dataset',
        'quantize.imatrix.entries_count',
        'quantize.imatrix.chunks_count',
        'tokenizer.chat_template',
        'tokenizer.ggml.eos_token_id'
    ]);

    const kvStore = Object.create(null);
    try {
        for (let k = 0; k < metadataKvCount; k++) {
            const keyObj = readGgufString(`kv[${k}].key`, 160);
            const key = keyObj.text;
            ensureBytes(4, `kv[${k}].type`);
            const vtype = buf.readUInt32LE(pos);
            pos += 4;
            const shouldCapture = INTERESTING_KV_KEYS.has(key) || key.startsWith('quantize.imatrix.');
            const val = skipOrReadValue(vtype, key, shouldCapture);
            if (shouldCapture) {
                kvStore[key] = val;
            }
        }
    } catch (err) {
        errors.push(err.message);
        return baseOut;
    }

    const alignment =
        Number.isInteger(kvStore['general.alignment']) && kvStore['general.alignment'] > 0
            ? kvStore['general.alignment']
            : DEFAULT_ALIGNMENT;
    baseOut.alignment = alignment;
    baseOut.architecture = typeof kvStore['general.architecture'] === 'string' ? kvStore['general.architecture'] : null;
    baseOut.modelName = typeof kvStore['general.name'] === 'string' ? kvStore['general.name'] : null;
    baseOut.fileTypeCode = Number.isInteger(kvStore['general.file_type']) ? kvStore['general.file_type'] : null;
    baseOut.quantizationLabel =
        baseOut.fileTypeCode !== null ? GGUF_FILE_TYPE_LABELS.get(baseOut.fileTypeCode) || `FT_${baseOut.fileTypeCode}` : null;
    baseOut.parameterCountBillions = parseParamCountBillions(kvStore['general.size_label'] || baseOut.modelName);
    baseOut.imatrixSource =
        typeof kvStore['quantize.imatrix.file'] === 'string'
            ? kvStore['quantize.imatrix.file']
            : typeof kvStore['quantize.imatrix.dataset'] === 'string'
              ? kvStore['quantize.imatrix.dataset']
              : null;
    baseOut.hasImatrix = Boolean(
        baseOut.imatrixSource ||
            (Number(kvStore['quantize.imatrix.entries_count']) > 0) ||
            (Number(kvStore['quantize.imatrix.chunks_count']) > 0)
    );
    baseOut.hasChatTemplate = Boolean(
        typeof kvStore['tokenizer.chat_template'] === 'string' && kvStore['tokenizer.chat_template'].trim().length > 0
    );
    baseOut.eosTokenId = Number.isInteger(kvStore['tokenizer.ggml.eos_token_id'])
        ? kvStore['tokenizer.ggml.eos_token_id']
        : null;

    // Parse tensor descriptors and compute exact data-section bounds + total parameters
    let maxTensorEndRel = 0;
    let totalElements = 0;
    const tensorTypeCounts = Object.create(null);

    try {
        for (let t = 0; t < tensorCount; t++) {
            const tName = readGgufString(`tensor[${t}].name`, 128).text;
            ensureBytes(4, `tensor[${t}].n_dims`);
            const nDims = buf.readUInt32LE(pos);
            pos += 4;
            if (nDims < 1 || nDims > 4) {
                errors.push(`Tensor "${tName}" has invalid n_dims=${nDims}`);
                return baseOut;
            }
            ensureBytes(nDims * 8 + 4 + 8, `tensor[${t}].dims_type_offset`);
            let elemCount = 1;
            const dims = [];
            for (let d = 0; d < nDims; d++) {
                const dimVal = Number(buf.readBigUInt64LE(pos));
                pos += 8;
                if (!Number.isFinite(dimVal) || dimVal <= 0) {
                    errors.push(`Tensor "${tName}" has non-positive dimension[${d}]=${dimVal}`);
                    return baseOut;
                }
                dims.push(dimVal);
                elemCount *= dimVal;
            }
            const ggmlType = buf.readUInt32LE(pos);
            pos += 4;
            const relOffset = Number(buf.readBigUInt64LE(pos));
            pos += 8;

            if (relOffset % alignment !== 0) {
                errors.push(`Tensor "${tName}" offset ${relOffset} violates ${alignment}-byte alignment`);
            }

            const typeSpec = GGML_TENSOR_TYPES.get(ggmlType);
            const typeLabel = typeSpec ? typeSpec[2] : `TYPE_${ggmlType}`;
            tensorTypeCounts[typeLabel] = (tensorTypeCounts[typeLabel] || 0) + 1;
            totalElements += elemCount;

            if (typeSpec) {
                const [blockElems, blockBytes] = typeSpec;
                if (dims[0] % blockElems !== 0) {
                    errors.push(`Tensor "${tName}" ne0=${dims[0]} not divisible by ${typeLabel} block size ${blockElems}`);
                } else {
                    const byteSize = (elemCount / blockElems) * blockBytes;
                    const endRel = relOffset + byteSize;
                    if (endRel > maxTensorEndRel) maxTensorEndRel = endRel;
                }
            } else {
                warnings.push(`Tensor "${tName}" uses unrecognized ggml_type=${ggmlType}`);
            }
        }
    } catch (err) {
        errors.push(err.message);
        return baseOut;
    }

    baseOut.tensorTypeCounts = { ...tensorTypeCounts };
    if (baseOut.parameterCountBillions === null && totalElements > 0) {
        baseOut.parameterCountBillions = Number((totalElements / 1e9).toFixed(3));
    }

    const dataSectionOffset = tensorCount > 0 ? alignOffset(pos, alignment) : pos;
    const expectedMinFileBytes = dataSectionOffset + maxTensorEndRel;
    baseOut.dataSectionOffset = dataSectionOffset;
    baseOut.expectedMinFileBytes = expectedMinFileBytes;

    if (fileSizeBytes < expectedMinFileBytes) {
        errors.push(
            `Truncated GGUF file: actual size ${fileSizeBytes}B < required tensor payload end ${expectedMinFileBytes}B`
        );
    }

    if (
        baseOut.quantizationLabel &&
        EXTREME_LOW_BIT_QUANT_RE.test(baseOut.quantizationLabel) &&
        !baseOut.hasImatrix &&
        baseOut.parameterCountBillions !== null &&
        baseOut.parameterCountBillions <= 4.5
    ) {
        warnings.push(
            `Broken-on-Arrival high-risk profile (arXiv:2609.05881): ${baseOut.parameterCountBillions}B model quantized to ${baseOut.quantizationLabel} without imatrix calibration`
        );
    }

    baseOut.valid = errors.length === 0;
    return baseOut;
}

/**
 * Inspects a GGUF file on disk using a bounded file-descriptor read (`<= 8 MiB`).
 *
 * @param {string} filePath
 * @returns {object}
 */
export function inspectGgufFileHeader(filePath) {
    let fd = null;
    try {
        const stat = fs.statSync(filePath);
        if (!stat.isFile()) {
            return errorEnvelope('INVALID_GGUF_PATH', `Not a regular file: ${filePath}`, null, 'jean.quantcheck');
        }
        fd = fs.openSync(filePath, 'r');
        let readLen = Math.min(stat.size, INITIAL_HEADER_READ_BYTES);
        let buf = Buffer.allocUnsafe(readLen);
        let bytesRead = fs.readSync(fd, buf, 0, readLen, 0);
        let parsed = parseGgufHeaderBuffer(buf.subarray(0, bytesRead), { fileSizeBytes: stat.size });

        if (
            !parsed.valid &&
            stat.size > readLen &&
            readLen < MAX_HEADER_READ_BYTES &&
            parsed.errors.some((e) => e.startsWith('Unexpected EOF reading '))
        ) {
            readLen = Math.min(stat.size, MAX_HEADER_READ_BYTES);
            buf = Buffer.allocUnsafe(readLen);
            bytesRead = fs.readSync(fd, buf, 0, readLen, 0);
            parsed = parseGgufHeaderBuffer(buf.subarray(0, bytesRead), { fileSizeBytes: stat.size });
        }

        if (!parsed.valid) {
            return errorEnvelope(
                'GGUF_INTEGRITY_FAILED',
                parsed.errors.join('; '),
                { filePath, ...parsed },
                'jean.quantcheck'
            );
        }
        return okEnvelope(
            'jean.quantcheck',
            `Valid GGUF v${parsed.version} (${parsed.quantizationLabel || 'mixed'}, ${parsed.tensorCount} tensors, imatrix=${parsed.hasImatrix})`,
            { filePath, fileSizeBytes: stat.size, ...parsed }
        );
    } catch (err) {
        return errorEnvelope('GGUF_READ_ERROR', err.message, { filePath }, 'jean.quantcheck');
    } finally {
        if (fd !== null) {
            try {
                fs.closeSync(fd);
            } catch {}
        }
    }
}

/**
 * Evaluates static Ollama / GGUF quantization risk profile (`arXiv:2609.05881` + Homelab 100% GPU rules).
 *
 * @param {{
 *   model?: string,
 *   quantizationLevel?: string|null,
 *   parameterSize?: string|number|null,
 *   families?: string[],
 *   capabilities?: string[],
 *   template?: string|null,
 *   hasImatrix?: boolean|null,
 *   isResident?: boolean,
 *   sizeBytes?: number|null,
 *   sizeVramBytes?: number|null
 * }} [meta={}]
 * @returns {{
 *   modality: 'embedding' | 'generative',
 *   supportsTools: boolean,
 *   quantizationLevel: string,
 *   parameterCountBillions: number|null,
 *   isExtremeLowBit: boolean,
 *   gpuResidencyRatio: number|null,
 *   riskTier: 'NOMINAL' | 'MODERATE_LOW_BIT' | 'UNPINNED_CPU_SPILL' | 'MISSING_CHAT_TEMPLATE' | 'HIGH_RISK_LOW_BIT_SMALL_MODEL',
 *   warnings: string[]
 * }}
 */
export function evaluateArtifactRiskProfile(meta = {}) {
    const modelName = String(meta.model || '');
    const quant = String(meta.quantizationLevel || 'UNKNOWN').trim();
    const paramsB = parseParamCountBillions(meta.parameterSize);
    const families = Array.isArray(meta.families) ? meta.families.map(String) : [];
    const capabilities = Array.isArray(meta.capabilities) ? meta.capabilities.map((c) => String(c).toLowerCase()) : [];

    const isEmbedding =
        capabilities.includes('embedding') ||
        (capabilities.length > 0 && !capabilities.includes('completion') && !capabilities.includes('chat')) ||
        EMBED_FAMILY_HINT_RE.test(modelName) ||
        families.some((f) => /^(?:bert|nomic-bert|xlm-roberta)$/i.test(f));

    const modality = isEmbedding ? 'embedding' : 'generative';
    const supportsTools = capabilities.includes('tools');
    const isExtremeLowBit = EXTREME_LOW_BIT_QUANT_RE.test(quant);
    const warnings = [];
    let riskTier = 'NOMINAL';

    let gpuResidencyRatio = null;
    if (meta.isResident && Number(meta.sizeBytes) > 0 && Number.isFinite(Number(meta.sizeVramBytes))) {
        gpuResidencyRatio = Number((Number(meta.sizeVramBytes) / Number(meta.sizeBytes)).toFixed(4));
        if (gpuResidencyRatio < 0.99) {
            riskTier = 'UNPINNED_CPU_SPILL';
            warnings.push(
                `Partial CPU spill detected on resident model "${modelName}": ${(gpuResidencyRatio * 100).toFixed(1)}% VRAM (< 99% GPU residency policy)`
            );
        }
    }

    if (modality === 'generative' && typeof meta.template === 'string' && meta.template.trim().length === 0) {
        riskTier = 'MISSING_CHAT_TEMPLATE';
        warnings.push(`Generative model "${modelName}" has an empty chat template`);
    }

    if (isExtremeLowBit) {
        if (paramsB !== null && paramsB <= 4.5 && meta.hasImatrix !== true) {
            riskTier = 'HIGH_RISK_LOW_BIT_SMALL_MODEL';
            warnings.push(
                `High Broken-on-Arrival risk (arXiv:2609.05881): ${paramsB}B model at ${quant} without confirmed imatrix calibration`
            );
        } else if (meta.hasImatrix !== true && riskTier === 'NOMINAL') {
            riskTier = 'MODERATE_LOW_BIT';
            warnings.push(`Low-bit quantization (${quant}) without confirmed imatrix metadata; require behavioral canary gate`);
        }
    }

    return {
        modality,
        supportsTools,
        quantizationLevel: quant,
        parameterCountBillions: paramsB,
        isExtremeLowBit,
        gpuResidencyRatio,
        riskTier,
        warnings
    };
}

/**
 * Detects degenerate repetition loops (`"the the the..."`, `"000000..."`, or repeated 4-grams)
 * characteristic of collapsed low-bit quantizations.
 *
 * @param {string} text
 * @returns {{ tokenCount: number, uniqueRatio: number, maxRepeatRun: number, collapsed: boolean }}
 */
export function computeTokenRepetitionMetrics(text = '') {
    const raw = String(text || '').trim();
    if (!raw) {
        return { tokenCount: 0, uniqueRatio: 0, maxRepeatRun: 0, collapsed: true };
    }
    const tokens = raw.split(/\s+/).filter(Boolean);
    if (tokens.length === 0) {
        return { tokenCount: 0, uniqueRatio: 0, maxRepeatRun: 0, collapsed: true };
    }

    const unique = new Set(tokens.map((t) => t.toLowerCase()));
    const uniqueRatio = Number((unique.size / tokens.length).toFixed(4));

    let maxRepeatRun = 1;
    let curRun = 1;
    for (let i = 1; i < tokens.length; i++) {
        if (tokens[i] === tokens[i - 1]) {
            curRun++;
            if (curRun > maxRepeatRun) maxRepeatRun = curRun;
        } else {
            curRun = 1;
        }
    }

    // Check single-character runaway (`"000000000000000000000000"`)
    const charRunMatch = raw.match(/(.)\1{31,}/);
    const hasCharRunaway = Boolean(charRunMatch);

    const collapsed =
        hasCharRunaway ||
        maxRepeatRun >= 10 ||
        (tokens.length >= 20 && uniqueRatio < 0.22);

    return {
        tokenCount: tokens.length,
        uniqueRatio,
        maxRepeatRun,
        collapsed
    };
}

function stripReasoningBlocks(raw = '') {
    return String(raw || '')
        .replace(/<think>[\s\S]*?<\/think>/gi, '')
        .replace(/<\|channel>thought[\s\S]*?<channel\|>/gi, '')
        .trim();
}

function extractJsCodeCandidate(raw = '') {
    const cleaned = stripReasoningBlocks(raw);
    const fenceMatch = cleaned.match(/```(?:javascript|js|ts)?\s*([\s\S]*?)```/i);
    const code = (fenceMatch ? fenceMatch[1] : cleaned).trim();
    return code
        .replace(/^\s*export\s+default\s+/m, '')
        .replace(/^\s*export\s+/gm, '');
}

/**
 * Compiles and executes a model-generated `clampRange(val, lo, hi)` candidate inside a
 * zero-privilege `node:vm` null-prototype sandbox (`timeout: 250ms`) across 5 test vectors.
 *
 * @param {string} rawOutput
 * @returns {{ ok: boolean, passedVectors: number, totalVectors: number, error?: string }}
 */
export function executeSandboxedClampCanary(rawOutput) {
    const code = extractJsCodeCandidate(rawOutput);
    if (!code || !/\bclampRange\b/.test(code)) {
        return { ok: false, passedVectors: 0, totalVectors: 5, error: 'Missing clampRange declaration' };
    }
    if (/\b(?:process|require|import|globalThis|Function|eval|Proxy|Reflect)\b/.test(code)) {
        return { ok: false, passedVectors: 0, totalVectors: 5, error: 'Forbidden identifier in canary snippet' };
    }

    const vectors = [
        { args: [5, 1, 10], expected: 5 },
        { args: [-3, 0, 8], expected: 0 },
        { args: [99, 0, 10], expected: 10 },
        { args: [4, 4, 4], expected: 4 },
        { args: [2.5, 1.5, 3.5], expected: 2.5 }
    ];

    try {
        const sandbox = Object.create(null);
        sandbox.Math = Math;
        sandbox.Number = Number;
        const context = vm.createContext(sandbox, {
            codeGeneration: { strings: false, wasm: false }
        });
        const harnessScript = new vm.Script(
            `${code}\n;typeof clampRange === 'function' ? clampRange : null;`,
            { filename: 'quantcheck-canary.js' }
        );
        const fn = harnessScript.runInContext(context, { timeout: 250 });
        if (typeof fn !== 'function') {
            return { ok: false, passedVectors: 0, totalVectors: vectors.length, error: 'clampRange did not evaluate to a function' };
        }

        let passed = 0;
        for (const v of vectors) {
            sandbox.__args = v.args;
            const callScript = new vm.Script(`clampRange(__args[0], __args[1], __args[2])`);
            const actual = callScript.runInContext(context, { timeout: 100 });
            if (typeof actual === 'number' && Math.abs(actual - v.expected) < 1e-9) {
                passed++;
            }
        }
        return {
            ok: passed === vectors.length,
            passedVectors: passed,
            totalVectors: vectors.length,
            ...(passed < vectors.length ? { error: `Passed ${passed}/${vectors.length} clampRange test vectors` } : {})
        };
    } catch (err) {
        return { ok: false, passedVectors: 0, totalVectors: vectors.length, error: err.message };
    }
}

function cosineSim(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || a.length !== b.length) return NaN;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
        const va = Number(a[i]);
        const vb = Number(b[i]);
        if (!Number.isFinite(va) || !Number.isFinite(vb)) return NaN;
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom > 1e-12 ? dot / denom : NaN;
}

async function fetchJsonWithTimeout(fetchImpl, url, init = {}, timeoutMs = 30_000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const res = await fetchImpl(url, { ...init, signal: controller.signal });
        if (!res.ok) {
            const txt = await res.text().catch(() => '');
            throw new Error(`HTTP ${res.status} from ${url}: ${txt.slice(0, 200)}`);
        }
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Runs the deterministic `quantcheck` behavioral canary suite against a single Ollama model.
 *
 * @param {{
 *   baseUrl: string,
 *   model: string,
 *   modality?: 'embedding' | 'generative',
 *   supportsTools?: boolean,
 *   keepAlive?: number | string,
 *   timeoutMs?: number,
 *   fetchImpl?: typeof fetch
 * }} options
 * @returns {Promise<{
 *   verdict: 'PASS' | 'WARN' | 'BROKEN_ON_ARRIVAL',
 *   modality: 'embedding' | 'generative',
 *   probes: Array<{ name: string, passed: boolean, detail: string, latencyMs: number }>,
 *   throughputTokPerSec: number|null,
 *   totalLatencyMs: number
 * }>}
 */
export async function runModelCanarySuite(options = {}) {
    const baseUrl = String(options.baseUrl || 'http://127.0.0.1:11434').replace(/\/+$/, '');
    const model = String(options.model || '').trim();
    const modality = options.modality || (EMBED_FAMILY_HINT_RE.test(model) ? 'embedding' : 'generative');
    const supportsTools = Boolean(options.supportsTools);
    const keepAlive = options.keepAlive !== undefined ? options.keepAlive : -1;
    const timeoutMs = options.timeoutMs ?? 45_000;
    const fetchImpl = options.fetchImpl || globalThis.fetch;

    const tSuite0 = performance.now();
    const probes = [];
    let throughputTokPerSec = null;

    if (modality === 'embedding') {
        const t0 = performance.now();
        try {
            const inputs = [
                'PostgreSQL write-ahead log checkpoint fsync durability guarantee',
                'Database WAL disk flush crash recovery persistence',
                'Homemade cinnamon sourdough brioche baking recipe'
            ];
            const json = await fetchJsonWithTimeout(
                fetchImpl,
                `${baseUrl}/api/embed`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, input: inputs, keep_alive: keepAlive })
                },
                timeoutMs
            );
            const vecs = json?.embeddings;
            const elapsed = Number((performance.now() - t0).toFixed(2));

            if (!Array.isArray(vecs) || vecs.length !== 3 || !Array.isArray(vecs[0]) || vecs[0].length < 32) {
                probes.push({
                    name: 'EMBED_FINITE_NORM',
                    passed: false,
                    detail: `Invalid embedding tensor shape (${vecs?.length}x${vecs?.[0]?.length})`,
                    latencyMs: elapsed
                });
            } else {
                const norms = vecs.map((v) => {
                    let sq = 0;
                    for (const x of v) {
                        if (typeof x !== 'number' || !Number.isFinite(x)) return NaN;
                        sq += x * x;
                    }
                    return Math.sqrt(sq);
                });
                const finiteOk = norms.every((n) => Number.isFinite(n) && n > 0.1);
                probes.push({
                    name: 'EMBED_FINITE_NORM',
                    passed: finiteOk,
                    detail: finiteOk
                        ? `dim=${vecs[0].length}, L2 norms=[${norms.map((n) => n.toFixed(3)).join(', ')}]`
                        : `NaN/Inf or zero-norm embedding detected`,
                    latencyMs: elapsed
                });

                const posSim = cosineSim(vecs[0], vecs[1]);
                const negSim = cosineSim(vecs[0], vecs[2]);
                const margin = posSim - negSim;
                const sepOk = Number.isFinite(margin) && margin >= 0.12;
                probes.push({
                    name: 'EMBED_SEMANTIC_SEPARATION',
                    passed: sepOk,
                    detail: `posCos=${Number.isFinite(posSim) ? posSim.toFixed(4) : 'NaN'}, negCos=${Number.isFinite(negSim) ? negSim.toFixed(4) : 'NaN'}, margin=${Number.isFinite(margin) ? margin.toFixed(4) : 'NaN'} (min 0.12)`,
                    latencyMs: 0
                });
            }
        } catch (err) {
            probes.push({
                name: 'EMBED_FINITE_NORM',
                passed: false,
                detail: err.message,
                latencyMs: Number((performance.now() - t0).toFixed(2))
            });
        }

        const allPassed = probes.length > 0 && probes.every((p) => p.passed);
        return {
            verdict: allPassed ? 'PASS' : 'BROKEN_ON_ARRIVAL',
            modality,
            probes,
            throughputTokPerSec: null,
            totalLatencyMs: Number((performance.now() - tSuite0).toFixed(2))
        };
    }

    const isGemma = /gemma/i.test(model);

    // Probe 1: Code Generation + Repetition Entropy + Sandboxed Execution (`arXiv:2609.05881` core smoke task)
    const tCode0 = performance.now();
    try {
        const body = {
            model,
            stream: false,
            keep_alive: keepAlive,
            ...(isGemma ? { think: false } : {}),
            options: { temperature: 0, seed: 42, num_predict: 220 },
            messages: [
                {
                    role: 'user',
                    content:
                        'Write only a plain JavaScript function named clampRange(val, lo, hi) that returns lo if val < lo, hi if val > hi, and val otherwise. Do not include explanations.'
                }
            ]
        };
        const resp = await fetchJsonWithTimeout(
            fetchImpl,
            `${baseUrl}/api/chat`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            },
            timeoutMs
        );
        const latencyMs = Number((performance.now() - tCode0).toFixed(2));
        const content = String(resp?.message?.content || '');
        if (Number(resp?.eval_count) > 0 && Number(resp?.eval_duration) > 0) {
            throughputTokPerSec = Number(((Number(resp.eval_count) / Number(resp.eval_duration)) * 1e9).toFixed(2));
        }

        const rep = computeTokenRepetitionMetrics(stripReasoningBlocks(content));
        probes.push({
            name: 'ENTROPY_REPETITION_GATE',
            passed: !rep.collapsed,
            detail: `tokens=${rep.tokenCount}, uniqueRatio=${rep.uniqueRatio}, maxRepeatRun=${rep.maxRepeatRun}`,
            latencyMs: 0
        });

        const execRes = executeSandboxedClampCanary(content);
        probes.push({
            name: 'CODE_SYNTAX_AND_EXEC',
            passed: execRes.ok && !rep.collapsed,
            detail: execRes.ok
                ? `Verified ${execRes.passedVectors}/${execRes.totalVectors} deterministic clampRange test vectors`
                : execRes.error || 'clampRange execution failed',
            latencyMs
        });
    } catch (err) {
        probes.push({
            name: 'CODE_SYNTAX_AND_EXEC',
            passed: false,
            detail: err.message,
            latencyMs: Number((performance.now() - tCode0).toFixed(2))
        });
    }

    // Probe 2: Structured JSON & Exact Arithmetic Invariant (`17 + 32 = 49`)
    const tJson0 = performance.now();
    try {
        const body = {
            model,
            stream: false,
            format: 'json',
            keep_alive: keepAlive,
            ...(isGemma ? { think: false } : {}),
            options: { temperature: 0, seed: 42, num_predict: 120 },
            messages: [
                {
                    role: 'user',
                    content: 'Compute 17 + 32 and return a JSON object with exact keys "sum" (number) and "valid" (boolean true).'
                }
            ]
        };
        const resp = await fetchJsonWithTimeout(
            fetchImpl,
            `${baseUrl}/api/chat`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            },
            timeoutMs
        );
        const latencyMs = Number((performance.now() - tJson0).toFixed(2));
        const cleaned = stripReasoningBlocks(String(resp?.message?.content || ''));
        const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
        const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : cleaned);
        const jsonOk = parsed && Number(parsed.sum) === 49 && parsed.valid === true;
        probes.push({
            name: 'STRUCTURED_JSON_INVARIANT',
            passed: Boolean(jsonOk),
            detail: jsonOk ? `{"sum":49,"valid":true} verified` : `Unexpected JSON payload: ${cleaned.slice(0, 120)}`,
            latencyMs
        });
    } catch (err) {
        probes.push({
            name: 'STRUCTURED_JSON_INVARIANT',
            passed: false,
            detail: err.message,
            latencyMs: Number((performance.now() - tJson0).toFixed(2))
        });
    }

    // Probe 3: Tool-Call Schema Gate (executed when `/api/show` capabilities include `'tools'`)
    if (supportsTools) {
        const tTool0 = performance.now();
        try {
            const body = {
                model,
                stream: false,
                keep_alive: keepAlive,
                ...(isGemma ? { think: false } : {}),
                options: { temperature: 0, seed: 42, num_predict: 160 },
                tools: [
                    {
                        type: 'function',
                        function: {
                            name: 'check_disk_space',
                            description: 'Check free disk space on a mount point',
                            parameters: {
                                type: 'object',
                                properties: { mount: { type: 'string' } },
                                required: ['mount']
                            }
                        }
                    }
                ],
                messages: [
                    {
                        role: 'user',
                        content: 'Call check_disk_space for mount "/var/lib/app".'
                    }
                ]
            };
            const resp = await fetchJsonWithTimeout(
                fetchImpl,
                `${baseUrl}/api/chat`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                },
                timeoutMs
            );
            const latencyMs = Number((performance.now() - tTool0).toFixed(2));
            const toolCalls = resp?.message?.tool_calls;
            let toolOk = false;
            let detail = 'No tool_calls emitted';

            if (Array.isArray(toolCalls) && toolCalls.length > 0) {
                const fn = toolCalls[0]?.function || {};
                const rawArgs = fn.arguments;
                const args = typeof rawArgs === 'string' ? JSON.parse(rawArgs) : rawArgs;
                if (fn.name === 'check_disk_space' && args && String(args.mount).includes('/var/lib/app')) {
                    toolOk = true;
                    detail = `Structured tool_calls verified (${fn.name} mount="${args.mount}")`;
                } else {
                    detail = `Mismatched tool_call: ${JSON.stringify(fn).slice(0, 120)}`;
                }
            } else {
                const rawText = stripReasoningBlocks(String(resp?.message?.content || ''));
                if (rawText.includes('check_disk_space') && rawText.includes('/var/lib/app')) {
                    toolOk = true;
                    detail = `Inline tool_call payload verified`;
                }
            }

            probes.push({
                name: 'TOOL_CALL_SCHEMA_GATE',
                passed: toolOk,
                detail,
                latencyMs
            });
        } catch (err) {
            probes.push({
                name: 'TOOL_CALL_SCHEMA_GATE',
                passed: false,
                detail: err.message,
                latencyMs: Number((performance.now() - tTool0).toFixed(2))
            });
        }
    }

    const coreProbes = probes.filter((p) => p.name !== 'TOOL_CALL_SCHEMA_GATE');
    const corePassed = coreProbes.length > 0 && coreProbes.every((p) => p.passed);
    const allPassed = probes.length > 0 && probes.every((p) => p.passed);

    return {
        verdict: !corePassed ? 'BROKEN_ON_ARRIVAL' : allPassed ? 'PASS' : 'WARN',
        modality,
        probes,
        throughputTokPerSec,
        totalLatencyMs: Number((performance.now() - tSuite0).toFixed(2))
    };
}

/**
 * Audits an Ollama endpoint with strict VRAM-residency safety (`INV-JEAN-QC-02`):
 *   - Statically profiles ALL installed `/api/tags` models via `/api/show` (0 GPU swaps).
 *   - Runs behavioral canary probes ONLY on currently VRAM-resident `/api/ps` models
 *     (or a specific `targetModel` when explicitly screening a newly pulled/fused model).
 *
 * @param {{
 *   baseUrl: string,
 *   targetModel?: string|null,
 *   includeUnloaded?: boolean,
 *   unloadTargetAfter?: boolean,
 *   timeoutMs?: number,
 *   fetchImpl?: typeof fetch
 * }} options
 * @returns {Promise<object>}
 */
export async function auditOllamaEndpointQuantization(options = {}) {
    const baseUrl = String(options.baseUrl || 'http://127.0.0.1:11434').replace(/\/+$/, '');
    const targetModel = options.targetModel ? String(options.targetModel).trim() : null;
    const includeUnloaded = Boolean(options.includeUnloaded);
    const unloadTargetAfter = Boolean(options.unloadTargetAfter);
    const timeoutMs = options.timeoutMs ?? 45_000;
    const fetchImpl = options.fetchImpl || globalThis.fetch;

    try {
        const [tagsResp, psResp] = await Promise.all([
            fetchJsonWithTimeout(fetchImpl, `${baseUrl}/api/tags`, {}, timeoutMs),
            fetchJsonWithTimeout(fetchImpl, `${baseUrl}/api/ps`, {}, timeoutMs)
        ]);

        const installedModels = Array.isArray(tagsResp?.models) ? tagsResp.models : [];
        const residentModels = Array.isArray(psResp?.models) ? psResp.models : [];
        const residentByName = new Map();
        for (const rm of residentModels) {
            if (rm?.name) residentByName.set(String(rm.name), rm);
            if (rm?.model) residentByName.set(String(rm.model), rm);
        }

        const candidates = targetModel
            ? installedModels.filter((m) => m.name === targetModel || m.model === targetModel)
            : installedModels;

        if (targetModel && candidates.length === 0) {
            candidates.push({ name: targetModel, model: targetModel, details: {} });
        }

        const reports = [];
        for (const m of candidates) {
            const name = String(m.name || m.model);
            const residentEntry = residentByName.get(name) || null;
            const isResident = Boolean(residentEntry);

            let showInfo = {};
            try {
                showInfo = await fetchJsonWithTimeout(
                    fetchImpl,
                    `${baseUrl}/api/show`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model: name })
                    },
                    timeoutMs
                );
            } catch {
                showInfo = {};
            }

            const details = { ...(m.details || {}), ...(showInfo.details || {}) };
            const riskProfile = evaluateArtifactRiskProfile({
                model: name,
                quantizationLevel: details.quantization_level || residentEntry?.details?.quantization_level || 'UNKNOWN',
                parameterSize: details.parameter_size || residentEntry?.details?.parameter_size || null,
                families: details.families || (details.family ? [details.family] : []),
                capabilities: showInfo.capabilities || [],
                template: typeof showInfo.template === 'string' ? showInfo.template : undefined,
                isResident,
                sizeBytes: residentEntry?.size ?? m.size ?? null,
                sizeVramBytes: residentEntry?.size_vram ?? null
            });

            const shouldRunLiveCanary = Boolean(targetModel) || isResident || includeUnloaded;
            let canary = null;
            if (shouldRunLiveCanary) {
                canary = await runModelCanarySuite({
                    baseUrl,
                    model: name,
                    modality: riskProfile.modality,
                    supportsTools: riskProfile.supportsTools,
                    keepAlive: unloadTargetAfter && targetModel === name ? 0 : -1,
                    timeoutMs,
                    fetchImpl
                });
            }

            let overallVerdict = 'STATIC_OK';
            if (canary) {
                if (canary.verdict === 'BROKEN_ON_ARRIVAL') {
                    overallVerdict = 'BROKEN_ON_ARRIVAL';
                } else if (canary.verdict === 'WARN' || riskProfile.riskTier !== 'NOMINAL') {
                    overallVerdict = 'WARN';
                } else {
                    overallVerdict = 'PASS';
                }
            } else if (riskProfile.riskTier !== 'NOMINAL') {
                overallVerdict = 'STATIC_WARN';
            }

            reports.push({
                model: name,
                isResident,
                verdict: overallVerdict,
                riskProfile,
                canary
            });
        }

        const brokenCount = reports.filter((r) => r.verdict === 'BROKEN_ON_ARRIVAL').length;
        const warnCount = reports.filter((r) => r.verdict === 'WARN' || r.verdict === 'STATIC_WARN').length;
        const liveProbedCount = reports.filter((r) => r.canary !== null).length;

        if (brokenCount > 0) {
            return errorEnvelope(
                'BROKEN_ON_ARRIVAL_DETECTED',
                `${brokenCount} broken quantized artifact(s) detected on ${baseUrl}`,
                {
                    endpoint: baseUrl,
                    installedCount: installedModels.length,
                    residentCount: residentModels.length,
                    liveProbedCount,
                    brokenCount,
                    warnCount,
                    models: reports
                },
                'jean.quantcheck'
            );
        }

        return okEnvelope(
            'jean.quantcheck',
            `Screened ${reports.length} tag(s) on ${baseUrl} (${liveProbedCount} live VRAM-resident canary PASS, ${warnCount} warning(s))`,
            {
                endpoint: baseUrl,
                installedCount: installedModels.length,
                residentCount: residentModels.length,
                liveProbedCount,
                brokenCount: 0,
                warnCount,
                models: reports
            }
        );
    } catch (err) {
        return errorEnvelope('ENDPOINT_UNREACHABLE', err.message, { endpoint: baseUrl }, 'jean.quantcheck');
    }
}

/**
 * Screens multiple fleet Ollama endpoints in parallel without VRAM thrashing.
 *
 * @param {string[]} urls
 * @param {{ includeUnloaded?: boolean, timeoutMs?: number, fetchImpl?: typeof fetch }} [options={}]
 * @returns {Promise<object>}
 */
export async function auditFleetQuantization(urls = [], options = {}) {
    const uniqueUrls = Array.from(
        new Set(
            (Array.isArray(urls) && urls.length > 0 ? urls : ['http://127.0.0.1:11434'])
                .map((u) => String(u || '').trim().replace(/\/+$/, ''))
                .filter(Boolean)
        )
    );

    const results = await Promise.all(
        uniqueUrls.map((baseUrl) =>
            auditOllamaEndpointQuantization({
                baseUrl,
                includeUnloaded: options.includeUnloaded,
                timeoutMs: options.timeoutMs,
                fetchImpl: options.fetchImpl
            })
        )
    );

    const brokenEndpoints = results.filter((r) => !r.ok);
    const totalTags = results.reduce((acc, r) => acc + (r.data?.models?.length || r.error?.details?.models?.length || 0), 0);
    const totalLiveProbed = results.reduce(
        (acc, r) => acc + (r.data?.liveProbedCount || r.error?.details?.liveProbedCount || 0),
        0
    );

    if (brokenEndpoints.length > 0) {
        return errorEnvelope(
            'FLEET_QUANTCHECK_FAILURE',
            `${brokenEndpoints.length}/${uniqueUrls.length} Ollama endpoint(s) failed quantcheck screening`,
            {
                endpointCount: uniqueUrls.length,
                totalTags,
                totalLiveProbed,
                endpoints: results
            },
            'jean.quantcheck'
        );
    }

    return okEnvelope(
        'jean.quantcheck',
        `All ${uniqueUrls.length} fleet Ollama endpoint(s) passed quantcheck (${totalTags} tags statically screened, ${totalLiveProbed} VRAM-resident models canary-verified)`,
        {
            endpointCount: uniqueUrls.length,
            totalTags,
            totalLiveProbed,
            endpoints: results
        }
    );
}

function normalizeHostUrl(raw) {
    const trimmed = String(raw || '').trim();
    if (!trimmed) return null;
    if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
    return `http://${trimmed.replace(/\/+$/, '')}`;
}

export function resolveDefaultFleetUrls(env = process.env) {
    if (env.OLLAMA_FLEET_URLS) {
        const list = String(env.OLLAMA_FLEET_URLS)
            .split(',')
            .map(normalizeHostUrl)
            .filter(Boolean);
        if (list.length > 0) return Array.from(new Set(list));
    }
    const configured = [
        env.PERSONA_OLLAMA_URL,
        env.EXECUTOR_OLLAMA_URL,
        env.OLLAMA_EMBED_HOST,
        env.OLLAMA_HOST
    ]
        .map(normalizeHostUrl)
        .filter(Boolean);

    return configured.length > 0 ? Array.from(new Set(configured)) : ['http://127.0.0.1:11434'];
}

function formatEndpointHumanReport(envelope, io) {
    const payload = envelope.ok ? envelope.data : envelope.error?.details;
    const statusIcon = envelope.ok ? 'PASS' : 'FAIL';
    io.stdout(`[${statusIcon}] ${envelope.summary}`);
    const models = payload?.models || [];
    for (const m of models) {
        const residency = m.isResident
            ? `VRAM=${m.riskProfile?.gpuResidencyRatio !== null ? `${(m.riskProfile.gpuResidencyRatio * 100).toFixed(1)}%` : 'resident'}`
            : 'unloaded (static-only)';
        const tokStr =
            m.canary?.throughputTokPerSec != null ? `, ${m.canary.throughputTokPerSec} tok/s` : '';
        io.stdout(
            `  - ${m.model} [${m.verdict}] (${m.riskProfile?.modality}, quant=${m.riskProfile?.quantizationLevel}, ${residency}${tokStr})`
        );
        if (Array.isArray(m.riskProfile?.warnings)) {
            for (const w of m.riskProfile.warnings) {
                io.stdout(`      warn: ${w}`);
            }
        }
        if (Array.isArray(m.canary?.probes)) {
            for (const p of m.canary.probes) {
                io.stdout(`      probe ${p.name}: ${p.passed ? 'OK' : 'FAIL'} (${p.detail})`);
            }
        }
    }
}

/**
 * Builds the `defineCli` runner for `jean-quantcheck` (`gguf`, `screen`, `fleet`).
 *
 * @param {object} [deps={}]
 */
export function createQuantcheckCli(deps = {}) {
    const inspectFn = deps.inspectGgufFileHeader || inspectGgufFileHeader;
    const auditEndpointFn = deps.auditOllamaEndpointQuantization || auditOllamaEndpointQuantization;
    const auditFleetFn = deps.auditFleetQuantization || auditFleetQuantization;

    return defineCli({
        name: 'jean-quantcheck',
        version: '1.0.0',
        description: 'GGUF binary integrity & Broken-on-Arrival quantization sanity gate (arXiv:2609.05881)',
        commands: {
            gguf: {
                description: 'Verify GGUF v2/v3 header, imatrix provenance, 32B alignment, and truncation bounds',
                options: {
                    file: { type: 'string', short: 'f', description: 'Path to .gguf weight or blob file' }
                },
                run: async ({ values, positionals, io }) => {
                    const targetFile = values.file || positionals[0];
                    if (!targetFile) {
                        return errorEnvelope(
                            'MISSING_FILE_ARG',
                            'Provide --file <path> or a positional GGUF file path',
                            null,
                            'jean-quantcheck'
                        );
                    }
                    const res = inspectFn(path.resolve(targetFile));
                    if (!values.json && res.ok) {
                        const d = res.data;
                        io.stdout(`[PASS] ${res.summary}`);
                        io.stdout(
                            `  arch=${d.architecture || 'unknown'} params=${d.parameterCountBillions ?? 'n/a'}B alignment=${d.alignment}B dataOffset=${d.dataSectionOffset}B minSize=${d.expectedMinFileBytes}B actualSize=${d.fileSizeBytes}B`
                        );
                        for (const w of d.warnings || []) {
                            io.stdout(`  warn: ${w}`);
                        }
                    }
                    return res;
                }
            },
            screen: {
                description: 'Screen a specific Ollama model tag (static risk + live behavioral canary suite)',
                options: {
                    url: { type: 'string', short: 'u', description: 'Ollama base URL (default: http://127.0.0.1:11434)' },
                    model: { type: 'string', short: 'm', description: 'Target Ollama model tag to screen' },
                    'unload-after': {
                        type: 'boolean',
                        description: 'Send keep_alive=0 so temporary candidate tag unloads after screening'
                    },
                    'timeout-ms': { type: 'string', description: 'Per-probe HTTP timeout in milliseconds' }
                },
                run: async ({ values, positionals, io }) => {
                    const model = values.model || positionals[0];
                    if (!model) {
                        return errorEnvelope(
                            'MISSING_MODEL_ARG',
                            'Provide --model <tag> or a positional model name',
                            null,
                            'jean-quantcheck'
                        );
                    }
                    const baseUrl = normalizeHostUrl(values.url || process.env.OLLAMA_HOST) || 'http://127.0.0.1:11434';
                    const timeoutMs = values['timeout-ms'] ? Number(values['timeout-ms']) : 45_000;
                    const res = await auditEndpointFn({
                        baseUrl,
                        targetModel: model,
                        unloadTargetAfter: Boolean(values['unload-after']),
                        timeoutMs
                    });
                    if (!values.json && res.ok) {
                        formatEndpointHumanReport(res, io);
                    }
                    return res;
                }
            },
            fleet: {
                description: 'Zero-thrash fleet audit (static all tags + live canary on VRAM-resident /api/ps models)',
                options: {
                    urls: { type: 'string', description: 'Comma-separated Ollama endpoint URLs' },
                    'include-unloaded': {
                        type: 'boolean',
                        description: 'Also run live behavioral probes on unloaded tags (causes VRAM swaps)'
                    },
                    'timeout-ms': { type: 'string', description: 'Per-probe HTTP timeout in milliseconds' }
                },
                run: async ({ values, io }) => {
                    const urls = values.urls
                        ? String(values.urls)
                              .split(',')
                              .map(normalizeHostUrl)
                              .filter(Boolean)
                        : resolveDefaultFleetUrls(process.env);
                    const timeoutMs = values['timeout-ms'] ? Number(values['timeout-ms']) : 45_000;
                    const res = await auditFleetFn(urls, {
                        includeUnloaded: Boolean(values['include-unloaded']),
                        timeoutMs
                    });
                    if (!values.json && res.ok) {
                        io.stdout(`[PASS] ${res.summary}`);
                        for (const ep of res.data?.endpoints || []) {
                            formatEndpointHumanReport(ep, io);
                        }
                    }
                    return res;
                }
            }
        }
    });
}
