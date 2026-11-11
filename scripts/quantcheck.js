#!/usr/bin/env node
/**
 * @file scripts/quantcheck.js
 * Jean SRE GGUF Quantization & "Broken-on-Arrival" Artifact Sanity Gate CLI (`INV-JEAN-QC-01..04`).
 */

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createQuantcheckCli, resolveDefaultFleetUrls } from '../src/quantcheck.js';

export { createQuantcheckCli, resolveDefaultFleetUrls };

const isDirectRun =
    process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (isDirectRun) {
    const cli = createQuantcheckCli();
    const { exitCode } = await cli.run(process.argv.slice(2));
    process.exitCode = exitCode;
}
