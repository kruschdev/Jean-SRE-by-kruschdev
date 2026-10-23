#!/usr/bin/env node

/**
 * @file workspace_optimizer.js
 * @description Sweep script to analyze and perform workspace optimizations in the homelab:
 * 1. Find and clean orphaned/stale node_modules and .venv directories in /home/kruschdev/homelab/projects.
 * 2. Compress or rotate historical Jean SRE markdown audit logs.
 * 3. Query local host Ollama service to unload unpinned/idle models and reclaim GPU VRAM.
 * 
 * Usage:
 *   node scripts/workspace_optimizer.js          (Dry-run report)
 *   node scripts/workspace_optimizer.js --clean  (Execute cleanup)
 */

import fs from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const HOMELAB_ROOT = resolve(__dirname, '../../..');
const PROJECTS_DIR = join(HOMELAB_ROOT, 'projects');
const JEAN_SRE_DIR = join(PROJECTS_DIR, 'jean-sre');

// Curated active pinned models (should NOT be unloaded from VRAM)
const PINNED_MODELS = ['qwen2.5-coder:7b', 'bge-large', 'qwen2.5-coder:3b', 'spectralquant:latest', 'qwen3-coder-30b-moe:latest'];

// Idle thresholds
const STALE_THRESHOLD_DAYS = 15;

/**
 * Calculate directory size recursively.
 * @param {string} dirPath 
 * @returns {number} size in bytes
 */
function getDirSize(dirPath) {
    let size = 0;
    try {
        const stats = fs.statSync(dirPath);
        if (!stats.isDirectory()) {
            return stats.size;
        }
        const files = fs.readdirSync(dirPath);
        for (const file of files) {
            size += getDirSize(join(dirPath, file));
        }
    } catch (e) {
        // Ignore permission or file not found issues
    }
    return size;
}

/**
 * Format bytes to human readable string.
 * @param {number} bytes 
 * @returns {string}
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Scan projects for node_modules and .venv directories.
 */
async function scanDependencyDirectories() {
    console.log(`📂 Scanning projects directory: ${PROJECTS_DIR}...`);
    const candidates = [];
    const now = Date.now();

    try {
        if (!fs.existsSync(PROJECTS_DIR)) return [];
        const items = fs.readdirSync(PROJECTS_DIR);

        for (const item of items) {
            try {
                const projectPath = join(PROJECTS_DIR, item);
                const itemStat = fs.statSync(projectPath);
                if (!itemStat.isDirectory()) continue;

                // Look inside the project directory for node_modules or .venv
                const subdirs = ['node_modules', 'venv', '.venv'];
                for (const subdir of subdirs) {
                    const targetPath = join(projectPath, subdir);
                    if (fs.existsSync(targetPath)) {
                        const stat = fs.statSync(targetPath);
                        // Determine last modified time by checking the subdirectory stat
                        // or package.json/requirements.txt inside the project.
                        let lastModified = stat.mtimeMs;
                        
                        const markerFiles = ['package.json', 'requirements.txt', 'setup.py'];
                        for (const marker of markerFiles) {
                            const markerPath = join(projectPath, marker);
                            if (fs.existsSync(markerPath)) {
                                const markerStat = fs.statSync(markerPath);
                                if (markerStat.mtimeMs > lastModified) {
                                    lastModified = markerStat.mtimeMs;
                                }
                            }
                        }

                        const ageDays = (now - lastModified) / (1000 * 60 * 60 * 24);
                        const isStale = ageDays > STALE_THRESHOLD_DAYS;

                        candidates.push({
                            project: item,
                            type: subdir,
                            path: targetPath,
                            lastModified: new Date(lastModified),
                            ageDays: Math.round(ageDays),
                            isStale,
                            size: 0 // Will populate size only for stale/orphaned items to speed up scanning
                        });
                    }
                }
            } catch (itemErr) {
                // Skip broken symlinks or missing files gracefully
            }
        }
    } catch (e) {
        console.error(`Error scanning directories: ${e.message}`);
    }

    // Populate sizes for stale candidates
    for (const cand of candidates) {
        if (cand.isStale) {
            console.log(`  Calculating size of stale ${cand.type} in [${cand.project}]...`);
            cand.size = getDirSize(cand.path);
        }
    }

    return candidates;
}

/**
 * Scan and clean historical markdown audit logs.
 */
function processAuditLogs(isCleanMode) {
    console.log(`\n📝 Auditing SRE markdown logs in ${JEAN_SRE_DIR}...`);
    let logSavings = 0;
    try {
        if (!fs.existsSync(JEAN_SRE_DIR)) return 0;
        const files = fs.readdirSync(JEAN_SRE_DIR);

        // Match logs with pattern: jean_infra_audit_log_2026-*.md or jean_sre_audit_log_2026-*.md
        const logPattern = /^(jean_infra_audit_log_|jean_sre_audit_log_)\d{4}-\d{2}-\d{2}T\d{4}\.md$/;
        
        for (const file of files) {
            if (logPattern.test(file)) {
                const filePath = join(JEAN_SRE_DIR, file);
                const stat = fs.statSync(filePath);
                const ageDays = (Date.now() - stat.mtimeMs) / (1000 * 60 * 60 * 24);

                console.log(`  Found historical log: ${file} (${formatBytes(stat.size)}, ${Math.round(ageDays)} days old)`);

                if (isCleanMode) {
                    fs.rmSync(filePath, { force: true });
                    console.log(`    Deleted: ${file}`);
                    logSavings += stat.size;
                } else {
                    logSavings += stat.size;
                }
            }
        }
    } catch (e) {
        console.error(`Error processing audit logs: ${e.message}`);
    }
    return logSavings;
}

/**
 * Check and unload unpinned/idle Ollama models to reclaim VRAM.
 */
async function processOllamaModels(isCleanMode) {
    console.log(`\n🤖 Auditing local host Ollama models...`);
    try {
        const res = await fetch('http://127.0.0.1:11434/api/ps').catch(() => null);
        if (!res || !res.ok) {
            console.log(`  Ollama service not running or unreachable on port 11434.`);
            return;
        }

        const data = await res.json();
        const models = data.models || [];
        if (models.length === 0) {
            console.log(`  No models currently loaded in GPU VRAM.`);
            return;
        }

        for (const model of models) {
            const name = model.name;
            const size = model.size;
            const isPinned = PINNED_MODELS.some(pin => name.startsWith(pin));
            
            console.log(`  Loaded model: ${name} (Size in VRAM: ${formatBytes(size)}) - Pinned: ${isPinned ? 'YES' : 'NO'}`);

            if (!isPinned && isCleanMode) {
                console.log(`    Unloading ${name} from GPU VRAM...`);
                const unloadRes = await fetch('http://127.0.0.1:11434/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: name, keep_alive: 0 })
                }).catch(() => null);

                if (unloadRes && unloadRes.ok) {
                    console.log(`    Successfully unloaded ${name}.`);
                } else {
                    console.log(`    Failed to unload ${name}.`);
                }
            }
        }
    } catch (e) {
        console.error(`Error querying Ollama models: ${e.message}`);
    }
}

/**
 * Main execution.
 */
async function main() {
    const isCleanMode = process.argv.includes('--clean');

    console.log(`===================================================`);
    console.log(`  HOMELAB WORKSPACE OPTIMIZER — Mode: ${isCleanMode ? 'CLEANUP' : 'DRY-RUN'}`);
    console.log(`===================================================`);

    const candidates = await scanDependencyDirectories();
    const staleCandidates = candidates.filter(c => c.isStale);

    console.log(`\n📋 Stale Dependency Directories Summary:`);
    let totalSavings = 0;
    if (staleCandidates.length === 0) {
        console.log(`  No stale node_modules or .venv directories found.`);
    } else {
        for (const cand of staleCandidates) {
            console.log(`  [STALE] ${cand.type} in projects/${cand.project} (Size: ${formatBytes(cand.size)}, Age: ${cand.ageDays} days)`);
            totalSavings += cand.size;

            if (isCleanMode) {
                try {
                    fs.rmSync(cand.path, { recursive: true, force: true });
                    console.log(`    Deleted: ${cand.path}`);
                } catch (e) {
                    console.error(`    Failed to delete ${cand.path}: ${e.message}`);
                }
            }
        }
    }

    const logSavings = processAuditLogs(isCleanMode);
    
    await processOllamaModels(isCleanMode);

    console.log(`\n===================================================`);
    if (isCleanMode) {
        console.log(`✨ Cleanup Complete!`);
        console.log(`Reclaimed Workspace Space: ${formatBytes(totalSavings + logSavings)}`);
    } else {
        console.log(`💡 Dry-run Complete.`);
        console.log(`Potential Reclaimable Space: ${formatBytes(totalSavings + logSavings)}`);
        console.log(`Run with --clean flag to execute the optimization.`);
    }
    console.log(`===================================================`);
}

main().catch(err => {
    console.error('Execution failure:', err);
    process.exit(1);
});
