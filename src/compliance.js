/**
 * @module compliance
 * Pre-Flight SRE Compliance Linter for Homelab Stability Policies.
 * Intercepts tool calls and audits their arguments recursively against rules.
 */

/**
 * Recursively extracts all string values from an object.
 * Hardens the linter by scanning all inputs regardless of argument key names.
 */
function extractStrings(obj, arr = []) {
    if (!obj) return arr;
    if (typeof obj === 'string') {
        arr.push(obj);
    } else if (typeof obj === 'object') {
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                extractStrings(obj[key], arr);
            }
        }
    }
    return arr;
}

/**
 * Checks for No Sudo policy violations.
 * Blocks any attempt to execute commands utilizing sudo, su, or root access.
 */
function checkSudoViolation(text) {
    // Matches "sudo", "su -", or "su " command blocks in scripts or terminal command args.
    const sudoRegex = /\b(sudo|su)\b/i;
    if (sudoRegex.test(text)) {
        return {
            violation: true,
            reason: "Use of 'sudo' or 'su' is strictly prohibited on production nodes to guarantee safety constraints."
        };
    }
    return { violation: false };
}

/**
 * Checks for rsync safety rules.
 * Blocks "rsync --delete" unless it explicitly excludes database/cache storage folders.
 */
function checkRsyncSafety(text) {
    if (text.includes('rsync') && text.includes('--delete')) {
        const hasExclusions = text.includes('--exclude');
        const excludesPostgres = text.includes('postgres');
        const excludesRedis = text.includes('redis');
        const excludesData = text.includes('data');

        if (!hasExclusions || !excludesPostgres || !excludesRedis || !excludesData) {
            return {
                violation: true,
                reason: "Executing 'rsync --delete' requires explicitly excluding database storage directories ('postgres/', 'redis/', 'data/') to avoid catastrophic state loss."
            };
        }
    }
    return { violation: false };
}

/**
 * Checks for OS Drive Protection Policy.
 * Blocks storing persistent databases, large Docker caches, or model weights directly on boot drives.
 */
function checkOSDriveProtection(text) {
    // Detect typical directories used for heavy volumes or caches
    const heavyDirs = [
        '/var/lib/docker',
        '/var/lib/postgresql',
        '/var/lib/mysql',
        '/var/lib/redis',
        '/.ollama',
        '/.cache/huggingface'
    ];

    for (const dir of heavyDirs) {
        if (text.includes(dir)) {
            // Verify if the path is safely mapped under a mounted data target
            const isRoutedToMount = text.includes('/mnt/data') || 
                                    text.includes('/mnt/data2') || 
                                    text.includes('/mnt/nvme') || 
                                    text.includes('/mnt/media1') || 
                                    text.includes('/mnt/sharedmedia');
            if (!isRoutedToMount) {
                return {
                    violation: true,
                    reason: `Storage configuration for heavy system directory '${dir}' detected on protected OS drive. All user data, databases, and Docker volumes must reside on designated mounts under /mnt/ (e.g., /mnt/nvme, /mnt/data, /mnt/media1).`
                };
            }
        }
    }
    return { violation: false };
}

/**
 * Checks for restricted OS credential/key paths (/etc/shadow, /etc/sudoers, /root/, .ssh/id_, docker.sock, mcp_config.json).
 */
function checkRestrictedSystemPaths(text) {
    const restrictedRegex = /(?:\/etc\/shadow|\/etc\/sudoers|\/root\/|\.ssh\/id_[a-z0-9_]+|\/var\/run\/docker\.sock|mcp_config\.json|\.vscode\/mcp\.json)/i;
    if (restrictedRegex.test(text)) {
        return {
            violation: true,
            reason: "Access to restricted system path (/etc/shadow, /etc/sudoers, /root/, SSH private keys, docker.sock, or MCP secret configs) is prohibited."
        };
    }
    return { violation: false };
}

/**
 * Audits a tool call before execution.
 * @param {string} name - Name of the MCP or native tool.
 * @param {object} args - Arguments passed to the tool.
 * @returns {object} - { valid: boolean, allowed: boolean, reason?: string }
 */
export function verifyToolCompliance(name, args) {
    // Target any tool execution that runs shell/agent scripts or commands
    const stringValues = extractStrings(args);
    const fullText = stringValues.join('\n');

    // 1. Sudo check
    const sudoCheck = checkSudoViolation(fullText);
    if (sudoCheck.violation) {
        return { valid: false, allowed: false, reason: sudoCheck.reason };
    }

    // 2. Rsync check
    const rsyncCheck = checkRsyncSafety(fullText);
    if (rsyncCheck.violation) {
        return { valid: false, allowed: false, reason: rsyncCheck.reason };
    }

    // 3. OS Drive check
    const osDriveCheck = checkOSDriveProtection(fullText);
    if (osDriveCheck.violation) {
        return { valid: false, allowed: false, reason: osDriveCheck.reason };
    }

    // 4. Restricted system path check
    const pathCheck = checkRestrictedSystemPaths(fullText);
    if (pathCheck.violation) {
        return { valid: false, allowed: false, reason: pathCheck.reason };
    }

    // 5. Block write/patch tools from mutating .env or secret files
    const lowerName = String(name || '').toLowerCase();
    if (lowerName.includes('write') || lowerName.includes('patch')) {
        const targetFile = String(args?.file_path || args?.path || '');
        if (/(?:^|\/)\.env(?:\.|$)/i.test(targetFile) && !targetFile.endsWith('.env.example')) {
            return {
                valid: false,
                allowed: false,
                reason: "Direct mutation of .env secret files via agent write/patch tools is prohibited."
            };
        }
    }

    return { valid: true, allowed: true };
}
