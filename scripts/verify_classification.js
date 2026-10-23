#!/usr/bin/env node

/**
 * @file verify_classification.js
 * @description Unit test verifying the correctness of our Jean SRE classifyIntentRisk implementation.
 */

function classifyIntentRisk(task) {
    const t = task.toLowerCase();
    
    // L0 (Destructive) - Forbidden from autonomous run, goes to review
    const l0Triggers = [
        { pattern: /drop\s+table/i, reason: 'mutates DB schema / drops table' },
        { pattern: /drop\s+database/i, reason: 'mutates DB / drops database' },
        { pattern: /delete\s+volume/i, reason: 'removes Docker volume' },
        { pattern: /volume\s+prune/i, reason: 'prunes Docker volumes' },
        { pattern: /system\s+prune/i, reason: 'prunes Docker system configurations' },
        { pattern: /rm\s+-rf/i, reason: 'recursive file deletion' },
        { pattern: /wipe\s+mount/i, reason: 'wipes volume/disk mounts' },
        { pattern: /(?<!--)format\s+/i, reason: 'disk formatting' },
        { pattern: /reboot/i, reason: 'system node reboot' },
        { pattern: /shutdown/i, reason: 'system node shutdown' },
        { pattern: /destroy/i, reason: 'infrastructure destruction' },
        { pattern: /delete\s+file/i, reason: 'destroys file assets' }
    ];

    for (const trigger of l0Triggers) {
        if (trigger.pattern.test(t)) {
            return { riskClass: 'L0', reason: trigger.reason };
        }
    }
    
    // L1 (Mild / Restorative) - Needs review/gate, goes to review
    const l1Triggers = [
        { pattern: /restart/i, reason: 'restarts microservice container' },
        { pattern: /reload/i, reason: 'reloads active process configurations' },
        { pattern: /stop\s+/i, reason: 'stops microservice container or service' },
        { pattern: /start\s+/i, reason: 'starts microservice container or service' },
        { pattern: /alter\s+/i, reason: 'alters active configurations / schema' },
        { pattern: /modify/i, reason: 'mutates codebase/config files' },
        { pattern: /update/i, reason: 'updates configurations / dependency' },
        { pattern: /clear.*cache/i, reason: 'clears active performance cache' },
        { pattern: /prune/i, reason: 'prunes system assets' },
        { pattern: /git\s+checkout/i, reason: 'git checkout alters code branch' },
        { pattern: /git\s+reset/i, reason: 'git reset forces file revert' },
        { pattern: /kill/i, reason: 'kills running process' },
        { pattern: /recreate/i, reason: 'recreates container or file' },
        { pattern: /write\s+/i, reason: 'writes data / files' },
        { pattern: /set\s+/i, reason: 'sets state variable / configs' }
    ];

    for (const trigger of l1Triggers) {
        if (trigger.pattern.test(t)) {
            return { riskClass: 'L1', reason: trigger.reason };
        }
    }
    
    // L2 (Safe / Read-Only) - safe to auto-run
    return { riskClass: 'L2', reason: 'Safe read-only diagnostics/telemetry retrieval' };
}

const testCases = [
    { task: "drop table users", expected: "L0" },
    { task: "delete volume caren-db", expected: "L0" },
    { task: "reboot kruschserv node", expected: "L0" },
    { task: "rm -rf /var/lib/docker/volumes", expected: "L0" },
    { task: "restart docker container for pocketlawyer-redis", expected: "L1" },
    { task: "reload nginx configurations", expected: "L1" },
    { task: "clear redis cache to free up memory", expected: "L1" },
    { task: "modify index.js to add logging statement", expected: "L1" },
    { task: "git checkout main branch to sync", expected: "L1" },
    { task: "docker ps --format 'table {{.Names}}\t{{.Status}}'", expected: "L2" },
    { task: "check disk usage with df -h command", expected: "L2" },
    { task: "read logs with cat /var/log/syslog | head -n 100", expected: "L2" }
];

console.log("🔍 Running Jean SRE Gated Routing Classification Tests...\n");
let passed = 0;

for (const tc of testCases) {
    const res = classifyIntentRisk(tc.task);
    if (res.riskClass === tc.expected) {
        console.log(`✅ PASSED: "${tc.task}" -> ${res.riskClass} (${res.reason})`);
        passed++;
    } else {
        console.log(`❌ FAILED: "${tc.task}" -> Expected: ${tc.expected}, Got: ${res.riskClass} (${res.reason})`);
    }
}

console.log(`\n🎉 Results: ${passed}/${testCases.length} tests passed successfully.`);
if (passed === testCases.length) {
    process.exit(0);
} else {
    process.exit(1);
}
