#!/usr/bin/env node

import { envOr } from '@krusch/toolkit/config';

const AGENT_URL = envOr('JEAN_AGENT_URL', null, 'http://localhost:3005');

function printUsage() {
    console.log(`
Usage:
  node trigger_emergency.js --type <type> --node <node> --desc "<description>"

Options:
  --type      Type of emergency (e.g. thermal, oom, container_crash)
  --node      Node name (kruschserv, kruschdev, kruschgame)
  --desc      Detailed description of the crisis

Example:
  node trigger_emergency.js --type thermal --node kruschdev --desc "CPU temperature reached 96°C on Xeon core 12"
`);
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.includes('--help') || args.includes('-h') || args.length === 0) {
        printUsage();
        process.exit(0);
    }

    let type = '';
    let node = '';
    let description = '';

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--type' && args[i + 1]) {
            type = args[i + 1];
        } else if (args[i] === '--node' && args[i + 1]) {
            node = args[i + 1];
        } else if (args[i] === '--desc' && args[i + 1]) {
            description = args[i + 1];
        }
    }

    if (!type || !node || !description) {
        console.error("Error: Missing required arguments: --type, --node, and --desc are required.");
        printUsage();
        process.exit(1);
    }

    console.log(`Sending emergency trigger to Jean harness at ${AGENT_URL}...`);
    console.log(`Payload:`);
    console.log(`  Type:        ${type}`);
    console.log(`  Node:        ${node}`);
    console.log(`  Description: ${description}`);

    try {
        const response = await fetch(`${AGENT_URL}/trigger`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(process.env.JEAN_API_KEY ? { 'x-api-key': process.env.JEAN_API_KEY } : {})
            },
            body: JSON.stringify({ type, node, description })
        });

        const data = await response.json();
        if (response.ok) {
            console.log(`\n✅ Success:`, data);
        } else {
            console.error(`\n❌ Error (${response.status}):`, data);
        }
    } catch (e) {
        console.error(`\n❌ Network Error: Failed to connect to Jean harness at ${AGENT_URL}:`, e.message);
    }
}

main().catch(e => {
    console.error("Fatal error:", e);
    process.exit(1);
});
