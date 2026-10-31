import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createPgPool } from '../../../lib/pg.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envPath = path.resolve(__dirname, '..', '.env');

let dbUrl = process.env.DBOS_DATABASE_URL || process.env.DATABASE_URL || '';
if (!dbUrl && fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    const dbUrlMatch = envContent.match(/^DBOS_DATABASE_URL=(.*)$/m);
    dbUrl = dbUrlMatch ? dbUrlMatch[1].trim() : '';
}

const pool = createPgPool(dbUrl, { serviceName: 'jean-sre-verify', max: 2 });

async function check() {
    try {
        console.log('=== Querying Verification Intents directly from PostgreSQL ===');
        const query = `
            SELECT job_id, status, payload, result, started_at_ms
            FROM agent_execution_queue
            WHERE job_id::text LIKE 'adaa97c4%'
               OR job_id::text LIKE '00b9a913%'
               OR job_id::text LIKE '5268c65d%'
            ORDER BY started_at_ms DESC;
        `;
        const res = await pool.query(query);
        console.log(JSON.stringify(res.rows, null, 2));
    } catch (e) {
        console.error(e);
    } finally {
        await pool.close();
    }
}
check();
