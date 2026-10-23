import pg from 'pg';
import fs from 'fs';

const envContent = fs.readFileSync('.env', 'utf8');
const dbUrlMatch = envContent.match(/DBOS_DATABASE_URL=(.*)/);
const dbUrl = dbUrlMatch ? dbUrlMatch[1] : '';

const pool = new pg.Pool({ connectionString: dbUrl });

async function check() {
    try {
        console.log("=== Querying Verification Intents directly from PostgreSQL ===");
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
    } catch(e) {
        console.error(e);
    } finally {
        pool.end();
    }
}
check();
