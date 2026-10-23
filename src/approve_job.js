import pg from 'pg';
import { randomUUID } from 'crypto';

const client = new pg.Client('postgresql://kruschdb:password@localhost:5433/kruschdb');
try {
    await client.connect();
    const job_id = '90a26716-6955-49ac-ba06-c3c5ed0cc064';
    
    // 1. Fetch original job
    const rowRes = await client.query("SELECT thread_id, payload FROM agent_execution_queue WHERE job_id = $1", [job_id]);
    const row = rowRes.rows[0];
    if (!row) {
        console.error("Job not found");
        process.exit(1);
    }
    
    // 2. Update status of the review job to 'approved'
    await client.query(
        "UPDATE agent_execution_queue SET status = $1, completed_at_ms = $2 WHERE job_id = $3",
        ['approved', Date.now(), job_id]
    );
    console.log("Updated review status of", job_id, "to approved");
    
    // 3. Spawn the execution job
    const new_job_id = randomUUID();
    const payload = row.payload || {};
    const instructions = payload.instructions || JSON.stringify(payload);
    
    const exec_payload = {
        task: "execute_approved_plan",
        instructions: instructions,
        context: {
            approved_from_review: job_id
        }
    };
    
    await client.query(
        `INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, status, started_at_ms)
         VALUES ($1, $2, $3, $4, 'pending', $5)`,
        [new_job_id, row.thread_id, 'execution', JSON.stringify(exec_payload), Date.now()]
    );
    console.log("Spawned execution job:", new_job_id);
    
} catch (e) {
    console.error(e);
} finally {
    await client.end();
}
