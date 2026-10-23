"""
Krusch-Nexus — DBOS Swarm Queue Integration.

Provides read access to the external openclaw-db PostgreSQL container
(agent_execution_queue table) for surfacing swarm debate results and
human review items in the Nexus frontend.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DBOS_DATABASE_URL = os.getenv("DBOS_DATABASE_URL")


def _get_dbos_conn():
    """Get a connection to the external DBOS queue database."""
    return psycopg2.connect(DBOS_DATABASE_URL)


def get_swarm_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch swarm jobs from the DBOS queue with optional filters."""
    conditions = []
    params = []

    if status:
        conditions.append("status = %s")
        params.append(status)
    if job_type:
        conditions.append("job_type = %s")
        params.append(job_type)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT job_id, thread_id, job_type, payload, result, status,
               started_at_ms, picked_at_ms, completed_at_ms, created_at
        FROM agent_execution_queue
        {where_clause}
        ORDER BY started_at_ms DESC
        LIMIT %s
    """
    params.append(limit)

    try:
        conn = _get_dbos_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        conn.close()

        jobs = []
        for row in rows:
            job = dict(row)
            # Convert UUID to string for JSON serialization
            job["job_id"] = str(job["job_id"])
            job["thread_id"] = str(job["thread_id"])
            # Convert timestamps
            if job.get("created_at"):
                job["created_at"] = job["created_at"].isoformat()
            # Compute human-readable elapsed time
            if job.get("started_at_ms"):
                started = datetime.fromtimestamp(job["started_at_ms"] / 1000, tz=timezone.utc)
                job["started_at"] = started.isoformat()
            jobs.append(job)
        return jobs

    except Exception as e:
        logger.error(f"[Swarm] Failed to query DBOS queue: {e}")
        return []


def get_swarm_stats() -> dict:
    """Get aggregate stats for the swarm queue."""
    query = """
        SELECT 
            COUNT(*) as all_time_jobs,
            COUNT(*) FILTER (WHERE status IN ('pending', 'running')) as active_backlog,
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'running') as running,
            COUNT(*) FILTER (WHERE status = 'completed') as completed,
            COUNT(*) FILTER (WHERE job_type = 'human_review' AND status = 'pending') as pending_human_reviews
        FROM agent_execution_queue
    """
    try:
        conn = _get_dbos_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception as e:
        logger.error(f"[Swarm] Failed to get stats: {e}")
        return {}


def get_jean_swarm_stats() -> dict:
    """Get swarm stats broken down by job_type for Jean SRE.

    Returns separate counts for Jean's intent jobs vs normal swarm
    debate activity, preventing Jean from conflating routine work
    with actionable problems.
    """
    query = """
        SELECT
            -- Jean's own intent jobs
            COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'pending') as intent_pending,
            COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'running') as intent_running,
            COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'completed') as intent_completed,
            COUNT(*) FILTER (WHERE job_type = 'intent' AND status = 'failed') as intent_failed,
            -- Normal swarm debate activity (NOT a problem)
            COUNT(*) FILTER (WHERE job_type IN ('debate_proposal', 'debate_rebuttal') AND status = 'running') as debate_running,
            COUNT(*) FILTER (WHERE job_type IN ('debate_proposal', 'debate_rebuttal') AND status = 'pending') as debate_pending,
            -- Other job types
            COUNT(*) FILTER (WHERE job_type NOT IN ('intent', 'debate_proposal', 'debate_rebuttal') AND status = 'pending') as other_pending,
            -- Overall
            COUNT(*) as total_jobs,
            COUNT(*) FILTER (WHERE status = 'completed') as total_completed,
            COUNT(*) FILTER (WHERE status = 'failed') as total_failed
        FROM agent_execution_queue
    """
    try:
        conn = _get_dbos_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception as e:
        logger.error(f"[Swarm] Failed to get Jean stats: {e}")
        return {}


def get_human_reviews(limit: int = 20) -> list[dict]:
    """Fetch pending human review items — the synthesized prototype plans."""
    return get_swarm_jobs(status="pending", job_type="human_review", limit=limit)


def get_debate_thread(thread_id: str) -> list[dict]:
    """Fetch all jobs in a debate thread to show the full Idea → Critique → Synthesis chain."""
    query = """
        SELECT job_id, thread_id, job_type, payload, result, status,
               started_at_ms, picked_at_ms, completed_at_ms, created_at
        FROM agent_execution_queue
        WHERE thread_id = %s
        ORDER BY started_at_ms ASC
    """
    try:
        conn = _get_dbos_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, [thread_id])
            rows = cur.fetchall()
        conn.close()

        jobs = []
        for row in rows:
            job = dict(row)
            job["job_id"] = str(job["job_id"])
            job["thread_id"] = str(job["thread_id"])
            if job.get("created_at"):
                job["created_at"] = job["created_at"].isoformat()
            if job.get("started_at_ms"):
                started = datetime.fromtimestamp(job["started_at_ms"] / 1000, tz=timezone.utc)
                job["started_at"] = started.isoformat()
            jobs.append(job)
        return jobs

    except Exception as e:
        logger.error(f"[Swarm] Failed to fetch thread {thread_id}: {e}")
        return []


def update_review_status(job_id: str, new_status: str) -> bool:
    """Mark a human review as approved/rejected/dismissed. If approved, dispatch for execution."""
    if new_status not in ("approved", "rejected", "dismissed"):
        return False
    try:
        import uuid
        import json
        conn = _get_dbos_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Fetch the original review job
            cur.execute("SELECT thread_id, payload FROM agent_execution_queue WHERE job_id = %s", [job_id])
            row = cur.fetchone()
            if not row:
                return False
                
            # Update the review job
            cur.execute(
                "UPDATE agent_execution_queue SET status = %s, completed_at_ms = %s WHERE job_id = %s",
                [new_status, int(datetime.now(timezone.utc).timestamp() * 1000), job_id]
            )
            
            # If approved, spawn an execution job for the headless worker
            if new_status == "approved":
                new_job_id = str(uuid.uuid4())
                payload = row["payload"] or {}
                instructions = payload.get("instructions", "No instructions provided.") if isinstance(payload, dict) else str(payload)
                
                exec_payload = {
                    "task": "execute_approved_plan",
                    "instructions": instructions,
                    "context": {
                        "approved_from_review": job_id
                    }
                }
                
                cur.execute(
                    """
                    INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, status, started_at_ms)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    """,
                    [new_job_id, row["thread_id"], 'execution', json.dumps(exec_payload), int(datetime.now(timezone.utc).timestamp() * 1000)]
                )
                
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[Swarm] Failed to update review {job_id}: {e}")
        return False

def execute_sandbox_test(command: str, cwd: str = "~") -> dict:
    """Submit a shell command to the DBOS queue to be executed on the kruschgame sandbox."""
    try:
        import uuid
        import time
        import json
        import re
        import shlex

        forbidden_re = re.compile(r"\b(sudo|rm\s+-rf|mkfs|dd\s+if=|shutdown|reboot|rsync\s+[^\n]*--delete)\b", re.IGNORECASE)
        if forbidden_re.search(command or "") or forbidden_re.search(cwd or ""):
            return {"error": "Command rejected by Jean SRE compliance gate (forbidden destructive pattern)."}

        job_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        safe_cwd = "~" if cwd == "~" else shlex.quote(cwd)

        payload = {
            "task": "execute_sandbox_test",
            "instructions": f"cd {safe_cwd} && {command}",
            "context": {
                "dispatched_by": "hermes_mcp",
                "target_node": "kruschgame"
            }
        }
        
        conn = _get_dbos_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_execution_queue (job_id, thread_id, job_type, payload, status, started_at_ms)
                VALUES (%s, %s, %s, %s, 'pending', %s)
                """,
                [job_id, thread_id, 'intent', json.dumps(payload), int(datetime.now(timezone.utc).timestamp() * 1000)]
            )
        conn.commit()
        
        # Poll for completion up to 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT status, result FROM agent_execution_queue WHERE job_id = %s", [job_id])
                row = cur.fetchone()
                
            if row:
                if row['status'] == 'completed':
                    conn.close()
                    return {"status": "completed", "result": row['result']}
                elif row['status'] == 'failed':
                    conn.close()
                    return {"status": "failed", "error": row['result']}
                    
            time.sleep(2)
            
        conn.close()
        return {"error": "Job timed out after 60 seconds."}
        
    except Exception as e:
        logger.error(f"[Swarm] Failed to execute sandbox test: {e}")
        return {"error": str(e)}
