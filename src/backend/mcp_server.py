import os
import json
import httpx
import subprocess
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import CrossEncoder
from src.backend.swarm import get_swarm_jobs, get_swarm_stats, get_jean_swarm_stats, get_human_reviews, get_debate_thread, update_review_status, execute_sandbox_test

DATABASE_URL = os.getenv("DBOS_DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
OLLAMA_EMBED_HOST = os.getenv("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Initialize fastMCP
mcp = FastMCP("KruschRetrievalMCP")

# DB connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Lazy loading for the reranker so it doesn't block startup
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker

def get_embedding(query: str, mode: str = "query", domain: str = "code"):
    """Call Ollama for embedding with domain-aware query prefixing and legacy 404 fallback."""
    formatted_input = query
    if mode == "query":
        if EMBED_MODEL.startswith("bge-large") and domain != "code":
            prefix = "Represent this sentence for searching relevant passages: "
            if not query.startswith(prefix):
                formatted_input = prefix + query
        elif EMBED_MODEL.startswith("snowflake-arctic-embed2"):
            prefix = "query: "
            if not query.startswith(prefix):
                formatted_input = prefix + query
        elif EMBED_MODEL.startswith("qwen3-embedding"):
            if not query.startswith("Instruct: "):
                formatted_input = (
                    "Instruct: Given a technical code or architecture query, retrieve the relevant source code module\nQuery:"
                    + query
                )

    payload = {
        "model": EMBED_MODEL,
        "input": formatted_input,
        "truncate": True,
        "keep_alive": -1,
    }
    if EMBED_MODEL.startswith("qwen3-embedding"):
        payload["dimensions"] = 1024

    base_url = OLLAMA_EMBED_HOST.rstrip("/")
    resp = httpx.post(
        f"{base_url}/api/embed",
        json=payload,
        timeout=30.0
    )
    if resp.status_code == 404:
        resp = httpx.post(
            f"{base_url}/api/embeddings",
            json={
                "model": EMBED_MODEL,
                "prompt": formatted_input[:1150],
                "truncate": True,
                "keep_alive": -1,
            },
            timeout=30.0,
        )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings", [[]])
    if isinstance(embeddings, list) and embeddings and embeddings[0]:
        return embeddings[0]
    if isinstance(data.get("embedding"), list) and data["embedding"]:
        return data["embedding"]
    return None

@mcp.tool()
def krusch_context_search_code(query: str, project: str = None) -> str:
    """
    Search the Kruschdb knowledge base for code or documents using semantic vector search and cross-encoder reranking.
    """
    # 1. Embed query
    vector = get_embedding(query)
    if not vector:
        return json.dumps({"error": "Failed to embed query"})
    
    vec_str = '[' + ','.join(str(v) for v in vector) + ']'
    
    # 2. HNSW Search (Top 50)
    db = SessionLocal()
    try:
        if project:
            sql = text("""
                SELECT file_name, file_path, summary, content,
                       1 - (embedding <=> CAST(:vec AS vector)) AS cosine_sim
                FROM blobs
                WHERE embedding IS NOT NULL AND repository_id IN (SELECT id FROM repositories WHERE name = :project)
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT 50
            """)
            params = {"vec": vec_str, "project": project}
        else:
            sql = text("""
                SELECT file_name, file_path, summary, content,
                       1 - (embedding <=> CAST(:vec AS vector)) AS cosine_sim
                FROM blobs
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT 50
            """)
            params = {"vec": vec_str}
            
        rows = db.execute(sql, params).fetchall()
        
        if not rows:
            return json.dumps({"results": [], "message": "No results found."})
            
        # 3. Rerank top 50
        reranker = get_reranker()
        pairs = []
        doc_map = []
        for row in rows:
            content_str = ""
            if row.content:
                if isinstance(row.content, (bytes, memoryview)):
                    content_str = bytes(row.content).decode('utf-8', errors='replace')
                else:
                    content_str = str(row.content)
            
            # Text used for reranking (combine summary + snippet)
            # Reranker has 512 token context limit (~2000 chars)
            snippet = content_str[:1000] if content_str else ""
            summary_part = f"Summary: {row.summary}\n" if row.summary and row.summary != '[SKIP]' else ""
            text_for_rerank = f"{row.file_name or row.file_path}\n{summary_part}{snippet}"
            
            pairs.append([query, text_for_rerank])
            doc_map.append({
                "file_name": row.file_name,
                "file_path": row.file_path,
                "summary": row.summary,
                "content_preview": content_str[:2000] if content_str else ""
            })
            
        scores = reranker.predict(pairs)
        
        # Combine and sort by score
        scored_docs = list(zip(doc_map, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Return Top 5
        top_5 = scored_docs[:5]
        
        results = []
        for doc, score in top_5:
            results.append({
                "file_path": doc["file_path"],
                "file_name": doc["file_name"],
                "summary": doc["summary"],
                "content_preview": doc["content_preview"],
                "rerank_score": float(score)
            })
            
        return json.dumps({"results": results})
        
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()


# ─── Episodic Memory Tools (ide_agent_memory) ──────────────────

@mcp.tool()
def krusch_context_search_memory(category: str, query: str, project: str = None, limit: int = 5) -> str:
    """
    Semantic search over episodic memory (lessons, bugs, outcomes, priorities, activity).
    Uses bge-large embeddings + cosine similarity on kruschdb.ide_agent_memory.
    """
    vector = get_embedding(query)
    if not vector:
        return json.dumps({"error": "Failed to embed query"})

    vec_str = '[' + ','.join(str(v) for v in vector) + ']'
    db = SessionLocal()
    try:
        if project:
            sql = text("""
                SELECT id, content, tags, project, created_at,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM ide_agent_memory
                WHERE category = :category AND project = :project AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
            """)
            params = {"vec": vec_str, "category": category, "project": project, "limit": limit}
        else:
            sql = text("""
                SELECT id, content, tags, project, created_at,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM ide_agent_memory
                WHERE category = :category AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
            """)
            params = {"vec": vec_str, "category": category, "limit": limit}

        rows = db.execute(sql, params).fetchall()
        results = []
        for row in rows:
            results.append({
                "id": str(row.id),
                "content": row.content,
                "tags": row.tags,
                "project": row.project,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "similarity": round(float(row.similarity), 4)
            })
        return json.dumps({"results": results, "count": len(results)})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()


@mcp.tool()
def krusch_context_write_state(category: str, content: str, author_id: str = "jean", project: str = None, tags: str = None) -> str:
    """
    Write a new episodic memory state (lesson, bug, outcome, priority, activity).
    Embeds the content with bge-large and stores in kruschdb.ide_agent_memory.
    Valid categories: priorities, bugs, outcomes, lessons, activity.
    """
    valid_cats = {"priorities", "bugs", "outcomes", "lessons", "activity"}
    if category not in valid_cats:
        return json.dumps({"error": f"Invalid category '{category}'. Valid: {valid_cats}"})

    vector = get_embedding(content, mode="document")
    if not vector:
        return json.dumps({"error": "Failed to embed content"})

    vec_str = '[' + ','.join(str(v) for v in vector) + ']'
    tag_str = tags if tags else f"author:{author_id}"

    db = SessionLocal()
    try:
        sql = text("""
            INSERT INTO ide_agent_memory (id, category, content, project, tags, embedding, created_at)
            VALUES (gen_random_uuid(), :category, :content, :project, :tags, CAST(:vec AS vector), NOW())
            RETURNING id
        """)
        result = db.execute(sql, {
            "category": category,
            "content": content,
            "project": project,
            "tags": tag_str,
            "vec": vec_str
        })
        db.commit()
        new_id = str(result.fetchone()[0])
        return json.dumps({"status": "success", "id": new_id, "message": f"Memory saved to '{category}' (project: {project or 'global'})"})
    except Exception as e:
        db.rollback()
        return json.dumps({"error": str(e)})
    finally:
        db.close()


@mcp.tool()
def krusch_context_list_memories(category: str, project: str = None, limit: int = 10) -> str:
    """
    List recent memories chronologically (newest first). No embedding needed.
    """
    db = SessionLocal()
    try:
        if project:
            sql = text("""
                SELECT id, content, tags, created_at
                FROM ide_agent_memory
                WHERE category = :category AND project = :project
                ORDER BY created_at DESC LIMIT :limit
            """)
            params = {"category": category, "project": project, "limit": limit}
        else:
            sql = text("""
                SELECT id, content, tags, project, created_at
                FROM ide_agent_memory
                WHERE category = :category
                ORDER BY created_at DESC LIMIT :limit
            """)
            params = {"category": category, "limit": limit}

        rows = db.execute(sql, params).fetchall()
        results = []
        for row in rows:
            entry = {
                "id": str(row.id),
                "content": row.content,
                "tags": row.tags,
                "created_at": row.created_at.isoformat() if row.created_at else None
            }
            if hasattr(row, 'project'):
                entry["project"] = row.project
            results.append(entry)
        return json.dumps({"results": results, "count": len(results)})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()

@mcp.tool()
def krusch_context_get_swarm_stats() -> str:
    """Get aggregate statistics for the DBOS agent execution queue."""
    return json.dumps(get_swarm_stats())

@mcp.tool()
def krusch_context_get_jean_swarm_stats() -> str:
    """Get swarm stats broken down by job_type for Jean SRE. Separates intent jobs from debate activity."""
    return json.dumps(get_jean_swarm_stats())

@mcp.tool()
def krusch_context_get_swarm_jobs(status: str = "pending", limit: int = 10) -> str:
    """Fetch jobs from the DBOS agent execution queue (e.g., status='pending', 'running', or 'completed')."""
    return json.dumps(get_swarm_jobs(status=status, limit=limit))

@mcp.tool()
def krusch_context_get_human_reviews(limit: int = 20) -> str:
    """Fetch pending human review items (synthesized prototype plans)."""
    return json.dumps(get_human_reviews(limit=limit))

@mcp.tool()
def krusch_context_get_debate_thread(thread_id: str) -> str:
    """Fetch all jobs in a debate thread to show the full Idea -> Critique -> Synthesis chain."""
    return json.dumps(get_debate_thread(thread_id))

@mcp.tool()
def krusch_context_update_review_status(job_id: str, new_status: str) -> str:
    """Mark a human review as 'approved' or 'rejected'. If approved, the swarm execution job will be dispatched."""
    success = update_review_status(job_id, new_status)
    if success:
        return json.dumps({"status": "success", "message": f"Job {job_id} successfully marked as {new_status}."})
    else:
        return json.dumps({"status": "error", "message": f"Failed to update job {job_id}. Verify the ID is correct and it is a pending review."})

@mcp.tool()
def krusch_context_execute_on_sandbox(command: str, cwd: str = "~") -> str:
    """Submit a shell command to the DBOS queue to be executed on the kruschgame sandbox."""
    return json.dumps(execute_sandbox_test(command, cwd))

if __name__ == "__main__":
    mcp.run(transport='stdio')
