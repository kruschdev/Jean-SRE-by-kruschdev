---
description: Project architecture standards derived from FTF and Pocket Lawyer
globs: ["**/package.json", "**/pyproject.toml", "**/Dockerfile*", "**/docker-compose*.yml", "**/Makefile"]
alwaysApply: false
---

# Project Architecture Standards

## Project Location
- All new projects **must** be created inside `projects/` (e.g. `projects/my-app/`)
- Never scaffold a project at the homelab repo root

## Every Project Must Have
- `AGENTS.md` — Universal project context: hazards, quick start, architecture. Per v1.1 spec, guidance accumulates up the directory tree; child files override ancestors on conflict
- `README.md` — Architecture overview + setup instructions
- `.env.example` — All env vars with comments, grouped by concern
- `config.json` — IDE-editable defaults (never secrets in production)
- `.gitignore` — node_modules, data/, *.db, .env (not .env.example)
- `GEMINI_INFLIGHT.md` — (optional) In-flight state for inter-session continuity. See [inflight template](../templates/GEMINI_INFLIGHT.md)

## Database
- PostgreSQL (`kruschdb`) as default for production data and ALL agentic/semantic memory.
- Semantic embeddings must use `pgvector` inside `kruschdb`.
- SQLite only for local development or simple tools (strictly forbidden for multi-agent semantic memory).
- Separate migration system (`db/migrate.js` or Alembic)
- Use connection pooling (pg Pool for Node, SQLAlchemy for Python)

## Docker
- Multi-stage Dockerfiles (build → production)
- `Makefile` with standard targets: `build`, `up`, `down`, `logs`, `shell`, `deploy`
- Follow container runtime standards in [infrastructure.md](infrastructure.md)

## Project Layout (Node.js)
```
├── AGENTS.md
├── README.md
├── config.json / config.js
├── .env.example
├── server.js
├── auth.js
├── db.js + db/pool.js + db/migrate.js
├── llm.js
├── Dockerfile
├── Makefile
├── client/          (React + Vite if frontend)
├── data/            (runtime, gitignored)
└── public/          (static assets)
```

## Project Layout (Python)
```
├── AGENTS.md
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example
├── config/
├── execution/       (modular by concern: ai/, legal/, core/)
├── services/        (microservices)
├── templates/       (persona-based: individual/, business/, shared/)
├── static/
├── Dockerfile
├── Makefile
└── tests/
```

## Multi-User Apps
- Persona/role-based route isolation (PL pattern)
- Separate templates per persona (`individual/`, `business/`, etc.)
- Tiered AI model selection: fast model for chat, pro for analysis
- JWT auth with httpOnly cookies (Node.js) or Flask-Login (Python)
