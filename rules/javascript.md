---
description: JavaScript coding standards for all projects
globs: ["**/*.js", "**/*.mjs"]
alwaysApply: false
---

# JavaScript Standards

## Language & Syntax
- Use ES modules (import/export), never CommonJS (require)
- Use async/await, never .then() chains
- Prefer const over let, never use var
- Use template literals over string concatenation
- Use early returns to reduce nesting
- Keep functions small and focused (under 40 lines)
- Use meaningful variable names (no single letters except loop counters)

## Documentation
- Add JSDoc comments to all exported functions
- Always handle errors with try/catch and meaningful error messages

## Toolkit First
- **Always check `@krusch/toolkit`** before writing utility code inline
- Use `@krusch/toolkit/llm` for LLM calls (`chat`, `chatJson`)
- Use `@krusch/toolkit/config` for config loading (`envOr`, `envInt`, `envBool`)
- Use `@krusch/toolkit/scheduler` for cron jobs (`createScheduler`)
- Use `@krusch/toolkit/imagen` for image generation (`createImageGenerator`)
- Use `@krusch/toolkit/json-parse` for AI response parsing (`parseAIJson`)
- Use `@krusch/toolkit/db` for SQLite databases (`initDb`, `query`, `run`)

## Configuration Pattern
- Layer: `config.json` (IDE-editable defaults) → `.env` → env vars
- Always provide `.env.example` with commented sections
- Use `envOr(key, default)` pattern for environment fallbacks

## Logging
- Use emoji-coded console logging:
  - `✅` success, `❌` error, `⚠️` warning
  - `🧠` AI/LLM, `🎯` targeting/routing, `📝` data/transcription
  - `💾` persistence, `🔒` security/auth

## Common Mistakes

| ❌ Anti-Pattern | ✅ Correct |
|----------------|-----------|
| `const fs = require('fs')` | `import fs from 'fs'` (ESM only) |
| `fetch().then(res => ...)` | `const res = await fetch(...)` |
| `var count = 0` | `let count = 0` or `const count = 0` |
| Missing `try/catch` on async calls | Wrap all `await` calls, log errors with `❌` prefix |
| Writing a custom `parseJSON()` | Use `@krusch/toolkit/json-parse` `parseAIJson()` |
| `const port = process.env.PORT \|\| 3000` | `const port = envInt('PORT', 3000)` via toolkit/config |
