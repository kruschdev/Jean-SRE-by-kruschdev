# Install `jean-sre` (`kruschdev/Jean-SRE-by-kruschdev`)

These are instructions for a coding agent (Claude Code, Codex, Cursor, Windsurf, or Antigravity) that a person asked to install and configure `jean-sre` (`kruschdev/Jean-SRE-by-kruschdev`) on their machine.

`jean-sre` is a Node.js `>= 22.0.0` autonomous 11-phase Principal Engineer code auditor, test-gated bug remediator, GGUF quantization sanity checker (`quantcheck`), and Teacher-Student QLoRA + DPO distillation harness (`PORT=3005`) built to drive [`kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack).

## 1. Check Prerequisites First (All at Once)

Run a single read-only check before making any changes:

```bash
id -u
node -v 2>/dev/null || echo "MISSING: node"
npm -v 2>/dev/null || echo "MISSING: npm"
npx --version 2>/dev/null || echo "MISSING: npx"
git --version 2>/dev/null || echo "MISSING: git"
docker compose version 2>/dev/null || echo "OPTIONAL_MISSING: docker-compose"
ollama --version 2>/dev/null || echo "OPTIONAL_MISSING: ollama"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "OPTIONAL_MISSING: nvidia-gpu"
```

Evaluate the results:
1. **User check**: If `id -u` prints `0` (`root`), stop and ask the person to run you under a standard non-root developer account.
2. **Node.js `>= 22.0.0` (Required)**: Needed for native `node:sqlite`, `node --test`, and `krusch-stack` integration. If `node -v` is missing or below `v22.0.0`, ask the person before switching or installing Node 22 (`nvm install 22` or `fnm use 22`).
3. **Docker, Ollama & GPU VRAM (Optional)**: None of these are required for Mode A (`npm test` runs `67` hermetic unit, security compliance, `state-graph`, `quantcheck`, and distillation tests in ~3 seconds with zero GPU or Docker). Only use Mode B if the person explicitly asks to run the live `:3005` background daemon and local GPU endpoints.

## 2. Choose the Right Setup Mode

- **Mode A: Zero-Docker Local Verification & Audit Harness (Recommended Default)**: Initializes `krusch-stack` MCP tools (`npx -y krusch-stack init`), clones `https://github.com/kruschdev/Jean-SRE-by-kruschdev.git`, installs Node dependencies, creates `.env` non-destructively, and runs all `67` hermetic tests (`npm test`) plus `npm run quantcheck -- --help`.
- **Mode B: Full Live Daemon on `:3005` + 3-Chair Bandit or Single-Node Ollama (Only if requested)**: Connects `jean-sre` on port `3005` to local OpenAI/Ollama GPU endpoints (`provider: "three-chair"` cascading across `gemma4:e2b`, `qwen3-coder-30b-moe:latest`, and `inkling-2-mini:ud-iq4_xs-dspark`, or single-node `ollama` with the published `@krush420` Hugging Face LoRA adapters) plus optional `HydraDB` (`HYDRADB_BOLT_URL`), and verifies `http://127.0.0.1:3005/health` and `http://127.0.0.1:3005/state`.

## 3. Install & Verify

### Mode A: Zero-Docker Local Verification & Audit Harness (Default)

1. **Initialize the companion 39-tool `krusch-stack` MCP spine** in the target workspace (if not already initialized):
   ```bash
   npx -y krusch-stack init
   npx -y krusch-stack doctor
   ```

2. **Clone `kruschdev/Jean-SRE-by-kruschdev` and install dependencies**:
   ```bash
   git clone https://github.com/kruschdev/Jean-SRE-by-kruschdev.git jean-sre
   cd jean-sre
   npm install
   ```
   If cloned standalone outside a parent monorepo where `@krusch/toolkit` is not already present in a parent `node_modules`, install the runtime peers locally without mutating `package.json`:
   ```bash
   node -e "import('@krusch/toolkit/llm').then(()=>process.exit(0)).catch(()=>process.exit(1))" 2>/dev/null || \
     PUPPETEER_SKIP_DOWNLOAD=true npm install --no-save github:kruschdev/toolkit @modelcontextprotocol/sdk express pg
   ```

3. **Create `.env` non-destructively and run the 67-test hermetic verification suite**:
   ```bash
   [ -f .env ] || cp .env.example .env
   npm test
   ```
   Confirm all `67` tests across `tests/*.test.js` and `src/sweep.test.js` pass (`0` failures).

### Mode B: Full Docker Compose or Local Daemon on `:3005` + GPU Endpoints (Only if Requested)

Before pulling any multi-GB model or starting background services:
1. Check free disk space (`df -h .`) and confirm with the person which model tier fits their hardware (`gemma4:e2b` or `llama3-hermes3-8b-sre-lora` for 8-12 GB GPUs, `qwen3-coder-30b-moe:latest` / `krush420/qwen3-coder-30b-sre-lora` or `inkling-2-mini:ud-iq4_xs-dspark` for 22 to 24 GB+ VRAM, or `krush420/gemma4-31b-sre-lora` / `krush420/gpt-oss-120b-sre-lora` on high-VRAM workstations). Remind the person to store `OLLAMA_MODELS`, GGUF weights, and Docker `data-root` on dedicated storage volumes rather than small OS boot partitions.
2. Check that port `3005` is free (`ss -ltnp | grep ':3005\b' || true`).
3. Update `.env` (`[ -f .env ] || cp .env.example .env`) so `PERSONA_PROVIDER` (`three-chair` or `ollama`), `PERSONA_OLLAMA_URL`, `PERSONA_MODEL` (default `inkling-2-mini:ud-iq4_xs-dspark`), `OLLAMA_THIRD_CHAIR_URL`, `LLAMA_SECOND_CHAIR_URL`, `LLAMA_FIRST_CHAIR_URL`, `HYDRADB_BOLT_URL`, `HOMELAB_PATH`, and `DATABASE_URL` point to the person's local endpoints.
4. Start the `:3005` harness and verify health:
   - **Direct Node process (works in any standalone `jean-sre` checkout)**:
     ```bash
     npm start &
     curl -fsS http://127.0.0.1:3005/health
     ```
   - **Docker Compose (`docker-compose.yml`, when running inside the full `HOMELAB_PATH` monorepo layout where `context: ../..` resolves `lib/` and `lib-py/`)**:
     ```bash
     docker compose up -d
     curl -fsS http://127.0.0.1:3005/health
     ```

## Never

- **Never point autonomous write sweeps (`npm run sweep` or `jean-sweep`) at dirty repositories with uncommitted work** without asking the person first (`git status` must be clean or backed by staged 2PC review).
- **Never pull multi-tens-of-GB GGUF or Ollama checkpoints** (`30B`, `31B`, `120B`) without explicit confirmation from the person after checking GPU VRAM (`nvidia-smi`) and available disk space (`df -h`), and remind them to keep model weights and container volumes off small OS partitions.
- **Never run `npm` or `npx` with `sudo` or as `root`**.
- **Never overwrite an existing `.env` or existing IDE MCP config files**; preserve user customizations and merge JSON keys non-destructively.

## Tell the Person After

Share a concise status report:
1. **Setup mode & paths**: Confirm where `jean-sre` was cloned and whether `krusch-stack doctor` passed.
2. **Hermetic verification**: Report the exact `npm test` result (`65/65` unit, CSO compliance, `state-graph`, `quantcheck`, and distillation tests passing).
3. **Daemon & model status (if Mode B was requested)**: Report `http://127.0.0.1:3005/health` and `http://127.0.0.1:3005/state` status plus which `PERSONA_PROVIDER` / `PERSONA_MODEL` endpoints are configured in `.env`.
4. **Safe next step**: Show how to inspect resident GGUF health (`npm run quantcheck`) or run a read-only audit pass once their target repo has a clean `git status`.
