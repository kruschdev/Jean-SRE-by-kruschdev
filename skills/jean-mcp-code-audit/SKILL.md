---
name: jean-mcp-code-audit
description: Explicit 3-Tier MCP tool playbook and verification protocol for Jean SRE across all 11 Code Audit phases and Companion Mode
---

# 🧭 Jean SRE 3-Tier MCP Code Audit Skill (`/jean-mcp-code-audit`)

Authoritative MCP tool execution and context-grounding playbook for **Jean SRE** (`jean_sre_sweep.js` and `harness.js`). Every determination made by Jean MUST be grounded in full symbol implementations, caller graphs, and project invariants retrieved via the 3-Tier MCP ecosystem.

---

## 🛠️ Explicit MCP Tool Catalog & Exact Call Signatures

When generating `### INVESTIGATION PLAN` intents (or executing interactive tool calls), select the exact MCP tool for the target question:

### Tier 1: Project Invariants & Working Memory (`krusch-context-mcp`)
- **`krusch_context_retrieve({"query": "<project_or_topic>", "include_state": true, "limit_tokens": 2000})`**
  - *When to use*: Phase `0_ORIENTATION`, `1.2_COMPLIANCE_MATRIX`, and before flagging architectural violations. Retrieves active invariants (`INV-*`), recent architectural decisions, and open blockers.
- **`krusch_context_nudge({"trigger": "pre_commit", "code": "<code_snippet>", "file_path": "<rel_path>"})`**
  - *When to use*: Phase `1.2_COMPLIANCE_MATRIX` and `COMPANION_AUDIT` to check code snippets against recorded project invariants.

### Tier 2A: AST Symbol Graphs, Callers & Semantic Code Search (`krusch-git`)
- **`krusch_git_file_symbols({"repo": "<project>", "file_path": "<rel_path>"})`**
  - *When to use*: Before inspecting or judging any module. Lists every declared function, class, method, and route in `<rel_path>` with its exact `startLine-endLine` range, signature, and implementation preview.
- **`search_symbols <symbol_or_keyword>`** / **`krusch_git_search_symbols({"repo": "<project>", "query": "<symbol_or_keyword>"})`**
  - *When to use*: Locate symbol declarations across the repository and inspect their full function body (up to 50 lines hydrated inline) plus exact `file:startLine-endLine` coordinates.
- **`dependency_graph <symbol_or_file>`** / **`krusch_git_dependency_graph({"repo": "<project>", "symbol": "<symbol_or_file>"})`**
  - *When to use*: Phase `1.3_CRITICAL_PATH`, `1.4_BOUNDARY_AUDIT`, and `1.6_STRUCTURAL_STRESS_TEST`. Traces outbound imports, inbound file dependents, and exact call-site line numbers across the codebase.
- **`krusch_git_semantic_search({"repo": "<project>", "query": "<natural_language_or_pattern_query>"})`**
  - *When to use*: Phase `1.5_ANTI_PATTERN` and `1.8_RISK_MAP`. Hybrid BM25 + dense vector similarity search across repository code blobs.
- **`krusch_git_read_blob({"repo": "<project>", "file_path": "<rel_path>"})`** or **`read_file <rel_path>:<startLine>-<endLine>`**
  - *When to use*: Read the complete implementation of a specific function or line range (`180` lines per window). Always pass `:startLine-endLine` from `krusch_git_file_symbols` or `search_symbols` when inspecting files $> 180$ lines.

### Tier 2B: OS Filesystem, Configs & Bounded Page Memory (`krusch-fs`)
- **`krusch_fs_locate({"pattern": "<glob_or_ext>", "node": "kruschdev"})`**
  - *When to use*: Phase `0_ORIENTATION` and `1.7_VERIFICATION_AUDIT`. Fast metadata/glob lookup for configs, `.env.example`, systemd units, SQL migrations, or test suites without dumping file contents.
- **`krusch_fs_page_tree({"path": "<project_path>/<doc_or_code_file>", "node": "kruschdev"})`**
  - *When to use*: Phase `0_ORIENTATION`, `1.1_STANDARD_MAPPING`, and `1.6_STRUCTURAL_STRESS_TEST`. Returns the structural Table of Contents (`page_number`, `section_title`, `[start_line, end_line]`, `page_hash`, `memory_count`).
- **`krusch_fs_read_page({"path": "<project_path>/<doc_or_code_file>", "page_number": 1, "node": "kruschdev"})`**
  - *When to use*: Read an exact logical page `[start_line, end_line]` of a specification, markdown document, or large source file along with any pinned `page_memories`.
- **`krusch_fs_search({"query": "<config_or_doc_query>", "node": "kruschdev"})`**
  - *When to use*: Hybrid BM25 + vector search across indexed project documentation, specs, and configuration files.

### Tier 2C: Frontend Design & Layout Verification (`krusch-frontend-mcp`)
- **`query_design_rules({"queryStr": "<ui_component_or_rule>"})`**
- **`analyze_layout_structure({"htmlString": "<html_or_jsx_snippet>"})`**
- **`list_canonical_templates({})`** / **`get_canonical_template({"templateKey": "article-shell"})`**
  - *When to use*: Auditing UI/React/HTML projects (`krusch-companion`, `krusch-dev-site`, `krusch-dashboard`, `krusch-law`, `krusch-biz`).

---

## 📋 Phase-by-Phase Mandatory MCP Tool Matrix

| Audit Phase | Required Primary MCP Tools | Mandatory Context Verification Rule |
| :--- | :--- | :--- |
| **`0_ORIENTATION`** | `krusch_context_retrieve`, `krusch_fs_page_tree`, `krusch_git_file_symbols` | Read `AGENTS.md`, `spec.md`, `package.json`/`pyproject.toml`, and top entrypoint symbols before summarizing architecture. |
| **`1.1_STANDARD_MAPPING`** | `krusch_fs_locate`, `krusch_git_search_symbols`, `read_file <path>:<start>-<end>` | Compare explicit config/linter rules against actual AST symbol & import conventions in `src/` or `server/`. |
| **`1.2_COMPLIANCE_MATRIX`** | `krusch_context_retrieve`, `krusch_git_search_symbols`, `read_file <path>:<start>-<end>` | Verify Letter (Pass A), Spirit (Pass B), and Project Purpose (Pass C) with verbatim `file:line` quotes. |
| **`1.3_CRITICAL_PATH`** | `dependency_graph <entry_symbol>`, `krusch_git_file_symbols`, `read_file <path>:<start>-<end>` | Trace the primary transaction across every hop using `dependency_graph` and read the full `[startLine-endLine]` of each hop. |
| **`1.4_BOUNDARY_AUDIT`** | `dependency_graph <module_path>`, `krusch_git_file_symbols` | Inspect inbound/outbound imports across module boundaries; verify whether shared utilities leak app-specific state. |
| **`1.5_ANTI_PATTERN`** | `krusch_git_semantic_search`, `search_symbols`, `read_file <path>:<start>-<end>` | If a search hit or grep snippet looks suspicious, ALWAYS read the full function `startLine-endLine` before flagging. |
| **`1.6_STRUCTURAL_STRESS_TEST`** | `krusch_git_file_symbols`, `dependency_graph <file_path>`, `krusch_fs_page_tree` | Use AST symbol count, inbound/outbound coupling, and section structure to identify files with $>3$ distinct responsibilities. |
| **`1.7_VERIFICATION_AUDIT`** | `krusch_fs_locate({"pattern": "*test*"})`, `krusch_git_file_symbols`, `run_shell_command` | Inspect actual test files and run read-only test discovery/execution (`npm test` / `pytest`) to verify critical path coverage. |
| **`1.8_RISK_MAP`** | `krusch_git_semantic_search`, `dependency_graph`, `read_file <path>:<start>-<end>` | Trace concurrency, transaction handling, timeouts, and trust boundaries across full function bodies. |
| **`1.9_ADVERSARIAL_REVIEW`** | `read_file <path>:<start>-<end>`, `dependency_graph <symbol>` | Re-read the exact `file:startLine-endLine` of your top 3 findings and check all callers via `dependency_graph` to steel-man the status quo and eliminate false positives. |
| **`2_MASTER_BACKLOG`** | *(Synthesis from verified Phase 0–1.9 checkpoints)* | Include ONLY findings verified by complete function bodies and exact `file:line` citations. Drop any refuted or unverified claims. |

---

## 🛡️ Zero-False-Positive Determination Rules

1. **Never Diagnose From a Truncated Preview**: If an output states `[showing lines X-Y of N]` or `[symbol body continues to line Z]`, you MUST call `read_file <path>:<startLine>-<endLine>` or `krusch_git_file_symbols` before claiming code is missing, unused, or a stub.
2. **Verify Call Sites Before Claiming Dead Code or Missing Validation**: Always run `dependency_graph <symbol>` to inspect how callers invoke a function and whether validation/sanitization happens at the caller boundary.
3. **Ground Truth Overrides Prior `JEAN_INFLIGHT`**: Live source code shown in `Executed Investigation Outputs` or `Live Source Verification for Mentioned Symbols/Files` always overrides any hypothesis carried over from an earlier `JEAN_INFLIGHT` block.
