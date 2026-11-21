---
name: jean-self-healing-infra
description: Closed-loop GPU VRAM, container health, and OpenTelemetry trace self-healing workflow for Jean SRE
---

# 🐳 Infrastructure Self-Healing & Tuning Workflow (/jean-self-healing-infra)

The system-level reliability playbook. Code-named `/jean-self-healing-infra`, this advanced manual enables **Jean SRE** and fleet workers to dynamically poll hardware performance telemetry, analyze GPU VRAM allocations, optimize LLM weight pinnings, execute container bounces, and run OpenTelemetry latency optimizations.

---

## 🌀 Self-Healing Infrastructure Loop

```mermaid
flowchart TD
    A[1. GPU & Telemetry Sweeps] -->|Assess VRAM & Load| B[2. Pin Status Check]
    B -->|Determine Model Allocation| C[3. Dynamic Weight Pinning]
    C -->|Swap / Pin Inference Weights| D[4. Container Recovery]
    D -->|Bounce Crashing Daemons| E[5. Telemetry Trace Optimization]
    E -->|Analyze Latency Bottlenecks| F[6. Nugget Consolidation]
```

---

## ⚡ Step-by-Step Infrastructure Tuning Guide

### Phase 1: Hardware & GPU Telemetry Scans
Monitor host loads, temperatures, and physical memory footprint across active fleet nodes:
- **Command (System performance probe)**:
  // turbo
  ```bash
  /home/krusch/homelab/scripts/check_nodes.sh
  ```
- **Command (NVIDIA GPU hardware check)**:
  // turbo
  ```bash
  ssh kruschdev "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free --format=csv,noheader"
  ```
  > [!IMPORTANT]
  > **Thermal Gating**: If any node GPU crosses `85°C`, instantly halt intensive processing, invoke dynamic offloading (see `/jean-thermal-rescale`), and send a warning card to the human loop gate.

### Phase 2: VRAM Allocation Audits
Determine which LLM weights and `llama-server` containers are currently active in physical GPU VRAM across all 3 primary nodes, verify `100% GPU` (`0% CPU`) residency, and confirm `kruschdev:11434` has `0` loaded Ollama models:
- **Command (Model & container status check across all 5 GPUs)**:
  // turbo
  ```bash
  ollama ps && ssh kruschdev "docker ps --filter name=krusch-first-chair-inkling; ollama ps; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" && ssh kruschserv "docker ps --filter name=krusch-second-chair-qwen30b; ollama ps"
  ```

### Phase 3: 3-Chair All-VRAM Residency (`100% GPU`)
Maintain permanent VRAM residency and prevent model load/unload swaps across the 3-Chair 5-GPU Fleet:
1. **1st Chair `kruschdev:18080` (Dual RTX 2080 Ti NVLink)**: Verify `krusch-first-chair-inkling` container serving `inkling-2-mini:ud-iq4_xs-dspark` (`32,768` ctx, `38.0 tok/s`, `100% GPU`) as Jean SRE `PERSONA_MODEL` & `EXECUTOR_MODEL` (`provider: "three-chair"`), and ensure `ollama ps` on `kruschdev:11434` has `0` loaded models (`scripts/ensure_first_chair_inkling.sh --host kruschdev`).
2. **2nd Chair `kruschserv:18080` + `:11434` (Dual RTX 3060)**: Verify `krusch-second-chair-qwen30b` serving `qwen3-coder-30b-moe:latest` (`16,384` ctx, `82.9 tok/s`, `100% GPU`) on `:18080` + pin `snowflake-arctic-embed2:latest` (`8,192` ctx, `keep_alive: -1`) on `:11434`.
3. **3rd Chair `krusch:11434` (Single RTX 2080 Ti)**: Pin `gemma4:e2b` (`16,384` ctx, `108.4 tok/s`, `keep_alive: -1`) + `snowflake-arctic-embed2:latest` (`8,192` ctx, `1024d`, `keep_alive: -1`).

### Phase 4: Container Auto-Recovery (Stale/Looping services)
If a staging or production container is unresponsive or continuously failing health endpoint checks:
1. Identify the container name and source node.
2. Gracefully bounce the container using Docker:
   ```bash
   ssh $TARGET_NODE "docker restart $CONTAINER_NAME"
   ```
3. Verify status: `docker ps | grep $CONTAINER_NAME`.

### Phase 5: OpenTelemetry / HALO Trace Optimizations
For deep trace analysis, execute the HALO optimization suite to pinpoint latency bottlenecks, database transaction lockups, or thread blockages:
// turbo
```bash
bash /home/krusch/homelab/projects/halo/run_halo.sh --analyze-latency
```

### Phase 6: Nuggets Memory Persistence
Record successfully applied hardware parameters as permanent operational hints:
- Call `krusch_context_remember({ category: "invariant", key: "gpu-tuning", content: "Verified 3-Chair 5-GPU residency: inkling-2-mini:ud-iq4_xs-dspark on kruschdev:18080 (Dual 2080 Ti NVLink, 0 models on :11434), qwen3-coder-30b-moe:latest on kruschserv:18080 + snowflake-arctic-embed2:latest on :11434 (Dual 3060), and gemma4:e2b + snowflake-arctic-embed2:latest on krusch:11434 (2080 Ti) at 100% GPU." })`.

---

## 🛠️ Diagnostics & Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| GPU stats command fails | NVIDIA driver desync or host GPU offline | Restart nvidia drivers: `sudo systemctl restart nvidia-persistenced` or reboot target host node |
| Container fails to recover after restart | Port binding collision or corrupted volume lockfile | Force-kill container, remove lockfiles from staging volumes, and restart via `docker-compose up -d --force-recreate` |
| Trace analyzer cannot locate trace logs | OpenTelemetry daemon is not running on the target node | Verify collector status: `docker ps | grep otel-collector` and verify config bindings |

---

## 🔧 Explicit MCP Tool Playbook (Jean Autonomous Infra Sweep)

When executing autonomous infrastructure sweeps across the 3 primary compute nodes (`krusch`, `kruschdev`, `kruschserv`), use the exact Sentinel & Context MCP tool calls matching the active focus area:

- **`docker` focus**:
  - `get_docker_containers({"node": "<node>"})`
  - `get_service_logs({"node": "<node>", "service": "<container_or_unit>", "lines": 50})`
  - `get_docker_daemon_config({"node": "<node>"})`
- **`storage` focus**:
  - `get_disk_usage({"node": "<node>"})`
  - `get_mount_options({"node": "<node>"})`
  - `find_large_files({"node": "<node>", "path": "/mnt", "min_size_mb": 500})`
  - `krusch_fs_locate({"pattern": "*", "node": "<node>"})`
- **`gpu` focus**:
  - `get_gpu_stats({"node": "<node>"})`
  - `get_process_health({"node": "<node>"})`
- **`system_tuning` focus**:
  - `get_system_stats({"node": "<node>"})`
  - `get_sysctl_params({"node": "<node>"})`
  - `get_cpu_governor({"node": "<node>"})`
  - `get_network_tuning({"node": "<node>"})`
- **`services` focus**:
  - `list_systemd_services({"node": "<node>"})`
  - `get_service_logs({"node": "<node>", "service": "<service_name>", "lines": 50})`
- **`security_self_state` focus**:
  - `audit_agent_self_state({"node": "<node>"})`
  - `detect_memory_anomaly({"node": "<node>"})`
  - `verify_state_integrity({"node": "<node>"})`

