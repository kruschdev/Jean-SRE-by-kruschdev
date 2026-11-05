---
base_model: google/gemma-4-31b-it
library_name: peft
model_name: gemma4-31b-sre-lora
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - krush420/sovereign-code-audit-sft
  - krush420/sovereign-g4-benchmarks
model-index:
  - name: gemma4-31b-sre-lora
    results:
      - task:
          type: text-generation
          name: Multi-Turn Long-Term Memory & Simulated App Tool Execution (#1 Open-Weights)
        dataset:
          name: DolphinBench (600 Tasks across 3 Personas / 1.5M Tokens, Official gpt-5.6-sol Judge)
          type: mem0ai/dolphinbench
          split: test
        metrics:
          - type: accuracy
            value: 57.33
            name: Official gpt-5.6-sol Pass Rate (344/600 %, #1 Open-Weights @ $0.00)
          - type: accuracy
            value: 61.22
            name: Official Conjunctive Mechanical Checks (835/1,364 %)
          - type: accuracy
            value: 63.67
            name: Invoked gpt-5.6-sol Semantic Judge Checks (354/556 %)
        source:
          name: Sovereign G4 & DolphinBench Hermetic Audit Logs
          url: https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks
      - task:
          type: text-generation
          name: Native <|channel>thought Reasoning, Partial-Briefing Gap-Search & SRE Tool-Calling SFT + DPO
        dataset:
          name: Sovereign Code Audit, Lego-First & Agentic Tool-Calling SFT + DPO
          type: krush420/sovereign-code-audit-sft
          split: agentic_tool_calling_sft
        metrics:
          - type: accuracy
            value: 81.56
            name: Stage-1 Assistant-Only SFT Token Accuracy (%)
          - type: accuracy
            value: 100.0
            name: Stage-2 BF16 Ref-Anchored DPO Preference Accuracy (%)
        source:
          name: Sovereign Code Audit SFT & DPO Distillation Pipeline
          url: https://huggingface.co/datasets/krush420/sovereign-code-audit-sft
tags:
  - base_model:adapter:google/gemma-4-31b-it
  - gemma4
  - krusch-stack
  - mcp
  - agentic-harness
  - tool-calling
  - reasoning
  - dolphinbench
  - lora
  - qlora
  - sft
  - dpo
  - code-audit
  - sre
  - ollama
  - gguf
  - transformers
  - trl
---

# Google Gemma 4 31B Dense — KruschStack Wave-6 Agentic Reasoning, Memory & SRE LoRA (`#1 Open-Weights` on Official DolphinBench)

[![KruschStack](https://img.shields.io/badge/Powered%20By-KruschStack-cyan.svg)](https://github.com/kruschdev/krusch-stack)
[![MCP Compatible](https://img.shields.io/badge/MCP-39%20Tools-blueviolet.svg)](https://github.com/kruschdev/krusch-stack)
[![DolphinBench #1 Open-Weights](https://img.shields.io/badge/DolphinBench%20600-57.33%25%20(344%2F600%20%231%20Open--Weights)-brightgreen.svg)](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks)
[![Base Model](https://img.shields.io/badge/Base-google%2Fgemma--4--31b--it-blue.svg)](https://huggingface.co/google/gemma-4-31b-it)

> **Trained Through [KruschStack](https://github.com/kruschdev/krusch-stack) to Benefit [KruschStack](https://github.com/kruschdev/krusch-stack)**
>
> `krush420/gemma4-31b-sre-lora` is the **Wave-6 Dense Thinking Agentic LoRA (`30.7B` `100%` active parameters, Apache-2.0)** for **[KruschStack (`krusch-stack`)](https://github.com/kruschdev/krusch-stack)** (`npx krusch-stack init`). Fine-tuning [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) with **assistant-only `<|channel>thought` loss masking**, **BF16 Reference-Adapter-Anchored `trl.DPOTrainer` preference calibration (`100.0%` accuracy)**, and a **Partial-Briefing Gap-Search curriculum**, and pairing at inference with **Krusch Grounded Harness (`v0.3`)** + `$0.00` Zero-LLM `SovereignMemoryStore`, it achieves **`57.33%` (`344/600` — `Morgan` `74.5%` [`149`], `Alex` `51.5%` [`103`], `Riley` `46.0%` [`92`]) at `$0.00` inference cost** under the official `openai/gpt-5.6-sol` (`reasoning_effort="medium"`) DolphinBench judge—setting the **`#1 Open-Weights` record (`+3.17 pp` / `+19` tasks over `54.17%` Stock `Gemma 4 31B`, beating all `8` non-`GPT-5.6-Luna` commercial cloud stacks up to `$1,830.57`, and lifting the `$0.00` 6-run `Gemma 4 31B Oracle@6` ceiling to `71.83%` [`431/600`] above Official `#1` Cloud `Mem0` `70.67%` @ `$96.21`)**.

---

## 🔗 The KruschStack & Jean SRE Ecosystem

- **🛠️ KruschStack (7-Pillar 39-Tool MCP Stack)**: [`github.com/kruschdev/krusch-stack`](https://github.com/kruschdev/krusch-stack) (`npx krusch-stack init`)
- **🤖 Jean SRE by Kruschdev (24/7 Autonomous Auditor & Distillation Harness)**: [`github.com/kruschdev/Jean-SRE-by-kruschdev`](https://github.com/kruschdev/Jean-SRE-by-kruschdev)
- **📊 Training Dataset (`18,131+` Zero-Leak SFT + DPO Records)**: [`krush420/sovereign-code-audit-sft`](https://huggingface.co/datasets/krush420/sovereign-code-audit-sft)
- **🏆 Hermetic Benchmark Audit Logs (`75/75` Gates, `49` Audit Logs = `48` Staged Raw JSON + `1` PR #169 Live Receipt across `333,000+` Queries/Traces, incl. `27` Official DolphinBench `600` Runs / `16,200` Traces)**: [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks) · [`krush420/sovereign-router-arena`](https://huggingface.co/spaces/krush420/sovereign-router-arena)
- **⚡ Companion 30B MoE Coding Specialist (`L2` Local Tier, `3.3B` Active)**: [`krush420/qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (`43.00%` [`258/600`] w/ Grounded v0.3)
- **🧠 Companion 120B MXFP4 Teacher (`L2.5` Referee Tier, `5.1B` Active)**: [`krush420/gpt-oss-120b-sre-lora`](https://huggingface.co/krush420/gpt-oss-120b-sre-lora) (`39.50%` [`237/600`], `92.37%` BEAM `100K` Recall)
- **🪶 Companion 8B Edge Agent (`L1.5` Tier)**: [`krush420/llama3-hermes3-8b-sre-lora`](https://huggingface.co/krush420/llama3-hermes3-8b-sre-lora) (`18.33%` [`110/600`], `15.7x` over Stock `8B`)

---

## ⚡ Quickstart

### 1. Run Locally with Ollama (`Modelfile`, `adapter.gguf` & Offline GGUF Fusion Included)

This repository includes pre-converted GGUF v3 LoRA tensors (`adapter.gguf`, `171.7 MiB`, `230` tensor pairs across all 60 `language_model.layers` attention projections) plus `fuse_lora_into_gguf.py` to bake the rank-16 adapter directly into Ollama's `Q4_K_M` base GGUF weights with **zero runtime GEMV overhead** while preserving Ollama's native **`RENDERER gemma4`** and **`PARSER gemma4`** (`<|channel>thought` reasoning + structured `<|tool_call>` parsing):

```bash
git clone https://huggingface.co/krush420/gemma4-31b-sre-lora
cd gemma4-31b-sre-lora
ollama pull gemma4:31b

# Option A (Recommended — keeps stock gemma4:31b untouched):
BASE_BLOB="$(ollama show --modelfile gemma4:31b | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py \
  --base-gguf "$BASE_BLOB" \
  --adapter-gguf ./adapter.gguf \
  --output-gguf ./gemma4-31b-sre.gguf
ollama create gemma4-31b-sre:latest -f Modelfile && rm -f ./gemma4-31b-sre.gguf
ollama run gemma4-31b-sre:latest
```

<details>
<summary><b>Option B: Zero-Extra-Disk In-Place Fusion (for Ephemeral Cloud GPU VMs)</b></summary>

On disk-constrained ephemeral GPU instances where `gemma4:31b` (`~19 GB`) is freshly pulled for a single workload, `fuse_lora_into_gguf.py` supports atomic temp-file in-place replacement directly into `"$BASE_BLOB"`:

```bash
BASE_BLOB="$(ollama show --modelfile gemma4:31b | awk '/^FROM / {print $2}')"
python3 fuse_lora_into_gguf.py --base-gguf "$BASE_BLOB" --adapter-gguf ./adapter.gguf --output-gguf "$BASE_BLOB"
ollama run gemma4:31b
```
</details>

> **Important Ollama `Modelfile` Rule for Gemma 4**: Always retain `TEMPLATE {{ .Prompt }}`, `RENDERER gemma4`, and `PARSER gemma4` (included in [`Modelfile`](./Modelfile) with `num_ctx 32768`, `top_k 64`, `top_p 0.95`, `temperature 0.2`). Replacing Gemma 4's native renderer with a manual ChatML template disables Ollama's built-in `<|channel>thought\n...\n<|thought|>` reasoning extractor and `<|tool_call>` parser.

### 2. Transformers + PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "google/gemma-4-31b-it"
adapter_id = "krush420/gemma4-31b-sre-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

### 3. vLLM Multi-LoRA Serving

```bash
vllm serve google/gemma-4-31b-it \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules gemma4-sre=krush420/gemma4-31b-sre-lora
```

---

## 🔬 What Wave-6 Solves (Root-Cause Engineering Over Wave-4 & Wave-5)

Across six controlled `600`-task Official `DolphinBench` (`openai/gpt-5.6-sol`) runs on identical frozen `0o444` SQLite stores, three subtle training & harness bottlenecks separated naive QLoRA (`53.67%`, `-0.50 pp` below `54.17%` Stock) from Wave-6 (`57.33%`, **`+3.17 pp` above Stock**):

1. **Assistant-Only `<|channel>thought` Loss Masking (`Gemma4AssistantOnlyCollator`)**: Standard full-sequence causal LM collators spend `>80%` of gradient norm predicting Turn-0 RAG context and JSON tool schemas, attenuating native `<|channel>thought` deliberation (`54.17% -> 53.67%` in Wave-4). Masking all `<|turn>system`, `<|turn>user`, and `<|turn>tool` spans to `-100` while training exclusively on `<|turn>model\n<|channel>thought...<|turn|>` lifted standalone QLoRA above Stock (`54.83%` in Wave-5).
2. **BF16 Reference-Adapter Anchoring in `trl` `1.17.1` (`verify_and_sync_trl_ref_adapter`)**: When `DPOTrainer` wraps an already-adapted `PeftModel` under 4-bit NF4 base quantization without an explicit `ref_adapter_name`, `disable_adapter()` falls back to the raw pre-SFT NF4 base model (`rewards/margins == 0.0` or unconstrained drift). Wave-6 explicitly clones all `460` Stage-1 SFT LoRA tensors into a frozen `torch.bfloat16` `"ref"` adapter (`max_diff_vs_sft = 0.0`, `ref_l2_norm = 35.285`), yielding clean KL-anchored preference convergence (`0.6931 -> 0.3578` loss, `+0.7520` reward margin, **`100.0%` preference accuracy**).
3. **Partial-Briefing Gap-Search Curriculum (`180` Targeted SFT + `66` DPO Pairs) + Grounded Harness v0.3**: Teaches the dense `30.7B` reasoner to audit Turn-0 memory briefings inside `<|channel>thought`, fire targeted entity/temporal `search_*` queries (`1.58x` search depth on `Alex` & `Morgan`) whenever Turn-0 context is incomplete or superseded, inspect current workspace state before mutating (`list_* -> update_*`), and respect explicit policy/privacy boundaries without over-refusing benign user requests.

### Wave-6 Training Hyperparameters & Metrics (`training_metrics.json`)

- **Hardware**: Single NVIDIA RTX PRO 6000 Blackwell Server Edition (`96 GB` VRAM, `sm_120`; peak allocated VRAM **`27.82 GB`**, `2,948.15s` total wall time)
- **Base Model**: `google/gemma-4-31b-it` (`30.7B` `100%` active dense parameters across 60 `language_model.layers` hybrid sliding + full `self_attn`)
- **LoRA Config**: `r=16`, `lora_alpha=16` (`alpha/r = 1.0`, conservative rank-1 scaling preserving pretrained dense attention geometry), `lora_dropout=0.05`, regex `.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$` (`85.9 MiB` SafeTensors / `171.7 MiB` GGUF v3 adapter)
- **Stage-1 Assistant-Only SFT (`8192` ctx, `460` trajectories, `lr=5e-5`, `1,196.91s`)**: Loss converged `2.1885 -> 0.9788` (`train_loss=1.2479`), assistant-only token accuracy reached **`81.56%`**
- **Stage-2 BF16 Ref-Anchored DPO (`114` pairs x `2` epochs, `beta=0.2`, `lr=5e-6`, `1,141.52s`)**: `ref_adapter_anchored: true` (`460` synced BF16 tensors, `max_diff_vs_sft: 0.0`), loss `0.6931 -> 0.3869` (`train_loss=0.3578`), token accuracy `80.49% -> 86.80%`, reward margin `0.000 -> +0.7520`, **`100.0%` preference accuracy**

---

## 📊 Complete 6-Run `Gemma 4 31B` Progression on Official `DolphinBench` (`3,600` Traces, `$0.00` Inference)

All 6 configurations read the **byte-identical `0o444` frozen `SovereignMemoryStore` SQLite files** (`13,539` sessions / `1.50M` tokens, `$0.00` zero-LLM ingestion) and are graded across all `600/600` tasks by the official `openai/gpt-5.6-sol` (`reasoning_effort="medium"`) judge:

| `#` | `Gemma 4 31B` Configuration (`Q4_K_M` Ollama Runtime, `$0.00`) | Official `gpt-5.6-sol` Overall (`600`) | `Morgan` (`200`) | `Alex` (`200`) | `Riley` (`200`) | Conjunctive Mech Checks | Invoked Judge Checks | Archive SHA-256 (`5/5` Gates) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`gemma4-31b-sre` Wave-6 QLoRA+DPO + `Grounded Harness v0.3`** | **`57.33%` (`344`)** 🏆 | **`74.5%` (`149`)** | **`51.5%` (`103`)** | `46.0%` (`92`) | **`835 / 1,364` (`61.22%`)** | **`354 / 556` (`63.67%`)** | `b7a26ef96ae99a93...` |
| **2** | `gemma4:31b` Stock (Thinking ON) + `Grounded Harness v0.2` | `56.67%` (`340`) | `73.0%` (`146`) | `48.5%` (`97`) | **`48.5%` (`97`)** | `808 / 1,363` (`59.28%`) | `349 / 557` (`62.66%`) | `eab600888c276d58...` |
| **3** | `gemma4-31b-sre` Wave-5 QLoRA+DPO (Assistant-Mask Fix, `Std.`) | `54.83%` (`329`) | `71.5%` (`143`) | `47.0%` (`94`) | `46.0%` (`92`) | `794 / 1,361` (`58.34%`) | `340 / 559` (`60.82%`) | `5b1a49b1885f1e62...` |
| **4** | `gemma4:31b` Stock (Thinking ON, `Std. Harness`, Queued Flagship) | `54.17%` (`325`) | `72.0%` (`144`) | `47.5%` (`95`) | `43.0%` (`86`) | `792 / 1,363` (`58.11%`) | `334 / 557` (`59.96%`) | `2a23c2a023473a66...` |
| **5** | `gemma4-31b-sre` Wave-4 QLoRA+DPO (Unmasked Full-Seq SFT, `Std.`) | `53.67%` (`322`) | `71.0%` (`142`) | `45.5%` (`91`) | `44.5%` (`89`) | `772 / 1,364` (`56.60%`) | `331 / 556` (`59.53%`) | `05e2d1061b5c1f0d...` |
| **6** | `gemma4:31b` Stock (Thinking OFF, `Std. Harness`) | `52.00%` (`312`) | `70.5%` (`141`) | `43.5%` (`87`) | `42.0%` (`84`) | `752 / 1,368` (`54.97%`) | `321 / 552` (`58.15%`) | `29d4b6d76bbf0abb...` |
| — | **`Oracle@6` `Gemma 4 31B` Family Union (`$0.00` Open-Weights)** | **`71.83%` (`431`)** | **`85.5%` (`171`)** | **`67.5%` (`135`)** | **`62.5%` (`125`)** | — | — | Beats `#1` Cloud `Mem0` (`70.67%` @ `$96.21`) |

### Architectural Takeaway: Dense `<|channel>thought` (`30.7B` Active) vs. Non-Thinking Sparse MoE (`3.05B` Active)

Testing the exact same **Wave-6 Gap-Search SFT + BF16 Ref-Anchored DPO** recipe across dense thinking **`Gemma 4 31B`** (`30.7B` active per token) and sparse non-thinking **`Qwen3-Coder-30B-A3B`** (`3.05B` active per token) revealed an important capacity-routing rule:
- **Dense Thinking `Gemma 4 31B` (`57.33%` [`344/600`])**: Holds multi-turn `search_*` results inside its `30.7B` `<|channel>thought` reasoning scratchpad (`+6.0 pp` on `Alex` `97 -> 103`, `+3.0 pp` on `Morgan` `146 -> 149`, `+27` conjunctive mechanical checks over Stock + Grounded).
- **Non-Thinking `3.05B`-Active MoE [`qwen3-coder-30b-sre-lora`](https://huggingface.co/krush420/qwen3-coder-30b-sre-lora) (`43.00%` [`258/600`])**: Without an internal reasoning scratchpad, aggressive multi-query memory expansion crowds `3B`-active per-token attention (`38.50%` on Wave-6 vs. **`43.00%` [`258/600`]** on concise Wave-4 + Grounded Harness v0.3). Route high-throughput local code/MCP execution (`124 tok/s`) to **Wave-4 `qwen3-coder-30b-sre-lora`**, and route multi-hop temporal contradiction resolution to **Wave-6 `gemma4-31b-sre-lora`**.

---

## 📜 License & Attribution

Distributed under the **Apache License 2.0** (`license: apache-2.0`), inheriting from [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it). Full SHA-256 benchmark provenance (`49` audit logs across `12` external suites, `27` DolphinBench `600`-task runs = `16,200` traces, `Oracle@27 = 97.83%`, `$0.00` `Oracle@17 = 83.17%`) is published in [`krush420/sovereign-g4-benchmarks`](https://huggingface.co/datasets/krush420/sovereign-g4-benchmarks).