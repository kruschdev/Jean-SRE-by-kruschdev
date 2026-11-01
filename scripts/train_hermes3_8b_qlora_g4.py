#!/usr/bin/env python3
"""
Unified Agentic SRE + Multi-Turn Tool-Caller LoRA + DPO Pipeline for NousResearch/Hermes-3-Llama-3.1-8B
Target Accelerator: Colab G4 96GB RTX PRO 6000 Blackwell (native BF16 LoRA across all 7 Llama-3.1 projections)

Why this exists:
1. Trains our fast 8.0B Hermes-3 agent (`hermes3:latest` on local 24GB GPU nodes) on the exact
   Wave-4 Unified SRE + Multi-Turn Tool-Calling curriculum (Referee CLEAN_PASS precision, P0/P1 recall,
   Lego-First `@krusch/toolkit` synthesis, multi-turn MCP/Workspace tool chains, and temporal state overrides).
2. Renders SFT & DPO trajectories with the exact fixed Ollama Hermes-3 ChatML + `<tools>` / `<tool_call>`
   template (zero train/serve skew) and emits a fixed role-neutral `Modelfile` that resolves stock Ollama's
   `{{- else if .System }}` bug (which previously dropped `.System` whenever `.Tools` were passed).
"""

import argparse
import gc
import inspect
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_BASE_MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"

HERMES3_TOOLS_PREAMBLE = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools></tools> XML tags. You may call one or more functions to assist with the user query. "
    "Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>"
)

HERMES3_TOOLS_EPILOGUE = (
    '  </tools> Use the following pydantic model json schema for each tool call you will make: '
    '{"properties": {"arguments": {"title": "Arguments", "type": "object"}, '
    '"name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], '
    '"title": "FunctionCall", "type": "object"} '
    "For each function call return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags as follows:\n"
    "<tool_call>\n"
    '{"arguments": <args-dict>, "name": <function-name>}\n'
    "</tool_call>"
)

FIXED_HERMES3_OLLAMA_TEMPLATE = '''{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Agentic SRE + Multi-Turn Tool-Caller LoRA + DPO trainer for NousResearch/Hermes-3-Llama-3.1-8B."
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "sft_dpo"],
        default="sft_dpo",
        help="Training stage: 'sft', 'dpo', or 'sft_dpo' (Stage-1 SFT -> Stage-2 KL-anchored DPO).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model ID for Hermes-3-Llama-3.1-8B.",
    )
    parser.add_argument(
        "--dataset",
        default="data/distillation/gpt_oss_120b_elitist_teacher.jsonl",
        help="Path to curated ChatML JSONL dataset captured from gpt-oss:120b.",
    )
    parser.add_argument(
        "--argon-dataset",
        default="data/argon_remediation_sft_corpus.jsonl",
        help="Optional path to Gemini 4 Argon verified Golden Remediation JSONL dataset (oversampled 2x).",
    )
    parser.add_argument(
        "--calibration-dataset",
        default="data/distillation/lego_and_referee_calibration_sft.jsonl",
        help="Pinned CLEAN_PASS + P0/P1 Remediation + Lego-First ChatML SFT split (never dropped).",
    )
    parser.add_argument(
        "--calibration-oversample",
        type=int,
        default=3,
        help="Oversample factor for the pinned Lego-First + Referee calibration SFT dataset.",
    )
    parser.add_argument(
        "--agentic-dataset",
        default="data/distillation/agentic_tool_calling_sft.jsonl",
        help="Pinned multi-turn SRE + Workspace tool-calling SFT split.",
    )
    parser.add_argument(
        "--agentic-oversample",
        type=int,
        default=3,
        help="Oversample factor for the pinned multi-turn tool-calling SFT dataset.",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="data/distillation/referee_calibration_dpo.jsonl",
        help="Balanced Precision + Recall + Architecture Calibration DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--agentic-dpo-dataset",
        default="data/distillation/agentic_tool_calling_dpo.jsonl",
        help="Multi-turn tool-action vs prose-dump / search-loop / false-refusal DPO preference JSONL dataset.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional existing PEFT LoRA checkpoint directory to resume/calibrate.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/jean-hermes3-8b-sre-lora",
        help="Directory to save LoRA adapter weights and Ollama Modelfile.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit NF4 QLoRA instead of native BF16.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of SFT epochs (or DPO epochs when --mode dpo).")
    parser.add_argument("--dpo-epochs", type=int, default=3, help="Number of Stage-2 DPO epochs when --mode sft_dpo.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max token context during LoRA training.")
    parser.add_argument("--max-samples", type=int, default=420, help="Max complete trajectories per epoch.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate for SFT.")
    parser.add_argument("--dpo-lr", type=float, default=1.2e-5, help="Learning rate for DPO calibration.")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="KL penalty beta for DPOTrainer.")
    return parser.parse_args()


def _clean_utf8(text: str) -> str:
    return str(text or "").encode("utf-8", "replace").decode("utf-8")


def _clean_message(m: dict, default_role: str = "user") -> dict:
    role = _clean_utf8(m.get("role", default_role))
    content = _clean_utf8(m.get("content", ""))
    out: dict = {"role": role, "content": content}
    raw_tcs = m.get("tool_calls")
    if isinstance(raw_tcs, list) and len(raw_tcs) > 0:
        norm_tcs = []
        for idx, tc in enumerate(raw_tcs):
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            args_val = fn.get("arguments", {})
            if isinstance(args_val, str):
                try:
                    args_val = json.loads(args_val)
                except Exception:
                    args_val = {}
            if not isinstance(args_val, dict):
                args_val = {}
            norm_tcs.append(
                {
                    "id": _clean_utf8(tc.get("id", f"call_{idx+1}")) if isinstance(tc, dict) else f"call_{idx+1}",
                    "type": "function",
                    "function": {
                        "name": _clean_utf8(fn.get("name", "tool")),
                        "arguments": args_val,
                    },
                }
            )
        out["tool_calls"] = norm_tcs
    if isinstance(m.get("tool_call_id"), str) and m["tool_call_id"]:
        out["tool_call_id"] = _clean_utf8(m["tool_call_id"])
    if isinstance(m.get("name"), str) and m["name"]:
        out["name"] = _clean_utf8(m["name"])
    return out


def render_hermes3_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Render messages + optional function schemas into the exact fixed Ollama Hermes-3 ChatML format
    so training trajectories match live Ollama inference byte-for-byte with zero skew.
    """
    sys_parts: list[str] = []
    conv_msgs: list[dict] = []
    for idx, m in enumerate(messages):
        if m.get("role") == "system" and not conv_msgs:
            txt = (m.get("content") or "").strip()
            if txt:
                sys_parts.append(txt)
        else:
            conv_msgs.append(m)

    system_text = "\n\n".join(sys_parts)
    has_tools = isinstance(tools, list) and len(tools) > 0

    chunks: list[str] = []
    if system_text or has_tools:
        sys_block = "<|im_start|>system"
        if system_text:
            sys_block += f"\n{system_text}"
        if has_tools:
            tool_defs = []
            for t in tools:
                fn = t.get("function", t) if isinstance(t, dict) else {}
                tool_defs.append(json.dumps({"type": "function", "function": fn}, ensure_ascii=False))
            sys_block += f"\n{HERMES3_TOOLS_PREAMBLE}\n" + "\n".join(tool_defs) + f"\n{HERMES3_TOOLS_EPILOGUE}"
        sys_block += "<|im_end|>\n"
        chunks.append(sys_block)

    for m in conv_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            chunks.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if isinstance(tcs, list) and len(tcs) > 0:
                tc_serialized = []
                for tc in tcs:
                    fn = tc.get("function", tc)
                    tc_serialized.append(
                        json.dumps(
                            {"name": fn.get("name", "tool"), "arguments": fn.get("arguments", {})},
                            ensure_ascii=False,
                        )
                    )
                prefix = f"{content.strip()}\n" if content and content.strip() else ""
                chunks.append(
                    f"<|im_start|>assistant\n{prefix}<tool_call>\n"
                    + "\n".join(tc_serialized)
                    + "\n</tool_call>\n<|im_end|>\n"
                )
            else:
                chunks.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            chunks.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
        else:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")

    return "".join(chunks)


def _read_rendered_sft_records(file_paths: list[str], max_chars: int) -> list[dict]:
    records = []
    for fp in file_paths:
        if not fp or not Path(fp).exists():
            continue
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages")
                if not isinstance(msgs, list) or len(msgs) < 2:
                    continue
                cleaned_msgs = [_clean_message(m) for m in msgs if isinstance(m, dict)]
                tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
                rendered = render_hermes3_chat(cleaned_msgs, tools=tools, add_generation_prompt=False)
                if len(rendered) <= max_chars:
                    records.append({"text": rendered})
    return records


def _normalize_turn_list(val, default_role: str = "assistant") -> list[dict]:
    if isinstance(val, list):
        return [_clean_message(m, default_role=default_role) for m in val if isinstance(m, dict)]
    return [{"role": default_role, "content": _clean_utf8(val)}]


def read_rendered_dpo_records(dpo_paths: list[Path], max_chars: int = 36000) -> list[dict]:
    records = []
    for dpo_path in dpo_paths:
        if not dpo_path.exists():
            continue
        with open(dpo_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = _normalize_turn_list(row.get("prompt", []), default_role="user")
                chosen = _normalize_turn_list(row.get("chosen", []), default_role="assistant")
                rejected = _normalize_turn_list(row.get("rejected", []), default_role="assistant")
                if not prompt or not chosen or not rejected:
                    continue
                tools = row.get("tools") if isinstance(row.get("tools"), list) and row.get("tools") else None
                prompt_str = render_hermes3_chat(prompt, tools=tools, add_generation_prompt=True)
                chosen_full = render_hermes3_chat(prompt + chosen, tools=tools, add_generation_prompt=False)
                rejected_full = render_hermes3_chat(prompt + rejected, tools=tools, add_generation_prompt=False)
                if not chosen_full.startswith(prompt_str) or not rejected_full.startswith(prompt_str):
                    raise ValueError(f"Hermes-3 DPO template prefix mismatch on {row.get('id')}")
                chosen_suffix = chosen_full[len(prompt_str) :]
                rejected_suffix = rejected_full[len(prompt_str) :]
                if len(prompt_str) + max(len(chosen_suffix), len(rejected_suffix)) <= max_chars:
                    records.append(
                        {
                            "prompt": prompt_str,
                            "chosen": chosen_suffix,
                            "rejected": rejected_suffix,
                        }
                    )
    return records


def write_ollama_modelfile(output_dir: Path) -> Path:
    modelfile_path = output_dir / "Modelfile"
    content = (
        "# Ollama Modelfile — Sovereign Unified Agentic SRE & Tool-Caller 8B (Wave-4 LoRA + DPO Fused into Hermes-3-Llama-3.1-8B)\n"
        "# Built with Meta Llama 3 (NousResearch/Hermes-3-Llama-3.1-8B). Base weights preserved as: hermes3:base\n"
        "# Preserves both .System and .Tools simultaneously inside the Hermes-3 system turn.\n"
        "FROM ./hermes3-8b-sre.gguf\n\n"
        f'TEMPLATE """{FIXED_HERMES3_OLLAMA_TEMPLATE}"""\n\n'
        "PARAMETER num_ctx 16384\n"
        "PARAMETER num_gpu 99\n"
        "PARAMETER stop <|im_start|>\n"
        "PARAMETER stop <|im_end|>\n"
    )
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def write_model_card_readme(output_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    content = """---
base_model: NousResearch/Hermes-3-Llama-3.1-8B
library_name: peft
model_name: jean-hermes3-8b-sre-lora
license: llama3.1
pipeline_tag: text-generation
tags:
  - base_model:adapter:NousResearch/Hermes-3-Llama-3.1-8B
  - lora
  - dpo
  - sft
  - tool-calling
  - code-audit
  - sre
---

# Hermes-3-Llama-3.1-8B — Unified Agentic SRE & Multi-Turn Tool-Caller LoRA

Built with Meta Llama 3 (`NousResearch/Hermes-3-Llama-3.1-8B`).
Wave-4 unified 2-stage `BF16` LoRA (`r=16`, `lora_alpha=32`, all 7 Llama-3.1 projections) + KL-anchored `DPO` (`beta=0.1`) adapter trained on clean-room SRE Referee calibration (`CLEAN_PASS` precision + P0/P1 recall), `@krusch/toolkit` Lego-First synthesis, and multi-turn MCP/Workspace tool-calling trajectories with temporal state overrides.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def _summarize_log_history(log_history: list[dict]) -> dict:
    step_logs = [entry for entry in log_history if "loss" in entry]
    summary = {
        "logged_steps": len(step_logs),
        "loss_start": step_logs[0].get("loss") if step_logs else None,
        "loss_end": step_logs[-1].get("loss") if step_logs else None,
        "history": log_history,
    }
    acc_logs = [e.get("mean_token_accuracy") for e in step_logs if e.get("mean_token_accuracy") is not None]
    if acc_logs:
        summary["token_accuracy_start"] = acc_logs[0]
        summary["token_accuracy_end"] = acc_logs[-1]
    margin_logs = [e.get("rewards/margins") for e in step_logs if e.get("rewards/margins") is not None]
    if margin_logs:
        summary["reward_margin_start"] = margin_logs[0]
        summary["reward_margin_end"] = margin_logs[-1]
    dpo_acc_logs = [e.get("rewards/accuracies") for e in step_logs if e.get("rewards/accuracies") is not None]
    if dpo_acc_logs:
        summary["reward_accuracy_start"] = dpo_acc_logs[0]
        summary["reward_accuracy_end"] = dpo_acc_logs[-1]
    return summary


def run_smoke_verification(model, tokenizer) -> dict:
    import torch

    model.eval()
    device = next(model.parameters()).device

    def gen(prompt_text: str, max_new_tokens: int = 320) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
        new_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=False)

    sre_prompt = render_hermes3_chat(
        [
            {
                "role": "system",
                "content": "You are an Adversarial Code Referee. Output a JSON verdict inside ```json ... ``` with keys verdict, checked_files, not_flagged_notes, and findings.",
            },
            {
                "role": "user",
                "content": "Audit `src/server.js`:\n```js\nimport { createApiServer, okEnvelope } from '@krusch/toolkit/http';\nexport const server = createApiServer({ name: 'svc', routes: { 'GET /ping': async (req, res) => okEnvelope(res, { pong: true }) } });\n```",
            },
        ],
        add_generation_prompt=True,
    )
    sre_out = gen(sre_prompt, max_new_tokens=320)

    tool_prompt = render_hermes3_chat(
        [
            {
                "role": "system",
                "content": "MEMORY RECORD [2026-10-02]: Q4 Architecture Review is Tuesday 09:00 UTC in Room A.\nMEMORY RECORD [2026-10-28]: UPDATE — Q4 Architecture Review moved to Thursday 14:00 UTC in Room C.",
            },
            {
                "role": "user",
                "content": "Create the workspace doc titled 'Q4 Architecture Review' with the current room and time from memory.",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "create_workspace_doc",
                    "description": "Create a workspace markdown document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["title", "body"],
                    },
                },
            }
        ],
        add_generation_prompt=True,
    )
    tool_out = gen(tool_prompt, max_new_tokens=256)

    return {
        "sre_clean_pass": "CLEAN_PASS" in sre_out,
        "sre_preview": sre_out[:450],
        "tool_call_emitted": "<tool_call>" in tool_out and "create_workspace_doc" in tool_out,
        "temporal_override_bound": ("Thursday" in tool_out or "14:00" in tool_out) and "Room C" in tool_out,
        "tool_preview": tool_out[:450],
    }


def main() -> None:
    import datasets as hf_datasets
    import peft as hf_peft
    import torch
    import transformers as hf_transformers
    import trl as hf_trl
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    t0 = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_chars = max(36000, args.max_seq_length * 4)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_ds = None
    dpo_ds = None

    if args.mode in ("sft", "sft_dpo"):
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

        data_files = [str(dataset_path)]
        argon_path = Path(args.argon_dataset)
        if argon_path.exists():
            data_files.extend([str(argon_path)] * 2)
            print(f"[LoRA-Hermes3-8B] Including Argon Golden Remediation dataset (2x oversampled) from {argon_path}", flush=True)

        pinned_files = []
        cal_path = Path(args.calibration_dataset)
        if cal_path.exists():
            oversample = max(1, int(args.calibration_oversample))
            pinned_files.extend([str(cal_path)] * oversample)
            print(f"[LoRA-Hermes3-8B] Pinning {oversample}x Lego-First + Calibrated CLEAN_PASS split from {cal_path}", flush=True)

        agentic_path = Path(args.agentic_dataset)
        if agentic_path.exists():
            ag_over = max(1, int(args.agentic_oversample))
            pinned_files.extend([str(agentic_path)] * ag_over)
            print(f"[LoRA-Hermes3-8B] Pinning {ag_over}x Wave-4 Agentic SRE + Workspace Tool-Calling split from {agentic_path}", flush=True)

        print(f"[LoRA-Hermes3-8B] Rendering fixed Hermes-3 ChatML + <tool_call> dataset (max_seq_length={args.max_seq_length})...", flush=True)
        pinned_records = _read_rendered_sft_records(pinned_files, max_chars=max_chars)
        regular_records = _read_rendered_sft_records(data_files, max_chars=max_chars)
        if regular_records:
            reg_ds = Dataset.from_list(regular_records).shuffle(seed=42)
            if args.max_samples and len(pinned_records) < args.max_samples:
                budget = args.max_samples - len(pinned_records)
                if len(reg_ds) > budget:
                    reg_ds = reg_ds.select(range(budget))
            regular_records = reg_ds.to_list()
        sft_ds = Dataset.from_list(pinned_records + regular_records).shuffle(seed=42)
        print(f"[LoRA-Hermes3-8B] Loaded {len(sft_ds)} unified Teacher + Argon + Lego-First + Tool-Calling trajectories.", flush=True)

    if args.mode in ("dpo", "sft_dpo"):
        dpo_path = Path(args.dpo_dataset)
        if not dpo_path.exists():
            raise FileNotFoundError(f"DPO preference dataset not found at {dpo_path}")
        dpo_paths = [dpo_path]
        ag_dpo_path = Path(args.agentic_dpo_dataset)
        if ag_dpo_path.exists():
            dpo_paths.extend([ag_dpo_path] * 2)
            print(f"[DPO-Hermes3-8B] Including 2x Wave-4 Tool-Action preference pairs from {ag_dpo_path}", flush=True)
        dpo_records = read_rendered_dpo_records(dpo_paths, max_chars=max_chars)
        dpo_ds = Dataset.from_list(dpo_records).shuffle(seed=42)
        print(f"[DPO-Hermes3-8B] Loaded {len(dpo_ds)} unified Referee + Tool-Action preference pairs.", flush=True)

    model_kwargs: dict = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[LoRA-Hermes3-8B] Loading base model {args.base_model} in 4-bit NF4...", flush=True)
    else:
        print(f"[LoRA-Hermes3-8B] Loading base model {args.base_model} in full native BF16 on 96GB G4...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    resumed_from = None
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if adapter_dir and (adapter_dir / "adapter_config.json").exists():
        print(f"[LoRA-Hermes3-8B] Resuming trainable PEFT adapter from {adapter_dir}...", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        resumed_from = str(adapter_dir)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    metrics: dict = {
        "base_model": args.base_model,
        "precision": "nf4_4bit" if args.load_in_4bit else "native_bf16",
        "mode": args.mode,
        "resumed_from": resumed_from,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "frameworks": {
            "torch": torch.__version__,
            "transformers": hf_transformers.__version__,
            "peft": hf_peft.__version__,
            "trl": hf_trl.__version__,
            "datasets": hf_datasets.__version__,
        },
    }

    if args.mode in ("sft", "sft_dpo"):
        sft_sig = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        sft_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "learning_rate": args.lr,
            "bf16": True,
            "logging_steps": 10,
            "save_strategy": "no",
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused",
            "report_to": "none",
        }
        if "dataset_text_field" in sft_sig:
            sft_kwargs["dataset_text_field"] = "text"
        if "max_length" in sft_sig:
            sft_kwargs["max_length"] = args.max_seq_length
        elif "max_seq_length" in sft_sig:
            sft_kwargs["max_seq_length"] = args.max_seq_length
        if "warmup_ratio" in sft_sig:
            sft_kwargs["warmup_ratio"] = 0.05

        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=sft_ds,
            args=SFTConfig(**sft_kwargs),
        )
        print("[LoRA-Hermes3-8B] Starting Stage-1 BF16 SFT training...", flush=True)
        t_sft = time.monotonic()
        sft_res = sft_trainer.train()
        sft_wall = round(time.monotonic() - t_sft, 2)
        metrics["sft"] = {
            "samples": len(sft_ds),
            "epochs": args.epochs,
            "lr": args.lr,
            "wall_seconds": sft_wall,
            "train_loss": getattr(sft_res, "training_loss", None),
            **_summarize_log_history(list(sft_trainer.state.log_history)),
        }
        model = sft_trainer.model
        del sft_trainer
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode in ("dpo", "sft_dpo"):
        dpo_sig = set(inspect.signature(DPOConfig.__init__).parameters.keys())
        can_use_ref_adapter = "model_adapter_name" in dpo_sig and "ref_adapter_name" in dpo_sig
        dpo_epochs = args.dpo_epochs if args.mode == "sft_dpo" else args.epochs
        with tempfile.TemporaryDirectory() as tmp_ref_dir:
            loaded_ref_adapter = False
            if can_use_ref_adapter:
                model.save_pretrained(tmp_ref_dir)
                model.load_adapter(tmp_ref_dir, adapter_name="sft_ref", is_trainable=False)
                model.set_adapter("default")
                loaded_ref_adapter = True
                print("[DPO-Hermes3-8B] Anchored Stage-2 DPO reference policy to frozen Stage-1 adapter ('sft_ref').", flush=True)

            dpo_kwargs = {
                "output_dir": str(output_dir),
                "num_train_epochs": dpo_epochs,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "learning_rate": args.dpo_lr,
                "beta": args.dpo_beta,
                "bf16": True,
                "logging_steps": 2,
                "save_strategy": "no",
                "lr_scheduler_type": "cosine",
                "optim": "adamw_torch_fused",
                "report_to": "none",
            }
            if "max_length" in dpo_sig:
                dpo_kwargs["max_length"] = args.max_seq_length
            if "max_prompt_length" in dpo_sig:
                dpo_kwargs["max_prompt_length"] = min(6144, int(args.max_seq_length * 0.75))
            if "warmup_ratio" in dpo_sig:
                dpo_kwargs["warmup_ratio"] = 0.05
            if loaded_ref_adapter:
                dpo_kwargs["model_adapter_name"] = "default"
                dpo_kwargs["ref_adapter_name"] = "sft_ref"

            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                processing_class=tokenizer,
                train_dataset=dpo_ds,
                args=DPOConfig(**dpo_kwargs),
            )
            print("[DPO-Hermes3-8B] Starting Stage-2 Referee + Tool-Action Preference Calibration (DPO)...", flush=True)
            t_dpo = time.monotonic()
            dpo_res = dpo_trainer.train()
            dpo_wall = round(time.monotonic() - t_dpo, 2)
            metrics["dpo"] = {
                "pairs": len(dpo_ds),
                "epochs": dpo_epochs,
                "lr": args.dpo_lr,
                "beta": args.dpo_beta,
                "ref_adapter_anchored": loaded_ref_adapter,
                "wall_seconds": dpo_wall,
                "train_loss": getattr(dpo_res, "training_loss", None),
                **_summarize_log_history(list(dpo_trainer.state.log_history)),
            }
            model = dpo_trainer.model
            del dpo_trainer
            if loaded_ref_adapter:
                model.set_adapter("default")
                model.delete_adapter("sft_ref")
            gc.collect()
            torch.cuda.empty_cache()

    print("[LoRA-Hermes3-8B] Running in-VRAM post-training Gate A + Gate B smoke verification...", flush=True)
    smoke = run_smoke_verification(model, tokenizer)
    print(f"[LoRA-Hermes3-8B] Smoke check: {json.dumps({k: v for k, v in smoke.items() if not k.endswith('_preview')})}", flush=True)
    metrics["smoke_verification"] = smoke

    metrics["total_wall_seconds"] = round(time.monotonic() - t0, 2)
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    print(f"[LoRA-Hermes3-8B] Saving LoRA adapter & tokenizer to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    modelfile = write_ollama_modelfile(output_dir)
    write_model_card_readme(output_dir)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[LoRA-Hermes3-8B] Done! Wrote Ollama Modelfile to {modelfile} and metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
