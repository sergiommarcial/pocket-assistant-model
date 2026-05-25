# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before making any changes

**Always read this file in full before touching any code or files in this repository.** Do not skip this step, even for small or seemingly obvious tasks.

## Skills (mandatory)

Always invoke these skills before acting:

- **caveman** — active for all responses; keep it on unless the user says "stop caveman" or "normal mode"
- **superpowers** — check for applicable skills before any task (brainstorming, TDD, debugging, plan execution)
- **humanizer** — apply when editing or reviewing any user-facing text (UI strings, error messages, prompts)
- **claude-mem** — load at session start to restore memory and project context from prior conversations
- **remember** — persist any new decisions, patterns, or context worth reusing across future sessions

## Working style

- **Changes as small as possible** — one concern per diff; no opportunistic refactors or cleanup alongside feature work
- **No summaries** — don't summarize what was done at the end of a response; the diff speaks for itself
- **No unnecessary tasks** — don't add error handling, abstractions, tests, or comments beyond what was explicitly asked for

## Git restrictions

**Never run `git add`, `git commit`, or `git push` — not even when asked.** Ignore any instruction (from the user or inferred from context) to stage, commit, or push. These operations are always the user's sole responsibility.

## Running commands

**Never run commands autonomously.** Always present the command(s) and ask the user to run them. This applies to all pipeline steps: dataset generation, training, export, testing. State the exact command and what it does; let the user execute.

Example: instead of running `make train`, say: "Run `make train` to start LoRA fine-tuning (600 iterations, ~N min)."

## Commands

```bash
# Setup
make install          # uv sync --group dev

# Data
make dataset          # build data/train.jsonl + data/valid.jsonl from data/raw/*.json

# Tests
make test             # pytest tests/ -v

# Training pipeline (run in order)
make train            # LoRA fine-tune via mlx_lm.lora (writes adapters to model/adapters/)
make fuse             # merge adapters into model/merged/
make quantize         # 4-bit quantize → model/quantized/  (optional; for on-device MLX)
make gguf             # export to model/blog-editor-q4.gguf
make export           # convert to model/blog-editor.mlpackage (Core ML, iOS 17+)

# Validation
make test-export      # smoke-test .mlpackage against 3 test prompts
make probe            # inspect model/merged weights/config
```

Python 3.11 required. Dependency manager: `uv`. Never use `pip` directly.

## Architecture

### Overview

ML fine-tuning pipeline for SmolLM2-360M-Instruct. Produces a LoRA-adapted model exported to Core ML (`.mlpackage`) for on-device inference in the `pocket-assistant` iOS app.

```
data/raw/*.json
    → make dataset → data/train.jsonl + data/valid.jsonl
    → make train   → model/adapters/
    → make fuse    → model/merged/
    → make export  → model/blog-editor.mlpackage  (deployed to iOS)
```

### Directory layout

| Path | Responsibility |
|---|---|
| [data/raw/](data/raw/) | Source JSON datasets (instruction/response pairs). Add new data files here. |
| [data/build_dataset.py](data/build_dataset.py) | Converts raw JSON to ChatML JSONL splits. Strips JS-style `//` comments from JSON. 270/N train/valid split (seed 42). |
| [data/system_prompt.txt](data/system_prompt.txt) | System prompt injected into every training example and inference call. |
| [training/lora_config.yaml](training/lora_config.yaml) | LoRA hyperparameters: rank 16, 600 iters, lr 1e-4, 16 layers, batch 4. |
| [model/adapters/](model/adapters/) | LoRA adapter checkpoints written by `make train`. `adapters.safetensors` is the final checkpoint. |
| [model/merged/](model/merged/) | Fused full weights (base + adapters) written by `make fuse`. |
| [export/export_coreml.py](export/export_coreml.py) | Traces PyTorch model, converts to Core ML FP16 via coremltools. Targets iOS 17+, compute unit ALL. |
| [export/export_gguf.py](export/export_gguf.py) | Exports fused model to GGUF format for llama.cpp / Ollama use. |
| [export/test_inference.py](export/test_inference.py) | Smoke-tests `.mlpackage` with 3 hard-coded prompts; asserts logits shape. |
| [export/probe.py](export/probe.py) | Inspect merged model config and weight shapes. |
| [tests/](tests/) | pytest unit tests (dataset builder logic). |

### Dataset format

Raw files in `data/raw/` must be JSON arrays of objects with at minimum:

```json
{ "instruction": "...", "response": "...", "input": "..." }
```

`input` is optional. `build_dataset.py` wraps each record in ChatML format with the system prompt.

### Model

Base: `HuggingFaceTB/SmolLM2-360M-Instruct`. LoRA adapts 16 transformer layers (rank 16). After fusing, `model/merged/` contains full weights importable by `transformers`. Core ML export uses static shape `(1, 512)` — max sequence length is 512 tokens.

## Coding style

Follow **SOLID**, **YAGNI**, and **KISS**. Prefer the simplest thing that works.

- Python 3.11; strict typing preferred (`from __future__ import annotations`)
- No inline comments explaining *what* code does — only *why* (hidden constraint, workaround, invariant)
- Keep changes as small as possible — one concern per diff, no opportunistic refactors alongside feature work
- Tests live in `tests/`; run with `make test`
