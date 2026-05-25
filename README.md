# pocket-assistant-model

Fine-tuning pipeline for the on-device personal assistant in [pocket-assistant](../pocket-assistant). Takes SmolLM2-360M-Instruct, LoRA fine-tunes it on a structured JSON task, and exports to Core ML for on-device inference.

## Contents

- [Requirements](#requirements)
- [Pipeline](#pipeline)
- [Training data](#training-data)
- [Probes](#probes)
- [Model](#model)
- [Directory layout](#directory-layout)

The model has one job: given the current Unix time and a natural-language request, output a JSON object matching one of four types — `calendar_event`, `notification`, `feed_card`, or `refusal`. Nothing else.

## Requirements

- Python 3.11 (exact — coremltools doesn't support 3.12 yet)
- [`uv`](https://github.com/astral-sh/uv)

```bash
make install
```

## Pipeline

Run these in order after updating training data:

```bash
make dataset    # build data/train.jsonl + data/valid.jsonl from data/raw/*.json
make train      # LoRA fine-tune (800 iters, ~20 min on M-series Mac)
make fuse       # merge adapters into model/merged/
make probe      # run 76 behavioral probes against model/merged
```

Optional export steps:

```bash
make export     # convert to model/pocket-assistant.mlpackage (Core ML, iOS 17+)
make quantize   # 4-bit quantize → model/quantized/ (smaller, for device deployment)
make gguf       # export to model/pocket-assistant-q4.gguf (for llama.cpp / Ollama)
```

## Training data

Raw data lives in `data/raw/*.json` — arrays of `{ "instruction": "...", "input": "...", "response": "..." }` records. `input` is typically `"Current time: <unix_timestamp>"`.

`make dataset` strips JS-style `//` comments (used for section headers in the JSON files), converts everything to ChatML, and splits 270/N train/valid with seed 42.

The system prompt is in `data/system_prompt.txt`. It's injected into every training example and needs to match what the iOS app sends at inference time.

## Probes

`make probe` runs 76 behavioral tests across six categories: scheduling (calendar, notification, feed card), refusal (scope, temporal, adversarial, ambiguous, dissonance). Each probe checks that the model output contains or excludes specific strings.

Run with `--verbose` for per-probe subprocess output:

```bash
uv run python export/probe.py --model model/merged --verbose
```

## Model

Base: `HuggingFaceTB/SmolLM2-360M-Instruct`. LoRA adapts 16 transformer layers (rank 16). After fusing, `model/merged/` contains full weights importable by `transformers`. Core ML export targets static shape `(1, 512)` — max 512 tokens.

Timestamp arithmetic is a known weak point at this scale. The model handles simple relative times ("tomorrow at 9am", "in 2 hours") reasonably well but makes errors on multi-step offset calculations.

## Directory layout

| Path | What it does |
|---|---|
| `data/raw/` | Source JSON datasets |
| `data/build_dataset.py` | Converts raw JSON to ChatML JSONL splits |
| `data/system_prompt.txt` | System prompt injected into every example |
| `training/lora_config.yaml` | LoRA hyperparameters |
| `model/adapters/` | LoRA checkpoints from `make train` |
| `model/merged/` | Fused weights from `make fuse` |
| `export/export_coreml.py` | Core ML export (FP16, iOS 17+) |
| `export/export_gguf.py` | GGUF export |
| `export/probe.py` | Behavioral probe suite |
| `tests/` | pytest unit tests (dataset builder logic) |
