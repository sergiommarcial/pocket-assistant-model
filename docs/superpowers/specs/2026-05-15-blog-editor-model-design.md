# blog-editor-model Design

**Date:** 2026-05-15
**Repo:** `blog-editor-model` (new)
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` fine-tuned via mlx-lm
**Target device:** iPhone (Core ML, on-device inference)

---

## 1. Overview

`blog-editor-model` is a standalone repo that fine-tunes SmolLM2-360M-Instruct on a curated blog writing editing dataset and exports the result as a Core ML package for on-device inference in the blog-agent iOS companion app.

The fine-tuned model handles lightweight local editing tasks (tone, clarity, hooks, CTAs, refusals) without a network call. Heavy pipeline work (research, drafting, peer review, publish) stays in the cloud blog-agent service.

**End-to-end flow:**
```
raw JSON dataset
  → build_dataset.py → train.jsonl / eval.jsonl
  → mlx_lm.lora     → model/adapters/
  → mlx_lm.fuse     → model/merged/
  → export_coreml.py → model/blog-editor.mlpackage
  → iOS app (swift-transformers + CoreML framework)
```

---

## 2. Project Layout

```
blog-editor-model/
  data/
    raw/
      blog_writing_dataset.json     # original ~50 examples (source of truth)
    generated/
      *.json                        # Claude-generated batches, reviewed before merge
    train.jsonl                     # 270 examples, ChatML format
    eval.jsonl                      # 30 examples, ChatML format
    system_prompt.txt               # canonical system prompt (shipped to iOS team)
    build_dataset.py                # merges raw + generated → JSONL splits
  training/
    lora_config.yaml                # mlx-lm LoRA hyperparams
  export/
    export_coreml.py                # merged weights → .mlpackage
    test_inference.py               # sanity-check .mlpackage locally
  model/
    adapters/                       # LoRA adapter weights (git-ignored if large)
    merged/                         # fused weights (git-ignored)
    blog-editor.mlpackage/          # final artifact (git LFS or excluded)
  pyproject.toml
  Makefile
  .python-version                   # pinned to 3.11 (coremltools ceiling)
  .gitignore                        # excludes model/adapters/, model/merged/, model/blog-editor.mlpackage/
```

---

## 3. Dataset Pipeline

### Target distribution (~300 total)

| Category          | Current | Target |
|-------------------|---------|--------|
| Tone & Voice      | 8       | 45     |
| Clarity           | 6       | 40     |
| Hooks & Openings  | 5       | 40     |
| CTAs              | 5       | 35     |
| Structure & Flow  | 7       | 45     |
| Grammar           | 6       | 35     |
| SEO-friendly      | 0       | 20     |
| Refusals          | 8       | 25     |
| Ambiguous         | 5       | 15     |
| **Total**         | **50**  | **300**|

### Expansion strategy

Generate synthetic examples via Claude API, category by category. Review each batch before merging into `data/generated/`. Source JSON uses JS-style `//` comments — `build_dataset.py` strips these with regex before parsing.

### ChatML format

SmolLM2-Instruct expects ChatML. Each `{instruction, input, response}` record converts to:

```json
{"messages": [
  {"role": "system", "content": "You are a blog writing editor. Improve, rewrite, or fix the text as instructed. If the request is out of scope, say so briefly."},
  {"role": "user", "content": "{instruction}\n\n{input}"},
  {"role": "assistant", "content": "{response}"}
]}
```

When `input` is empty (refusals, some ambiguous cases), omit the trailing `\n\n{input}`.

### `build_dataset.py` responsibilities

1. Strip `//` comments from all JSON files in `data/raw/` and `data/generated/`
2. Validate each record has `instruction` and `response` keys (raise on missing)
3. Merge all records into one list
4. Convert each record to ChatML message list
5. Shuffle with `seed=42`
6. Write first 270 → `train.jsonl`, remaining 30 → `eval.jsonl`

---

## 4. Training

### `training/lora_config.yaml`

```yaml
model: HuggingFaceTB/SmolLM2-360M-Instruct
data: data/
train: true
seed: 42

# LoRA — verify exact key names against installed mlx-lm version
lora_layers: 16
lora_parameters:
  rank: 16
  alpha: 32
  dropout: 0.05
  scale: 10.0

# Training
batch_size: 4
iters: 600
learning_rate: 1e-4
warmup_steps: 50
val_batches: 10
steps_per_eval: 100
steps_per_save: 200
adapter_path: model/adapters

# Data
train_split: train
valid_split: eval
```

**Rationale:**
- `iters: 600` ≈ 2 epochs over 270 examples at batch 4 — enough signal, low overfitting risk
- `rank: 16` — adequate capacity for a narrow editing task, keeps adapter small
- `lora_layers: 16` — fine-tune last 16 transformer layers, freeze embeddings
- Expected runtime on M-series Mac: ~8–15 min

---

## 5. Export Pipeline

### `pyproject.toml`

```toml
[project]
name = "blog-editor-model"
requires-python = ">=3.11,<3.12"
dependencies = [
    "mlx-lm>=0.15",
    "coremltools>=7.2",
    "transformers>=4.40",
    "datasets>=2.18",
]

[tool.uv]
python-preference = "only-managed"
```

`.python-version` pinned to `3.11` — coremltools does not support 3.12+ as of this writing.

### `export/export_coreml.py`

Conversion config:
- Precision: `float16` (~180–220 MB output)
- Compute units: `ALL` (Neural Engine + GPU + CPU fallback)
- Max sequence length: `512` tokens (blog editing inputs are short)
- No KV-cache export — single-shot generation, not streaming chat

### `export/test_inference.py`

Loads `blog-editor.mlpackage` via coremltools and runs 3 test prompts (one tone, one refusal, one ambiguous). Asserts non-empty string output and prints results. Run before handing artifact to iOS team.

### Makefile

```makefile
install:
    uv sync

dataset:
    uv run python data/build_dataset.py

train:
    uv run mlx_lm.lora --config training/lora_config.yaml

fuse:
    uv run mlx_lm.fuse --model HuggingFaceTB/SmolLM2-360M-Instruct \
                        --adapter-path model/adapters \
                        --save-path model/merged

export:
    uv run python export/export_coreml.py \
        --merged-path model/merged \
        --output model/blog-editor.mlpackage \
        --max-length 512

test-export:
    uv run python export/test_inference.py --model model/blog-editor.mlpackage
```

---

## 6. iOS Integration (sketch)

This repo's responsibility ends at `model/blog-editor.mlpackage`. iOS implementation lives in the companion app repo.

**Deliverables shipped to iOS team:**
- `model/blog-editor.mlpackage` (~180–220 MB fp16)
- `data/system_prompt.txt` — canonical system prompt used at training time (must match exactly at inference)

**Recommended iOS runtime:** HuggingFace [`swift-transformers`](https://github.com/huggingface/swift-transformers) — handles tokenization and Core ML generation loop.

**ChatML prompt format at inference** (must match training exactly):
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
```

**On-device vs cloud split:**

| On-device (this model) | Cloud (blog-agent pipeline) |
|---|---|
| Tone / clarity / hook rewrites | propose_topics, gather_sources |
| CTA suggestions | write_draft, peer_review |
| Refusal detection | translate, publish |

---

## 7. Out of Scope

- iOS app implementation
- Dataset generation scripts (manual + Claude API, reviewed by hand)
- Model quantization beyond fp16 (evaluate after baseline)
- Streaming / KV-cache inference
