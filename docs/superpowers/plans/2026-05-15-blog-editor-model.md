# blog-editor-model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fine-tuning pipeline that trains SmolLM2-360M-Instruct on a blog editing dataset and exports it as a Core ML package for on-device iPhone inference.

**Architecture:** mlx-lm LoRA fine-tuning on Apple Silicon, adapter fusion, then Core ML export via `optimum[coreml]`. Dataset is converted from raw JSON (with JS-style comments) to ChatML-formatted JSONL. The repo produces one artifact: `model/blog-editor.mlpackage`.

**Tech Stack:** Python 3.11, uv, mlx-lm ≥0.15, coremltools ≥7.2, optimum[coreml] ≥1.20, transformers ≥4.40, torch ≥2.2, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Create | Dependencies, Python constraint, pytest config |
| `.python-version` | Create | Pin Python 3.11 (coremltools ceiling) |
| `.gitignore` | Create | Exclude model artifacts and generated JSONL |
| `Makefile` | Create | All pipeline targets |
| `data/__init__.py` | Create | Makes `data/` a package for pytest imports |
| `data/generated/.gitkeep` | Create | Placeholder dir for generated batches |
| `data/build_dataset.py` | Create | Strip comments, validate, convert to ChatML, split |
| `data/system_prompt.txt` | Create | Canonical system prompt (must match at inference) |
| `data/raw/blog_writing_dataset.json` | Create | Copy from Downloads |
| `training/lora_config.yaml` | Create | mlx-lm LoRA hyperparams |
| `export/__init__.py` | Create | Package marker |
| `export/export_coreml.py` | Create | Fused weights → .mlpackage via optimum |
| `export/test_inference.py` | Create | Load .mlpackage, run 3 smoke prompts |
| `tests/test_build_dataset.py` | Create | Unit tests for build_dataset.py |

---

## Task 1: Repo Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `Makefile` (install target only)
- Create: `data/__init__.py`
- Create: `data/generated/.gitkeep`
- Create directory stubs: `model/`, `tests/`, `training/`, `export/`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "blog-editor-model"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "mlx-lm>=0.15",
    "coremltools>=7.2",
    "optimum[coreml]>=1.20",
    "transformers>=4.40",
    "torch>=2.2",
]

[tool.uv]
python-preference = "only-managed"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Create `.gitignore`**

```
model/adapters/
model/merged/
model/blog-editor.mlpackage/
data/train.jsonl
data/eval.jsonl
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 4: Create stub `Makefile`**

```makefile
.PHONY: install

install:
	uv sync --group dev
```

- [ ] **Step 5: Create directory structure and package markers**

```bash
mkdir -p data/raw data/generated model tests training export
touch data/__init__.py data/generated/.gitkeep export/__init__.py
```

- [ ] **Step 6: Install deps and verify Python version**

```bash
uv python install 3.11
uv sync --group dev
uv run python --version
```

Expected: `Python 3.11.x`

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml .python-version .gitignore Makefile data/__init__.py data/generated/.gitkeep export/__init__.py
git commit -m "feat: scaffold repo"
```

---

## Task 2: `build_dataset.py` — comment stripping

**Files:**
- Create: `tests/test_build_dataset.py`
- Create: `data/build_dataset.py` (partial — `strip_js_comments` only)

- [ ] **Step 1: Create `tests/test_build_dataset.py` with failing tests**

```python
import pytest
from data.build_dataset import strip_js_comments


class TestStripJsComments:
    def test_removes_line_comment(self):
        text = '[\n  // CATEGORY\n  {"a": 1}\n]'
        result = strip_js_comments(text)
        assert "//" not in result
        assert '"a": 1' in result

    def test_preserves_url_in_string(self):
        text = '{"url": "http://example.com"}'
        result = strip_js_comments(text)
        assert "http://example.com" in result

    def test_no_comments_unchanged(self):
        text = '{"instruction": "Fix this.", "input": "", "response": "Done."}'
        assert strip_js_comments(text) == text

    def test_comment_after_object(self):
        text = '{"a": 1} // trailing\n{"b": 2}'
        result = strip_js_comments(text)
        assert "//" not in result
        assert '"a": 1' in result
        assert '"b": 2' in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_build_dataset.py::TestStripJsComments -v
```

Expected: `ModuleNotFoundError` (no implementation yet)

- [ ] **Step 3: Create `data/build_dataset.py` with `strip_js_comments`**

```python
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any


def strip_js_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif not in_string and text[i : i + 2] == "//":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            result.append(ch)
        i += 1
    return "".join(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_build_dataset.py::TestStripJsComments -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_dataset.py data/build_dataset.py
git commit -m "feat: add strip_js_comments with tests"
```

---

## Task 3: `build_dataset.py` — validation + ChatML conversion

**Files:**
- Modify: `tests/test_build_dataset.py` (append new test classes)
- Modify: `data/build_dataset.py` (append `SYSTEM`, `validate_record`, `to_chatml`)

- [ ] **Step 1: Append failing tests to `tests/test_build_dataset.py`**

```python
from data.build_dataset import SYSTEM, validate_record, to_chatml


class TestValidateRecord:
    def test_valid_record_passes(self):
        validate_record({"instruction": "Fix this.", "input": "", "response": "Done."})

    def test_missing_instruction_raises(self):
        with pytest.raises(ValueError, match="instruction"):
            validate_record({"input": "", "response": "Done."})

    def test_missing_response_raises(self):
        with pytest.raises(ValueError, match="response"):
            validate_record({"instruction": "Fix this.", "input": ""})


class TestToChatml:
    def test_with_input(self):
        record = {"instruction": "Make concise.", "input": "Long text.", "response": "Short."}
        result = to_chatml(record, SYSTEM)
        assert result["messages"][0] == {"role": "system", "content": SYSTEM}
        assert result["messages"][1]["content"] == "Make concise.\n\nLong text."
        assert result["messages"][2]["content"] == "Short."

    def test_empty_input_omitted(self):
        record = {"instruction": "Fix this.", "input": "", "response": "Done."}
        result = to_chatml(record, SYSTEM)
        assert result["messages"][1]["content"] == "Fix this."

    def test_whitespace_only_input_omitted(self):
        record = {"instruction": "Fix.", "input": "   ", "response": "Done."}
        result = to_chatml(record, SYSTEM)
        assert result["messages"][1]["content"] == "Fix."

    def test_roles_are_correct(self):
        record = {"instruction": "Fix.", "input": "text", "response": "Fixed."}
        result = to_chatml(record, SYSTEM)
        assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_build_dataset.py::TestValidateRecord tests/test_build_dataset.py::TestToChatml -v
```

Expected: `ImportError` (functions not yet defined)

- [ ] **Step 3: Append to `data/build_dataset.py`**

```python
SYSTEM = "You are a blog writing editor. Improve, rewrite, or fix the text as instructed. If the request is out of scope, say so briefly."


def validate_record(record: dict[str, Any]) -> None:
    if "instruction" not in record:
        raise ValueError(f"Record missing 'instruction': {record}")
    if "response" not in record:
        raise ValueError(f"Record missing 'response': {record}")


def to_chatml(record: dict[str, Any], system: str) -> dict:
    instruction = record["instruction"]
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["response"]},
        ]
    }
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
uv run pytest tests/test_build_dataset.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_dataset.py data/build_dataset.py
git commit -m "feat: add validate_record and to_chatml with tests"
```

---

## Task 4: `build_dataset.py` — splits + main + CLI

**Files:**
- Modify: `tests/test_build_dataset.py` (append `TestBuildSplits`)
- Modify: `data/build_dataset.py` (append `TRAIN_N`, `build_splits`, `load_json_file`, `main`, `__main__` block)

- [ ] **Step 1: Append failing tests to `tests/test_build_dataset.py`**

```python
from data.build_dataset import build_splits


class TestBuildSplits:
    def _records(self, n: int) -> list[dict]:
        return [{"instruction": f"Fix {i}.", "input": "", "response": f"Done {i}."} for i in range(n)]

    def test_split_sizes(self):
        train, eval_ = build_splits(self._records(10), train_n=8, seed=42)
        assert len(train) == 8
        assert len(eval_) == 2

    def test_split_is_deterministic(self):
        records = self._records(10)
        train1, _ = build_splits(records, train_n=8, seed=42)
        train2, _ = build_splits(records, train_n=8, seed=42)
        assert train1 == train2

    def test_covers_all_records(self):
        records = self._records(10)
        train, eval_ = build_splits(records, train_n=8, seed=42)
        all_instructions = {r["instruction"] for r in train + eval_}
        assert len(all_instructions) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_build_dataset.py::TestBuildSplits -v
```

Expected: `ImportError`

- [ ] **Step 3: Append to `data/build_dataset.py`**

```python
TRAIN_N = 270


def build_splits(
    records: list[dict], train_n: int, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled[:train_n], shuffled[train_n:]


def load_json_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return json.loads(strip_js_comments(text))


def main(data_dir: Path = Path("data"), train_n: int = TRAIN_N) -> None:
    raw_dir = data_dir / "raw"
    generated_dir = data_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for json_file in sorted(raw_dir.glob("*.json")) + sorted(generated_dir.glob("*.json")):
        batch = load_json_file(json_file)
        for record in batch:
            validate_record(record)
        records.extend(batch)

    train_records, eval_records = build_splits(records, train_n=train_n)

    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(to_chatml(r, SYSTEM)) for r in train_records) + "\n",
        encoding="utf-8",
    )
    (data_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(to_chatml(r, SYSTEM)) for r in eval_records) + "\n",
        encoding="utf-8",
    )
    print(f"Written: {len(train_records)} train, {len(eval_records)} eval")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--train-n", default=TRAIN_N, type=int)
    args = parser.parse_args()
    main(args.data_dir, args.train_n)
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_dataset.py data/build_dataset.py
git commit -m "feat: complete build_dataset with splits and CLI"
```

---

## Task 5: Raw dataset, system prompt, and training config

**Files:**
- Create: `data/raw/blog_writing_dataset.json`
- Create: `data/system_prompt.txt`
- Create: `training/lora_config.yaml`

No unit tests — config and data files. Validated by running the dataset build.

- [ ] **Step 1: Copy raw dataset**

```bash
cp ~/Downloads/blog_writing_dataset.json data/raw/blog_writing_dataset.json
```

- [ ] **Step 2: Create `data/system_prompt.txt`**

```
You are a blog writing editor. Improve, rewrite, or fix the text as instructed. If the request is out of scope, say so briefly.
```

- [ ] **Step 3: Create `training/lora_config.yaml`**

Before writing this file, verify the LoRA config key names match the installed mlx-lm version:

```bash
uv run mlx_lm.lora --help | grep -i lora
```

Then create `training/lora_config.yaml`:

```yaml
model: HuggingFaceTB/SmolLM2-360M-Instruct
data: data/
train: true
seed: 42

# LoRA
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

# Data splits
train_split: train
valid_split: eval
```

If `mlx_lm.lora --help` shows flat keys like `--lora-rank` instead of `lora_parameters`, replace the nested block with:

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_scale: 10.0
```

- [ ] **Step 4: Run dataset build smoke test**

With only ~50 raw examples, use a smaller `--train-n` so `eval.jsonl` is non-empty (mlx-lm requires a non-empty validation set):

```bash
uv run python data/build_dataset.py --train-n 45
```

Expected:
```
Written: 45 train, 5 eval
```

Verify line counts:

```bash
wc -l data/train.jsonl data/eval.jsonl
```

Expected:
```
45 data/train.jsonl
 5 data/eval.jsonl
50 total
```

Expand the dataset to ~300 examples before running `make train`. The full `--train-n 270` split is only valid once 300+ records exist.

- [ ] **Step 5: Commit**

```bash
git add data/raw/blog_writing_dataset.json data/system_prompt.txt training/lora_config.yaml
git commit -m "feat: add raw dataset, system prompt, and LoRA config"
```

---

## Task 6: Export script and smoke test

**Files:**
- Create: `export/export_coreml.py`
- Create: `export/test_inference.py`

These scripts run after training is complete. No unit tests — they require a real `.mlpackage`.

- [ ] **Step 1: Create `export/export_coreml.py`**

```python
"""Convert fused MLX model weights to Core ML via optimum-cli."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def export(merged_path: str, output_path: str, max_length: int) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "optimum.exporters.coreml",
        "--model", merged_path,
        "--task", "text-generation",
        "--sequence-length", str(max_length),
        output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-path", required=True, help="Path to fused model (mlx_lm.fuse output)")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    export(args.merged_path, args.output, args.max_length)
```

**If `optimum.exporters.coreml` raises `ModuleNotFoundError`:** verify the entry point first:

```bash
uv run optimum-cli export coreml --help
```

If that works, replace the `cmd` in `export()` with the CLI form:

```python
cmd = [
    "uv", "run", "optimum-cli", "export", "coreml",
    "--model", merged_path,
    "--task", "text-generation",
    "--sequence-length", str(max_length),
    output_path,
]
```

- [ ] **Step 2: Create `export/test_inference.py`**

```python
"""Smoke-test a converted .mlpackage: load it and run 3 test prompts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

SYSTEM = Path("data/system_prompt.txt").read_text().strip()

TEST_CASES = [
    ("Make this more concise.", "In this blog post I am going to be talking about various ways to improve content."),
    ("Write a full 1500 word blog post about SEO.", ""),
    ("Fix this.", "blog good write"),
]


def build_prompt(instruction: str, input_text: str) -> str:
    user = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main(model_path: str, tokenizer_path: str, max_length: int = 512) -> None:
    print(f"Loading: {model_path}")
    mlmodel = ct.models.MLModel(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for instruction, input_text in TEST_CASES:
        prompt = build_prompt(instruction, input_text)
        tokens = tokenizer(prompt, return_tensors="np", truncation=True, max_length=max_length)
        n = tokens["input_ids"].shape[1]

        input_ids = np.zeros((1, max_length), dtype=np.int32)
        attn_mask = np.zeros((1, max_length), dtype=np.int32)
        input_ids[0, :n] = tokens["input_ids"][0]
        attn_mask[0, :n] = 1

        result = mlmodel.predict({"input_ids": input_ids, "attention_mask": attn_mask})
        assert result.get("logits") is not None, f"No logits in output: {list(result.keys())}"
        print(f"PASS | logits shape: {result['logits'].shape} | prompt: {instruction[:40]!r}")

    print("\nAll 3 smoke tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .mlpackage")
    parser.add_argument("--tokenizer", required=True, help="Path to merged model (for tokenizer)")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    main(args.model, args.tokenizer, args.max_length)
```

- [ ] **Step 3: Commit**

```bash
git add export/export_coreml.py export/test_inference.py
git commit -m "feat: add Core ML export script and smoke test"
```

---

## Task 7: Complete Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace `Makefile` with complete version**

```makefile
.PHONY: install dataset test train fuse export test-export

install:
	uv sync --group dev

dataset:
	uv run python data/build_dataset.py

test:
	uv run pytest tests/ -v

train:
	uv run mlx_lm.lora --config training/lora_config.yaml

fuse:
	uv run mlx_lm.fuse \
		--model HuggingFaceTB/SmolLM2-360M-Instruct \
		--adapter-path model/adapters \
		--save-path model/merged

export:
	uv run python export/export_coreml.py \
		--merged-path model/merged \
		--output model/blog-editor.mlpackage \
		--max-length 512

test-export:
	uv run python export/test_inference.py \
		--model model/blog-editor.mlpackage \
		--tokenizer model/merged \
		--max-length 512
```

- [ ] **Step 2: Verify Makefile parses cleanly**

```bash
make --dry-run install
make --dry-run train
make --dry-run export
```

Expected: no syntax errors, commands printed.

- [ ] **Step 3: Run unit tests via make**

```bash
make test
```

Expected: `14 passed`

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: complete Makefile with all pipeline targets"
```

---

## Full Pipeline Sequence (run after dataset expanded to 300 examples)

```bash
make install        # install deps
make dataset        # build train.jsonl / eval.jsonl (270/30 split)
make test           # run unit tests
make train          # LoRA fine-tune (~10 min on M-series)
make fuse           # merge adapters into base model
make export         # convert to Core ML (.mlpackage)
make test-export    # smoke test the .mlpackage
```

Artifacts handed to iOS team:
- `model/blog-editor.mlpackage`
- `data/system_prompt.txt`
