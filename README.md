# pocket-assistant-model

Fine-tuning pipeline for the on-device personal assistant in [pocket-assistant](../pocket-assistant). Takes SmolLM2-360M-Instruct, LoRA fine-tunes it on a structured scheduling task, and exports to Core ML for on-device inference.

## Contents

- [Fact sheet](#fact-sheet)
- [Approbal gates](#approbal-gates)
- [Requirements](#requirements)
- [Output format](#output-format)
- [Pipeline](#pipeline)
- [Training data](#training-data)
- [Verification](#verification)
- [Known failures](#known-failures)
- [Interaction patterns](#interaction-patterns)
- [Directory layout](#directory-layout)

## Fact sheet

| | |
|---|---|
| **Base model** | HuggingFaceTB/SmolLM2-360M-Instruct |
| **Parameters** | 360M |
| **LoRA rank** | 16 |
| **Layers adapted** | 16 |
| **Training iterations** | 1000 |
| **Learning rate** | 1e-4 |
| **Batch size** | 4 |
| **Max sequence length** | 512 tokens |
| **Training records** | 270 train / ~154 valid |
| **Targeted patches** | 43 (always in TRAIN, bypass shuffle) |
| **Probe results** | 74/80 (92.5% pre-arbitration baseline) |
| **Accepted failures** | 2 (see [Known failures](#known-failures)) |
| **Reference time** | Monday, June 2, 2025 4:00 PM (Unix `1748880000`) |
| **Output types** | 4 — `CALENDAR`, `NOTIFICATION`, `FEED_CARD`, `REFUSAL` |
| **Export format** | Core ML `.mlpackage` (iOS 17+) |
| **Training time** | ~25 min on M-series Mac |


## Approbal gates

```bash
  [PASS] ADVERSARIAL: 17/17 (100.0% ≥ 100%)
  [PASS] AMBIGUOUS: 12/12 (100.0% ≥ 100%)
  [PASS] ARBITRATION: 4/4 (100.0% ≥ 75%)
  [PASS] CALENDAR: 11/12 (91.7% ≥ 91%)
  [PASS] DISSONANCE: 8/9 (88.9% ≥ 88%)
  [PASS] FEED_CARD: 6/6 (100.0% ≥ 100%)
  [PASS] NOTIFICATION: 9/9 (100.0% ≥ 100%)
  [PASS] SCOPE: 6/6 (100.0% ≥ 100%)
  [PASS] TEMPORAL: 5/5 (100.0% ≥ 100%)

  Advisory (known failures — tracked, not blocking):
    · cot: reasoning field present when asked
    · dissonance: duration contradicts time span
```

## Requirements

- Python 3.11 (exact — coremltools doesn't support 3.12 yet)
- [`uv`](https://github.com/astral-sh/uv)

```bash
make install
```

## Output format

The model has one job: given the current time and a natural-language request, output a single pipe-delimited string matching one of four schemas. Nothing else.

```
CALENDAR|title|startUnix|endUnix|location|attendees
NOTIFICATION|title|body|fireAtUnix
FEED_CARD|title|body|priority
REFUSAL|reason
```

### Field rules

- `startUnix`, `endUnix`, `fireAtUnix` — Unix timestamps in seconds since epoch, computed from the current time passed in the user message.
- `location`, `attendees` — optional; leave empty (trailing `|`) if not provided.
- `attendees` — comma-separated list of email addresses when multiple.
- `priority` — one of `low`, `medium`, or `high`.

### Current time injection

The model does not infer the current time on its own. The caller must prepend it to every user message:

```
Current time: Monday, June 2, 2025 4:00 PM
<user instruction>
```

The system prompt (`data/system_prompt.txt`) must match exactly what the iOS app sends at inference time.

### Examples

```
# Calendar event with attendees
CALENDAR|Kickoff Call|1748998800|1749002400||alice@example.com,bob@example.com

# Notification 30 minutes from now (current time = 4:00 PM → fireAt = 4:30 PM)
NOTIFICATION|Drink Water|Time to drink some water.|1748881800

# Feed card
FEED_CARD|Production Server Down|The production server is currently down. Immediate attention required.|high

# Refusal for out-of-scope request
REFUSAL|Writing blog posts is outside my scope. I can schedule events, set reminders, or create feed cards.
```

### Chain-of-thought (REASON suffix)

When the user asks the model to explain its reasoning, it appends `|REASON|explanation` to any output type:

```
NOTIFICATION|Drink Water|Time to drink some water.|1748881800|REASON|Current time is Monday June 2 4:00 PM UTC = 1748880000. 30 minutes = 1800s. Fire at: 1748880000 + 1800 = 1748881800.
```

This suffix is only emitted when explicitly requested. The `REASON` field is for human inspection — the iOS app strips it before parsing.

## Pipeline

Full pipeline from data to verified model:

```bash
make dataset    # build data/train.jsonl + data/valid.jsonl from data/raw/*.json
make train      # LoRA fine-tune (1000 iters, ~25 min on M-series Mac)
make fuse       # merge adapters into model/merged/
make probe      # run 80 behavioral probes against model/merged
```

To regenerate `model/merged/` from existing adapters (without retraining):

```bash
make fuse
```

To retrain from scratch:

```bash
make dataset && rm -rf model/adapters/* && make train && make fuse
```

Optional export steps:

```bash
make export     # convert to model/pocket-assistant.mlpackage (Core ML, iOS 17+)
make quantize   # 4-bit quantize → model/quantized/ (smaller, for device deployment)
make gguf       # export to model/pocket-assistant-q4.gguf (for llama.cpp / Ollama)
```

## Training data

### Format

Raw data lives in `data/raw/*.json` — arrays of `{ "instruction": "...", "input": "...", "response": "..." }` records. `input` carries context, typically `"Current time: Monday, June 2, 2025 4:00 PM"`. `make dataset` strips JS-style `//` comments (used for section headers in the JSON files), wraps each record in ChatML with the system prompt, and splits 270/N train/valid with seed 42.

All training examples use a fixed reference time: **Monday, June 2, 2025 4:00 PM** (Unix `1748880000`). All timestamps in the dataset are computed relative to that anchor.

### Targeted patches

`data/raw/targeted_patches.json` contains 43 high-priority records that always go to the training set, bypassing the random shuffle. These fix specific failure modes that the base dataset alone doesn't cover reliably: relative-time notifications, multi-attendee events, past-date refusals, scope refusals, and adversarial refusals.

`build_dataset.py` sets `TRAIN_N=270`. With 43 patches always in TRAIN, `base_train_n = 270 - 43 = 227` base records are drawn from the shuffled pool. Increasing `TRAIN_N` without adding patches pulls more base records in; increasing both together keeps `base_train_n` constant — the TRAIN_N+1 trick for adding patches without displacing critical records.

### Data quality fixes

Already applied. `data/fix_data_quality.py` fixed:
- 9 records with stale Unix-timestamp input format → human-readable date string
- 12 CoT REASON records with wrong date in arithmetic (2025-06-03 → 2025-06-02)
- 2 medication records with wrong `fireAt` timestamp
- 19 REFUSAL records referencing "JSON" output → "pipe-delimited string"
- 3 conflict examples renamed to prevent hallucinated conflict detection on common event names (Daily Standup → Morning Briefing, Team Standup → Morning Briefing, Conference Call → Vendor Sync)

## Verification

`export/probe.py` runs 76 behavioral tests against a fused model. Each probe sends a natural-language instruction (with current-time context) and checks that the output contains or excludes specific strings.

```bash
uv run python export/probe.py --model model/merged
```

Add `--verbose` for full model output per probe.

### Probe categories

| Category | Count | What it tests |
|---|---|---|
| Scheduling · Calendar | 12 | CALENDAR output for events: timestamps, location, attendees, no-op CoT |
| Scheduling · Notification | 9 | NOTIFICATION output: relative times, absolute times, CoT absent when not asked |
| Scheduling · Feed card | 6 | FEED_CARD output: priority levels, urgency detection |
| Refusal · Adversarial | 17 | Resistance to prompt injection, role override, social engineering, scope redefinition |
| Refusal · Ambiguous | 12 | REFUSAL for vague or multi-action requests |
| Refusal · Dissonance | 9 | REFUSAL for self-contradictory inputs (conflicting times, durations, past events) |
| Refusal · Scope | 6 | REFUSAL for out-of-scope requests (writing, code, search) |
| Refusal · Temporal | 5 | REFUSAL for past-date scheduling; CoT on temporal refusal |
| Arbitration | 4 | Correct output type when wording is ambiguous (attend vs create, block vs remind, or-phrased) |

Current result: **74/80** (4 arbitration probes untested against current model). The 2 known failing probes are accepted — see below.

## Known failures

Both are accepted as the ceiling for a 360M model on ~390 training records. Every fix attempt caused regressions elsewhere.

### `cot: reasoning field present when asked`

**Probe:** "Schedule a standup tomorrow at 9am for one hour. Explain your reasoning."  
**Expected:** `CALENDAR|...|REASON|...`  
**Actual:** `CALENDAR|...|` (no REASON suffix, or REFUSAL)

Every CoT standup patch we tried caused the model to start refusing valid future times (8pm medication, 5pm report) as "already passed", and hallucinate self-referential responses like "I take my medication at 8pm. I can't remind myself." The 360M model can't hold CoT for CALENDAR events without breaking other things.

### `dissonance: duration contradicts time span`

**Probe:** "Schedule a 30-minute standup tomorrow from 9am to 11am."  
**Expected:** `REFUSAL|The time span (9am to 11am) is 2 hours, not 30 minutes...`  
**Actual:** `CALENDAR|30-Minute Standup|...|`

A base training record for this exists at position [40] in the seed=42 shuffle (always TRAIN). We added a targeted patch to reinforce it using the TRAIN_N+1 trick to avoid displacing a critical CoT record ([226]). With 2 TRAIN copies the model still outputs CALENDAR. Adding more copies triggers the same timestamp confusion regressions as the CoT fix attempts.

## Interaction patterns

Decisions about how the model output maps to app behavior. Each pattern is a settled design choice — change here first, then retrain if the model needs new training data.

### NOTIFICATION → calendar offer

**Scenario:** User says "create an event for me in 30 minutes" (or similar time-block request without explicit calendar intent).

**Model output:** `NOTIFICATION` — fires a local notification at the specified time.

**App behavior (Option A):** After delivering the notification, the app appends a follow-up assistant message: "Also add this to your calendar?" If the user confirms, the same request is resent with "add to calendar" appended, and the model responds with `CALENDAR`.

**Why not train the model to offer this directly:** The model outputs a single pipe-delimited string per turn. A combined action+question would require a new output type, new training records, and parser changes. The app layer handles this more cleanly.

**Example exchange:**

```
User:  Create a block for me in 30 minutes.
Model: NOTIFICATION|Focus Block|Time for your focus block.|1748881800

App:   Also add this to your calendar? [Yes] [No]

User:  Yes
Model: CALENDAR|Focus Block|1748881800|1748885400||
```

### Ambiguous duration → REFUSAL (ask)

**Scenario:** User says "create an event in 30 minutes" without specifying how long.

**Model output:** `REFUSAL|I need more details. How long is the event?`

This is correct — CALENDAR requires both `startUnix` and `endUnix`. Don't patch this behavior.

### Personal event (no attendees)

**Scenario:** User creates an event for themselves with no other participants.

**Model output:** `CALENDAR|title|startUnix|endUnix||` — attendees field is empty.

The device owner is always the calendar owner. No need to add them to the attendees list.

## Directory layout

| Path | What it does |
|---|---|
| `data/raw/` | Source JSON datasets |
| `data/raw/targeted_patches.json` | High-priority records guaranteed to TRAIN |
| `data/build_dataset.py` | Converts raw JSON to ChatML JSONL splits |
| `data/system_prompt.txt` | System prompt injected into every example |
| `data/fix_data_quality.py` | One-time data quality cleanup (already applied) |
| `training/lora_config.yaml` | LoRA hyperparameters (rank 16, 1000 iters, lr 1e-4) |
| `model/adapters/` | LoRA checkpoints from `make train` |
| `model/merged/` | Fused weights from `make fuse` |
| `export/export_coreml.py` | Core ML export (FP16, iOS 17+) |
| `export/export_gguf.py` | GGUF export |
| `export/probe.py` | 76-probe behavioral test suite |
| `tests/` | pytest unit tests (dataset builder logic) |
