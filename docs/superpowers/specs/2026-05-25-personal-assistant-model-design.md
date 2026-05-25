# personal-assistant-model Design

**Date:** 2026-05-25
**Repo:** `pocket-assistant-model`
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` fine-tuned via mlx-lm
**Target device:** iPhone (Core ML, on-device inference)

---

## 1. Overview

Transform the existing blog-editor fine-tune pipeline into an on-device personal assistant that classifies user intent and responds with structured JSON. The model handles scheduling, reminders, and feed card creation without a network call.

**End-to-end flow (unchanged):**
```
raw JSON dataset
  → build_dataset.py → train.jsonl / valid.jsonl
  → mlx_lm.lora     → model/adapters/
  → mlx_lm.fuse     → model/merged/
  → export_coreml.py → model/pocket-assistant.mlpackage
  → iOS app (swift-transformers + CoreML framework)
```

---

## 2. Output Contract

Model outputs exactly one JSON object per turn. No prose outside the object.

### Schemas (system-generated fields omitted — app fills `id`, `timestamp`)

**Calendar event**
```json
{
  "type": "calendar_event",
  "title": "string",
  "startDate": 1748880000,
  "endDate": 1748883600,
  "location": "string (optional)",
  "attendees": ["string"]
}
```

**Notification / reminder**
```json
{
  "type": "notification",
  "title": "string",
  "body": "string",
  "fireAt": 1748880000 // optional — omit when no specific time given
}
```

**Feed card**
```json
{
  "type": "feed_card",
  "title": "string",
  "body": "string",
  "priority": "high | normal | low"
}
```

**Refusal** (out of scope or ambiguous without enough info)
```json
{
  "type": "refusal",
  "reason": "string"
}
```

All dates are Unix timestamps in seconds. The app injects current time into the user message so the model can resolve relative references ("tomorrow", "next Friday").

---

## 3. System Prompt

```
You are a personal assistant running on-device. Given the current time, respond only with valid JSON matching one of these types: calendar_event, notification, feed_card, or refusal. Never output anything outside the JSON object.
```

Stored in `data/system_prompt.txt`. Must match exactly at inference time.

**User message format at inference:**
```
Current time: {unix_timestamp}
{user request}
```

---

## 4. Dataset

### Record format

Same `{instruction, input, response}` → ChatML as the blog-editor pipeline:

- `instruction` — natural language request
- `input` — optional context (e.g. `"Current time: 1748880000"`)
- `response` — JSON string (one of the 4 schemas above)

### Category distribution (~300 total)

| Category | Count | Notes |
|---|---|---|
| calendar_event | 75 | Vary: single attendee, multi-attendee, all-day, with location |
| notification | 75 | Vary: relative time ("in 2 hours"), absolute time, no fireAt |
| feed_card | 60 | Vary: priority levels, short/long body |
| refusal | 50 | Clear out-of-scope (code, recipes, math) |
| ambiguous → refusal | 40 | Missing required info (no time given for reminder, etc.) |
| **Total** | **300** | |

### Files

- `data/raw/personal_assistant_dataset.json` — seed examples (~50, source of truth)
- `data/generated/batch_*.json` — Claude-generated batches, reviewed before merge
- `data/train.jsonl` — 270 ChatML records
- `data/valid.jsonl` — 30 ChatML records

Old blog-editor files (`data/raw/blog_writing_dataset.json`, `data/generated/batch_*.json`) are removed.

### `build_dataset.py` changes

Only the `SYSTEM` constant changes. All logic (comment stripping, validation, ChatML conversion, splits) is unchanged.

---

## 5. Training

`training/lora_config.yaml` changes from blog-editor config:

| Parameter | Blog-editor | Personal assistant | Reason |
|---|---|---|---|
| `iters` | 600 | 800 | JSON schema precision needs more signal |
| everything else | — | unchanged | — |

Expected runtime on M-series Mac: ~10–18 min.

---

## 6. Export

`export/export_coreml.py` — unchanged except output path:
- `model/blog-editor.mlpackage` → `model/pocket-assistant.mlpackage`

Precision, compute units, and max sequence length (512) stay the same.

### `export/test_inference.py` — 3 test prompts

1. **Calendar event:** `"Current time: 1748880000\nSchedule a team standup tomorrow at 9am"` → assert `type == "calendar_event"`
2. **Notification:** `"Current time: 1748880000\nRemind me to take my meds at 8pm"` → assert `type == "notification"`
3. **Refusal:** `"Write me a poem about the ocean"` → assert `type == "refusal"`

---

## 7. iOS Integration

Deliverables shipped to iOS team:
- `model/pocket-assistant.mlpackage`
- `data/system_prompt.txt`

App responsibility:
- Inject `Current time: {unix_timestamp}` at the start of user message
- Parse JSON response and map to iOS types (`FeedCard`, `LocalNotification`, `CalendarEvent` from `src/types/index.ts`)
- Fill system-generated fields: `id` (UUID), `timestamp` (current time), `sourceModule` (for FeedCard)

---

## 8. Out of Scope

- Context-aware Q&A (model reading AssistantContext: emails, past events) — deferred
- Multi-turn conversation — deferred
- Base model upgrade (Qwen2.5-0.5B) — revisit if JSON reliability is insufficient after baseline
- Quantization beyond fp16 — evaluate after baseline
