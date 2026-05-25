# personal-assistant-model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blog-editor fine-tune with a personal assistant that outputs structured JSON for calendar events, notifications, feed cards, and refusals.

**Architecture:** Update the system prompt, swap the training dataset, bump LoRA iters, and rename export artifacts. The ChatML record format, `build_dataset.py` logic, and Core ML export chain are unchanged.

**Tech Stack:** mlx-lm, SmolLM2-360M-Instruct, coremltools, pytest, uv

---

### Task 1: Update system prompt and SYSTEM constant

**Files:**
- Modify: `data/system_prompt.txt`
- Modify: `data/build_dataset.py:41`

- [ ] **Step 1: Replace `data/system_prompt.txt`**

Overwrite the entire file with:

```
You are a personal assistant running on-device. Given the current time, respond only with valid JSON matching one of these types: calendar_event, notification, feed_card, or refusal. Never output anything outside the JSON object.
```

- [ ] **Step 2: Update `SYSTEM` in `data/build_dataset.py`**

Line 41 — change from:

```python
SYSTEM = "You are a blog writing editor. Improve, rewrite, or fix the text as instructed. If the request is out of scope, say so briefly."
```

to:

```python
SYSTEM = "You are a personal assistant running on-device. Given the current time, respond only with valid JSON matching one of these types: calendar_event, notification, feed_card, or refusal. Never output anything outside the JSON object."
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -v
```

Expected: all 18 tests pass. `TestToChatml` tests import `SYSTEM` and compare output against the same constant, so they pass regardless of the constant's value.

- [ ] **Step 4: Commit**

```bash
git add data/system_prompt.txt data/build_dataset.py
git commit -m "feat: update system prompt for personal assistant"
```

---

### Task 2: Replace raw dataset

**Files:**
- Delete: `data/raw/blog_writing_dataset.json`
- Delete: `data/generated/batch_*.json` (14 files)
- Create: `data/raw/personal_assistant_dataset.json`

- [ ] **Step 1: Delete old files**

```bash
rm data/raw/blog_writing_dataset.json
rm data/generated/batch_*.json
```

Verify:

```bash
ls data/raw/ && ls data/generated/
```

Expected: `data/raw/` contains only `__init__.py`, `data/generated/` is empty (no `.json` files).

- [ ] **Step 2: Create `data/raw/personal_assistant_dataset.json`**

Create the file with exactly this content (52 seed examples):

```json
[
  // ============================================================
  // PERSONAL ASSISTANT TRAINING DATASET
  // Reference current time for all examples: 1748880000
  // (2025-06-03 00:00:00 UTC, Tuesday)
  // All timestamps are Unix seconds in UTC.
  // Categories: calendar_event (13), notification (13),
  //             feed_card (10), refusal (9), ambiguous->refusal (7)
  // ============================================================

  // --- CALENDAR EVENTS ---
  {
    "instruction": "Schedule a team standup tomorrow at 9am for one hour.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Team Standup\", \"startDate\": 1748998800, \"endDate\": 1749002400, \"attendees\": []}"
  },
  {
    "instruction": "Book a dentist appointment next Monday at 2pm for one hour.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Dentist Appointment\", \"startDate\": 1749448800, \"endDate\": 1749452400, \"location\": \"Dentist Office\", \"attendees\": []}"
  },
  {
    "instruction": "Add a lunch meeting with Sarah tomorrow at noon for one hour.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Lunch with Sarah\", \"startDate\": 1749009600, \"endDate\": 1749013200, \"attendees\": [\"Sarah\"]}"
  },
  {
    "instruction": "Schedule a 30-minute 1:1 with Alex today at 4pm.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"1:1 with Alex\", \"startDate\": 1748937600, \"endDate\": 1748939400, \"attendees\": [\"Alex\"]}"
  },
  {
    "instruction": "Block tomorrow from 2pm to 4pm for a project review.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Project Review\", \"startDate\": 1749016800, \"endDate\": 1749024000, \"attendees\": []}"
  },
  {
    "instruction": "Add a conference call tomorrow at 10am for one hour with ana@example.com and tom@example.com.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Conference Call\", \"startDate\": 1749002400, \"endDate\": 1749006000, \"attendees\": [\"ana@example.com\", \"tom@example.com\"]}"
  },
  {
    "instruction": "Schedule a team dinner next Monday at 7pm for two hours.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Team Dinner\", \"startDate\": 1749466800, \"endDate\": 1749473600, \"attendees\": []}"
  },
  {
    "instruction": "Add a gym session tomorrow at 7am for one hour.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Gym\", \"startDate\": 1748991600, \"endDate\": 1748995200, \"attendees\": []}"
  },
  {
    "instruction": "Create an event for my flight departure next Monday at 6am at JFK airport.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Flight Departure\", \"startDate\": 1749420000, \"endDate\": 1749423600, \"location\": \"JFK Airport\", \"attendees\": []}"
  },
  {
    "instruction": "Book a doctor appointment tomorrow at 11am for 30 minutes at City Medical Center.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Doctor Appointment\", \"startDate\": 1749006000, \"endDate\": 1749007800, \"location\": \"City Medical Center\", \"attendees\": []}"
  },
  {
    "instruction": "Schedule a two-hour marketing sync next Monday at 10am.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Marketing Sync\", \"startDate\": 1749434400, \"endDate\": 1749441600, \"attendees\": []}"
  },
  {
    "instruction": "Add a birthday dinner for Mom today at 7pm for two hours.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Birthday Dinner for Mom\", \"startDate\": 1748948400, \"endDate\": 1748955600, \"attendees\": [\"Mom\"]}"
  },
  {
    "instruction": "Block tomorrow from 8am to 5pm for an off-site workshop.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"calendar_event\", \"title\": \"Off-site Workshop\", \"startDate\": 1748995200, \"endDate\": 1749027600, \"attendees\": []}"
  },

  // --- NOTIFICATIONS / REMINDERS ---
  {
    "instruction": "Remind me to take my medication at 8pm today.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Take Medication\", \"body\": \"Time to take your medication.\", \"fireAt\": 1748952000}"
  },
  {
    "instruction": "Set a reminder to call the plumber in 2 hours.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Call the Plumber\", \"body\": \"Time to call the plumber.\", \"fireAt\": 1748887200}"
  },
  {
    "instruction": "Remind me to send the project report by 5pm today.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Send Project Report\", \"body\": \"Send the project report before 5pm.\", \"fireAt\": 1748941200}"
  },
  {
    "instruction": "Alert me to leave for the airport tomorrow at 5am.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Leave for Airport\", \"body\": \"Time to leave for the airport.\", \"fireAt\": 1748984400}"
  },
  {
    "instruction": "Remind me to pick up the kids at 3pm today.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Pick Up Kids\", \"body\": \"Time to pick up the kids.\", \"fireAt\": 1748934000}"
  },
  {
    "instruction": "Set a reminder for my team standup tomorrow at 9am.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Team Standup\", \"body\": \"Team standup starts now.\", \"fireAt\": 1748998800}"
  },
  {
    "instruction": "Set an alarm for 6am tomorrow.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Wake Up Alarm\", \"body\": \"Good morning!\", \"fireAt\": 1748988000}"
  },
  {
    "instruction": "Remind me to reply to John's email in 30 minutes.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Reply to John\", \"body\": \"Reply to John's email now.\", \"fireAt\": 1748881800}"
  },
  {
    "instruction": "Notify me when it's 6pm today.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"6pm Check-in\", \"body\": \"It's 6pm.\", \"fireAt\": 1748944800}"
  },
  {
    "instruction": "Remind me to submit the expense form before end of business today.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Submit Expense Form\", \"body\": \"Submit the expense form before end of business.\", \"fireAt\": 1748941200}"
  },
  {
    "instruction": "Set a reminder for my anniversary dinner reservation tonight at 7:30pm.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Anniversary Dinner\", \"body\": \"Head to your anniversary dinner reservation.\", \"fireAt\": 1748950200}"
  },
  {
    "instruction": "Remind me to take a break in one hour.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Take a Break\", \"body\": \"Time to step away from your desk.\", \"fireAt\": 1748883600}"
  },
  {
    "instruction": "Remind me to water the plants tomorrow morning at 8am.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"notification\", \"title\": \"Water the Plants\", \"body\": \"Time to water the plants.\", \"fireAt\": 1748995200}"
  },

  // --- FEED CARDS ---
  {
    "instruction": "Note that the quarterly report is ready for review.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Quarterly Report Ready\", \"body\": \"The quarterly report is ready for your review.\", \"priority\": \"normal\"}"
  },
  {
    "instruction": "Add a card: the API documentation was updated today.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"API Docs Updated\", \"body\": \"The API documentation was updated today.\", \"priority\": \"low\"}"
  },
  {
    "instruction": "Create a note that we need to order office supplies.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Order Office Supplies\", \"body\": \"We need to order office supplies.\", \"priority\": \"low\"}"
  },
  {
    "instruction": "Flag this as important: the client deadline moved to Friday.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Client Deadline Moved\", \"body\": \"The client deadline has moved to Friday.\", \"priority\": \"high\"}"
  },
  {
    "instruction": "Add a high priority card: the production server is down.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Production Server Down\", \"body\": \"The production server is down. Immediate action required.\", \"priority\": \"high\"}"
  },
  {
    "instruction": "Save a note that John's feedback on the proposal was positive.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Positive Feedback from John\", \"body\": \"John's feedback on the proposal was positive.\", \"priority\": \"normal\"}"
  },
  {
    "instruction": "Create a low priority reminder to update the team wiki.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Update Team Wiki\", \"body\": \"Remember to update the team wiki when you get a chance.\", \"priority\": \"low\"}"
  },
  {
    "instruction": "Add a card: the design files are now available in Figma.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Design Files in Figma\", \"body\": \"The design files are now available in Figma.\", \"priority\": \"normal\"}"
  },
  {
    "instruction": "Mark as important: budget approval received.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Budget Approved\", \"body\": \"Budget approval has been received.\", \"priority\": \"high\"}"
  },
  {
    "instruction": "Create a card: team lunch is rescheduled to Thursday.",
    "input": "",
    "response": "{\"type\": \"feed_card\", \"title\": \"Team Lunch Rescheduled\", \"body\": \"Team lunch has been rescheduled to Thursday.\", \"priority\": \"normal\"}"
  },

  // --- REFUSALS ---
  {
    "instruction": "Write a poem about the ocean.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Writing poems is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "What's the capital of France?",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Answering general knowledge questions is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "Help me debug this Python function.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Debugging code is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "Generate a cover letter for my job application.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Writing cover letters is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "What is 256 multiplied by 17?",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Performing arithmetic is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "Translate 'hello world' to Spanish.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Translation is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "What's the weather like today?",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I don't have access to weather data. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "Summarize this article for me.",
    "input": "The quick brown fox jumps over the lazy dog. This sentence has been used as a typing test for decades.",
    "response": "{\"type\": \"refusal\", \"reason\": \"Summarizing articles is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },
  {
    "instruction": "Tell me a joke.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"Telling jokes is outside my scope. I can schedule events, set reminders, or create feed cards.\"}"
  },

  // --- AMBIGUOUS → REFUSAL ---
  {
    "instruction": "Set a reminder.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What should I remind you about, and when?\"}"
  },
  {
    "instruction": "Schedule a meeting.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What is the meeting about, and when should it be scheduled?\"}"
  },
  {
    "instruction": "Add a calendar event for next week.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. Which day next week and at what time?\"}"
  },
  {
    "instruction": "Remind me about it later.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What should I remind you about, and when exactly?\"}"
  },
  {
    "instruction": "Create a card.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What should the card say?\"}"
  },
  {
    "instruction": "Book something for tomorrow.",
    "input": "Current time: 1748880000",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What would you like to book and at what time?\"}"
  },
  {
    "instruction": "Remind me in a bit.",
    "input": "",
    "response": "{\"type\": \"refusal\", \"reason\": \"I need more details. What should I remind you about, and how long is 'a bit'?\"}"
  }
]
```

- [ ] **Step 3: Smoke-test the dataset pipeline with reduced split**

The seed has 52 records — below `TRAIN_N=270`. Use `--train-n` to test with the seed:

```bash
uv run python data/build_dataset.py --train-n 44
```

Expected output:
```
Written: 44 train, 8 valid
```

Verify the first record is personal-assistant format:

```bash
python -c "import json; r = json.loads(open('data/train.jsonl').readline()); print(r['messages'][0]['content'][:60])"
```

Expected: `You are a personal assistant running on-device.`

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ -v
```

Expected: all 18 tests pass.

> **Note:** Training requires ~300 records. Before running `make train`, generate additional batches and place them in `data/generated/`. Once total records ≥ 300, run `make dataset` (uses `TRAIN_N=270`). Use the same `{instruction, input, response}` record format with the same 5 categories and reference timestamps.

- [ ] **Step 5: Commit**

```bash
git add data/raw/personal_assistant_dataset.json
git commit -m "feat: add personal assistant seed dataset (52 examples)"
```

---

### Task 3: Update LoRA training config

**Files:**
- Modify: `training/lora_config.yaml:17`

- [ ] **Step 1: Bump iters from 600 to 800**

In `training/lora_config.yaml`, change:

```yaml
iters: 600
```

to:

```yaml
iters: 800
```

- [ ] **Step 2: Commit**

```bash
git add training/lora_config.yaml
git commit -m "feat: bump LoRA iters to 800 for JSON schema precision"
```

---

### Task 4: Update export pipeline

**Files:**
- Modify: `Makefile` (lines 29–44: `gguf` and `export` / `test-export` targets)
- Modify: `export/test_inference.py` (TEST_CASES + build_prompt)
- Modify: `export/export_gguf.py` (docstring + print)

- [ ] **Step 1: Update Makefile artifact paths**

In `Makefile`, change the `gguf` target:

```makefile
gguf:
	uv run python export/export_gguf.py \
		--merged-path model/merged \
		--output model/blog-editor-q4.gguf
```

to:

```makefile
gguf:
	uv run python export/export_gguf.py \
		--merged-path model/merged \
		--output model/pocket-assistant-q4.gguf
```

Change the `export` target:

```makefile
export:
	uv run python export/export_coreml.py \
		--merged-path model/merged \
		--output model/blog-editor.mlpackage \
		--max-length 512
```

to:

```makefile
export:
	uv run python export/export_coreml.py \
		--merged-path model/merged \
		--output model/pocket-assistant.mlpackage \
		--max-length 512
```

Change the `test-export` target:

```makefile
test-export:
	uv run python export/test_inference.py \
		--model model/blog-editor.mlpackage \
		--tokenizer model/merged \
		--max-length 512
```

to:

```makefile
test-export:
	uv run python export/test_inference.py \
		--model model/pocket-assistant.mlpackage \
		--tokenizer model/merged \
		--max-length 512
```

- [ ] **Step 2: Update `export/export_gguf.py` docstring and print**

At the top of the file, change the module docstring from:

```python
"""
Convert fused HuggingFace weights to GGUF Q4_K_M for rnllama (React Native).

Requires llama.cpp installed:
  brew install llama.cpp

Run after `make fuse`:
  make gguf
  # output: model/blog-editor-q4.gguf (~230 MB)

Then copy to React Native app:
  cp model/blog-editor-q4.gguf ../blog-notifier/assets/models/
"""
```

to:

```python
"""
Convert fused HuggingFace weights to GGUF Q4_K_M for rnllama (React Native).

Requires llama.cpp installed:
  brew install llama.cpp

Run after `make fuse`:
  make gguf
  # output: model/pocket-assistant-q4.gguf (~230 MB)

Then copy to React Native app:
  cp model/pocket-assistant-q4.gguf ../pocket-assistant/assets/models/
"""
```

On line 87, change:

```python
    print(f"\nNext: cp {out} ../blog-notifier/assets/models/")
```

to:

```python
    print(f"\nNext: cp {out} ../pocket-assistant/assets/models/")
```

- [ ] **Step 3: Update `export/test_inference.py`**

Replace `TEST_CASES` and `build_prompt` to match the personal assistant format.

Change `TEST_CASES` from:

```python
TEST_CASES = [
    ("Make this more concise.", "In this blog post I am going to be talking about various ways to improve content."),
    ("Write a full 1500 word blog post about SEO.", ""),
    ("Fix this.", "blog good write"),
]
```

to:

```python
CURRENT_TIME = 1748880000  # fixed reference for smoke tests

TEST_CASES = [
    ("Schedule a team standup tomorrow at 9am for one hour.", f"Current time: {CURRENT_TIME}"),
    ("Remind me to take my medication at 8pm today.", f"Current time: {CURRENT_TIME}"),
    ("Write a poem about the ocean.", ""),
]
```

Change `build_prompt` from:

```python
def build_prompt(instruction: str, input_text: str) -> str:
    user = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

to:

```python
def build_prompt(instruction: str, input_text: str) -> str:
    user = f"{input_text}\n{instruction}" if input_text.strip() else instruction
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

The prompt format now matches training: context line (`Current time: ...`) comes first, then the instruction.

- [ ] **Step 4: Commit**

```bash
git add Makefile export/test_inference.py export/export_gguf.py
git commit -m "feat: rename export artifacts to pocket-assistant, update smoke test prompts"
```

---

## Self-review checklist (already resolved)

- System prompt: updated in both `system_prompt.txt` and `build_dataset.py` ✓
- Output schemas: aligned to iOS types in `src/types/index.ts` ✓
- Dataset seed: 52 examples across all 5 categories ✓
- Training: iters bumped to 800 ✓
- Export: all `blog-editor` references renamed to `pocket-assistant` ✓
- Smoke tests: prompts use personal assistant format with `Current time:` prefix ✓
- `TRAIN_N` note: training blocked until dataset reaches 300+ records ✓
- `build_dataset.py` pipeline logic: unchanged ✓
