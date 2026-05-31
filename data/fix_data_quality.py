"""One-time script to fix data quality issues in personal_assistant_dataset.json.

Fixes:
1. 9 stale input format records: "Current time: 1748880000" → human-readable
2. 12 CoT REASON records with wrong date: "2025-06-03" → "2025-06-02"
3. 2 medication records: wrong fireAt timestamp 1748952000 → 1748894400
4. 19 REFUSAL records: stale "JSON" reference → "pipe-delimited string"
5. 3 conflict examples: rename "Team Standup"/"Daily Standup"/"Conference Call"
   in Existing events context — these exact phrases appear in probe requests and
   cause the model to hallucinate conflicts when scheduling those event types.

Run from repo root: uv run python data/fix_data_quality.py
"""
from __future__ import annotations
import json
from pathlib import Path

PATH = Path("data/raw/personal_assistant_dataset.json")

STALE_INPUT = "Current time: 1748880000"
CORRECT_INPUT = "Current time: Monday, June 2, 2025 4:00 PM"

DATE_REPLACEMENTS = [
    ("2025-06-03 00:00 UTC", "2025-06-02 16:00 UTC"),
    ("Tuesday 2025-06-03", "Monday 2025-06-02"),
    ("2025-06-03 (Tuesday)", "2025-06-02 (Monday)"),
    # plain date string — must come last to avoid double-replacing the above
    ("2025-06-03", "2025-06-02"),
]


def fix_stale_input(record: dict) -> bool:
    inp = record.get("input", "")
    if inp.startswith(STALE_INPUT):
        record["input"] = inp.replace(STALE_INPUT, CORRECT_INPUT, 1)
        return True
    return False


def fix_cot_dates(record: dict) -> bool:
    response = record.get("response", "")
    if "REASON" not in response:
        return False
    original = response
    for wrong, right in DATE_REPLACEMENTS:
        response = response.replace(wrong, right)
    if response != original:
        record["response"] = response
        return True
    return False


def fix_medication_timestamp(record: dict) -> bool:
    resp = record.get("response", "")
    if "medication" in record.get("instruction", "").lower() and "1748952000" in resp:
        record["response"] = resp.replace("1748952000", "1748894400")
        # fix CoT arithmetic text if present
        record["response"] = record["response"].replace(
            "8pm = 20 hours = 72000 seconds after midnight. 1748880000 + 72000 = 1748894400",
            "8pm = 4 hours after 4:00 PM. 1748880000 + 14400 = 1748894400",
        )
        return True
    return False


def fix_json_reference(record: dict) -> bool:
    resp = record.get("response", "")
    if resp.startswith("REFUSAL") and "json" in resp.lower():
        fixed = resp.replace(
            "I only output JSON.", "I only output pipe-delimited strings."
        ).replace(
            "I only output the defined JSON types.", "I only output the defined pipe-delimited types."
        )
        if fixed != resp:
            record["response"] = fixed
            return True
    return False


CONFLICT_EVENT_RENAMES = [
    # "Existing events" names that collide with probe scheduling requests.
    # Renaming prevents the model from hallucinating a conflict when asked to
    # schedule an event whose name matches an existing-event training label.
    ("Daily Standup", "Morning Briefing"),
    ("Team Standup", "Morning Briefing"),
    ("Conference Call", "Vendor Sync"),
]


def fix_conflict_event_names(record: dict) -> bool:
    inp = record.get("input", "")
    resp = record.get("response", "")
    if "Existing events" not in inp:
        return False
    original_inp, original_resp = inp, resp
    for wrong, right in CONFLICT_EVENT_RENAMES:
        inp = inp.replace(wrong, right)
        resp = resp.replace(wrong, right)
    if inp != original_inp or resp != original_resp:
        record["input"] = inp
        record["response"] = resp
        return True
    return False


def main() -> None:
    data: list[dict] = json.loads(PATH.read_text(encoding="utf-8"))

    counts = {
        "stale_input": 0,
        "cot_date": 0,
        "medication_ts": 0,
        "json_ref": 0,
        "conflict_event_rename": 0,
    }
    for r in data:
        if fix_stale_input(r):
            counts["stale_input"] += 1
        if fix_cot_dates(r):
            counts["cot_date"] += 1
        if fix_medication_timestamp(r):
            counts["medication_ts"] += 1
        if fix_json_reference(r):
            counts["json_ref"] += 1
        if fix_conflict_event_names(r):
            counts["conflict_event_rename"] += 1

    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Fixed {len(data)} records:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
