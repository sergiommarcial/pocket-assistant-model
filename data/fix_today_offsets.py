"""
Fix stale midnight-based offsets in NOTIFICATION records with empty anchor.

In anchor+offset format, empty anchor means offset is seconds from NOW.
Many records were authored with offsets that are seconds from midnight (clock times).

REF_TOD = 57600 = 4pm in seconds from midnight (16 * 3600).

For same-day clock times with empty anchor:
  offset > REF_TOD  → future: new_offset = offset - REF_TOD
  offset == REF_TOD → now:    new_offset = 0
  offset < REF_TOD  → past:   convert to REFUSAL (if offset > FIX_THRESHOLD)

FIX_THRESHOLD = 7200: offsets <= 2h from now are genuine relative offsets;
leave those alone (1800=30min, 3600=1h, 7200=2h are all valid durations).
"""

from __future__ import annotations
import json
from pathlib import Path

REF_TOD = 57600       # 4pm in seconds from midnight
FIX_THRESHOLD = 7200  # offsets <= 2h kept as genuine relative durations


def fix_response(response: str) -> str:
    reason_idx = response.find("|REASON|")
    main = response[:reason_idx] if reason_idx != -1 else response

    parts = main.split("|")
    if parts[0] != "NOTIFICATION" or len(parts) < 5:
        return response

    if parts[3].strip():
        return response  # has anchor, nothing to fix

    try:
        offset = int(parts[4].strip())
    except ValueError:
        return response

    if offset <= FIX_THRESHOLD:
        return response  # genuine short duration

    new_offset = offset - REF_TOD

    if new_offset < 0:
        reason_suffix = "|REASON|Current time is Monday June 2, 2025 4:00 PM. That time has already passed." if reason_idx != -1 else ""
        return f"REFUSAL|I can't set reminders in the past. That time has already passed.{reason_suffix}"

    parts[4] = str(new_offset)
    # Drop stale REASON text — the offset semantics changed, old arithmetic is wrong.
    return "|".join(parts)


def main() -> None:
    path = Path("data/raw/personal_assistant_dataset.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    changed = 0
    for record in data:
        old = record.get("response", "")
        new = fix_response(old)
        if new != old:
            record["response"] = new
            changed += 1
            print(f"  {old!r}")
            print(f"→ {new!r}")
            print()

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{changed} records updated.")


if __name__ == "__main__":
    main()
