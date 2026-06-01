"""
Migrate training data from absolute Unix timestamps to anchor + offset format.

CALENDAR|title|startUnix|endUnix|... → CALENDAR|title|anchor|startOffset|endOffset|...
NOTIFICATION|title|body|fireAtUnix   → NOTIFICATION|title|body|anchor|fireAtOffset

anchor: "tomorrow", weekday name ("Friday"), "next Monday", or empty for same-day offsets
offset: seconds from midnight of anchor day; or seconds from now when anchor is empty

Uses REF_TIME as origin to preserve the intended day/time semantics from training data
(training records were authored as REF_TIME + N_days*86400 + hour*3600).
"""

from __future__ import annotations
import json
import re
from datetime import date, timedelta
from pathlib import Path

REF_TIME = 1748880000  # Monday, June 2, 2025 4:00 PM UTC
REF_DATE = date(2025, 6, 2)  # Monday

UNIX_RE = re.compile(r"^\d{9,10}$")


def days_to_anchor(days: int) -> str:
    if days == 0:
        return ""
    if days == 1:
        return "tomorrow"
    target = REF_DATE + timedelta(days=days)
    if days <= 7:
        return target.strftime("%A")  # "Wednesday", "Friday", "Monday"
    return f"in {days} days"


def ts_to_anchor_offset(ts_str: str) -> tuple[str, str] | None:
    v = ts_str.strip()
    if not UNIX_RE.match(v):
        return None
    ts = int(v)
    delta = ts - REF_TIME
    if delta < 0:
        return None
    days = delta // 86400
    time_of_day = delta % 86400
    return days_to_anchor(days), str(time_of_day)


def convert_response(response: str) -> str:
    reason_idx = response.find("|REASON|")
    main = response[:reason_idx] if reason_idx != -1 else response
    reason = response[reason_idx:] if reason_idx != -1 else ""

    parts = main.split("|")
    kind = parts[0]

    if kind == "CALENDAR" and len(parts) >= 4:
        start_v = parts[2].strip()
        end_v = parts[3].strip()
        if UNIX_RE.match(start_v) and UNIX_RE.match(end_v):
            start = ts_to_anchor_offset(start_v)
            end = ts_to_anchor_offset(end_v)
            if start and end:
                anchor, start_off = start
                _, end_off = end
                tail = parts[4:]
                return "|".join([parts[0], parts[1], anchor, start_off, end_off] + tail) + reason
        return response

    if kind == "NOTIFICATION" and len(parts) >= 4:
        fire_v = parts[3].strip()
        if UNIX_RE.match(fire_v):
            result = ts_to_anchor_offset(fire_v)
            if result:
                anchor, fire_off = result
                tail = parts[4:]
                return "|".join([parts[0], parts[1], parts[2], anchor, fire_off] + tail) + reason
        return response

    return response


def migrate_file(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for record in data:
        old = record.get("response", "")
        new = convert_response(old)
        if new != old:
            record["response"] = new
            changed += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


if __name__ == "__main__":
    raw = Path("data/raw")
    for f in sorted(raw.glob("*.json")):
        n = migrate_file(f)
        print(f"{f.name}: {n} records updated")
