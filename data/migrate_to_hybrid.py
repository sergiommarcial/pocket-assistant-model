"""Convert CALENDAR/NOTIFICATION time fields from human-readable back to Unix timestamps.

Migrates personal_assistant_dataset.json responses from the intermediate
human-readable pipe format to the hybrid format:
  - Input: human-readable current time (unchanged)
  - Output: Unix timestamps for CALENDAR start/end and NOTIFICATION fireAt

Run from repo root: uv run python data/migrate_to_hybrid.py
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _strip_js_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                result.append(ch)
                result.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            result.append(ch)
        else:
            if ch == '"':
                in_string = True
                result.append(ch)
            elif ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            else:
                result.append(ch)
        i += 1
    return "".join(result)


def _human_to_ts(s: str) -> int:
    dt = datetime.strptime(s.strip(), "%A, %B %d, %Y %I:%M %p")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _parse_duration(d: str) -> int:
    total = 0
    m = re.search(r"(\d+)h", d)
    if m:
        total += int(m.group(1)) * 3600
    m = re.search(r"(\d+)min", d)
    if m:
        total += int(m.group(1)) * 60
    return total


def _to_hybrid(response: str) -> str:
    reason_suffix = ""
    if "|REASON|" in response:
        idx = response.index("|REASON|")
        reason_suffix = response[idx:]
        response = response[:idx]

    parts = response.split("|")
    t = parts[0]

    if t == "CALENDAR":
        title = parts[1]
        start_ts = _human_to_ts(parts[2])
        end_ts = start_ts + _parse_duration(parts[3])
        location = parts[4] if len(parts) > 4 else ""
        attendees = parts[5] if len(parts) > 5 else ""
        return f"CALENDAR|{title}|{start_ts}|{end_ts}|{location}|{attendees}{reason_suffix}"

    if t == "NOTIFICATION":
        title, body = parts[1], parts[2]
        fire_ts = _human_to_ts(parts[3])
        return f"NOTIFICATION|{title}|{body}|{fire_ts}{reason_suffix}"

    return response + reason_suffix


def main() -> None:
    path = Path("data/raw/personal_assistant_dataset.json")
    records: list[dict] = json.loads(_strip_js_comments(path.read_text()))
    for r in records:
        r["response"] = _to_hybrid(r["response"])
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Migrated {len(records)} records → {path}")


if __name__ == "__main__":
    main()
