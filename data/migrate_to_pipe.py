"""One-time migration: convert personal_assistant_dataset.json from JSON responses
to pipe-delimited format and Unix timestamps to human-readable dates.

Run from repo root: uv run python data/migrate_to_pipe.py
"""
from __future__ import annotations
import json
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


def _ts_to_human(ts: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour, minute = dt.hour, dt.minute
    if hour == 0:
        time_str = f"12:{minute:02d} AM"
    elif hour < 12:
        time_str = f"{hour}:{minute:02d} AM"
    elif hour == 12:
        time_str = f"12:{minute:02d} PM"
    else:
        time_str = f"{hour - 12}:{minute:02d} PM"
    return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year} {time_str}"


def _duration(seconds: int) -> str:
    h, m = divmod(seconds // 60, 60)
    if h and m:
        return f"{h}h {m}min"
    return f"{h}h" if h else f"{m}min"


def _to_pipe(response_str: str) -> str:
    r = json.loads(response_str)
    t = r["type"]
    reasoning = r.get("reasoning", "")
    suffix = f"|REASON|{reasoning}" if reasoning else ""

    if t == "calendar_event":
        title = r["title"]
        start = _ts_to_human(r["startDate"])
        duration = _duration(r["endDate"] - r["startDate"])
        location = r.get("location", "")
        attendees = ",".join(r.get("attendees", []))
        return f"CALENDAR|{title}|{start}|{duration}|{location}|{attendees}{suffix}"

    if t == "notification":
        return f"NOTIFICATION|{r['title']}|{r['body']}|{_ts_to_human(r['fireAt'])}{suffix}"

    if t == "feed_card":
        return f"FEED_CARD|{r['title']}|{r['body']}|{r['priority']}{suffix}"

    if t == "refusal":
        return f"REFUSAL|{r['reason']}{suffix}"

    raise ValueError(f"Unknown type: {t}")


def _convert_input(input_str: str) -> str:
    if not input_str or not input_str.startswith("Current time:"):
        return input_str
    raw = input_str.replace("Current time:", "").strip()
    if not raw.isdigit():
        return input_str
    return f"Current time: {_ts_to_human(int(raw))}"


def main() -> None:
    path = Path("data/raw/personal_assistant_dataset.json")
    records: list[dict] = json.loads(_strip_js_comments(path.read_text()))

    for r in records:
        if r.get("input"):
            r["input"] = _convert_input(r["input"])
        r["response"] = _to_pipe(r["response"])

    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Converted {len(records)} records → {path}")


if __name__ == "__main__":
    main()
