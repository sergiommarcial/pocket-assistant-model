from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any


def strip_js_comments(text: str) -> str:
    """
    Remove JS-style // comments from JSON text.

    This function processes JSON files (which use only double-quoted strings)
    and strips // comments that appear outside of strings. Single-quoted
    strings are not tracked because they are invalid in JSON and never appear
    in the input.

    Args:
        text: JSON text potentially containing JS-style // comments

    Returns:
        The input text with // comments removed (everything from // to end of line)
    """
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


SYSTEM = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8").strip()


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


PATCHES_FILE = "targeted_patches.json"


def main(data_dir: Path = Path("data"), train_n: int = TRAIN_N) -> None:
    raw_dir = data_dir / "raw"
    generated_dir = data_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # targeted_patches.json always goes to train — these are failure-fix records
    # that must be in the training set to have any effect.
    patches_path = raw_dir / PATCHES_FILE
    patches: list[dict] = []
    if patches_path.exists():
        patches = load_json_file(patches_path)
        for record in patches:
            validate_record(record)

    base_records: list[dict] = []
    for json_file in sorted(raw_dir.glob("*.json")) + sorted(generated_dir.glob("*.json")):
        if json_file == patches_path:
            continue
        batch = load_json_file(json_file)
        for record in batch:
            validate_record(record)
        base_records.extend(batch)

    base_train_n = max(0, train_n - len(patches))
    base_train, valid_records = build_splits(base_records, train_n=base_train_n)
    # Shuffle patches into base_train so they're distributed across the epoch,
    # not front-loaded — front-loading can bias early gradient updates.
    combined = patches + base_train
    random.seed(43)
    random.shuffle(combined)
    train_records = combined

    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(to_chatml(r, SYSTEM)) for r in train_records) + "\n",
        encoding="utf-8",
    )
    (data_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(to_chatml(r, SYSTEM)) for r in valid_records) + "\n",
        encoding="utf-8",
    )
    print(f"Written: {len(train_records)} train ({len(patches)} patches + {len(base_train)} base), {len(valid_records)} valid")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--train-n", default=TRAIN_N, type=int)
    args = parser.parse_args()
    main(args.data_dir, args.train_n)
