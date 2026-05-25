"""Smoke-test a converted .mlpackage: load it and run 3 test prompts."""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

SYSTEM = Path("data/system_prompt.txt").read_text().strip()

CURRENT_TIME = 1748880000  # fixed reference for smoke tests

TEST_CASES = [
    (
        "Schedule a team standup tomorrow at 9am for one hour.",
        f"Current time: {CURRENT_TIME}",
    ),
    ("Remind me to take my medication at 8pm today.", f"Current time: {CURRENT_TIME}"),
    ("Write a poem about the ocean.", ""),
]


def build_prompt(instruction: str, input_text: str) -> str:
    user = f"{input_text}\n{instruction}" if input_text.strip() else instruction
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
        tokens = tokenizer(
            prompt, return_tensors="np", truncation=True, max_length=max_length
        )
        n = tokens["input_ids"].shape[1]

        input_ids = np.zeros((1, max_length), dtype=np.int32)
        attn_mask = np.zeros((1, max_length), dtype=np.int32)
        input_ids[0, :n] = tokens["input_ids"][0]
        attn_mask[0, :n] = 1

        result = mlmodel.predict({"input_ids": input_ids, "attention_mask": attn_mask})
        assert (
            result.get("logits") is not None
        ), f"No logits in output: {list(result.keys())}"
        print(
            f"PASS | logits shape: {result['logits'].shape} | prompt: {instruction[:40]!r}"
        )

    print("\nAll 3 smoke tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .mlpackage")
    parser.add_argument(
        "--tokenizer", required=True, help="Path to merged model (for tokenizer)"
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    main(args.model, args.tokenizer, args.max_length)
