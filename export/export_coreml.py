"""Convert fused MLX model weights to Core ML via coremltools."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM


def export(merged_path: str, output_path: str, max_length: int) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {merged_path} ...")
    model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.float32)
    model.eval()

    example_ids = torch.zeros(1, max_length, dtype=torch.long)
    example_mask = torch.ones(1, max_length, dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(model, (example_ids, example_mask), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(output_path)
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-path", required=True, help="Path to fused model (mlx_lm.fuse output)")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    export(args.merged_path, args.output, args.max_length)
