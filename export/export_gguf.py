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

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_convert_script() -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py."""
    candidates = [
        Path("/opt/homebrew/bin/convert_hf_to_gguf.py"),
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/opt/homebrew/opt/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: find via brew --prefix
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "llama.cpp"], text=True
        ).strip()
        candidate = Path(prefix) / "share/llama.cpp/convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Install llama.cpp:\n  brew install llama.cpp"
    )


def find_quantize_bin() -> str:
    """Locate llama-quantize binary."""
    for name in ("llama-quantize", "quantize"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "llama-quantize not found. Install llama.cpp:\n  brew install llama.cpp"
    )


def export(merged_path: str, output: str) -> None:
    merged = Path(merged_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp16_gguf = out.with_suffix(".fp16.gguf")

    convert_script = find_convert_script()
    quantize_bin = find_quantize_bin()

    # Step 1: HuggingFace → GGUF fp16
    print(f"Step 1: Converting {merged} → {fp16_gguf} (fp16) ...")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(merged),
            "--outfile",
            str(fp16_gguf),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    # Step 2: GGUF fp16 → GGUF Q4_K_M
    print(f"Step 2: Quantizing {fp16_gguf} → {out} (Q4_K_M) ...")
    subprocess.run(
        [quantize_bin, str(fp16_gguf), str(out), "Q4_K_M"],
        check=True,
    )

    fp16_gguf.unlink()  # remove intermediate
    size_mb = out.stat().st_size / (1024**2)
    print(f"Done: {out}  ({size_mb:.0f} MB)")
    print(f"\nNext: cp {out} ../pocket-assistant/assets/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.merged_path, args.output)
