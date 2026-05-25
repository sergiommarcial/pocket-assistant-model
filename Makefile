.PHONY: install dataset test train fuse quantize gguf export test-export probe

install:
	uv sync --group dev

dataset:
	uv run python data/build_dataset.py

test:
	uv run pytest tests/ -v

train:
	uv run mlx_lm.lora --config training/lora_config.yaml

fuse:
	uv run mlx_lm.fuse \
		--model HuggingFaceTB/SmolLM2-360M-Instruct \
		--adapter-path model/adapters \
		--save-path model/merged

quantize:
	uv run mlx_lm.convert \
		--hf-path model/merged \
		--mlx-path model/quantized \
		--quantize \
		--q-bits 4

gguf:
	uv run python export/export_gguf.py \
		--merged-path model/merged \
		--output model/pocket-assistant-q4.gguf

export:
	uv run python export/export_coreml.py \
		--merged-path model/merged \
		--output model/pocket-assistant.mlpackage \
		--max-length 512

test-export:
	uv run python export/test_inference.py \
		--model model/pocket-assistant.mlpackage \
		--tokenizer model/merged \
		--max-length 512

probe:
	uv run python export/probe.py --model model/merged
