.PHONY: install dataset test train fuse quantize gguf export test-export probe full_train help
.DEFAULT_GOAL := help

CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
BOLD   := \033[1m
DIM    := \033[2m
RESET  := \033[0m

help:
	@printf "$(BOLD)pocket-assistant-model$(RESET)\n\n"
	@printf "$(YELLOW)Setup$(RESET)\n"
	@printf "  $(CYAN)install$(RESET)       $(DIM)Install dependencies (uv sync --group dev)$(RESET)\n"
	@printf "\n$(YELLOW)Data & training$(RESET)\n"
	@printf "  $(CYAN)dataset$(RESET)       $(DIM)Build train/valid JSONL from data/raw/$(RESET)\n"
	@printf "  $(CYAN)train$(RESET)         $(DIM)LoRA fine-tune (800 iters)$(RESET)\n"
	@printf "  $(CYAN)fuse$(RESET)          $(DIM)Merge adapters into model/merged/$(RESET)\n"
	@printf "\n$(YELLOW)Validation$(RESET)\n"
	@printf "  $(CYAN)test$(RESET)          $(DIM)Run pytest unit tests$(RESET)\n"
	@printf "  $(CYAN)probe$(RESET)         $(DIM)Run 76 behavioral probes against model/merged$(RESET)\n"
	@printf "  $(CYAN)test-export$(RESET)   $(DIM)Smoke-test .mlpackage against 3 prompts$(RESET)\n"
	@printf "\n$(YELLOW)Pipelines$(RESET)\n"
	@printf "  $(CYAN)full_train$(RESET)    $(DIM)dataset → train → fuse → probe$(RESET)\n"
	@printf "\n$(YELLOW)Export$(RESET)\n"
	@printf "  $(CYAN)export$(RESET)        $(DIM)Convert to Core ML .mlpackage (iOS 17+)$(RESET)\n"
	@printf "  $(CYAN)quantize$(RESET)      $(DIM)4-bit quantize → model/quantized/$(RESET)\n"
	@printf "  $(CYAN)gguf$(RESET)          $(DIM)Export to GGUF (llama.cpp / Ollama)$(RESET)\n"
	@printf "\n"

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

full_train:
	make dataset && make train && make fuse && make probe