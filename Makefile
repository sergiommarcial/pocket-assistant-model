.PHONY: install dataset test train fuse quantize gguf export test-export probe full_train clean help
.DEFAULT_GOAL := help

CYAN  := \033[96m
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[92m
YELLOW := \033[93m

VENV            := .venv
PYTHON          := $(VENV)/bin/python
UV_RUN          := uv run --python $(PYTHON)
REQUIRED_PYTHON := 3.11

help: ## Show this help message
	@echo "$(BOLD)pocket-assistant-model$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-14s$(RESET) %s\n", $$1, $$2}'

check-python: ## Verify the venv is running Python $(REQUIRED_PYTHON)
	@ACTUAL=$$($(PYTHON) --version 2>&1 | awk '{print $$2}'); \
	MAJOR_MINOR=$$(echo "$$ACTUAL" | cut -d. -f1-2); \
	if [ "$$MAJOR_MINOR" = "$(REQUIRED_PYTHON)" ]; then \
		echo "$(GREEN)Python $$ACTUAL — OK$(RESET)"; \
	else \
		echo "$(YELLOW)Expected Python $(REQUIRED_PYTHON).x but found $$ACTUAL$(RESET)"; \
		exit 1; \
	fi

install: check-python ## Sync dependencies including dev group
	uv sync --group dev

format: check-python ## Auto-format src/ with black
	$(UV_RUN) black export/ tests/

lint: check-python ## Check src/ formatting without modifying files
	$(UV_RUN) black --check export/ tests/
	$(UV_RUN) bandit -r export/ tests/
	$(UV_RUN) pyflakes export/ tests/
	$(UV_RUN) vulture export/ tests/

dataset: check-python ## Build train/valid JSONL from data/raw/
	$(UV_RUN) python data/build_dataset.py

test: check-python ## Run pytest unit tests
	$(UV_RUN) pytest tests/ -v

train: check-python ## LoRA fine-tune (800 iters, writes to model/adapters/)
	$(UV_RUN) mlx_lm.lora --config training/lora_config.yaml

fuse: check-python ## Merge adapters into model/merged/
	$(UV_RUN) mlx_lm.fuse \
		--model HuggingFaceTB/SmolLM2-360M-Instruct \
		--adapter-path model/adapters \
		--save-path model/merged

quantize: check-python ## 4-bit quantize model/merged → model/quantized/
	$(UV_RUN) mlx_lm.convert \
		--hf-path model/merged \
		--mlx-path model/quantized \
		--quantize \
		--q-bits 4

gguf: check-python ## Export fused model to GGUF (llama.cpp / Ollama)
	$(UV_RUN) python export/export_gguf.py \
		--merged-path model/merged \
		--output model/pocket-assistant-q4.gguf

export: check-python ## Convert to Core ML .mlpackage (iOS 17+, max-length 512)
	$(UV_RUN) python export/export_coreml.py \
		--merged-path model/merged \
		--output model/pocket-assistant.mlpackage \
		--max-length 512

test-export: check-python ## Smoke-test .mlpackage against 3 prompts
	$(UV_RUN) python export/test_inference.py \
		--model model/pocket-assistant.mlpackage \
		--tokenizer model/merged \
		--max-length 512

probe: check-python ## Inspect model/merged weights and config
	$(UV_RUN) python export/probe.py --model model/merged

full_train: ## Run full pipeline: dataset → train → fuse → probe
	$(MAKE) dataset && $(MAKE) train && $(MAKE) fuse && $(MAKE) probe

clean: ## Remove build artifacts and caches
	rm -rf model/quantized/ model/merged/ model/*.mlpackage model/*.gguf \
		$(shell find . -name "__pycache__" -o -name "*.pyc") 2>/dev/null; true
