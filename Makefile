# Variables
GOINFRE_USER   = /goinfre/tlaghzal
CACHE_DIR      = $(GOINFRE_USER)/uv-cache
VENV_DIR       = $(GOINFRE_USER)/call_venv
HF_HOME_DIR    = $(GOINFRE_USER)/hf-cache
CPU_INDEX      = https://download.pytorch.org/whl/cpu

.PHONY: install run debug clean lint lint-strict

install:
	@echo "Creating storage directories in goinfre..."
	@mkdir -p $(CACHE_DIR)
	@mkdir -p $(VENV_DIR)
	@mkdir -p $(HF_HOME_DIR)
	@if [ ! -L .venv ]; then \
		echo "Creating symbolic link for .venv..."; \
		rm -rf .venv; \
		ln -s $(VENV_DIR) .venv; \
	fi
	@echo "Installing dependencies..."
	@export UV_CACHE_DIR=$(CACHE_DIR); \
	unset TMPDIR; \
	uv sync --extra-index-url $(CPU_INDEX)
	@echo "Setup complete!"

run:
	@export UV_CACHE_DIR=$(CACHE_DIR); \
	export HF_HOME=$(HF_HOME_DIR); \
	unset TMPDIR; \
	uv run python -m src $(ARGS)

debug:
	@export UV_CACHE_DIR=$(CACHE_DIR); \
	export HF_HOME=$(HF_HOME_DIR); \
	unset TMPDIR; \
	uv run python -m pdb -m src $(ARGS)

clean:
	@rm -rf .venv
	@rm -rf .mypy_cache
	@export UV_CACHE_DIR=$(CACHE_DIR); uv cache clean
	@find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	@export UV_CACHE_DIR=$(CACHE_DIR); \
	unset TMPDIR; \
	uv run flake8 . && uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@export UV_CACHE_DIR=$(CACHE_DIR); \
	unset TMPDIR; \
	uv run flake8 . && uv run mypy . --strict
