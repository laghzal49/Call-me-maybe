SRC_DIR = src

CACHE_DIR := $(if $(wildcard /goinfre/.),/goinfre/$(or $(USER),$(shell whoami)),$(HOME)/.cache/call-me-maybe)
UV_ENV = UV_CACHE_DIR="$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(CACHE_DIR)/.venv"
HF_ENV = HF_HOME="$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(CACHE_DIR)/.venv"
LINT_ENV = $(UV_ENV) MYPY_CACHE_DIR="$(CACHE_DIR)/.mypy_cache"

.PHONY: all install run debug clean fclean cache-clean lint lint-strict

all: install run

install:
	@$(UV_ENV) uv sync

run:
	@$(HF_ENV) uv run python3 -m $(SRC_DIR)

debug:
	@$(HF_ENV) uv run python3 -m pdb -m $(SRC_DIR)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf data/output

fclean: clean
	@rm -rf .venv

cache-clean:
	@rm -rf $(CACHE_DIR)

lint:
	@$(LINT_ENV) uv run flake8 .
	@$(LINT_ENV) uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@$(LINT_ENV) uv run flake8 .
	@$(LINT_ENV) uv run mypy . --strict
