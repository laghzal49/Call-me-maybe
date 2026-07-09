SRC_DIR = src

# Use 42's per-user /goinfre cache when available (bigger quota than $HOME);
# fall back to a plain home cache dir on any other machine.
CACHE_DIR := $(if $(wildcard /goinfre/.),/goinfre/$(USER),$(HOME)/.cache/call-me-maybe)
UV_ENV = UV_CACHE_DIR="$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(CACHE_DIR)/.venv"
HF_ENV = HF_HOME="$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(CACHE_DIR)/.venv"

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

fclean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf data/output
	@rm -rf .venv

lint:
	@$(UV_ENV) uv run flake8 .
	@$(UV_ENV) uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@$(UV_ENV) uv run flake8 .
	@$(UV_ENV) uv run mypy . --strict
