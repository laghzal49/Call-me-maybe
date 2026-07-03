CACHE_DIR = goinfre/
SRC_DIR = src
UV_ENV = UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(HOME)/$(CACHE_DIR)/.venv"
HF_ENV = HF_HOME="$(HOME)/$(CACHE_DIR)" UV_PROJECT_ENVIRONMENT="$(HOME)/$(CACHE_DIR)/.venv"

all: install run

add:
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" uv add pydantic numpy ./llm_sdk accelerate
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" uv add --dev flake8 mypy

install:
	@$(UV_ENV) uv sync

run:
	@$(HF_ENV) uv run python3 -m $(SRC_DIR)

test:
	@$(HF_ENV) uv run python3 -m src --input data/input/invalid_test_case.json

debug:
	@$(HF_ENV) uv run python3 -m pdb -m $(SRC_DIR)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache

lint:
	@$(UV_ENV) uv run flake8 src/
	@$(UV_ENV) uv run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@$(UV_ENV) uv run flake8 src/
	@$(UV_ENV) uv run mypy src/ --strict
