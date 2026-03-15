SRC_DIR = src


.PHONY: all install run debug clean fclean cache-clean lint lint-strict

all: install run

install:
	uv sync

run:
	uv run python3 -m $(SRC_DIR)

debug:
	uv run python3 -m pdb -m $(SRC_DIR)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf data/output

fclean: clean
	@rm -rf .venv

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
