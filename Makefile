# Minimal developer targets. The full Makefile (one-command export, android, publish, ...) is owned
# by 05_code_plans/01 (#28); these are the lint/typecheck/test entrypoints plan #5 needs now. All
# targets are thin wrappers over uv/ruff/mypy/pytest — no logic lives here.

.PHONY: lint format typecheck parity test test-train check

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/mobiletransformers

# Cross-language enum/schema parity gate (Python source of truth vs. checked-in Kotlin/schemas).
parity:
	uv run python -m mobiletransformers.codegen.enums --check

test:
	uv run pytest tests/unit tests/fixtures tests/export

# ORT-training smoke — requires the cp312 source-built wheel (ort-training-local, Python 3.12).
test-train:
	uv run --python 3.12 --group ort-training-local pytest tests/integration/test_ort_training_smoke.py

check: lint typecheck parity test
