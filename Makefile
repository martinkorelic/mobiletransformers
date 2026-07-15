# MobileTransformers developer Makefile (owned by 05_code_plans/01, #28).
#
# Every target is a THIN WRAPPER (<= ~3 lines) over the `mobiletransformers` CLI (#15) or Gradle —
# no build/export logic lives here. `setup*` targets honor the dependency-profile isolation from #2
# (the onnxruntime-import-colliding profiles can never co-install; each `setup` variant syncs its own
# environment). CI (#29) invokes these targets, not raw commands.

.PHONY: help setup setup-export setup-train setup-genai \
        lint format typecheck parity test test-smoke test-train check \
        export-model package-model android-build build-aar publish-local docs clean-generated

# Overridable export knobs (used by `export-model`).
MODEL   ?=
OUTPUT  ?= build/package
PEFT    ?= lora
QUANT   ?= int4
CONFIG  ?=
GRADLE  := cd android/MobileTransformersApp && ./gradlew

help:  ## List every target with its description.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- environment setup (profile-isolated; never combine the conflicting pairs) ------------------
setup:  ## Install core + dev tooling (no onnxruntime provider).
	uv sync --frozen --group dev

setup-export:  ## Install the export profile (optimum-onnx[onnxruntime]; Python >= 3.11).
	uv sync --extra export

setup-train:  ## Install the source-built ORT-training profile (cp312 only).
	uv sync --python 3.12 --group ort-training-local

setup-genai:  ## Install the onnxruntime-genai smoke profile (Python >= 3.11).
	uv sync --group genai-smoke

# --- standing checks (the #5 tooling; CI re-invokes these) --------------------------------------
lint:  ## Formatter + linter check (ruff).
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:  ## Auto-format + auto-fix (ruff).
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:  ## Static type check (mypy).
	uv run mypy src/mobiletransformers

parity:  ## Cross-language enum/schema parity gate (Python source of truth vs. Kotlin/schemas).
	uv run python -m mobiletransformers.codegen.enums --check

test:  ## Python unit tests (core env, no heavy deps).
	uv run pytest tests/unit tests/fixtures tests/export tests/hub tests/support tests/cli tests/adapter tests/federated

test-smoke:  ## Export/package/manifest wiring smoke (core-runnable subset).
	uv run pytest tests/export tests/hub tests/cli -q

test-train:  ## ORT-training smoke — requires the cp312 source-built wheel (ort-training-local).
	uv run --python 3.12 --group ort-training-local pytest tests/integration/test_ort_training_smoke.py

check: lint typecheck parity test  ## lint + typecheck + parity + tests (the standing gate).

# --- one-command export / package (wraps #15; no logic here) ------------------------------------
export-model:  ## MODEL=<hf-id> [OUTPUT= PEFT= QUANT=] -> device-ready package (wraps #15).
	mobiletransformers export --model $(MODEL) --output $(OUTPUT) --peft $(PEFT) --quant $(QUANT)

package-model:  ## Validate + assemble an existing build dir into a Hub package (wraps #15).
	mobiletransformers package-model $(if $(CONFIG),--config $(CONFIG),)

# --- android / publish (bodies owned by #30; thin wrappers over Gradle + scripts/) --------------
android-build:  ## gradle assembleDebug (SDK + sample app).
	$(GRADLE) :MobileTransformers:assembleDebug :app:assembleDebug

build-aar:  ## Assemble the release AAR (#30 owns the script body).
	scripts/android_build_aar.sh

publish-local:  ## Publish the library to mavenLocal (#30 owns the script body).
	scripts/publish_local_maven.sh

docs:  ## Build the docs set if a docs system is added (#31).
	@echo "docs: no docs system wired yet (owned by 05_code_plans/04, #31)."

# --- cleanup (generated artifacts ONLY; never user caches / cache_dir/) --------------------------
clean-generated:  ## Remove generated build/model artifacts ONLY (never user caches).
	rm -rf build/ dist/ onnx_models/ *.egg-info src/*.egg-info
	find . -maxdepth 2 -type d -name '__pycache__' -prune -exec rm -rf {} +
