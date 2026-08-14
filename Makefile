# MobileTransformers developer Makefile (owned by 05_code_plans/01, #28).
#
# Every target is a THIN WRAPPER (<= ~3 lines) over the `mobiletransformers` CLI (#15) or Gradle —
# no build/export logic lives here. `setup*` targets honor the dependency-profile isolation from #2
# (the onnxruntime-import-colliding profiles can never co-install; each `setup` variant syncs its own
# environment). CI (#29) invokes these targets, not raw commands.

.PHONY: help setup setup-export setup-train setup-genai \
        lint format typecheck parity guard test test-smoke test-train test-jvm test-cpp test-integration check consumer-app \
        export-model package-model android-build device-package device-test device-rss device-federated build-aar publish-local docs requirements clean-generated

# Overridable export knobs (used by `export-model`).
MODEL   ?=
OUTPUT  ?= build/package
PEFT    ?= lora
QUANT   ?= int4
CONFIG  ?=
# `package-model` re-emits the manifest of an ALREADY-EXPORTED package; defaults to what export wrote.
PACKAGE ?= $(OUTPUT)
# device-package knobs (the script re-applies its own defaults for empty values).
VARIANT ?= cpu-int4
TRAIN   ?= 0
RAG     ?= 1
# #33: explicit Optimum task. Empty means auto-select, which never picks `text-classification` —
# an encoder fine-tune must name it (e.g. TASK=text-classification).
TASK    ?=
# #36: the host-side package the federated gateway aggregates against. It must be the SAME export the
# device holds — `weight_handoff_map.json` is the authority on tensor names/shapes for both sides.
FED_PKG ?= build/pkg
# Gradle 8.7 / AGP 8.5.1 need JDK 17. The system `java` is often 11, so fall back to Android Studio's
# bundled JBR the same way scripts/{android_build_aar,publish_local_maven}.sh do. An explicit JAVA_HOME
# in the environment always wins.
JAVA_HOME ?= /opt/android-studio/jbr
GRADLE  := cd android/MobileTransformers && JAVA_HOME=$(JAVA_HOME) ./gradlew
CPP_DIR := android/MobileTransformers/MobileTransformers/src/main/cpp

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

guard:  ## CI ratchets: #4 secret reads + #6 registry-dispatch literals (Python, legacy roots, C++).
	uv run pytest tests/unit/test_guards.py -q

test-jvm:  ## Android SDK JVM unit tests (no device, no NDK, no vendored native libs).
	$(GRADLE) :MobileTransformers:testDebugUnitTest

test-cpp:  ## C++ host unit tests (googletest; ORT-free headers only, no NDK/device).
	cmake -S $(CPP_DIR)/tests -B build/cpp-tests -DCMAKE_BUILD_TYPE=Release >/dev/null
	cmake --build build/cpp-tests -j
	ctest --test-dir build/cpp-tests --output-on-failure

test-integration:  ## Integration tests (env-gated; skip when their profile is absent).
	uv run pytest tests/integration

test:  ## Python unit tests (core env, no heavy deps).
	uv run pytest tests/unit tests/fixtures tests/export tests/hub tests/support tests/cli tests/adapter tests/federated

test-smoke:  ## Export/package/manifest wiring smoke (core-runnable subset).
	uv run pytest tests/export tests/hub tests/cli -q

test-train:  ## ORT-training smoke — requires the cp312 source-built wheel (ort-training-local).
	uv run --python 3.12 --group ort-training-local pytest tests/integration/test_ort_training_smoke.py

check: lint typecheck parity guard test  ## lint + typecheck + parity + guards + tests (the standing gate).

# --- one-command export / package (wraps #15; no logic here) ------------------------------------
export-model:  ## MODEL=<hf-id> [OUTPUT= PEFT= QUANT=] -> device-ready package (wraps #15).
	uv run mobiletransformers export --model $(MODEL) --output $(OUTPUT) --peft $(PEFT) --quant $(QUANT)

package-model:  ## PACKAGE=<dir> [CONFIG=] -> re-hash + re-emit that package's manifest and checksums.
	uv run mobiletransformers package-model --package $(PACKAGE) $(if $(CONFIG),--config $(CONFIG),)

# --- android / publish (bodies owned by #30; thin wrappers over Gradle + scripts/) --------------
android-build:  ## gradle assembleDebug (SDK + sample app).
	$(GRADLE) :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug

device-package:  ## MODEL=<hf-id> [VARIANT= TRAIN=1 RAG=1 TASK=] -> export + adb push a real package for device tests (#1-29 W6).
	MODEL=$(MODEL) VARIANT=$(VARIANT) TRAIN=$(TRAIN) RAG=$(RAG) TASK=$(TASK) scripts/device_package.sh

device-test:  ## Run the instrumented device suites over the pushed package (skips w/o a device/package).
	$(GRADLE) :MobileTransformers:connectedDebugAndroidTest

device-rss:  ## Collect the four-point RSS table (copy vs mmap, both engines) and evaluate Gate 0.1 #4 / Gate 0.2.
	scripts/device_rss.sh

device-federated:  ## #36 round-trip: export factors on device -> `federated serve` on host -> import back.
	PKG=$(FED_PKG) scripts/federated_round_device.sh

consumer-app:  ## Build examples/consumer-app against the mavenLocal artifact (#30 proof).
	cd examples/consumer-app && ./gradlew assembleDebug

build-aar:  ## Assemble the release AAR (#30 owns the script body).
	scripts/android_build_aar.sh

publish-local:  ## Publish the library to mavenLocal (#30 owns the script body).
	scripts/publish_local_maven.sh

docs:  ## Regenerate the derived docs (compatibility matrix) and check every page's links/tables.
	uv run mobiletransformers support-matrix --md docs/COMPATIBILITY_MATRIX.md
	uv run pytest tests/unit/test_docs.py -q

requirements:  ## Regenerate requirements/*.lock.txt from uv.lock (they had no producer and rotted).
	uv export --no-emit-project --group dev --format requirements.txt -o requirements/requirements-dev.lock.txt
	uv export --no-emit-project --extra export --format requirements.txt -o requirements/requirements-export.lock.txt
	uv export --no-emit-project --extra rag --format requirements.txt -o requirements/requirements-rag.lock.txt
	uv export --python 3.12 --no-emit-project --group ort-training-local --format requirements.txt -o requirements/requirements-train-local.lock.txt

# --- cleanup (generated artifacts ONLY; never user caches / cache_dir/) --------------------------
clean-generated:  ## Remove generated build/model artifacts ONLY (never user caches).
	rm -rf build/ dist/ onnx_models/ *.egg-info src/*.egg-info
	find . -maxdepth 2 -type d -name '__pycache__' -prune -exec rm -rf {} +
