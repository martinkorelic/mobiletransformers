# MobileTransformers developer Makefile.
#
# Every target is a THIN WRAPPER (<= ~3 lines) over the `mobiletransformers` CLI or Gradle —
# no build/export logic lives here. `setup*` targets honor the dependency-profile isolation
# (the onnxruntime-import-colliding profiles can never co-install; each `setup` variant syncs its own
# environment). CI invokes these targets, not raw commands.

.PHONY: help setup setup-export setup-train setup-genai doctor fetch-native-deps \
        lint format typecheck parity guard test test-smoke test-train test-jvm test-cpp test-integration check consumer-app \
        export-model package-model publish-catalog publish-artifacts android-build device-package device-test device-hub-test device-rss device-federated build-aar publish-local docs docs-build docs-serve requirements clean-generated

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
# Explicit Optimum task. Empty means auto-select, which never picks `text-classification` —
# an encoder fine-tune must name it (e.g. TASK=text-classification).
TASK    ?=
# The host-side package the federated gateway aggregates against. It must be the SAME export the
# device holds — `weight_handoff_map.json` is the authority on tensor names/shapes for both sides.
FED_PKG ?= build/pkg
# Gradle 8.7 / AGP 8.5.1 need JDK 17.
#
# Resolution order, and the order matters: an explicit JAVA_HOME in the environment wins; else a JDK
# 17+ already on PATH; else Android Studio's bundled JBR at its Linux default. The PATH probe is
# second rather than absent because the hardcoded path is a *Linux Android Studio* location that
# exists on exactly one kind of machine — on macOS, on a CI runner, or under any other JDK install it
# silently resolves to a directory that is not there, and Gradle then reports a Java version error
# naming neither JAVA_HOME nor this file. `make doctor` reports which branch you are on.
JAVA_HOME ?= $(shell \
  if command -v java >/dev/null 2>&1 && \
     java -version 2>&1 | head -1 | grep -qE '"(1[7-9]|2[0-9])'; then \
    dirname "$$(dirname "$$(readlink -f "$$(command -v java)")")"; \
  else echo /opt/android-studio/jbr; fi)
# `uv run`, but WITHOUT re-resolving the lock.
#
# `[tool.uv.sources]` points `onnxruntime-training` at a local path under `third_party/wheels/`, and
# that wheel is git-ignored (632 MB, source-built). A bare `uv run` validates every source in the lock
# before executing, so on a machine that does not have the wheel it fails with
#
#   Failed to read from the distribution cache ... No such file or directory
#
# for `make lint` — a target with no connection to training whatsoever. That made `make check`
# impossible on a fresh clone and would have turned CI red on the first run. `--frozen` skips the
# re-resolution; the lock is committed and covers every profile, so there is nothing to re-resolve.
#
# Targets that genuinely need the wheel (`test-train`) deliberately do NOT use this.
UVRUN   := uv run --frozen
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

doctor:  ## Report every prerequisite and the command that fixes each. Read-only; always exits 0.
	@scripts/doctor.sh

fetch-native-deps:  ## Download + verify the gitignored Android native libraries (see third_party/android/manifest.json).
	@scripts/fetch_native_deps.sh

# --- standing checks (CI re-invokes these) ------------------------------------------------------
lint:  ## Formatter + linter check (ruff).
	$(UVRUN) ruff check src/ tests/
	$(UVRUN) ruff format --check src/ tests/

format:  ## Auto-format + auto-fix (ruff).
	$(UVRUN) ruff format src/ tests/
	$(UVRUN) ruff check --fix src/ tests/

typecheck:  ## Static type check (mypy).
	$(UVRUN) mypy src/mobiletransformers

parity:  ## Cross-language enum/schema parity gate (Python source of truth vs. Kotlin/schemas).
	$(UVRUN) python -m mobiletransformers.codegen.enums --check

guard:  ## CI ratchets: secret reads, registry-dispatch literals, plan identifiers, machine paths.
	$(UVRUN) pytest tests/unit/test_guards.py -q

test-jvm:  ## Android SDK JVM unit tests (no device, no NDK, no vendored native libs).
	$(GRADLE) :MobileTransformers:testDebugUnitTest

test-cpp:  ## C++ host unit tests (googletest; ORT-free headers only, no NDK/device).
	cmake -S $(CPP_DIR)/tests -B build/cpp-tests -DCMAKE_BUILD_TYPE=Release >/dev/null
	cmake --build build/cpp-tests -j
	ctest --test-dir build/cpp-tests --output-on-failure

test-integration:  ## Integration tests (env-gated; skip when their profile is absent).
	$(UVRUN) pytest tests/integration

test:  ## Python unit tests (core env, no heavy deps).
	$(UVRUN) pytest tests/unit tests/fixtures tests/export tests/hub tests/support tests/cli tests/adapter tests/federated

test-smoke:  ## Export/package/manifest wiring smoke (core-runnable subset).
	$(UVRUN) pytest tests/export tests/hub tests/cli -q

test-train:  ## ORT-training smoke — requires the cp312 source-built wheel (ort-training-local).
	uv run --python 3.12 --group ort-training-local pytest tests/integration/test_ort_training_smoke.py

check: lint typecheck parity guard test  ## lint + typecheck + parity + guards + tests (the standing gate).

# --- one-command export / package (no logic here) -----------------------------------------------
export-model:  ## MODEL=<hf-id> [OUTPUT= PEFT= QUANT=] -> device-ready package.
	uv run mobiletransformers export --model $(MODEL) --output $(OUTPUT) --peft $(PEFT) --quant $(QUANT)

package-model:  ## PACKAGE=<dir> [CONFIG=] -> re-hash + re-emit that package's manifest and checksums.
	uv run mobiletransformers package-model --package $(PACKAGE) $(if $(CONFIG),--config $(CONFIG),)

publish-catalog: ## [ONLY=<key> PUSH=0 KEEP=1] -> export + verify + publish the showcase model catalog.
	ONLY=$(ONLY) PUSH=$(if $(PUSH),$(PUSH),1) KEEP=$(if $(KEEP),$(KEEP),0) scripts/publish_catalog.sh

# --- android / publish (thin wrappers over Gradle + scripts/) -----------------------------------
android-build:  ## gradle assembleDebug (SDK + sample app).
	$(GRADLE) :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug

device-package:  ## MODEL=<hf-id> [VARIANT= TRAIN=1 RAG=1 TASK=] -> export + adb push a real package for device tests.
	MODEL=$(MODEL) VARIANT=$(VARIANT) TRAIN=$(TRAIN) RAG=$(RAG) TASK=$(TASK) scripts/device_package.sh

device-test:  ## Run the instrumented device suites over the pushed package (skips w/o a device/package).
	$(GRADLE) :MobileTransformers:connectedDebugAndroidTest

device-hub-test:  ## REPO=<org>/<name> [HUB_TOKEN=] -> pull that repo from the Hub ONTO THE DEVICE and load it.
	@[ -n "$(REPO)" ] || { echo "set REPO=<org>/<name>, e.g. REPO=mobiletransformers/functiongemma-270m-it" >&2; exit 1; }
	$(GRADLE) :MobileTransformers:connectedDebugAndroidTest \
	  -Pandroid.testInstrumentationRunnerArguments.class=com.martinkorelic.mobiletransformers.HubPullDeviceTest \
	  -Pandroid.testInstrumentationRunnerArguments.mtHubRepoId=$(REPO) \
	  $(if $(HUB_TOKEN),-Pandroid.testInstrumentationRunnerArguments.mtHubToken=$(HUB_TOKEN),)

device-rss:  ## Collect the four-point RSS table (copy vs mmap, both engines) and evaluate the memory gates.
	scripts/device_rss.sh

device-federated:  ## Round-trip: export factors on device -> `federated serve` on host -> import back.
	PKG=$(FED_PKG) scripts/federated_round_device.sh

consumer-app:  ## Build examples/consumer-app against the mavenLocal artifact (publication proof).
	cd examples/consumer-app && ./gradlew assembleDebug

build-aar:  ## Assemble the release AAR (scripts/android_build_aar.sh owns the body).
	scripts/android_build_aar.sh

publish-local:  ## Publish the library to mavenLocal (scripts/publish_local_maven.sh owns the body).
	scripts/publish_local_maven.sh

docs:  ## Regenerate the derived docs (compatibility matrix) and check every page's links/tables.
	uv run --frozen mobiletransformers support-matrix --md docs/COMPATIBILITY_MATRIX.md
	$(UVRUN) pytest tests/unit/test_docs.py -q

docs-build:  ## Build the documentation site. `--strict` fails on any unresolved internal link.
	uv run --frozen --group docs mkdocs build --strict

docs-serve:  ## Serve the documentation site locally with live reload (http://127.0.0.1:8000).
	uv run --frozen --group docs mkdocs serve

publish-artifacts:  ## Upload the gitignored build artifacts to the Hub (needs HF_TOKEN_ORG).
	uv run --frozen python scripts/publish_build_artifacts.py $(if $(DRY_RUN),--dry-run,)

requirements:  ## Regenerate requirements/*.lock.txt from uv.lock (they had no producer and rotted).
	uv export --no-emit-project --group dev --format requirements.txt -o requirements/requirements-dev.lock.txt
	uv export --no-emit-project --extra export --format requirements.txt -o requirements/requirements-export.lock.txt
	uv export --no-emit-project --extra rag --format requirements.txt -o requirements/requirements-rag.lock.txt
	uv export --python 3.12 --no-emit-project --group ort-training-local --format requirements.txt -o requirements/requirements-train-local.lock.txt
	# The SBOM had no producer at all, so it sat at 0.1.0 for a month while the project moved on.
	# A dependency inventory nobody regenerates is a claim that quietly stops being true.
	uv export --no-emit-project --group dev --format cyclonedx1.5 -o requirements/sbom-cyclonedx.json

# --- cleanup (generated artifacts ONLY; never user caches / cache_dir/) --------------------------
clean-generated:  ## Remove generated build/model artifacts ONLY (never user caches).
	rm -rf build/ dist/ onnx_models/ *.egg-info src/*.egg-info
	find . -maxdepth 2 -type d -name '__pycache__' -prune -exec rm -rf {} +
