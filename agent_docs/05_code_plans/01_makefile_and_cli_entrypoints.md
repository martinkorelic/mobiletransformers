# Makefile & CLI Entrypoints

**Priority #28 | Prerequisites: #1 (`00_code_plans/01_python_package_and_uv_scaffolding.md`), #15 (`02_code_plans/05_one_command_export_cli.md`), #2 (`00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`) | Blocks: #29 (`05_code_plans/02`, CI), #30 (`05_code_plans/03`, AAR)**

> Runs alongside the tiers; gated by the contracts it wraps. Thin wrappers only — no logic in the Makefile.

## Purpose

Lower clone-to-running to minutes with one repeatable command path. The `Makefile` and `scripts/` are **thin wrappers** over the Python CLI (`mobiletransformers ...`, from #15) and Gradle; they must not encode logic that belongs in those tools. The repo currently has **no `Makefile` and no `scripts/` directory** — both are created here.

## Touched / new files

- NEW `Makefile` (repo root).
- NEW `scripts/` (repo root) — `android_build_aar.sh`, `publish_local_maven.sh` (bodies owned by #30); `run_smoke.sh`.
- `pyproject.toml` (from #1) — provides the `mobiletransformers` console entrypoint the Makefile calls.
- `android/ORTransformer/gradlew` — wrapped by `android-build` / `build-aar` targets.

## Data contracts / interfaces

### Makefile targets (canonical names — these become a compatibility surface, see `00_code_plans` reimplementation gates)

```make
setup:            ## install python package with dev+export extras (uv sync / pip install -e .[export,train])
lint:             ## formatter + linter (after toolchain chosen)
test:             ## python unit tests
test-smoke:       ## tiny export/artifact/inference smoke
export-model:     ## MODEL=.. PEFT=.. QUANT=.. → MobileTransformers-ready package (wraps #15)
package-model:    ## validate+package an existing build dir
android-build:    ## gradle assembleDebug (app + library)
build-aar:        ## assemble release AAR (post-rename target; legacy fallback documented in #30)
publish-local:    ## publish library to mavenLocal (#30)
docs:             ## build docs if a docs system is added (#31)
clean-generated:  ## remove generated build/model artifacts ONLY (never user caches)
```

Each is `≤ ~3 lines` calling the CLI/Gradle. Example:

```make
export-model:
	mobiletransformers export onnx --model $(MODEL) --peft $(PEFT) --quant $(QUANT)
```

### Profile isolation guard

`setup` must respect the dependency-profile isolation from #2 (the `onnxruntime` / `onnxruntime-training` / `onnxruntime-genai` / `optimum-onnx` envs must never collide). Provide `setup-train`, `setup-export`, `setup-genai` if a single env cannot host all profiles (decision inherited from #2).

## Implementation steps

1. Create `pyproject.toml` console-script (`mobiletransformers = mobiletransformers.cli:main`) — already in #1/#15; the Makefile depends on it existing.
2. Write `Makefile` with the targets above as thin wrappers; add `.PHONY` and `## help` self-documentation.
3. Create `scripts/` with executable shell stubs; `android_build_aar.sh` / `publish_local_maven.sh` bodies land in #30.
4. `clean-generated` removes only `build/`, generated packages, and `onnx_models/` — never `cache_dir/` or anything outside the repo (Tier-5 risk: destructive cleanup).
5. Document the happy path in `README.md` and `docs/EXPORT.md` (#31).

## Interactions

- **#15 (export CLI)**: `export-model`/`package-model` wrap it; do not duplicate export logic.
- **#29 (CI)**: CI invokes `make test`, `make test-smoke`, `make android-build`.
- **#30 (AAR/Maven)**: `build-aar`/`publish-local` call the `scripts/` bodies.
- **#2 (dep profiles)**: `setup*` targets honor profile isolation.

## Worked example

A few real target bodies — each a thin wrapper, no logic:

```make
export-model:
	mobiletransformers export --model $(MODEL) --peft $(PEFT) --quant $(QUANT)

package-model:
	mobiletransformers package --build-dir $(BUILD_DIR)

lint:
	ruff check src/ && ruff format --check src/

typecheck:
	mypy src/mobiletransformers

android-build:
	cd android/ORTransformer && ./gradlew assembleDebug
```

`lint` and `typecheck` wire the #5 tooling (ruff + mypy); they are the standing checks CI re-invokes (#29).

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `make help` lists every target with its `##` description (grep-checkable).
- `make lint` / `make typecheck` run clean on the committed tree (ruff + mypy, the #5 tooling).
- `make clean-generated` removes only generated artifacts (assert `cache_dir/` and anything outside the repo are untouched).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `make test` / `make test-smoke` run from a fresh checkout and pass against the tiny fixture.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- `make setup` (and any `setup-train`/`setup-export`/`setup-genai` variants) installs at least core+export profiles cleanly in a fresh env.
- `make export-model MODEL=<tiny> PEFT=lora QUANT=qint8` produces a validatable package (wraps #15; can be a multi-minute real export).

**Definition of done** — the `Makefile` and `scripts/` exist with every canonical target as a `≤ ~3 line` thin wrapper, `make help` self-documents them, `setup` honors the #2 profile isolation, and `clean-generated` is proven non-destructive to user caches.
