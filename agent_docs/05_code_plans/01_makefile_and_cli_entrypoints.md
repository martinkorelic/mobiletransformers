# Makefile & CLI Entrypoints

**Priority #27 | Prerequisites: #1 (`00_code_plans/01_python_package_and_uv_scaffolding.md`), #14 (`02_code_plans/05_one_command_export_cli.md`), #2 (`00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`) | Blocks: #28 (`05_code_plans/02`, CI), #29 (`05_code_plans/03`, AAR)**

> Runs alongside the tiers; gated by the contracts it wraps. Thin wrappers only — no logic in the Makefile.

## Purpose

Lower clone-to-running to minutes with one repeatable command path. The `Makefile` and `scripts/` are **thin wrappers** over the Python CLI (`mobiletransformers ...`, from #14) and Gradle; they must not encode logic that belongs in those tools. The repo currently has **no `Makefile` and no `scripts/` directory** — both are created here.

## Touched / new files

- NEW `Makefile` (repo root).
- NEW `scripts/` (repo root) — `android_build_aar.sh`, `publish_local_maven.sh` (bodies owned by #29); `run_smoke.sh`.
- `pyproject.toml` (from #1) — provides the `mobiletransformers` console entrypoint the Makefile calls.
- `android/ORTransformer/gradlew` — wrapped by `android-build` / `build-aar` targets.

## Data contracts / interfaces

### Makefile targets (canonical names — these become a compatibility surface, see `00_code_plans` reimplementation gates)

```make
setup:            ## install python package with dev+export extras (uv sync / pip install -e .[export,train])
lint:             ## formatter + linter (after toolchain chosen)
test:             ## python unit tests
test-smoke:       ## tiny export/artifact/inference smoke
export-model:     ## MODEL=.. PEFT=.. QUANT=.. → MobileTransformers-ready package (wraps #14)
package-model:    ## validate+package an existing build dir
android-build:    ## gradle assembleDebug (app + library)
build-aar:        ## assemble release AAR (post-rename target; legacy fallback documented in #29)
publish-local:    ## publish library to mavenLocal (#29)
docs:             ## build docs if a docs system is added (#30)
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

1. Create `pyproject.toml` console-script (`mobiletransformers = mobiletransformers.cli:main`) — already in #1/#14; the Makefile depends on it existing.
2. Write `Makefile` with the targets above as thin wrappers; add `.PHONY` and `## help` self-documentation.
3. Create `scripts/` with executable shell stubs; `android_build_aar.sh` / `publish_local_maven.sh` bodies land in #29.
4. `clean-generated` removes only `build/`, generated packages, and `onnx_models/` — never `cache_dir/` or anything outside the repo (Tier-5 risk: destructive cleanup).
5. Document the happy path in `README.md` and `docs/EXPORT.md` (#30).

## Interactions

- **#14 (export CLI)**: `export-model`/`package-model` wrap it; do not duplicate export logic.
- **#28 (CI)**: CI invokes `make test`, `make test-smoke`, `make android-build`.
- **#29 (AAR/Maven)**: `build-aar`/`publish-local` call the `scripts/` bodies.
- **#2 (dep profiles)**: `setup*` targets honor profile isolation.

## Tests & smokes

- `make setup` installs at least core+export profiles cleanly.
- `make test` / `make test-smoke` run from a fresh checkout.
- `make help` lists every target with its `##` description.
- `make clean-generated` removes only generated artifacts (assert `cache_dir/` untouched).
- `make export-model MODEL=<tiny> PEFT=lora QUANT=qint8` produces a validatable package.
