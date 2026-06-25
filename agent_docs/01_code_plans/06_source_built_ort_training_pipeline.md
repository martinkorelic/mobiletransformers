# Source-Built ONNX Runtime Training Pipeline

**Priority #3 | Prerequisites: #2 (`00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`) | Blocks: every training-artifact path that imports `onnxruntime.training` (notably #9 via the merger, #18 training lifecycle); proves Gate 0.3**

## Purpose

`onnxruntime-training==1.23.0+cpu` (`requirements-ort.txt:9`) is **not a public PyPI package** — public `onnxruntime-training` stalls at ~1.19.2. The repo's wheel is **source-built** and must be treated as a first-class local artifact with reproducible provenance, or onboarding silently fails. This plan makes the build a documented, checksummed, CI-smoked pipeline: a Python wheel build script, an Android AAR build script with matching ORT revision/NDK/ABIs/API level, a provenance manifest, a `BUILD.md`, the uv local-source wiring, and a CI smoke that imports `onnxruntime.training.artifacts` and runs `generate_artifacts` on a tiny fixture.

This is **Priority #3 — the first gate after toolchain lock** — because it proves the train toolchain is alive before any API work builds on it. The repo never used Optimum's `ORTTrainer`; it calls `onnxruntime.training.artifacts.generate_artifacts` directly (`artifact/onnx_builder.py:65`) and the training runtime API `from onnxruntime.training.api import CheckpointState, Module, Optimizer` (`artifact/onnx_builder.py:21`), so Optimum v2's training deprecation does **not** affect this path — only the source build's survival does.

**THE UNKNOWN TO LOCK:** the exact `torch` version the wheel was compiled against. ORT-training links a specific torch ABI; a mismatch crashes at `import`. It is unknown today and must be resolved from a clean resolve, then pinned (see step 4 / `00_code_plans/03`).

## Touched / new files

- NEW `scripts/build_ort_training_wheel.sh` — Python wheel build (`--enable_training_apis --build_wheel`).
- NEW `scripts/build_ort_training_android.sh` — Android AAR build (`--build_java`), pinned to the **same** ORT commit, NDK, ABIs, API level.
- NEW `third_party/onnxruntime/manifest.json` — provenance: ORT git SHA, build flags, tool versions, produced filenames, SHA256, and `torch_version`.
- NEW `third_party/onnxruntime/BUILD.md` — manual rebuild steps + known-good versions.
- EDIT `pyproject.toml` (from `00_code_plans/01`) — `[tool.uv.sources]` local wheel + `[dependency-groups] ort-training-local`.
- NEW `.github/workflows/ort-training-smoke.yml` (or `ci/` equivalent) — the import + `generate_artifacts` smoke.
- NEW `tests/fixtures/tiny_trainable.onnx` (+ `training_config.json`) — minimal fixture for the smoke.
- Built wheels/AARs/headers/libs stay **out of tracked source** (per Tier 0) — referenced by path/checksum, not committed.

## Data contracts / interfaces

- **`scripts/build_ort_training_wheel.sh`**: clones ORT at a pinned SHA, builds with `./build.sh --config Release --enable_training_apis --build_wheel --parallel` (CPU), emits `onnxruntime_training-1.23.0+cpu-*.whl`. Script echoes and the manifest records: ORT SHA, full build flags, compiler/CMake/Python versions, output wheel filename + SHA256.
- **`scripts/build_ort_training_android.sh`**: builds with `--android --build_java --enable_training_apis` and explicit `--android_sdk_path`, `--android_ndk_path`, `--android_abi <list>`, `--android_api <level>`, **matching the wheel's ORT SHA**. Emits the training-enabled AAR consumed by the Android module. The native headers in `session_cache.h:8` (`onnxruntime/onnxruntime_training_cxx_api.h`) and the training C++ API used by `TrainingSessionCache` (`session_cache.h:819-1042`) must come from this AAR — the manifest records its SHA256 and the ABIs/API level so the device build matches the desktop wheel's ORT revision.
- **`third_party/onnxruntime/manifest.json`** fields: `ort_git_sha`, `ort_version` (`"1.23.0+cpu"`), `build_flags` (wheel + android, separately), `compiler`, `cmake_version`, `ndk_version`, `android_abis`, `android_api_level`, `python_version`, `wheel_filename`, `wheel_sha256`, `aar_filename`, `aar_sha256`, **`torch_version`** (the locked unknown).
- **uv wiring** (mirrors `00_code_plans/03`): `[tool.uv.sources] onnxruntime-training = { path = "third_party/onnxruntime/<wheel>", marker = "..." }`; `[dependency-groups] ort-training-local = ["onnxruntime-training", "torch==<locked>", ...]`. `optimum-onnx` here is **bare** (no `[onnxruntime]` extra) so it does not pull public onnxruntime over the training wheel. This group is declared mutually-exclusive with `export` / `genai-smoke` via `[tool.uv] conflicts`.
- **CI smoke contract:** in the `ort-training-local` env, `python -c "import torch, onnxruntime; from onnxruntime.training import artifacts, onnxblock; from onnxruntime.training.api import CheckpointState, Module, Optimizer"` must succeed, then `generate_artifacts` on the fixture must produce `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint`.

## Implementation steps

1. **Write `build_ort_training_wheel.sh`**: parameterize ORT SHA + config; run `--enable_training_apis --build_wheel`; on success compute SHA256 and append/update `manifest.json`. Keep CPU-only for the desktop artifact-generation profile.

2. **Write `build_ort_training_android.sh`**: same ORT SHA; `--build_java --enable_training_apis` with explicit NDK/ABIs/API level; compute the AAR SHA256 into `manifest.json`. Document that NNAPI does not work with ORT training (already noted in `session_cache.h:259,602,981`) — the device training session is CPU.

3. **Author `manifest.json` + `BUILD.md`**: fill every provenance field; `BUILD.md` gives copy-paste rebuild steps and the known-good tool versions. Leave `torch_version` as a `TODO` until step 4.

4. **Lock the torch unknown** (the critical step, procedure from `00_code_plans/03`):
   1. Build a clean venv from current `requirements-ort.txt` pins (`onnxruntime-training==1.23.0+cpu`, `optimum==1.23.3`, `peft==0.13.2`, `onnxscript==0.3.1`).
   2. Record exactly what torch resolves to: `uv pip freeze | grep -i '^torch'`.
   3. Pin `torch==X.Y.Z` into the `ort-training-local` group **and** `manifest.json` `torch_version`.
   4. Until done, leave torch unpinned with `# TODO(plan06): pin to resolved ORT-training torch ABI`.

5. **Wire `pyproject.toml`**: add `[tool.uv.sources]` local wheel + `ort-training-local` group + the `conflicts` entries. Confirm `uv sync --frozen --group ort-training-local` resolves with no public `onnxruntime` shadowing the local wheel.

6. **Build the fixture**: a tiny ONNX graph with at least one trainable initializer + a `training_config.json` shaped like the real one (`requires_grad` / `frozen_params` lists, the exact fields `gen_artifacts` reads at `artifact/onnx_builder.py:52-59`). Keep it sub-MB so CI is fast.

7. **Write the CI smoke** (`ort-training-smoke.yml`): sync the `ort-training-local` group, run the import assertion, then mirror `gen_artifacts` (`artifact/onnx_builder.py:33-90`): split initializers into `requires_grad`/`frozen_params`, call `artifacts.generate_artifacts(..., optimizer=artifacts.OptimType.AdamW, ...)`, assert the four output artifacts exist. Optionally run one `Module`/`Optimizer` step (the `onnx_checktrain` shape at `artifact/onnx_builder.py:111-153`) as an extended check.

## Interactions

- **#2 (dependency profiles)**: this plan implements the build + uv wiring that #2 specified; #2's "one onnxruntime provider per env" and `conflicts` rules are enforced here. The torch unknown is shared between the two docs — resolve once, record in `manifest.json`.
- **#9 (unified merger)** and **#18 (training lifecycle)**: both depend on `generate_artifacts` and the `onnxruntime.training.api` runtime working; this plan is the gate that proves they can.
- **Android module** (`session_cache.h` `TrainingSessionCache`): consumes the training AAR from `build_ort_training_android.sh`; the AAR's ORT SHA must match the wheel so desktop-generated artifacts run on-device.
- **Gate 0.3** (`01_tier0_foundation_decisions.md`): the documented, checksummed build + the passing smoke is the Gate 0.3 evidence (Path A: direct `onnxruntime.training.artifacts` on a source-built wheel).

## Tests & acceptance

Per Gate 0.3's framing: **building the wheel and the Android AAR is Manual/CI** (long, toolchain-heavy), while the **`generate_artifacts` fixture smoke is Integration** — fast and automated once the wheel exists. The conflicts/resolution guards that don't need the built wheel are fast Unit checks.

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Fixture well-formedness** (`pytest tests/fixtures/test_tiny_trainable.py`): `tiny_trainable.onnx` has ≥1 trainable initializer and `training_config.json` carries the exact `requires_grad` / `frozen_params` fields `gen_artifacts` reads (`artifact/onnx_builder.py:52-59`); sub-MB.
- **Conflicts-guard resolution:** `uv sync --frozen --extra export` resolves (public packages, no source wheel needed); co-syncing `--extra export` with `--group ort-training-local` **fails** at resolution time (proves the `[tool.uv] conflicts` guard) — fast, no build.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Import smoke:** in the `ort-training-local` env, `import torch, onnxruntime; from onnxruntime.training import artifacts, onnxblock; from onnxruntime.training.api import CheckpointState, Module, Optimizer` succeeds (catches the torch-ABI mismatch — the locked unknown).
- **`generate_artifacts` smoke:** run on `tests/fixtures/tiny_trainable.onnx`; assert `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint` are produced with the AdamW optimizer. (This is the CI smoke in `ort-training-smoke.yml`.)
- **Extended one-step train smoke (optional):** load the artifacts via `Module`/`Optimizer`, run one `train()` + `optimizer.step()` (per `onnx_checktrain`), assert a finite loss.
- **uv local-group sync smoke:** `uv sync --frozen --group ort-training-local` resolves with the local wheel and **no public `onnxruntime` shadowing** it.
- **Provenance smoke:** re-hashing the local wheel/AAR matches `manifest.json` SHA256 (catches a silently-swapped artifact).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Build the training wheel:** run `scripts/build_ort_training_wheel.sh` at the pinned ORT SHA; emits `onnxruntime_training-1.23.0+cpu-*.whl` and updates `manifest.json` (SHA256 + flags).
- **Build the Android AAR:** run `scripts/build_ort_training_android.sh` at the **same** ORT SHA / NDK / ABIs / API level; emits the training-enabled AAR and records its SHA256.
- **Lock the torch unknown:** clean-resolve from `requirements-ort.txt`, `uv pip freeze | grep -i '^torch'`, pin `torch==X.Y.Z` into the group + `manifest.json`.
- **Android AAR compile/link:** the library module compiles and links against the training AAR; verify the headers/libs ORT version matches `manifest.json` `ort_version` (real NDK build).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- A reproducible, checksummed build exists: `build_ort_training_wheel.sh` + `build_ort_training_android.sh` + `manifest.json` (all provenance fields incl. the resolved `torch_version`) + `BUILD.md`, with the wheel/AAR referenced by path/checksum and **not** committed.
- `pyproject.toml` wires the local wheel via `[tool.uv.sources]` + `ort-training-local` group + `conflicts`; `uv sync --frozen --group ort-training-local` resolves with no public `onnxruntime` shadow.
- The CI smoke (`ort-training-smoke.yml`) passes: imports succeed and `generate_artifacts` produces the four artifacts on the tiny fixture — the Gate 0.3 evidence that the train toolchain is alive.
