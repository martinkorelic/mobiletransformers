# Tier 0 — 00_code_plans audit

Audited 2026-08-07 against branch `restructure` @ `54e0a8e` (clean tree). Host gates re-run this
session (no repo mutation, `--frozen --no-sync`):

- `pytest tests/{unit,fixtures,export,hub,support,cli,adapter,federated}` → **215 passed, 10 skipped**
- `ruff check src/ tests/` → clean · `mypy src/mobiletransformers` → 62 files, no issues
- `python -m mobiletransformers.codegen.enums --check` → `parity OK`
- `mobiletransformers --help` / `import mobiletransformers` → exit 0, `__version__ == 0.1.0`

`uv sync` was **not** run (would mutate the shared `.venv`), so the plan-#2 conflict-pair assertions
(`uv sync --extra export --group ort-training-local` must fail) are **declared but unverified** here.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `01_python_package_and_uv_scaffolding` | `[x]` done | yes | **97%** | Genuinely done |
| 2 | `03_dependency_profiles_and_ort_training_wheel` | `[x]` done | mostly | **85%** | Done w/ documented deferrals (`export-rocm` empty, Android manifest fields null) |
| 4 | `02_config_layering_settings_constants` | `[x]` done | mostly | **85%** | Layers real; step 8 (legacy secret-read migration) + step 9 (CI grep guard) not done |
| 5 | `10_python_code_quality_and_module_health` | `[x]` done | mostly | **80%** | Tooling green; exception-discipline test is weak and violated; decomposition round-trip test missing |
| 6 | `09_typed_models_enums_and_registries` | `[x]` done | **partly** | **55%** | Owner contract layer done; **grep-guard DoD fails**, `build_adapter_mapping` missing, Kotlin step 9 not done |
| 8 | `07_weight_handoff_map_and_tensor_codec` | `[x]` done | yes+ | **85%** | Better than claimed (C++ both sides landed); integration tests missing |
| 13 | `06_manifest_first_package_and_cache_bridge` | `[x]` done | mostly | **78%** | Installer skips verify + post-install probe/rollback; no Robolectric integration test |
| 16 | `04_android_gradle_rename_migration` | `[x]` done | yes | **90%** | Done via option B; naming contract drifted (`:app`, not `:MobileTransformersApp`); no aliases, no maven coord |
| 17 | `05_android_facade_foundation` | `[ ]` code-complete | mostly | **85%** | Surface real; `ORT*` leak in public `TrainingResult`; public-surface lint test missing; device leg open |
| 18 | `08_training_lifecycle_and_checkpoint_contracts` | `[ ]` code-complete | mostly | **72%** | Classes real but **unwired dead code**; cancel test missing; `canResume` drift |

**Tier 0 overall: ~81%.** The Python foundation (#1/#2/#4/#5/#8) is real and CI-gated. The two soft
spots are **#6's consumption half** (legacy Python/C++/Kotlin dispatch was never migrated, so the
plan's own "no `if/elif`" grep guard fails) and **#18's wiring** (`TrainingJob` has no consumer).

---

## Per-plan findings

### #1 — Python package & uv scaffolding (`00_code_plans/01`)

**Required:** `src/mobiletransformers/` tree per doc-00 hierarchy + `__init__` everywhere;
`__version__ = "0.1.0"`; hatchling `pyproject.toml` w/ extras, groups, `[tool.uv.sources]`,
console script; `cli/main.py` argparse dispatcher + `export`/`validate`/`package-model` stubs with
`add_parser`/`run`; `tests/{unit,integration,smoke}`; `requirements/`, `scripts/`,
`third_party/onnxruntime/`; `tests/unit/test_import_compat.py`; wheel contains only `src/`.

**Verified present:**
- Full subpackage tree: `config/`, `config/registry/`, `codegen/`, `export/`, `artifacts/`,
  `peft/{mars,lora_xs,ablation}/`, `training/`, `inference/`, `rag/`, `hub/`, `federated/`,
  `evaluation/{benchmarks,mobile,smoke}/`, `utils/`, `cli/`, `support/`, `adapter/` — all with
  `__init__.py`.
- `src/mobiletransformers/__init__.py:26` `__version__ = "0.1.0"`; `__all__` declared `:29-46`.
- `pyproject.toml:1-24` hatchling + `[project.scripts] mobiletransformers = "mobiletransformers.cli.main:main"`;
  `:76-79` `[tool.uv.sources]`; `:93-97` hatch wheel target = `["src/mobiletransformers"]` + `allow-direct-references`.
- `src/mobiletransformers/cli/main.py:1-40` — canonical argparse dispatcher, 9 subcommands, all via
  `add_parser`/`run`. No `cli/__main__.py`, no typer/click (contract honored by #15/#21/#22/#35).
- `tests/unit/test_import_compat.py` green (asserts `tools.parser_config` legacy import + `import mobiletransformers`).
- `third_party/wheels/README.md`, `requirements/` (4 locks + SBOM), `scripts/` (6 scripts) all populated.
- Re-verified live: `mobiletransformers --help` exits 0, module import clean.

**Gaps:** (d) `uv build` wheel-content assertion not re-run this session (would write `dist/`);
hatch config makes it structurally certain. Nothing else.

**Drift / doubtful claims:** none. Self-check #1 is justified.

---

### #2 — Dependency profiles & ORT-training wheel (`00_code_plans/03`)

**Required:** exact pins in every extra/group; `genai-smoke` + `export-rocm` groups;
`[tool.uv] conflicts` for all mutually-exclusive onnxruntime providers (incl. `export ✕ export-rocm`);
source-built wheel at the `[tool.uv.sources]` path with matching `sha256`;
`third_party/onnxruntime/{manifest.json,BUILD.md}` with **non-empty** `torch_version`;
`scripts/build_ort_training_{wheel,android}.sh`; generated `requirements/*.lock.txt` + SBOM.

**Verified present:**
- `pyproject.toml:39-74` — `dev`/`docs`/`smoke`/`android-build`/`genai-smoke`/`ort-training-local`/`export-rocm`.
- `pyproject.toml:81-87` `[tool.uv] conflicts` — the three declared pairs.
- `pyproject.toml:52-69` `ort-training-local` carries the resolved ABI pins: `torch==2.7.1`,
  `transformers==4.46.2`, `numpy<2; python_version=='3.12'`, `onnx<1.19; python_version=='3.12'`,
  `onnxruntime-training; python_version=='3.12'` — all justified inline.
- `pyproject.toml:16` `pydantic>=2` **is** a core dependency (the #6 prerequisite).
- `third_party/onnxruntime/manifest.json` — `ort_git_sha 9b25b6a838…`, `torch_version "2.7.1"` (non-empty,
  the "one unknown" resolved), wheel `sha256 87e6f3c6…`, `paired_stack` block.
- `third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` present (662 MB),
  git-ignored via `.gitignore` `third_party/wheels/*.whl`.
- `scripts/build_ort_training_wheel.sh`, `scripts/build_ort_training_android.sh`, `third_party/onnxruntime/BUILD.md`.
- `requirements/{requirements-dev,requirements-export,requirements-rag,requirements-train-local}.lock.txt`
  + `sbom-cyclonedx.json`.

**Gaps:**
- (d) `export-rocm = []` (`pyproject.toml:74`) — declared-but-empty. The plan's **fourth** onnxruntime
  provider (`onnxruntime-rocm==1.18.0` + `torch*+rocm`) and the `export ✕ export-rocm` conflict pair are
  therefore **not wired**. Documented deferral (needs an AMD index).
- (d) `manifest.json` `ndk_version: null`, `android_api_level: null`, `abis: []`, `android.aar_sha256: null` —
  `build_ort_training_android.sh` has never been run. Documented.
- (b) Isolation assertions (`uv sync --frozen --extra export` alive; `--extra export --group ort-training-local`
  must fail; `uv export` re-runs with no diff) not re-verified here — env-mutating.

**Drift / doubtful claims:** the DoD says "**every** declared conflict pair errors on co-sync"; only 3 of the
plan's 4 pairs are declared. Otherwise the self-check's deviation list (a)–(e) is accurate and matches code.

---

### #4 — Config layering, Settings & constants (`00_code_plans/02`)

**Required:** three layers (`config/config.yml` YAML-only; `config/settings.py` `Settings`+`get_settings()`;
`config/constants.py`); `utils/yaml.load_config_from_file` **replacing the six duplicate loaders**;
root `config.py` + `tools/parser_config.py` deprecation shims; CLI>env>YAML>default `resolve()`;
step 8 — migrate business-module `os.environ['HF_TOKEN'|'HF_CACHE'|…]` to `get_settings()`;
step 9 — a **CI** grep guard banning direct secret reads in `src/`.

**Verified present:**
- `config/config.yml` exists; `config/` holds YAML only (no `__init__.py`, no `.py`) — the import-collision
  constraint is honored.
- `src/mobiletransformers/config/settings.py:24-63` — frozen `Settings` dataclass (8 fields incl. `gemini_api_key`),
  `require_hf_token()`, `@lru_cache get_settings()` with `load_dotenv()`. Exactly the plan's shape; stdlib only.
- `src/mobiletransformers/config/constants.py` — section names, `TASK_NAME_TO_DATASET`, experiment constants,
  `:158 SUPPORTED_PEFT_METHODS = tuple(m.value for m in PEFTMethod)` (correctly derived, not a stale tuple).
- `src/mobiletransformers/config/__init__.py:18-27` `resolve(cli, env, yaml, default)`.
- `src/mobiletransformers/utils/yaml.py:19` `load_config_from_file`.
- `config.py` (root) and `tools/parser_config.py` are re-export shims emitting `DeprecationWarning`;
  both proven by `tests/unit/test_settings_precedence.py` + `test_import_compat.py` (green, warnings observed).
- Grep over `src/` for `os.environ[HF_TOKEN|HF_CACHE|GEMINI_API_KEY|AZURE_]` → **empty**.

**Gaps:**
- (a) **Step 8 not done.** 13 direct secret reads remain in the legacy roots:
  `trainer/builder.py:262,264,265`; `trainer/validator.py:154`; `inference/validator.py:45,113`;
  `inference/builder.py:3415,3416`; `artifact/onnx_builder.py:128,334,498,620,650`;
  `evaluation/openehr/openehr_eval.py:28`. The plan explicitly lists these as "the migration checklist".
- (a) **Step 9 not done as specified.** There is **no** grep guard in `Makefile` or
  `.github/workflows/ci.yml` — the guard exists only as a one-off command someone ran. The `fast` job
  runs `make lint typecheck parity test` only.
- (d) The six duplicate `load_config_from_file` copies all survive
  (`trainer/builder.py:503`, `trainer/validator.py:1140`, `trainer/merge_validator.py:435`,
  `artifact/onnx_builder.py:685`, `inference/validator.py:396`, `inference/builder.py:3400`) —
  explicitly deferred in the self-check and documented in `utils/yaml.py`'s docstring.

**Drift / doubtful claims:** self-check line "the CI grep guard (`os.environ[...]` over `src/`) is empty"
implies an enforced gate. **It is not wired into CI or the Makefile** — a future `src/` regression would
not fail the build. The DoD bullet "CI grep guard passes" is therefore unsupported.

---

### #5 — Python code quality & module health (`00_code_plans/10`)

**Required:** `[tool.ruff]`+`[tool.mypy]` w/ ratchet; `exceptions.py` hierarchy mirroring Kotlin names;
`utils/logging.py` `get_logger` (no `print` in library code, no bare `raise Exception`);
`_typing.py` + `py.typed`; `__all__` + `public_api.txt` golden; `# DECOMPOSE:` note on every monolith;
`research/` quarantined; lint/typecheck/parity + public-API smoke green in CI; a test asserting
"every public module raises only `MobileTransformersError` subclasses"; a **decomposition round-trip**
integration test (one per-arch inference builder through the registry ⇒ same ONNX as the monolith).

**Verified present:**
- `pyproject.toml:99-135` — ruff (`E,F,I,W,UP,B`, line 110, legacy roots excluded) + mypy
  (lenient global, `disallow_untyped_defs` for `mobiletransformers.*`, `follow_imports=skip` for legacy roots).
- `src/mobiletransformers/exceptions.py` — `MobileTransformersError` → the 7 named subclasses (verified
  by `tests/unit/test_exceptions.py`).
- `src/mobiletransformers/utils/logging.py`, `_typing.py`, `py.typed`, `public_api.txt` all present;
  `tests/unit/test_public_api.py` guards the golden.
- All 7 monoliths carry the note: `trainer/{utils,embedding_builder,validator}.py:1`,
  `database/builder.py:2`, `inference/builder.py:1`, `artifact/{onnx_builder,merger}.py:1`
  (+ an 8th on `inference/export_inference_package.py:1`).
- `research/` + legacy roots excluded from wheel, ruff and mypy.
- `Makefile` `lint`/`format`/`typecheck`/`parity`/`test`/`check`; `ci.yml` `fast` job invokes them.
  All three gates re-run green this session.

**Gaps:**
- (b) `tests/unit/test_exceptions.py` only has 3 tests, all about **hierarchy shape**
  (`test_every_error_derives_from_root`, `test_catchable_as_root`, `test_root_is_an_exception`). The plan's
  actual requirement — "every public module raises only `MobileTransformersError` subclasses on its
  documented failure paths" — is **not tested**, and it is **violated**:
  `src/mobiletransformers/hub/package_format.py:128,132` raise bare `ValueError`;
  `src/mobiletransformers/config/settings.py:38` and `export/onnx_config_with_loss.py:132,154` raise
  `RuntimeError`/`ValueError`.
- (a) The **decomposition round-trip** integration test does not exist, because
  `inference/builder.py` was never split into `inference/graph/<arch>.py`. The only golden-equivalence
  test that landed is the *merger* one (`tests/unit/test_merger_builder.py`, delivered by #9). The plan's
  DoD bullet "the #6/#9-adjacent splits have landed" is half-true: #9's merger split landed, #6's per-arch
  inference-builder split did not.
- (d) `codegen/enums.py:130-138` uses `print()` — it is a CLI-ish codegen entry point, defensible, but it
  is under `src/` library code and the ban is stated unconditionally.
- (d) Naming drift only: the plan names `tests/utils/test_exceptions.py` and `tests/parity/`;
  actual are `tests/unit/test_exceptions.py` and `tests/unit/test_enum_parity.py`. Harmless.

**Drift / doubtful claims:** self-check bullet "Does new code use `get_logger()` + the `exceptions.py`
hierarchy (**no bare `raise Exception`**)" is checked `[x]` but three `src/` modules raise stdlib
exceptions on documented failure paths, and nothing tests it.

---

### #6 — Typed models, enums & registries (`00_code_plans/09`) ← **weakest plan in the tier**

**Required (A1–A5 + steps 4–10):** Pydantic v2 config layer; 11 mirrored enums; PEFT registry **incl.
`build_adapter_mapping`**; architecture registry replacing `trainer/builder.py:260-272` **and**
`inference/builder.py:3234-3271`; merger registry + `build_merger_model` replacing the four factories;
C++ `get_merger_type`/`run_merger_model`/paths resolved from the handoff map; code-only config lifted
(`QuantizationOptions`, lora-xs reconstruction, `SessionOptions`, `large_model`); Kotlin typed enum fields
in the three config data classes + fail-closed `FileUtil` parsing + `ConfigurationScreen` from `entries`;
**grep guard**: no `architectures[0] ==` / `train_method ==` / `peft_method ==` / `merger_type ==` literal
branch survives in `trainer/`, `inference/`, `artifact/`, `weight_merger.cpp`.

**Verified present (the owned contract layer — genuinely good):**
- `config/constants.py` — 13 `str,Enum` classes (the 11 required + `IndexingMode` (#27) + Python-only
  `ExportFrontend` (#7)) + `ENUM_REGISTRY:107`.
- `config/models.py:37-113` — `_Base` (`populate_by_name`, `extra="ignore"`, `use_enum_values`),
  `_CrossBoundary` (schemaVersion block), `SamplingConfig`, `DeviceOptions`, `Linear/CosineScheduler`,
  discriminated `SchedulerConfig:79`, `QuantizationOptions`, `GenerationConfig`, `TrainingConfig`, `RagConfig`.
- `config/registry/peft.py:21-73` (`AdapterComponent`, `PEFTMethodSpec`, `PEFT_REGISTRY`, `get_peft_spec`),
  `architecture.py:31-100` (`ArchitectureSpec`, `ARCHITECTURE_REGISTRY`, `resolve_architecture`,
  lazy `import_from_path`), `merger.py:33-222` (`MergerSpec`, `resolve_merger`, `build_merger_model`,
  `_build_lora_merger`, `_build_mars_merger`).
- `codegen/enums.py` + checked-in `schemas/{enums.json,GenerationConfig,RagConfig,TrainingConfig}.schema.json`;
  `--check` re-run green.
- 12 Kotlin mirrors under `…/mobiletransformers/constants/*.kt`, parity-gated.
- `trainer/builder.py`'s **architecture** ladder is gone (grep for `architectures[0] ==` in `trainer/`: no hits) —
  replaced by `resolve_architecture` (delivered by #7).
- The four `create_*_merger_model{,_2}` factories are gone; `tests/unit/test_merger_builder.py` pins the
  unified builder byte-for-byte to committed goldens (delivered by #9).

**Gaps — (a) genuinely missing:**
1. **`build_adapter_mapping` does not exist.** Grep across `src/`, `trainer/`, `artifact/` → zero hits.
   A3 declares it as the single mapping builder; `trainer/utils.py:535 create_mars_adapter_mapping` and
   `:672 create_lora_mapping` are both untouched.
2. **PEFT dispatch chain intact.** `trainer/builder.py:286,295,320,332,342` (`train_method == "lora"/"lora-xs"/
   "mars"/"all"/"nolora"`) and `:347,349` (mapping dispatch) — exactly the branches A3 was written to delete.
3. **Inference architecture ladder intact.** `inference/builder.py:3237-3269` — 13 `config.architectures[0] ==`
   branches (Gemma/Gemma2/Llama/Mistral/Phi/Phi3×2/PhiMoE/Phi3Small×2/Phi3V/Qwen2/Nemotron/ChatGLM).
4. **C++ merger dispatch intact.** `weight_merger.cpp:672,708,759,833,853` still branch on
   `merger_type == "lora"/"lora_q"/"mars_q"`. (The *paths* and *variant tag* now come from the handoff
   map via `load_handoff_map:585` — so half of step 7 landed; the `if/elif` selection did not.)
5. **Duplicate target-module tables intact.** `peft_models/mars/utils.py:1 TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING`
   and `peft_models/ablation/utils.py:1 TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING` — still two
   18-line copies, not collapsed into `ArchitectureSpec.target_modules`.
6. **Code-only config not lifted (step 8, 3 of 4 items):** `peft_models/lora_xs/merger.py:34,42` still carry
   `# TODO: Hardcoded` / `# TODO: hardcoded` reconstruction config; `trainer/validator.py:405` still builds a
   raw `SessionOptions()` (no typed model — grep `SessionOptions` in `config/models.py` → none);
   `artifact/onnx_builder.py:555 large_model = True` still hardcoded. Only `QuantizationOptions` landed.
7. **Kotlin step 9 essentially not done.** `ORTGenerationConfig.kt:4 SamplingOptions.method: String`,
   `ORTTrainingConfig.kt:31-33 coreConfigId/memoryConfigId/executionProvider: String`,
   `:54 schedulerType: String`, `ORTRagConfig.kt:8,10 searchType/indexingMode: String` — every closed-set
   field is still `String`. `FileUtil.kt` still parses via `.optString(...)` (`:29,59-61,123-125,133-135`).
8. **`FileUtil.kt:139-158` actively contradicts the fail-closed canonical decision**: an unknown
   `schedulerType` prints `"Warning: Unknown scheduler type … using linear scheduler"` and **silently
   defaults to Linear** (same in `parseSchedulerConfigFromRoot:160+`). The canonical decision requires
   enum `fromWire()` that throws on unknown value. This is a live correctness hazard, not cosmetic.
9. **`ConfigurationScreen.kt:639,769,969`** still hardcode `listOf("greedy","top_p","top_k")`,
   `listOf("opt1","opt2","opt3")`, `listOf("linear","cosine")` instead of `SamplingMethod.entries` etc.

**Landed of step 9:** only `ORTGeneratorNative.kt:302-305` — `methodMap` retired in favour of
`SamplingMethod.fromWire(args.method).nativeOrdinal` (delivered by #24).

**Drift / doubtful claims:** the plan's **Definition of done** says "adding a method/architecture/merger is a
registry row + enum member with **no** new `if/elif` branch (**grep guard passes**)". The grep guard
demonstrably **fails** in all four named locations. The `Done` box is `[x]` and IMPLEMENTATION_ORDER's
first self-check bullet is `[x]` with the caveat "*true for `src/`; legacy `trainer/`,`inference/`,`artifact/`
still branch — their rewrites ride with #7/#9*". #7 and #9 are both marked done/code-complete and **did not**
do those rewrites (they fixed the arch ladder in `trainer/` and the merger factories only). So the deferral
target no longer exists — this work is currently **orphaned with no owner**.

---

### #8 — Weight handoff map & tensor codec (`00_code_plans/07`)

**Required:** `artifacts/handoff_map.py` (`TensorSpec`/`HandoffEntry`/`HandoffMap`/`TrainableTensorCodec`),
byte-deterministic `to_json`, `validate()` fail-closed invariants incl. the quantized scale-naming bug,
`check_compat()` per the pinned algorithm + shared `check_compat_cases.json`; build-side emit
(`observed_inference_inits` + `peft_mapping`); C++ `load_handoff_map` + map-driven
`save_merged_parameters`; C++ map-driven `WeightSessionCache::init`; Kotlin JNI thread-through;
integration tests (C++ smoke; cross-language golden parsed by Python **and** C++).

**Verified present:**
- `src/mobiletransformers/artifacts/handoff_map.py:33 HANDOFF_MAP_READER_VERSION`, `:41 INFERENCE_SUFFIX_TO_ROLE`,
  `:52 TensorSpec`, `:64 ObservedInit`, `:79 HandoffEntry`, `:159 HandoffMap`, `:285 TrainableTensorCodec`
  (`canonical_inference_name`, `from_peft_mapping:307`).
- `src/mobiletransformers/artifacts/versioning.py:19 SchemaVersionError`, `:23 parse_version`, `:37 check_compat`.
- `tests/unit/test_handoff_map.py` + `test_tensor_codec.py` (26 tests, incl. naming-drift raise and
  observed-name quantized roles) — green.
- Shared fixture `tests/fixtures/check_compat_cases.json` consumed by **both**
  `tests/unit/test_handoff_map.py` and Kotlin `PackagesTest.kt:44-53` (via `PackageFormat.checkCompat:49`) —
  the cross-language oracle really is shared.
- **Build-side emit landed** (contrary to the plan's own trailing "deferred" note):
  `inference/export_inference_package.py:255,268,291` builds `ObservedInit`s → `from_peft_mapping` →
  `HandoffMap(...)` with `merger_models`.
- **C++ both sides landed:** `cpp/handoff_io.h:26 struct HandoffEntry`, `:47 check_compat` (C++ mirror),
  `:62 load_handoff_entries`; `weight_merger.cpp:585 load_handoff_map`, `:984 externalDataLocation[role]`
  atomic write; `session_cache.h:72-82` loads flat `<name>.bin` keyed by `inferenceInitializerNames[role]`
  with `:143-153` dtype/shape fail-closed validation before `:778 AddExternalInitializers`.

**Gaps:**
- (a) **Integration tests missing entirely.** No C++ test harness exists in the repo (no `cpp/test`,
  no CMake test target). The plan's two integration tests — the tiny-fixture C++ save→load smoke and
  the cross-language golden parsed by the **C++** loader — have no implementation. The Kotlin side
  mirrors `check_compat_cases.json`; the C++ side does not.
- (d) Kotlin JNI: the plan's step 10 says pass the handoff-map path alongside `peftMappingPath`.
  Implementation instead passes the `inference/` dir and derives the path in C++
  (`weight_merger.cpp:1051 merger_models_directory + "/weight_handoff_map.json"`;
  `ORTTrainerNative.kt:602-632`). Functionally equivalent; contract drift only.
- (c) Device merge→load leg outstanding (owned by #9/#23 manual legs).

**Drift / doubtful claims:** plan step 8 said keep `inference_name` compiled as a loud-deprecation
fallback. The implementation **deleted** it (`weight_merger.cpp:968` "inference_name path is gone. Fail
closed if a merged layer has no map entry"). This is a deliberate improvement consistent with the
"fail closed is the default" canonical rule, but it contradicts the plan text and the DoD bullet
"`inference_name` remains only as a loud-deprecation fallback" — that bullet should be rewritten, not
checked. Also: the plan's trailing "Deferred to owning plans → Build-side emit wiring" note is now
**stale** (the wiring landed in #9).

---

### #13 — Manifest-first package & cache bridge (`00_code_plans/06`)

**Required:** Kotlin `packages/{MobileTransformersManifest,ManifestValidator,VariantSelector,
ChecksumVerifier,ModelPackageInstaller,CacheIndex}.kt`; Python `artifacts/manifest.py` + `test_manifest.py`;
install flow = plan file set → staging → **verify** → materialize → **assert `updatePaths()` probes for
claimed features, roll back on miss** → atomic rename; `CacheIndex.list()` tolerating legacy dirs;
tests incl. an atomicity test and a Robolectric/JVM integration test asserting
`LLMRepository.availableModels` + the three availability flags.

**Verified present:**
- All six Kotlin classes exist with the specified surface: `MobileTransformersManifest.kt:14/39/44/48`,
  `ManifestValidator.kt:10 validate` (check_compat → variants non-empty → defaultVariant ∈ variants →
  per-feature `paths` coverage → weightHandoff resolvable → `requiredFiles` on disk),
  `VariantSelector.kt:9 select`, `ChecksumVerifier.kt:8/22/31`, `ModelPackageInstaller.kt:19 install`,
  `CacheIndex.kt:16 list`, plus `PackageFormat.kt:19 sanitizeRepoId` / `:49 checkCompat`.
- Python `src/mobiletransformers/artifacts/manifest.py:24 MANIFEST_READER_VERSION`, `:28 SelectedVariant`,
  `:42 MobileTransformersManifest`; `tests/unit/test_manifest.py` (10 tests, green).
- `PackagesTest.kt` — 10 JVM tests incl. both shared-oracle parity tests, manifest validate/reject,
  three variant-selection cases, checksum-corruption, install-materializes-cache-shape, legacy-dir tolerance.
- Atomic publish path present: `ModelPackageInstaller.kt:26 .staging/<id>` → `:43 renameTo` with a
  cross-mount copy fallback; test `:124` asserts no leftover staging dir.

**Gaps:**
- (a) **Install step 5 (verify) is not in the installer.** `ModelPackageInstaller.install` never calls
  `ChecksumVerifier` or `ManifestValidator`. Verification exists only in the #21 download path
  (`HubDownloader.kt:53-60` passes `expectedSha = manifest.sha256` to `PackageDownloader`). So an
  install from an **already-staged/local** dir — the exact flow this plan defines and tests — is unverified,
  and `ManifestValidator.validate` is called by **no production code path** (only by tests).
- (a) **Install step 7 (post-install feature-probe assertion + rollback) missing.** `install()` copies
  whatever exists and returns; it never asserts `train/training_config.json` / `inference/generation_config.json` /
  `embedding/rag_config.json` / `tokenizer/` for the features the variant claims, and has no rollback.
  It also `deleteRecursively()`s the previous `target` (`:42`) **before** the rename — a failed rename
  after that point destroys the prior install.
- (b) **Atomicity test missing** ("kill install mid-copy → no partial dir under `cacheDir`").
- (b) **Integration test missing.** The plan's `ModelPackageInstaller` (Robolectric) test asserting
  `LLMRepository(cacheDir).availableModels` contains the sanitized id and the three availability flags
  flip is absent — and cannot be written as-is: `MobileTransformers/build.gradle.kts:96-98` declares only
  `junit`, `kotlinx-coroutines-test`, `okhttp-mockwebserver`; **no Robolectric dependency exists**.
  `PackagesTest.kt:118-121` asserts raw file existence as a proxy.
- (c) On-device generate-after-install smoke deferred.

**Drift / doubtful claims:** self-check "*Does the installer materialize the Hub layout into the
`LLMRepository` cache shape atomically? (stage → renameTo; JVM-tested)*" is true for the rename; the
`LLMRepository`-shape half is asserted by file paths, not by `LLMRepository`. The plan's own integration
requirement is unmet.

---

### #16 — Android Gradle rename migration (`00_code_plans/04`)

**Required (as written, option A):** root → `android/MobileTransformers/`;
`rootProject.name = "MobileTransformers"`, `include(":MobileTransformersApp")` + `include(":MobileTransformers")`;
SDK namespace `com.martinkorelic.mobiletransformers`; app namespace/applicationId
`com.martinkorelic.mobiletransformers.app`; inter-module dep `:MobileTransformers`;
**keep** `libortmobile.so` / `loadLibrary("ortmobile")` / `Java_com_martinkorelic_ortmobile_*`;
`Aliases.kt` deprecated typealiases; optional maven-publish coordinate `mobiletransformers-android`;
build gate + native packaging check for arm64-v8a **and** x86_64.

**Verified present:**
- SDK `namespace = "com.martinkorelic.mobiletransformers"` (`MobileTransformers/build.gradle.kts:8`).
- App `namespace`/`applicationId` = `com.martinkorelic.mobiletransformers.app`
  (`app/build.gradle.kts:9,14`); `implementation(project(":MobileTransformers"))` (`:66`).
- **Zero** residual `ortmobile` / `orttransformer` / `ORTransformer` anywhere in the Android tree
  (grep over `*.kt,*.kts,*.cpp,*.h,*.xml,*.txt`, excluding `build/`) → empty.
- Native fully renamed (option **B**, superseding the doc): `CMakeLists.txt:12 project("mobiletransformers")`,
  `loadLibrary("mobiletransformers")` in `GenAISpike.kt:13`, `ORTGeneratorGenAI.kt:135`, `MainActivity.kt:99`;
  30 `Java_com_martinkorelic_mobiletransformers_*` symbols (22 in `native-lib.cpp`, 8 in `genai_runtime.cpp`);
  no `Java_com_martinkorelic_ortmobile_*` remains.
- `libmobiletransformers.so` built for **arm64-v8a** (`build/intermediates/.../arm64-v8a/`) and a
  `MobileTransformers-debug.aar` exists in `build/outputs/aar/`.
- `abiFilters += listOf("arm64-v8a", "x86_64")` (`MobileTransformers/build.gradle.kts:36`).
- Parity generator's Kotlin path (`codegen/enums.py::KOTLIN_CONSTANTS_RELPATH`) follows the new tree — `make parity` green.

**Gaps:**
- (a) **`Aliases.kt` never created** (neither `#16`'s nor `#17`'s `internal/legacy/Aliases.kt`).
  Deliberate + documented in #19's notes ("#16 fully retired the `ortmobile` brand, so there are no
  consumers to alias") — but the plan's DoD bullet "`Aliases.kt` deprecated typealiases compile" and
  #17's "compat typealiases … compile" are checked without an artifact. Classify (d), with a stale DoD.
- (a) **maven-publish coordinate not added.** No `publishing`/`artifactId` block in
  `MobileTransformers/build.gradle.kts`. Plan called it optional-but-cheap; owned by #30.
- (b) x86_64 native link never verified (blocked on the incomplete vendored `jniLibs/x86_64` — documented
  in the #9 self-check). So the "native packaging check for arm64-v8a **and** x86_64" is half-done.
- (c) Instrumented stub + launch smoke are device legs.

**Drift / doubtful claims:** the actual layout is `android/MobileTransformersApp/` with
`rootProject.name = "MobileTransformersApp"` and `include(":app")` — **not** the plan's
`android/MobileTransformers/` + `include(":MobileTransformersApp")`. Every downstream reference had to
follow (`Makefile` uses `:MobileTransformers:assembleDebug :app:assembleDebug`; `ci.yml` likewise). The
plan's "Data contracts / interfaces" `settings.gradle.kts` end state is therefore **stale and wrong**;
no plan text records the deviation. Also the plan's option-A JNI contract (keep `ortmobile`) is inverted
by the implementation — recorded in IMPLEMENTATION_ORDER but **not** in the plan file itself, so a cold
agent reading `00_code_plans/04` alone would be misled.

---

### #17 — Android facade foundation (`00_code_plans/05`)

**Required:** `MobileTransformers.fromPretrained` + `MobileTransformerModel`; `config/*.kt` public configs;
`runtime/{ModelSession(internal),InferenceEngine(reused),RuntimeCapabilities}.kt`; `packages/ModelFeature.kt`;
`hub/{HuggingFaceHubClient,HubDownloadRequest,HubAuthProvider}.kt`;
`internal/runtime/RepositoryBackedModelSession.kt`; `internal/config/ConfigMappers.kt`;
compat typealiases; tests incl. a **public-surface lint** ("no `ORT*` symbol in the public API") and a
JVM/Robolectric `ModelPackageValidator` layout integration test.

**Verified present:**
- `MobileTransformers.kt:20-31` `object MobileTransformers { suspend fun fromPretrained(...) }`, wired to
  `HubResolver`/`HubDownloader`/`RepositoryBackedModelSession`/`LLMRepository`/`PackageFormat`.
- `MobileTransformerModel.kt:29-82` — `applyPeft/train/merge/generate/retrieve/ingest/generateWithRag/
  pushAdapter/close` + `capabilities`/`engine`/`installedFeatures`. No `ORT*`/`Job`/repository type in
  this class's signatures.
- `config/PublicConfigs.kt:20-99` — `DeviceConfig`, `SamplingConfig`, `TrainConfig`, `GenerationConfig`,
  `RagConfig`, `DatasetConfig`, `HubConfig`; `config/PeftConfig.kt` (sealed, from #19).
- `runtime/ModelSession.kt` (full contract), `runtime/RuntimeCapabilities.kt` (exact plan shape),
  `runtime/Results.kt` (8 result types), `packages/ModelFeature.kt` (7 values + `isEngineSelector`).
- `internal/runtime/RepositoryBackedModelSession.kt`, `internal/config/ConfigMappers.kt`.
- `GenerationConfig.maxNewTokens` is the public field from day one (`PublicConfigs.kt:59`) and maps to
  `maxSequenceLength` (`ConfigMappers.kt:89`) — the canonical-decision lock holds.
- 40 facade JVM tests across 8 files (ConfigMapper 7, ConfigMappingDelta 4, ExceptionMessage 6,
  FacadeDelegation 2, FeatureAndVariant 4, PeftMapping 8, RagConfigMapper 4, SamplingMapping 5).

**Gaps:**
- (a) **Public-surface lint test missing** — and it would currently **fail**:
  `runtime/Results.kt:19` declares `val summary: com.martinkorelic.mobiletransformers.ORTTrainerNative.TrainingSummary?`
  inside the public `TrainingResult`. An `ORT*`/`*Native` type **does** leak into a public signature,
  contradicting #17's DoD. (Note this is *mandated* by #18's data contract, which literally specifies
  `summary: ORTTrainerNative.TrainingSummary?` — the two plans contradict each other and nobody resolved it.)
- (a) `hub/HuggingFaceHubClient.kt` / `HubDownloadRequest.kt` / `HubAuthProvider.kt` never created.
  Superseded by #21's `hub/{HubResolver,DownloadPlanner,PackageDownloader,HubDownloader,PackageDownloadWorker}.kt`
  — a better outcome, but the substitution is undocumented in either plan.
- (a) `ModelSession` is declared **public**, not `internal` as the plan's contract block specifies
  (`runtime/ModelSession.kt` — `interface ModelSession`). It therefore is part of the ABI.
- (a) `internal/legacy/Aliases.kt` not created (see #16).
- (b) The plan's integration test (drive the validator over a fixture cache dir; reject one missing
  `inference/generation_config.json`) is not present; `FeatureAndVariantTest` covers manifest variant
  selection, not the cache-layout probe.
- (c) Device leg open: `FacadeLoadGenerateTest.kt` exists under `androidTest/` but is device-run.

**Drift / doubtful claims:** self-check bullet 2 is `[x]` on "engine selector + exception hierarchy in
place"; both hold (`runtime/InferenceEngine.kt` from #11, full sealed hierarchy from #19). Bullet 1's
"no `ORT*` in public signatures" is contradicted by `Results.kt:19`.

---

### #18 — Training lifecycle & checkpoint contracts (`00_code_plans/08`)

**Required:** `training/{TrainingJob,TrainingStatus,TrainingEvent,CheckpointInfo,TrainingJobManager}.kt`;
a `TrainingCallback`→event adapter mapping 1:1; `@Volatile cancelRequested` checked in the epoch/step loops
with the existing `saveModel`+`saveTrainingState` path; `CheckpointInfo` a read-only projection with
**no format change**; `canResume` = state file exists **AND** `loadFromState` would load it;
`TrainingJobManager` one job per `sanitizedRepoId` + a `TrainingJobSpec` WorkManager seam (no WM dep);
tests: status/event mapping, **cancel (both `saveCheckpoint` values)**, checkpoint round-trip.

**Verified present:**
- All five files exist: `training/TrainingStatus.kt:11` (sealed), `TrainingEvent.kt:11` (sealed),
  `CheckpointInfo.kt:13-34` (Gson projection of `TrainingState`, `read(checkpointDirPath, stateJsonPath)`),
  `TrainingEventAdapter.kt:20-99` (`status: StateFlow`, `events: SharedFlow`, `markCancelled`),
  `TrainingJob.kt:17-56` (`status`/`events`/`start`/`cancel`/`checkpoint`/`canResume`),
  `TrainingJobManager.kt:11-30` (`TrainingJobSpec`, `getOrCreate` keyed by `PackageFormat.sanitizeRepoId`).
- `ORTTrainerNative.kt:44-47` — `@Volatile var cancelRequested` (annotation present), checked at `:224`
  (epoch loop) and `:254` (step loop). No `TrainingState`/`SchedulerState` format change.
- No WorkManager dependency added to the SDK module (confirmed: `build.gradle.kts` deps).
- Tests: `CheckpointInfoTest` (2: `projectsStateAndPreservesFormat`, `absentStateFileProjectsExistsFalse`),
  `TrainingEventAdapterTest` (3: `statusTransitionsFollowCallbacks`, `eventStreamOrderMatchesScriptedCallbacks`,
  `errorTransitionsToFailed`).

**Gaps:**
- (b) **`TrainingJob` / `TrainingJobManager` have no consumer.** Grep across all of `main/` finds zero
  references outside `training/` itself (the only hit is a comment in `ORTTrainerNative.kt:44`).
  `MobileTransformerModel.train` (`:43`) returns `TrainingResult` via `session.train`, never a
  `TrainingJob`; `RepositoryBackedModelSession` builds its own callback path. So the plan's central
  deliverable — "a public, lifecycle-shaped training surface **the facade can expose**" — is present as
  code but is **dead**: no `status`/`events`/`cancel`/`canResume` is reachable from the public API.
- (a) **Cancel unit test missing.** The plan spells out two cases (`saveCheckpoint=true` → loop breaks,
  `saveTrainingState` invoked, `Cancelled(checkpoint)` emitted, `currentGlobalStep` reflects persisted step;
  `saveCheckpoint=false` → no state write). Neither exists. `TrainingEventAdapter.markCancelled` is untested.
- (a) **`canResume` drift.** `TrainingJob.kt:55-56` — `get() = checkpoint()?.exists == true`. The plan
  requires "state file exists **AND** `trainingConfig.loadFromState` would load it"; `loadFromState` is
  never consulted, so `canResume` can report true for a config that will not resume.
- (c) Device legs open: resume/no-double-count, `profileMetrics` summary, train→merge→generate
  (`androidTest/TrainMergeGenerateTest.kt` exists but is device-run).

**Drift / doubtful claims:** self-check bullet 3 (`session lock + cooperative cancellation defined for reuse
by the scheduler (#34)`) is `[x]`; cooperative cancel exists, but there is **no session lock** anywhere
(`TrainingJobManager` holds a `jobs` map with no locking, and nothing prevents two callers driving
`LLMRepository.runTraining` concurrently). Bullet 2's "adapter complete" is supportable. Bullet 1's
"without hiding the native lifecycle" is supportable in isolation but moot given nothing consumes it.

---

## Cross-plan gaps for this tier

1. **The registry-consumption half of #6 is orphaned.** Four `if/elif` sites the plan was written to delete
   still exist (`trainer/builder.py:286-350`, `inference/builder.py:3237-3269`,
   `weight_merger.cpp:672/708/759/833/853`, `peft_models/{mars,ablation}/utils.py`). #6 deferred them to
   #7/#9; both are closed and did not do them. **No plan currently owns this work** — it needs an explicit
   re-assignment, and #6's `[x]` + "grep guard passes" DoD should be un-checked until then.
2. **Kotlin closed-set fields are still `String` with a silent-default parser.** `FileUtil.kt:154`
   defaults an unknown `schedulerType` to Linear with a `println` — a direct violation of the canonical
   "typed fail-closed parsing, closed-set values through enum `fromWire()`" decision. The Kotlin enum
   mirrors exist and pass parity, but nothing on the read path uses them (except `SamplingMethod` at
   `ORTGeneratorNative.kt:305`). Parity being green gives **false assurance**: it only checks the mirror
   files, not that they're used.
3. **No CI enforcement for two named guards.** The #4 secrets grep and the #6 dispatch grep are both
   described as CI guards; neither is in `Makefile` or `.github/workflows/ci.yml`. Both currently pass by
   luck of the last manual run.
4. **No C++ test harness exists at all.** #8's two integration tests, and any future C++ regression test,
   have nowhere to live. `check_compat` is mirrored in three languages; only Python and Kotlin are pinned
   to the shared `check_compat_cases.json` oracle.
5. **No Robolectric in the SDK module** (`MobileTransformers/build.gradle.kts:96-98`), which is why #13's
   and #17's `LLMRepository`-shape integration tests are absent and substituted with file-existence proxies.
6. **`ORT*` leak / plan contradiction.** #17 DoD forbids `ORT*` in public signatures; #18's data contract
   mandates `summary: ORTTrainerNative.TrainingSummary?`. Implementation followed #18
   (`runtime/Results.kt:19`). Needs an ownership ruling (wrap `TrainingSummary` in a neutral type, or
   amend #17's DoD).
7. **`00_code_plans/04` (#16) is stale as written** — its Gradle-naming data contract, its option-A JNI
   decision, and its `Aliases.kt` deliverable were all superseded. Only IMPLEMENTATION_ORDER records this;
   the plan file itself would mislead a cold agent.

---

## Remaining work, ordered

### Host-doable now (no device, no user)
1. **#6 legacy dispatch migration (largest item).** In order: `build_adapter_mapping` in
   `config/registry/peft.py` → replace `trainer/builder.py:286-350` PEFT chain and
   `trainer/utils.py:535/672` mapping builders → collapse `peft_models/{mars,ablation}/utils.py` tables into
   `ArchitectureSpec.target_modules` → `inference/builder.py:3237-3269` through `resolve_architecture` →
   `weight_merger.cpp` `merger_type` branches through the registry/`MergerVariant`.
2. **#6 Kotlin step 9.** Swap the closed-set `String` fields in `ORTGenerationConfig.kt` /
   `ORTTrainingConfig.kt` / `ORTRagConfig.kt` to the existing enums; make `FileUtil.kt:139-175` fail closed
   via `fromWire()` (delete the silent Linear default + `println`); `ConfigurationScreen.kt:639/769/969` →
   `SamplingMethod.entries` / `CoreConfigId.entries` / `SchedulerType.entries`.
3. **#6 step 8 leftovers.** `SessionOptions` Pydantic model (`trainer/validator.py:405`), `large_model`
   config field (`artifact/onnx_builder.py:555`), lora-xs reconstruction config
   (`peft_models/lora_xs/merger.py:34,42`).
4. **#13 installer hardening.** Call `ManifestValidator.validate` + `ChecksumVerifier.verify` inside
   `ModelPackageInstaller.install`; add the post-install feature-probe assertion + rollback; publish via
   rename-then-delete-old rather than delete-then-rename.
5. **#18 wiring + tests.** Expose `TrainingJob` from the facade (or delete it and fold status/events into
   `ModelSession`); add the two cancel tests; fix `canResume` to consult `loadFromState`.
6. **#17 public-surface lint.** Add the "no `ORT*` in public API" test, and resolve the
   `TrainingResult.summary` leak (neutral `TrainingSummary` data class in `runtime/`).
7. **#5 exception discipline.** Convert `hub/package_format.py:128,132`, `config/settings.py:38`,
   `export/onnx_config_with_loss.py:132,154` to typed subclasses; extend `tests/unit/test_exceptions.py`
   to actually assert the documented failure paths.
8. **#4/#6 CI guards.** Add `make guard` (secrets grep over `src/` + dispatch-literal grep over
   `trainer/ inference/ artifact/ cpp/`) and wire it into the `fast` job. Note: the dispatch guard will
   fail until item 1 lands — land it as a ratchet.
9. **Test infrastructure.** Add Robolectric to `MobileTransformers/build.gradle.kts` testImplementation so
   #13's and #17's `LLMRepository`-shape integration tests can be written; stand up a C++ test target
   (googletest under `cpp/`) for #8's save→load smoke and the `check_compat_cases.json` C++ mirror.
10. **Doc hygiene.** Rewrite `00_code_plans/04`'s Gradle-naming contract + JNI decision to option B;
    delete the stale "build-side emit deferred" note in `00_code_plans/07`; un-check #6's `[x]` and
    #4's CI-guard bullet; record the `hub/HuggingFaceHubClient*` → `hub/HubResolver*` substitution in #17.

### Device-required
- #16: x86_64 native link + AAR/APK packaging check (needs a complete vendored `jniLibs/x86_64`).
- #17: facade load → generate one token (`androidTest/FacadeLoadGenerateTest.kt`).
- #18: resume/no-double-count; `profileMetrics` summary from `training_logs.json`; train→merge→generate
  (`androidTest/TrainMergeGenerateTest.kt`).
- #8/#13: on-device merge → map-driven load → generate over a real #9 package; install-then-generate smoke.

### Manual / user-run
- #2: `uv sync --frozen --group ort-training-local` alive-check; `uv sync --extra export --group
  ort-training-local` must-fail check; `sha256sum` of the wheel vs `manifest.json`; `uv export`
  determinism re-run. (All env-mutating — must be run by the user, not an agent sharing the `.venv`.)
- #2/#16: `scripts/build_ort_training_android.sh` to populate the manifest's Android fields.
- #5: full real-model export regression after the #6 decomposition lands (rides #15).
