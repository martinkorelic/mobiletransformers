# Python Code Quality, Conventions & Module Health

**Priority (global #):** 5  |  **Prerequisites:** #1 (`00_code_plans/01_python_package_and_uv_scaffolding.md`), #2 (`00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`), #4 (`00_code_plans/02_config_layering_settings_constants.md`)  |  **Blocks:** #6 (`00_code_plans/09_typed_models_enums_and_registries.md`), #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), and every Python builder plan that touches a monolith

## Purpose

Make the Python codebase **consistent, typed, lint-clean, and decomposable** so the framework is extensible
and its public API is stable — **without a big-bang rewrite**. Today the export toolkit is a set of large,
loosely-typed modules with no enforced conventions: `inference/builder.py` (~200 KB), `trainer/validator.py`
(~51 KB), `artifact/merger.py` (~44 KB), `artifact/onnx_builder.py` (~38 KB), `trainer/embedding_builder.py`
(~27 KB), `trainer/utils.py` (~26 KB), `database/builder.py` (~28 KB), plus a sprawling `research/` tree, no
`ruff`/`mypy` config, ad-hoc `print()` logging, and no declared library surface.

This plan establishes the **tooling, shared foundation, and decomposition strategy** *before* the registry
work (#6) and the merger/builder rewrites (#9, #7) land — so those plans inherit the conventions instead of
each reinventing them. It is foundational but **non-blocking for behaviour**: it sets standards and ratchets,
not a freeze.

> Scope boundary: this plan owns *conventions, tooling, shared utils, the decomposition map, and the public
> API surface*. The actual per-module splits are executed by the plans that already touch those modules
> (#6 registries → per-architecture inference builders; #9 merger → name-resolution split), so decomposition
> rides along rather than as a separate risky pass.

## Touched / new files

- `pyproject.toml` (from #1/#2) — add `[tool.ruff]` (lint + format) and `[tool.mypy]` (typing) sections;
  add `ruff`, `mypy` to the `dev` group.
- NEW `src/mobiletransformers/exceptions.py` — the Python exception hierarchy, deliberately **parallel to the
  Kotlin facade's** (`00_code_plans/05`): `MobileTransformersError` → `ExportError`, `ManifestError`,
  `HandoffError`, `MergeError`, `ConfigValidationError`, `UnsupportedModelError`, `HubError`.
- NEW `src/mobiletransformers/utils/logging.py` — structured, module-level `get_logger(__name__)`; bans
  `print()` in library code (CLI user output stays in `cli/`).
- NEW `src/mobiletransformers/_typing.py` (or `py.typed` marker + shared aliases) — common typing aliases
  (`TensorName`, `PathLike`, `JsonDict`) and the `py.typed` marker so downstream consumers get types.
- `src/mobiletransformers/__init__.py` (from #1) — declare `__all__`: the stable public surface
  (`export_model`, `package_model`, `pull_package`, `push_adapter`, the public config models, the enums, the
  registry `register_*` helpers, `__version__`). Everything else is internal (`_`-prefixed module or
  documented unstable).
- **Decomposition targets (strategy here; splits land with their owning plan):**
  - `inference/builder.py` (~200 KB) → `inference/graph/<arch>.py` per-architecture builders resolved through
    the **architecture registry** (#6). The Gemma/Gemma2 inference-model branch (`inference/builder.py:3234-3236`)
    becomes registry entries (also unblocks Gemma-3 in #37).
  - `artifact/merger.py` (~44 KB) — the hard-coded name-rewrite logic (mirrors C++ `weight_merger.cpp`
    `replace_prefix`) moves behind the `weight_handoff_map.json` resolver (#8/#9).
  - `trainer/validator.py` (~51 KB), `trainer/embedding_builder.py`, `artifact/onnx_builder.py` — split by
    concern (validation rules vs. graph ops vs. quantization) as they're touched.
- `research/` — **quarantine**: excluded from the built package (hatchling include rules from #1) and from
  strict lint/type gates; never part of the public API.

## Data contracts / interfaces

- **`ruff`**: line length (match repo norm), rule set (pyflakes + pycodestyle + import-sort + pyupgrade);
  `ruff format` is the formatter (no separate `black`).
- **`mypy` strictness ratchet** (an existing untyped codebase cannot go strict overnight):
  - global: `ignore_missing_imports = true` to start; CI-green from day one.
  - **new** modules (`src/mobiletransformers/**` created from #1 onward): `disallow_untyped_defs = true`.
  - legacy modules: per-module overrides relaxed, tightened over time; each tightening is a small PR.
- **Exception hierarchy** (must mirror Kotlin names from `00_code_plans/05` so errors read the same on both
  sides): single root `MobileTransformersError`; raise typed subclasses, never bare `Exception`.
- **Logging**: `logger = get_logger(__name__)` per module; levels not `print`. CLI-facing output is the CLI's
  responsibility, not library modules.
- **Module-size guideline**: no **new** module > ~800 LOC without a decomposition note; each remaining monolith
  carries a top-of-file `# DECOMPOSE: split into … (owned by #N)` tracking comment naming its target split.
- **Public API**: `__all__` is the SemVer-governed surface (`05_code_plans/05`); a `public_api.txt` golden can
  be checked in so accidental surface changes fail CI.

## Implementation steps

1. Add `[tool.ruff]` + `[tool.mypy]` to `pyproject.toml`; add `make lint` / `make typecheck` targets
   (bodies owned by `05_code_plans/01`). Baseline the ratchet so CI is **green immediately** (no mass reformat
   that buries real diffs — format-on-touch, not a single giant commit).
2. Add `exceptions.py` + `utils/logging.py` + `py.typed`; adopt in all new code and migrate the hottest paths
   (export/merge/manifest) opportunistically.
3. Declare `mobiletransformers.__all__`; mark internals; add the `public_api.txt` golden.
4. Write the **decomposition map** (target sub-module layout per monolith) and add the `# DECOMPOSE:` tracking
   comments; land the splits that ride with #6 (per-arch inference builders) and #9 (merger name resolution).
5. Quarantine `research/` (exclude from build + strict gates).
6. Add `lint`, `typecheck`, and the cross-language `parity` job to CI (wired by `05_code_plans/02`), plus a
   public-API import smoke.

## Interactions

- **#6 (`09` typed models/enums/registries)**: decomposition pivots on the architecture registry; enums/config
  are the parity source of truth this plan helps enforce.
- **#1 (scaffolding) / #4 (`02` config)**: extends their `pyproject.toml` and `config/` layout.
- **#8/#9 (handoff map / merger)**: the `artifact/merger.py` name-rewrite split lands with the merger unification.
- **`05_code_plans/01` (Makefile)**: owns the `lint`/`typecheck` target bodies.
- **`05_code_plans/02` (CI)**: owns the `lint`/`typecheck`/`parity` gates and the public-API smoke.
- **`05_code_plans/04` (docs)**: `PUBLIC_API.md` documents the Python surface declared here.
- **`05_code_plans/05` (versioning)**: SemVer needs this declared, stable Python API.

## References

- ruff (lint + format): https://docs.astral.sh/ruff/
- mypy (gradual typing, per-module strictness): https://mypy.readthedocs.io/en/stable/existing_code.html
- pytest: https://docs.pytest.org/
- PEP 561 (`py.typed` / distributing types): https://peps.python.org/pep-0561/

## Tests & acceptance

**Unit (automated)**
- `ruff check src/` and `ruff format --check src/` are clean.
- `mypy src/mobiletransformers` passes at the agreed ratchet (new modules strict).
- Public-API import smoke: `python -c "import mobiletransformers; assert mobiletransformers.__all__"`.
- `pytest tests/utils/test_exceptions.py` — every public module raises only `MobileTransformersError` subclasses
  on its documented failure paths.

**Integration (automated)**
- Parity test (`pytest tests/parity/`): regenerate `schemas/*.schema.json` + `enums.json` from the Pydantic/enum
  source of truth (#6) and assert they match the checked-in Kotlin/C++ mirrors (fails on drift).
- Decomposition round-trip: one per-architecture inference builder, invoked through the registry, produces the
  **same ONNX** (golden compare) as the pre-split monolith on a tiny model.

**Manual (user-run)**
- After the decomposition splits land, a full export of a real model (rides on #15's export E2E) still yields a
  working device package — confirms no behaviour regression from restructuring.

**Definition of done**
- `lint` + `typecheck` + `parity` + public-API import smoke are green in CI.
- `exceptions.py` + `utils/logging.py` + `py.typed` exist and are used by all new code.
- `mobiletransformers.__all__` + `public_api.txt` golden are in place.
- The decomposition map is documented, every monolith carries a `# DECOMPOSE:` note, and the #6/#9-adjacent
  splits have landed.
- `research/` is excluded from the built package and strict gates.
- *(Not a user-workflow checkpoint — foundational; workflow validation rides on #15/#19.)*
