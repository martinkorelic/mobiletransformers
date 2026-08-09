"""Migration net: every legacy module's public symbols survive its move into the package.

The Migration Map moves ~17.5k lines of essentially untested code. These modules cannot be imported in
the core environment (torch / onnxruntime / optimum), and `inference/builder.py` is unimportable under
*every* declared profile — so the net is static. That is the right shape anyway: the failure mode being
guarded is "a symbol silently disappeared during a `git mv` + split", which AST comparison catches
exactly.

**To move a module:** move it, then add one line to `MODULE_LOCATIONS`. If the symbols still match, the
move was symbol-preserving. If they do not, the diff names precisely what was dropped.

Regenerate the golden only to add a module or record a deliberate, reviewed change:
`python tests/fixtures/gen_legacy_symbol_golden.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.gen_legacy_symbol_golden import public_symbols

REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG = "src/mobiletransformers"
GOLDEN = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "legacy_symbol_golden.json").read_text(encoding="utf-8")
)

#: Logical module name -> its CURRENT repo-relative path. Add an entry when a module moves; the key
#: stays the original name so the golden keeps working as the index.
#:
#: A module whose old path still holds a deprecation SHIM needs no entry: the shim re-exports the same
#: `__all__`, so the golden matches at the original path either way. Entries here are for modules
#: whose old path is gone, or that were SPLIT across several new modules (recorded as the split's
#: primary home — the shim is what proves the whole surface survived).
MODULE_LOCATIONS: dict[str, str] = {
    # S5 — TFLite export needs tensorflow/keras/keras_nlp, which are in NO dependency profile, so it
    # cannot ship in the package. Moved to the research tree rather than migrated.
    "artifact.tflite_builder": "research/tflite/tflite_builder.py",
    # S1 — split by concern; tools/utils.py remains as a shim re-exporting all ten names.
    #   move_onnx_model/move_files_excluding/delete_directory -> utils/paths.py
    #   create_chat_input/render_template                     -> utils/templating.py
    #   load_and_save_dataset/trim_dataset/save_as_jsonl/preload_dataset -> training/data.py
    #   MemoryLoggerCallback                                  -> training/callbacks.py
    # S7 — the whole database/ root moved to rag/; the old paths keep import shims, but the golden
    # should verify the REAL module, so it is pointed at the new home rather than at the re-export.
    "database.builder": "src/mobiletransformers/rag/builder.py",
    "database.query": "src/mobiletransformers/rag/query.py",
    "database.vector_entity": "src/mobiletransformers/rag/vector_entity.py",
    "database.json2entity": "src/mobiletransformers/rag/json2entity.py",
    # S8 — the reusable evaluators moved into the package; the old paths keep import shims, but the
    # golden should verify the REAL module, so it follows them to their new home.
    "evaluation.eval_adapter_models": "src/mobiletransformers/evaluation/eval_adapter_models.py",
    "evaluation.eval_adapter_onnx_model": "src/mobiletransformers/evaluation/eval_adapter_onnx_model.py",
    "evaluation.mobile_evaluator": "src/mobiletransformers/evaluation/mobile_evaluator.py",
    "evaluation.mobile.base_mobile_eval": "src/mobiletransformers/evaluation/mobile/base_mobile_eval.py",
    "evaluation.mobile.mobile_eval": "src/mobiletransformers/evaluation/mobile/mobile_eval.py",
    "evaluation.mobile.recommendation_eval": (
        "src/mobiletransformers/evaluation/mobile/recommendation_eval.py"
    ),
    "evaluation.openehr.openehr_eval": "src/mobiletransformers/evaluation/openehr/openehr_eval.py",
    "evaluation.openehr.openehr_eval_plots": (
        "src/mobiletransformers/evaluation/openehr/openehr_eval_plots.py"
    ),
    # S8 — hardcoded-path experiment scripts have NO importable API (zero classes/defs, work done in
    # top-level statements), so they went to research/ rather than into an installable wheel — the same
    # call S5 made for artifact/tflite_builder.py. See research/evaluation/README.md.
    "evaluation.benchmark.arc_eval": "research/evaluation/benchmark/arc_eval.py",
    "evaluation.benchmark.boolq_eval": "research/evaluation/benchmark/boolq_eval.py",
    "evaluation.benchmark.hellaswag_eval": "research/evaluation/benchmark/hellaswag_eval.py",
    "evaluation.benchmark.logiqa_eval": "research/evaluation/benchmark/logiqa_eval.py",
    "evaluation.benchmark.winogrande_eval": "research/evaluation/benchmark/winogrande_eval.py",
    "evaluation.test.test_eval_onnx": "research/evaluation/scripts/test_eval_onnx.py",
    "evaluation.test.test_gen": "research/evaluation/scripts/test_gen.py",
    "evaluation.test.test_gen_viz": "research/evaluation/scripts/test_gen_viz.py",
    # S6b — inference/validator.py -> artifacts/validation.py
    "inference.validator": "src/mobiletransformers/artifacts/validation.py",
    "trainer.validator": "src/mobiletransformers/training/validators.py",
    "trainer.merge_validator": "src/mobiletransformers/training/merge_validators.py",
    "inference.builder": "src/mobiletransformers/inference/builder.py",
    # S9 — the deprecation shims are GONE, so every remaining golden module needs its real home
    # recorded here. Until S9 these resolved by falling back to the shim still sitting at the old
    # path; that fallback no longer exists.
    "inference.generator_genai": "research/genai/generator_genai.py",
    "artifact.merger": "src/mobiletransformers/config/registry/merger.py",
    "artifact.onnx_builder": "src/mobiletransformers/artifacts/builder.py",
    "inference.export_inference_package": "src/mobiletransformers/export/inference_package.py",
    "inference.generator": "src/mobiletransformers/inference/generator.py",
    "peft_models.ablation.config": "src/mobiletransformers/peft/ablation/config.py",
    "peft_models.ablation.layer": "src/mobiletransformers/peft/ablation/layer.py",
    "peft_models.ablation.model": "src/mobiletransformers/peft/ablation/model.py",
    "peft_models.ablation.utils": "src/mobiletransformers/peft/ablation/utils.py",
    "peft_models.lora_xs.initialization_utils": "src/mobiletransformers/peft/lora_xs/initialization_utils.py",
    "peft_models.lora_xs.latent_utils": "src/mobiletransformers/peft/lora_xs/latent_utils.py",
    "peft_models.lora_xs.merger": "src/mobiletransformers/peft/lora_xs/merger.py",
    "peft_models.lora_xs.svd_utils": "src/mobiletransformers/peft/lora_xs/svd_utils.py",
    "peft_models.mars.config": "src/mobiletransformers/peft/mars/config.py",
    "peft_models.mars.layer": "src/mobiletransformers/peft/mars/layer.py",
    "peft_models.mars.model": "src/mobiletransformers/peft/mars/model.py",
    "peft_models.mars.study": "src/mobiletransformers/peft/mars/study.py",
    "peft_models.mars.utils": "src/mobiletransformers/peft/mars/utils.py",
    "tools.parser_config": "src/mobiletransformers/config/constants.py",
    "tools.tokenizer_export": "src/mobiletransformers/export/tokenizer_export.py",
    "trainer.builder": "src/mobiletransformers/export/training_export.py",
    "trainer.embedding_builder": "src/mobiletransformers/export/embedding_export.py",
}

#: Modules that now live under the package, keyed by their new path. Used by the placeholder check in
#: `test_import_weight.py` so migrated code is recognised as migrated rather than as a stray file.
MIGRATED_PATHS: set[str] = {
    "src/mobiletransformers/utils/paths.py",
    "src/mobiletransformers/utils/templating.py",
    "src/mobiletransformers/training/data.py",
    "src/mobiletransformers/training/callbacks.py",
    "src/mobiletransformers/inference/generator.py",
    "src/mobiletransformers/export/tokenizer_export.py",
    # S2
    "src/mobiletransformers/export/inference_package.py",
    # S4
    "src/mobiletransformers/training/preprocessing.py",
    "src/mobiletransformers/peft/mapping.py",
    "src/mobiletransformers/export/embedding_export.py",
    "src/mobiletransformers/export/training_export.py",
    # S5
    "src/mobiletransformers/artifacts/builder.py",
    # S8 — evaluation/ -> evaluation/ (library evaluators only)
    "src/mobiletransformers/evaluation/eval_adapter_models.py",
    "src/mobiletransformers/evaluation/eval_adapter_onnx_model.py",
    "src/mobiletransformers/evaluation/mobile_evaluator.py",
    "src/mobiletransformers/evaluation/mobile/base_mobile_eval.py",
    "src/mobiletransformers/evaluation/mobile/mobile_eval.py",
    "src/mobiletransformers/evaluation/mobile/recommendation_eval.py",
    "src/mobiletransformers/evaluation/openehr/openehr_eval.py",
    "src/mobiletransformers/evaluation/openehr/openehr_eval_plots.py",
    # S6b — inference/validator.py -> artifacts/validation.py
    "src/mobiletransformers/artifacts/validation.py",
    # S8 side-effect — `load_mars_adapters` extracted from research/utils.py, because its only caller
    # is now a PACKAGED module and `research/` is not part of the distribution.
    "src/mobiletransformers/peft/adapters.py",
    # S6 — the last and largest legacy module
    "src/mobiletransformers/inference/builder.py",
    # S6b — trainer validators
    "src/mobiletransformers/training/validators.py",
    "src/mobiletransformers/training/merge_validators.py",
    # S6b side-effect — the benchmark dataset registry extracted from research/offline_train_eval.py,
    # same reason as peft/adapters.py above.
    "src/mobiletransformers/training/benchmark_datasets.py",
    # S7 — database/ -> rag/
    "src/mobiletransformers/rag/builder.py",
    "src/mobiletransformers/rag/query.py",
    "src/mobiletransformers/rag/vector_entity.py",
    "src/mobiletransformers/rag/json2entity.py",
    # S3 — peft_models/{mars,lora_xs,ablation} + create_orthogonal_matrices vendored from research/.
    *(
        f"src/mobiletransformers/peft/{sub}/{name}.py"
        for sub, names in {
            "mars": ("config", "layer", "model", "study", "utils", "matrices"),
            "lora_xs": ("initialization_utils", "latent_utils", "merger", "svd_utils", "__init__"),
            "ablation": ("config", "layer", "model", "utils"),
        }.items()
        for name in names
    ),
}


#: Modules that were SPLIT BY CONCERN rather than moved, so no single path holds them. `MODULE_LOCATIONS`
#: maps one module to one file and cannot express this; before S9 these resolved via the shim that stayed
#: behind re-exporting all the parts. With the shims deleted the golden has to check the union.
MODULE_SPLITS: dict[str, tuple[str, ...]] = {
    # S1
    "tools.utils": (
        f"{_PKG}/utils/paths.py",
        f"{_PKG}/utils/templating.py",
        f"{_PKG}/training/data.py",
        f"{_PKG}/training/callbacks.py",
    ),
    # S4
    "trainer.utils": (
        f"{_PKG}/training/preprocessing.py",
        f"{_PKG}/peft/mapping.py",
    ),
}

#: Legacy PACKAGE `__init__.py` files. The golden records ZERO public symbols for each, so deleting them
#: in S9 loses nothing — asserted below rather than assumed.
REMOVED_EMPTY_PACKAGES = frozenset(
    {"artifact", "inference", "peft_models", "peft_models.lora_xs", "tools", "trainer"}
)


def current_path(dotted: str) -> Path:
    relocated = MODULE_LOCATIONS.get(dotted)
    if relocated:
        return REPO_ROOT / relocated
    candidate = REPO_ROOT / (dotted.replace(".", "/") + ".py")
    if candidate.is_file():
        return candidate
    return REPO_ROOT / dotted.replace(".", "/") / "__init__.py"


@pytest.mark.parametrize("dotted", sorted(GOLDEN), ids=lambda d: d)
def test_public_symbols_survive_relocation(dotted: str) -> None:
    expected = set(GOLDEN[dotted])

    if dotted in REMOVED_EMPTY_PACKAGES:
        # A legacy package `__init__.py` deleted in S9. Only safe because it defined nothing.
        assert not expected, (
            f"{dotted} is listed as an empty legacy package but the golden records {sorted(expected)} — "
            "it carried symbols, so it cannot simply be deleted"
        )
        return

    if dotted in MODULE_SPLITS:
        paths = [REPO_ROOT / rel for rel in MODULE_SPLITS[dotted]]
        missing_files = [str(p.relative_to(REPO_ROOT)) for p in paths if not p.is_file()]
        assert not missing_files, f"{dotted} split into missing file(s): {missing_files}"
        actual: set[str] = set()
        for part in paths:
            actual |= set(public_symbols(part))
        dropped = sorted(expected - actual)
        assert not dropped, f"{dotted} was split across {MODULE_SPLITS[dotted]} and the union lost: {dropped}"
        return

    path = current_path(dotted)
    assert path.is_file(), (
        f"{dotted} is at neither its original location nor a MODULE_LOCATIONS entry — "
        "if it moved, record the new path there"
    )

    actual = set(public_symbols(path))
    dropped = sorted(expected - actual)
    assert not dropped, f"{dotted} ({path.relative_to(REPO_ROOT)}) lost public symbol(s): {dropped}"


def test_golden_covers_every_legacy_module() -> None:
    """A new legacy module must be recorded before it can be moved."""
    from tests.fixtures.gen_legacy_symbol_golden import collect

    missing = sorted(set(collect()) - set(GOLDEN))
    assert not missing, (
        f"legacy modules absent from the golden: {missing} — run "
        "`python tests/fixtures/gen_legacy_symbol_golden.py`"
    )


#: Modules deliberately relocated OUTSIDE the package, with the reason. Anything else must land in
#: `src/mobiletransformers/`.
RELOCATED_OUT_OF_PACKAGE = {
    "artifact.tflite_builder": "needs tensorflow/keras/keras_nlp — in no dependency profile",
    # S8 — these have NO importable API: zero classes, zero functions, all work in top-level statements
    # (so importing one runs a benchmark), against hardcoded `experiment_results/...` paths. An
    # installable wheel must not ship modules that execute an experiment on import. They stay in the
    # repo as the record of how published numbers were produced. See research/evaluation/README.md.
    **{
        f"evaluation.benchmark.{name}_eval": "top-level experiment script, hardcoded paths — not a library"
        for name in ("arc", "boolq", "hellaswag", "logiqa", "winogrande")
    },
    **{
        f"evaluation.test.{name}": "top-level experiment script, hardcoded paths — not a test either"
        for name in ("test_eval_onnx", "test_gen", "test_gen_viz")
    },
    # S9 — a desktop prototype with no importers, kept as the reference the plans cite.
    "inference.generator_genai": "desktop GenAI prototype, hardcoded path — reference, not library",
}


def test_relocations_point_at_the_package() -> None:
    """A relocation must land under `src/mobiletransformers/` unless it is a recorded exception."""
    for dotted, location in MODULE_LOCATIONS.items():
        if dotted in RELOCATED_OUT_OF_PACKAGE:
            assert not location.startswith("src/"), (
                f"{dotted} is recorded as out-of-package but points into src/"
            )
            continue
        assert location.startswith("src/mobiletransformers/"), (
            f"{dotted} relocated to {location}, which is outside the package"
        )


# --- decorators ------------------------------------------------------------------------------------
DECORATOR_GOLDEN = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "legacy_decorator_golden.json").read_text(encoding="utf-8")
)


def _decorators_at(path: Path) -> dict[str, list[str]]:
    from tests.fixtures.gen_legacy_symbol_golden import decorated_definitions

    return decorated_definitions(path) if path.is_file() else {}


@pytest.mark.parametrize("dotted", sorted(DECORATOR_GOLDEN), ids=lambda d: d)
def test_decorators_survive_relocation(dotted: str) -> None:
    """A dropped DECORATOR changes behaviour while every symbol name survives.

    Caught for real during S4: slicing `trainer/utils.py` by line started at
    `class DataCollatorForSupervisedDataset` and left `@dataclass` behind, silently removing the
    generated `__init__`. The symbol golden saw nothing wrong.
    """
    expected = DECORATOR_GOLDEN[dotted]

    # A definition may still be at its original path, or have moved into any migrated module. Search
    # both, rather than hand-maintaining a second relocation map that could itself go stale.
    found: dict[str, list[str]] = dict(_decorators_at(current_path(dotted)))
    for migrated in sorted(MIGRATED_PATHS):
        for symbol, decorators in _decorators_at(REPO_ROOT / migrated).items():
            found.setdefault(symbol, decorators)

    for symbol, decorators in expected.items():
        assert symbol in found, (
            f"{dotted}.{symbol} carried {decorators} but is no longer a decorated top-level "
            "definition anywhere — a decorator was dropped during a move"
        )
        lost = sorted(set(decorators) - set(found[symbol]))
        assert not lost, f"{dotted}.{symbol} lost decorator(s): {lost}"
