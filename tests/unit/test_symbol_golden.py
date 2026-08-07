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
    path = current_path(dotted)
    assert path.is_file(), (
        f"{dotted} is at neither its original location nor a MODULE_LOCATIONS entry — "
        "if it moved, record the new path there"
    )

    expected = set(GOLDEN[dotted])
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
