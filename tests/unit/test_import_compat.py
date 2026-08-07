"""Import-compatibility smoke: the new package imports, and legacy root imports
still resolve during migration."""

import warnings

import pytest


def test_mobiletransformers_imports():
    import mobiletransformers

    # The version is asserted against pyproject.toml (the single write-site) in test_version_sites.py;
    # hardcoding it here made a THIRD place to update on every release.
    assert isinstance(mobiletransformers.__version__, str)
    assert mobiletransformers.__version__


def test_legacy_parser_config_imports():
    # Root package must keep resolving throughout migration (shim added in config plan).
    from tools.parser_config import ARTIFACT_CONFIG, INFERENCE_CONFIG, TRAIN_CONFIG

    assert TRAIN_CONFIG == "TRAIN_BUILDER"
    assert ARTIFACT_CONFIG == "ARTIFACT_BUILDER"
    assert INFERENCE_CONFIG == "INFERENCE_BUILDER"


# --- Migration Map shims: every old import path still resolves -------------------------------------
# `old dotted module` -> symbols the shim must still expose. Identity (`is`) is asserted where the
# symbol is core-importable, which proves the shim RE-EXPORTS rather than redefining.
S1_SHIMS = {
    "tools.utils": [
        ("move_onnx_model", "mobiletransformers.utils.paths"),
        ("move_files_excluding", "mobiletransformers.utils.paths"),
        ("delete_directory", "mobiletransformers.utils.paths"),
        ("create_chat_input", "mobiletransformers.utils.templating"),
        ("render_template", "mobiletransformers.utils.templating"),
        ("load_and_save_dataset", "mobiletransformers.training.data"),
        ("trim_dataset", "mobiletransformers.training.data"),
        ("save_as_jsonl", "mobiletransformers.training.data"),
        ("preload_dataset", "mobiletransformers.training.data"),
    ],
    "tools.tokenizer_export": [
        ("export_tokenizer_config", "mobiletransformers.export.tokenizer_export"),
        ("export_tokenizer_config_advanced", "mobiletransformers.export.tokenizer_export"),
    ],
    "inference.generator": [
        ("generate_tokens_onnx", "mobiletransformers.inference.generator"),
    ],
}


@pytest.mark.parametrize(
    ("old_module", "symbol", "new_module"),
    [(old, sym, new) for old, pairs in S1_SHIMS.items() for sym, new in pairs],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_shim_reexports_the_moved_symbol(old_module, symbol, new_module):
    import importlib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim = importlib.import_module(old_module)
    new = importlib.import_module(new_module)
    assert getattr(shim, symbol) is getattr(new, symbol), (
        f"{old_module}.{symbol} is not the same object as {new_module}.{symbol} — "
        "the shim redefines rather than re-exports"
    )


@pytest.mark.parametrize("old_module", sorted(S1_SHIMS))
def test_shim_warns_on_import(old_module):
    """A shim must be visibly deprecated, or callers never migrate off it."""
    import importlib
    import sys

    sys.modules.pop(old_module, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(old_module)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"{old_module} does not emit a DeprecationWarning"
    )
