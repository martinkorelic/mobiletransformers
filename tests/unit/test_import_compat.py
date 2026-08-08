"""Import-compatibility smoke: the new package imports, and legacy root imports
still resolve during migration."""




def test_mobiletransformers_imports():
    import mobiletransformers

    # The version is asserted against pyproject.toml (the single write-site) in test_version_sites.py;
    # hardcoding it here made a THIRD place to update on every release.
    assert isinstance(mobiletransformers.__version__, str)
    assert mobiletransformers.__version__


# --- Migration Map S9: the shim tests were DELETED WITH THE SHIMS ------------------------------------
#
# This file used to assert that every legacy import path still resolved, re-exported the identical
# object (`is`, not `==`, so a shim could not redefine), and emitted a DeprecationWarning. Those tests
# were the reason the shims were trustworthy while they existed.
#
# S9 removed the shims: `trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`
# and `evaluation/` are gone from the repo root, so there is no longer an old path to assert about.
# What replaced this coverage is stronger, not weaker:
#
#   * `test_symbol_golden.py` proves every public symbol survived to its new home — including the two
#     modules that were SPLIT across several files (`tools.utils`, `trainer.utils`), which no shim
#     test ever checked as a union.
#   * `test_no_src_to_legacy_imports.py` runs with both allow-lists EMPTY, so nothing in `src/` can
#     reach for a legacy root again.
#   * `test_import_weight.py` proves the built wheel is self-contained.
