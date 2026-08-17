"""Import smoke for the package itself.

The legacy-shim tests that lived here were deleted with the shims (`trainer/`, `artifact/`,
`inference/`, `tools/`, `peft_models/`, `database/`, `evaluation/` in S9; root `config.py` on
2026-08-14). Their coverage moved, and got stronger: `test_symbol_golden.py` proves every public
symbol survived to its new home (including the two modules that were SPLIT across files, which no
shim test ever checked as a union), `test_no_src_to_legacy_imports.py` keeps both allow-lists empty,
and `test_import_weight.py` proves the built wheel is self-contained.
"""


def test_mobiletransformers_imports():
    import mobiletransformers

    # The version is asserted against pyproject.toml (the single write-site) in test_version_sites.py;
    # hardcoding it here made a THIRD place to update on every release.
    assert isinstance(mobiletransformers.__version__, str)
    assert mobiletransformers.__version__
