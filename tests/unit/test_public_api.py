"""Public-API guard: `__all__` is non-empty, importable, and matches the checked-in golden.

If you intentionally change the public surface, regenerate the golden:
    python -c "import mobiletransformers,pathlib; \
      pathlib.Path('src/mobiletransformers/public_api.txt').write_text(chr(10).join(sorted(mobiletransformers.__all__))+chr(10))"
"""

from __future__ import annotations

from importlib.resources import files

import mobiletransformers


def test_all_is_declared_and_nonempty():
    assert mobiletransformers.__all__


def test_all_names_are_importable():
    for name in mobiletransformers.__all__:
        assert hasattr(mobiletransformers, name), f"{name} in __all__ but not importable"


def test_all_matches_golden():
    golden = files("mobiletransformers").joinpath("public_api.txt").read_text().split()
    assert sorted(mobiletransformers.__all__) == sorted(golden), (
        "public surface drifted from public_api.txt; regenerate the golden if intentional"
    )
