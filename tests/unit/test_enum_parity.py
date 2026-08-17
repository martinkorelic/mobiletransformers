"""Cross-language enum parity (F2): Kotlin wire values == Python enum values, schemas regenerable."""

from __future__ import annotations

from mobiletransformers.codegen.enums import (
    KOTLIN_CONSTANTS_RELPATH,
    check,
    enums_golden,
    find_repo_root,
    parse_kotlin_enums,
)
from mobiletransformers.config.constants import ENUM_REGISTRY


def test_parity_check_passes():
    # Equivalent to `python -m mobiletransformers.codegen.enums --check`.
    drifts = check(find_repo_root())
    assert drifts == [], "enum/schema parity drift:\n" + "\n".join(drifts)


def test_kotlin_wire_values_equal_python():
    repo_root = find_repo_root()
    kotlin = parse_kotlin_enums(repo_root / KOTLIN_CONSTANTS_RELPATH)
    for name, enum in ENUM_REGISTRY.items():
        assert kotlin.get(name) == {m.value for m in enum}, f"{name} Kotlin/Python drift"


def test_golden_enums_json_matches_source():
    import json

    repo_root = find_repo_root()
    golden = json.loads((repo_root / "schemas" / "enums.json").read_text())
    # declaration-order lists in enums.json == the Python source
    assert golden == enums_golden()
