"""Parity generator + checker for the cross-language enum/schema contract.

Run ``python -m mobiletransformers.codegen.enums`` to (re)generate the checked-in artifacts:
  * ``schemas/<Model>.schema.json`` — from ``Model.model_json_schema(by_alias=True)``.
  * ``schemas/enums.json`` — the golden ``{enumName: [wire values...]}`` from ``config.constants``.

Run ``python -m mobiletransformers.codegen.enums --check`` (the CI parity gate) to fail on drift:
  * regenerated schemas / enums.json differ from the checked-in ones, or
  * the hand-written Kotlin ``constants/*.kt`` wire values != the Python enum values.

It parses the Kotlin mirrors (regex over ``NAME("wire")``) but never writes ``.kt`` files — the
Kotlin enums are the hand-maintained mirror; this only verifies they agree with the Python source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel

from mobiletransformers.config.constants import ENUM_REGISTRY
from mobiletransformers.config.models import CROSS_BOUNDARY_MODELS


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from this file (or ``start``) until a directory containing ``pyproject.toml``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not locate repo root (no pyproject.toml found)")


# Kotlin enum mirrors live under the MobileTransformers library module (Android rename #16, done 2026-07-14).
KOTLIN_CONSTANTS_RELPATH = (
    "android/MobileTransformersApp/MobileTransformers/src/main/java/"
    "com/martinkorelic/mobiletransformers/constants"
)

_KOTLIN_ENTRY_RE = re.compile(r'\b([A-Z][A-Z0-9_]*)\s*\(\s*"([^"]*)"\s*\)')
_KOTLIN_CLASS_RE = re.compile(r"enum\s+class\s+([A-Za-z0-9_]+)")


def enums_golden() -> dict[str, list[str]]:
    """The Python source of truth: ``{enumName: [wire values in declaration order]}``."""
    return {name: [m.value for m in enum] for name, enum in ENUM_REGISTRY.items()}


def schema_for(model: type[BaseModel]) -> dict:
    return model.model_json_schema(by_alias=True)


def generate(repo_root: Path) -> None:
    """Write the checked-in schemas + enums.json from the Python source of truth."""
    schemas_dir = repo_root / "schemas"
    schemas_dir.mkdir(exist_ok=True)
    (schemas_dir / "enums.json").write_text(
        json.dumps(enums_golden(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, model in CROSS_BOUNDARY_MODELS.items():
        (schemas_dir / f"{name}.schema.json").write_text(
            json.dumps(schema_for(model), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_kotlin_enums(constants_dir: Path) -> dict[str, set[str]]:
    """Parse ``NAME("wire")`` entries per enum file. Returns ``{enumName: {wire values}}``."""
    result: dict[str, set[str]] = {}
    for kt in sorted(constants_dir.glob("*.kt")):
        text = kt.read_text(encoding="utf-8")
        class_match = _KOTLIN_CLASS_RE.search(text)
        if not class_match:
            continue
        enum_name = class_match.group(1)
        # Only entries within the enum body (before the companion object, if any).
        body = text.split("companion", 1)[0]
        wires = {m.group(2) for m in _KOTLIN_ENTRY_RE.finditer(body)}
        result[enum_name] = wires
    return result


def check(repo_root: Path) -> list[str]:
    """Return a list of drift messages (empty == parity holds)."""
    drifts: list[str] = []
    schemas_dir = repo_root / "schemas"

    # 1) enums.json + schemas must match regeneration byte-for-byte.
    expected_enums = json.dumps(enums_golden(), indent=2, sort_keys=True) + "\n"
    enums_file = schemas_dir / "enums.json"
    if not enums_file.is_file() or enums_file.read_text(encoding="utf-8") != expected_enums:
        drifts.append("schemas/enums.json is stale (run: python -m mobiletransformers.codegen.enums)")
    for name, model in CROSS_BOUNDARY_MODELS.items():
        expected = json.dumps(schema_for(model), indent=2, sort_keys=True) + "\n"
        path = schemas_dir / f"{name}.schema.json"
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            drifts.append(f"schemas/{name}.schema.json is stale (regenerate)")

    # 2) Kotlin wire values must equal the Python enum values.
    constants_dir = repo_root / KOTLIN_CONSTANTS_RELPATH
    if not constants_dir.is_dir():
        drifts.append(f"Kotlin constants dir missing: {KOTLIN_CONSTANTS_RELPATH}")
        return drifts
    kotlin = parse_kotlin_enums(constants_dir)
    python_values = {name: {m.value for m in enum} for name, enum in ENUM_REGISTRY.items()}
    for name, py_wires in python_values.items():
        kt_wires = kotlin.get(name)
        if kt_wires is None:
            drifts.append(f"Kotlin enum missing: {name}.kt")
        elif kt_wires != py_wires:
            drifts.append(f"{name} drift — python={sorted(py_wires)} kotlin={sorted(kt_wires)}")
    for name in kotlin:
        if name not in python_values:
            drifts.append(f"Kotlin enum {name} has no Python counterpart")
    return drifts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mobiletransformers.codegen.enums")
    parser.add_argument("--check", action="store_true", help="Fail on drift instead of regenerating.")
    args = parser.parse_args(argv)
    repo_root = find_repo_root()

    if args.check:
        drifts = check(repo_root)
        if drifts:
            print("PARITY DRIFT:", file=sys.stderr)
            for d in drifts:
                print(f"  - {d}", file=sys.stderr)
            return 1
        print("parity OK: schemas + enums.json + Kotlin mirrors agree with the Python source.")
        return 0

    generate(repo_root)
    print(f"regenerated schemas/ + enums.json under {repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
