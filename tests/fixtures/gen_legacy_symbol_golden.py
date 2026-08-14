"""ARCHIVAL — regenerated `tests/fixtures/legacy_symbol_golden.json` while the legacy roots existed.

**This script can no longer run.** `LEGACY_ROOTS` names the seven repo-root packages S9 deleted, so
`collect()` walks paths that are not there and would rewrite the golden to `{}` — silently destroying
the evidence that the migration was symbol-preserving. It is kept as the record of HOW the golden was
produced, not as a tool to re-run.

The goldens it produced are still enforced, by `tests/unit/test_symbol_golden.py`, against the modules'
CURRENT homes. The pure AST helpers those tests need moved to `tests/fixtures/symbol_tools.py`
(2026-08-14) so the live code no longer lives inside a dead generator.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.symbol_tools import decorated_definitions, public_symbols

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOTS = ("trainer", "artifact", "inference", "tools", "peft_models", "evaluation", "database")
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "legacy_symbol_golden.json"


def collect() -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for root in LEGACY_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT)
            dotted = str(rel.with_suffix("")).replace("/", ".").removesuffix(".__init__")
            modules[dotted] = public_symbols(path)
    return modules


def collect_decorators() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for root in LEGACY_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT)
            dotted = str(rel.with_suffix("")).replace("/", ".").removesuffix(".__init__")
            found = decorated_definitions(path)
            if found:
                out[dotted] = found
    return out


DECORATOR_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "legacy_decorator_golden.json"

if __name__ == "__main__":  # pragma: no cover - see the module docstring; do NOT run this
    raise SystemExit(
        "refusing to run: the seven legacy roots this generator walks were deleted in S9, so "
        "regenerating would overwrite the goldens with an empty document. See the docstring."
    )
    payload = collect()
    GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v) for v in payload.values())
    print(f"wrote {GOLDEN.relative_to(REPO_ROOT)}: {len(payload)} modules, {total} public symbols")

    decorators = collect_decorators()
    DECORATOR_GOLDEN.write_text(json.dumps(decorators, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(v) for v in decorators.values())
    print(f"wrote {DECORATOR_GOLDEN.relative_to(REPO_ROOT)}: {n} decorated definitions")
