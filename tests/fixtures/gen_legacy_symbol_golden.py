"""Regenerate `tests/fixtures/legacy_symbol_golden.json`.

    python tests/fixtures/gen_legacy_symbol_golden.py

Records the top-level public symbols of every legacy-root module so the Migration Map's moves can be
proven symbol-preserving. AST-only: none of these modules is importable in the core environment (most
need torch/onnxruntime), so a runtime `dir()` is not an option.

Run this ONLY to add a module or to record a deliberate, reviewed symbol change — never to paper over
a symbol that disappeared during a move.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOTS = ("trainer", "artifact", "inference", "tools", "peft_models", "evaluation", "database")
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "legacy_symbol_golden.json"


def decorated_definitions(path: Path) -> dict[str, list[str]]:
    """Top-level defs/classes -> their decorator names.

    A move that drops a DECORATOR changes behaviour while preserving every symbol name, so the symbol
    golden alone cannot see it. This is not hypothetical: slicing ``trainer/utils.py`` by line during
    S4 started the slice at `class DataCollatorForSupervisedDataset` and silently left `@dataclass`
    behind, removing the generated ``__init__``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        names = []
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Attribute):
                names.append(target.attr)
        if names:
            out[node.name] = sorted(names)
    return out


def public_symbols(path: Path) -> list[str]:
    """The module's public surface.

    ``__all__`` when the module declares one — that is Python's own answer, and it is what makes a
    deprecation shim comparable to the module it replaces: the shim *imports* the names rather than
    defining them, so a defs-only walk would report every symbol as dropped.

    Otherwise: top-level defs/classes/assigned names, excluding ``_``-prefixed ones.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, ast.List | ast.Tuple):
                declared = [
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                return sorted(set(declared))

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return sorted(n for n in names if not n.startswith("_"))


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

if __name__ == "__main__":
    payload = collect()
    GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v) for v in payload.values())
    print(f"wrote {GOLDEN.relative_to(REPO_ROOT)}: {len(payload)} modules, {total} public symbols")

    decorators = collect_decorators()
    DECORATOR_GOLDEN.write_text(json.dumps(decorators, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(v) for v in decorators.values())
    print(f"wrote {DECORATOR_GOLDEN.relative_to(REPO_ROOT)}: {n} decorated definitions")
