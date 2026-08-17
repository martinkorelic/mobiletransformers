"""Pure AST helpers for the symbol/decorator goldens.

Lifted out of `gen_legacy_symbol_golden.py` on 2026-08-14. That generator can no longer run — the seven
legacy roots it walks were deleted in S9 — but these two functions are still imported by
`tests/unit/test_symbol_golden.py` and `tests/unit/test_import_weight.py`, which compare the frozen
goldens against the modules' CURRENT homes. Keeping live helpers inside a dead generator was the kind
of "load-bearing by accident" arrangement this repo has been paying for elsewhere.

Nothing here touches the legacy roots; both functions take a path and parse it.
"""

from __future__ import annotations

import ast
from pathlib import Path


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

    Otherwise: top-level defs/classes/assigned names **and module-level import bindings**, excluding
    ``_``-prefixed ones. Imports count because they genuinely are attributes of the module — a
    de-duplication that replaces a private copy of a helper with ``from ...utils.yaml import
    load_config_from_file`` keeps the name importable from exactly where it was, and a defs-only walk
    would report that as a lost public symbol (it did, for five modules, on 2026-08-14).
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
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".", 1)[0])
    return sorted(n for n in names if not n.startswith("_"))
