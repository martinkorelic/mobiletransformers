"""The wheel-installability gate, and the Migration Map's objective progress meter.

`pyproject.toml` packages only `src/mobiletransformers`, so every import from `src/` into a legacy root
is a module that exists in a checkout and is **absent from an installed wheel**. Today the full export
path works from a checkout and fails from a wheel — this test makes that fact countable instead of
folkloric.

`ALLOWED` shrinks as the migration proceeds and must reach **empty**, at which point `uv build` yields a
self-contained wheel. Entries may only be REMOVED.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mobiletransformers"

#: Top-level packages that live at the repo root and are NOT shipped in the wheel.
#:
#: `config` was added 2026-08-14: the root `config.py` deprecation shim was NOT in this set, so the
#: gate did not catch `evaluation/mobile/recommendation_eval.py`'s `from config import AZURE_*` — an
#: installed wheel raised `ModuleNotFoundError: config`. The shim is now deleted, and the name stays
#: listed so a re-introduced root `config.py` cannot reopen the hole.
LEGACY_ROOTS = frozenset(
    {
        "trainer",
        "artifact",
        "inference",
        "tools",
        "peft_models",
        "evaluation",
        "database",
        "research",
        "config",
    }
)

#: The subset scanned for lazy dotted-path STRING literals (see `_dotted_string_references`).
DOTTED_ROOTS = LEGACY_ROOTS - {"config"}

#: `src/`-relative module path -> the legacy roots it may still import. ENTRIES MAY ONLY BE REMOVED.
#:
#: All four original arrows lived in the export pipeline's lazily-imported stage builders and were
#: removed one per step: tools (S1), inference.export_inference_package (S2), trainer (S4),
#: artifact (S5).
#: EMPTY as of Migration Map S5 — the package no longer imports any unpackaged legacy root, so
#: `uv build` yields a wheel from which the full export path runs with no repository checkout.
#: Adding an entry here is a REGRESSION, not a step.
ALLOWED: dict[str, set[str]] = {}


def _imported_roots(path: Path) -> set[str]:
    """Top-level legacy packages imported anywhere in `path`, including function-local imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in LEGACY_ROOTS:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".", 1)[0]
            if root in LEGACY_ROOTS:
                found.add(root)
    return found


def _scan() -> dict[str, set[str]]:
    offenders: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        roots = _imported_roots(path)
        if roots:
            offenders[str(path.relative_to(SRC))] = roots
    return offenders


def _dotted_string_references() -> dict[str, list[str]]:
    """Legacy roots named in DOTTED-PATH STRING literals (the registries' lazy-import convention).

    `import` statements are only half the story: `config/registry/*.py` resolves classes from strings
    like `"peft_models.mars.config.MarsConfig"` via `import_from_path`. An AST import walk cannot see
    those, so a move that forgets one leaves a runtime-only `ModuleNotFoundError` that no static gate
    catches — exactly what happened to the MARS config path during S3.
    """
    import re

    # Non-capturing group: findall must return the FULL dotted path, not just the root.
    # Scans DOTTED_ROOTS, not LEGACY_ROOTS: `config` is a legacy *import* root but a terrible
    # dotted-literal root — `"config.yml"` / `"config.json"` are filenames, not module paths.
    pattern = re.compile(rf'"((?:{"|".join(sorted(DOTTED_ROOTS))})\.[\w.]+)"')
    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        if hits:
            offenders[str(path.relative_to(SRC))] = sorted(set(hits))
    return offenders


#: `src/`-relative module -> dotted paths into a legacy root it may still name. ONLY REMOVALS.
#: The registries resolve these lazily via `import_from_path`; S4 moves `trainer/utils.py`.
#: **EMPTY.** The only entry was the architecture registry's lazy `inference.builder` paths, held open
#: for "the deferred move of inference/builder.py (unimportable under every declared profile)". S6 moved
#: it: the registry now names `mobiletransformers.inference.builder`, which is inside the wheel. There
#: is no longer any dotted string in `src/` that resolves into an unpackaged root — the failure mode
#: this guard exists for (works from a checkout, NameError/ImportError from an installed wheel) is gone.
ALLOWED_DOTTED: dict[str, set[str]] = {}


def test_no_lazy_dotted_paths_into_legacy_roots() -> None:
    offenders = {
        module: sorted(set(hits) - ALLOWED_DOTTED.get(module, set()))
        for module, hits in _dotted_string_references().items()
        if set(hits) - ALLOWED_DOTTED.get(module, set())
    }
    assert not offenders, (
        "dotted-path string(s) resolving into an unpackaged legacy root — these fail at RUNTIME from "
        f"an installed wheel and no import guard sees them:\n{offenders}"
    )


def test_dotted_allowlist_shrinks() -> None:
    actual = _dotted_string_references()
    for module, allowed in ALLOWED_DOTTED.items():
        resolved = sorted(allowed - set(actual.get(module, [])))
        assert not resolved, f"{module} no longer names {resolved} — remove them from ALLOWED_DOTTED"


def test_no_new_src_to_legacy_imports() -> None:
    actual = _scan()
    unlisted = {
        module: sorted(roots - ALLOWED.get(module, set()))
        for module, roots in actual.items()
        if roots - ALLOWED.get(module, set())
    }
    assert not unlisted, (
        "new import(s) from the package into an UNPACKAGED legacy root — these work from a checkout "
        f"and fail from an installed wheel:\n{unlisted}"
    )


def test_allowlist_shrinks_and_never_goes_stale() -> None:
    """A resolved arrow must be removed from ALLOWED, so the meter cannot silently stall."""
    actual = _scan()
    for module, allowed in ALLOWED.items():
        remaining = actual.get(module, set())
        resolved = sorted(allowed - remaining)
        assert not resolved, (
            f"{module} no longer imports {resolved} — remove them from ALLOWED so the "
            "migration's progress stays honest"
        )


def test_wheel_is_self_contained_once_the_allowlist_empties() -> None:
    """The migration's finish line, stated as an assertion.

    While ALLOWED is non-empty this records what is left. When it empties, the assertion below becomes
    the standing guarantee that `mobiletransformers export` runs from a wheel with no checkout.
    """
    assert not ALLOWED, "ALLOWED is non-empty — the migration regressed"
    assert not _scan(), "ALLOWED is empty but src/ still imports a legacy root"
