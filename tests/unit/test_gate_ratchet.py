"""Every migration ratchet must shrink, and none may go stale.

The migration relies on several allow-lists that trade "gated" for "tracked":

* `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` — lint codes tolerated in freshly-moved code
* `pyproject.toml` `[[tool.mypy.overrides]]` marked `MIGRATION RATCHET` — modules not yet typed
* `tests/unit/test_guards.py::DISPATCH_ALLOWLIST` — remaining #6 dispatch literals
* `tests/unit/test_no_src_to_legacy_imports.py::ALLOWED` — remaining `src/` → legacy arrows

Each of those has its own shrink check. This module guards the property they share: an entry naming a
file that no longer exists, or a code that no longer fires, is a lie that makes the debt look bigger
than it is — and hides the next real violation behind a blanket exemption.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

_RATCHET_MARKER = "MIGRATION RATCHET"


def _per_file_ignores() -> dict[str, list[str]]:
    """Parse `[tool.ruff.lint.per-file-ignores]` without a TOML parser (core env is Python 3.10)."""
    if "[tool.ruff.lint.per-file-ignores]" not in PYPROJECT:
        return {}
    block = PYPROJECT.split("[tool.ruff.lint.per-file-ignores]", 1)[1].split("\n[", 1)[0]
    entries: dict[str, list[str]] = {}
    for match in re.finditer(r'^"([^"]+)"\s*=\s*\[([^\]]*)\]', block, re.MULTILINE):
        codes = re.findall(r'"([^"]+)"', match.group(2))
        entries[match.group(1)] = codes
    return entries


def _ratchet_override_modules() -> list[str]:
    """Modules in `MIGRATION RATCHET`-marked mypy override blocks."""
    modules: list[str] = []
    for block in PYPROJECT.split("[[tool.mypy.overrides]]")[1:]:
        head = block.split("\n[", 1)[0]
        if _RATCHET_MARKER not in head:
            continue
        modules.extend(re.findall(r'"([^"]+)"', head.split("module", 1)[1].split("\n", 2)[0] or ""))
        for match in re.finditer(r"module\s*=\s*\[([^\]]*)\]", head, re.DOTALL):
            modules.extend(re.findall(r'"([^"]+)"', match.group(1)))
    return sorted(set(modules))


def test_ruff_per_file_ignores_name_existing_files() -> None:
    stale = [path for path in _per_file_ignores() if not (REPO_ROOT / path).exists()]
    assert not stale, f"per-file-ignores name file(s) that no longer exist: {stale}"


def test_ruff_per_file_ignores_still_fire() -> None:
    """An ignore whose codes no longer trigger must be deleted — otherwise the ratchet never tightens."""
    for path, codes in _per_file_ignores().items():
        if not codes or not (REPO_ROOT / path).is_file():
            continue
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "--isolated", "--select", ",".join(codes), path],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode != 0, (
            f"{path}: none of {codes} fire any more — drop the per-file-ignores entry"
        )


def test_mypy_ratchet_overrides_name_existing_modules() -> None:
    for module in _ratchet_override_modules():
        package = module.removesuffix(".*").replace(".", "/")
        candidates = [REPO_ROOT / "src" / f"{package}.py", REPO_ROOT / "src" / package]
        assert any(c.exists() for c in candidates), (
            f"MIGRATION RATCHET override names {module}, which does not exist — drop it"
        )


def test_dispatch_allowlist_is_tracked_not_forgotten() -> None:
    """Cross-check: the #6 allow-list must name real files and carry a stated owner *while it has debt*.

    The owner requirement is conditional on the allow-list being non-empty. It is empty now — #6 closed
    when `inference/builder.py`'s 14-branch ladder became registry rows — and demanding an owner for
    debt that no longer exists would force a fake entry to keep the gate green.
    """
    from tests.unit.test_guards import DISPATCH_ALLOWLIST

    for path in DISPATCH_ALLOWLIST:
        assert (REPO_ROOT / path).is_file(), f"DISPATCH_ALLOWLIST names a missing file: {path}"

    if DISPATCH_ALLOWLIST:
        guards = (REPO_ROOT / "tests" / "unit" / "test_guards.py").read_text(encoding="utf-8")
        assert "Owner:" in guards, "DISPATCH_ALLOWLIST must record who owns the remaining debt"


def test_legacy_import_allowlist_is_tracked() -> None:
    from tests.unit.test_no_src_to_legacy_imports import ALLOWED, SRC

    for module in ALLOWED:
        assert (SRC / module).is_file(), f"ALLOWED names a missing module: {module}"
