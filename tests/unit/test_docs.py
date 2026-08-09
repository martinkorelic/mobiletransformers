"""Docs guards: relative links resolve, and pages don't drift out of sync with the code.

`docs/RAG.md` claimed #26/#27 were "not yet implemented" long after they landed, and
`docs/PUBLIC_API.md`'s CLI table omitted a registered subcommand. Both are the kind of drift a test
catches for free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
MARKDOWN = sorted(DOCS.glob("*.md")) + [REPO_ROOT / "README.md", REPO_ROOT / "CHANGELOG.md"]

_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_resolve(page: Path) -> None:
    broken = []
    for match in _LINK.finditer(page.read_text(encoding="utf-8")):
        target = match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.partition("#")[0]
        if path and not (page.parent / path).resolve().exists():
            broken.append(f"[{match.group(1)}]({target})")
    assert not broken, f"{page.name}: broken relative link(s): {broken}"


def test_cli_table_lists_every_registered_subcommand() -> None:
    """`federated` was registered in the parser but missing from the documented table."""
    from mobiletransformers.cli.main import build_parser

    parser = build_parser()
    registered = {
        name
        for action in parser._subparsers._group_actions  # noqa: SLF001 - argparse has no public API
        for name in action.choices
    }
    documented = (DOCS / "PUBLIC_API.md").read_text(encoding="utf-8")
    missing = sorted(cmd for cmd in registered if f"`{cmd}`" not in documented)
    assert not missing, f"docs/PUBLIC_API.md does not document: {missing}"


#: Kotlin sources for the public facade. The internal packages are deliberately excluded — the point is
#: to prove the *documented* surface exists, not to inventory everything under the namespace.
_KOTLIN_FACADE_ROOT = (
    REPO_ROOT
    / "android"
    / "MobileTransformers"
    / "MobileTransformers"
    / "src"
    / "main"
    / "java"
    / "com"
    / "martinkorelic"
    / "mobiletransformers"
)

_KOTLIN_DECL = re.compile(
    r"\b(?:data class|sealed class|enum class|value class|abstract class|open class|class|interface|"
    r"fun interface|object)\s+([A-Z][A-Za-z0-9_]*)"
)


def test_documented_kotlin_facade_symbols_exist() -> None:
    """Every type named in the Kotlin facade table must be a real declaration.

    The Python `__all__` surface is guarded by `public_api.txt` and the CLI table by the test above;
    the Kotlin half of the same page had no guard at all, and its table sat marked "pending #17/#19"
    long after both landed. This closes that asymmetry: the doc can now only name types that exist.

    Method names inside the table cells are not checked here — `FacadeDelegationTest` (Android JVM)
    already pins the model handle's methods by calling them.
    """
    if not _KOTLIN_FACADE_ROOT.is_dir():
        pytest.skip("Android sources not present in this checkout")

    declared: set[str] = set()
    for source in _KOTLIN_FACADE_ROOT.rglob("*.kt"):
        declared.update(_KOTLIN_DECL.findall(source.read_text(encoding="utf-8")))

    page = (DOCS / "PUBLIC_API.md").read_text(encoding="utf-8")
    section = page.partition("## Kotlin facade")[2].partition("\n## ")[0]
    assert section.strip(), "docs/PUBLIC_API.md has no Kotlin facade section"

    # Types are the backticked identifiers in UpperCamelCase. SCREAMING_CASE tokens are enum *values*
    # (`NATIVE`, `GENAI`), which `make parity` already checks against the Python enums wire-value by
    # wire-value — a stronger check than existence, so re-testing them here would add nothing.
    named = {
        token
        for token in re.findall(r"`([^`]+)`", section)
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", token) and not token.isupper()
    }
    assert named, "the Kotlin facade table names no types — the guard would pass vacuously"

    missing = sorted(name for name in named if name not in declared)
    assert not missing, (
        f"docs/PUBLIC_API.md names Kotlin types that do not exist: {missing}. "
        "Either the facade renamed them or the doc drifted."
    )


def test_every_doc_page_is_reachable_from_the_readme() -> None:
    """A page nobody links to is a page nobody reads."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # Generated//checklist pages are referenced from their owning docs, not the README index.
    exempt = {"RELEASE_CHECKLIST.md", "mobile_evaluation.md", "COMPATIBILITY_MATRIX.md"}
    unlinked = sorted(
        p.name for p in DOCS.glob("*.md") if p.name not in exempt and f"docs/{p.name}" not in readme
    )
    assert not unlinked, f"not linked from README.md: {unlinked}"
