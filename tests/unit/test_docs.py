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


def test_every_doc_page_is_reachable_from_the_readme() -> None:
    """A page nobody links to is a page nobody reads."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # Generated//checklist pages are referenced from their owning docs, not the README index.
    exempt = {"RELEASE_CHECKLIST.md", "mobile_evaluation.md", "COMPATIBILITY_MATRIX.md"}
    unlinked = sorted(
        p.name for p in DOCS.glob("*.md") if p.name not in exempt and f"docs/{p.name}" not in readme
    )
    assert not unlinked, f"not linked from README.md: {unlinked}"
