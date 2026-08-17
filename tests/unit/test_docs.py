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

#: EVERY tracked markdown file.
#:
#: The link check used to cover `docs/` + README + CHANGELOG only, and was widened in 2026-08-14 to
#: every tracked page after the review found the worst reference rot in `agent_docs/`. That directory
#: was untracked on 2026-08-17, so it now falls out of this scan by the same `git ls-files` rule that
#: excludes build output — deliberately, and worth stating: the shipped docs are what this gate is
#: for, and the planning material is no longer part of the repo's public surface.
#:
#: Deliberately relative links only. Checking external URLs needs the network, goes red for reasons
#: outside this repo, and would make the one gate that always runs the flakiest one.
#:
#: Enumerated with `git ls-files` rather than `rglob` + an exclude list. An `rglob` sweep reported 8
#: "broken" pages that were all vendored nlohmann/json docs under `.cxx/_deps/` — build output whose
#: links are not ours to fix — and every new vendored dependency would have needed another exclusion.
#: Tracked-files-only is self-maintaining: anything gitignored is out by construction.


def _tracked_markdown() -> list[Path]:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    return sorted(REPO_ROOT / line for line in result.stdout.split() if (REPO_ROOT / line).is_file())


_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _broken_links(page: Path) -> list[str]:
    broken = []
    for match in _LINK.finditer(page.read_text(encoding="utf-8", errors="replace")):
        target = match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.partition("#")[0]
        if path and not (page.parent / path).resolve().exists():
            broken.append(f"[{match.group(1)}]({target})")
    return broken


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_resolve(page: Path) -> None:
    assert not _broken_links(page), f"{page.name}: broken relative link(s): {_broken_links(page)}"


def test_no_tracked_markdown_links_to_a_missing_file() -> None:
    """The repo-wide sweep, reported in one place so a rename shows every page it broke at once."""
    pages = _tracked_markdown()
    if not pages:
        pytest.skip("git not available or not a checkout")
    # Vacuity floor: this sweep is only meaningful if it is actually seeing the repo, and a `git
    # ls-files` that returns nothing useful would otherwise pass silently. Lowered from 50 to 20 on
    # 2026-08-17 when `agent_docs/` (52 tracked pages) was untracked — re-pointed rather than deleted,
    # because the assertion still does its job at the new size (30 pages today).
    assert len(pages) > 20, f"only {len(pages)} markdown files found — the sweep is not seeing the repo"

    broken = {str(page.relative_to(REPO_ROOT)): links for page in pages if (links := _broken_links(page))}
    assert not broken, (
        "markdown link(s) pointing at files that do not exist — a renamed or deleted file leaves "
        f"these behind silently:\n{broken}"
    )


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

#: UpperCamelCase enum ENTRIES (`ModelFeature.Inference`), which are not declarations but which the
#: docs legitimately name. The existing SCREAMING_CASE carve-out does not reach them.
_KOTLIN_ENUM_BODY = re.compile(r"\benum class\s+[A-Z][A-Za-z0-9_]*[^{]*\{(.*?)(?:\n\s*;|\n\})", re.DOTALL)
_KOTLIN_ENUM_ENTRY = re.compile(r"^\s*([A-Z][A-Za-z0-9_]*)\s*(?:\(|,|$)", re.MULTILINE)

#: Kotlin stdlib types a doc table names as a *column value* (`Boolean`, `Long`), not as a symbol this
#: repo declares. Listing them beats loosening the identifier pattern, which would stop catching a
#: genuinely renamed type.
_KOTLIN_BUILTINS = frozenset(
    {"Boolean", "Long", "Int", "String", "Float", "Double", "Set", "List", "Map", "Unit", "Any"}
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

    declared: set[str] = set(_KOTLIN_BUILTINS)
    for source in _KOTLIN_FACADE_ROOT.rglob("*.kt"):
        text = source.read_text(encoding="utf-8")
        declared.update(_KOTLIN_DECL.findall(text))
        for body in _KOTLIN_ENUM_BODY.findall(text):
            declared.update(_KOTLIN_ENUM_ENTRY.findall(body))

    page = (DOCS / "PUBLIC_API.md").read_text(encoding="utf-8")
    section = page.partition("## Kotlin facade")[2].partition("\n## ")[0]
    assert section.strip(), "docs/PUBLIC_API.md has no Kotlin facade section"

    # ANDROID_SDK.md's capability/result/tool-call tables name the same types and were unguarded, so
    # the 2026-08-17 doc sweep could have invented a property name and nothing would have noticed.
    # Only the TABLE rows are scanned: the prose around them legitimately names Java/Android types
    # (`WorkManager`, `Play`) that are not declarations in this repo.
    sdk = (DOCS / "ANDROID_SDK.md").read_text(encoding="utf-8")
    section += "\n" + "\n".join(line for line in sdk.splitlines() if line.startswith("| `"))

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


def test_every_doc_page_is_reachable() -> None:
    """A page nobody links to is a page nobody reads.

    There are now **two** front doors, and a page needs only one of them:

    - `README.md`, for someone reading the repository on GitHub.
    - `mkdocs.yml`'s `nav`, for someone reading the published site.

    Checking the README alone was right when it was the only index. It stopped being right when the
    documentation site moved into this repository: `index.md` is the site's home page and would be a
    strange thing to link from the README, while a page missing from the **nav** is invisible on the
    site no matter how well the README links it. Requiring either catches both kinds of orphan.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    # Generated/checklist pages are referenced from their owning docs, not the README index.
    exempt = {"RELEASE_CHECKLIST.md", "mobile_evaluation.md", "COMPATIBILITY_MATRIX.md"}
    unlinked = sorted(
        p.name
        for p in DOCS.glob("*.md")
        if p.name not in exempt and f"docs/{p.name}" not in readme and f": {p.name}" not in mkdocs
    )
    assert not unlinked, (
        f"unreachable from both README.md and the mkdocs nav: {unlinked}. Add it to the README's "
        "documentation table, or to `nav` in mkdocs.yml, or both."
    )


def test_the_site_nav_names_only_pages_that_exist() -> None:
    """The other half: `nav` must not point at a page that is not there.

    `mkdocs build --strict` catches this too, but only where mkdocs is installed — this keeps it in
    the gate that always runs, so a renamed page fails in `make check` rather than in CI.
    """
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    referenced = set(re.findall(r"^\s+\S.*?:\s+(\S+\.md)\s*$", mkdocs, re.MULTILINE))
    assert referenced, "mkdocs.yml nav names no pages — this guard would pass vacuously"
    missing = sorted(name for name in referenced if not (DOCS / name).is_file())
    assert not missing, f"mkdocs.yml nav names pages that do not exist: {missing}"


#: Types a Kotlin snippet may legitimately name without the SDK declaring them: Kotlin/Java stdlib,
#: Android framework, coroutines, and the generated `BuildConfig`. Kept explicit rather than "skip
#: anything we cannot find", which would make the guard pass on a typo.
_COOKBOOK_EXTERNAL_TYPES = frozenset(
    {
        "Boolean",
        "Byte",
        "ByteArray",
        "Double",
        "Float",
        "Int",
        "List",
        "Long",
        "Map",
        "Set",
        "String",
        "Unit",
        "Array",
        "Pair",
        "Context",
        "Intent",
        "Log",
        "File",
        "System",
        "BuildConfig",
        "Exception",
        "Throwable",
    }
)

#: `object`/`companion` members and enum values are not declarations `_KOTLIN_DECL` can see; they are
#: covered by `make parity` (enum wire values) or by the JVM suites that call them.
# Enum constants and sealed-class members: the scan cannot tell `MemoryConfigId.HIGH_PERF` from a
# type name, and the enum itself is already checked.
_COOKBOOK_IGNORED_TOKENS = frozenset(
    {
        "GREEDY",
        "NATIVE",
        "GENAI",
        "HIGH_PERF",
        "DEFAULT",
        "Accepted",
        "Rejected",
        "Inference",
        "Training",
    }
)


def test_cookbook_snippets_only_name_kotlin_types_that_exist() -> None:
    """`docs/COOKBOOK.md`'s snippets must not name a Kotlin type the facade does not declare.

    The cookbook's whole value is that it can be pasted into a real app. A snippet naming a renamed or
    deleted type is worse than no snippet, and this is exactly how `docs/RAG.md` rotted before the
    Kotlin-facade guard above was added — nothing checked the code, only the prose links.

    Scans ```kotlin fences (not the whole page), so prose may still discuss a type by name in the past
    tense while the code stays honest.
    """
    if not _KOTLIN_FACADE_ROOT.is_dir():
        pytest.skip("Android sources not present in this checkout")

    page = DOCS / "COOKBOOK.md"
    assert page.is_file(), "docs/COOKBOOK.md is missing"

    declared: set[str] = set()
    for source in _KOTLIN_FACADE_ROOT.rglob("*.kt"):
        declared.update(_KOTLIN_DECL.findall(source.read_text(encoding="utf-8")))
    assert declared, "no Kotlin declarations found — the guard would pass vacuously"

    blocks = re.findall(r"```kotlin\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "COOKBOOK.md has no kotlin snippets — the guard would pass vacuously"

    named: set[str] = set()
    for block in blocks:
        # Strip line comments and string literals: a repo id, an intent name or a prose comment is
        # not a type reference, and treating one as such would make the guard fail on correct code.
        code = re.sub(r"//.*", "", block)
        code = re.sub(r'"[^"]*"', '""', code)
        named.update(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", code))

    unknown = sorted(
        name
        for name in named
        if name not in declared
        and name not in _COOKBOOK_EXTERNAL_TYPES
        and name not in _COOKBOOK_IGNORED_TOKENS
    )
    assert not unknown, (
        f"docs/COOKBOOK.md names Kotlin types that do not exist: {unknown}. "
        "Either the facade renamed them or the cookbook drifted — fix the snippet, because it is "
        "meant to be copy-pasteable."
    )
