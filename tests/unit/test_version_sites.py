"""#32: the version appears in several places and they must never disagree.

`pyproject.toml` is the single write-site. Everything else either derives from it at runtime or is a
declaration that this test pins to it. Previously `__version__` was hardcoded (a second write-site) and
`CITATION.cff` advertised a `1.0.0 / 2025-10-18` release that did not exist — against a `0.1.0` package
with zero git tags.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GRADLE_PROPERTIES = REPO_ROOT / "android/MobileTransformers/gradle.properties"
CITATION = REPO_ROOT / "CITATION.cff"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].+)?$")


def declared_version() -> str:
    """The one authoritative version: ``pyproject.toml``'s ``[project] version``.

    Parsed with a regex rather than ``tomllib`` because the core gate runs on Python 3.10, where
    ``tomllib`` does not exist — and this test must not need a dependency to check a version.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert match, "pyproject.toml [project] declares no version"
    return match.group(1)


def _properties(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_declared_version_is_semver() -> None:
    assert _SEMVER.match(declared_version()), declared_version()


def test_runtime_version_derives_from_the_package_metadata() -> None:
    """`__version__` must be read, not restated — a literal here is a second write-site."""
    import mobiletransformers

    assert mobiletransformers.__version__ == declared_version()

    source = (REPO_ROOT / "src/mobiletransformers/__init__.py").read_text(encoding="utf-8")
    assert 'version("mobiletransformers")' in source, "__version__ should come from importlib.metadata"
    assert f'__version__ = "{declared_version()}"' not in source, "__version__ is hardcoded again"


@pytest.mark.skipif(not GRADLE_PROPERTIES.is_file(), reason="Android tree not present")
def test_gradle_version_matches() -> None:
    props = _properties(GRADLE_PROPERTIES)
    assert "version" in props, f"{GRADLE_PROPERTIES.name} declares no `version`"
    assert props["version"] == declared_version()


@pytest.mark.skipif(not GRADLE_PROPERTIES.is_file(), reason="Android tree not present")
def test_gradle_group_is_the_publication_coordinate() -> None:
    """`00_code_plans/04` said `com.martinkorelic`; `05_code_plans/03` (the publication plan) wins."""
    assert _properties(GRADLE_PROPERTIES).get("group") == "com.martinkorelic.mobiletransformers"


def test_citation_version_matches() -> None:
    text = CITATION.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff declares no version"
    assert match.group(1) == declared_version()


def test_citation_release_date_is_not_in_the_future() -> None:
    """A citation must not advertise a release that has not happened."""
    from datetime import date

    text = CITATION.read_text(encoding="utf-8")
    match = re.search(r"^date-released:\s*(\d{4})-(\d{2})-(\d{2})\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff declares no date-released"
    released = date(*(int(g) for g in match.groups()))
    assert released <= date.today(), f"date-released {released} is in the future"
