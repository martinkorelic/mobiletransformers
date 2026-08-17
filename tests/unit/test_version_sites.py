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
CONSUMER_PROPERTIES = REPO_ROOT / "examples/consumer-app/gradle.properties"
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


@pytest.mark.skipif(not CONSUMER_PROPERTIES.is_file(), reason="consumer example not present")
def test_the_consumer_example_resolves_a_version_that_exists() -> None:
    """`examples/consumer-app` is the proof that the published AAR is consumable from outside this
    repo — so it has to ask for the version this repo actually publishes.

    It was pinned at `0.1.0` against a `0.2.0` project, one minor behind, under a comment in that
    same file promising it was kept in step. The guard stopped one directory short of the file that
    drifted, which is the whole reason this exists: a version site nobody checks is a version site
    that rots, and this one rots in the example a newcomer is most likely to copy.
    """
    props = _properties(CONSUMER_PROPERTIES)
    assert "mobiletransformersVersion" in props, (
        f"{CONSUMER_PROPERTIES} declares no `mobiletransformersVersion`"
    )
    assert props["mobiletransformersVersion"] == declared_version(), (
        "the consumer example resolves an SDK version this repository does not publish"
    )


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


def test_manifest_version_site_is_derived_not_hardcoded() -> None:
    """The 5th version site: the manifest's `mobiletransformersVersion`.

    `export/pipeline.py` derives it from `importlib.metadata`, so a real export is always correct.
    The FIXTURES were the rot risk: three of them hardcoded "0.1.0", so a version bump would have left
    the committed package, its generator and the pipeline test disagreeing with `pyproject.toml`
    while every other version test stayed green.
    """
    import json

    expected = declared_version()
    fixture = REPO_ROOT / "tests/fixtures/tiny_package/mobiletransformers_manifest.json"
    manifest = json.loads(fixture.read_text(encoding="utf-8"))
    assert manifest["mobiletransformersVersion"] == expected, (
        f"{fixture.relative_to(REPO_ROOT)} says {manifest['mobiletransformersVersion']!r}, "
        f"pyproject says {expected!r} — regenerate it with tests/fixtures/make_tiny_package.py"
    )

    for path in (
        REPO_ROOT / "tests/fixtures/make_tiny_package.py",
        REPO_ROOT / "tests/export/test_pipeline.py",
    ):
        text = path.read_text(encoding="utf-8")
        if "mobiletransformersVersion" not in text:
            continue
        for line in text.splitlines():
            if "mobiletransformersVersion" in line and '"' in line.split(":", 1)[-1]:
                literal = re.search(r'"mobiletransformersVersion":\s*"([^"]+)"', line)
                if literal:
                    assert literal.group(1) == expected, (
                        f"{path.relative_to(REPO_ROOT)} hardcodes "
                        f"{literal.group(1)!r}; pyproject says {expected!r}"
                    )
