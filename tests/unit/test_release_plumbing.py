"""#28/#30/#32: release plumbing that should be checkable without a release.

`make help` completeness and `clean-generated` non-destructiveness are #28 DoD items that were never
written; the publication coordinates are #30's contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
SDK_BUILD = REPO_ROOT / "android/MobileTransformers/MobileTransformers/build.gradle.kts"


def _targets() -> set[str]:
    """Rule targets declared in the Makefile (excluding pattern rules and variables)."""
    return {
        m.group(1)
        for m in re.finditer(r"^([a-zA-Z][\w-]*):", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    }


def _documented() -> set[str]:
    return {
        m.group(1)
        for m in re.finditer(
            r"^([a-zA-Z][\w-]*):.*?##\s+\S", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE
        )
    }


def test_every_make_target_is_self_documented() -> None:
    """`make help` greps for `## ` docstrings; an undocumented target is invisible to users."""
    undocumented = sorted(_targets() - _documented())
    assert not undocumented, f"targets missing a `## ` doc comment: {undocumented}"


def test_phony_declares_every_target() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    phony_block = text.split(".PHONY:", 1)[1].split("\n\n", 1)[0]
    declared = set(phony_block.replace("\\", " ").split())
    missing = sorted(_targets() - declared - {"help"})
    assert not missing, f"targets missing from .PHONY: {missing}"


def test_clean_generated_is_not_destructive() -> None:
    """`clean-generated` must only remove BUILD output — never sources, tests or vendored deps."""
    text = MAKEFILE.read_text(encoding="utf-8")
    recipe = text.split("clean-generated:", 1)[1].split("\n\n", 1)[0]
    forbidden = ("src/", "tests/", "android/", "docs/", "agent_docs/", "jniLibs", "third_party")
    for token in forbidden:
        assert f"rm -rf {token}" not in recipe, f"clean-generated would delete {token}"
        assert f"rm -r {token}" not in recipe, f"clean-generated would delete {token}"
    assert "build/" in recipe, "clean-generated should remove build/"


@pytest.mark.skipif(not SDK_BUILD.is_file(), reason="Android tree not present")
def test_publication_coordinates_are_the_agreed_ones() -> None:
    text = SDK_BUILD.read_text(encoding="utf-8")
    assert "`maven-publish`" in text, "the SDK module does not apply maven-publish"
    assert 'artifactId = "mobiletransformers-android"' in text
    assert "withSourcesJar()" in text, "#30 requires a sources jar"


@pytest.mark.skipif(not SDK_BUILD.is_file(), reason="Android tree not present")
def test_pom_license_matches_the_repository_license() -> None:
    """A consumer resolving the POM relies on it; it must not advertise a licence we do not use."""
    pom = SDK_BUILD.read_text(encoding="utf-8")
    license_md = (REPO_ROOT / "LICENSE.md").read_text(encoding="utf-8")
    if "Attribution-NonCommercial" in license_md:
        assert "NonCommercial" in pom, "LICENSE.md is CC-BY-NC but the POM claims otherwise"
    elif "Apache License" in license_md:
        assert "Apache-2.0" in pom or "Apache License" in pom


def test_third_party_notices_exist_and_cover_the_vendored_natives() -> None:
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for component in ("ONNX Runtime", "tokenizers", "ObjectBox", "nlohmann/json"):
        assert component in notices, f"THIRD_PARTY_NOTICES.md does not mention {component}"


def test_changelog_records_the_required_non_goals() -> None:
    """#32 requires the non-goals to be explicit, so they are not re-litigated per release."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    non_goals = changelog.split("### Non-goals", 1)[1]
    for phrase in ("GPU/NPU", "Multimodal"):
        assert phrase in non_goals, f"CHANGELOG non-goals omit {phrase}"
