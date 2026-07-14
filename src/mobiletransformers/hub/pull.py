"""Hub pull + install (#21, Python-first) — download a package and materialize it into the cache shape.

``pull_package`` fetches the manifest first, selects a variant, downloads only the files its
``downloadPlan`` names (sha256-verified), and ``install_package`` materializes the selected variant into
the exact ``<cacheDir>/<sanitizedRepoId>/{train,inference,embedding,tokenizer}`` layout `LLMRepository`
probes (the Python mirror of #13's Kotlin ``ModelPackageInstaller``). The Android downloader is a
separate device leg (deferred).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.exceptions import HubError
from mobiletransformers.hub.package_format import MANIFEST_FILENAME, VARIANT_SUBDIRS, sanitize_repo_id
from mobiletransformers.hub.variant_select import (
    Constraints,
    default_desktop_constraints,
    select_variant,
)

Downloader = Callable[..., str]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_downloader(**kwargs: object) -> str:
    from huggingface_hub import snapshot_download  # core dep, imported lazily to keep import light

    return snapshot_download(**kwargs)  # type: ignore[arg-type]


def _allow_patterns(manifest: MobileTransformersManifest, variant_id: str, features: set[str]) -> list[str]:
    plan = manifest.to_dict().get("downloadPlan", {}).get(variant_id, {})
    groups = set(features) | {"core", "checksums"}
    patterns: list[str] = []
    for group in sorted(groups):
        patterns.extend(plan.get(group, []))
    return patterns


def pull_package(
    repo_id: str,
    *,
    revision: str = "main",
    variant: str | None = None,
    features: tuple[str, ...] = ("inference",),
    token: str | None = None,
    dest: str | Path | None = None,
    constraints: Constraints | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Download the selected variant's files for ``features`` into a staging dir; verify sha256.

    ``downloader`` is injectable (defaults to ``huggingface_hub.snapshot_download``) so tests/offline
    runs can serve the fixture. Returns the staging directory (a full package subtree).
    """
    downloader = downloader or _default_downloader
    staging = Path(dest) if dest is not None else Path(f"./.mt-pull/{sanitize_repo_id(repo_id)}")
    staging.mkdir(parents=True, exist_ok=True)

    # 1. Manifest first.
    downloader(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(staging),
        allow_patterns=[MANIFEST_FILENAME],
    )
    manifest_path = staging / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise HubError(f"manifest {MANIFEST_FILENAME} not present after pull of {repo_id!r}")
    manifest = MobileTransformersManifest.load(manifest_path)

    # 2. Select variant.
    feature_set = set(features) | {"core", "inference"}
    variant_id = variant or select_variant(manifest, constraints or default_desktop_constraints())

    # 3+4. Download the file set for the requested features.
    patterns = _allow_patterns(manifest, variant_id, feature_set)
    downloader(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(staging),
        allow_patterns=patterns,
    )

    # 5. Verify sha256 of every downloaded file we have a digest for.
    sha_map: dict[str, str] = manifest.to_dict().get("sha256", {})
    for rel, expected in sha_map.items():
        f = staging / rel
        if f.is_file() and _sha256(f) != expected:
            raise HubError(f"sha256 mismatch for {rel} in {repo_id!r} (corrupt download)")
    return staging


def install_package(
    staging_dir: str | Path, cache_root: str | Path, repo_id: str, *, variant: str | None = None
) -> Path:
    """Materialize a staged package into ``<cache_root>/<sanitizedRepoId>/`` (the LLMRepository shape).

    Atomic: builds under ``<cache_root>/.partial/<sanitized>`` then ``os.replace``. Flattens
    ``shared/tokenizer`` -> ``tokenizer/`` and ``shared/chat_template.jinja`` alongside it. Validates the
    manifest before publishing.
    """
    staging_dir = Path(staging_dir)
    cache_root = Path(cache_root)
    manifest = MobileTransformersManifest.load(staging_dir / MANIFEST_FILENAME)
    variant_id = variant or manifest.default_variant

    # A feature-partial pull only downloaded ONE variant's files, so the whole-package validate()
    # (which checks every variant's handoff) is inappropriate here — check just the selected variant.
    selected = next((v for v in manifest.variants if v.get("id") == variant_id), None)
    if selected is None:
        raise HubError(f"variant {variant_id!r} not in manifest for install")
    handoff_rel = selected.get("weightHandoff")
    if handoff_rel and not (staging_dir / handoff_rel).is_file():
        raise HubError(f"selected variant {variant_id!r} weightHandoff missing in staging: {handoff_rel}")

    sanitized = sanitize_repo_id(repo_id)
    partial = cache_root / ".partial" / sanitized
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)

    variant_root = staging_dir / "variants" / variant_id
    for sub in VARIANT_SUBDIRS:
        src = variant_root / sub
        if src.is_dir():
            shutil.copytree(src, partial / sub, dirs_exist_ok=True)
    # Flatten shared/ tokenizer + chat template into the conventional cache layout.
    shared_tok = staging_dir / "shared" / "tokenizer"
    if shared_tok.is_dir():
        shutil.copytree(shared_tok, partial / "tokenizer", dirs_exist_ok=True)
    chat_template = staging_dir / "shared" / "chat_template.jinja"
    if chat_template.is_file():
        shutil.copy2(chat_template, partial / "tokenizer" / "chat_template.jinja")
    # Manifest + per-variant checksums at the cache root.
    shutil.copy2(staging_dir / MANIFEST_FILENAME, partial / MANIFEST_FILENAME)
    checksums = variant_root / "checksums.json"
    if checksums.is_file():
        shutil.copy2(checksums, partial / "checksums.json")

    target = cache_root / sanitized
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)
    return target


__all__ = ["pull_package", "install_package"]
