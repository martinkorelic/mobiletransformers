"""#21 pull + install smokes — offline (injected downloader over the tiny_package fixture)."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path

import pytest

from mobiletransformers.exceptions import HubError
from mobiletransformers.hub.pull import install_package, pull_package

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package"


def _fake_downloader(remote: Path):
    """A snapshot_download stand-in: copy files under `remote` matching allow_patterns into local_dir."""

    def _dl(*, repo_id, revision, token, local_dir, allow_patterns, **_):
        dst = Path(local_dir)
        rels = [p.relative_to(remote).as_posix() for p in remote.rglob("*") if p.is_file()]
        for rel in rels:
            if _matches(rel, allow_patterns):
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote / rel, out)
        return str(dst)

    return _dl


def _matches(rel: str, patterns) -> bool:
    for pat in patterns:
        if pat.endswith("/**"):
            if rel.startswith(pat[:-2]) or rel.startswith(pat[:-3] + "/"):
                return True
        elif rel == pat or fnmatch.fnmatch(rel, pat):
            return True
    return False


def test_pull_inference_only_downloads_expected_and_verifies(tmp_path):
    staging = pull_package(
        "org/tiny-model",
        features=("inference",),
        dest=tmp_path / "stg",
        downloader=_fake_downloader(FIXTURE),
    )
    # core + inference + checksums present; train/ and embedding/ absent (not requested).
    assert (staging / "mobiletransformers_manifest.json").is_file()
    assert (staging / "shared/tokenizer/tokenizer.json").is_file()
    assert (staging / "variants/cpu-int4/inference/model.onnx").is_file()
    assert not (staging / "variants/cpu-int4/train").exists()
    assert not (staging / "variants/cpu-int4/embedding").exists()


def test_pull_detects_sha256_mismatch(tmp_path):
    remote = tmp_path / "remote"
    shutil.copytree(FIXTURE, remote)
    # Corrupt one inference file's bytes (its manifest sha256 no longer matches).
    (remote / "variants/cpu-int4/inference/model.onnx").write_text("CORRUPTED")
    with pytest.raises(HubError, match="sha256 mismatch.*model.onnx"):
        pull_package(
            "org/tiny-model",
            features=("inference",),
            dest=tmp_path / "stg",
            downloader=_fake_downloader(remote),
        )


def test_install_materializes_cache_layout(tmp_path):
    staging = pull_package(
        "org/tiny-model",
        features=("inference", "train", "rag"),
        dest=tmp_path / "stg",
        downloader=_fake_downloader(FIXTURE),
    )
    cache_root = tmp_path / "cache"
    target = install_package(staging, cache_root, "org/Tiny-Model", variant="cpu-int4")
    assert target.name == "org__Tiny-Model"
    # LLMRepository-shaped layout; tokenizer flattened out of shared/.
    assert (target / "inference/model.onnx").is_file()
    assert (target / "train/training_config.json").is_file()
    assert (target / "embedding/rag_config.json").is_file()
    assert (target / "tokenizer/tokenizer.json").is_file()
    assert (target / "mobiletransformers_manifest.json").is_file()
    # atomic: no leftover staging.
    assert not (cache_root / ".partial" / "org__Tiny-Model").exists()
