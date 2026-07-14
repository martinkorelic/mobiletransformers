"""Hub model-package format (#14) — the on-Hub repo shape + ``mobiletransformers_manifest.json`` schema.

This module OWNS the manifest field list and the package layout. A consumer (Python CLI, Android
downloader, sample app) fetches the small manifest first, then resolves which files to pull from
``downloadPlan`` before touching large ONNX blobs. One shared package, dual engine (native ORT + ONNX
Runtime GenAI both read the same folder); per-tensor external initializers live flat in ``inference/``
beside ``model.onnx``, the frozen quantized base is the immutable ``inference/frozen_base.onnx.data``.

Scope boundary: the manifest *validator*, variant-selection, and cache-install semantics are owned by
#13 (``artifacts/manifest.py``); the ``weight_handoff_map.json`` schema by #8; the per-tensor merge
contract by #9. This module pins the schema + shape and provides ``build_manifest`` (the emit helper the
export CLI #15 calls) and ``sanitize_repo_id`` (byte-identical to the Kotlin cache-bridge sanitizer).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
MIN_READER_VERSION = "1.0"
ARTIFACT_FORMAT_VERSION = 1
MANIFEST_FILENAME = "mobiletransformers_manifest.json"

#: Feature groups a downloader can request; each maps to repo-relative glob patterns in ``downloadPlan``.
FEATURE_GROUPS = ("core", "inference", "train", "rag", "genai", "checksums")

#: Per-variant subdirectories.
VARIANT_SUBDIRS = ("train", "inference", "embedding")

#: Files that must exist regardless of which features are requested (validation floor).
REQUIRED_TOP_LEVEL_FILES = (MANIFEST_FILENAME, "shared/tokenizer/tokenizer.json")

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def sanitize_repo_id(repo_id: str) -> str:
    """Map an HF repo id to a filesystem-safe cache directory name.

    Canonical algorithm (mirrored byte-for-byte in the Kotlin cache bridge, #13):
      1. every ``/`` becomes ``__`` (double underscore);
      2. every remaining char not in ``[A-Za-z0-9._-]`` becomes a single ``_``;
      3. no trimming, no case-folding, no length cap.

    Example: ``mobiletransformers/Qwen2-0.5B`` -> ``mobiletransformers__Qwen2-0.5B``.
    """
    out: list[str] = []
    for ch in repo_id:
        if ch == "/":
            out.append("__")
        elif ch in _SAFE_CHARS:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _download_plan_for(variant_id: str, features: tuple[str, ...]) -> dict[str, list[str]]:
    """Build the per-variant ``{group: [glob patterns]}`` map. Groups whose feature the variant does
    not declare are present but empty (a downloader still keys off them deterministically)."""
    has = set(features)
    plan: dict[str, list[str]] = {
        "core": [
            MANIFEST_FILENAME,
            "shared/tokenizer/**",
            "shared/chat_template.jinja",
            "shared/config.json",
            "shared/generation_config.json",
        ],
        "inference": [f"variants/{variant_id}/inference/**"] if "inference" in has else [],
        "train": [f"variants/{variant_id}/train/**"] if "train" in has else [],
        "rag": [f"variants/{variant_id}/embedding/**"] if "rag" in has else [],
        "genai": [f"variants/{variant_id}/inference/genai_config.json"] if "genai" in has else [],
        "checksums": [f"variants/{variant_id}/checksums.json"],
    }
    return plan


def _variant_paths(package_dir: Path, variant_id: str) -> dict[str, str]:
    """Resolve the ``paths`` map for a variant from the subdirs that actually exist on disk."""
    paths: dict[str, str] = {"tokenizer": "shared/tokenizer"}
    for sub in VARIANT_SUBDIRS:
        rel = f"variants/{variant_id}/{sub}"
        if (package_dir / rel).is_dir():
            paths[sub] = rel
    return paths


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_files(package_dir: Path) -> list[str]:
    """Repo-relative POSIX paths of every file under ``package_dir`` (excluding the manifest itself)."""
    rels: list[str] = []
    for p in sorted(package_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_FILENAME:
            rels.append(p.relative_to(package_dir).as_posix())
    return rels


def build_manifest(
    package_dir: str | Path,
    variants: list[dict[str, Any]],
    base_model_id: str,
    report: dict[str, Any],
    *,
    default_variant: str | None = None,
    exported_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the full ``mobiletransformers_manifest.json`` dict from the on-disk package tree.

    ``variants`` — list of descriptors, each with at least ``id``, ``executionProvider``,
    ``quantization``, ``supportedEngines``, ``abi``, ``features``, ``minimumAndroidApi``,
    ``recommendedDeviceMemoryMb``. ``paths``/``weightHandoff``/``downloadPlan`` are computed here.
    ``report`` — provenance (architectures, supportedTasks, selectedTask, trustRemoteCode, version pins,
    peftMethods, quantization, mobiletransformersVersion, license, androidRuntime). Integrity
    (``fileSizes``/``sha256``) is stream-hashed from disk. Deterministic given the same tree + inputs.
    """
    package_dir = Path(package_dir)
    if not variants:
        raise ValueError("build_manifest requires at least one variant")
    default_variant = default_variant or variants[0]["id"]
    variant_ids = {v["id"] for v in variants}
    if default_variant not in variant_ids:
        raise ValueError(f"defaultVariant {default_variant!r} not among variants {sorted(variant_ids)}")

    rels = _walk_files(package_dir)
    file_sizes = {rel: (package_dir / rel).stat().st_size for rel in rels}
    sha256 = {rel: _sha256_of(package_dir / rel) for rel in rels}

    manifest_variants: list[dict[str, Any]] = []
    download_plan: dict[str, dict[str, list[str]]] = {}
    for v in variants:
        vid = v["id"]
        features = tuple(v.get("features", ()))
        handoff = f"variants/{vid}/inference/weight_handoff_map.json"
        manifest_variants.append(
            {
                "id": vid,
                "executionProvider": v["executionProvider"],
                "quantization": v["quantization"],
                "supportedEngines": list(v.get("supportedEngines", ["native"])),
                "abi": v.get("abi"),
                "features": list(features),
                "minimumAndroidApi": v.get("minimumAndroidApi"),
                "recommendedDeviceMemoryMb": v.get("recommendedDeviceMemoryMb"),
                "weightHandoff": handoff,
                "paths": _variant_paths(package_dir, vid),
            }
        )
        download_plan[vid] = _download_plan_for(vid, features)

    default_handoff = f"variants/{default_variant}/inference/weight_handoff_map.json"
    required = [f for f in REQUIRED_TOP_LEVEL_FILES if f != MANIFEST_FILENAME]
    required = [MANIFEST_FILENAME, *required, f"variants/{default_variant}/inference/model.onnx"]

    return {
        "schemaVersion": SCHEMA_VERSION,
        "minReaderVersion": MIN_READER_VERSION,
        "baseModelId": base_model_id,
        "exportedAt": exported_at or "",
        "mobiletransformersVersion": report.get("mobiletransformersVersion", ""),
        "artifactFormatVersion": ARTIFACT_FORMAT_VERSION,
        "architectures": list(report.get("architectures", [])),
        "supportedTasks": list(report.get("supportedTasks", [])),
        "selectedTask": report.get("selectedTask"),
        "trustRemoteCode": bool(report.get("trustRemoteCode", False)),
        "optimumOnnxVersion": report.get("optimumOnnxVersion"),
        "transformersVersion": report.get("transformersVersion"),
        "onnxRuntimeTrainingVersion": report.get("onnxRuntimeTrainingVersion"),
        "onnxRuntimeGenAIVersion": report.get("onnxRuntimeGenAIVersion"),
        "peftMethods": list(report.get("peftMethods", [])),
        "quantization": list(report.get("quantization", [])),
        "defaultVariant": default_variant,
        "variants": manifest_variants,
        "downloadPlan": download_plan,
        "requiredFiles": required,
        "fileSizes": file_sizes,
        "sha256": sha256,
        "weightHandoff": default_handoff,
        "androidRuntime": report.get(
            "androidRuntime",
            {"minimumAndroidApi": None, "recommendedDeviceMemoryMb": None, "requiredAbis": []},
        ),
        "license": report.get(
            "license", {"framework": "Apache-2.0", "baseModelWeights": None, "noticeFile": None}
        ),
    }


def write_manifest(package_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest deterministically (sorted keys, trailing newline) to the package root."""
    package_dir = Path(package_dir)
    path = package_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_variant_checksums(package_dir: str | Path, manifest: dict[str, Any]) -> list[Path]:
    """Emit each variant's ``checksums.json`` (the subset of ``sha256`` under its subtree) so a
    downloaded variant subtree is independently verifiable."""
    package_dir = Path(package_dir)
    sha = manifest["sha256"]
    written: list[Path] = []
    for v in manifest["variants"]:
        vid = v["id"]
        prefix = f"variants/{vid}/"
        subset = {rel: digest for rel, digest in sha.items() if rel.startswith(prefix)}
        out = package_dir / "variants" / vid / "checksums.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(out)
    return written


__all__ = [
    "SCHEMA_VERSION",
    "MIN_READER_VERSION",
    "ARTIFACT_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "FEATURE_GROUPS",
    "VARIANT_SUBDIRS",
    "REQUIRED_TOP_LEVEL_FILES",
    "sanitize_repo_id",
    "build_manifest",
    "write_manifest",
    "write_variant_checksums",
]
