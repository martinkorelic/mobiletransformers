"""#14 Hub package format: sanitize_repo_id parity, build_manifest round-trip, dual-engine sanity.

onnx-free (core env). Uses the committed tiny package fixture (tests/fixtures/tiny_package).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobiletransformers.artifacts.versioning import SchemaVersionError, check_compat
from mobiletransformers.hub.package_format import (
    MANIFEST_FILENAME,
    build_manifest,
    sanitize_repo_id,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PKG = FIXTURES / "tiny_package"


def _load_manifest() -> dict:
    return json.loads((PKG / MANIFEST_FILENAME).read_text())


# --- sanitize_repo_id parity ------------------------------------------------


def _sanitize_cases() -> list[tuple[str, str]]:
    data = json.loads((FIXTURES / "sanitize_repo_id_cases.json").read_text())
    return [(c["input"], c["expected"]) for c in data["cases"]]


@pytest.mark.parametrize("raw,expected", _sanitize_cases())
def test_sanitize_repo_id_matches_parity_oracle(raw, expected):
    assert sanitize_repo_id(raw) == expected


# --- build_manifest round-trip ---------------------------------------------


def test_manifest_integrity_keys_exist_on_disk():
    m = _load_manifest()
    for rel in m["sha256"]:
        assert (PKG / rel).is_file(), f"sha256 references missing file {rel}"
    for rel in m["fileSizes"]:
        assert (PKG / rel).stat().st_size == m["fileSizes"][rel]


def test_required_files_present():
    m = _load_manifest()
    for rel in m["requiredFiles"]:
        assert (PKG / rel).exists(), f"requiredFile missing: {rel}"


def test_download_plan_patterns_resolve():
    m = _load_manifest()
    for variant_id, groups in m["downloadPlan"].items():
        for group, patterns in groups.items():
            for pat in patterns:
                if pat.endswith("/**"):
                    matches = list(PKG.glob(pat.replace("/**", "/**/*")))
                    assert matches, f"{variant_id}/{group} glob {pat} matched nothing"
                else:
                    assert (PKG / pat).exists(), f"{variant_id}/{group} path {pat} missing"


def test_build_manifest_is_deterministic():
    # Rebuilding over the same tree reproduces identical integrity maps.
    m = _load_manifest()
    variants = [
        {k: v for k, v in var.items() if k not in ("weightHandoff", "paths")} for var in m["variants"]
    ]
    report = {
        "mobiletransformersVersion": m["mobiletransformersVersion"],
        "architectures": m["architectures"],
        "supportedTasks": m["supportedTasks"],
        "selectedTask": m["selectedTask"],
        "peftMethods": m["peftMethods"],
        "quantization": m["quantization"],
    }
    rebuilt = build_manifest(
        PKG,
        variants,
        base_model_id=m["baseModelId"],
        report=report,
        default_variant=m["defaultVariant"],
        exported_at=m["exportedAt"],
    )
    assert rebuilt["sha256"] == m["sha256"]
    assert rebuilt["fileSizes"] == m["fileSizes"]
    assert rebuilt["downloadPlan"] == m["downloadPlan"]


def test_parameter_counts_reach_the_manifest():
    """The training stage reports both counts; `build_manifest` used to drop them on the floor.

    Every shipped package (decoder and encoder alike) read `null` for these while
    `train/trainable_parameters.json` carried the real number — so a package could not be audited
    from its manifest alone.
    """
    m = _load_manifest()
    variants = [
        {k: v for k, v in var.items() if k not in ("weightHandoff", "paths")} for var in m["variants"]
    ]
    report = {"trainableParameterCount": 442_368, "trainingParameterCount": 135_000_000}
    rebuilt = build_manifest(
        PKG, variants, base_model_id=m["baseModelId"], report=report, default_variant=m["defaultVariant"]
    )
    assert rebuilt["trainableParameterCount"] == 442_368
    assert rebuilt["trainingParameterCount"] == 135_000_000

    # Inference-only packages legitimately have neither: the key is present and null, not absent.
    inference_only = build_manifest(
        PKG, variants, base_model_id=m["baseModelId"], report={}, default_variant=m["defaultVariant"]
    )
    assert inference_only["trainableParameterCount"] is None
    assert inference_only["trainingParameterCount"] is None


# --- dual-engine sanity -----------------------------------------------------


def test_genai_variant_has_both_configs_native_only_omits():
    m = _load_manifest()
    by_id = {v["id"]: v for v in m["variants"]}
    # cpu-int4 supports genai -> genai_config.json present, downloadPlan.genai non-empty.
    assert "genai" in by_id["cpu-int4"]["supportedEngines"]
    assert (PKG / "variants/cpu-int4/inference/genai_config.json").exists()
    assert m["downloadPlan"]["cpu-int4"]["genai"]
    # cpu-fp16 is native-only -> no genai_config.json, downloadPlan.genai empty.
    assert "genai" not in by_id["cpu-fp16"]["supportedEngines"]
    assert not (PKG / "variants/cpu-fp16/inference/genai_config.json").exists()
    assert m["downloadPlan"]["cpu-fp16"]["genai"] == []


# --- schema versioning (F1) -------------------------------------------------


def test_manifest_carries_schema_versions_and_reader_gate():
    m = _load_manifest()
    assert m["schemaVersion"] and m["minReaderVersion"]
    # A reader at 1.0 accepts a 1.0 doc; a future major fails closed.
    check_compat(m["schemaVersion"], m["minReaderVersion"], "1.0")
    with pytest.raises(SchemaVersionError):
        check_compat("2.0", "2.0", "1.0")
