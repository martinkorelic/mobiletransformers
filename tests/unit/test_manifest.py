"""#13 manifest validator + variant selection. Uses the committed tiny package fixture (onnx-free)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.artifacts.versioning import SchemaVersionError
from mobiletransformers.exceptions import ManifestError, NoCompatibleVariant

PKG = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package"
MANIFEST = PKG / "mobiletransformers_manifest.json"


def _copy_pkg(dst: Path) -> Path:
    shutil.copytree(PKG, dst)
    return dst


def test_valid_fixture_passes():
    m = MobileTransformersManifest.load(MANIFEST)
    m.validate(PKG)  # no raise


def test_round_trip_preserves_unknown_fields():
    data = json.loads(MANIFEST.read_text())
    data["someFutureField"] = {"nested": 1}
    m = MobileTransformersManifest.from_dict(data)
    assert m.to_dict()["someFutureField"] == {"nested": 1}
    # deterministic serialization
    assert m.to_json() == json.dumps(data, indent=2, sort_keys=True) + "\n"


def test_bad_default_variant_rejected(tmp_path):
    pkg = _copy_pkg(tmp_path / "p")
    data = json.loads((pkg / "mobiletransformers_manifest.json").read_text())
    data["defaultVariant"] = "does-not-exist"
    (pkg / "mobiletransformers_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="defaultVariant"):
        MobileTransformersManifest.from_dict(data).validate(pkg)


def test_unresolvable_weight_handoff_rejected(tmp_path):
    pkg = _copy_pkg(tmp_path / "p")
    # Delete a per-tensor .bin the handoff map references.
    (pkg / "variants/cpu-int4/inference/model.layers.0.attn.q_proj.MatMul.weight.bin").unlink()
    m = MobileTransformersManifest.load(pkg / "mobiletransformers_manifest.json")
    with pytest.raises(ManifestError, match="missing external file"):
        m.validate(pkg)


def test_feature_without_path_rejected(tmp_path):
    pkg = _copy_pkg(tmp_path / "p")
    data = json.loads((pkg / "mobiletransformers_manifest.json").read_text())
    for v in data["variants"]:
        if v["id"] == "cpu-int4":
            v["paths"].pop("inference")
    with pytest.raises(ManifestError, match="inference"):
        MobileTransformersManifest.from_dict(data).validate(pkg)


def test_schema_major_bump_fails_closed(tmp_path):
    pkg = _copy_pkg(tmp_path / "p")
    data = json.loads((pkg / "mobiletransformers_manifest.json").read_text())
    data["schemaVersion"] = "2.0"
    data["minReaderVersion"] = "2.0"
    with pytest.raises(SchemaVersionError):
        MobileTransformersManifest.from_dict(data).validate(pkg)


# --- variant selection ------------------------------------------------------


def test_select_prefers_smallest_memory_then_default():
    m = MobileTransformersManifest.load(MANIFEST)
    # arm64 + core/inference on native: cpu-int4 (3072) beats cpu-fp16 (6144, and abi=null).
    sel = m.select_variant(abis=["arm64-v8a"], requested_features=["core", "inference"])
    assert sel.id == "cpu-int4"


def test_select_requires_genai_engine_filters_out_native_only():
    m = MobileTransformersManifest.load(MANIFEST)
    sel = m.select_variant(abis=["arm64-v8a"], requested_engine="genai", requested_features=["genai"])
    assert sel.id == "cpu-int4"  # only cpu-int4 supports genai


def test_select_no_match_raises(tmp_path):
    m = MobileTransformersManifest.load(MANIFEST)
    # rag requested on an engine/abi combo that has it only on cpu-int4, but force a memory ceiling
    # below both variants' requirements.
    with pytest.raises(NoCompatibleVariant):
        m.select_variant(abis=["arm64-v8a"], total_mem_mb=1024, requested_features=["core"])


def test_select_memory_ceiling_picks_only_fitting_variant():
    m = MobileTransformersManifest.load(MANIFEST)
    # 4096 MB fits cpu-int4 (3072) but not cpu-fp16 (6144); abi any-match via arm64.
    sel = m.select_variant(abis=["arm64-v8a"], total_mem_mb=4096, requested_features=["core", "inference"])
    assert sel.id == "cpu-int4"
    assert sel.recommended_device_memory_mb == 3072
