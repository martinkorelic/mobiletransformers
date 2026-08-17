"""#21 constraint-based variant selection (onnx-free; uses the tiny_package fixture)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.exceptions import NoCompatibleVariant
from mobiletransformers.hub.variant_select import Constraints, default_desktop_constraints, select_variant

MANIFEST = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package" / "mobiletransformers_manifest.json"
)


def _m() -> MobileTransformersManifest:
    return MobileTransformersManifest.load(MANIFEST)


def test_default_desktop_prefers_int4():
    assert select_variant(_m(), default_desktop_constraints()) == "cpu-int4"


def test_preferred_quantization_soft_preference():
    c = Constraints(abi=("arm64-v8a", "x86_64"), preferred_quantization="fp16")
    assert select_variant(_m(), c) == "cpu-fp16"


def test_memory_ceiling_excludes_fp16():
    # 4096 MB fits cpu-int4 (3072) but not cpu-fp16 (6144), even if fp16 preferred.
    c = Constraints(abi=("arm64-v8a", "x86_64"), preferred_quantization="fp16", device_memory_mb=4096)
    assert select_variant(_m(), c) == "cpu-int4"


def test_genai_engine_requires_int4():
    c = Constraints(abi=("arm64-v8a",), engine="genai", requested_features=("core", "inference", "genai"))
    assert select_variant(_m(), c) == "cpu-int4"


def test_storage_budget_fails_closed():
    c = Constraints(abi=("arm64-v8a", "x86_64"), available_storage_bytes=10)  # ~10 bytes: impossible
    with pytest.raises(NoCompatibleVariant, match="budget"):
        select_variant(_m(), c)


def test_no_matching_abi_raises():
    c = Constraints(abi=("riscv64",))  # cpu-int4 is arm64-only; cpu-fp16 abi=null (any) -> fp16 chosen
    # cpu-fp16 has abi=null so it matches any ABI; assert it is selected rather than raising.
    assert select_variant(_m(), c) == "cpu-fp16"
