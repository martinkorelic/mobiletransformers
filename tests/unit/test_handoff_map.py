"""Unit tests for HandoffMap.validate() invariants + the canonical check_compat (#8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobiletransformers.artifacts.handoff_map import (
    HANDOFF_MAP_READER_VERSION,
    HandoffEntry,
    HandoffMap,
)
from mobiletransformers.artifacts.versioning import SchemaVersionError, check_compat
from mobiletransformers.config.constants import HandoffMode
from mobiletransformers.exceptions import HandoffError

_CASES = json.loads((Path(__file__).parent.parent / "fixtures" / "check_compat_cases.json").read_text())[
    "cases"
]


def _good_entry(idx: int = 0) -> HandoffEntry:
    name = f"model.layers.{idx}.attn.q_proj.MatMul.weight"
    return HandoffEntry(
        training_base_layer_name=f"backbone.model.layers.{idx}.self_attn.q_proj.base_layer",
        dtype="float16",
        shape=(4096, 4096),
        merged_tensor_names={"weight": name},
        inference_initializer_names={"weight": name},
        external_data_location={"weight": name + ".bin"},
    )


# --- check_compat (shared cross-language fixture) --------------------------------------------------
@pytest.mark.parametrize("case", _CASES, ids=[c["why"] for c in _CASES])
def test_check_compat_matches_shared_fixture(case: dict) -> None:
    if case["expect"] == "accept":
        check_compat(case["doc"], case["minReader"], case["reader"])  # must not raise
    else:
        with pytest.raises(SchemaVersionError):
            check_compat(case["doc"], case["minReader"], case["reader"])


def test_reader_version_is_one_zero() -> None:
    assert HANDOFF_MAP_READER_VERSION == "1.0"


# --- validate() invariants ------------------------------------------------------------------------
def test_valid_external_initializer_map_passes() -> None:
    HandoffMap(entries=[_good_entry(0), _good_entry(1)]).validate()


def test_merged_must_equal_inference_name() -> None:
    e = _good_entry()
    e.merged_tensor_names = {"weight": "model.layers.0.attn.q_proj.MatMul.WRONG"}
    with pytest.raises(HandoffError, match="mergedTensorNames"):
        HandoffMap(entries=[e]).validate()


def test_quantized_scale_from_base_layer_name_is_rejected() -> None:
    # The documented bug: scale derived from base_layer_name instead of the observed inference init.
    seed = "model.layers.0.attn.q_proj.MatMul"
    e = HandoffEntry(
        training_base_layer_name="backbone.model.layers.0.self_attn.q_proj.base_layer",
        dtype="int4",
        shape=(4096, 2048),
        merged_tensor_names={
            "weight_quantized": f"{seed}.qweight",
            "scale": f"{seed}.scales",
            "zero_point": f"{seed}.qzeros",
        },
        inference_initializer_names={
            "weight_quantized": f"{seed}.qweight",
            "scale": f"{seed}.scales",
            "zero_point": f"{seed}.qzeros",
        },
        external_data_location={
            "weight_quantized": f"{seed}.qweight.bin",
            "scale": f"{seed}.scales.bin",
            "zero_point": f"{seed}.qzeros.bin",
        },
        quantization={
            "weightQuantizedName": f"{seed}.qweight",
            # BUG: derived from base_layer_name, not the observed inference init
            "scaleName": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight_scale",
            "zeroPointName": f"{seed}.qzeros",
        },
    )
    with pytest.raises(HandoffError, match="not derived from base_layer_name"):
        HandoffMap(entries=[e]).validate()


def test_duplicate_external_location_rejected() -> None:
    a, b = _good_entry(0), _good_entry(1)
    b.external_data_location = dict(a.external_data_location)  # collide
    with pytest.raises(HandoffError, match="duplicate externalDataLocation"):
        HandoffMap(entries=[a, b]).validate()


def test_duplicate_inference_name_rejected() -> None:
    a, b = _good_entry(0), _good_entry(1)
    b.inference_initializer_names = dict(a.inference_initializer_names)
    b.merged_tensor_names = dict(a.merged_tensor_names)
    with pytest.raises(HandoffError, match="duplicate inferenceInitializerName"):
        HandoffMap(entries=[a, b]).validate()


def test_model_input_mode_fails_closed_v1() -> None:
    with pytest.raises(HandoffError, match="not supported"):
        HandoffMap(entries=[_good_entry()], handoff_mode=HandoffMode.MODEL_INPUT).validate()


def test_adapter_mode_fails_closed_v1() -> None:
    with pytest.raises(HandoffError, match="not supported"):
        HandoffMap(entries=[_good_entry()], handoff_mode=HandoffMode.ADAPTER).validate()


def test_unsupported_major_fails_closed() -> None:
    with pytest.raises(SchemaVersionError):
        HandoffMap(entries=[_good_entry()], schema_version="2.0").validate()


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "weight_handoff_map.json"
    HandoffMap(entries=[_good_entry(0), _good_entry(1)]).save(path)
    loaded = HandoffMap.load(path)
    assert len(loaded.entries) == 2
    assert loaded.handoff_mode == HandoffMode.EXTERNAL_INITIALIZER
