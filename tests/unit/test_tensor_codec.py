"""Unit tests for TrainableTensorCodec + HandoffEntry/HandoffMap round-trip determinism (#8)."""

from __future__ import annotations

import json

import pytest

from mobiletransformers.artifacts.handoff_map import (
    HandoffEntry,
    HandoffMap,
    ObservedInit,
    TrainableTensorCodec,
)
from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.config.registry.architecture import resolve_architecture
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import HandoffError


class _FakeConfig:
    architectures = ["LlamaForCausalLM"]


def _entry() -> HandoffEntry:
    name = "model.layers.0.attn.q_proj.MatMul.weight"
    return HandoffEntry(
        training_base_layer_name="backbone.model.layers.0.self_attn.q_proj.base_layer",
        dtype="float16",
        shape=(4096, 4096),
        checkpoint_names={"weight": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight"},
        merged_tensor_names={"weight": name},
        inference_initializer_names={"weight": name},
        external_data_location={"weight": name + ".bin"},
    )


def test_entry_round_trip_preserves_fields() -> None:
    entry = _entry()
    again = HandoffEntry.from_dict(entry.to_dict())
    assert again == entry
    assert again.shape == (4096, 4096)
    assert again.dtype == "float16"


def test_map_json_is_byte_deterministic_across_runs_and_entry_order() -> None:
    a = _entry()
    b = HandoffEntry(
        training_base_layer_name="backbone.model.layers.1.self_attn.k_proj.base_layer",
        dtype="float16",
        shape=(4096, 1024),
        merged_tensor_names={"weight": "model.layers.1.attn.k_proj.MatMul.weight"},
        inference_initializer_names={"weight": "model.layers.1.attn.k_proj.MatMul.weight"},
        external_data_location={"weight": "model.layers.1.attn.k_proj.MatMul.weight.bin"},
    )
    json1 = HandoffMap(entries=[a, b]).to_json()
    json2 = HandoffMap(entries=[b, a]).to_json()  # different input order -> identical output
    assert json1 == json2
    # and it survives a load round-trip
    reparsed = HandoffMap.from_dict(json.loads(json1))
    assert reparsed.to_json() == json1


def test_canonical_inference_name_applies_registry_rewrite() -> None:
    arch = resolve_architecture(_FakeConfig())  # LlamaForCausalLM, attention_module_name="self_attn"
    got = TrainableTensorCodec.canonical_inference_name(
        "backbone.model.layers.0.self_attn.q_proj.base_layer", arch
    )
    assert got == "model.layers.0.attn.q_proj.MatMul"


def test_from_peft_mapping_builds_entry_from_observed_names() -> None:
    arch = resolve_architecture(_FakeConfig())
    peft = get_peft_spec(PEFTMethod.LORA)
    peft_mapping = {
        "backbone.model.layers.0.self_attn.q_proj": {
            "adapter_A": "backbone.model.layers.0.self_attn.q_proj.lora_A.default",
            "adapter_B": "backbone.model.layers.0.self_attn.q_proj.lora_B.default",
        }
    }
    observed = [ObservedInit("model.layers.0.attn.q_proj.MatMul.weight", "float16", (4096, 4096), "weight")]
    entries = TrainableTensorCodec.from_peft_mapping(
        peft_mapping, requires_grad=[], observed_inference_inits=observed, peft_spec=peft, arch_spec=arch
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.inference_initializer_names["weight"] == "model.layers.0.attn.q_proj.MatMul.weight"
    # external_initializer invariant holds by construction
    assert e.merged_tensor_names == e.inference_initializer_names
    assert e.checkpoint_names["adapter_A"].endswith("lora_A.default")
    HandoffMap(entries=entries).validate()  # must pass


def test_from_peft_mapping_raises_on_naming_drift() -> None:
    arch = resolve_architecture(_FakeConfig())
    peft = get_peft_spec(PEFTMethod.LORA)
    peft_mapping = {"backbone.model.layers.0.self_attn.q_proj": {"adapter_A": "x", "adapter_B": "y"}}
    # observed init for a DIFFERENT layer -> no match for the canonical seed -> drift error
    observed = [ObservedInit("model.layers.9.attn.k_proj.MatMul.weight", "float16", (1, 1), "weight")]
    with pytest.raises(HandoffError):
        TrainableTensorCodec.from_peft_mapping(
            peft_mapping, requires_grad=[], observed_inference_inits=observed, peft_spec=peft, arch_spec=arch
        )


def test_from_peft_mapping_quantized_names_come_from_observed() -> None:
    arch = resolve_architecture(_FakeConfig())
    peft = get_peft_spec(PEFTMethod.MARS)
    seed = "model.layers.0.attn.q_proj.MatMul"
    peft_mapping = {"backbone.model.layers.0.self_attn.q_proj": {"shared_A": "a", "adapter_B": "b"}}
    observed = [
        ObservedInit(f"{seed}.qweight", "int4", (4096, 2048), "weight_quantized"),
        ObservedInit(f"{seed}.scales", "float16", (4096, 32), "scale"),
        ObservedInit(f"{seed}.qzeros", "uint8", (4096, 32), "zero_point"),
    ]
    entries = TrainableTensorCodec.from_peft_mapping(
        peft_mapping, requires_grad=[], observed_inference_inits=observed, peft_spec=peft, arch_spec=arch
    )
    q = entries[0].quantization
    assert q is not None
    assert q["scaleName"] == f"{seed}.scales"  # from observed, NOT base_layer_name
    assert q["weightQuantizedName"] == f"{seed}.qweight"
    HandoffMap(entries=entries).validate()


def test_from_peft_mapping_derives_one_orientation_for_the_whole_package() -> None:
    """The WIRING, not the pure function: a real export must reach the right orientation.

    `derive_transpose_policy` has its own unit tests, but a correct function that nothing calls is
    what caused the defect this guards — `ObservedInit.transposed` was a field with a sensible
    meaning that no producer ever set. This drives `from_peft_mapping`, the single call site the
    export actually uses, and asserts the value that lands on the entries.

    Shapes mirror SmolLM2-135M with GQA, the mix that makes the package decidable:
      * `q_proj` is SQUARE (576x576) and cannot decide its own orientation;
      * `v_proj` is (576x192) on disk while `B @ A` is (192x576), so it is stored transposed.

    Both must come out `already_transposed_for_inference` — the square one inheriting the answer
    from the layer that could prove it. If the square layer were left to decide alone it would say
    `no_transpose`, which is how a package ends up describing one export two different ways.
    """
    from mobiletransformers.artifacts.handoff_map import ALREADY_TRANSPOSED

    arch = resolve_architecture(_FakeConfig())
    peft = get_peft_spec(PEFTMethod.LORA)

    def mapping(layer: str) -> dict[str, str]:
        # The PEFT module path spelling, as the real export emits it (see any shipped
        # weight_handoff_map.json `checkpointNames`) — NOT the training-graph initializer spelling.
        base = f"base_model.model.model.layers.0.self_attn.{layer}"
        return {"adapter_A": f"{base}.lora_A.lora", "adapter_B": f"{base}.lora_B.lora"}

    peft_mapping = {
        "backbone.model.layers.0.self_attn.q_proj": mapping("q_proj"),
        "backbone.model.layers.0.self_attn.v_proj": mapping("v_proj"),
    }
    observed = [
        ObservedInit("model.layers.0.attn.q_proj.MatMul.weight", "float32", (576, 576), "weight"),
        ObservedInit("model.layers.0.attn.v_proj.MatMul.weight", "float32", (576, 192), "weight"),
    ]
    # Keyed by the TRAINING GRAPH initializer name, which is a different spelling again.
    specs = {
        "backbone.model.layers.0.self_attn.q_proj.lora_A.lora.weight": {
            "dtype": "float32",
            "shape": [8, 576],
        },
        "backbone.model.layers.0.self_attn.q_proj.lora_B.lora.weight": {
            "dtype": "float32",
            "shape": [576, 8],
        },
        "backbone.model.layers.0.self_attn.v_proj.lora_A.lora.weight": {
            "dtype": "float32",
            "shape": [8, 576],
        },
        "backbone.model.layers.0.self_attn.v_proj.lora_B.lora.weight": {
            "dtype": "float32",
            "shape": [192, 8],
        },
    }

    entries = TrainableTensorCodec.from_peft_mapping(
        peft_mapping,
        requires_grad=[],
        observed_inference_inits=observed,
        peft_spec=peft,
        arch_spec=arch,
        trainable_tensor_specs=specs,
    )

    assert len(entries) == 2
    assert {e.transpose_policy for e in entries} == {ALREADY_TRANSPOSED}, (
        "the export did not resolve one orientation for the package: "
        f"{ {e.training_base_layer_name: e.transpose_policy for e in entries} }"
    )
    HandoffMap(entries=entries).validate()


def test_from_peft_mapping_refuses_a_map_that_describes_no_adapter_factors() -> None:
    """Total adapter-spec drift must fail closed, not degrade to `no_transpose`.

    Each adapter role is looked up by a DERIVED training-graph initializer name. A single miss is
    tolerated (the role is left undescribed). But if specs were supplied and nothing matched at all,
    every entry loses its `adapterShapes`, orientation becomes unobservable, and the package quietly
    declares `no_transpose` — reproducing the 2026-08-14 defect through a name drift instead of an
    unassigned field. This test uses the spelling my own first draft got wrong, which is how the
    hazard was found.
    """
    arch = resolve_architecture(_FakeConfig())
    peft = get_peft_spec(PEFTMethod.LORA)
    base = "base_model.model.model.layers.0.self_attn.v_proj"
    peft_mapping = {
        "backbone.model.layers.0.self_attn.v_proj": {
            "adapter_A": f"{base}.lora_A.lora",
            "adapter_B": f"{base}.lora_B.lora",
        }
    }
    observed = [ObservedInit("model.layers.0.attn.v_proj.MatMul.weight", "float32", (576, 192), "weight")]
    # Real-looking specs under a spelling `to_checkpoint_name` does not produce.
    drifted = {
        f"{base}.lora_A.lora.weight": {"dtype": "float32", "shape": [8, 576]},
        f"{base}.lora_B.lora.weight": {"dtype": "float32", "shape": [192, 8]},
    }

    with pytest.raises(HandoffError, match="none matched any adapter role"):
        TrainableTensorCodec.from_peft_mapping(
            peft_mapping,
            requires_grad=[],
            observed_inference_inits=observed,
            peft_spec=peft,
            arch_spec=arch,
            trainable_tensor_specs=drifted,
        )
