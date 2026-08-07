"""Registries are the single source of truth: resolvers cover every legacy branch, fail closed,
and no new business code reintroduces a string-literal dispatch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobiletransformers.config.constants import MergerVariant, PEFTMethod
from mobiletransformers.config.registry import (
    build_merger_model,
    get_peft_spec,
    resolve_architecture,
    resolve_merger,
)
from mobiletransformers.config.registry.architecture import ARCHITECTURE_REGISTRY
from mobiletransformers.config.registry.peft import (
    PEFT_TARGET_MODULES_BY_MODEL_TYPE,
    build_adapter_mapping,
)
from mobiletransformers.exceptions import UnsupportedModelError

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every architecture the legacy trainer/builder.py:260-272 dispatch handled.
LEGACY_TRAINING_ARCHES = [
    "LlamaForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "OPTForCausalLM",
    "BertModel",
]


@pytest.mark.parametrize("arch", LEGACY_TRAINING_ARCHES)
def test_resolve_architecture_covers_legacy_branches(arch):
    spec = resolve_architecture(SimpleNamespace(architectures=[arch]))
    assert spec.architecture == arch
    assert spec.onnx_config_class  # a dotted path is present


def test_resolve_architecture_unknown_fails_closed():
    with pytest.raises(UnsupportedModelError):
        resolve_architecture(SimpleNamespace(architectures=["TotallyMadeUpForCausalLM"]))
    with pytest.raises(UnsupportedModelError):
        resolve_architecture(SimpleNamespace(architectures=[]))


@pytest.mark.parametrize("method", list(PEFTMethod))
def test_get_peft_spec_covers_all_methods(method):
    spec = get_peft_spec(method)
    assert spec.method == method


@pytest.mark.parametrize(
    "method,quant_in,expected",
    [
        (PEFTMethod.LORA, False, MergerVariant.LORA),
        (PEFTMethod.LORA, True, MergerVariant.LORA_Q),
        (PEFTMethod.LORA_XS, False, MergerVariant.LORA),
        (PEFTMethod.LORA_XS, True, MergerVariant.LORA_Q),
        (PEFTMethod.MARS, False, MergerVariant.MARS_Q),
        (PEFTMethod.MARS, True, MergerVariant.MARS_Q),
    ],
)
def test_resolve_merger_variant_and_filename(method, quant_in, expected):
    spec = resolve_merger(method, quant_in=quant_in, quant_out=True)
    assert spec.variant == expected
    assert "_2" not in spec.output_filename  # descriptive names, no legacy _2 duplication
    assert spec.output_filename.endswith(".onnx")


@pytest.mark.parametrize("method", [PEFTMethod.ALL, PEFTMethod.NOLORA])
def test_resolve_merger_no_merger_methods_fail_closed(method):
    with pytest.raises(UnsupportedModelError):
        resolve_merger(method, quant_in=True, quant_out=True)


def test_build_merger_model_emits_valid_graph(tmp_path):
    """build_merger_model is wired (#9); byte-equivalence to the legacy factories lives in
    tests/unit/test_merger_builder.py. Here we just confirm it emits a checkable graph."""
    import onnx

    spec = resolve_merger(PEFTMethod.LORA, quant_in=True, quant_out=True)
    out = tmp_path / "out.onnx"
    build_merger_model(spec, out)
    onnx.checker.check_model(str(out))


def test_architecture_registry_nonempty_and_consistent():
    for name, spec in ARCHITECTURE_REGISTRY.items():
        assert spec.architecture == name


def test_no_string_literal_dispatch_in_src():
    """New package code must dispatch through registries, never `x == "lora"` / `architectures[0] ==`.

    Shares ONE definition of "banned pattern" and one comment-aware matcher with
    ``tests/unit/test_guards.py``, which applies the same rules to the legacy roots and the C++ tree.
    Keeping them in step matters during the migration: as a legacy module moves under ``src/`` its
    dispatch debt moves from the guards' ``DISPATCH_ALLOWLIST`` to *this* test, which has no
    allow-list — so a module must be clean before it is allowed to move.
    """
    from tests.unit.test_guards import DISPATCH_PATTERNS, _grep

    src = REPO_ROOT / "src" / "mobiletransformers"
    hits = _grep(DISPATCH_PATTERNS, [src], ("*.py",))
    assert not hits, (
        "string-literal dispatch found in src/ — resolve it through config/registry/ "
        "(a module carrying dispatch debt must be cleaned BEFORE it migrates):\n" + "\n".join(hits)
    )


# --- #6 A3: build_adapter_mapping + the deduplicated PEFT target table -----------------------------
def test_build_adapter_mapping_returns_empty_for_methods_without_adapters():
    for method in (PEFTMethod.ALL, PEFTMethod.NOLORA):
        assert build_adapter_mapping(method, object()) == {}


def test_build_adapter_mapping_dispatches_per_method():
    """The registry resolves WHICH builder runs; callers never branch on the method string."""
    calls: list[tuple[str, dict]] = []

    def fake(model, **kwargs):
        calls.append((model, kwargs))
        return {"layer": {"adapter_A": "a"}}

    for method in (PEFTMethod.LORA, PEFTMethod.LORA_XS, PEFTMethod.MARS):
        spec = get_peft_spec(method)
        assert spec.mapping_builder is not None, f"{method} claims builds_mapping but has no builder"
        with mock.patch("mobiletransformers.config.registry.peft.import_from_path", return_value=fake):
            assert build_adapter_mapping(method, "model-obj") == {"layer": {"adapter_A": "a"}}
    assert [c[0] for c in calls] == ["model-obj"] * 3


def test_peft_target_modules_table_has_one_source():
    """#6: the MARS and ablation tables were byte-identical copies under two names."""
    from peft_models.ablation.utils import TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING
    from peft_models.mars.utils import TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING

    assert TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING is PEFT_TARGET_MODULES_BY_MODEL_TYPE
    assert TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING is PEFT_TARGET_MODULES_BY_MODEL_TYPE


def test_peft_target_table_is_wider_than_the_architecture_registry():
    """Guards the reason these two tables are NOT merged: different key spaces, different coverage."""
    assert "t5" in PEFT_TARGET_MODULES_BY_MODEL_TYPE  # model_type keys...
    assert "LlamaForCausalLM" not in PEFT_TARGET_MODULES_BY_MODEL_TYPE  # ...not architecture keys
    assert len(PEFT_TARGET_MODULES_BY_MODEL_TYPE) > len(ARCHITECTURE_REGISTRY)
