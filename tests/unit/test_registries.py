"""Registries are the single source of truth: resolvers cover every legacy branch, fail closed,
and no new business code reintroduces a string-literal dispatch."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobiletransformers.config.constants import MergerVariant, PEFTMethod
from mobiletransformers.config.registry import (
    build_merger_model,
    get_peft_spec,
    resolve_architecture,
    resolve_merger,
)
from mobiletransformers.config.registry.architecture import ARCHITECTURE_REGISTRY
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
    """New package code must dispatch through registries, never `x == "lora"` / `architectures[0] ==`."""
    banned = [
        r"architectures\[0\]\s*==",
        r'peft_method\s*==\s*["\']',
        r'train_method\s*==\s*["\']',
        r'merger_type\s*==\s*["\']',
    ]
    src = REPO_ROOT / "src" / "mobiletransformers"
    result = subprocess.run(
        ["grep", "-rnE", "--include=*.py", "|".join(banned), str(src)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"string-literal dispatch found in src/:\n{result.stdout}"
