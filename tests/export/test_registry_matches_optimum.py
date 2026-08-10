"""Cross-check: every `ArchitectureSpec.onnx_config_class` is the one Optimum itself would pick.

Env-gated on the `export` profile (needs optimum + transformers); no network, no model downloads.

## Why this exists

`ArchitectureSpec.onnx_config_class` is a **lazy dotted path**. It is resolved only when a training
export actually runs, so a wrong binding is invisible to every other gate — it does not fail import,
lint, typecheck or any unit test. The registry carried the note *"corrected but NOT exercised end to
end"* for exactly this reason.

It was wrong. `Gemma3ForCausalLM` bound `Gemma3OnnxConfig`, which is the **multimodal** config: its
`__init__` does `super().__init__(config.text_config, ...)`, and a text-only Gemma-3 config
(`google/gemma-3-270m`, `model_type: gemma3_text`) has no `text_config`. Every training export of a
Gemma-3 would have died with `AttributeError`. Optimum maps `gemma3_text` to `Gemma3TextOnnxConfig`.

The generalizing fix is this test rather than one corrected row: Optimum's `TasksManager` already
knows the right answer for every model type it supports, so the registry is checked **against** it
instead of being maintained in parallel and hoping the two agree.
"""

from __future__ import annotations

import pytest

pytest.importorskip("optimum.exporters.onnx", reason="export profile only")
pytest.importorskip("transformers", reason="export profile only")

from mobiletransformers.config.registry.architecture import ARCHITECTURE_REGISTRY  # noqa: E402


def _optimum_config_for(model_type: str) -> type | None:
    """The ONNX config class Optimum's TasksManager maps `model_type` to, or None if unsupported."""
    import optimum.exporters.onnx.model_configs  # noqa: F401  # decorator registration
    from optimum.exporters.tasks import TasksManager

    entry = TasksManager._SUPPORTED_MODEL_TYPE.get(model_type, {}).get("onnx")
    if not entry:
        return None
    # Values are `functools.partial(SomeOnnxConfig, task=..., use_past=...)`.
    constructor = next(iter(entry.values()))
    return getattr(constructor, "func", constructor)


def _model_types_for_architecture(architecture: str) -> set[str]:
    """Every `model_type` whose auto-mapping declares this architecture class name."""
    from transformers.models.auto import modeling_auto

    found: set[str] = set()
    for attr in dir(modeling_auto):
        if not attr.endswith("_MAPPING_NAMES"):
            continue
        mapping = getattr(modeling_auto, attr)
        if not isinstance(mapping, dict):
            continue
        for model_type, names in mapping.items():
            candidates = {names} if isinstance(names, str) else set(names)
            if architecture in candidates:
                found.add(model_type)
    return found


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURE_REGISTRY))
def test_registry_binding_matches_optimum_task_manager(architecture: str) -> None:
    """A row's ONNX config must be the class Optimum resolves for that architecture's model type."""
    spec = ARCHITECTURE_REGISTRY[architecture]
    if spec.onnx_config_class is None:
        pytest.skip(f"{architecture} is inference-only (no Optimum config by design)")

    model_types = _model_types_for_architecture(architecture)
    if not model_types:
        pytest.skip(f"{architecture} is not in this transformers line's auto-mappings")

    expected = {_optimum_config_for(mt) for mt in model_types}
    expected.discard(None)
    if not expected:
        pytest.skip(f"Optimum has no ONNX config for {sorted(model_types)}")

    declared = spec.onnx_config_class.rsplit(".", 1)[-1]
    expected_names = {cls.__name__ for cls in expected}

    assert declared in expected_names, (
        f"{architecture} (model_type {sorted(model_types)}) is bound to {declared!r}, but Optimum "
        f"resolves {sorted(expected_names)}. A lazy dotted path makes this invisible until an export "
        "runs — fix the registry row, not this test."
    )


def test_gemma3_binds_the_text_config_not_the_multimodal_one() -> None:
    """The specific defect this file was written for, pinned by name.

    `Gemma3OnnxConfig` reads `config.text_config`; a `Gemma3TextConfig` has no such attribute, so the
    binding was not merely imprecise — it could not construct at all.
    """
    spec = ARCHITECTURE_REGISTRY["Gemma3ForCausalLM"]
    assert spec.onnx_config_class.endswith("Gemma3TextOnnxConfig")

    from transformers import Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=37,
        head_dim=16,
    )
    onnx_config = spec.load_onnx_config_class()(config, task="text-generation", use_past=True)
    outputs = list(onnx_config.outputs)
    # The canonical contract `export/normalize.py` enforces on every package.
    assert outputs[0] == "logits"
    assert outputs[1:3] == ["present.0.key", "present.0.value"]
