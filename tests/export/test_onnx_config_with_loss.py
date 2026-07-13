"""Training-graph export via the vendored OnnxConfigWithLoss (plan #7 Fallback resolution).

Automates what the plan listed as a manual test: proves the vendored wrapper turns an inference
OnnxConfig into a training graph (labels in, loss out) on optimum's surviving export(). Uses a tiny
synthetic Llama config (no download). Needs the export profile (optimum + torch)."""

from __future__ import annotations

import importlib.util

import pytest

_HAS = importlib.util.find_spec("optimum") is not None and importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS, reason="needs export profile (optimum + torch)")


def _tiny_llama_config():  # type: ignore[no-untyped-def]
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=64,
    )


def test_vendored_wrapper_adds_labels_and_loss() -> None:
    from optimum.exporters.onnx.model_configs import LlamaOnnxConfig

    from mobiletransformers.export.onnx_config_with_loss import OnnxConfigWithLoss

    base = LlamaOnnxConfig(
        _tiny_llama_config(), task="text-generation", use_past=False, use_past_in_inputs=False
    )
    ocl = OnnxConfigWithLoss(base)
    assert "labels" in ocl.inputs
    assert "loss" in ocl.outputs
    dummy = ocl.generate_dummy_inputs(framework="pt")
    assert "labels" in dummy


def test_training_graph_export_emits_loss(tmp_path) -> None:
    import onnx
    import torch
    from optimum.exporters.onnx import export
    from optimum.exporters.onnx.model_configs import LlamaOnnxConfig
    from transformers import LlamaForCausalLM

    from mobiletransformers.export.onnx_config_with_loss import OnnxConfigWithLoss

    class _TrainerWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = model
            self.config = model.config
            self.training = True

        def forward(self, input_ids, attention_mask, position_ids, labels):  # type: ignore[no-untyped-def]
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )

    cfg = _tiny_llama_config()
    model = LlamaForCausalLM(cfg)
    base = LlamaOnnxConfig(cfg, task="text-generation", use_past=False, use_past_in_inputs=False)
    ocl = OnnxConfigWithLoss(base)
    wrapper = _TrainerWrapper(model)
    wrapper.train()

    out = tmp_path / "model.onnx"
    inputs, outputs = export(wrapper, ocl, out, opset=20, do_constant_folding=False)

    assert "labels" in inputs
    assert "loss" in outputs
    graph = onnx.load(str(out))
    graph_inputs = {i.name for i in graph.graph.input}
    graph_outputs = {o.name for o in graph.graph.output}
    assert "labels" in graph_inputs
    assert "loss" in graph_outputs
