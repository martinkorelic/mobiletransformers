"""#33 encoder training gate: export -> generate_artifacts -> real train step -> a metric.

This is the plan's Definition-of-done for encoder support minus the Android smoke (device-gated). It
downloads a small real encoder, so it is env-gated on the ``ort-training-local`` profile AND on
network access, exactly like the other integration smokes.

Run:  uv run --python 3.12 --group ort-training-local --no-default-groups \\
          pytest tests/integration/test_encoder_training_gate.py -q

## What this pins, and why each part earned a test

Three defects had to be fixed to get here, none of which a decoder run could have surfaced:

1. **PEFT was wrapped as ``CAUSAL_LM``** at both LoRA call sites regardless of task, mis-configuring
   any encoder. Now from ``TaskSpec.peft_task_type``.
2. **Activation quantization on the gradient path.** ORT rewrites ``Gemm`` -> ``MatMul`` *before*
   quantizing and matches ``nodes_to_exclude`` against the rewritten name, so excluding BERT's
   pooler/classifier ``Gemm`` silently missed and they came back as ``MatMulInteger`` fed by
   ``DynamicQuantizeLinear`` — which has no gradient. Quantized *weights* are fine (they dequantize to
   float and are frozen); quantized *activations* are not.
3. **``LayerNormalization`` exported with one output.** ORT's gradient reads its optional saved
   mean/inv-std. RMSNorm decoders export ``SimplifiedLayerNormalization`` with 2 outputs already.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="ort-training-local profile only")
pytest.importorskip("onnxruntime.training.artifacts", reason="ort-training-local profile only")
pytest.importorskip("optimum.exporters.onnx", reason="optimum required for the export leg")
pytest.importorskip("transformers", reason="transformers required for the export leg")

#: Small real encoder (22.7M params) that the project already ships as the RAG embedder, so this adds
#: no new download to a machine that has run an export.
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: A tiny, deliberately separable sentiment set: the point is to prove gradients reach the encoder and
#: move a real metric, not to measure generalisation.
POSITIVE = [
    "this film was wonderful",
    "an absolute delight to watch",
    "brilliant and moving",
    "a masterpiece of storytelling",
]
NEGATIVE = [
    "this film was terrible",
    "a complete waste of time",
    "boring and painfully dull",
    "an awful, incoherent mess",
]


@pytest.fixture(scope="module")
def encoder_artifacts(tmp_path_factory):
    """Export the encoder training graph and generate ORT training artifacts once."""
    from mobiletransformers.artifacts.builder import gen_artifacts
    from mobiletransformers.export.training_export import optimum_hf_export

    root = tmp_path_factory.mktemp("encoder_gate")
    export_dir, artifact_dir = root / "export", root / "artifacts"
    export_dir.mkdir()
    artifact_dir.mkdir()

    optimum_hf_export(
        model_id=ENCODER_MODEL,
        model_output=str(export_dir),
        training_mode=True,
        train_method="lora",
        lora_rank=8,
        lora_alpha=8,
        quantize=True,
        # No lora_target: the architecture registry's row for BertForSequenceClassification decides.
        task_type="text-classification",
    )
    config = gen_artifacts(
        train_dir=str(export_dir),
        artifact_dir=str(artifact_dir),
        model_name="quant_model.onnx",
        training_config={},
    )
    return export_dir, artifact_dir, config


def test_exported_graph_takes_per_sequence_labels_and_emits_a_loss(encoder_artifacts):
    """`labels[batch]` — one per sequence — is the contract that defines this objective."""
    import onnx

    export_dir, _, _ = encoder_artifacts
    model = onnx.load(str(export_dir / "model.onnx"), load_external_data=False)

    shapes = {
        i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input
    }
    assert shapes["labels"] == ["batch_size"], "classification supervises one label per sequence"
    assert shapes["input_ids"] == ["batch_size", "sequence_length"]
    # token_type_ids, not position_ids — the encoder wrapper's signature is the exported input set.
    assert "token_type_ids" in shapes and "position_ids" not in shapes
    assert [o.name for o in model.graph.output] == ["loss", "logits"]


def test_no_activation_quantization_survives_on_the_gradient_path(encoder_artifacts):
    """Quantized weights are fine; quantized activations have no gradient at all.

    `DequantizeLinear` on a frozen weight is what the project wants. `DynamicQuantizeLinear` means an
    *activation* was quantized, and ORT registers no gradient builder for it.
    """
    import onnx

    export_dir, _, _ = encoder_artifacts
    model = onnx.load(str(export_dir / "quant_model.onnx"), load_external_data=False)
    op_types = [n.op_type for n in model.graph.node]

    assert op_types.count("DynamicQuantizeLinear") == 0
    assert op_types.count("MatMulInteger") == 0
    assert op_types.count("DequantizeLinear") > 0, "weights should still be quantized"


def test_training_artifacts_are_generated(encoder_artifacts):
    _, artifact_dir, config = encoder_artifacts

    for name in ("training_model.onnx", "eval_model.onnx", "optimizer_model.onnx", "checkpoint"):
        assert (artifact_dir / name).exists(), f"{name} missing"
    assert config["trainable_parameter_count"] > 0
    assert config["source_parameter_count"] > config["trainable_parameter_count"]


def test_the_encoder_actually_learns(encoder_artifacts):
    """The gate itself: gradients reach the encoder and move a real metric.

    Asserts on **accuracy**, not only loss — a falling loss shows the optimizer runs, while accuracy
    going to 1.0 on a separable set shows the update is in the right direction.
    """
    from onnxruntime.training.api import CheckpointState, Module, Optimizer
    from transformers import AutoTokenizer

    _, artifact_dir, _ = encoder_artifacts

    state = CheckpointState.load_checkpoint(str(artifact_dir / "checkpoint"))
    module = Module(str(artifact_dir / "training_model.onnx"), state, str(artifact_dir / "eval_model.onnx"))
    optimizer = Optimizer(str(artifact_dir / "optimizer_model.onnx"), module)

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    encoded = tokenizer(
        POSITIVE + NEGATIVE,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)
    labels = np.array([1] * len(POSITIVE) + [0] * len(NEGATIVE), dtype=np.int64)

    def accuracy() -> float:
        module.eval()
        logits = module(input_ids, attention_mask, token_type_ids, labels)[1]
        return float((logits.argmax(-1) == labels).mean())

    module.train()
    losses = []
    for _ in range(30):
        losses.append(float(module(input_ids, attention_mask, token_type_ids, labels)[0]))
        optimizer.step()
        module.lazy_reset_grad()

    assert all(np.isfinite(losses)), "loss went non-finite; gradients are not well-formed"
    assert losses[-1] < losses[0] * 0.95, f"loss did not fall materially: {losses[0]} -> {losses[-1]}"
    assert accuracy() == 1.0, "the encoder did not learn a deliberately separable 8-example set"
