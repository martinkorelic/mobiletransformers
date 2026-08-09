"""Unit tests for the task registry (`config/registry/task.py`, #6 pattern, consumed by #33).

The registry replaced three task-shaped branches in `export/training_export.py`. Two were style; the
third — `LoraConfig(..., task_type="CAUSAL_LM")` at both LoRA call sites — was a latent defect for any
non-decoder model, which is why these tests assert the *encoder* row's values explicitly rather than
just that lookups work.
"""

from __future__ import annotations

import pytest

from mobiletransformers.config.constants import TaskType
from mobiletransformers.config.registry.task import TASK_REGISTRY, get_task_spec
from mobiletransformers.exceptions import UnsupportedModelError


def test_every_task_type_has_a_row():
    """A TaskType with no row would fall through to a branch somewhere — that is the thing being removed."""
    assert set(TASK_REGISTRY) == set(TaskType)


def test_rows_are_self_consistent():
    for task, spec in TASK_REGISTRY.items():
        assert spec.task is task, f"{task} row declares task={spec.task}"


def test_decoder_uses_causal_lm_and_a_kv_cache():
    spec = get_task_spec(TaskType.TEXT_GENERATION)
    assert spec.auto_model_class == "transformers.AutoModelForCausalLM"
    assert spec.uses_kv_cache is True
    assert spec.peft_task_type == "CAUSAL_LM"


def test_encoder_uses_feature_extraction_and_no_kv_cache():
    """The regression for the hardcoded `CAUSAL_LM`: an encoder must not be wrapped as a decoder."""
    spec = get_task_spec(TaskType.FEATURE_EXTRACTION)
    assert spec.auto_model_class == "transformers.AutoModel"
    assert spec.uses_kv_cache is False
    assert spec.peft_task_type == "FEATURE_EXTRACTION"


def test_kv_cache_kwargs_are_absent_for_encoders_not_merely_false():
    """A BertOnnxConfig does not accept `use_past` at all — passing `use_past=False` still raises."""
    assert get_task_spec(TaskType.FEATURE_EXTRACTION).onnx_config_kwargs(training_mode=False) == {}
    assert get_task_spec(TaskType.FEATURE_EXTRACTION).onnx_config_kwargs(training_mode=True) == {}


def test_training_graphs_never_use_the_cache():
    """The backward pass needs the full sequence; only the inference export asks for past-KV."""
    decoder = get_task_spec(TaskType.TEXT_GENERATION)
    assert decoder.onnx_config_kwargs(training_mode=True) == {
        "use_past": False,
        "use_past_in_inputs": False,
    }
    assert decoder.onnx_config_kwargs(training_mode=False) == {
        "use_past": True,
        "use_past_in_inputs": True,
    }


@pytest.mark.parametrize(
    "wire",
    ["text-generation", "text-generation-with-past", TaskType.TEXT_GENERATION],
)
def test_wire_strings_resolve_including_the_with_past_variant(wire):
    """`-with-past` selects graph shape, not task identity — TasksManager owns that suffix."""
    assert get_task_spec(wire).task is TaskType.TEXT_GENERATION


def test_unknown_task_fails_closed_naming_the_alternatives():
    with pytest.raises(UnsupportedModelError) as excinfo:
        get_task_spec("image-classification")
    message = str(excinfo.value)
    assert "image-classification" in message
    assert "feature-extraction" in message and "text-generation" in message


# --- #33: sequence classification as a training objective ------------------------------------


def test_sequence_classification_supervises_one_label_per_sequence():
    """The axis that separates this objective from every decoder task."""
    spec = get_task_spec(TaskType.SEQUENCE_CLASSIFICATION)

    assert spec.label_shape == ("batch_size",)
    assert spec.is_token_level is False
    assert get_task_spec(TaskType.TEXT_GENERATION).is_token_level is True


def test_sequence_classification_loads_a_head_and_wraps_it_as_seq_cls():
    spec = get_task_spec(TaskType.SEQUENCE_CLASSIFICATION)
    assert spec.auto_model_class == "transformers.AutoModelForSequenceClassification"
    assert spec.peft_task_type == "SEQ_CLS"
    assert spec.model_init_kwargs == {"num_labels": 2}
    assert spec.uses_kv_cache is False


def test_feature_extraction_is_declared_untrainable():
    """`AutoModel` has no head and no loss, so a training graph is impossible — say so up front.

    Without this the export dies deep inside torch with
    `BertModel.forward() got an unexpected keyword argument 'labels'`.
    """
    assert get_task_spec(TaskType.FEATURE_EXTRACTION).trainable is False
    assert get_task_spec(TaskType.TEXT_GENERATION).trainable is True
    assert get_task_spec(TaskType.SEQUENCE_CLASSIFICATION).trainable is True


def test_classification_keeps_its_head_out_of_quantization():
    """ORT registers no gradient for DynamicQuantizeLinear.

    Anything on the gradient path between the loss and the adapters must stay unquantized. A decoder
    only routes through the LM head; BERT-family classification also routes through `pooler` and
    `classifier`, and quantizing those fails `generate_artifacts` outright.
    """
    excluded = get_task_spec(TaskType.SEQUENCE_CLASSIFICATION).quantization_exclude_layers
    assert "pooler" in excluded and "classifier" in excluded
    assert get_task_spec(TaskType.TEXT_GENERATION).quantization_exclude_layers == ("embed_head",)


def test_each_trainable_task_declares_a_wrapper_and_a_label_shape():
    """A new objective is a row; this is the invariant a row has to satisfy to be usable."""
    for task, spec in TASK_REGISTRY.items():
        if not spec.trainable:
            continue
        assert spec.trainer_wrapper_class, f"{task} declares no trainer wrapper"
        assert spec.label_shape, f"{task} is trainable but supervises no labels"
