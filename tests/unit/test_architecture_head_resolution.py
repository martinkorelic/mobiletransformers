"""The architecture registry keys on the loaded class, not the checkpoint's declared architecture (#33).

`config.architectures` describes the **checkpoint**. A sentence-transformers encoder declares
`["BertModel"]` even when loaded through `AutoModelForSequenceClassification` as a
`BertForSequenceClassification` — so resolving from the config alone sends an encoder fine-tune to the
un-headed, untrainable row. The head is part of the architecture identity.
"""

from __future__ import annotations

import pytest

from mobiletransformers.config.constants import TaskType
from mobiletransformers.config.registry.architecture import (
    ARCHITECTURE_REGISTRY,
    resolve_architecture,
)
from mobiletransformers.exceptions import UnsupportedModelError


class _Config:
    """Stands in for a HF config; only `architectures` is read."""

    def __init__(self, architectures):
        self.architectures = architectures


def test_config_lookup_is_unchanged_when_no_override_is_given():
    assert resolve_architecture(_Config(["LlamaForCausalLM"])).architecture == "LlamaForCausalLM"


def test_loaded_class_overrides_the_checkpoints_declared_architecture():
    """The #33 case: same config, different head, different row."""
    config = _Config(["BertModel"])

    assert resolve_architecture(config).task is TaskType.FEATURE_EXTRACTION
    resolved = resolve_architecture(config, architecture="BertForSequenceClassification")
    assert resolved.architecture == "BertForSequenceClassification"
    assert resolved.task is TaskType.SEQUENCE_CLASSIFICATION


def test_override_still_fails_closed_on_an_unknown_head():
    with pytest.raises(UnsupportedModelError, match="ElectraForSequenceClassification"):
        resolve_architecture(_Config(["BertModel"]), architecture="ElectraForSequenceClassification")


def test_decoder_rows_agree_with_what_automodel_would_load():
    """For every decoder the loaded class name IS `architectures[0]`, so the override is a no-op.

    This is what makes passing `type(model).__name__` strictly more accurate rather than a behaviour
    change for the paths that already worked.
    """
    for name, spec in ARCHITECTURE_REGISTRY.items():
        if spec.task is not TaskType.TEXT_GENERATION:
            continue
        assert resolve_architecture(_Config([name]), architecture=name) is spec


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("BertForSequenceClassification", ("query", "value")),
        ("RobertaForSequenceClassification", ("query", "value")),
        # DistilBERT names its projections differently — the per-architecture difference the registry
        # exists to hold as data rather than as a branch.
        ("DistilBertForSequenceClassification", ("q_lin", "v_lin")),
    ],
)
def test_encoder_rows_carry_their_own_projection_names(architecture, expected):
    spec = ARCHITECTURE_REGISTRY[architecture]
    assert spec.target_modules == expected
    assert spec.attention_module_name == "attention"
    assert spec.task is TaskType.SEQUENCE_CLASSIFICATION
