"""The requested-vs-realized trainable-tensor gate.

Runs in the core env: the decision lives in `artifacts/trainable_gate.py` precisely so it does not
need the `ort-training-local` profile to be tested (`artifacts/builder.py` imports
`onnxruntime.training` at module scope).

The fixtures below are the REAL names from the exports that exposed this — a BERT encoder and a Llama
decoder under MARS — so the test fails for the same reason the export did.
"""

from __future__ import annotations

import pytest

from mobiletransformers.artifacts.trainable_gate import (
    assert_every_requested_tensor_is_trainable,
    is_quant_companion,
)
from mobiletransformers.exceptions import ExportError

# One MARS-adapted BERT layer: the per-module up-projection plus the block's shared pair.
_UP = "backbone.bert.encoder.layer.0.attention.self.query.up_project.mars.weight"
_SHARED_DOWN = "backbone.bert.encoder.layer.0.attention.self.query.shared_qkv.mars_down_qkv.weight"
_SHARED_MIX = "backbone.bert.encoder.layer.0.attention.self.query.shared_qkv.mars.weight"


def test_quant_companion_suffixes():
    assert is_quant_companion(f"{_SHARED_MIX}_quantized")
    assert is_quant_companion(f"{_SHARED_MIX}_scale")
    assert is_quant_companion(f"{_SHARED_MIX}_zero_point")
    assert not is_quant_companion(_SHARED_MIX)


def test_passes_when_every_requested_tensor_survives():
    requested = [_UP, _SHARED_DOWN, _SHARED_MIX]
    assert_every_requested_tensor_is_trainable(requested, list(requested), frozen=[])


def test_catches_the_defect_it_was_written_for():
    """MARS's shared adapter quantized away, the per-module factor kept — the exact shipped state.

    Before the quantizer learned to exclude declared-trainable tensors, this is what every quantized
    MARS export produced: half the requested set demoted to frozen, no error, training still running,
    loss still falling. 12 realized of 24 requested on an encoder; 4 of 8 on a decoder.
    """
    requested = [_UP, _SHARED_DOWN, _SHARED_MIX]
    realized = [_UP]
    frozen = [
        f"{_SHARED_DOWN}_quantized",
        f"{_SHARED_DOWN}_scale",
        f"{_SHARED_DOWN}_zero_point",
        f"{_SHARED_MIX}_quantized",
        f"{_SHARED_MIX}_scale",
        f"{_SHARED_MIX}_zero_point",
    ]

    with pytest.raises(ExportError) as excinfo:
        assert_every_requested_tensor_is_trainable(requested, realized, frozen)

    message = str(excinfo.value)
    assert "2 of 3" in message
    # It must say WHY, not just that a count differs — the whole failure mode is that quantized
    # means frozen, and nothing else in the pipeline says so.
    assert "QUANTIZED" in message
    assert "shared_qkv" in message, "the error must name a lost tensor"


def test_distinguishes_quantized_away_from_absent():
    """Two different causes need two different fixes, so the message must separate them."""
    requested = [_UP, _SHARED_MIX]
    with pytest.raises(ExportError) as excinfo:
        assert_every_requested_tensor_is_trainable(
            requested, realized=[], frozen=[f"{_SHARED_MIX}_quantized"]
        )
    message = str(excinfo.value)
    assert "QUANTIZED" in message
    assert "absent from the graph entirely" in message


def test_compares_sets_not_counts():
    """A count can coincide while the wrong tensors are frozen — that must still fail."""
    requested = [_UP, _SHARED_DOWN]
    # Same number realized as requested, but it is the same tensor twice and _SHARED_DOWN is gone.
    realized = [_UP, f"{_UP}_duplicate_suffix"]
    with pytest.raises(ExportError, match="1 of 2"):
        assert_every_requested_tensor_is_trainable(requested, realized, frozen=[f"{_SHARED_DOWN}_quantized"])


def test_substring_matching_matches_the_selection_rule():
    """`gen_artifacts` selects by substring, so the gate must judge by the same rule or disagree.

    A requested parameter `…lora_B.lora.weight` is realized as the graph initializer of that name;
    judging by equality would report a false loss for any name the graph decorates.
    """
    requested = ["backbone.model.layers.0.self_attn.q_proj.lora_B.lora.weight"]
    realized = ["prefix/backbone.model.layers.0.self_attn.q_proj.lora_B.lora.weight"]
    assert_every_requested_tensor_is_trainable(requested, realized, frozen=[])
