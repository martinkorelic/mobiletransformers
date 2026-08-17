"""Unit tests for the train-vs-inference parity check (`artifacts/train_inference_parity.py`).

Only the pure leg — the loss computation and the skip behaviour — runs here. Running either graph needs
`onnxruntime` (and `onnxruntime.training` for the training half), which live in profiles that conflict
with the core one; the wired end-to-end run is the `make device-package TRAIN=1` leg.

The causal-shift tests matter more than they look: a double shift is exactly what made the old
host-side `onnx_checktrain` loss incomparable to the device's, and it is invisible unless asserted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mobiletransformers.artifacts.train_inference_parity import (
    MAX_LOSS_DELTA_NATS,
    ParityResult,
    causal_cross_entropy,
    verify_train_inference_parity,
)


def test_perfect_prediction_gives_zero_loss():
    """A graph that assigns all mass to the true next token scores 0 nats."""
    vocab = 8
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    logits = np.full((1, 4, vocab), -1e4, dtype=np.float32)
    # Position i must predict token i+1.
    for pos, target in enumerate(input_ids[0, 1:]):
        logits[0, pos, target] = 1e4

    assert causal_cross_entropy(logits, input_ids) == pytest.approx(0.0, abs=1e-6)


def test_uniform_logits_give_the_uniform_floor():
    """Uniform prediction is `ln(vocab_size)` — the reference the device test's message cites."""
    vocab = 50
    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    logits = np.zeros((1, 5, vocab), dtype=np.float32)

    assert causal_cross_entropy(logits, input_ids) == pytest.approx(math.log(vocab), abs=1e-6)


def test_the_shift_is_causal_and_applied_once():
    """Predicting token i from position i (no shift) must NOT score zero.

    A double shift is the defect this pins: the loss is defined over `logits[:, :-1]` against
    `input_ids[:, 1:]`, so logits aligned to the *current* token are wrong by exactly one position and
    must be penalised.
    """
    vocab = 8
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)

    aligned_to_current = np.full((1, 4, vocab), -1e4, dtype=np.float32)
    for pos, target in enumerate(input_ids[0]):
        aligned_to_current[0, pos, target] = 1e4

    assert causal_cross_entropy(aligned_to_current, input_ids) > 100.0


def test_loss_is_stable_at_extreme_magnitudes():
    """Observed fp values reached 1e8 on a broken graph; the log-sum-exp must not overflow to nan."""
    vocab = 16
    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    logits = np.full((1, 3, vocab), 1.5e8, dtype=np.float32)

    loss = causal_cross_entropy(logits, input_ids)
    assert math.isfinite(loss)
    assert loss == pytest.approx(math.log(vocab), abs=1e-6)


def test_rejects_non_sequence_logits():
    with pytest.raises(ValueError, match=r"\[batch, seq, vocab\]"):
        causal_cross_entropy(np.zeros((4, 8), dtype=np.float32), np.zeros((4, 8), dtype=np.int64))


def test_parity_result_delta_is_absolute():
    """Either half may be the higher one; the bound is on the magnitude of the disagreement."""
    assert ParityResult(inference_loss=3.0, training_loss=3.4).delta == pytest.approx(0.4)
    assert ParityResult(inference_loss=3.4, training_loss=3.0).delta == pytest.approx(0.4)


def test_quantization_sized_gap_is_inside_the_bound_and_a_broken_graph_is_not():
    """The bound must admit the measured quantization gap and exclude a weights-lost graph.

    0.39 nats is the measured SmolLM2-135M fp32-vs-uint8 gap; a graph that had lost its pretrained
    weights sits at the uniform floor (10.80 for this tokenizer), nats away from a ~3 nat reference.
    """
    assert ParityResult(inference_loss=13.861, training_loss=14.254).delta < MAX_LOSS_DELTA_NATS
    assert ParityResult(inference_loss=3.0, training_loss=10.80).delta > MAX_LOSS_DELTA_NATS


def test_skips_loudly_when_the_training_runtime_is_absent(tmp_path, caplog, monkeypatch):
    """In the core profile the check cannot run — it must say so, never silently pass."""
    import builtins

    real_import = builtins.__import__

    def _no_ort_training(name, *args, **kwargs):
        if name.startswith("onnxruntime"):
            raise ImportError("no onnxruntime in the core profile")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ort_training)

    assert verify_train_inference_parity(tmp_path / "model.onnx", tmp_path / "train") is None
    assert "SKIPPED" in caplog.text
