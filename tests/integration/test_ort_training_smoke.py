"""ORT-training toolchain smoke (Gate 0.3): prove the source-built wheel is *alive*.

Skips unless ``onnxruntime.training`` imports, so it only runs in the ``ort-training-local`` profile
(Python 3.12). Mirrors ``artifact/onnx_builder.py::gen_artifacts``: split initializers into
requires_grad / frozen_params, call ``generate_artifacts`` with AdamW, and assert the four training
artifacts are produced. An extended check loads them via ``Module``/``Optimizer`` and runs one
train step, asserting a finite loss.

Run:  uv run --python 3.12 --group ort-training-local --no-default-groups \
          pytest tests/integration/test_ort_training_smoke.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest

pytest.importorskip("torch", reason="ort-training-local profile only")
ort_artifacts = pytest.importorskip(
    "onnxruntime.training.artifacts", reason="ort-training-local profile only"
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
MODEL_PATH = FIXTURE_DIR / "tiny_trainable.onnx"
CONFIG_PATH = FIXTURE_DIR / "training_config.json"

ARTIFACT_NAMES = ("training_model.onnx", "eval_model.onnx", "optimizer_model.onnx", "checkpoint")


def _split_initializers() -> tuple[list[str], list[str]]:
    """Reproduce gen_artifacts' requires_grad / frozen split from the fixture + config."""
    model = onnx.load(str(MODEL_PATH))
    config = json.loads(CONFIG_PATH.read_text())
    requires_grad, frozen = [], []
    for init in model.graph.initializer:
        if any(sub in init.name for sub in config["requires_grad"]):
            requires_grad.append(init.name)
        else:
            frozen.append(init.name)
    return requires_grad, frozen


def test_generate_artifacts_produces_four_outputs(tmp_path):
    requires_grad, frozen = _split_initializers()
    assert requires_grad, "fixture must have at least one trainable initializer"

    ort_artifacts.generate_artifacts(
        str(MODEL_PATH),
        requires_grad=requires_grad,
        frozen_params=frozen,
        optimizer=ort_artifacts.OptimType.AdamW,
        artifact_directory=str(tmp_path),
    )

    for name in ARTIFACT_NAMES:
        produced = tmp_path / name
        assert produced.exists(), f"generate_artifacts did not produce {name}"


def test_one_train_step_finite_loss(tmp_path):
    from onnxruntime.training.api import CheckpointState, Module, Optimizer

    requires_grad, frozen = _split_initializers()
    ort_artifacts.generate_artifacts(
        str(MODEL_PATH),
        requires_grad=requires_grad,
        frozen_params=frozen,
        optimizer=ort_artifacts.OptimType.AdamW,
        artifact_directory=str(tmp_path),
    )

    state = CheckpointState.load_checkpoint(str(tmp_path / "checkpoint"))
    model = Module(str(tmp_path / "training_model.onnx"), state, str(tmp_path / "eval_model.onnx"))
    optimizer = Optimizer(str(tmp_path / "optimizer_model.onnx"), model)

    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    model.train()
    loss = model(x)
    optimizer.step()
    model.lazy_reset_grad()

    # Module returns a bare scalar (single 0-d output) or a list of outputs — handle both.
    loss_arr = loss[0] if isinstance(loss, (list, tuple)) else loss
    loss_value = float(np.asarray(loss_arr).reshape(-1)[0])
    assert np.isfinite(loss_value), f"loss not finite: {loss_value}"
