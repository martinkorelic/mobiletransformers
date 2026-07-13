"""Fixture well-formedness (onnx-only; runs in the core env — no onnxruntime-training needed)."""

from __future__ import annotations

import json
from pathlib import Path

import onnx

FIXTURE_DIR = Path(__file__).parent
MODEL_PATH = FIXTURE_DIR / "tiny_trainable.onnx"
CONFIG_PATH = FIXTURE_DIR / "training_config.json"


def test_model_is_valid_and_tiny():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size < 1_000_000  # sub-MB so CI stays fast
    onnx.checker.check_model(str(MODEL_PATH), full_check=True)


def test_has_trainable_initializer():
    model = onnx.load(str(MODEL_PATH))
    config = json.loads(CONFIG_PATH.read_text())
    names = [init.name for init in model.graph.initializer]
    # At least one initializer matches a requires_grad substring (the trainable split gen_artifacts does).
    trainable = [n for n in names if any(sub in n for sub in config["requires_grad"])]
    assert trainable, f"no trainable initializer among {names}"


def test_config_has_gen_artifacts_fields():
    config = json.loads(CONFIG_PATH.read_text())
    # Exact fields artifact/onnx_builder.py::gen_artifacts reads.
    for field in (
        "requires_grad",
        "peft_mapping",
        "rank",
        "alpha",
        "peft_target",
        "trainable_parameter_count",
    ):
        assert field in config, f"training_config.json missing {field}"
    assert isinstance(config["requires_grad"], list) and config["requires_grad"]
