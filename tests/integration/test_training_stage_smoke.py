"""#15 training-stage smoke (ort-training-local profile): the `gen_artifacts` leg the training stage wires.

Skips unless ``onnxruntime.training`` imports, so it only runs in the ``ort-training-local`` profile
(Python 3.12). This is the automated-under-profile leg of `export/pipeline.py::_build_training_stage`
step 2 — it drives `artifact/onnx_builder.py::gen_artifacts` over the committed tiny fixture and asserts
the four training artifacts + the extended training_config (with the peft_mapping the trainable-split
step consumes). Step 1 (`optimum_hf_export`, a real HF model) and step 3 (`export_inference_package`,
already core-tested) are the manual `make device-package` / #9-package legs.

Run:  uv run --python 3.12 --group ort-training-local --no-default-groups \
          pytest tests/integration/test_training_stage_smoke.py -q
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="ort-training-local profile only")
pytest.importorskip("onnxruntime.training.artifacts", reason="ort-training-local profile only")

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def test_gen_artifacts_produces_training_artifacts_and_extended_config(tmp_path):
    from artifact.onnx_builder import gen_artifacts  # type: ignore[import-not-found]

    # gen_artifacts reads <train_dir>/quant_model.onnx + training_config.json (the shape #15 step 1 writes).
    train_dir = tmp_path / "train_export"
    train_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "tiny_trainable.onnx", train_dir / "quant_model.onnx")
    shutil.copy(FIXTURE_DIR / "training_config.json", train_dir / "training_config.json")

    artifact_dir = tmp_path / "train"
    extended = gen_artifacts(
        train_dir=str(train_dir),
        artifact_dir=str(artifact_dir),
        model_name="quant_model.onnx",
        training_config={},
    )

    for name in ("training_model.onnx", "eval_model.onnx", "optimizer_model.onnx"):
        assert (artifact_dir / name).is_file(), f"missing {name}"
    assert (artifact_dir / "checkpoint").exists()

    # The extended config carries the peft_mapping the trainable split (step 3) consumes.
    assert extended["peft_mapping"] == {"linear.weight": "linear.weight"}
    on_disk = json.loads((artifact_dir / "training_config.json").read_text())
    assert on_disk["peft_mapping"] == extended["peft_mapping"]
