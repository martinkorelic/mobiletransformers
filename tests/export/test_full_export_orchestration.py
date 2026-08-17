"""#15 real-export orchestration: stage-gated `_full_export` with injected builders (no heavy deps).

The heavy stage builders (optimum/ORT-training) run only under their profiles; here we inject fakes that
write tiny synthetic stage dirs, and assert the orchestration: stage selection, effective features honest
to what's on disk, GenAI feature gating, a valid #13 manifest, and fail-closed on an unavailable stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.exceptions import ExportError
from mobiletransformers.export.pipeline import (
    StageOutput,
    _default_builders,
    _full_export,
    plan_export,
)


def _fake_inference(with_genai: bool):
    def build(plan, dest, *, token, embedding_model):
        inf = Path(dest) / "inference"
        tok = Path(dest) / "tokenizer"
        inf.mkdir(parents=True, exist_ok=True)
        tok.mkdir(parents=True, exist_ok=True)
        (inf / "model.onnx").write_bytes(b"\x00")
        (inf / "model.onnx_data").write_bytes(b"\x00")
        HandoffMap(entries=[]).save(inf / "weight_handoff_map.json")  # all-frozen base, valid + empty
        if with_genai and "genai" in plan.supported_engines:
            (inf / "genai_config.json").write_text("{}\n")
        (inf / "generation_config.json").write_text("{}\n")
        (tok / "tokenizer.json").write_text("{}\n")
        return StageOutput(
            stage_dirs={"inference": inf, "tokenizer": tok},
            report={"architectures": ["LlamaForCausalLM"], "supportedTasks": [plan.task]},
        )

    return build


def _plan(tmp_path, *, engines):
    return plan_export(
        model="org/tiny",
        output=tmp_path / "pkg",
        quant="int4",
        engines=engines,
        discover=lambda m: "text-generation-with-past",
    )


def _variant_features(pkg):
    manifest = MobileTransformersManifest.load(pkg.manifest_path)
    manifest.validate(pkg.output_dir)  # the #13 checkpoint
    return set(manifest.variants[0]["features"])


def test_inference_only_yields_valid_package_without_train(tmp_path):
    plan = _plan(tmp_path, engines=("native", "genai"))
    builders = {**_default_builders(), "inference": _fake_inference(with_genai=True)}
    pkg = _full_export(plan, token=None, embedding_model=None, stages={"inference"}, builders=builders)

    feats = _variant_features(pkg)
    assert "inference" in feats
    assert "train" not in feats  # nothing on disk claims train
    assert "genai" in feats
    assert (pkg.output_dir / "variants/cpu-int4/inference/model.onnx").is_file()
    assert (pkg.output_dir / "variants/cpu-int4/inference/weight_handoff_map.json").is_file()
    assert (pkg.output_dir / "shared/tokenizer/tokenizer.json").is_file()


def test_genai_feature_requires_both_engine_and_config(tmp_path):
    # genai requested but no genai_config.json emitted -> feature dropped (Native-only).
    plan = _plan(tmp_path, engines=("native", "genai"))
    builders = {**_default_builders(), "inference": _fake_inference(with_genai=False)}
    pkg = _full_export(plan, token=None, embedding_model=None, stages={"inference"}, builders=builders)
    assert "genai" not in _variant_features(pkg)


def test_native_only_engine_never_claims_genai(tmp_path):
    plan = _plan(tmp_path, engines=("native",))
    builders = {**_default_builders(), "inference": _fake_inference(with_genai=True)}
    pkg = _full_export(plan, token=None, embedding_model=None, stages={"inference"}, builders=builders)
    assert "genai" not in _variant_features(pkg)


def test_unavailable_training_stage_fails_closed(tmp_path):
    plan = _plan(tmp_path, engines=("native",))
    with pytest.raises(ExportError, match="training stage"):
        _full_export(plan, token=None, embedding_model=None, stages={"training"})


def test_unknown_stage_rejected(tmp_path):
    from mobiletransformers.exceptions import ConfigValidationError

    plan = _plan(tmp_path, engines=("native",))
    with pytest.raises(ConfigValidationError, match="unknown export stage"):
        _full_export(plan, token=None, embedding_model=None, stages={"bogus"})


# --- provenance across the two-profile export (#15/#33 debt) ----------------


def _fake_training(plan, dest, *, token, embedding_model):
    """A training stage that reports what the real one reports: nothing about the inference graph."""
    train = Path(dest) / "train"
    train.mkdir(parents=True, exist_ok=True)
    (train / "training_config.json").write_text("{}\n")
    (train / "checkpoint").write_bytes(b"\x00")
    return StageOutput(stage_dirs={"train": train}, report={"trainableTensorCount": 4})


def test_training_only_reexport_keeps_the_inference_provenance(tmp_path):
    """A `--stages training` run must not erase what the inference run recorded.

    Producing a train-capable package REQUIRES two profile-scoped runs (the onnxruntime profiles cannot
    co-install), and the second rebuilds the manifest. It used to rebuild it from a report that knows
    nothing about the graph, so every pushed device package carried `transformersVersion: null` — the
    field that attributes a package to a transformers line, which is exactly what diagnosing an export
    regression needs.
    """
    plan = _plan(tmp_path, engines=("native",))
    builders = {**_default_builders(), "inference": _fake_inference(with_genai=False)}
    inference_pkg = _full_export(
        plan, token=None, embedding_model=None, stages={"inference"}, builders=builders
    )
    # The real inference stage writes this side-car next to the graph it describes.
    (inference_pkg.output_dir / "variants/cpu-int4/inference/optimum_config.json").write_text(
        '{"modelId": "org/tiny", "task": "text-generation-with-past", "modelType": "llama",'
        ' "optimumOnnxVersion": "0.1.0", "transformersVersion": "4.46.2", "trustRemoteCode": true}\n'
    )

    pkg = _full_export(
        plan,
        token=None,
        embedding_model=None,
        stages={"training"},
        builders={**_default_builders(), "training": _fake_training},
    )

    manifest = MobileTransformersManifest.load(pkg.manifest_path)
    assert manifest.data["transformersVersion"] == "4.46.2"
    assert manifest.data["optimumOnnxVersion"] == "0.1.0"
    assert manifest.data["architectures"] == ["llama"]
    assert manifest.data["trustRemoteCode"] is True
    # ... and the stage this run DID build is still there.
    assert "train" in set(manifest.variants[0]["features"])


def test_an_inference_run_wins_over_the_recorded_side_car(tmp_path):
    """Carrying forward must never overwrite what THIS run's inference stage reported."""
    plan = _plan(tmp_path, engines=("native",))
    (tmp_path / "pkg/variants/cpu-int4/inference").mkdir(parents=True)
    (tmp_path / "pkg/variants/cpu-int4/inference/optimum_config.json").write_text(
        '{"modelType": "stale-arch", "transformersVersion": "0.0.1"}\n'
    )

    def inference_with_versions(plan, dest, *, token, embedding_model):
        out = _fake_inference(with_genai=False)(plan, dest, token=token, embedding_model=embedding_model)
        out.report.update({"architectures": ["LlamaForCausalLM"], "transformersVersion": "4.46.2"})
        return out

    pkg = _full_export(
        plan,
        token=None,
        embedding_model=None,
        stages={"inference"},
        builders={**_default_builders(), "inference": inference_with_versions},
    )

    manifest = MobileTransformersManifest.load(pkg.manifest_path)
    assert manifest.data["architectures"] == ["LlamaForCausalLM"]
    assert manifest.data["transformersVersion"] == "4.46.2"
