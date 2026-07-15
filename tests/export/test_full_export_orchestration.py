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
