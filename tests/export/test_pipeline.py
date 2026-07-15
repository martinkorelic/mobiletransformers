"""#15 one-command export pipeline: arg mapping, dry-run plan, assemble+manifest checkpoint, model card."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.exceptions import ConfigValidationError
from mobiletransformers.export.model_card import render_model_card
from mobiletransformers.export.pipeline import (
    assemble_package,
    export_package,
    manifest_skeleton,
    parse_peft,
    plan_export,
    quant_spec,
)

PKG = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package"


# --- arg mapping ------------------------------------------------------------


@pytest.mark.parametrize(
    "value,method,opt",
    [
        ("lora", PEFTMethod.LORA, None),
        ("lora-xs", PEFTMethod.LORA_XS, None),
        ("mars", PEFTMethod.MARS, 0),
        ("mars-opt1", PEFTMethod.MARS, 1),
        ("mars-opt4", PEFTMethod.MARS, 4),
    ],
)
def test_parse_peft(value, method, opt):
    assert parse_peft(value) == (method, opt)


@pytest.mark.parametrize("bad", ["mars-opt9", "mars-optx", "banana"])
def test_parse_peft_rejects_invalid(bad):
    with pytest.raises(ConfigValidationError):
        parse_peft(bad)


def test_quant_spec_known_and_unknown():
    assert quant_spec("qint8")["weight_type"] == "QInt8"
    assert quant_spec("int4")["weight_type"] == "MatMul4Bits"
    with pytest.raises(ConfigValidationError):
        quant_spec("int2")


# --- dry-run planning (no heavy deps; task injected) ------------------------


def test_plan_and_dry_run(tmp_path):
    plan = export_package(
        model="org/tiny",
        output=tmp_path / "out",
        peft="mars-opt1",
        quant="int4",
        include_rag=True,
        dry_run=True,
        discover=lambda m: "text-generation-with-past",
    )
    assert plan.task == "text-generation-with-past"
    assert plan.peft_method is PEFTMethod.MARS and plan.optimization_level == 1
    assert plan.variant_id == "cpu-int4"
    assert "rag" in plan.features
    # nothing written on dry-run
    assert not (tmp_path / "out").exists()
    skel = manifest_skeleton(plan)
    assert skel["defaultVariant"] == "cpu-int4" and skel["_dryRun"] is True


def test_plan_export_auto_task_uses_injected_discover(tmp_path):
    plan = plan_export(model="org/x", output=tmp_path, discover=lambda m: "feature-extraction")
    assert plan.task == "feature-extraction"


# --- assemble + manifest CHECKPOINT (validates against #13) -----------------


def test_assemble_package_produces_valid_13_package(tmp_path):
    # Reuse the fixture's cpu-int4 variant subtrees as synthetic stage outputs.
    src = PKG / "variants" / "cpu-int4"
    stage_dirs = {
        "inference": src / "inference",
        "train": src / "train",
        "embedding": src / "embedding",
        "tokenizer": PKG / "shared" / "tokenizer",
    }
    plan = plan_export(
        model="org/tiny",
        output=tmp_path / "pkg",
        quant="int4",
        include_rag=True,
        discover=lambda m: "text-generation-with-past",
    )
    report = {
        "mobiletransformersVersion": "0.1.0",
        "architectures": ["LlamaForCausalLM"],
        "supportedTasks": ["text-generation-with-past"],
        "selectedTask": "text-generation-with-past",
        "peftMethods": ["lora"],
        "quantization": ["int4"],
        "androidRuntime": {"minimumAndroidApi": 28, "recommendedDeviceMemoryMb": 3072, "requiredAbis": []},
        "license": {"framework": "Apache-2.0", "baseModelWeights": "Apache-2.0", "noticeFile": None},
    }
    pkg = assemble_package(
        plan, stage_dirs, base_model_id="org/tiny", report=report, exported_at="2026-07-14T00:00:00Z"
    )
    # The emitted tree validates against the #13 manifest validator — this is the export-E2E checkpoint.
    manifest = MobileTransformersManifest.load(pkg.manifest_path)
    manifest.validate(pkg.output_dir)
    assert (pkg.output_dir / "variants/cpu-int4/inference/model.onnx").exists()
    assert (pkg.output_dir / "shared/tokenizer/tokenizer.json").exists()


# --- model card -------------------------------------------------------------


def test_render_model_card_contains_key_sections():
    manifest = json.loads((PKG / "mobiletransformers_manifest.json").read_text())
    card = render_model_card(manifest)
    assert manifest["baseModelId"] in card
    assert "## Licenses" in card and "## Variants" in card
    assert "cpu-int4" in card and "cpu-fp16" in card
