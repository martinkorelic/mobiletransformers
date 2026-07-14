"""Generate the shared tiny MobileTransformers Hub package fixture (#14) at ``tests/fixtures/tiny_package``.

A minimal but *structurally complete* two-variant package: ``cpu-int4`` (native+genai, all features) and
``cpu-fp16`` (native-only, no rag/genai). Placeholder ONNX/blob files are tiny; the real bits are the
directory shape, a valid tiny ``weight_handoff_map.json`` per variant (so #13's resolvability check
passes), and a consistent generated ``mobiletransformers_manifest.json`` + per-variant ``checksums.json``.

Shared by #14 (`build_manifest` round-trip), #13 (validator), and #21 (pull smoke). Regenerate with:
``python tests/fixtures/make_tiny_package.py`` (core env, onnx not required — pure files).
"""

from __future__ import annotations

import json
from pathlib import Path

from mobiletransformers.artifacts.handoff_map import HandoffEntry, HandoffMap
from mobiletransformers.hub.package_format import (
    build_manifest,
    write_manifest,
    write_variant_checksums,
)

FIXTURE_DIR = Path(__file__).parent / "tiny_package"
TRAINABLE = "model.layers.0.attn.q_proj.MatMul.weight"

_REPORT = {
    "mobiletransformersVersion": "0.1.0",
    "architectures": ["LlamaForCausalLM"],
    "supportedTasks": ["text-generation", "text-generation-with-past"],
    "selectedTask": "text-generation-with-past",
    "trustRemoteCode": False,
    "optimumOnnxVersion": "0.1.0",
    "transformersVersion": "4.46.2",
    "onnxRuntimeTrainingVersion": "1.23.0",
    "onnxRuntimeGenAIVersion": "0.14.0",
    "peftMethods": ["lora"],
    "quantization": ["int4", "fp16"],
    "androidRuntime": {
        "minimumAndroidApi": 28,
        "recommendedDeviceMemoryMb": 3072,
        "requiredAbis": ["arm64-v8a"],
    },
    "license": {
        "framework": "Apache-2.0",
        "baseModelWeights": "Apache-2.0",
        "noticeFile": "licenses/BASE_MODEL_LICENSE",
    },
}

_VARIANTS = [
    {
        "id": "cpu-int4",
        "executionProvider": "cpu",
        "quantization": "int4",
        "supportedEngines": ["native", "genai"],
        "abi": ["arm64-v8a"],
        "features": ["core", "inference", "train", "rag", "genai"],
        "minimumAndroidApi": 28,
        "recommendedDeviceMemoryMb": 3072,
    },
    {
        "id": "cpu-fp16",
        "executionProvider": "cpu",
        "quantization": "fp16",
        "supportedEngines": ["native"],
        "abi": None,
        "features": ["core", "inference", "train"],
        "minimumAndroidApi": 28,
        "recommendedDeviceMemoryMb": 6144,
    },
]


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tiny_handoff() -> str:
    entry = HandoffEntry(
        training_base_layer_name="backbone.model.layers.0.self_attn.q_proj.base_layer",
        dtype="float32",
        shape=(4, 3),
        checkpoint_names={"weight": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight"},
        merger_output_names={"weight": "merged_weight"},
        merged_tensor_names={"weight": TRAINABLE},
        inference_initializer_names={"weight": TRAINABLE},
        external_data_location={"weight": f"{TRAINABLE}.bin"},
    )
    return HandoffMap(entries=[entry]).to_json()


def _write_variant_tree(root: Path, variant: dict, *, with_genai: bool, with_rag: bool) -> None:
    vid = variant["id"]
    inf = root / "variants" / vid / "inference"
    _w(inf / "model.onnx", "ONNX_PLACEHOLDER\n")
    _w(inf / "frozen_base.onnx.data", "FROZEN_BASE_PLACEHOLDER\n")
    _w(inf / f"{TRAINABLE}.bin", "TRAINABLE_TENSOR_BYTES\n")
    _w(inf / f"{TRAINABLE}.bin.sha256", "0" * 64 + "\n")
    _w(inf / "weight_handoff_map.json", _tiny_handoff())
    _w(inf / "generation_config.json", json.dumps({"type": "native"}, indent=2) + "\n")
    if with_genai:
        _w(inf / "genai_config.json", json.dumps({"model": {"type": "llama"}}, indent=2) + "\n")

    train = root / "variants" / vid / "train"
    _w(train / "training_config.json", json.dumps({"peftMethod": "lora", "rank": 8}, indent=2) + "\n")
    _w(train / "weight_handoff_map.json", _tiny_handoff())

    if with_rag:
        emb = root / "variants" / vid / "embedding"
        _w(emb / "embedding_model.onnx", "EMBED_ONNX_PLACEHOLDER\n")
        _w(emb / "rag_config.json", json.dumps({"embeddingDimension": 384}, indent=2) + "\n")


def main() -> None:
    root = FIXTURE_DIR
    # shared/
    _w(root / "shared" / "tokenizer" / "tokenizer.json", json.dumps({"version": "1.0"}) + "\n")
    _w(root / "shared" / "chat_template.jinja", "{{ messages }}\n")
    _w(root / "shared" / "config.json", json.dumps({"model_type": "llama"}, indent=2) + "\n")
    _w(root / "shared" / "generation_config.json", json.dumps({"eos_token_id": 2}, indent=2) + "\n")
    # optimum/
    _w(root / "optimum" / "export_report.json", json.dumps({"selectedTask": _REPORT["selectedTask"]}) + "\n")
    _w(root / "optimum" / "supported_tasks.json", json.dumps(_REPORT["supportedTasks"]) + "\n")
    _w(root / "optimum" / "optimum_config.json", json.dumps({"opset": 20}) + "\n")
    # licenses/ + README
    _w(root / "licenses" / "BASE_MODEL_LICENSE", "Apache-2.0\n")
    _w(root / "licenses" / "FRAMEWORK_LICENSE", "Apache-2.0\n")
    _w(root / "README.md", "# Tiny MobileTransformers package fixture\n")
    # variants/
    _write_variant_tree(root, _VARIANTS[0], with_genai=True, with_rag=True)
    _write_variant_tree(root, _VARIANTS[1], with_genai=False, with_rag=False)

    manifest = build_manifest(
        root,
        _VARIANTS,
        base_model_id="MobileTransformers/Tiny-0.1B",
        report=_REPORT,
        default_variant="cpu-int4",
        exported_at="2026-07-14T00:00:00Z",
    )
    write_variant_checksums(root, manifest)
    # Recompute so the manifest's sha256/fileSizes include the just-written checksums.json files.
    manifest = build_manifest(
        root,
        _VARIANTS,
        base_model_id="MobileTransformers/Tiny-0.1B",
        report=_REPORT,
        default_variant="cpu-int4",
        exported_at="2026-07-14T00:00:00Z",
    )
    write_manifest(root, manifest)
    print(f"wrote tiny package fixture to {root} ({len(manifest['sha256'])} files hashed)")


if __name__ == "__main__":
    main()
