"""Model-card (README) renderer for a MobileTransformers package (#15, shared with #22 push-back)."""

from __future__ import annotations

from typing import Any


def render_model_card(manifest: dict[str, Any], package_dir: str | None = None) -> str:
    """Render a Hub README from a ``mobiletransformers_manifest.json`` dict.

    Includes the base model, both licenses, version pins, Android runtime requirements, and a variant
    table (id / EP / quant / engines / features / min API / recommended RAM). Pure string building.
    """
    base = manifest.get("baseModelId", "unknown")
    lic = manifest.get("license", {}) or {}
    android = manifest.get("androidRuntime", {}) or {}
    lines: list[str] = []
    lines.append(f"# {base} — MobileTransformers package")
    lines.append("")
    lines.append(f"On-device (Android) package exported from **{base}** with MobileTransformers.")
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"- Base model: `{base}`")
    lines.append(f"- Selected task: `{manifest.get('selectedTask')}`")
    lines.append(f"- PEFT methods: {', '.join(manifest.get('peftMethods', [])) or 'n/a'}")
    lines.append(f"- Quantization: {', '.join(manifest.get('quantization', [])) or 'n/a'}")
    lines.append(
        "- Toolchain: "
        f"optimum-onnx {manifest.get('optimumOnnxVersion')}, "
        f"transformers {manifest.get('transformersVersion')}, "
        f"ort-training {manifest.get('onnxRuntimeTrainingVersion')}, "
        f"ort-genai {manifest.get('onnxRuntimeGenAIVersion')}"
    )
    lines.append("")
    lines.append("## Licenses")
    lines.append(f"- Framework: {lic.get('framework', 'Apache-2.0')}")
    lines.append(f"- Base model weights: {lic.get('baseModelWeights', 'see upstream')}")
    lines.append("")
    lines.append("## Android runtime")
    lines.append(f"- Minimum API: {android.get('minimumAndroidApi')}")
    lines.append(f"- Recommended device memory (MB): {android.get('recommendedDeviceMemoryMb')}")
    lines.append(f"- Required ABIs: {', '.join(android.get('requiredAbis', [])) or 'any'}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| id | EP | quant | engines | features | min API | rec. RAM (MB) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for v in manifest.get("variants", []):
        lines.append(
            f"| {v.get('id')} | {v.get('executionProvider')} | {v.get('quantization')} "
            f"| {', '.join(v.get('supportedEngines', []))} | {', '.join(v.get('features', []))} "
            f"| {v.get('minimumAndroidApi')} | {v.get('recommendedDeviceMemoryMb')} |"
        )
    lines.append("")
    default = manifest.get("defaultVariant")
    lines.append(f"Default variant: `{default}`.")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_model_card"]
