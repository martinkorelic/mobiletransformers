"""Model-card (README) renderer for a MobileTransformers package (#15, shared with #22 push-back)."""

from __future__ import annotations

from typing import Any

#: `selectedTask` -> the Hub's `pipeline_tag` vocabulary. The Hub rejects tags outside its own list,
#: and our task names are Optimum's, which overlap but are not identical (`-with-past` is an export
#: detail the Hub has never heard of). Unmapped tasks emit no tag rather than an invalid one.
_PIPELINE_TAG_BY_TASK: dict[str, str] = {
    "text-generation": "text-generation",
    "text-generation-with-past": "text-generation",
    "feature-extraction": "feature-extraction",
    "text-classification": "text-classification",
    "token-classification": "token-classification",
    "fill-mask": "fill-mask",
    "question-answering": "question-answering",
}


def _frontmatter(manifest: dict[str, Any], base: str, lic: dict[str, Any]) -> list[str]:
    """The YAML block the Hub parses for a model page's metadata.

    Without it a published repo renders with no licence, no link back to the base model and no task
    filter — the card body says all three in prose, which the Hub does not read. `base_model` in
    particular is what makes the package show up as a derivative of the model it was exported from,
    which for a repo that ships no original weights is the main thing a reader needs to see.

    Emits only fields whose value is actually known: a `license: null` line is worse than no line,
    because the Hub renders it as a licence literally named "null". The framework licence is
    deliberately NOT defaulted here — it is #32's open decision, and guessing it in published metadata
    would be the loudest possible place to guess wrong.
    """
    out: list[str] = ["---"]
    if base and base != "unknown":
        out.append(f"base_model: {base}")
    out.append("library_name: mobiletransformers")
    tag = _PIPELINE_TAG_BY_TASK.get(str(manifest.get("selectedTask") or ""))
    if tag:
        out.append(f"pipeline_tag: {tag}")
    weights_licence = lic.get("baseModelWeights")
    if weights_licence:
        # The WEIGHTS' licence, not the framework's: this repo redistributes an export of the base
        # model, so the upstream terms are the ones that govern what is in it.
        out.append(f"license: {weights_licence}")
    tags = ["mobiletransformers", "onnx", "on-device", "android"]
    if "lora" in (manifest.get("peftMethods") or []):
        tags.append("lora")
    for quant in manifest.get("quantization") or []:
        tags.append(str(quant))
    out.append("tags:")
    out.extend(f"  - {t}" for t in tags)
    out.append("---")
    out.append("")
    return out


def render_model_card(manifest: dict[str, Any], package_dir: str | None = None) -> str:
    """Render a Hub README from a ``mobiletransformers_manifest.json`` dict.

    Includes the base model, both licenses, version pins, Android runtime requirements, and a variant
    table (id / EP / quant / engines / features / min API / recommended RAM). Pure string building.
    """
    base = manifest.get("baseModelId", "unknown")
    lic = manifest.get("license", {}) or {}
    android = manifest.get("androidRuntime", {}) or {}
    lines: list[str] = []
    lines.extend(_frontmatter(manifest, base, lic))
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
    # `or`, not a dict default: the keys EXIST with a null value on every package the exporter has
    # produced, so `get(k, fallback)` returned None and the published page read "Framework: None" —
    # which a reader can only interpret as "there is no licence".
    lines.append(
        f"- Framework: {lic.get('framework') or 'not declared in this package — see the repository'}"
    )
    lines.append(
        f"- Base model weights: {lic.get('baseModelWeights') or 'see the base model above'} "
        "(this package redistributes an export of those weights, so their terms govern its contents)"
    )
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
