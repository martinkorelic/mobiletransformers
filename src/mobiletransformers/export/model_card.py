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


#: The framework a reader needs in order to do anything with one of these packages.
#:
#: A MobileTransformers package is NOT loadable by `transformers`, `optimum` or plain `onnxruntime`:
#: it is a manifest plus per-variant stages with a weight-handoff map, and the thing that reads it is
#: the Android SDK. Without this link the card describes an artifact with no stated way to run it.
FRAMEWORK_REPOSITORY = "https://github.com/martinkorelic/mobiletransformers"

#: The banner filename as referenced from a published card. It is uploaded ALONGSIDE the README into
#: the model repo rather than hot-linked from GitHub, so the image renders from the moment the repo
#: is published and keeps rendering regardless of the framework repository's visibility, default
#: branch or later reorganisation. A card whose header image 404s looks abandoned.
BANNER_FILENAME = "mobiletransformers_banner.png"

#: Cite the framework, not just the base model. Kept here rather than in a doc so every published
#: card carries it without anyone remembering to paste it.
_CITATION = r"""@misc{mobiletransformers2025,
  author       = {Koreli\v{c}, Martin and Pejovi{\'c}, Veljko},
  title        = {MobileTransformers: An On-Device LLM PEFT Framework for Fine-Tuning and Inference},
  year         = {2025},
  howpublished = {\url{https://gitlab.fri.uni-lj.si/lrk/mobiletransformers}}
}"""


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
    # EVERY declared method, not a hardcoded `lora` test. A MARS package is this project's own
    # research contribution and used to publish with no tag naming it at all, so a reader browsing the
    # org — or anyone filtering the Hub by tag — could not tell a MARS export from a LoRA one.
    for method in manifest.get("peftMethods") or []:
        tags.append(str(method))
    for quant in manifest.get("quantization") or []:
        tags.append(str(quant))
    out.append("tags:")
    out.extend(f"  - {t}" for t in tags)
    out.append("---")
    out.append("")
    return out


#: Human-readable names for the PEFT vocabulary. A card that prints the bare enum value asks its
#: reader to already know what `lora-xs` is; the point of the page is that they do not.
_PEFT_DESCRIPTIONS: dict[str, str] = {
    "lora": "LoRA — low-rank adapters on the attention projections.",
    "lora-xs": "LoRA-XS — LoRA with a frozen SVD basis, training only a small r x r core.",
    "mars": (
        "MARS (Multi-Adapter Rank Sharing) — this project's own method: adapters shared across "
        "layers, so parameter count grows with rank rather than with depth."
    ),
}


def _read_peft_details(package_dir: str | None) -> dict[str, Any]:
    """Rank and adapted modules for the training stage, or ``{}`` when there is no train stage.

    These are **not** in the manifest — it carries `peftMethods` and nothing else about the tuning
    setup. They live beside the training graph, in `train/trainable_parameters.json` (`peftMethod`,
    `rank`) and `train/training_config.json` (`rank`, `peft_target`).

    Best-effort by construction: a package exported for inference only has no train stage, and an
    unreadable file must not fail a push. Every failure path returns ``{}`` and the caller omits the
    lines rather than printing `None` — the mistake that once published "Framework: None" on a live
    model page.
    """
    if not package_dir:
        return {}
    import json
    from pathlib import Path

    root = Path(package_dir)
    details: dict[str, Any] = {}
    # Variant-scoped, and the variant id is not knowable here — glob rather than guess a layout.
    for name, keys in (
        ("trainable_parameters.json", ("peftMethod", "rank")),
        ("training_config.json", ("rank", "peft_target")),
    ):
        for path in sorted(root.glob(f"variants/*/train/{name}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key in keys:
                if payload.get(key) not in (None, "", [], {}):
                    details.setdefault(key, payload[key])
            break
    return details


def _cell(variant: dict[str, Any], key: str) -> str:
    """One variant-table cell, rendering an undeclared value as ``—`` rather than ``None``.

    A table full of the word ``None`` reads as a broken renderer, and leaves the reader unable to tell
    it from a value that is genuinely absent. Every package the exporter has produced has nulls here.
    """
    value = variant.get(key)
    return str(value) if value not in (None, "") else "—"


def render_model_card(
    manifest: dict[str, Any],
    package_dir: str | None = None,
    *,
    banner: str | None = BANNER_FILENAME,
    repo_id: str | None = None,
) -> str:
    """Render a Hub README from a ``mobiletransformers_manifest.json`` dict.

    Includes the base model, both licenses, version pins, Android runtime requirements, and a variant
    table (id / EP / quant / engines / features / min API / recommended RAM). Pure string building.

    ``banner`` is the filename to reference for the header image, or ``None`` to omit it. A parameter
    rather than a constant read from disk because this function does no IO — the caller (``push``)
    owns deciding whether the file will actually be uploaded, and passing a name for an image that is
    not there would render a broken image on a public page.

    ``repo_id`` makes the usage snippets copy-pasteable. It is not in the manifest — a package does
    not know where it will be published — so the publisher supplies it; without it the snippets fall
    back to a placeholder.
    """
    base = manifest.get("baseModelId", "unknown")
    lic = manifest.get("license", {}) or {}
    android = manifest.get("androidRuntime", {}) or {}
    lines: list[str] = []
    lines.extend(_frontmatter(manifest, base, lic))
    if banner:
        lines.append(f"![MobileTransformers]({banner})")
        lines.append("")
    lines.append(f"# {base} — MobileTransformers package")
    lines.append("")
    lines.append(f"On-device (Android) package exported from **{base}** with MobileTransformers.")
    lines.append("")
    # What the package can DO, before how it was made: it decides which of the app's screens light up,
    # and it is the first thing someone choosing between shelf entries needs.
    features: list[str] = []
    for variant in manifest.get("variants", []):
        for feature in variant.get("features", []):
            if feature not in features:
                features.append(str(feature))
    if features:
        lines.append("## What this package can do")
        explanations = {
            "inference": "generate or score on device",
            "train": "**fine-tune on device**, then merge the adapter back into the base weights",
            "rag": "retrieve over documents you ingest, and ground answers in them",
            "core": "shared files every other group needs",
        }
        for feature in features:
            lines.append(f"- `{feature}` — {explanations.get(feature, 'see the docs')}")
        if "train" not in features:
            lines.append("- *(no `train` group: this package is inference-only)*")
        lines.append("")

    peft_methods = [str(m) for m in manifest.get("peftMethods") or []]
    if peft_methods:
        details = _read_peft_details(package_dir)
        lines.append("## Fine-tuning method")
        for method in peft_methods:
            described = _PEFT_DESCRIPTIONS.get(method)
            lines.append(f"- **{method}**{' — ' + described if described else ''}")
        # Omit rather than guess: a rank printed for a package whose train stage says nothing is a
        # number the reader would reasonably trust.
        rank = details.get("rank")
        if rank is not None:
            lines.append(f"- Rank: `{rank}`")
        targets = details.get("peft_target")
        if targets:
            lines.append(f"- Adapted modules: {', '.join(f'`{t}`' for t in targets)}")
        lines.append("")

    lines.append("## Provenance")
    lines.append(f"- Base model: `{base}`")
    lines.append(f"- Selected task: `{manifest.get('selectedTask')}`")
    lines.append(f"- Quantization: {', '.join(manifest.get('quantization', [])) or 'n/a'}")
    # Same rule as the licence lines below: a toolchain entry whose version is null is not evidence
    # that the tool was absent, only that the export did not record it — and "optimum-onnx None"
    # reads as a version literally named None. List the ones that are known; say so when none are.
    toolchain = [
        (label, manifest.get(key))
        for label, key in (
            ("optimum-onnx", "optimumOnnxVersion"),
            ("transformers", "transformersVersion"),
            ("ort-training", "onnxRuntimeTrainingVersion"),
            ("ort-genai", "onnxRuntimeGenAIVersion"),
        )
    ]
    known = [f"{label} {version}" for label, version in toolchain if version]
    lines.append(f"- Toolchain: {', '.join(known) if known else 'not recorded in this package'}")
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
    lines.append(f"- Minimum API: {android.get('minimumAndroidApi') or 'not declared'}")
    memory = android.get("recommendedDeviceMemoryMb")
    # Only when measured. A device-memory recommendation is the number a reader uses to decide whether
    # their phone can run this at all, so inventing one — or printing None — is worse than omitting it.
    if memory:
        lines.append(f"- Recommended device memory (MB): {memory}")
    lines.append(f"- Required ABIs: {', '.join(android.get('requiredAbis', [])) or 'any'}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| id | EP | quant | engines | features | min API | rec. RAM (MB) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for v in manifest.get("variants", []):
        lines.append(
            f"| {_cell(v, 'id')} | {_cell(v, 'executionProvider')} | {_cell(v, 'quantization')} "
            f"| {', '.join(v.get('supportedEngines', [])) or '—'} "
            f"| {', '.join(v.get('features', [])) or '—'} "
            f"| {_cell(v, 'minimumAndroidApi')} | {_cell(v, 'recommendedDeviceMemoryMb')} |"
        )
    lines.append("")
    default = manifest.get("defaultVariant")
    lines.append(f"Default variant: `{default}`.")
    lines.append("")
    lines.append("## Running this model")
    lines.append("")
    lines.append(
        "This is a **MobileTransformers package**, not a plain Hugging Face model: it is a manifest "
        "plus per-variant ONNX stages and a weight-handoff map. `transformers`, `optimum` and plain "
        "`onnxruntime` cannot load it. Use the framework:"
    )
    lines.append("")
    lines.append(f"**{FRAMEWORK_REPOSITORY}**")
    lines.append("")
    lines.append("```kotlin")
    lines.append("// Android — pulls, verifies and installs on first use.")
    lines.append("val model = MobileTransformers.fromPretrained(")
    lines.append("    context = context,")
    lines.append(f'    repoId  = "{repo_id or "<org>/<this-repo>"}",')
    lines.append(")")
    lines.append("```")
    lines.append("")
    lines.append("```bash")
    lines.append("# Host — download and inspect the package without a device.")
    lines.append(f"mobiletransformers pull --repo-id {repo_id or '<org>/<this-repo>'}")
    lines.append("```")
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append("If you are using this framework for your own work, please cite:")
    lines.append("")
    lines.append("```bibtex")
    lines.append(_CITATION)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_model_card"]
