"""Adapter model-card renderer (#22) — wraps #15's ``render_model_card`` with the mandatory adapter
disclosures: a bold privacy warning, the exact upstream base-model license, PEFT/MARS details, and
re-apply instructions. ``assert_required_sections`` fails closed before any upload."""

from __future__ import annotations

from mobiletransformers.adapter.export import AdapterPackage
from mobiletransformers.config.constants import PEFTMethod

PRIVACY_WARNING = (
    "**⚠️ Privacy warning:** this adapter was fine-tuned on-device and its weights may encode private "
    "user data. Uploading it is a deliberate act of publication — do not push adapters trained on data "
    "you would not publish."
)


def render_adapter_card(pkg: AdapterPackage, *, mode: str, base_model_license: str = "see upstream") -> str:
    """Render the adapter README. ``mode`` is ``"peft"`` (Mode 1) or ``"native"`` (Mode 2)."""
    lines: list[str] = []
    lines.append(f"# {pkg.base_model_id} — MobileTransformers adapter")
    lines.append("")
    lines.append(PRIVACY_WARNING)
    lines.append("")
    lines.append("## Adapter")
    lines.append(f"- Base model: `{pkg.base_model_id}`")
    lines.append(
        f"- PEFT method: **{pkg.peft_method}**"
        + (
            f" (MARS optimization level {pkg.mars_optimization_level})"
            if pkg.peft_method == PEFTMethod.MARS.value
            else ""
        )
    )
    lines.append(f"- Rank: {pkg.rank}    Alpha: {pkg.alpha}")
    lines.append(f"- Target modules: {', '.join(pkg.peft_target) or 'n/a'}")
    lines.append(f"- Trainable parameters: {pkg.trainable_parameter_count}")
    lines.append(f"- Handoff mode: `{pkg.handoff_mode}`")
    lines.append("")
    lines.append("## Licenses")
    lines.append("- Framework: Apache-2.0")
    lines.append(f"- Base model weights: {base_model_license}")
    lines.append("")
    lines.append("## Re-apply")
    if mode == "peft":
        lines.append(
            "This is a standard PEFT LoRA adapter — load it with "
            "`PeftModel.from_pretrained(base_model, <this_repo>)`."
        )
    else:
        lines.append(
            "This is a **MobileTransformers-native** adapter (not a drop-in PEFT adapter): it ships the "
            "merged per-tensor external initializers + `weight_handoff_map.json`. Re-apply it through "
            "MobileTransformers (install the package, load via the native runtime)."
        )
    lines.append("")
    return "\n".join(lines)


def assert_required_sections(card: str, pkg: AdapterPackage) -> None:
    """Fail closed unless the mandatory disclosures are present in ``card``."""
    missing: list[str] = []
    if "Privacy warning" not in card:
        missing.append("privacy warning")
    if "## Licenses" not in card:
        missing.append("licenses section")
    if pkg.peft_method and pkg.peft_method not in card:
        missing.append("peft method")
    if "Rank:" not in card:
        missing.append("rank/alpha")
    if missing:
        from mobiletransformers.exceptions import ExportError

        raise ExportError(f"adapter model card missing mandatory sections: {', '.join(missing)}")


__all__ = ["PRIVACY_WARNING", "render_adapter_card", "assert_required_sections"]
