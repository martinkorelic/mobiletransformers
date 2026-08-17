"""What a published model page says about how the package was fine-tuned.

The card used to name the PEFT method in exactly one place — a `PEFT methods: mars` bullet buried in
Provenance — and tag only `lora` in the Hub frontmatter, from a hardcoded string test. So a **MARS**
package, which is this project's own research contribution, published with no tag naming it and
nothing on the page explaining what it is. Someone browsing the org could not tell a MARS export from
a LoRA one, and Hub tag search could not find it at all.

Rank and adapted modules are the other half, and they are **not in the manifest** — they live beside
the training graph in `train/trainable_parameters.json` and `train/training_config.json`. Reading them
is best-effort by construction, which is exactly why it needs tests: every failure path returns `{}`
and the lines are silently omitted, so a mistake here is invisible on a page that still looks complete.
"""

from __future__ import annotations

import json
from pathlib import Path

from mobiletransformers.export.model_card import render_model_card


def _manifest(**overrides) -> dict:
    base = {
        "baseModelId": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "selectedTask": "text-generation-with-past",
        "peftMethods": ["mars"],
        "quantization": ["int4"],
        "license": {"baseModelWeights": "apache-2.0", "framework": None},
        "androidRuntime": {},
        "variants": [{"id": "cpu-int4", "features": ["core", "inference", "train", "rag"]}],
        "defaultVariant": "cpu-int4",
    }
    base.update(overrides)
    return base


def _write_train_stage(root: Path, **payloads) -> None:
    stage = root / "variants" / "cpu-int4" / "train"
    stage.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (stage / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


# --- the frontmatter tag ----------------------------------------------------------------------


def test_every_peft_method_becomes_a_hub_tag():
    """A `mars` package must be findable by tag. The old code tagged `lora` or nothing."""
    card = render_model_card(_manifest(peftMethods=["mars"]))
    frontmatter = card.split("---")[1]
    assert "  - mars" in frontmatter, f"no `mars` tag in frontmatter:\n{frontmatter}"


def test_multiple_methods_are_all_tagged():
    card = render_model_card(_manifest(peftMethods=["lora", "lora-xs"]))
    frontmatter = card.split("---")[1]
    assert "  - lora" in frontmatter
    assert "  - lora-xs" in frontmatter


# --- the named section ------------------------------------------------------------------------


def test_mars_is_named_and_explained_in_its_own_section():
    card = render_model_card(_manifest(peftMethods=["mars"]))
    assert "## Fine-tuning method" in card
    assert "**mars**" in card
    assert "Multi-Adapter Rank Sharing" in card, "MARS is named but never explained"


def test_rank_and_targets_are_read_from_the_train_stage(tmp_path):
    """They are NOT in the manifest — this asserts the package is actually read."""
    _write_train_stage(
        tmp_path,
        trainable_parameters={"peftMethod": "mars", "rank": 16},
        training_config={"rank": 16, "peft_target": ["q_proj", "v_proj"]},
    )
    card = render_model_card(_manifest(), str(tmp_path))
    assert "Rank: `16`" in card
    assert "`q_proj`" in card and "`v_proj`" in card


def test_a_missing_rank_omits_the_line_rather_than_printing_none(tmp_path):
    """`None` rendered as text is worse than silence — the "Framework: None" mistake, repeated."""
    card = render_model_card(_manifest(), str(tmp_path))  # no train stage at all
    assert "## Fine-tuning method" in card, "the method is known even with no train stage"
    assert "Rank:" not in card
    assert "None" not in card.split("## Fine-tuning method")[1].split("##")[0]


def test_an_unreadable_train_stage_does_not_fail_the_card(tmp_path):
    """Best-effort: a corrupt file must not block a push."""
    stage = tmp_path / "variants" / "cpu-int4" / "train"
    stage.mkdir(parents=True)
    (stage / "trainable_parameters.json").write_text("{not json", encoding="utf-8")
    card = render_model_card(_manifest(), str(tmp_path))
    assert "## Fine-tuning method" in card
    assert "Rank:" not in card


def test_no_peft_methods_means_no_section():
    card = render_model_card(_manifest(peftMethods=[]))
    assert "## Fine-tuning method" not in card


# --- the capability section -------------------------------------------------------------------


def test_features_are_listed_so_a_reader_knows_what_the_package_supports():
    card = render_model_card(_manifest())
    assert "## What this package can do" in card
    assert "fine-tune on device" in card
    assert "`rag`" in card


def test_an_inference_only_package_says_so():
    manifest = _manifest(variants=[{"id": "cpu-int4", "features": ["core", "inference"]}])
    card = render_model_card(manifest)
    assert "inference-only" in card


# --- no `None` reaches a published page ---------------------------------------------------------
#
# The "Framework: None" bug shipped on a real model page: `lic.get(k, fallback)` returned None
# because the key EXISTED with a null value, and a reader can only read that as "there is no licence".
# The same shape was still live in three other places — the toolchain line, the device-memory line and
# every variant-table cell — on every package the exporter has produced, since those fields are null
# in all of them.


def test_no_none_reaches_the_page_when_the_manifest_is_full_of_nulls():
    manifest = _manifest(
        optimumOnnxVersion=None,
        transformersVersion=None,
        onnxRuntimeTrainingVersion=None,
        onnxRuntimeGenAIVersion=None,
        androidRuntime={"minimumAndroidApi": None, "recommendedDeviceMemoryMb": None},
        variants=[{"id": "cpu-int4", "features": ["core", "inference"]}],
    )
    card = render_model_card(manifest)
    assert "None" not in card, f"a null rendered as the literal 'None':\n{card}"


def test_a_known_toolchain_version_is_still_shown():
    """The fix must not hide real data — only the nulls."""
    card = render_model_card(_manifest(onnxRuntimeTrainingVersion="1.23.0+cpu"))
    assert "ort-training 1.23.0+cpu" in card


def test_a_recommended_memory_figure_is_shown_when_measured():
    card = render_model_card(_manifest(androidRuntime={"recommendedDeviceMemoryMb": 3000}))
    assert "3000" in card
