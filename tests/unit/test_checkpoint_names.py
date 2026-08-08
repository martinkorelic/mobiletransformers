"""The export-time merge-contract check (`artifacts/checkpoint_names.py`).

Each test here encodes a defect that reached a device because nothing verified, on the host, that the
names the handoff map publishes are names the checkpoint actually contains.
"""

from __future__ import annotations

import json

import pytest

from mobiletransformers.artifacts.checkpoint_names import (
    checkpoint_weight_param,
    to_checkpoint_name,
    verify_handoff_names_resolve,
    with_base_layer,
)
from mobiletransformers.exceptions import ExportError

RAW = "base_model.model.model.layers.9.self_attn.q_proj"
CHECKPOINT_PARAM = "backbone.model.layers.9.self_attn.q_proj.base_layer.weight"


def _write_map(tmp_path, training_names):
    path = tmp_path / "weight_handoff_map.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"inferenceName": f"t{i}", "trainingBaseLayerName": n}
                    for i, n in enumerate(training_names)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_checkpoint(tmp_path, param_names):
    """A stand-in for the ORT flatbuffer: names appear as plain UTF-8, which is all the check reads."""
    path = tmp_path / "checkpoint"
    path.write_bytes(b"\x00\x11flatbufferish\x00" + b"\x00".join(n.encode() for n in param_names))
    return path


def test_name_derivation_matches_the_cpp_twin():
    assert to_checkpoint_name(RAW) == "backbone.model.layers.9.self_attn.q_proj"
    assert checkpoint_weight_param(RAW) == CHECKPOINT_PARAM


def test_base_layer_suffix_is_idempotent():
    """The map records `trainingBaseLayerName` WITH the suffix; doubling it misses as surely as omitting."""
    once = with_base_layer(RAW)
    assert with_base_layer(once) == once
    assert checkpoint_weight_param(once) == CHECKPOINT_PARAM


def test_passes_when_every_name_resolves(tmp_path):
    handoff = _write_map(tmp_path, [RAW])
    adapter = "backbone.model.layers.9.self_attn.q_proj.lora_A.lora.weight"
    ckpt = _write_checkpoint(tmp_path, [CHECKPOINT_PARAM, adapter])

    assert verify_handoff_names_resolve(handoff, ckpt) == [CHECKPOINT_PARAM]


def test_catches_the_missing_base_layer_defect(tmp_path):
    """The real bug: the merger asked for `<layer>.weight`; peft stores `<layer>.base_layer.weight`.

    A checkpoint holding only the un-suffixed name must fail the check — that package's merge would
    find no base weight for any layer.
    """
    handoff = _write_map(tmp_path, [RAW])
    ckpt = _write_checkpoint(tmp_path, ["backbone.model.layers.9.self_attn.q_proj.weight"])

    with pytest.raises(ExportError, match="does not exist"):
        verify_handoff_names_resolve(handoff, ckpt)


def test_catches_a_wrong_prefix(tmp_path):
    """The other half: a checkpoint keyed with peft's raw wrapper rather than ORT's `backbone.`."""
    handoff = _write_map(tmp_path, [RAW])
    ckpt = _write_checkpoint(tmp_path, [RAW + ".base_layer.weight"])

    with pytest.raises(ExportError, match="name-shape disagreement"):
        verify_handoff_names_resolve(handoff, ckpt)


def test_error_names_the_offenders_and_counts_them(tmp_path):
    handoff = _write_map(tmp_path, [f"base_model.model.model.layers.{i}.self_attn.q_proj" for i in range(9)])
    ckpt = _write_checkpoint(tmp_path, ["unrelated"])

    with pytest.raises(ExportError) as excinfo:
        verify_handoff_names_resolve(handoff, ckpt)
    message = str(excinfo.value)
    assert "9 of 9" in message
    assert "and 4 more" in message  # 5 shown, the rest summarised


def test_empty_map_fails_closed(tmp_path):
    """An empty map means nothing can be merged; publishing it as train-capable is a lie."""
    handoff = _write_map(tmp_path, [])
    ckpt = _write_checkpoint(tmp_path, [CHECKPOINT_PARAM])

    with pytest.raises(ExportError, match="no entries"):
        verify_handoff_names_resolve(handoff, ckpt)


def test_missing_checkpoint_fails_closed(tmp_path):
    handoff = _write_map(tmp_path, [RAW])

    with pytest.raises(ExportError, match="checkpoint not found"):
        verify_handoff_names_resolve(handoff, tmp_path / "absent")


def test_entry_without_a_training_name_is_reported(tmp_path):
    path = tmp_path / "weight_handoff_map.json"
    path.write_text(json.dumps({"entries": [{"inferenceName": "t0"}]}), encoding="utf-8")
    ckpt = _write_checkpoint(tmp_path, [CHECKPOINT_PARAM])

    with pytest.raises(ExportError, match="no trainingBaseLayerName"):
        verify_handoff_names_resolve(path, ckpt)
