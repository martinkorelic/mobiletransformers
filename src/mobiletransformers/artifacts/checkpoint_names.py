"""Export-time proof that the handoff map names parameters the checkpoint actually contains.

## Why this exists

`weight_handoff_map.json` records a `trainingBaseLayerName` per entry. At merge time the on-device
`WeightMerger` turns that name into checkpoint parameter lookups (`<layer>.base_layer.weight`,
`.weight_scale`, …) and asks `Ort::CheckpointState` for them. Nothing ever checked that those lookups
could succeed, so a name-shape disagreement between producer and consumer was undetectable until a
phone ran the merge — and it was undetectable *there* too, for a while, because the merge reported
success having merged nothing.

Three separate defects were this one gap:

* the C++ asked for `<layer>.weight`; peft stores the frozen weight under `<layer>.base_layer.weight`,
  so no layer ever resolved;
* `find_handoff_entry` varied the `.base_layer` suffix but not the `base_model.model.model.` /
  `backbone.model.` prefix, so all 60 merges wrote nothing;
* the quantized roles (`.weight_scale` / `.weight_zero_point`) were derived by the same unchecked rule
  and are equally wrong whenever the non-quantized one is.

Each cost a full export → push → run cycle to find. This check runs on the host in milliseconds and
fails the export instead.

## How it reads the checkpoint

ORT's checkpoint is a flatbuffer; `CheckpointState.load_checkpoint` in ORT 1.23 exposes no iterable
`.parameters`, and requiring `onnxruntime-training` here would make a core-profile export depend on the
heaviest optional profile. Parameter names are stored as plain UTF-8 in the file, so membership is an
exact byte-substring test — no parsing, no schema assumptions, no false "present" answers. The
direction that matters is one-way: we assert names ARE there, never enumerate what is.
"""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.exceptions import ExportError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: Mirrors `cpp/layer_name.h` and `handoff_map._TRAINING_WRAPPER_PREFIXES`. One wire format, three
#: implementations — if this set changes, change it in all three.
RAW_PREFIX = "base_model.model.model."
CHECKPOINT_PREFIX = "backbone.model."
BASE_LAYER_SUFFIX = ".base_layer"

#: The roles `WeightMerger::extract_base_layer_params` looks up. `weight` is required; the quantized
#: companions are only expected when the package is quantized, so their absence is not an error here.
REQUIRED_ROLE = "weight"


def to_checkpoint_name(name: str) -> str:
    """`base_model.model.model.<layer>` -> `backbone.model.<layer>`; twin of `layer_name::to_checkpoint`."""
    if name.startswith(RAW_PREFIX):
        return CHECKPOINT_PREFIX + name[len(RAW_PREFIX) :]
    return name


def with_base_layer(name: str) -> str:
    """Append `.base_layer` unless already present. Idempotent."""
    return name if name.endswith(BASE_LAYER_SUFFIX) else name + BASE_LAYER_SUFFIX


def checkpoint_weight_param(training_base_layer_name: str, role: str = REQUIRED_ROLE) -> str:
    """The exact checkpoint parameter the device merger will ask for."""
    return f"{with_base_layer(to_checkpoint_name(training_base_layer_name))}.{role}"


def verify_handoff_names_resolve(
    handoff_map_path: str | Path,
    checkpoint_path: str | Path,
) -> list[str]:
    """Assert every `trainingBaseLayerName` resolves to a real checkpoint parameter.

    Returns the resolved parameter names (one per entry) so callers can log the count.

    Raises:
        ExportError: naming the first few unresolved parameters. Fails closed: a package whose merge
            cannot possibly work must not be published as if it can.
    """
    import json

    handoff_map_path = Path(handoff_map_path)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ExportError(f"checkpoint not found for handoff-name verification: {checkpoint_path}")

    data = json.loads(handoff_map_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not entries:
        raise ExportError(f"{handoff_map_path} declares no entries; nothing could be merged on device")

    blob = checkpoint_path.read_bytes()

    resolved: list[str] = []
    missing: list[str] = []
    for entry in entries:
        training_name = entry.get("trainingBaseLayerName")
        if not training_name:
            missing.append(f"<entry {entry.get('inferenceName', '?')} has no trainingBaseLayerName>")
            continue
        param = checkpoint_weight_param(training_name)
        if param.encode("utf-8") in blob:
            resolved.append(param)
        else:
            missing.append(param)

    if missing:
        shown = "\n  ".join(missing[:5])
        more = f"\n  … and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ExportError(
            f"{len(missing)} of {len(entries)} handoff entries name a checkpoint parameter that does "
            f"not exist in {checkpoint_path.name}. The on-device merge would find no base weight for "
            f"these layers:\n  {shown}{more}\n"
            "This is a producer/consumer name-shape disagreement — reconcile "
            "artifacts/checkpoint_names.py, cpp/layer_name.h and artifacts/handoff_map.py, which must "
            "all describe the same wire format."
        )

    logger.info(
        "handoff-name check: %d/%d trainingBaseLayerName(s) resolve to real checkpoint parameters",
        len(resolved),
        len(entries),
    )
    return resolved
