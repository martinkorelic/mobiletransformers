"""Normalize a raw Optimum export into repo conventions.

Optimum's ``main_export`` already emits HF-canonical IO names (``input_ids``/``attention_mask``/
``position_ids`` in, ``logits`` + ``present.<i>.key/value`` out, ``past_key_values.<i>.key/value`` in) —
the same scheme ``inference/builder.py``'s ``make_genai_config`` writes into ``genai_config.json``, so
the Native and GenAI engines agree. So normalization here **verifies** those names (fail-closed if a
required output is missing) and **consolidates external data into a single blob** beside ``model.onnx``,
producing the flat package shape plan #9 (unified merger / handoff map) consumes.

``onnx`` is imported lazily so this module stays importable in the core env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mobiletransformers.exceptions import ExportError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: Canonical decoder input names (subset expected present; optimum emits these for text-generation).
CANONICAL_TEXTGEN_INPUTS = ("input_ids", "attention_mask")

#: Known tokenizer / processing artifacts optimum copies next to the graph.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "added_tokens.json",
)


@dataclass
class NormalizedPackage:
    onnx_model: Path
    external_data: Path | None
    io_inputs: tuple[str, ...]
    io_outputs: tuple[str, ...]
    kv_layers: int
    tokenizer_files: tuple[str, ...]
    generation_config: bool


def _count_kv_layers(names: tuple[str, ...], prefix: str) -> int:
    """Count distinct layer indices among ``<prefix>.<i>.key/value`` names."""
    layers: set[str] = set()
    for name in names:
        if name.startswith(prefix + "."):
            parts = name[len(prefix) + 1 :].split(".")
            if parts and parts[0].isdigit():
                layers.add(parts[0])
    return len(layers)


def normalize_package(
    out_dir: str | Path,
    *,
    model_id: str,
    task: str | None = None,
    model_filename: str = "model.onnx",
    external_data_filename: str = "model.onnx_data",
) -> NormalizedPackage:
    """Verify canonical IO and consolidate external data for the export at ``out_dir``.

    Fails closed (``ExportError``) if the ONNX graph is absent, has no ``logits`` output for a
    text-generation task, or is missing the ``present.*`` KV outputs a ``*-with-past`` task requires.
    """
    import onnx

    out_dir = Path(out_dir)
    model_path = out_dir / model_filename
    if not model_path.is_file():
        raise ExportError(f"export produced no {model_filename} in {out_dir} (model={model_id!r})")

    onnx_model = onnx.load(str(model_path), load_external_data=True)
    io_inputs = tuple(i.name for i in onnx_model.graph.input)
    io_outputs = tuple(o.name for o in onnx_model.graph.output)

    task = task or ""
    if task.startswith("text-generation"):
        if "logits" not in io_outputs:
            raise ExportError(
                f"text-generation export missing `logits` output (model={model_id!r}, "
                f"outputs={list(io_outputs)})"
            )
        missing_inputs = [n for n in CANONICAL_TEXTGEN_INPUTS if n not in io_inputs]
        if missing_inputs:
            raise ExportError(
                f"export missing canonical inputs {missing_inputs} (model={model_id!r}, "
                f"inputs={list(io_inputs)})"
            )
        if task.endswith("-with-past") and _count_kv_layers(io_outputs, "present") == 0:
            raise ExportError(
                f"`{task}` export has no `present.*` KV outputs (model={model_id!r}, "
                f"outputs={list(io_outputs)})"
            )
    elif not io_outputs:
        raise ExportError(f"export graph has no outputs (model={model_id!r})")

    kv_layers = _count_kv_layers(io_outputs, "present") or _count_kv_layers(io_inputs, "past_key_values")

    # Consolidate all initializers into a single external blob beside model.onnx (flat #9 shape).
    external_data = out_dir / external_data_filename
    onnx.save_model(
        onnx_model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )
    logger.info(
        "normalized %s: inputs=%s outputs=%s kv_layers=%s", model_id, io_inputs, io_outputs, kv_layers
    )

    tokenizer_files = tuple(f for f in _TOKENIZER_FILES if (out_dir / f).is_file())
    generation_config = (out_dir / "generation_config.json").is_file()

    return NormalizedPackage(
        onnx_model=model_path,
        external_data=external_data if external_data.is_file() else None,
        io_inputs=io_inputs,
        io_outputs=io_outputs,
        kv_layers=kv_layers,
        tokenizer_files=tokenizer_files,
        generation_config=generation_config,
    )


__all__ = ["NormalizedPackage", "normalize_package"]
