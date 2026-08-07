# DECOMPOSE(#5): this is the single inference-export orchestrator (#9, 01_code_plans/01). It replaces
# the overlapping gen_genai (artifact/onnx_builder.py) + Model.make_genai_config (inference/builder.py)
# export paths with one entry that emits the flat external-data package + weight_handoff_map.json.
# It lives under the legacy `inference/` root (out of the ruff/mypy gate) until inference/ moves into
# src/; it imports the owner contracts from the packaged `mobiletransformers` namespace.
"""Unified inference-export package builder (#9).

Given a normalized inference ``model.onnx`` (as produced by ``mobiletransformers.export`` / #7) plus the
training config that describes which layers were adapted, this produces the device-ready package:

    <output_dir>/
      model.onnx                 # graph; initializers are EXTERNAL refs only
      genai_config.json          # augmented with the external-initializers session entry (if present)
      weight_handoff_map.json    # SINGLE SOURCE OF TRUTH for tensor identity (#8 schema)
      frozen_base.onnx.data      # frozen base tensors — one immutable flat blob, never merged-over
      <trainable>.bin (+ .sha256)# one file per trainable/merge-target tensor (overwritten by merge)

The handoff map is built through #8's ``TrainableTensorCodec`` / ``HandoffMap`` — names are *observed*
from the actual graph initializers, never re-derived — and the required merger ONNX graphs are emitted
via #9's ``build_merger_model``. Both the offline (``artifact/merger.py``) and on-device
(``weight_merger.cpp``) mergers then write to the exact filenames this map records.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import helper, numpy_helper
from onnx.external_data_helper import write_external_data_tensors

from mobiletransformers.artifacts.handoff_map import HandoffMap, ObservedInit, TrainableTensorCodec
from mobiletransformers.config.constants import HandoffMode, PEFTMethod
from mobiletransformers.config.registry.architecture import resolve_architecture
from mobiletransformers.config.registry.merger import build_merger_model, resolve_merger
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import ExportError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_FILENAME = "model.onnx"
GENAI_CONFIG_FILENAME = "genai_config.json"
HANDOFF_MAP_FILENAME = "weight_handoff_map.json"
FROZEN_BASE_BLOB = "frozen_base.onnx.data"

#: The real ORT session-options key that points file/buffer loads at the external-initializer folder.
EXTERNAL_INITIALIZERS_FOLDER_KEY = "session.model_external_initializers_file_folder_path"

#: Reconcile BOTH quantized-tensor vocabularies into the 4 canonical handoff roles: the QDQ /
#: DequantizeLinear naming the merger emits (weight_quantized/weight_scale/weight_zero_point, see
#: weight_merger.cpp::save_merged_parameters) and the GPTQ/int4 naming (qweight/scales/qzeros, see
#: handoff_map.INFERENCE_SUFFIX_TO_ROLE). This single map closes Tier-0 finding #10 — the role of an
#: observed initializer is data here, not a scattered string rule.
_ROLE_BY_TOKEN = {
    "weight": "weight",
    "weight_quantized": "weight_quantized",
    "weight_scale": "scale",
    "weight_zero_point": "zero_point",
    "qweight": "weight_quantized",
    "scales": "scale",
    "qzeros": "zero_point",
}


@dataclass
class ExportedPackage:
    output_dir: Path
    model_path: Path
    handoff_map_path: Path
    frozen_base_blob: Path | None
    trainable_bins: tuple[Path, ...]
    merger_models: dict[str, str]


def _seed_and_token(name: str) -> tuple[str, str]:
    """Split an initializer name into (seed, role-token): ``a.b.MatMul.weight`` -> (``a.b.MatMul``, ``weight``)."""
    seed, _, token = name.rpartition(".")
    return seed, token


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_sidecar(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via temp + fsync + os.replace (never a truncated sidecar)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _convert_initializers_to_inputs(
    model: onnx.ModelProto, initializer_names: set[str], opset_version: int
) -> onnx.ModelProto:
    """Ported from ``gen_genai``'s ``weight_input=True`` path: promote the named initializers to graph
    inputs (GenAI ``set_model_input`` handoff). Reserved for the ``model_input`` fallback mode, which is
    not wired in v1 — kept here so a later plan implements that mode behind the existing enum key."""
    initializers = {init.name: init for init in model.graph.initializer}
    new_inputs = []
    for name in initializer_names:
        init = initializers.get(name)
        if init is not None:
            new_inputs.append(helper.make_tensor_value_info(name, init.data_type, init.dims))
            model.graph.initializer.remove(init)
    new_graph = helper.make_graph(
        nodes=list(model.graph.node),
        name=model.graph.name,
        inputs=list(model.graph.input) + new_inputs,
        outputs=list(model.graph.output),
        initializer=[i for i in model.graph.initializer if i.name not in initializer_names],
    )
    return helper.make_model(new_graph, opset_imports=[helper.make_opsetid("", opset_version)])


def _classify_initializers(
    model: onnx.ModelProto, trainable_seeds: set[str]
) -> tuple[list[ObservedInit], set[str]]:
    """Return (observed trainable inits, set of trainable initializer names).

    An initializer is a *trainable / merge-target* tensor iff its seed matches an adapted layer AND its
    role-token is a recognized weight role. Everything else is frozen base.
    """
    observed: list[ObservedInit] = []
    trainable_names: set[str] = set()
    for init in model.graph.initializer:
        seed, token = _seed_and_token(init.name)
        role = _ROLE_BY_TOKEN.get(token)
        if seed in trainable_seeds and role is not None:
            dtype = str(helper.tensor_dtype_to_np_dtype(init.data_type))
            observed.append(ObservedInit(name=init.name, dtype=dtype, shape=tuple(init.dims), role=role))
            trainable_names.add(init.name)
    return observed, trainable_names


def _split_external_data(
    model: onnx.ModelProto, output_dir: Path, trainable_names: set[str]
) -> tuple[Path | None, list[Path]]:
    """Point each initializer at its external file: trainables at a per-tensor ``<name>.bin``, frozen
    base tensors at the single flat ``frozen_base.onnx.data``. Writes the blobs to ``output_dir``.
    Returns (frozen_base_blob_path_or_None, trainable_bin_paths)."""
    from onnx.external_data_helper import set_external_data

    trainable_bins: list[Path] = []
    wrote_base = False
    for init in model.graph.initializer:
        # set_external_data needs raw_data present.
        if not init.HasField("raw_data"):
            arr = numpy_helper.to_array(init)
            for field_name in (
                "float_data",
                "int32_data",
                "int64_data",
                "string_data",
                "uint64_data",
                "double_data",
            ):
                init.ClearField(field_name)
            init.raw_data = arr.tobytes()
        if init.name in trainable_names:
            location = f"{init.name}.bin"
            trainable_bins.append(output_dir / location)
        else:
            location = FROZEN_BASE_BLOB
            wrote_base = True
        set_external_data(init, location=location)

    write_external_data_tensors(model, str(output_dir))
    frozen_base = output_dir / FROZEN_BASE_BLOB if wrote_base else None
    return frozen_base, trainable_bins


def _emit_merger_models(
    output_dir: Path, peft_method: PEFTMethod, quant_in: bool, quant_out: bool
) -> dict[str, str]:
    """Emit the merger ONNX graph(s) this package needs and return ``{MergerVariant.value: filename}``."""
    spec = resolve_merger(peft_method, quant_in=quant_in, quant_out=quant_out)
    build_merger_model(spec, output_dir / spec.output_filename)
    return {spec.variant.value: spec.output_filename}


def _ensure_session_config_entries(output_dir: Path, runtime_inference_dir: str) -> None:
    """Belt-and-suspenders: add the external-initializers-folder ORT entry to an existing
    ``genai_config.json``. No-op (logged) if the file is not present — full genai_config production is
    owned by the inference builder / one-command CLI (#15), not this phase."""
    path = output_dir / GENAI_CONFIG_FILENAME
    if not path.exists():
        logger.info("no %s to augment (produced upstream); skipping session entry", GENAI_CONFIG_FILENAME)
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    session_options = (
        config.setdefault("model", {}).setdefault("decoder", {}).setdefault("session_options", {})
    )
    entries = session_options.setdefault("config_entries", [])
    if not any(e and e[0] == EXTERNAL_INITIALIZERS_FOLDER_KEY for e in entries):
        entries.append([EXTERNAL_INITIALIZERS_FOLDER_KEY, runtime_inference_dir])
    _atomic_write_sidecar(path, json.dumps(config, indent=2) + "\n")


def export_inference_package(
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    training_config: dict,
    model_config: object,
    peft_method: PEFTMethod,
    quant_in: bool,
    quant_out: bool,
    handoff_mode: HandoffMode = HandoffMode.EXTERNAL_INITIALIZER,
    runtime_inference_dir: str | None = None,
) -> ExportedPackage:
    """Build the unified inference package from a normalized ``model.onnx`` + training config.

    ``training_config`` must carry ``requires_grad`` (list of trainable-name substrings) and
    ``peft_mapping`` (``{base_layer_name: {role: checkpoint_name}}``). ``model_config`` is the HF config
    (or any object with ``architectures``) resolved to the architecture spec for name rewrites.

    Fails closed: only ``external_initializer`` is supported in v1; ``model_input`` and ``adapter`` raise
    ``NotImplementedError`` (F7 stubs behind the existing enum keys).
    """
    if handoff_mode is not HandoffMode.EXTERNAL_INITIALIZER:
        raise NotImplementedError(
            f"handoff mode {handoff_mode.value!r} not supported in v1 "
            f"(only {HandoffMode.EXTERNAL_INITIALIZER.value!r} resolves); "
            "model_input/adapter are reserved for a later plan"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_mapping = training_config.get("peft_mapping") or {}
    requires_grad = training_config.get("requires_grad") or []
    if not peft_mapping:
        raise ExportError("training_config has no peft_mapping; nothing to hand off to the merger")

    arch_spec = resolve_architecture(model_config)
    peft_spec = get_peft_spec(peft_method)

    model = onnx.load(str(model_path), load_external_data=True)

    # Trainable seeds are the adapted layers' inference MatMul seeds (canonical, from #8's codec rule).
    trainable_seeds = {
        TrainableTensorCodec.canonical_inference_name(
            base if base.endswith(".base_layer") else base + ".base_layer", arch_spec
        )
        for base in peft_mapping
    }
    observed, trainable_names = _classify_initializers(model, trainable_seeds)
    if not observed:
        raise ExportError(
            "no inference initializers matched the adapted layers; "
            f"inference/training naming drifted (trainable seeds: {sorted(trainable_seeds)})"
        )

    # #8 codec joins training-side mapping with observed inference inits -> one entry per trainable MatMul.
    entries = TrainableTensorCodec.from_peft_mapping(
        peft_mapping=peft_mapping,
        requires_grad=requires_grad,
        observed_inference_inits=observed,
        peft_spec=peft_spec,
        arch_spec=arch_spec,
    )

    frozen_base, trainable_bins = _split_external_data(model, output_dir, trainable_names)

    # sha256 the bytes actually written for each trainable .bin, and record on the matching role.
    bin_sha: dict[str, str] = {}
    for bin_path in trainable_bins:
        digest = _sha256_file(bin_path)
        bin_sha[bin_path.name] = digest
        _atomic_write_sidecar(bin_path.with_suffix(bin_path.suffix + ".sha256"), digest + "\n")
    for entry in entries:
        for role, location in entry.external_data_location.items():
            if location in bin_sha:
                entry.sha256[role] = bin_sha[location]

    merger_models = _emit_merger_models(output_dir, peft_method, quant_in, quant_out)

    handoff = HandoffMap(entries=entries, handoff_mode=handoff_mode, merger_models=merger_models)
    handoff_map_path = output_dir / HANDOFF_MAP_FILENAME
    handoff.save(handoff_map_path)  # validate() runs inside save(): fails closed on any contract breach

    # Save the graph with external refs only (raw_data was cleared by write_external_data_tensors).
    final_model_path = output_dir / MODEL_FILENAME
    onnx.save(model, str(final_model_path))

    _ensure_session_config_entries(output_dir, runtime_inference_dir or str(output_dir))

    logger.info(
        "exported inference package: %d trainable tensor(s), frozen_base=%s, mergers=%s -> %s",
        len(trainable_bins),
        bool(frozen_base),
        merger_models,
        output_dir,
    )
    return ExportedPackage(
        output_dir=output_dir,
        model_path=final_model_path,
        handoff_map_path=handoff_map_path,
        frozen_base_blob=frozen_base,
        trainable_bins=tuple(trainable_bins),
        merger_models=merger_models,
    )
