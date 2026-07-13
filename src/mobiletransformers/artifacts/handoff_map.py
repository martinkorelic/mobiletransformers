"""``weight_handoff_map.json`` schema + ``TrainableTensorCodec`` — the ONE source of tensor identity.

This module OWNS the handoff-map contract (schemaVersion 1.0). It replaces the implicit four-way name
agreement between the merge writer (``weight_merger.cpp``), the load side (``session_cache.h``), and the
inference graph (``inference/builder.py``) with one declarative artifact. Every consumer (#9 merger, #13
manifest, #23 native load) reads this shape; none may redefine it.

**Boundary for this cycle (#8):** the Python owner layer — the dataclasses, deterministic
serialization, fail-closed ``validate()``, and the codec join — all pure and fully tested. The
build-side *emit* wiring (accumulating observed initializer names in the inference-graph builder;
feeding ``peft_mapping`` at export time) and the C++/Kotlin consumers ride with their integration plans
(#9 owns the on-device merge/save; #23 the map-driven load), exactly as #6/#7 deferred cross-boundary
consumption. The contract here is what those plans build against.

Consumes #6: adapter role vocabulary from ``PEFTMethodSpec.component_schema`` and the name-rewrite rule
(attention-module name) from ``ArchitectureSpec`` — no naming is hand-rolled here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.versioning import check_compat
from mobiletransformers.config.constants import HandoffMode
from mobiletransformers.exceptions import HandoffError

#: Reader schema version for this contract (see ``check_compat``). Bump only when the reader learns a
#: new schema.
HANDOFF_MAP_READER_VERSION = "1.0"

#: Deterministic role order within an entry (JSON is additionally sort_keys=True for byte-stability).
ROLE_ORDER = ("weight", "weight_quantized", "scale", "zero_point")
_QUANTIZED_ROLES = ("weight_quantized", "scale", "zero_point")

#: Inference-graph initializer suffix -> handoff role. The (deferred) inference-export accumulation
#: uses this to tag each observed initializer; recorded here so both sides share one mapping.
INFERENCE_SUFFIX_TO_ROLE = {
    "weight": "weight",  # fp16/fp32 MatMul weight
    "qweight": "weight_quantized",  # int4 packed weight
    "scales": "scale",
    "qzeros": "zero_point",
}

_DEFAULT_ENGINES = ("native", "genai")


@dataclass(frozen=True)
class TensorSpec:
    """Deterministic description of one trainable tensor role (codec-internal view of an entry)."""

    name: str  # canonical inference name
    dtype: str  # "float16" | "float32" | "int8" | "uint8" | "int4"
    shape: tuple[int, ...]
    role: str  # "weight" | "weight_quantized" | "scale" | "zero_point"
    transpose_policy: str
    aggregation_role: str  # "merged_base_plus_adapter" | "frozen" | "adapter_only"


@dataclass(frozen=True)
class ObservedInit:
    """One initializer as actually emitted by the inference-graph builder (the codec's ground truth).

    The inference-export accumulation (deferred to the inference-builder migration) produces these;
    the codec joins them against the training-side ``peft_mapping`` so names are *observed*, never
    re-derived — a canonical/observed disagreement raises at build time, not silently at runtime."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str
    transposed: bool = False


@dataclass
class HandoffEntry:
    """One trainable MatMul's full identity across train -> merge -> inference."""

    training_base_layer_name: str
    dtype: str
    shape: tuple[int, ...]
    checkpoint_names: dict[str, str] = field(default_factory=dict)
    merger_output_names: dict[str, str] = field(default_factory=dict)
    merged_tensor_names: dict[str, str] = field(default_factory=dict)
    inference_initializer_names: dict[str, str] = field(default_factory=dict)
    external_data_location: dict[str, str] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    genai_input_names: dict[str, str] = field(default_factory=dict)
    quantization: dict[str, str] | None = None
    transpose_policy: str = "no_transpose"

    @property
    def roles(self) -> tuple[str, ...]:
        present = set(self.inference_initializer_names)
        return tuple(r for r in ROLE_ORDER if r in present)

    @property
    def is_quantized(self) -> bool:
        return self.quantization is not None or any(
            r in self.inference_initializer_names for r in _QUANTIZED_ROLES
        )

    def tensor_specs(self) -> list[TensorSpec]:
        """One :class:`TensorSpec` per role, in canonical role order."""
        specs = []
        for role in self.roles:
            specs.append(
                TensorSpec(
                    name=self.inference_initializer_names[role],
                    dtype=self.dtype,
                    shape=self.shape,
                    role=role,
                    transpose_policy=self.transpose_policy,
                    aggregation_role="merged_base_plus_adapter",
                )
            )
        return specs

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trainingBaseLayerName": self.training_base_layer_name,
            "checkpointNames": dict(self.checkpoint_names),
            "mergerOutputNames": dict(self.merger_output_names),
            "mergedTensorNames": dict(self.merged_tensor_names),
            "inferenceInitializerNames": dict(self.inference_initializer_names),
            "externalDataLocation": dict(self.external_data_location),
            "sha256": dict(self.sha256),
            "genaiInputNames": dict(self.genai_input_names),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "transposePolicy": self.transpose_policy,
        }
        if self.quantization is not None:
            out["quantization"] = dict(self.quantization)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffEntry:
        return cls(
            training_base_layer_name=data["trainingBaseLayerName"],
            dtype=data["dtype"],
            shape=tuple(data.get("shape", [])),
            checkpoint_names=dict(data.get("checkpointNames", {})),
            merger_output_names=dict(data.get("mergerOutputNames", {})),
            merged_tensor_names=dict(data.get("mergedTensorNames", {})),
            inference_initializer_names=dict(data.get("inferenceInitializerNames", {})),
            external_data_location=dict(data.get("externalDataLocation", {})),
            sha256=dict(data.get("sha256", {})),
            genai_input_names=dict(data.get("genaiInputNames", {})),
            quantization=dict(data["quantization"]) if data.get("quantization") is not None else None,
            transpose_policy=data.get("transposePolicy", "no_transpose"),
        )


@dataclass
class HandoffMap:
    """The whole ``weight_handoff_map.json`` document."""

    entries: list[HandoffEntry] = field(default_factory=list)
    handoff_mode: HandoffMode = HandoffMode.EXTERNAL_INITIALIZER
    schema_version: str = "1.0"
    min_reader_version: str = "1.0"
    engines: tuple[str, ...] = _DEFAULT_ENGINES
    external_data_layout: str = "one_file_per_tensor"
    frozen_base_blob: str = "frozen_base.onnx.data"
    merger_models: dict[str, str] = field(default_factory=dict)

    def _sorted_entries(self) -> list[HandoffEntry]:
        """Entries sorted by their canonical weight name (byte-deterministic serialization)."""

        def key(e: HandoffEntry) -> str:
            return e.inference_initializer_names.get("weight") or e.training_base_layer_name

        return sorted(self.entries, key=key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "minReaderVersion": self.min_reader_version,
            "handoffMode": self.handoff_mode.value,
            "engines": list(self.engines),
            "externalDataLayout": self.external_data_layout,
            "frozenBaseBlob": self.frozen_base_blob,
            "mergerModels": dict(self.merger_models),
            "entries": [e.to_dict() for e in self._sorted_entries()],
        }

    def to_json(self) -> str:
        """Byte-deterministic JSON (sorted keys + sorted entries) for stable checksums (#13)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffMap:
        return cls(
            entries=[HandoffEntry.from_dict(e) for e in data.get("entries", [])],
            handoff_mode=HandoffMode(data.get("handoffMode", HandoffMode.EXTERNAL_INITIALIZER.value)),
            schema_version=data.get("schemaVersion", "1.0"),
            min_reader_version=data.get("minReaderVersion", "1.0"),
            engines=tuple(data.get("engines", _DEFAULT_ENGINES)),
            external_data_layout=data.get("externalDataLayout", "one_file_per_tensor"),
            frozen_base_blob=data.get("frozenBaseBlob", "frozen_base.onnx.data"),
            merger_models=dict(data.get("mergerModels", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> HandoffMap:
        """Parse + version-gate + validate a map file. Fail closed on any problem."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        check_compat(
            data.get("schemaVersion", "1.0"),
            data.get("minReaderVersion", "1.0"),
            HANDOFF_MAP_READER_VERSION,
        )
        handoff_map = cls.from_dict(data)
        handoff_map.validate()
        return handoff_map

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def validate(self) -> None:
        """Enforce the handoff-map invariants. Raise :class:`HandoffError` naming the offender."""
        check_compat(self.schema_version, self.min_reader_version, HANDOFF_MAP_READER_VERSION)

        # v1 supports only external_initializer; the other modes are fail-closed stubs (F7).
        if self.handoff_mode != HandoffMode.EXTERNAL_INITIALIZER:
            raise HandoffError(
                f"handoffMode {self.handoff_mode.value!r} is not supported in this version "
                "(v1 supports only 'external_initializer')"
            )

        seen_external: dict[str, str] = {}
        seen_inference: dict[str, str] = {}
        for entry in self.entries:
            where = entry.training_base_layer_name

            # external_initializer: the merged name the writer stamps MUST equal the inference name.
            for role, inf_name in entry.inference_initializer_names.items():
                merged = entry.merged_tensor_names.get(role)
                if merged is None:
                    raise HandoffError(f"{where}: role {role!r} missing mergedTensorNames entry")
                if merged != inf_name:
                    raise HandoffError(
                        f"{where}: mergedTensorNames[{role!r}]={merged!r} != "
                        f"inferenceInitializerNames[{role!r}]={inf_name!r} (external_initializer)"
                    )

            # Quantized names MUST come from the observed inference initializers, never base_layer_name.
            if entry.quantization is not None:
                for map_key, role in (
                    ("weightQuantizedName", "weight_quantized"),
                    ("scaleName", "scale"),
                    ("zeroPointName", "zero_point"),
                ):
                    q_name = entry.quantization.get(map_key)
                    obs_name = entry.inference_initializer_names.get(role)
                    if q_name is None or obs_name is None:
                        raise HandoffError(f"{where}: quantized entry missing {map_key!r}/{role!r} name")
                    if q_name != obs_name:
                        raise HandoffError(
                            f"{where}: quantization[{map_key!r}]={q_name!r} must equal the observed "
                            f"inference initializer {obs_name!r} (not derived from base_layer_name)"
                        )

            # No two entries may claim the same external file or the same inference initializer name.
            for loc in entry.external_data_location.values():
                if loc in seen_external:
                    raise HandoffError(
                        f"duplicate externalDataLocation {loc!r} ({where} and {seen_external[loc]})"
                    )
                seen_external[loc] = where
            for inf_name in entry.inference_initializer_names.values():
                if inf_name in seen_inference:
                    raise HandoffError(
                        f"duplicate inferenceInitializerName {inf_name!r} "
                        f"({where} and {seen_inference[inf_name]})"
                    )
                seen_inference[inf_name] = where


class TrainableTensorCodec:
    """Pure (no I/O) builder of :class:`HandoffEntry` objects from the three name sources."""

    @staticmethod
    def canonical_inference_name(base_layer_name: str, arch_spec: Any) -> str:
        """The single Python implementation of the ``weight_merger.cpp:904`` rewrite rules.

        Strip a leading ``backbone.``; rewrite the architecture's attention-module token
        (``arch_spec.attention_module_name``, e.g. ``self_attn``) to ``attn``; rewrite ``base_layer``
        to ``MatMul``. Rules are *data* from #6's architecture registry, not literals baked here. Used
        only to seed a lookup against observed names — the observed name wins, and a mismatch raises.
        """
        name = base_layer_name
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        attn = getattr(arch_spec, "attention_module_name", "self_attn")
        if attn and attn != "attn":
            name = name.replace(f".{attn}.", ".attn.")
        name = name.replace(".base_layer", ".MatMul")
        return name

    @classmethod
    def from_peft_mapping(
        cls,
        peft_mapping: dict[str, dict[str, str]],
        requires_grad: Iterable[str],
        observed_inference_inits: Iterable[ObservedInit],
        peft_spec: Any,
        arch_spec: Any,
    ) -> list[HandoffEntry]:
        """Join training-side (``peft_mapping`` + ``requires_grad``) with inference-side observed
        initializers, one :class:`HandoffEntry` per trainable MatMul.

        The ``peft_spec.component_schema`` supplies the adapter role vocabulary (order-of-truth); the
        ``arch_spec`` supplies the name-rewrite. Raises :class:`HandoffError` if a mapping's
        canonical-derived name has no observed inference initializer (drift caught at build time).
        """
        # Group observed inits by their base (canonical seed) name: "seed.weight" -> "seed".
        by_seed: dict[str, list[ObservedInit]] = {}
        for obs in observed_inference_inits:
            seed = obs.name.rsplit(".", 1)[0]
            by_seed.setdefault(seed, []).append(obs)

        known_roles = {c.role for c in getattr(peft_spec, "component_schema", ())}
        entries: list[HandoffEntry] = []
        for base_layer_name, role_names in peft_mapping.items():
            training_base = (
                base_layer_name
                if base_layer_name.endswith(".base_layer")
                else base_layer_name + ".base_layer"
            )
            seed = cls.canonical_inference_name(training_base, arch_spec)
            group = by_seed.get(seed)
            if not group:
                raise HandoffError(
                    f"no observed inference initializer for {base_layer_name!r} "
                    f"(canonical seed {seed!r}); inference/training naming drifted"
                )

            inference_names = {obs.role: obs.name for obs in group}
            weight_like = next((o for o in group if o.role in ("weight", "weight_quantized")), group[0])

            # checkpointNames = adapter roles from the mapping (validated against the PEFT schema)
            # plus the frozen base weight the merger reads from the CheckpointState.
            checkpoint_names = {
                role: name for role, name in role_names.items() if not known_roles or role in known_roles
            }
            checkpoint_names.setdefault("weight", f"{training_base}.weight")

            quantization = None
            if any(o.role in _QUANTIZED_ROLES for o in group):
                quantization = {
                    "weightQuantizedName": inference_names.get("weight_quantized", ""),
                    "scaleName": inference_names.get("scale", ""),
                    "zeroPointName": inference_names.get("zero_point", ""),
                }

            entries.append(
                HandoffEntry(
                    training_base_layer_name=training_base,
                    dtype=weight_like.dtype,
                    shape=weight_like.shape,
                    checkpoint_names=checkpoint_names,
                    merger_output_names={role: f"merged_{role}" for role in inference_names},
                    merged_tensor_names=dict(inference_names),
                    inference_initializer_names=dict(inference_names),
                    external_data_location={role: f"{name}.bin" for role, name in inference_names.items()},
                    quantization=quantization,
                    transpose_policy=(
                        "already_transposed_for_inference" if weight_like.transposed else "no_transpose"
                    ),
                )
            )
        return entries


__all__ = [
    "HANDOFF_MAP_READER_VERSION",
    "ROLE_ORDER",
    "INFERENCE_SUFFIX_TO_ROLE",
    "TensorSpec",
    "ObservedInit",
    "HandoffEntry",
    "HandoffMap",
    "TrainableTensorCodec",
]
