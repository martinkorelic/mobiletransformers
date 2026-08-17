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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.checkpoint_names import to_checkpoint_name
from mobiletransformers.artifacts.versioning import check_compat
from mobiletransformers.config.constants import HandoffMode
from mobiletransformers.config.registry.architecture import DEFAULT_ATTENTION_MODULE_NAME
from mobiletransformers.exceptions import HandoffError

#: Reader schema version for this contract (see ``check_compat``). Bump only when the reader learns a
#: new schema.
#: What THIS reader understands. Must be >= any map's `minReaderVersion`; a map at a higher
#: MINOR version than this still loads (additive fields are ignored).
HANDOFF_MAP_READER_VERSION = "1.1"

#: Deterministic role order within an entry (JSON is additionally sort_keys=True for byte-stability).
ROLE_ORDER = ("weight", "weight_quantized", "scale", "zero_point")
_QUANTIZED_ROLES = ("weight_quantized", "scale", "zero_point")

#: The on-disk inference weight is stored in the SAME orientation the merger computes in.
NO_TRANSPOSE = "no_transpose"
#: The on-disk inference weight is the TRANSPOSE of the merger's orientation, so a merged tensor must
#: be transposed before it is written back.
ALREADY_TRANSPOSED = "already_transposed_for_inference"


def derive_transpose_policy(
    weight_shape: tuple[int, ...],
    adapter_shapes: dict[str, tuple[int, ...]],
) -> str:
    """Which orientation the on-disk inference weight is in, relative to the merger's.

    **This replaces a field that was never assigned.** ``ObservedInit.transposed`` defaulted to
    ``False`` and nothing in the codebase ever set it, so every package ever produced declared
    ``no_transpose`` by omission — including packages where it was demonstrably wrong. The merge
    honoured that value and wrote every weight transposed, undetected by four independent gates —
    a norm, a sum, a byte count and a checksum are all transpose-invariant, so none of them can
    detect a permutation of elements.

    The orientation is *observable*, so it is observed rather than declared. The merger computes
    ``base + scale * (adapter_B @ adapter_A)``, whose shape is ``(B.rows, A.cols)``. If that equals the
    on-disk weight shape the two orientations agree; if it equals its reverse, the on-disk tensor is
    the transpose.

    Square weights are genuinely ambiguous from shape alone — ``[576,576]`` satisfies both — which is
    exactly why the defect survived: the adapted decoder layers are ``q_proj`` (square) and ``v_proj``
    (not). Callers should resolve a package-wide policy from the entries that *can* be decided; see
    :func:`resolve_package_transpose_policy`.
    """
    # The down-projection is `adapter_A` under LoRA and `shared_A` under MARS — MARS shares one A
    # across the layers of a block, which is what the method IS, so the name differs by design.
    #
    # Reading only `adapter_A` made every MARS layer take the fail-open branch below and declare
    # `no_transpose` for an orientation that is fully observable. That is the same shape as the defect
    # this function was written to kill: a value nobody computed, honoured by a consumer. Caught by
    # `test_the_derivation_agrees_with_a_real_exported_package` on the first MARS package ever
    # exported (2026-08-17, gemma-3-270m-it) — the test reads a real artifact for exactly this reason.
    a = adapter_shapes.get("adapter_A") or adapter_shapes.get("shared_A")
    b = adapter_shapes.get("adapter_B")

    if b and not a:
        # B described but A under neither known name: a THIRD convention this function does not
        # understand. Refuse rather than default — silently returning `no_transpose` here is precisely
        # how every merged weight came to be written transposed.
        raise ValueError(
            f"adapter shapes {sorted(adapter_shapes)} describe an up-projection but no "
            "down-projection under `adapter_A` or `shared_A`, so orientation cannot be observed. "
            "Teach this function the new name rather than letting it guess."
        )

    if not a or not b or len(a) != 2 or len(b) != 2 or len(weight_shape) != 2:
        # Nothing to observe (no adapters described, or not a 2-D weight): keep the historical value
        # rather than inventing one.
        return NO_TRANSPOSE
    delta = (b[0], a[1])
    if tuple(weight_shape) == delta:
        return NO_TRANSPOSE
    if tuple(weight_shape) == delta[::-1]:
        return ALREADY_TRANSPOSED
    raise ValueError(
        f"adapter factors {b} @ {a} produce a {delta} delta, which is neither the on-disk weight "
        f"shape {tuple(weight_shape)} nor its transpose. The merge would add tensors that cannot be "
        f"added; refusing to describe this layer."
    )


def resolve_package_transpose_policy(entries: Sequence[HandoffEntry]) -> str:
    """One orientation for the whole package, decided by the entries that are not square.

    A single export uses one convention throughout, so an unambiguous layer settles it for the
    ambiguous (square) ones. Fails closed when non-square layers disagree with each other — that would
    mean the package mixes conventions, which no consumer could honour.
    """
    decided = {
        e.transpose_policy
        for e in entries
        if len(e.shape) == 2 and e.shape[0] != e.shape[1] and e.adapter_shapes
    }
    if len(decided) > 1:
        raise ValueError(
            f"handoff entries disagree about weight orientation ({sorted(decided)}); a package must "
            "use one convention throughout"
        )
    return decided.pop() if decided else NO_TRANSPOSE


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

    # NOTE: a `transposed: bool = False` field used to live here and was the sole input to
    # `transposePolicy`. **Nothing in the codebase ever assigned it**, so every package ever produced
    # declared `no_transpose` by omission, the on-device merge honoured that, and every merged weight
    # was written transposed (2026-08-14; a magnitude-based check cannot
    # detect a permutation"). It is deliberately NOT re-added: the orientation is observable from the
    # adapter and weight shapes, so `derive_transpose_policy` observes it instead of trusting a flag
    # someone must remember to set.


@dataclass
class HandoffEntry:
    """One trainable MatMul's full identity across train -> merge -> inference."""

    training_base_layer_name: str
    #: Entry-level dtype/shape: the *weight-like* role's. Kept for compatibility and as the fallback
    #: for readers that predate ``tensorDtypes``/``tensorShapes``.
    dtype: str
    shape: tuple[int, ...]
    #: Per-role on-disk dtype/shape, one pair per role in ``external_data_location``. Required by the
    #: device loader: each ``<name>.bin`` holds RAW external-data bytes with no header, so the reader
    #: has no other way to learn the element type and shape of a packed ``weight_quantized`` / ``scale``
    #: / ``zero_point`` tensor, whose layout differs from the entry-level weight's.
    tensor_dtypes: dict[str, str] = field(default_factory=dict)
    tensor_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    checkpoint_names: dict[str, str] = field(default_factory=dict)
    #: Per-adapter-role dtype/shape of the TRAINING-side factors (`adapter_A`, `adapter_B`,
    #: `shared_A`, `intermediate`), read from the training graph's initializers at artifact time.
    #:
    #: `checkpoint_names` already NAMED these; nothing described them, so the map was the single
    #: source of tensor identity for the merged inference initializers only. A consumer exchanging the
    #: factors themselves (#35 rank-r federation, #36's Kotlin codec) had to infer shapes from the
    #: rank. ADDITIVE: absent on packages exported before this, and readers tolerate unknown fields by
    #: the canonical rule, so this is a minor `schemaVersion` bump rather than a breaking one.
    adapter_dtypes: dict[str, str] = field(default_factory=dict)
    adapter_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
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

    def dtype_for(self, role: str) -> str:
        """On-disk dtype of ``role``'s ``.bin``, falling back to the entry-level weight dtype."""
        return self.tensor_dtypes.get(role, self.dtype)

    def shape_for(self, role: str) -> tuple[int, ...]:
        """On-disk shape of ``role``'s ``.bin``, falling back to the entry-level weight shape."""
        return self.tensor_shapes.get(role, self.shape)

    def tensor_specs(self) -> list[TensorSpec]:
        """One :class:`TensorSpec` per role, in canonical role order."""
        specs = []
        for role in self.roles:
            specs.append(
                TensorSpec(
                    name=self.inference_initializer_names[role],
                    # Per-role, not the entry-level weight's: a scale/zero_point tensor has its own
                    # dtype and shape, and reporting the weight's here was silently wrong.
                    dtype=self.dtype_for(role),
                    shape=self.shape_for(role),
                    role=role,
                    transpose_policy=self.transpose_policy,
                    aggregation_role="merged_base_plus_adapter",
                )
            )
        return specs

    #: Adapter roles in a canonical, deterministic order. Federated exchange serializes in this order
    #: within each entry, exactly as the merged path serializes in `ROLE_ORDER`.
    ADAPTER_ROLE_ORDER: tuple[str, ...] = ("shared_A", "intermediate", "adapter_A", "adapter_B")

    def adapter_tensor_specs(self) -> list[TensorSpec]:
        """One :class:`TensorSpec` per ADAPTER factor, in canonical adapter-role order.

        The rank-r counterpart of :meth:`tensor_specs`. That one describes the MERGED inference
        initializer (one full-size weight per adapted layer); this one describes the factors the
        optimizer actually updates (`lora_A` + `lora_B`, or MARS's shared pair), which is what
        federation exchanges as of #35's vocabulary decision.

        Empty when the map predates the adapter-identity fields — the caller decides whether that is
        fatal, because silently falling back to the merged specs is precisely the ambiguity that made
        the two vocabularies collide in the first place.
        """
        specs = []
        for role in self.ADAPTER_ROLE_ORDER:
            if role not in self.adapter_dtypes or role not in self.adapter_shapes:
                continue
            specs.append(
                TensorSpec(
                    # The ORT CHECKPOINT PARAMETER name, not the raw PEFT module path: this is the
                    # identity a client looks the tensor up by, and #36's Kotlin codec mirrors it.
                    # `to_checkpoint_name` is the existing normalizer (twin of `cpp/layer_name.h`).
                    name=f"{to_checkpoint_name(self.checkpoint_names[role])}.weight",
                    dtype=self.adapter_dtypes[role],
                    shape=self.adapter_shapes[role],
                    role=role,
                    transpose_policy=self.transpose_policy,
                    aggregation_role="adapter_only",
                )
            )
        return specs

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trainingBaseLayerName": self.training_base_layer_name,
            "checkpointNames": dict(self.checkpoint_names),
            "adapterDtypes": dict(self.adapter_dtypes),
            "adapterShapes": {r: list(sh) for r, sh in self.adapter_shapes.items()},
            "mergerOutputNames": dict(self.merger_output_names),
            "mergedTensorNames": dict(self.merged_tensor_names),
            "inferenceInitializerNames": dict(self.inference_initializer_names),
            "externalDataLocation": dict(self.external_data_location),
            "sha256": dict(self.sha256),
            "genaiInputNames": dict(self.genai_input_names),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "tensorDtypes": dict(self.tensor_dtypes),
            "tensorShapes": {role: list(shape) for role, shape in self.tensor_shapes.items()},
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
            tensor_dtypes=dict(data.get("tensorDtypes", {})),
            tensor_shapes={role: tuple(shape) for role, shape in data.get("tensorShapes", {}).items()},
            checkpoint_names=dict(data.get("checkpointNames", {})),
            # Absent on packages exported before the adapter-identity fields existed; an empty dict
            # simply means "this map cannot describe the factors", which readers must handle.
            adapter_dtypes=dict(data.get("adapterDtypes", {})),
            adapter_shapes={role: tuple(shape) for role, shape in data.get("adapterShapes", {}).items()},
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
    #: 1.1 adds `adapterDtypes`/`adapterShapes` per entry — purely ADDITIVE, so this is a MINOR bump
    #: and `min_reader_version` deliberately stays 1.0: a 1.0 reader ignores the new fields by the
    #: canonical unknown-fields rule and keeps working, and packages written at 1.0 still load.
    schema_version: str = "1.1"
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

            # Every role with a .bin must declare its OWN on-disk dtype/shape: the device loader reads
            # raw external-data bytes (no TensorProto header), so a missing pair leaves the tensor
            # unloadable. Entries predating tensorDtypes/tensorShapes carry neither and fall back to
            # the entry-level pair — sound only for a single non-quantized role, hence the extra gate.
            if entry.tensor_dtypes or entry.tensor_shapes:
                for role in entry.external_data_location:
                    if role not in entry.tensor_dtypes:
                        raise HandoffError(f"{where}: role {role!r} missing tensorDtypes entry")
                    if role not in entry.tensor_shapes:
                        raise HandoffError(f"{where}: role {role!r} missing tensorShapes entry")
            elif entry.quantization is not None:
                raise HandoffError(
                    f"{where}: quantized entry must declare per-role tensorDtypes/tensorShapes "
                    "(the entry-level dtype/shape describes only the weight-like role)"
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


#: Wrapper prefixes the training stack puts in front of the model's own module path. `OnnxTrainerWrapper`
#: contributes ``backbone.``; peft's ``get_peft_model`` contributes ``base_model.model.`` — so a LoRA'd
#: SmolLM2 layer arrives as ``base_model.model.model.layers.0.self_attn.q_proj``. The inference graph
#: knows nothing of either, so they must come off before the two sides can be compared.
_TRAINING_WRAPPER_PREFIXES = ("backbone.", "base_model.model.", "base_model.")


def _strip_wrapper_prefixes(name: str) -> str:
    """Reduce a training-side parameter path to the model's own module path.

    Applied repeatedly because the wrappers nest (``backbone.base_model.model.…``). Order matters:
    ``base_model.model.`` is tried before ``base_model.`` so the longer wrapper wins.
    """
    changed = True
    while changed:
        changed = False
        for prefix in _TRAINING_WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


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
        return TrainableTensorCodec.candidate_inference_names(base_layer_name, arch_spec)[0]

    @staticmethod
    def candidate_inference_names(base_layer_name: str, arch_spec: Any) -> tuple[str, ...]:
        """Every spelling an adapted layer's inference MatMul seed may legitimately take.

        Two inference exporters are in play and they name the attention module differently:

        * the legacy ``inference/builder.py`` graphs use ``attn`` — which is what the
          ``weight_merger.cpp:904`` rewrite (and :meth:`canonical_inference_name`) was written against;
        * the Optimum export that #7 made the front door preserves HF-canonical ``self_attn``.

        Seeding the lookup with only the rewritten spelling meant **no** trainable tensor in an
        Optimum-produced package could be matched, so `export_inference_package` failed with
        "inference/training naming drifted" and no handoff map could be built for the very packages the
        project now ships. The seed is only a lookup key — the *observed* initializer name is what is
        recorded and what the device reads back out of `inferenceInitializerNames` — so accepting both
        spellings is safe and keeps the C++ mirror valid for legacy packages.

        Ordered: the rewritten (legacy) spelling first, so :meth:`canonical_inference_name` and the C++
        mirror keep their existing meaning.
        """
        name = _strip_wrapper_prefixes(base_layer_name)
        name = name.replace(".base_layer", ".MatMul")

        # Falls back to the registry's declared default rather than a literal spelled here —
        # `arch_spec` is `Any` and legacy callers pass None.
        attn = getattr(arch_spec, "attention_module_name", None) or DEFAULT_ATTENTION_MODULE_NAME
        candidates = []
        if attn and attn != "attn":
            candidates.append(name.replace(f".{attn}.", ".attn."))
        candidates.append(name)
        # Preserve order, drop duplicates (an architecture whose module already *is* `attn`).
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def from_peft_mapping(
        cls,
        peft_mapping: dict[str, dict[str, str]],
        requires_grad: Iterable[str],
        observed_inference_inits: Iterable[ObservedInit],
        peft_spec: Any,
        arch_spec: Any,
        trainable_tensor_specs: dict[str, dict[str, Any]] | None = None,
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
            seeds = cls.candidate_inference_names(training_base, arch_spec)
            group = next((g for g in (by_seed.get(s) for s in seeds) if g), None)
            if not group:
                raise HandoffError(
                    f"no observed inference initializer for {base_layer_name!r} "
                    f"(tried seeds {list(seeds)}); inference/training naming drifted"
                )

            inference_names = {obs.role: obs.name for obs in group}
            weight_like = next((o for o in group if o.role in ("weight", "weight_quantized")), group[0])

            # checkpointNames = adapter roles from the mapping (validated against the PEFT schema)
            # plus the frozen base weight the merger reads from the CheckpointState.
            checkpoint_names = {
                role: name for role, name in role_names.items() if not known_roles or role in known_roles
            }
            checkpoint_names.setdefault("weight", f"{training_base}.weight")

            # Describe the adapter factors, not just name them. `trainable_tensor_specs` is keyed by
            # the TRAINING GRAPH's initializer name (`backbone.model...lora_A.lora.weight`), while
            # `checkpoint_names` holds the PEFT module path (`base_model.model.model...lora_A.lora`) —
            # two of the five spellings of one layer. `to_checkpoint_name` is the existing normalizer
            # (twin of `cpp/layer_name.h`); re-deriving the rewrite here is what the layer-identity
            # work exists to prevent. A role whose tensor is not found is simply left undescribed
            # rather than guessed.
            adapter_dtypes: dict[str, str] = {}
            adapter_shapes: dict[str, tuple[int, ...]] = {}
            if trainable_tensor_specs:
                for role, module_path in checkpoint_names.items():
                    if role == "weight":
                        continue  # the frozen base, described by tensor_dtypes/tensor_shapes already
                    initializer = f"{to_checkpoint_name(module_path)}.weight"
                    spec = trainable_tensor_specs.get(initializer)
                    if spec is None:
                        continue
                    adapter_dtypes[role] = str(spec["dtype"])
                    adapter_shapes[role] = tuple(int(d) for d in spec["shape"])

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
                    # Keep each role's OWN observed dtype/shape. The device reads raw external-data
                    # bytes with no header, so collapsing these onto the weight-like role left a
                    # packed weight_quantized/scale/zero_point unloadable.
                    tensor_dtypes={obs.role: obs.dtype for obs in group},
                    tensor_shapes={obs.role: obs.shape for obs in group},
                    checkpoint_names=checkpoint_names,
                    adapter_dtypes=adapter_dtypes,
                    adapter_shapes=adapter_shapes,
                    merger_output_names={role: f"merged_{role}" for role in inference_names},
                    merged_tensor_names=dict(inference_names),
                    inference_initializer_names=dict(inference_names),
                    external_data_location={role: f"{name}.bin" for role, name in inference_names.items()},
                    quantization=quantization,
                    transpose_policy=derive_transpose_policy(weight_like.shape, adapter_shapes),
                )
            )

        # Fail closed on total adapter-spec drift. Every role above is looked up by a DERIVED
        # initializer name, and a miss leaves the role merely "undescribed" — deliberate, since a
        # partially-described entry is still usable. But if specs were supplied and NOT ONE entry
        # matched, the derivation is not lenient, it is broken: `adapter_shapes` is empty everywhere,
        # so orientation becomes unobservable and the whole package silently reverts to
        # `no_transpose`. That is precisely the defect this field was rebuilt to prevent, reached by
        # a name spelling drift instead of by an unassigned field. Name the lookup that missed.
        if trainable_tensor_specs and entries and not any(e.adapter_shapes for e in entries):
            probe = entries[0]
            attempted = [
                f"{to_checkpoint_name(path)}.weight"
                for role, path in probe.checkpoint_names.items()
                if role != "weight"
            ]
            raise HandoffError(
                f"{len(trainable_tensor_specs)} trainable tensor specs were supplied but none matched "
                f"any adapter role across {len(entries)} entries — for example "
                f"{probe.training_base_layer_name!r} looked for {attempted}. The training-graph "
                "initializer spelling has drifted from `to_checkpoint_name`. Refusing to emit a map "
                "that describes no adapter factors: weight orientation would be unobservable and the "
                "package would silently declare 'no_transpose'."
            )

        # One convention per package. A square weight cannot decide its own orientation, so the
        # entries that CAN decide settle it for the ones that cannot — otherwise `q_proj` (square) and
        # `v_proj` (not) would describe the same export two different ways.
        package_policy = resolve_package_transpose_policy(entries)
        for entry in entries:
            entry.transpose_policy = package_policy
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
