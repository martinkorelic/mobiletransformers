"""``FederatedAdapterRecord`` — a thin wrapper over the canonical tensor codec (#8) for federation (#35).

The record **invents no tensor ordering**: the exchanged tensor list (names, order, dtype, shape) comes
straight from :class:`~mobiletransformers.artifacts.handoff_map.HandoffMap` /
:meth:`HandoffEntry.tensor_specs` — the ONE source of tensor identity. ``adapterFormatVersion`` **equals**
the handoff-map ``schemaVersion`` and is gated by the shared ``check_compat`` helper, so a record built
against an incompatible codec fails closed.

**Byte serialization (pinned — #36's JNI ``ByteArray`` uses exactly this):** a 4-byte little-endian
``uint32`` header length, then the UTF-8 JSON header (each tensor entry carrying ``byteOffset``/``byteLength``
instead of inline bytes), then the concatenated raw tensor payloads in **codec order** — each little-endian,
contiguous C-order, dtype/shape exactly as declared. No compression, no alignment padding in v1.

**v1 simplification (not a silent cap):** the federatable set is the codec's ordered *trainable* tensor
specs (the per-layer LoRA-shaped trainable weights). True A/B-factor-only exchange is a future refinement;
using the handoff map's tensor identity keeps us from inventing a second ordering (F8).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mobiletransformers.artifacts.handoff_map import HandoffMap, TensorSpec
from mobiletransformers.artifacts.versioning import check_compat
from mobiletransformers.exceptions import HandoffError

if TYPE_CHECKING:
    import numpy as np

#: Reader schema version for the federated record contract (see ``check_compat``).
FEDERATED_RECORD_READER_VERSION = "1.0"

#: handoff dtype name -> little-endian numpy dtype string. Only float/int8 trainables are federatable in v1;
#: int4 (packed quantized base weights) are never exchanged and fail closed.
_DTYPE_TO_NP = {
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "int8": "|i1",
    "uint8": "|u1",
    "int32": "<i4",
}


def _np_dtype(dtype: str) -> str:
    np_dtype = _DTYPE_TO_NP.get(dtype)
    if np_dtype is None:
        raise HandoffError(f"dtype {dtype!r} is not federatable in v1 (float/int8 trainables only)")
    return np_dtype


def codec_tensor_specs(handoff: HandoffMap) -> list[TensorSpec]:
    """The deterministic, codec-derived list of federatable tensor specs (order == serialization order).

    **These are the rank-r ADAPTER FACTORS, not the merged weights (#35, decided 2026-08-09).**

    The two vocabularies are genuinely different objects, and conflating them is what stopped the
    simulation dead: `HandoffEntry.tensor_specs()` describes ONE merged inference initializer per
    adapted layer (60 tensors at full weight shape on SmolLM2-135M), while the ORT checkpoint holds
    `lora_A` + `lora_B` per adapted layer (120 tensors at rank-r shape). Federation now exchanges the
    factors:

    * per-round traffic drops by roughly ``d_in * d_out / (r * (d_in + d_out))`` — about 36x at r=8 on
      this model, and the ratio grows with ``d/r``;
    * it matches the tier doc's "do not aggregate merged base weights";
    * a client no longer has to merge locally before it can send anything.

    Fails closed on a map that cannot describe its factors (any package exported before handoff-map
    schema 1.1). Falling back to the merged specs would silently resurrect the exact ambiguity this
    decision removed, and the failure would surface as a shape mismatch several layers away.
    """
    specs: list[TensorSpec] = []
    missing: list[str] = []
    for entry in handoff._sorted_entries():
        entry_specs = entry.adapter_tensor_specs()
        if not entry_specs:
            missing.append(entry.training_base_layer_name)
        specs.extend(entry_specs)

    if missing:
        raise HandoffError(
            f"{len(missing)} handoff entries carry no adapter dtype/shape (e.g. {missing[0]}), so the "
            "rank-r factors cannot be described. This package predates weight_handoff_map schema 1.1 "
            "— re-export it with the training stage. (Federation exchanges adapter factors as of #35; "
            "it does not fall back to merged weights.)"
        )
    return specs


#: The only aggregation v1 produces or accepts (#35, decided 2026-08-08).
#:
#: The dataclass used to advertise `"weighted_average" | "average" | "server_only"` in a comment, but
#: no caller ever set the other two — they were vocabulary with no implementation behind them. An
#: unreachable enum value in a **wire format** is worse than absent: a peer may legitimately emit it,
#: and this side would have accepted it and then aggregated as if it were a weighted average.
#: Rejected on read instead. Adding a value back means implementing it AND regenerating the golden.
SUPPORTED_AGGREGATIONS = frozenset({"weighted_average"})

#: The tensor roles the record carries — `TrainableTensorCodec`'s vocabulary (#8), which #35 ratified
#: as normative over the tier doc's never-implemented `{adapter, trainable_weight, head}`.
SUPPORTED_ROLES = frozenset(
    {
        # Adapter factors — what v1 exchanges as of the #35 rank-r decision.
        "shared_A",
        "intermediate",
        "adapter_A",
        "adapter_B",
        # Merged-weight roles. Still accepted on READ so records written under the previous
        # merged-weight vocabulary deserialize rather than failing as "unknown role"; nothing
        # produces them any more.
        "weight",
        "weight_quantized",
        "scale",
        "zero_point",
    }
)


@dataclass
class FederatedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str  # one of SUPPORTED_ROLES
    aggregation: str  # SUPPORTED_AGGREGATIONS — single-valued in v1


@dataclass
class FederatedAdapterRecord:
    """One round's adapter payload — metadata wrapper + the raw tensor arrays (in codec order)."""

    base_model_id: str
    peft_method: str
    adapter_format_version: str
    tensors: list[FederatedTensor]
    arrays: list[np.ndarray]
    round: int = 0
    mobiletransformers_package_revision: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    min_reader_version: str = "1.0"

    def __post_init__(self) -> None:
        if len(self.tensors) != len(self.arrays):
            raise HandoffError(
                f"tensor/array count mismatch: {len(self.tensors)} specs vs {len(self.arrays)} arrays"
            )

    # --- construction from the canonical codec ---------------------------------------------------
    @classmethod
    def from_handoff(
        cls,
        handoff: HandoffMap,
        arrays: Sequence[np.ndarray],
        *,
        base_model_id: str,
        peft_method: str,
        round: int = 0,
        package_revision: str = "",
        metrics: dict[str, Any] | None = None,
        aggregation: str = "weighted_average",
    ) -> FederatedAdapterRecord:
        specs = codec_tensor_specs(handoff)
        if len(arrays) != len(specs):
            raise HandoffError(f"expected {len(specs)} arrays (codec-derived), got {len(arrays)}")
        tensors = [FederatedTensor(s.name, s.dtype, tuple(s.shape), s.role, aggregation) for s in specs]
        return cls(
            base_model_id=base_model_id,
            peft_method=peft_method,
            adapter_format_version=handoff.schema_version,
            tensors=tensors,
            arrays=list(arrays),
            round=round,
            mobiletransformers_package_revision=package_revision,
            metrics=metrics or {},
        )

    def check_format(self, handoff: HandoffMap) -> None:
        """Fail closed unless this record's ``adapterFormatVersion`` matches the codec it rides on (F1/F8)."""
        check_compat(self.schema_version, self.min_reader_version, FEDERATED_RECORD_READER_VERSION)
        if self.adapter_format_version != handoff.schema_version:
            raise HandoffError(
                f"adapterFormatVersion {self.adapter_format_version!r} != weight_handoff_map "
                f"schemaVersion {handoff.schema_version!r}"
            )

    # --- ndarray view ----------------------------------------------------------------------------
    def to_ndarrays(self) -> list[np.ndarray]:
        return list(self.arrays)

    # --- pinned byte serialization ---------------------------------------------------------------
    def serialize(self) -> bytes:
        import numpy as np

        payloads: list[bytes] = []
        tensor_headers: list[dict[str, Any]] = []
        offset = 0
        for spec, arr in zip(self.tensors, self.arrays, strict=True):
            np_dtype = _np_dtype(spec.dtype)
            buf = np.ascontiguousarray(arr, dtype=np_dtype).tobytes(order="C")
            tensor_headers.append(
                {
                    "name": spec.name,
                    "dtype": spec.dtype,
                    "shape": list(spec.shape),
                    "role": spec.role,
                    "aggregation": spec.aggregation,
                    "byteOffset": offset,
                    "byteLength": len(buf),
                }
            )
            payloads.append(buf)
            offset += len(buf)

        header = {
            "schemaVersion": self.schema_version,
            "minReaderVersion": self.min_reader_version,
            "baseModelId": self.base_model_id,
            "mobiletransformersPackageRevision": self.mobiletransformers_package_revision,
            "peftMethod": self.peft_method,
            "adapterFormatVersion": self.adapter_format_version,
            "round": self.round,
            "tensors": tensor_headers,
            "metrics": self.metrics,
        }
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        return struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(payloads)

    @classmethod
    def deserialize(cls, data: bytes) -> FederatedAdapterRecord:
        import numpy as np

        if len(data) < 4:
            raise HandoffError("truncated federated record (no header length)")
        (header_len,) = struct.unpack("<I", data[:4])
        header_end = 4 + header_len
        if len(data) < header_end:
            raise HandoffError("truncated federated record (header shorter than declared)")
        header = json.loads(data[4:header_end].decode("utf-8"))
        payload = data[header_end:]

        # F1/F8: gate the WIRE record's schema before trusting any of its offsets. This used to run
        # only if a caller happened to invoke check_format afterwards, and no caller did — so a peer
        # record built against an incompatible codec was accepted by the only ingest path there is.
        check_compat(
            header.get("schemaVersion", "1.0"),
            header.get("minReaderVersion", "1.0"),
            FEDERATED_RECORD_READER_VERSION,
        )

        tensors: list[FederatedTensor] = []
        arrays: list[np.ndarray] = []
        for t in header["tensors"]:
            start, length = t["byteOffset"], t["byteLength"]
            # Bounds-check before slicing: byteOffset/byteLength come from an untrusted peer, and a
            # Python slice silently clamps rather than raising.
            if start < 0 or length < 0 or start + length > len(payload):
                raise HandoffError(
                    f"tensor {t['name']!r} declares bytes [{start}, {start + length}) "
                    f"outside the {len(payload)}-byte payload"
                )
            chunk = payload[start : start + length]
            if len(chunk) != length:
                raise HandoffError(f"truncated payload for tensor {t['name']!r}")
            expected = int(np.dtype(_np_dtype(t["dtype"])).itemsize)
            for dim in t["shape"]:
                expected *= int(dim)
            if expected != length:
                raise HandoffError(
                    f"tensor {t['name']!r}: shape {tuple(t['shape'])} of {t['dtype']} needs "
                    f"{expected} bytes, header declares {length}"
                )
            arr = np.frombuffer(chunk, dtype=_np_dtype(t["dtype"])).reshape(tuple(t["shape"]))
            if t["aggregation"] not in SUPPORTED_AGGREGATIONS:
                raise HandoffError(
                    f"tensor {t['name']!r}: unsupported aggregation {t['aggregation']!r}; v1 supports "
                    f"only {sorted(SUPPORTED_AGGREGATIONS)}. Accepting it would silently aggregate the "
                    "tensor as a weighted average, which is not what the peer asked for."
                )
            if t["role"] not in SUPPORTED_ROLES:
                raise HandoffError(
                    f"tensor {t['name']!r}: unknown role {t['role']!r}; expected one of "
                    f"{sorted(SUPPORTED_ROLES)} (#35 codec vocabulary)."
                )
            tensors.append(
                FederatedTensor(t["name"], t["dtype"], tuple(t["shape"]), t["role"], t["aggregation"])
            )
            arrays.append(arr)

        return cls(
            base_model_id=header["baseModelId"],
            peft_method=header["peftMethod"],
            adapter_format_version=header["adapterFormatVersion"],
            tensors=tensors,
            arrays=arrays,
            round=header.get("round", 0),
            mobiletransformers_package_revision=header.get("mobiletransformersPackageRevision", ""),
            metrics=header.get("metrics", {}),
            schema_version=header.get("schemaVersion", "1.0"),
            min_reader_version=header.get("minReaderVersion", "1.0"),
        )

    def comm_size_bytes(self) -> int:
        """Serialized size of this record (per-round communication cost)."""
        return len(self.serialize())


__all__ = [
    "FEDERATED_RECORD_READER_VERSION",
    "FederatedTensor",
    "FederatedAdapterRecord",
    "codec_tensor_specs",
]
