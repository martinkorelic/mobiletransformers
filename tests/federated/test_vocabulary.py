"""#35 role/aggregation vocabulary — decided 2026-08-08, enforced on read.

The record is a **wire format**: a peer builds it and this side consumes it. An enum value that is
declared but unimplemented is therefore not harmless documentation — a peer may legitimately emit it,
and before this change `from_bytes` accepted it and carried on, so a tensor marked `server_only` would
have been aggregated as a weighted average. Unknown values now fail closed.

**Updated 2026-08-09 (#35 rank-r decision).** The exchanged vocabulary is now the ADAPTER FACTOR
roles (`shared_A`/`intermediate`/`adapter_A`/`adapter_B`). The merged-weight roles stay in
`SUPPORTED_ROLES` for READ compatibility — a peer may still hold a record written under the previous
vocabulary, and rejecting it as "unknown role" would be worse than accepting and reporting it — but
nothing produces them any more. The golden was regenerated for this change.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.adapter_record import (
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_ROLES,
    FederatedAdapterRecord,
)
from tests.federated._helpers import make_arrays, make_handoff


def _record() -> FederatedAdapterRecord:
    return FederatedAdapterRecord.from_handoff(
        make_handoff(), make_arrays(), base_model_id="tiny/model", peft_method="lora"
    )


def _rewrite_header(blob: bytes, mutate) -> bytes:
    """Patch the JSON header in place, keeping the payload and the length prefix consistent."""
    (header_len,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + header_len])
    payload = blob[4 + header_len :]
    mutate(header)
    new_header = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(new_header)) + new_header + payload


def test_v1_vocabulary_is_the_codec_vocabulary():
    # What v1 PRODUCES: the adapter factors (#35 rank-r decision).
    assert {"shared_A", "intermediate", "adapter_A", "adapter_B"} <= SUPPORTED_ROLES
    # What it still ACCEPTS on read: the previous merged-weight roles.
    assert {"weight", "weight_quantized", "scale", "zero_point"} <= SUPPORTED_ROLES
    # The tier doc's original set was never implemented by anything, and still is not.
    assert not SUPPORTED_ROLES & {"adapter", "trainable_weight", "head"}


def test_aggregation_is_single_valued_in_v1():
    assert SUPPORTED_AGGREGATIONS == {"weighted_average"}


def test_round_trip_still_works():
    record = _record()
    restored = FederatedAdapterRecord.deserialize(record.serialize())

    assert [t.role for t in restored.tensors] == [t.role for t in record.tensors]
    assert all(t.aggregation in SUPPORTED_AGGREGATIONS for t in restored.tensors)
    for got, want in zip(restored.arrays, record.arrays, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("bad", ["average", "server_only", "median", ""])
def test_unknown_aggregation_is_rejected(bad):
    """`average`/`server_only` were declared-but-unreachable; a peer emitting one must not be misread."""
    blob = _rewrite_header(_record().serialize(), lambda h: h["tensors"][0].update(aggregation=bad))

    with pytest.raises(HandoffError, match="unsupported aggregation"):
        FederatedAdapterRecord.deserialize(blob)


@pytest.mark.parametrize("bad", ["adapter", "trainable_weight", "head", "bias"])
def test_unknown_role_is_rejected(bad):
    """Including the tier doc's old vocabulary: it is not silently accepted as an alias."""
    blob = _rewrite_header(_record().serialize(), lambda h: h["tensors"][0].update(role=bad))

    with pytest.raises(HandoffError, match="unknown role"):
        FederatedAdapterRecord.deserialize(blob)
