"""#36 server half: aggregation, dropout, and refusing to publish an aggregate nobody agreed on."""

from __future__ import annotations

import numpy as np
import pytest

from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.adapter_record import (
    FederatedAdapterRecord,
    codec_tensor_specs,
)
from mobiletransformers.federated.gateway import FederatedGateway

from ._helpers import make_handoff


def _record(handoff, scale: float, round_number: int = 0) -> bytes:
    """A client record whose every tensor is a constant `scale`, so averages are checkable by hand."""
    specs = codec_tensor_specs(handoff)
    arrays = [np.full(tuple(s.shape), scale, dtype=np.float32) for s in specs]
    return FederatedAdapterRecord.from_handoff(
        handoff, arrays, base_model_id="org/base", peft_method="lora", round=round_number
    ).serialize()


def _gateway(handoff, **kw) -> FederatedGateway:
    return FederatedGateway(handoff, base_model_id="org/base", **kw)


def test_fedavg_is_weighted_by_num_examples() -> None:
    handoff = make_handoff()
    gw = _gateway(handoff)

    # 1.0 with weight 1, 3.0 with weight 3 -> (1*1 + 3*3)/4 = 2.5
    result = gw.aggregate([("a", _record(handoff, 1.0), 1), ("b", _record(handoff, 3.0), 3)])

    decoded = FederatedAdapterRecord.deserialize(result.blob)
    for array in decoded.arrays:
        assert np.allclose(array, 2.5), "aggregate is not the example-weighted mean"
    assert result.accepted == 2
    assert result.total_examples == 4


def test_a_dropped_client_does_not_end_the_round() -> None:
    # Devices go offline mid-round; that is normal operation, not an error.
    handoff = make_handoff()
    gw = _gateway(handoff, min_clients=2)

    result = gw.aggregate(
        [("a", _record(handoff, 2.0), 1), ("b", _record(handoff, 2.0), 1), ("gone", b"", 1)]
    )

    assert result.accepted == 2
    assert [client for client, _ in result.rejected] == ["gone"]


def test_a_round_below_min_clients_refuses_to_publish() -> None:
    # Publishing an aggregate two devices decided is worse than publishing nothing.
    handoff = make_handoff()
    gw = _gateway(handoff, min_clients=3)

    with pytest.raises(HandoffError, match="min_clients"):
        gw.aggregate([("a", _record(handoff, 1.0), 1), ("b", _record(handoff, 1.0), 1)])


def test_a_client_running_a_different_package_is_rejected_not_coerced() -> None:
    # Averaging in a record whose tensors do not match the package would silently corrupt the global
    # adapter — the shapes here differ, which is exactly what "a different package" looks like.
    handoff = make_handoff()
    specs = codec_tensor_specs(handoff)
    wrong = FederatedAdapterRecord.from_handoff(
        handoff,
        [np.full(tuple(s.shape), 1.0, dtype=np.float32) for s in specs],
        base_model_id="org/base",
        peft_method="lora",
    )
    # Corrupt one declared shape so the decoded array disagrees with the package.
    wrong.arrays[0] = np.full((7, 7), 1.0, dtype=np.float32)
    wrong.tensors[0].shape = (7, 7)

    gw = _gateway(handoff, min_clients=1)
    result = gw.aggregate([("ok", _record(handoff, 1.0), 1), ("bad", wrong.serialize(), 1)])

    assert result.accepted == 1
    assert result.rejected and result.rejected[0][0] == "bad"


def test_tensors_are_matched_by_name_not_by_position() -> None:
    # The #35 defect: pairing by iteration order would write one layer's lora_A over another's.
    # A record serialized in a DIFFERENT order must still aggregate correctly.
    handoff = make_handoff()
    specs = codec_tensor_specs(handoff)
    arrays = [np.full(tuple(s.shape), float(i + 1), dtype=np.float32) for i, s in enumerate(specs)]
    record = FederatedAdapterRecord.from_handoff(
        handoff, arrays, base_model_id="org/base", peft_method="lora"
    )
    # Reverse tensors AND arrays together: same content, different wire order.
    record.tensors = list(reversed(record.tensors))
    record.arrays = list(reversed(record.arrays))

    gw = _gateway(handoff, min_clients=1)
    result = gw.aggregate([("shuffled", record.serialize(), 1)])

    decoded = FederatedAdapterRecord.deserialize(result.blob)
    # Each tensor keeps ITS OWN value; a positional match would have permuted them.
    for i, array in enumerate(decoded.arrays):
        assert np.allclose(array, float(i + 1)), f"tensor {i} picked up another tensor's values"


def test_a_client_with_no_examples_is_rejected() -> None:
    handoff = make_handoff()
    gw = _gateway(handoff, min_clients=1)

    result = gw.aggregate([("ok", _record(handoff, 1.0), 2), ("empty", _record(handoff, 9.0), 0)])

    assert result.accepted == 1
    assert result.rejected[0][0] == "empty"


def test_the_global_record_round_trips_through_the_same_codec() -> None:
    # The bytes the server hands back must be bytes a client can read.
    handoff = make_handoff()
    gw = _gateway(handoff, min_clients=1)

    result = gw.aggregate([("a", _record(handoff, 1.5), 1)], round_number=3)

    decoded = FederatedAdapterRecord.deserialize(result.blob)
    decoded.check_format(handoff)
    assert decoded.round == 3
    assert [t.name for t in decoded.tensors] == [s.name for s in codec_tensor_specs(handoff)]


def test_min_clients_below_one_is_rejected_at_construction() -> None:
    with pytest.raises(HandoffError, match="min_clients"):
        _gateway(make_handoff(), min_clients=0)
