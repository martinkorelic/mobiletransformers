"""#35: one server round — aggregate, wrap in the codec-ordered record, persist.

``build_server_app`` used to discard all five of its arguments and return a bare ``ServerApp()``: no
strategy, no round loop, no handler. ``federated_average`` was wired to nothing and
``save_global_adapter`` had no caller, so ``run_simulation`` reported success for an ``--output``
directory that was guaranteed to be empty. :func:`aggregate_round` is the whole server side of a round
minus the Flower messaging, which is why it can be tested here without ``flwr``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from mobiletransformers.federated.flower_sim import ClientUpdate, aggregate_round
from tests.federated._helpers import make_arrays, make_handoff


def _round(tmp_path, updates, round_index=1):
    return aggregate_round(
        make_handoff(),
        updates,
        base_model_id="org/base",
        peft_method="lora",
        round_index=round_index,
        output_dir=tmp_path,
    )


def test_round_writes_a_readable_global_adapter(tmp_path):
    updates = [ClientUpdate(make_arrays(scale=1.0), 2), ClientUpdate(make_arrays(scale=3.0), 2)]

    aggregated, path = _round(tmp_path, updates)

    assert path.is_file(), "the round must persist an artifact — --output was never written before"
    assert path.name == "global_adapter_round1.mtfed"

    restored = FederatedAdapterRecord.deserialize(path.read_bytes())
    assert restored.round == 1
    assert restored.base_model_id == "org/base"
    assert len(restored.arrays) == len(aggregated)
    for got, expected in zip(restored.to_ndarrays(), aggregated, strict=True):
        assert np.allclose(got, expected)


def test_round_result_is_the_weighted_mean(tmp_path):
    a, b = make_arrays(scale=1.0), make_arrays(scale=3.0)
    aggregated, _ = _round(tmp_path, [ClientUpdate(a, 1), ClientUpdate(b, 3)])
    for got, base in zip(aggregated, a, strict=True):
        assert np.allclose(got, base * 2.5)  # (1*a + 3*3a)/4


def test_round_records_participation_metrics(tmp_path):
    updates = [ClientUpdate(make_arrays(), 5), None, ClientUpdate(make_arrays(), 7)]
    _, path = _round(tmp_path, updates)
    metrics = FederatedAdapterRecord.deserialize(path.read_bytes()).metrics
    assert metrics["clients"] == 2
    assert metrics["dropped"] == 1
    assert metrics["numExamples"] == 12


def test_rounds_produce_distinct_artifacts(tmp_path):
    _, first = _round(tmp_path, [ClientUpdate(make_arrays(), 1)], round_index=1)
    _, second = _round(tmp_path, [ClientUpdate(make_arrays(), 1)], round_index=2)
    assert first != second
    assert sorted(p.name for p in tmp_path.glob("*.mtfed")) == [
        "global_adapter_round1.mtfed",
        "global_adapter_round2.mtfed",
    ]


def test_round_fails_closed_when_every_client_dropped(tmp_path):
    with pytest.raises(HandoffError, match="no surviving client updates"):
        _round(tmp_path, [None, None])
    assert not list(tmp_path.glob("*.mtfed")), "a failed round must not leave an artifact behind"


def test_round_fails_closed_on_tensor_count_mismatch(tmp_path):
    short = make_arrays()[:-1]
    with pytest.raises(HandoffError):
        _round(tmp_path, [ClientUpdate(make_arrays(), 1), ClientUpdate(short, 1)])
