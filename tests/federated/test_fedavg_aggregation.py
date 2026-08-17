"""#35: FedAvg aggregation equals the (weighted) mean; the global artifact is written."""

from __future__ import annotations

import numpy as np

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from mobiletransformers.federated.flower_sim import (
    ClientUpdate,
    federated_average,
    save_global_adapter,
)
from tests.federated._helpers import make_arrays, make_handoff


def test_weighted_average_matches_manual_mean():
    a = make_arrays(scale=1.0)  # client 1 tensors
    b = make_arrays(scale=3.0)  # client 2 tensors
    updates = [ClientUpdate(a, num_examples=1), ClientUpdate(b, num_examples=3)]

    agg = federated_average(updates)

    # weighted mean: (1*a + 3*b) / 4 = (a + 3*(3a)) / 4 = (a + 9a)/4 = 2.5*a  (since b == 3a)
    for i, expected in enumerate(a):
        assert np.allclose(agg[i], expected * 2.5)


def test_equal_weight_average():
    a = make_arrays(scale=2.0)
    b = make_arrays(scale=4.0)
    agg = federated_average([ClientUpdate(a, 1), ClientUpdate(b, 1)])
    for i in range(len(a)):
        assert np.allclose(agg[i], (a[i] + b[i]) / 2)


def test_global_artifact_written(tmp_path):
    handoff = make_handoff()
    agg = federated_average([ClientUpdate(make_arrays(1.0), 1), ClientUpdate(make_arrays(3.0), 3)])
    rec = FederatedAdapterRecord.from_handoff(
        handoff, agg, base_model_id="org/base", peft_method="lora", round=2
    )
    path = save_global_adapter(rec, tmp_path)
    assert path.name == "global_adapter_round2.mtfed"
    assert path.exists()
    # the saved artifact deserializes back to the aggregated tensors.
    back = FederatedAdapterRecord.deserialize(path.read_bytes())
    for orig, restored in zip(agg, back.to_ndarrays(), strict=True):
        assert np.array_equal(orig, restored)
