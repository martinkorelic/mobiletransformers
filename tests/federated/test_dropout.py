"""#35: a dropped client (missing reply) does not stall aggregation — it completes over the survivors."""

from __future__ import annotations

import numpy as np
import pytest

from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.flower_sim import ClientUpdate, federated_average
from tests.federated._helpers import make_arrays


def test_dropped_client_skipped():
    a = make_arrays(scale=1.0)
    c = make_arrays(scale=5.0)
    # middle client dropped (None); aggregation runs over clients 1 and 3.
    agg = federated_average([ClientUpdate(a, 1), None, ClientUpdate(c, 1)])
    for i in range(len(a)):
        assert np.allclose(agg[i], (a[i] + c[i]) / 2)


def test_all_clients_dropped_fails_closed():
    with pytest.raises(HandoffError, match="no surviving client"):
        federated_average([None, None])


def test_tensor_count_mismatch_fails_closed():
    a = make_arrays()
    bad = [a[0]]  # one tensor instead of two
    with pytest.raises(HandoffError, match="tensor-count mismatch"):
        federated_average([ClientUpdate(a, 1), ClientUpdate(bad, 1)])
