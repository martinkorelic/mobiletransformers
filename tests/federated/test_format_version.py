"""#35: adapterFormatVersion must equal the handoff schemaVersion; mismatch fails closed (F1/F8)."""

from __future__ import annotations

import pytest

from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from tests.federated._helpers import make_arrays, make_handoff


def test_matching_format_version_passes():
    handoff = make_handoff()
    rec = FederatedAdapterRecord.from_handoff(
        handoff, make_arrays(), base_model_id="org/base", peft_method="lora"
    )
    assert rec.adapter_format_version == handoff.schema_version
    rec.check_format(handoff)  # no raise


def test_mismatched_format_version_fails_closed():
    handoff = make_handoff()
    rec = FederatedAdapterRecord.from_handoff(
        handoff, make_arrays(), base_model_id="org/base", peft_method="lora"
    )
    rec.adapter_format_version = "2.0"  # simulate a record built against an incompatible codec
    with pytest.raises(HandoffError, match="adapterFormatVersion"):
        rec.check_format(handoff)
