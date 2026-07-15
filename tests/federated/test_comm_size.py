"""#35: per-round communication size is measured and bounded (payload = raw tensor bytes + JSON header)."""

from __future__ import annotations

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from tests.federated._helpers import make_arrays, make_handoff


def test_comm_size_bounded_and_accounts_for_payload():
    handoff = make_handoff()
    arrays = make_arrays()
    rec = FederatedAdapterRecord.from_handoff(handoff, arrays, base_model_id="org/base", peft_method="lora")

    # Raw tensor payload = sum of each float32 array's nbytes ((2*3 + 3) * 4 = 36 bytes).
    raw_payload = sum(a.nbytes for a in arrays)
    assert raw_payload == 36

    size = rec.comm_size_bytes()
    # size = 4 (header length) + JSON header + raw payload; must exceed the payload and stay small.
    assert size > raw_payload + 4
    assert size < raw_payload + 4096  # header is tiny for a 2-tensor LoRA record
