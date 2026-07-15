"""#35: the federated record round-trips byte-identically with codec-derived tensor order."""

from __future__ import annotations

import numpy as np

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord, codec_tensor_specs
from tests.federated._helpers import make_arrays, make_handoff


def test_tensor_order_is_codec_derived():
    handoff = make_handoff()
    specs = codec_tensor_specs(handoff)
    # Order comes from HandoffMap._sorted_entries() (canonical weight name), not the record.
    assert [s.name for s in specs] == [
        "model.layers.0.attn.q_proj.MatMul.weight",
        "model.layers.1.attn.v_proj.MatMul.weight",
    ]


def test_record_roundtrip_byte_identical():
    handoff = make_handoff()
    arrays = make_arrays()
    rec = FederatedAdapterRecord.from_handoff(
        handoff, arrays, base_model_id="org/base", peft_method="lora", round=1
    )

    blob = rec.serialize()
    back = FederatedAdapterRecord.deserialize(blob)

    assert back.base_model_id == "org/base"
    assert back.peft_method == "lora"
    assert back.round == 1
    assert [t.name for t in back.tensors] == [t.name for t in rec.tensors]
    for orig, restored in zip(arrays, back.to_ndarrays(), strict=True):
        assert np.array_equal(orig, restored)
    # re-serializing the deserialized record reproduces the exact bytes.
    assert back.serialize() == blob
