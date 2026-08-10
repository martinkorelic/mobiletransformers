"""#35: freeze the pinned byte serialization as a golden (the #36 cross-language JNI contract).

If this test fails after an intentional format change, regenerate the golden with:
    python -m tests.federated.gen_serialization_golden
"""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from tests.federated._helpers import make_arrays, make_handoff

_GOLDEN = Path(__file__).parent / "fixtures" / "federated_record.golden.bin"


def _deterministic_record() -> FederatedAdapterRecord:
    return FederatedAdapterRecord.from_handoff(
        make_handoff(),
        make_arrays(),
        base_model_id="org/base",
        peft_method="lora",
        round=0,
        package_revision="rev-1",
    )


def test_serialization_matches_golden():
    blob = _deterministic_record().serialize()
    assert _GOLDEN.exists(), f"missing golden {_GOLDEN}; run gen_serialization_golden.py"
    assert blob == _GOLDEN.read_bytes()


def test_golden_deserializes_back():
    back = FederatedAdapterRecord.deserialize(_GOLDEN.read_bytes())
    assert back.base_model_id == "org/base"
    assert back.mobiletransformers_package_revision == "rev-1"
    # Rank-r factors as of #35, in codec order: entries by canonical weight name, then adapter role.
    assert [t.name for t in back.tensors] == [
        "l0.lora_A.lora.weight",
        "l0.lora_B.lora.weight",
        "l1.lora_A.lora.weight",
        "l1.lora_B.lora.weight",
    ]
    assert [t.role for t in back.tensors] == ["adapter_A", "adapter_B", "adapter_A", "adapter_B"]
