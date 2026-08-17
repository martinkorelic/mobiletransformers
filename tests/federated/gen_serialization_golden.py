"""Regenerate the federated-record byte golden (run from the repo root under the core/dev env).

python -m tests.federated.gen_serialization_golden
"""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from tests.federated._helpers import make_arrays, make_handoff


def main() -> None:
    rec = FederatedAdapterRecord.from_handoff(
        make_handoff(),
        make_arrays(),
        base_model_id="org/base",
        peft_method="lora",
        round=0,
        package_revision="rev-1",
    )
    out = Path(__file__).parent / "fixtures" / "federated_record.golden.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rec.serialize())
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
