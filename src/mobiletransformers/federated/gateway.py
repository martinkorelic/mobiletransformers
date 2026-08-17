"""Server side of a federated round: collect device records, aggregate, hand back a global record.

## What this is, and what it is not

This is the **round state machine**, not a web server. It takes serialized
:class:`FederatedAdapterRecord` blobs, validates each against the package the server holds, aggregates
the survivors with :func:`federated_average`, and returns a serialized global record. Whether those
blobs arrive over HTTP, gRPC or a Flower ``ServerApp`` is the transport's problem — keeping the two
apart is what lets the aggregation logic be tested without a socket, and what lets the same logic sit
behind the Flower strategy that ``flower_sim.py`` already drives.

## The rules it enforces, and why each exists

Every one of these corresponds to a defect that already happened, either in the #35 simulation or in
the class of bug this project keeps hitting:

* **Tensors are matched by NAME, never by position.** The simulation paired them by checkpoint
  iteration order, which would write one layer's ``lora_A`` over another's; differing shapes caught it
  "mostly", and "mostly" is not a guarantee.
* **A client whose record disagrees with the package is dropped, not coerced.** A record naming an
  unknown tensor, or the right tensor at the wrong shape, is a client running a different package —
  averaging it in would silently corrupt the global adapter.
* **Dropout is normal.** Devices go offline mid-round; the round completes over the survivors and says
  how many there were. What is *not* tolerated is completing with too few to be meaningful, hence
  ``min_clients``.
* **The global record is built through the same codec** the clients use, so the bytes the server hands
  back are the bytes a client can read — pinned by the same cross-language golden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.exceptions import HandoffError
from mobiletransformers.federated.adapter_record import (
    FederatedAdapterRecord,
    codec_tensor_specs,
)
from mobiletransformers.federated.flower_sim import ClientUpdate, federated_average
from mobiletransformers.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass
class RoundResult:
    """Outcome of one aggregation round."""

    round: int
    #: The serialized global record, ready to hand back to clients.
    blob: bytes
    #: Clients whose record was accepted and averaged in.
    accepted: int
    #: Clients rejected, with the reason — kept rather than discarded so a systematically broken
    #: client population is visible instead of looking like light dropout.
    rejected: list[tuple[str, str]]
    #: Total examples behind the aggregate, the weight FedAvg used.
    total_examples: int

    def describe(self) -> str:
        return (
            f"round {self.round}: {self.accepted} accepted, {len(self.rejected)} rejected, "
            f"{self.total_examples} examples, {len(self.blob)} B global record"
        )


class FederatedGateway:
    """Aggregates one round at a time against a fixed package.

    :param handoff: the server's copy of ``weight_handoff_map.json``. This is the authority on tensor
        identity, order and shape — a client's record is checked against it rather than against the
        other clients, so a cohort that is uniformly wrong is still caught.
    :param min_clients: fewest accepted clients for a round to be considered meaningful. Below this the
        round FAILS rather than publishing an aggregate a couple of devices decided.
    """

    def __init__(
        self,
        handoff: HandoffMap,
        *,
        base_model_id: str,
        peft_method: str = "lora",
        package_revision: str = "",
        min_clients: int = 2,
    ) -> None:
        if min_clients < 1:
            raise HandoffError(f"min_clients must be >= 1, got {min_clients}")
        self.handoff = handoff
        self.base_model_id = base_model_id
        self.peft_method = peft_method
        self.package_revision = package_revision
        self.min_clients = min_clients
        #: Declared tensor order — computed once, so every round agrees with the package and with itself.
        self.specs = codec_tensor_specs(handoff)
        self._expected = {spec.name: tuple(spec.shape) for spec in self.specs}

    def aggregate(
        self,
        submissions: list[tuple[str, bytes, int]],
        *,
        round_number: int = 0,
    ) -> RoundResult:
        """Aggregate one round.

        :param submissions: ``(client_id, serialized_record, num_examples)`` per client.
        :raises HandoffError: when fewer than :attr:`min_clients` records survive validation.
        """

        accepted: list[ClientUpdate] = []
        rejected: list[tuple[str, str]] = []

        for client_id, blob, num_examples in submissions:
            try:
                arrays = self._validated_arrays(blob)
            except HandoffError as exc:
                # A rejection is data, not an exception to propagate: one broken client must not end
                # the round for everyone else.
                logger.warning("round %s: dropping client %s (%s)", round_number, client_id, exc)
                rejected.append((client_id, str(exc)))
                continue
            if num_examples <= 0:
                rejected.append((client_id, f"num_examples must be > 0, got {num_examples}"))
                continue
            accepted.append(ClientUpdate(arrays=arrays, num_examples=num_examples))

        if len(accepted) < self.min_clients:
            raise HandoffError(
                f"round {round_number} had {len(accepted)} usable client update(s), below the "
                f"min_clients={self.min_clients} floor; refusing to publish an aggregate. "
                f"Rejections: {rejected or 'none'}"
            )

        averaged = federated_average(list(accepted))
        blob = self._build_global_record(averaged, round_number=round_number)

        return RoundResult(
            round=round_number,
            blob=blob,
            accepted=len(accepted),
            rejected=rejected,
            total_examples=sum(u.num_examples for u in accepted),
        )

    def _validated_arrays(self, blob: bytes) -> list[np.ndarray]:
        """Decode one client record into arrays **in the package's declared order**.

        Matching by name is the whole point; the record's own ordering is not trusted, so a client that
        serialized in a different order still contributes correctly rather than corrupting a layer.
        """
        import numpy as np

        record = FederatedAdapterRecord.deserialize(blob)
        record.check_format(self.handoff)

        # `tensors` and `arrays` are parallel lists (the record's own invariant, enforced in its
        # __post_init__), so this is where the two are joined into a name-keyed view.
        by_name = dict(zip([t.name for t in record.tensors], record.arrays, strict=True))
        unknown = set(by_name) - set(self._expected)
        if unknown:
            raise HandoffError(f"record carries tensor(s) this package does not declare: {sorted(unknown)}")

        arrays: list[np.ndarray] = []
        for spec in self.specs:
            if spec.name not in by_name:
                raise HandoffError(f"record is missing declared tensor {spec.name!r}")
            array = np.asarray(by_name[spec.name])
            if tuple(array.shape) != self._expected[spec.name]:
                raise HandoffError(
                    f"tensor {spec.name!r} has shape {tuple(array.shape)}, package declares "
                    f"{self._expected[spec.name]}"
                )
            arrays.append(array)
        return arrays

    def _build_global_record(self, arrays: list[np.ndarray], *, round_number: int) -> bytes:
        """Serialize the aggregate through the SAME codec clients read, so the bytes round-trip."""
        record = FederatedAdapterRecord.from_handoff(
            self.handoff,
            arrays,
            base_model_id=self.base_model_id,
            peft_method=self.peft_method,
            round=round_number,
            package_revision=self.package_revision,
        )
        return record.serialize()


__all__ = ["FederatedGateway", "RoundResult"]
