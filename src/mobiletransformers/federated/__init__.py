"""Federated adapter codec + Python Flower simulation (#35, Tier-3 showcase).

Framing: *Flower-compatible federated adapter experiments*, not production federated Android LLM training.
Option A (Python-only in-process simulation) only. The exchange record (:mod:`adapter_record`) and the
aggregation math (:mod:`flower_sim`) are pure numpy + stdlib — importable and testable without Flower or an
ORT-training runtime. The Flower ``ClientApp``/``run_simulation`` orchestration (:mod:`flower_client`) imports
``flwr`` lazily and is exercised in the manual workflow leg.
"""

from __future__ import annotations

from mobiletransformers.federated.adapter_record import (
    FederatedAdapterRecord,
    FederatedTensor,
    codec_tensor_specs,
)
from mobiletransformers.federated.flower_sim import ClientUpdate, federated_average, save_global_adapter

__all__ = [
    "FederatedAdapterRecord",
    "FederatedTensor",
    "codec_tensor_specs",
    "ClientUpdate",
    "federated_average",
    "save_global_adapter",
]
