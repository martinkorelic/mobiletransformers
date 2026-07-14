"""Support-matrix dataclasses + the canonical (list-shaped) JSON envelope (#20 owns this schema).

Reconciles #7's seed-row shape (``export/support_matrix.py``, a dict keyed by model id, with
``chosenTask``/``blocker``) into the #20 contract: ``models`` is a *list*, task is ``selectedTask``,
blockers are a ``blockers[]`` list, and each row carries the full six-status map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mobiletransformers.support.statuses import STATUS_ORDER, USER_FACING_STATUSES

SCHEMA_VERSION = "1.0"
MIN_READER_VERSION = "1.0"
TRANSFORMERS_CEILING = "<4.58"


@dataclass
class CandidateEntry:
    model_id: str
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    optimum_onnx_version: str | None = None
    transformers_version: str | None = None
    opset: int = 20
    supported_tasks: tuple[str, ...] = ()
    selected_task: str | None = None
    trust_remote_code: bool = False
    mars_target_modules_known: bool = False
    statuses: dict[str, bool] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """The per-model wire row (camelCase, list-shaped) for ``model_support_matrix.json``."""
        return {
            "modelId": self.model_id,
            "modelType": self.model_type,
            "architectures": list(self.architectures),
            "optimumOnnxVersion": self.optimum_onnx_version,
            "transformersVersion": self.transformers_version,
            "opset": self.opset,
            "supportedTasks": list(self.supported_tasks),
            "selectedTask": self.selected_task,
            "trustRemoteCode": self.trust_remote_code,
            "marsTargetModulesKnown": self.mars_target_modules_known,
            "statuses": {k: bool(self.statuses.get(k, False)) for k in STATUS_ORDER},
            "blockers": list(self.blockers),
        }


@dataclass
class SupportMatrix:
    models: list[CandidateEntry]
    generated_at: str | None = None
    toolchain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "minReaderVersion": MIN_READER_VERSION,
            "generatedAt": self.generated_at or "",
            "toolchain": {"transformersCeiling": TRANSFORMERS_CEILING, **self.toolchain},
            "statusOrder": list(STATUS_ORDER),
            "userFacingStatuses": sorted(USER_FACING_STATUSES),
            "models": [m.to_row() for m in self.models],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def filtered_docs_dict(self) -> dict[str, Any]:
        """User-facing view: only models with ≥1 user-facing status true; strip contributor-only
        (non-user-facing) statuses from each row."""
        d = self.to_dict()
        kept = []
        for row in d["models"]:
            if any(row["statuses"].get(s) for s in USER_FACING_STATUSES):
                row = dict(row)
                row["statuses"] = {s: row["statuses"][s] for s in sorted(USER_FACING_STATUSES)}
                kept.append(row)
        d["models"] = kept
        return d


__all__ = [
    "SCHEMA_VERSION",
    "MIN_READER_VERSION",
    "TRANSFORMERS_CEILING",
    "CandidateEntry",
    "SupportMatrix",
]
