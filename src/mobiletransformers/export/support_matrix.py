"""Seed/merge ``model_support_matrix.json`` — the per-model export-status truth.

This plan (#7) sets the two statuses it can prove: ``optimum_exportable`` (from task discovery) and
``mobile_package_exportable`` (from a successful normalized export). The remaining canonical statuses
(``train_artifacts_exportable``, ``android_inference_ready``, ``android_training_ready``, ``rag_ready``)
are seeded as ``None`` and flipped by later plans. ``02_code_plans/02`` (#20) owns the canonical schema
+ field list and is the reporting layer that reads/extends this file — so ``merge_row`` preserves any
field it does not own.

Pure stdlib/JSON (no onnx/optimum), so it runs in any profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Envelope versioning (uniform contract, see canonical decisions). #20 finalizes the schema.
SCHEMA_VERSION = "1.0"
MIN_READER_VERSION = "1.0"

#: Statuses #7 owns. The other canonical statuses are seeded None and owned by later plans.
_OWNED_STATUSES = ("optimum_exportable", "mobile_package_exportable")
_DEFERRED_STATUSES = (
    "train_artifacts_exportable",
    "android_inference_ready",
    "android_training_ready",
    "rag_ready",
)


@dataclass
class SupportRow:
    """One model's export status, as far as #7 can determine it."""

    model_id: str
    model_type: str | None
    optimum_exportable: bool
    mobile_package_exportable: bool
    supported_tasks: tuple[str, ...] = ()
    chosen_task: str | None = None
    blocker: str | None = None
    toolchain: dict[str, str] = field(default_factory=dict)

    def owned_fields(self) -> dict[str, Any]:
        """The subset of the wire row this plan owns (merged over any existing row)."""
        return {
            "modelType": self.model_type,
            "supportedTasks": list(self.supported_tasks),
            "chosenTask": self.chosen_task,
            "optimum_exportable": self.optimum_exportable,
            "mobile_package_exportable": self.mobile_package_exportable,
            "blocker": self.blocker,
            "toolchain": dict(self.toolchain),
        }


def empty_matrix() -> dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, "minReaderVersion": MIN_READER_VERSION, "models": {}}


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Load an existing matrix, or a fresh empty one if the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return empty_matrix()
    matrix = json.loads(path.read_text(encoding="utf-8"))
    matrix.setdefault("schemaVersion", SCHEMA_VERSION)
    matrix.setdefault("minReaderVersion", MIN_READER_VERSION)
    matrix.setdefault("models", {})
    return matrix


def merge_row(matrix: dict[str, Any], row: SupportRow) -> dict[str, Any]:
    """Merge ``row`` into ``matrix`` in place, keyed by ``model_id``.

    Updates only the fields #7 owns and preserves everything else (statuses set by later plans).
    Idempotent: merging the same row twice yields the same matrix.
    """
    models = matrix.setdefault("models", {})
    existing = models.get(row.model_id, {})
    # Seed deferred statuses as None the first time we see a model; never clobber a later plan's value.
    for status in _DEFERRED_STATUSES:
        existing.setdefault(status, None)
    existing.update(row.owned_fields())
    models[row.model_id] = existing
    return matrix


def write_matrix(matrix: dict[str, Any], path: str | Path) -> None:
    """Write ``matrix`` deterministically (sorted keys) for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_support_matrix(path: str | Path, row: SupportRow) -> dict[str, Any]:
    """Load → merge ``row`` → write. Returns the updated matrix."""
    matrix = load_matrix(path)
    merge_row(matrix, row)
    write_matrix(matrix, path)
    return matrix


__all__ = [
    "SCHEMA_VERSION",
    "MIN_READER_VERSION",
    "SupportRow",
    "empty_matrix",
    "load_matrix",
    "merge_row",
    "write_matrix",
    "update_support_matrix",
]
