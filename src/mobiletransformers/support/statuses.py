"""Ordered readiness statuses + inheritance for the support matrix (#20).

Six statuses along the readiness pipeline; each *implies* all earlier ones. The moment one is false,
every later status is forced false and a blocker is recorded — so a row can never claim it trains on a
device it cannot even export.
"""

from __future__ import annotations

from enum import Enum


class SupportStatus(str, Enum):
    OPTIMUM_EXPORTABLE = "optimum_exportable"
    MOBILE_PACKAGE_EXPORTABLE = "mobile_package_exportable"
    TRAIN_ARTIFACTS_EXPORTABLE = "train_artifacts_exportable"
    ANDROID_INFERENCE_READY = "android_inference_ready"
    ANDROID_TRAINING_READY = "android_training_ready"
    RAG_READY = "rag_ready"


#: Canonical order; a later status may only be true if every earlier one is true.
STATUS_ORDER: tuple[str, ...] = tuple(s.value for s in SupportStatus)

#: Statuses that surface in user-facing starter-zoo docs (earlier ones are contributor-only).
USER_FACING_STATUSES: frozenset[str] = frozenset(
    {
        SupportStatus.ANDROID_INFERENCE_READY.value,
        SupportStatus.ANDROID_TRAINING_READY.value,
        SupportStatus.RAG_READY.value,
    }
)


def apply_inheritance(statuses: dict[str, bool]) -> dict[str, bool]:
    """Return a copy where every status after the first ``false`` is forced ``false``.

    Missing keys default to ``false``. The result always has all six keys in ``STATUS_ORDER``.
    """
    result: dict[str, bool] = {}
    blocked = False
    for key in STATUS_ORDER:
        if blocked:
            result[key] = False
            continue
        value = bool(statuses.get(key, False))
        result[key] = value
        if not value:
            blocked = True
    return result


def first_blocked(statuses: dict[str, bool]) -> str | None:
    """Name of the earliest status that is ``false`` (the first blocker), or ``None`` if all true."""
    for key in STATUS_ORDER:
        if not statuses.get(key, False):
            return key
    return None


__all__ = [
    "SupportStatus",
    "STATUS_ORDER",
    "USER_FACING_STATUSES",
    "apply_inheritance",
    "first_blocked",
]
