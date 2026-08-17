"""MobileTransformers Python exception hierarchy.

Deliberately parallel to the Kotlin facade's ``MobileTransformersException`` hierarchy
so errors read the same on both sides.
Library code raises a typed subclass — never a bare ``Exception``.
"""

from __future__ import annotations


class MobileTransformersError(Exception):
    """Root of every MobileTransformers error. Catch this to catch anything the library raises."""


class ConfigValidationError(MobileTransformersError):
    """A config object or file failed validation (bad/missing/typed-wrong fields)."""


class ExportError(MobileTransformersError):
    """An ONNX/GenAI/mobile export step failed."""


class ManifestError(MobileTransformersError):
    """A ``mobiletransformers_manifest.json`` is missing, malformed, or version-incompatible."""


class NoCompatibleVariant(ManifestError):
    """No package variant satisfies the requested device capabilities / features / engine."""


class HandoffError(MobileTransformersError):
    """A ``weight_handoff_map.json`` lookup/contract could not be satisfied."""


class MergeError(MobileTransformersError):
    """An adapter/weight merge (offline or on-device) failed."""


class UnsupportedModelError(MobileTransformersError):
    """The requested architecture/task/feature is not supported in this version."""


class HubError(MobileTransformersError):
    """A Hugging Face Hub download/upload/auth operation failed."""


__all__ = [
    "MobileTransformersError",
    "ConfigValidationError",
    "ExportError",
    "ManifestError",
    "NoCompatibleVariant",
    "HandoffError",
    "MergeError",
    "UnsupportedModelError",
    "HubError",
]
