"""Manifest validator + variant selection (#13) — the read/validate/select half of the package contract.

The manifest *schema/field-list* is owned by #14 (``hub/package_format.py``); this module owns the
**validator**, the **variant-selection** algorithm, and (on the Kotlin side, mirrored) the cache-install
semantics. It reuses the one canonical ``check_compat`` (``artifacts/versioning.py``) with a
``MANIFEST_READER_VERSION`` and the handoff-map contract (``artifacts/handoff_map.py``) to assert every
``externalDataLocation`` a variant advertises actually resolves to a file on disk.

``select_variant`` is the language-agnostic algorithm the Kotlin ``VariantSelector`` mirrors byte-for-byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.handoff_map import HandoffMap
from mobiletransformers.artifacts.versioning import check_compat
from mobiletransformers.exceptions import ManifestError, NoCompatibleVariant

#: Oldest manifest schema this reader understands (see ``check_compat``).
MANIFEST_READER_VERSION = "1.0"


@dataclass(frozen=True)
class SelectedVariant:
    """The variant chosen by :meth:`MobileTransformersManifest.select_variant`."""

    id: str
    execution_provider: str
    quantization: str
    supported_engines: tuple[str, ...]
    features: tuple[str, ...]
    paths: dict[str, str]
    weight_handoff: str
    recommended_device_memory_mb: int | None


@dataclass
class MobileTransformersManifest:
    """A parsed ``mobiletransformers_manifest.json``. Unknown fields are preserved for round-trip."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MobileTransformersManifest:
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        return cls(data=data)

    @classmethod
    def load(cls, path: str | Path) -> MobileTransformersManifest:
        path = Path(path)
        if not path.is_file():
            raise ManifestError(f"manifest not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=True) + "\n"

    # -- typed accessors ----------------------------------------------------
    @property
    def schema_version(self) -> str:
        return self.data.get("schemaVersion", "")

    @property
    def min_reader_version(self) -> str:
        return self.data.get("minReaderVersion", "")

    @property
    def default_variant(self) -> str:
        return self.data.get("defaultVariant", "")

    @property
    def variants(self) -> list[dict[str, Any]]:
        return self.data.get("variants", [])

    def _variant(self, variant_id: str) -> dict[str, Any] | None:
        return next((v for v in self.variants if v.get("id") == variant_id), None)

    # -- validation ---------------------------------------------------------
    def validate(self, package_dir: str | Path) -> None:
        """Fail closed (``ManifestError``) unless the manifest is version-compatible, internally
        consistent, and every advertised file/handoff tensor resolves on disk."""
        package_dir = Path(package_dir)
        # F1 schema gate (raises SchemaVersionError, a MobileTransformersError, on incompatibility).
        check_compat(self.schema_version, self.min_reader_version, MANIFEST_READER_VERSION)

        if not self.variants:
            raise ManifestError("manifest declares no variants")
        if self._variant(self.default_variant) is None:
            raise ManifestError(
                f"defaultVariant {self.default_variant!r} is not among variants "
                f"{[v.get('id') for v in self.variants]}"
            )

        for v in self.variants:
            vid = v.get("id")
            paths = v.get("paths", {})
            features = set(v.get("features", ()))
            # Every claimed feature that needs a subtree must have a path entry.
            for feature, required_path in (
                ("train", "train"),
                ("inference", "inference"),
                ("rag", "embedding"),
            ):
                if feature in features and required_path not in paths:
                    raise ManifestError(
                        f"variant {vid!r} claims feature {feature!r} but has no '{required_path}' path"
                    )
            # weightHandoff must resolve, and every externalDataLocation it names must exist.
            self._validate_handoff(package_dir, vid, v.get("weightHandoff"), paths.get("inference"))

        # requiredFiles floor.
        for rel in self.data.get("requiredFiles", []):
            if not (package_dir / rel).exists():
                raise ManifestError(f"requiredFile missing on disk: {rel}")

    def _validate_handoff(
        self, package_dir: Path, variant_id: Any, handoff_rel: str | None, inference_rel: str | None
    ) -> None:
        if not handoff_rel:
            raise ManifestError(f"variant {variant_id!r} has no weightHandoff pointer")
        handoff_path = package_dir / handoff_rel
        if not handoff_path.is_file():
            raise ManifestError(f"variant {variant_id!r} weightHandoff does not resolve: {handoff_rel}")
        inference_dir = (package_dir / inference_rel) if inference_rel else handoff_path.parent
        handoff = HandoffMap.load(handoff_path)  # runs check_compat + validate() on the handoff itself
        for entry in handoff.entries:
            for role, location in entry.external_data_location.items():
                if not (inference_dir / location).is_file():
                    raise ManifestError(
                        f"variant {variant_id!r} handoff entry {entry.training_base_layer_name!r} "
                        f"role {role!r} points at missing external file: {location}"
                    )

    # -- variant selection --------------------------------------------------
    def select_variant(
        self,
        *,
        abis: tuple[str, ...] | list[str],
        quantization: str | None = None,
        total_mem_mb: int | None = None,
        requested_features: tuple[str, ...] | list[str] = (),
        requested_engine: str = "native",
    ) -> SelectedVariant:
        """Pick the best variant for the given device caps + requests, or raise ``NoCompatibleVariant``.

        Filters: ABI overlap (variant ``abi=null`` means any), quantization (if requested), memory
        (``recommendedDeviceMemoryMb <= total_mem_mb``), features ⊇ requested, engine ∈ supportedEngines.
        Tie-break: smallest ``recommendedDeviceMemoryMb``, then the ``defaultVariant``, then id order.
        """
        abis_set = set(abis)
        req_features = set(requested_features)
        candidates: list[dict[str, Any]] = []
        for v in self.variants:
            v_abi = v.get("abi")
            if v_abi is not None and not (abis_set & set(v_abi)):
                continue
            if quantization is not None and v.get("quantization") != quantization:
                continue
            mem = v.get("recommendedDeviceMemoryMb")
            if total_mem_mb is not None and mem is not None and mem > total_mem_mb:
                continue
            if not req_features.issubset(set(v.get("features", ()))):
                continue
            if requested_engine not in set(v.get("supportedEngines", ())):
                continue
            candidates.append(v)

        if not candidates:
            raise NoCompatibleVariant(
                f"no variant matches abis={sorted(abis_set)} quant={quantization} "
                f"mem={total_mem_mb} features={sorted(req_features)} engine={requested_engine!r}"
            )

        def _key(v: dict[str, Any]) -> tuple[int, int, str]:
            mem = v.get("recommendedDeviceMemoryMb")
            return (
                mem if mem is not None else 1 << 30,
                0 if v.get("id") == self.default_variant else 1,
                str(v.get("id")),
            )

        best = min(candidates, key=_key)
        return SelectedVariant(
            id=best["id"],
            execution_provider=best.get("executionProvider", ""),
            quantization=best.get("quantization", ""),
            supported_engines=tuple(best.get("supportedEngines", ())),
            features=tuple(best.get("features", ())),
            paths=dict(best.get("paths", {})),
            weight_handoff=best.get("weightHandoff", ""),
            recommended_device_memory_mb=best.get("recommendedDeviceMemoryMb"),
        )


__all__ = [
    "MANIFEST_READER_VERSION",
    "SelectedVariant",
    "MobileTransformersManifest",
]
