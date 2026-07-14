"""Constraint-based variant selection for hub pull (#21).

Layers the #21 download-time policy — soft quantization preference, download-size tie-break, and a
storage-budget ceiling — over #13's hard-filter :meth:`MobileTransformersManifest.select_variant`
(ABI / engine / features / memory). The algorithm is deterministic and mirrored by the Android
`VariantSelector.kt` (device leg, deferred).
"""

from __future__ import annotations

from dataclasses import dataclass

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.exceptions import NoCompatibleVariant
from mobiletransformers.hub.package_format import FEATURE_GROUPS

#: Fraction of free storage a package may occupy before selection fails closed.
_STORAGE_BUDGET_FRACTION = 0.9


@dataclass(frozen=True)
class Constraints:
    """Device/desktop capabilities + requests that drive variant selection."""

    abi: tuple[str, ...] = ("arm64-v8a",)
    preferred_quantization: str = "int4"
    engine: str = "native"
    requested_features: tuple[str, ...] = ("core", "inference")
    available_storage_bytes: int | None = None
    device_memory_mb: int | None = None
    extra_abis_any: bool = False  # if True, an abi=null variant always matches (desktop pull)


def default_desktop_constraints() -> Constraints:
    """Permissive constraints for a desktop `mobiletransformers pull` (any ABI, no memory ceiling)."""
    return Constraints(abi=("arm64-v8a", "x86_64"), available_storage_bytes=None, device_memory_mb=None)


def _variant_download_bytes(manifest: MobileTransformersManifest, variant_id: str, features: set[str]) -> int:
    """Sum ``fileSizes`` over the files the variant's ``downloadPlan`` would pull for ``features``."""
    data = manifest.to_dict()
    file_sizes: dict[str, int] = data.get("fileSizes", {})
    plan = data.get("downloadPlan", {}).get(variant_id, {})
    groups = set(features) | {"core", "checksums"}
    total = 0
    seen: set[str] = set()
    for group in groups:
        for pattern in plan.get(group, []):
            prefix = pattern[:-3] if pattern.endswith("/**") else None
            for rel, size in file_sizes.items():
                if rel in seen:
                    continue
                if (prefix is not None and rel.startswith(prefix)) or rel == pattern:
                    seen.add(rel)
                    total += size
    return total


def select_variant(manifest: MobileTransformersManifest, constraints: Constraints) -> str:
    """Return the chosen variant id, or raise ``NoCompatibleVariant``.

    Hard filters (ABI / engine / features / memory) come from #13's ``select_variant``; among the
    survivors this prefers ``preferred_quantization``, then smallest download size, then the
    ``defaultVariant``. Fails closed if the estimated download exceeds
    ``available_storage_bytes * _STORAGE_BUDGET_FRACTION``.
    """
    features = set(constraints.requested_features) | {"core", "inference"}
    # Reuse #13's hard-filter selector to get *a* compatible variant + prove compatibility exists.
    #: (#13 raises NoCompatibleVariant if nothing passes the hard filters.)
    manifest.select_variant(
        abis=list(constraints.abi),
        total_mem_mb=constraints.device_memory_mb,
        requested_features=list(features),
        requested_engine=constraints.engine,
    )

    # Re-derive the full candidate set here so we can apply the soft preference + size tie-break.
    abi_set = set(constraints.abi)
    candidates = [
        v
        for v in manifest.variants
        if (v.get("abi") is None or set(v.get("abi") or []) & abi_set)
        and features.issubset(set(v.get("features", ())))
        and constraints.engine in set(v.get("supportedEngines", ()))
        and (
            constraints.device_memory_mb is None
            or v.get("recommendedDeviceMemoryMb") is None
            or v["recommendedDeviceMemoryMb"] <= constraints.device_memory_mb
        )
    ]
    if not candidates:  # pragma: no cover - #13 selector already guarantees non-empty
        raise NoCompatibleVariant("no compatible variant after hard filters")

    def _key(v: dict) -> tuple[int, int, int, str]:
        return (
            0 if v.get("quantization") == constraints.preferred_quantization else 1,
            _variant_download_bytes(manifest, v["id"], features),
            0 if v["id"] == manifest.default_variant else 1,
            str(v["id"]),
        )

    chosen = min(candidates, key=_key)
    chosen_id = chosen["id"]

    if constraints.available_storage_bytes is not None:
        need = _variant_download_bytes(manifest, chosen_id, features)
        budget = int(constraints.available_storage_bytes * _STORAGE_BUDGET_FRACTION)
        if need > budget:
            raise NoCompatibleVariant(
                f"variant {chosen_id!r} needs ~{need} bytes but budget is {budget} "
                f"({_STORAGE_BUDGET_FRACTION:.0%} of {constraints.available_storage_bytes})"
            )
    return chosen_id


__all__ = ["Constraints", "default_desktop_constraints", "select_variant", "FEATURE_GROUPS"]
