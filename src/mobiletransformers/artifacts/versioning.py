"""The ONE canonical ``check_compat`` for every schema-versioned cross-boundary contract.

Owned by #8 (weight-handoff-map plan), reused by the manifest (#13), support matrix (#20), and the
federated record (#35). Mirrored byte-for-byte in Kotlin/C++ (a shared table-driven fixture,
``tests/fixtures/check_compat_cases.json``, pins the behaviour across languages).

Semantics (do not improvise): versions are ``"MAJOR.MINOR"`` strings; comparison is on the
``(major, minor)`` **integer tuple**, never string comparison. A document with a *lower* major than the
reader is accepted (readers keep old-major compatibility until a deliberate major bump); a *higher*
document minor is accepted (additive fields, ignored). Fail closed on: malformed versions, a document
major beyond the reader, or a reader below the document's ``minReaderVersion``.
"""

from __future__ import annotations

from mobiletransformers.exceptions import MobileTransformersError


class SchemaVersionError(MobileTransformersError):
    """A versioned contract is incompatible with this SDK (needs a newer/older reader)."""


def parse_version(version: str) -> tuple[int, int]:
    """Parse ``"MAJOR.MINOR"`` into ``(major, minor)``. Fail closed on anything malformed."""
    parts = str(version).split(".")
    if len(parts) != 2:
        raise SchemaVersionError(f"malformed schema version {version!r} (expected 'MAJOR.MINOR')")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SchemaVersionError(f"non-integer schema version {version!r}") from exc
    if major < 0 or minor < 0:
        raise SchemaVersionError(f"negative schema version {version!r}")
    return major, minor


def check_compat(doc_schema_version: str, doc_min_reader_version: str, reader_schema_version: str) -> None:
    """Raise :class:`SchemaVersionError` unless a reader at ``reader_schema_version`` can read a
    document at ``doc_schema_version`` that requires ``doc_min_reader_version``. Returns ``None`` on
    accept (fail-closed: only accepts explicitly)."""
    doc_major, doc_minor = parse_version(doc_schema_version)
    req_major, req_minor = parse_version(doc_min_reader_version)
    rdr_major, rdr_minor = parse_version(reader_schema_version)

    if doc_major > rdr_major:
        raise SchemaVersionError(
            f"document schema v{doc_major}.{doc_minor} needs a newer SDK (reader supports major {rdr_major})"
        )
    if (rdr_major, rdr_minor) < (req_major, req_minor):
        raise SchemaVersionError(
            f"document requires reader >= {doc_min_reader_version}; this SDK is {reader_schema_version}"
        )


__all__ = ["SchemaVersionError", "parse_version", "check_compat"]
