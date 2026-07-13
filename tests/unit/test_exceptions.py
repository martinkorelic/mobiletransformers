"""The exception hierarchy is rooted at MobileTransformersError and mirrors the Kotlin names."""

from __future__ import annotations

import pytest

from mobiletransformers.exceptions import (
    ConfigValidationError,
    ExportError,
    HandoffError,
    HubError,
    ManifestError,
    MergeError,
    MobileTransformersError,
    UnsupportedModelError,
)

_SUBCLASSES = [
    ConfigValidationError,
    ExportError,
    ManifestError,
    HandoffError,
    MergeError,
    UnsupportedModelError,
    HubError,
]


@pytest.mark.parametrize("exc", _SUBCLASSES)
def test_every_error_derives_from_root(exc):
    assert issubclass(exc, MobileTransformersError)


@pytest.mark.parametrize("exc", _SUBCLASSES)
def test_catchable_as_root(exc):
    with pytest.raises(MobileTransformersError):
        raise exc("boom")


def test_root_is_an_exception():
    assert issubclass(MobileTransformersError, Exception)
