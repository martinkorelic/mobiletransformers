"""#26/#27 embedding (RAG) stage: the pure decisions that gate the on-device retriever.

The stage body itself needs optimum + a network fetch, so it runs under the `export` profile. What is
testable here — and what actually broke the device leg — is the contract the stage must honour: the
pooled vector width the graph will emit, and the fail-closed check that this width is one the Android
vector store can index. A package that pools to an unindexable width installs fine and only fails at
first ingest on device, which is exactly the failure mode the check exists to prevent.
"""

from __future__ import annotations

import pytest

from mobiletransformers.exceptions import ConfigValidationError
from mobiletransformers.export.pipeline import (
    SUPPORTED_EMBEDDING_DIMENSIONS,
    _pooled_embedding_dimension,
)


def test_single_pooling_mode_keeps_the_word_dimension():
    config = {"word_embedding_dimension": 384, "pooling_mode_mean_tokens": True}
    assert _pooled_embedding_dimension(config) == 384


def test_concatenated_modes_multiply_the_width():
    """sentence-transformers concatenates every enabled mode, so two modes emit 2x the word width.

    Reading `word_embedding_dimension` alone would declare 384 in `rag_config.json` while the graph
    emitted 768 — the store would reject every vector at insert time.
    """
    config = {
        "word_embedding_dimension": 384,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_cls_token": True,
    }
    assert _pooled_embedding_dimension(config) == 768


def test_no_active_mode_falls_back_to_the_word_dimension():
    assert _pooled_embedding_dimension({"word_embedding_dimension": 512}) == 512


@pytest.mark.parametrize("config", [{}, {"word_embedding_dimension": 0}, {"word_embedding_dimension": "x"}])
def test_unusable_pooling_config_fails_closed(config):
    with pytest.raises(ConfigValidationError):
        _pooled_embedding_dimension(config)


def test_supported_dimensions_mirror_the_kotlin_registry():
    """Pinned against `rag/VectorStoreRegistry.kt`'s `DimensionRegistry.SUPPORTED_DIMENSIONS`.

    These two lists are the same contract in two languages: the exporter refuses to write a package the
    device cannot index, so they must not drift.
    """
    assert set(SUPPORTED_EMBEDDING_DIMENSIONS) == {64, 128, 256, 384, 512, 768, 1024, 1536}
