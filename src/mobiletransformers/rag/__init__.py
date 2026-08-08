"""Python-side RAG / vector-database helpers (Migration Map S7, formerly the ``database/`` root).

Builds and queries the ObjectBox vector store that ships beside a package's ``embedding/`` stage. The
**on-device** store is the Android module's ObjectBox (``ORTVectorDatabase``); this half prepares and
inspects it on a host, so a corpus can be embedded once and pushed rather than indexed on a phone.

Modules:

* :mod:`~mobiletransformers.rag.vector_entity` — the fixed-dimension entity classes (64…1536). The
  dimensions here must stay in step with the Kotlin ``DimensionRegistry``; a package whose encoder
  emits an unlisted dimension cannot be indexed on device.
* :mod:`~mobiletransformers.rag.builder` — document ingestion, chunking and embedding precompute.
* :mod:`~mobiletransformers.rag.query` — similarity/text search over a built store.
* :mod:`~mobiletransformers.rag.json2entity` — ObjectBox model-JSON <-> entity UID plumbing.

Imports are deliberately NOT re-exported at package level: these modules need ``objectbox`` (and
``builder`` additionally needs LangChain), which the core profile does not install. Importing this
package must stay cheap, so callers import the submodule they need.

``database/`` still holds deprecation shims re-exporting these names; they are removed in S9.
"""
