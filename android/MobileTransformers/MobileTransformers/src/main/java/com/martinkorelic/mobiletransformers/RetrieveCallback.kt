package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.runtime.RetrievalResult

/**
 * Public retrieval callback for [MobileTransformerModel.retrieve] (#19). Mirrors the internal
 * `RagCallback`, delivering the neutral [RetrievalResult] (never `RagResult`/`RagMatch`/`ORT*`).
 */
interface RetrieveCallback {
    fun onQueryResults(result: RetrievalResult) {}

    fun onQueryEnd() {}

    fun onError(error: Throwable) {}
}
