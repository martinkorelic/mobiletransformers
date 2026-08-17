package com.martinkorelic.mobiletransformers.hub

/**
 * #21: builds Hugging Face Hub `resolve` URLs and the auth header. Pure (JVM-testable).
 */
object HubResolver {
    const val DEFAULT_ENDPOINT = "https://huggingface.co"

    /** `<endpoint>/<repoId>/resolve/<revision>/<repoRelativePath>`. */
    fun fileUrl(endpoint: String, repoId: String, revision: String, path: String): String =
        "${endpoint.trimEnd('/')}/$repoId/resolve/$revision/$path"

    fun authHeaders(token: String?): Map<String, String> =
        if (token.isNullOrBlank()) emptyMap() else mapOf("Authorization" to "Bearer $token")
}
