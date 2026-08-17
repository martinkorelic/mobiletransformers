package com.martinkorelic.mobiletransformers

data class TokenAttributes(
    var tokenId : Int?,
    val content: String = "",
    val lstrip: Boolean = false,
    val normalized: Boolean = false,
    val rstrip: Boolean = false,
    val single_word: Boolean = false
)

data class SpecialTokensMap(val tokens: Map<String, TokenAttributes>)