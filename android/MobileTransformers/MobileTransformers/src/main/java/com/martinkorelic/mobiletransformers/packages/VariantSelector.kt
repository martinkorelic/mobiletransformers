package com.martinkorelic.mobiletransformers.packages

/**
 * Device-capability variant selection (#13) — mirror of the Python
 * `artifacts.manifest.MobileTransformersManifest.select_variant`. Pure (device caps are parameters, so
 * this is JVM-unit-testable; the caller passes `Build.SUPPORTED_ABIS` / `ActivityManager` memory).
 */
object VariantSelector {
    fun select(
        manifest: MobileTransformersManifest,
        abis: List<String>,
        quantization: String? = null,
        totalMemMb: Int? = null,
        requestedFeatures: List<String> = emptyList(),
        requestedEngine: String = "native",
    ): MobileTransformersManifest.Variant {
        val abiSet = abis.toSet()
        val reqFeatures = requestedFeatures.toSet()
        val candidates = manifest.variants.filter { v ->
            (v.abi == null || v.abi.any { it in abiSet }) &&
                (quantization == null || v.quantization == quantization) &&
                (totalMemMb == null || v.recommendedDeviceMemoryMb == null || v.recommendedDeviceMemoryMb <= totalMemMb) &&
                reqFeatures.all { it in v.features } &&
                requestedEngine in v.supportedEngines
        }
        if (candidates.isEmpty()) {
            throw NoCompatibleVariantException(
                "no variant matches abis=$abis quant=$quantization mem=$totalMemMb " +
                    "features=$requestedFeatures engine='$requestedEngine'",
            )
        }
        // Tie-break: smallest recommended memory, then the defaultVariant, then id order.
        return candidates.minWith(
            compareBy(
                { it.recommendedDeviceMemoryMb ?: Int.MAX_VALUE },
                { if (it.id == manifest.defaultVariant) 0 else 1 },
                { it.id },
            ),
        )
    }
}
