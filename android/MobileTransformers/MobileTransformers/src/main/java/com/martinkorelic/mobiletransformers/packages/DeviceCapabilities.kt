package com.martinkorelic.mobiletransformers.packages

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/**
 * What this device can run, for `VariantSelector` to select against before a byte is downloaded.
 *
 * Shared rather than duplicated because omitting it is silent and expensive. `HubDownloader` falls
 * back to `manifest.defaultVariant` when `abis` is empty — "select on device capability" degrades to
 * "take whatever the publisher listed first" — so an incompatible variant downloads happily and only
 * fails at load, after the user has waited for a multi-gigabyte transfer.
 *
 * That is exactly what `PackageDownloadWorker` did: `MobileTransformers.fromPretrained` passed both
 * values and the worker passed neither, so the two download paths disagreed about which variant this
 * phone can run. One copy, called by both.
 */
internal object DeviceCapabilities {

    /** ABIs this device supports, most-preferred first. Empty only on a device that reports none. */
    fun abis(): List<String> = Build.SUPPORTED_ABIS?.toList() ?: emptyList()

    /**
     * Total physical RAM in MB, or null when it cannot be read.
     *
     * Null is a real answer and must stay distinguishable from zero: `VariantSelector` treats null as
     * "unknown, do not filter on memory", whereas 0 would reject every variant that declares a
     * minimum.
     */
    fun totalMemoryMb(context: Context): Int? =
        runCatching {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val info = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
            (info.totalMem / (1024L * 1024L)).toInt()
        }.getOrNull()

    /**
     * The download groups a feature set implies — the wire names `HubDownloader` plans files from.
     *
     * `inference` is unconditional: every package has one, and a pull that omitted it would install a
     * training stage with nothing to train against. `GenAI`/`ManualInference` select an engine over
     * the shared package rather than adding a group, which is why they appear nowhere here.
     */
    fun downloadGroups(features: Set<ModelFeature>): Set<String> = buildSet {
        add("inference")
        if (ModelFeature.Training in features) add("train")
        if (ModelFeature.Rag in features || ModelFeature.Embedding in features) add("rag")
    }
}
