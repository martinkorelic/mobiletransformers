package com.martinkorelic.mobiletransformers.runtime

import com.martinkorelic.mobiletransformers.NativeLibrary

/**
 * Process resident-set-size sampler (#12, Gate 0.2).
 *
 * Reads `VmRSS` from `/proc/self/status` in native code. `Debug.getPss()` measures the JVM's accounting
 * of the process; the weight blobs are mapped by native code outside it, so the gate's four-point table
 * (base/merged x copy/mmap) is specified against `VmRSS`.
 *
 * The zero-copy load is default-off and toggled with `adb shell setprop debug.mtf.mmap_weights 1` (or
 * the `MTF_MMAP_WEIGHTS` environment variable off-device). [mmapWeightsEnabled] reports what native
 * code actually resolved, so a harness can assert the flip took effect instead of assuming it.
 */
object MemoryProbe {

    init {
        NativeLibrary.ensureLoaded()
    }

    /** Resident set size in KiB, or -1 when `/proc/self/status` is unreadable. */
    fun currentRssKb(): Long = runCatching { nativeCurrentRssKb() }.getOrDefault(-1L)

    /** True when the next weight load will take the mmap branch. */
    fun mmapWeightsEnabled(): Boolean = runCatching { nativeMmapWeightsEnabled() }.getOrDefault(false)

    private external fun nativeCurrentRssKb(): Long

    private external fun nativeMmapWeightsEnabled(): Boolean
}
