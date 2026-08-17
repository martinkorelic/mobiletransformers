package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.views.downloadPhaseLabel
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * What the download card and the model bar say a pull is doing.
 *
 * ### The defect
 *
 * Downloads default to Wi-Fi only. With no Wi-Fi, WorkManager parks the worker in `ENQUEUED` and it
 * waits — correct, deliberate behaviour, and the reason `ModelHolder` went to the trouble of mapping
 * that state to the sentence `"waiting for Wi-Fi"`.
 *
 * [downloadPhaseLabel] then threw it away. It matched `Resolving`/`Verifying`/`Installing` and sent
 * **everything else** to `"Downloading"` — so the app displayed an active download, with a progress
 * bar, that never advanced. The one state the sentence existed to distinguish from a stall was
 * rendered as a stall.
 *
 * Both halves were individually right; nothing tested the seam. Reported from a real phone on
 * 2026-08-17, not by any suite.
 *
 * ### Why the flag, and why it is checked first
 *
 * An enqueued job still reports whatever phase it last reached, so a waiting pull can legitimately
 * carry `phase == "Downloading"`. Matching on the phase string alone cannot distinguish the two —
 * which is why [DownloadUi.waitingForConstraints] is a boolean and why it takes precedence.
 */
class DownloadPhaseLabelTest {

    @Test
    fun `a job waiting on its network constraint says so, whatever phase it last reached`() {
        // The exact shape that broke: the worker is parked, but the last phase it reported was a
        // download in progress. Matching the phase alone renders this as "Downloading".
        assertEquals(
            "Waiting for Wi-Fi",
            downloadPhaseLabel("Downloading", waitingForConstraints = true),
        )
        assertEquals(
            "Waiting for Wi-Fi",
            downloadPhaseLabel("Resolving", waitingForConstraints = true),
        )
    }

    @Test
    fun `a running job reports its real phase`() {
        assertEquals("Resolving", downloadPhaseLabel("Resolving", waitingForConstraints = false))
        assertEquals("Verifying", downloadPhaseLabel("Verifying", waitingForConstraints = false))
        assertEquals("Installing", downloadPhaseLabel("Installing", waitingForConstraints = false))
    }

    @Test
    fun `an unrecognized phase still reads as downloading rather than as a raw enum name`() {
        // The `else` branch is deliberate: the SDK's phase vocabulary can grow, and a user should see
        // "Downloading" rather than `RUNNING` or `ENQUEUED`.
        assertEquals("Downloading", downloadPhaseLabel("SomeFuturePhase", waitingForConstraints = false))
    }

    @Test
    fun `the default keeps existing callers on the running path`() {
        assertEquals("Downloading", downloadPhaseLabel("Downloading"))
    }
}
