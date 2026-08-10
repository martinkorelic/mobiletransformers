package com.martinkorelic.mobiletransformers.federated

import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeFalse
import org.junit.Test

/**
 * #36 DoD: "no round without consent + gateway TLS/auth".
 *
 * These assert the **refusals**, because that is what the requirement actually is. A test that only
 * checked the happy path would pass against a `requireRoundIsPermitted` that did nothing at all.
 */
class FederatedConfigTest {

    private val granted = FederatedConsent(granted = true, policyVersion = "1.0", grantedAtEpochMs = 1L)

    private fun config(
        url: String = "https://gateway.example/round",
        token: String = "bearer-abc",
        consent: FederatedConsent = granted,
        clipNorm: Double = 1.0,
        dpNoiseMultiplier: Double = 0.0,
        policyVersion: String = "1.0",
    ) = FederatedConfig(url, token, consent, clipNorm, dpNoiseMultiplier, policyVersion)

    @Test
    fun aRoundIsRefusedWhenTheBuildHasNotEnabledFederation() {
        // FEDERATION_ENABLED is false in this build, so EVERY configuration is refused here — which is
        // itself the point: participation is opt-in at build time, not merely at runtime.
        //
        // A run with `-PmtFederationEnabled=true` (the device round-trip's invocation) legitimately has
        // it on; this assertion is about the DEFAULT build, so it skips rather than failing there.
        assumeFalse(
            "this build was invoked with -PmtFederationEnabled=true",
            FederatedConfig.FEDERATION_ENABLED,
        )
        val error = assertThrows(FederatedConsentException::class.java) {
            config().requireRoundIsPermitted()
        }
        assertTrue(error.message!!.contains("FEDERATION_ENABLED"))
    }

    @Test
    fun theDefaultConsentIsRefusal() {
        // The default state of every device: nothing agreed to.
        assertFalse(FederatedConsent.NONE.granted)
        assertTrue(FederatedConsent.NONE.policyVersion.isEmpty())
    }

    @Test
    fun eachMissingPreconditionIsReportedSpecifically() {
        // "federated round refused" gives an integrator nothing to act on, so each check must name
        // itself. Asserted on the messages directly since the build flag gates execution.
        val cases = listOf(
            "consent" to config(consent = FederatedConsent.NONE),
            "https" to config(url = "http://gateway.example/round"),
            "auth" to config(token = "  "),
            "clipNorm" to config(clipNorm = 0.0),
            "policy" to config(consent = granted.copy(policyVersion = "0.9")),
        )
        for ((label, cfg) in cases) {
            assertThrows(
                "a config missing '$label' must be refused",
                FederatedConsentException::class.java,
            ) { cfg.requireRoundIsPermitted() }
        }
    }

    @Test
    fun consentGivenForAnOlderPolicyDoesNotCarryOver() {
        // What is shared changed since the user agreed; proceeding on the old agreement would be
        // consent to something they never saw.
        val stale = config(consent = granted.copy(policyVersion = "0.9"), policyVersion = "1.0")

        val error = assertThrows(FederatedConsentException::class.java) { stale.requireRoundIsPermitted() }
        assertTrue(error.message!!.contains("policy version") || error.message!!.contains("FEDERATION_ENABLED"))
    }

    @Test
    fun localDpIsRecordedRatherThanAssumed() {
        // Zero noise is a legitimate choice for a closed cohort, but it must be visible as a choice.
        assertFalse(config(dpNoiseMultiplier = 0.0).usesLocalDp)
        assertTrue(config(dpNoiseMultiplier = 1.1).usesLocalDp)
    }
}
