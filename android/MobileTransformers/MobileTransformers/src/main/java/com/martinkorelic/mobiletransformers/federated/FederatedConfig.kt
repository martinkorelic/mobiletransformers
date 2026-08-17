package com.martinkorelic.mobiletransformers.federated

import com.martinkorelic.mobiletransformers.MobileTransformersException

/** A federated round was requested without a precondition that protects the user. */
class FederatedConsentException(message: String) : MobileTransformersException(message)

/**
 * What a user actually agreed to. Absence is refusal.
 *
 * Modelled as a record with a timestamp and a version rather than a boolean because consent is not a
 * flag: it is given for a stated purpose at a point in time, and a change to what is shared invalidates
 * it. [policyVersion] is what makes that enforceable — a round declaring a newer policy than the one
 * the user saw must stop, not proceed on the old agreement.
 */
data class FederatedConsent(
    val granted: Boolean,
    val policyVersion: String,
    val grantedAtEpochMs: Long,
) {
    companion object {
        /** The absence of consent, which is the default state of every device. */
        val NONE = FederatedConsent(granted = false, policyVersion = "", grantedAtEpochMs = 0L)
    }
}

/**
 * Preconditions for participating in a federated round.
 *
 * These are a **precondition to any real-user run, not a follow-up**, so they
 * are enforced at the point a round starts rather than documented and hoped for. Before this the repo
 * had exactly one flag (`BuildConfig.ADAPTER_UPLOAD_ENABLED`) and no notion of consent at all — a grep
 * for "consent" returned nothing.
 *
 * @property gatewayUrl must be `https://`. Plain HTTP would put adapter factors — which are derived
 *   from the user's own data — on the wire in clear.
 * @property clientAuthToken bearer credential. Without client auth a gateway cannot tell a participant
 *   from anyone who found the URL, so "only this cohort contributes" would be unenforceable.
 * @property clipNorm L2 bound applied to an update before it leaves the device. Federated updates leak
 *   information about the data that produced them; clipping bounds any single example's influence and
 *   is the precondition for meaningful local DP noise.
 * @property dpNoiseMultiplier Gaussian noise scale relative to [clipNorm]. `0.0` means **no local DP**,
 *   which is a legitimate configuration for a closed test cohort but must be a deliberate choice — so
 *   it is required to be stated rather than defaulted.
 */
data class FederatedConfig(
    val gatewayUrl: String,
    val clientAuthToken: String,
    val consent: FederatedConsent = FederatedConsent.NONE,
    val clipNorm: Double = 1.0,
    val dpNoiseMultiplier: Double = 0.0,
    val policyVersion: String = "1.0",
) {
    /**
     * Throws unless every precondition holds. Call before a round does anything observable.
     *
     * Fail-closed and specific: each message says which protection is missing, because "federated round
     * refused" gives an integrator nothing to act on.
     */
    fun requireRoundIsPermitted() {
        if (!FEDERATION_ENABLED) {
            throw FederatedConsentException(
                "federated participation is disabled in this build " +
                    "(BuildConfig.FEDERATION_ENABLED=false); it is off by default and must be " +
                    "enabled deliberately by the app that ships it"
            )
        }
        if (!consent.granted) {
            throw FederatedConsentException(
                "no user consent on record for federated training; a round must not start without it"
            )
        }
        if (consent.policyVersion != policyVersion) {
            throw FederatedConsentException(
                "consent was given for policy version '${consent.policyVersion}' but this round " +
                    "declares '$policyVersion'. What is shared has changed since the user agreed — " +
                    "ask again rather than proceeding on the old agreement."
            )
        }
        if (!gatewayUrl.startsWith("https://")) {
            throw FederatedConsentException(
                "gateway URL '$gatewayUrl' is not https; adapter updates are derived from the user's " +
                    "own data and must not travel in clear"
            )
        }
        if (clientAuthToken.isBlank()) {
            throw FederatedConsentException(
                "no client auth token; an unauthenticated gateway cannot tell a participant from " +
                    "anyone who found the URL"
            )
        }
        if (clipNorm <= 0.0) {
            throw FederatedConsentException(
                "clipNorm must be > 0 (got $clipNorm): an unclipped update lets a single example " +
                    "dominate what leaves the device"
            )
        }
        if (dpNoiseMultiplier < 0.0) {
            throw FederatedConsentException("dpNoiseMultiplier must be >= 0 (got $dpNoiseMultiplier)")
        }
    }

    /** True when this configuration adds local differential-privacy noise. Recorded, not assumed. */
    val usesLocalDp: Boolean get() = dpNoiseMultiplier > 0.0

    internal companion object {
        /**
         * Off by default, mirroring `BuildConfig.ADAPTER_UPLOAD_ENABLED` (#22).
         *
         * Read reflectively so this class stays unit-testable and so the library does not fail to link
         * in a consumer whose BuildConfig predates the field. Absent means **false** — the safe value.
         *
         * `internal` rather than private so the test suites can *read* the build's answer and skip
         * accordingly (the device round-trip needs it on, the refusal tests need it off). Reading it is
         * not a way to change it: the value comes from `BuildConfig`, i.e. from the Gradle invocation.
         */
        internal val FEDERATION_ENABLED: Boolean = runCatching {
            Class.forName("com.martinkorelic.mobiletransformers.BuildConfig")
                .getField("FEDERATION_ENABLED")
                .getBoolean(null)
        }.getOrDefault(false)
    }
}
