package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.federated.FederatedConfig
import com.martinkorelic.mobiletransformers.federated.FederatedConsent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * #35/#36 — one federated round: import → train locally → export, with the consent gate visible.
 *
 * ### The disabled state is the honest default
 *
 * `BuildConfig.FEDERATION_ENABLED` is **false** in shipped builds, and this screen shows that rather
 * than hiding the feature. Pressing Run in such a build produces a `FederatedConsentException` naming
 * the missing protection, which is exactly what an integrator needs to see — so the screen surfaces the
 * refusal instead of pre-emptively greying everything out and explaining nothing.
 *
 * ### Nothing is uploaded
 *
 * The round returns bytes. Handing them to a gateway is deliberately the caller's problem, which is
 * what lets the whole loop run against a local `federated serve` (or over `adb`) with no server in the
 * app. This screen stops at "here is the payload and its size" — `payloadBytes` being the #36 DoD
 * measurement.
 */
class FederatedViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(FederatedUiState())
    val ui: StateFlow<FederatedUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    fun onGatewayChanged(value: String) {
        _ui.value = _ui.value.copy(gatewayUrl = value)
    }

    fun onTokenChanged(value: String) {
        _ui.value = _ui.value.copy(token = value)
    }

    fun onConsentChanged(granted: Boolean) {
        _ui.value = _ui.value.copy(consentGranted = granted)
    }

    fun runRound() {
        val loaded = ModelHolder.state.value as? ModelState.Loaded ?: return
        val s = _ui.value
        viewModelScope.launch {
            _ui.value = s.copy(running = true, error = null, result = null)
            try {
                val result = loaded.model.federatedRound(
                    config = FederatedConfig(
                        gatewayUrl = s.gatewayUrl,
                        clientAuthToken = s.token,
                        // Consent carries the policy version it was granted against, so a policy
                        // change invalidates it rather than riding on the old agreement.
                        consent = if (s.consentGranted) {
                            FederatedConsent(
                                granted = true,
                                policyVersion = "1.0",
                                grantedAtEpochMs = System.currentTimeMillis(),
                            )
                        } else {
                            FederatedConsent.NONE
                        },
                    ),
                    // Round 0 has nothing to import: a device must be able to join a cohort that has
                    // not published an aggregate yet.
                    globalRecord = null,
                    roundNumber = s.round,
                    localTraining = { _ ->
                        loaded.model.train(
                            dataset = AppConfig.dataset.value,
                            config = AppConfig.train.value.copy(maxSteps = 5, mergeAtEnd = false),
                        )
                        Unit
                    },
                )
                _ui.value = _ui.value.copy(
                    result = RoundSummary(
                        round = result.round,
                        importedTensors = result.importedTensors,
                        payloadBytes = result.payloadBytes,
                        trainedLocally = result.trainedLocally,
                    ),
                    round = s.round + 1,
                )
            } catch (e: Throwable) {
                // Includes the fail-closed consent/TLS/auth refusals, which name the missing protection.
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(running = false)
            }
        }
    }
}

data class FederatedUiState(
    val gatewayUrl: String = "https://localhost:8443",
    val token: String = "",
    val consentGranted: Boolean = false,
    val round: Int = 0,
    val running: Boolean = false,
    val result: RoundSummary? = null,
    val error: String? = null,
)

data class RoundSummary(
    val round: Int,
    val importedTensors: Int,
    val payloadBytes: Int,
    val trainedLocally: Boolean,
)

/** Kept beside the state so the screen's copy and the SDK's default cannot drift apart. */
val federationDefaultTrainConfig: TrainConfig = TrainConfig(maxSteps = 5, mergeAtEnd = false)
