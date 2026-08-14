package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.BuildConfig
import com.martinkorelic.mobiletransformers.app.viewmodels.FederatedViewModel

/**
 * #35/#36 — one federated round, with the consent gate on screen rather than implied.
 *
 * The disabled state is shown honestly. `FEDERATION_ENABLED` is false by default, and rather than
 * hiding the feature the screen says so and still lets the button be pressed — the resulting
 * `FederatedConsentException` names the missing protection, which is more useful to an integrator than
 * a greyed-out control with no explanation.
 */
@Composable
fun FederatedScreen(vm: FederatedViewModel) {
    val ui by vm.ui.collectAsState()
    val state by vm.modelState.collectAsState()

    ModelGate(state, needs = "A federated round trains locally, so it needs a train-capable package.") {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            if (!BuildConfig.FEDERATION_ENABLED) {
                EmptyState(
                    title = "Federation is disabled in this build",
                    detail = "BuildConfig.FEDERATION_ENABLED = false. It is off by default and must be " +
                        "enabled deliberately by the app that ships it. Running a round below will " +
                        "fail closed and name this as the reason.",
                )
            }

            Section("Gateway") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextField("Gateway URL", ui.gatewayUrl, vm::onGatewayChanged)
                    TextField("Client auth token", ui.token, vm::onTokenChanged)
                    LabeledSwitch("Consent granted", ui.consentGranted, vm::onConsentChanged)
                    Text(
                        "Consent, TLS and auth are checked before any tensor is read. Only adapter " +
                            "factors and aggregate metrics ever leave the device — never examples.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Section("Round ${ui.round}") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "import global adapter → train locally → export this device's update. " +
                            "Round 0 imports nothing: a device must be able to join a cohort that has " +
                            "not published an aggregate yet.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Button(onClick = vm::runRound, enabled = !ui.running) {
                        Text(if (ui.running) "Running…" else "Run one round")
                    }
                }
            }

            ui.result?.let { r ->
                Section("Result") {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("round ${r.round}")
                        Text("imported tensors: ${r.importedTensors}")
                        Text("trained locally: ${r.trainedLocally}")
                        Text("upload payload: ${r.payloadBytes} B", style = MaterialTheme.typography.bodyLarge)
                        Text(
                            "Nothing was uploaded. The round returns bytes; handing them to a gateway " +
                                "is the caller's choice, which is what lets the whole loop run against " +
                                "a local `federated serve`.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            ui.error?.let { Section("Refused / error") { Text(it) } }
        }
    }
}
