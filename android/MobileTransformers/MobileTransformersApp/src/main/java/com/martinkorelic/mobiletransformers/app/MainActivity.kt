package com.martinkorelic.mobiletransformers.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.ui.theme.AppTheme
import com.martinkorelic.mobiletransformers.app.ui.theme.AppThemedContent
import com.martinkorelic.mobiletransformers.app.views.ChatScreen
import com.martinkorelic.mobiletransformers.app.views.ConfigurationScreen
import com.martinkorelic.mobiletransformers.app.views.FederatedScreen
import com.martinkorelic.mobiletransformers.app.views.ModelsScreen
import com.martinkorelic.mobiletransformers.app.views.ToolCallScreen
import com.martinkorelic.mobiletransformers.app.views.TrainScreen

/**
 * The MobileTransformers showcase app — **the reference example for the public SDK**.
 *
 * Every screen is a self-contained worked example of one capability, and every one of them talks to the
 * library only through the public facade: `MobileTransformers.fromPretrained`,
 * `MobileTransformerModel`, and the `config/` types. No `ORT*`, `*Native` or `*Repository` type appears
 * anywhere in this module, and `tests/unit/test_guards.py::test_the_sample_app_uses_only_the_public_facade`
 * fails the build if one does.
 *
 * That rule is the point of the app's existence. The previous version drove
 * `LLMRepository`/`TrainingRepository`/`RagRepository`/`InferenceRepository` directly, which meant the
 * public API #17/#19 shipped had never been exercised by anything — its ergonomics had never met a real
 * screen, and there was no worked example of the interface every consumer is told to adopt. Building
 * these six screens found six facade gaps, each fixed **in the facade** and recorded against #17/#19.
 *
 * Screen order is dependency order: Models first, because on a real device nothing else is reachable
 * until a package is installed.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppThemedContent(theme = AppTheme.BETTER) {
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ShowcaseApp()
                }
            }
        }
    }
}

private enum class Destination(val label: String) {
    Models("Models"),
    Chat("Chat"),
    Train("Train"),
    ToolCalls("Tool calls"),
    Federated("Federated"),
    Configuration("Config"),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ShowcaseApp() {
    var destination by remember { mutableStateOf(Destination.Models) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("MobileTransformers") }) },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            // Scrollable rather than a bottom bar: six destinations do not fit a NavigationBar, and
            // dropping one would mean a shipped capability with no worked example.
            ScrollableTabRow(selectedTabIndex = destination.ordinal, edgePadding = 8.dp) {
                Destination.entries.forEach { d ->
                    Tab(
                        selected = d == destination,
                        onClick = { destination = d },
                        text = { Text(d.label) },
                    )
                }
            }

            when (destination) {
                Destination.Models -> ModelsScreen(viewModel())
                Destination.Chat -> ChatScreen(viewModel())
                Destination.Train -> TrainScreen(viewModel())
                Destination.ToolCalls -> ToolCallScreen(viewModel())
                Destination.Federated -> FederatedScreen(viewModel())
                Destination.Configuration -> ConfigurationScreen(viewModel())
            }
        }
    }
}
