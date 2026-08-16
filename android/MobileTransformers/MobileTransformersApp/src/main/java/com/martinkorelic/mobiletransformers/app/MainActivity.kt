package com.martinkorelic.mobiletransformers.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Hub
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Label
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.School
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.martinkorelic.mobiletransformers.app.ui.theme.AppTheme
import com.martinkorelic.mobiletransformers.app.ui.theme.AppThemedContent
import com.martinkorelic.mobiletransformers.app.views.AboutScreen
import com.martinkorelic.mobiletransformers.app.views.ChatScreen
import com.martinkorelic.mobiletransformers.app.views.ClassifyScreen
import com.martinkorelic.mobiletransformers.app.views.ConfigurationScreen
import com.martinkorelic.mobiletransformers.app.views.ConfigurationTab
import com.martinkorelic.mobiletransformers.app.views.FederatedScreen
import com.martinkorelic.mobiletransformers.app.views.ModelBar
import com.martinkorelic.mobiletransformers.app.views.ModelsScreen
import com.martinkorelic.mobiletransformers.app.views.RetrievalScreen
import com.martinkorelic.mobiletransformers.app.views.TrainScreen
import kotlinx.coroutines.launch

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
 * screen, and there was no worked example of the interface every consumer is told to adopt.
 *
 * ### The shell
 *
 * Three parts, all of them global, because all three answer questions that arise on whichever screen
 * the user happens to be on:
 *
 * - a **drawer** grouping the destinations by purpose, with each one saying why it is unusable when it
 *   is (see [Destination.availability]) — replacing a `ScrollableTabRow` whose later tabs were
 *   off-screen and whose dependency order was invisible;
 * - a **[ModelBar]** pinned under the app bar, so "what is loaded, what can it do, is a pull running"
 *   never requires navigating away from the question;
 * - a **snackbar** fed by [AppSnackbar], so outcomes that complete while the user is elsewhere — a
 *   download finishing, training failing — are not delivered to a screen nobody is looking at.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppThemedContent(theme = AppTheme.FRI) {
                // The SDK declares POST_NOTIFICATIONS but nothing requested it, so on API 33+ every
                // foreground training notification was dropped by the system.
                RequestNotificationPermissionOnce()
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ShowcaseApp()
                }
            }
        }
    }
}

private val Destination.icon: ImageVector
    get() = when (this) {
        Destination.Models -> Icons.Outlined.Download
        Destination.Chat -> Icons.AutoMirrored.Outlined.Chat
    Destination.Retrieval -> Icons.Outlined.Search
    Destination.Classify -> Icons.Outlined.Label
        Destination.Train -> Icons.Outlined.School
        Destination.Federated -> Icons.Outlined.Hub
        Destination.Configuration -> Icons.Outlined.Tune
        Destination.About -> Icons.Outlined.Info
    }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ShowcaseApp() {
    var destination by remember { mutableStateOf(Destination.Models) }
    // Which Configuration tab to open when something links into it (Chat's "Settings").
    var configurationTab by remember { mutableStateOf(ConfigurationTab.Generation) }
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    val modelState by ModelHolder.state.collectAsState()
    val download by ModelHolder.download.collectAsState()
    val activity by ModelHolder.activity.collectAsState()

    // Loading a classifier while sitting on Chat would leave the user on a screen that has just left
    // the drawer, with no visible way back.
    LaunchedEffect(modelState) { destination = redirectFor(destination, modelState) }

    LaunchedEffect(Unit) {
        AppSnackbar.events.collect { event ->
            val result = snackbarHostState.showSnackbar(
                message = event.message,
                actionLabel = event.actionLabel,
                // An error is the one kind worth making the user dismiss: it is the only one whose
                // text they may need to read twice.
                duration = if (event.severity == AppSnackbar.Severity.Error) {
                    SnackbarDuration.Long
                } else {
                    SnackbarDuration.Short
                },
            )
            if (result == SnackbarResult.ActionPerformed) event.onAction?.invoke()
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                DrawerContent(
                    current = destination,
                    modelState = modelState,
                    onSelect = {
                        destination = it
                        scope.launch { drawerState.close() }
                    },
                )
            }
        },
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(destination.label) },
                    navigationIcon = {
                        androidx.compose.material3.IconButton(
                            onClick = { scope.launch { drawerState.open() } },
                        ) { Icon(Icons.Outlined.Menu, contentDescription = "Open navigation drawer") }
                    },
                )
            },
            snackbarHost = {
                // Top-anchored rather than Material's default bottom: the bottom of every screen here
                // is where the primary controls live (Send, Start, the prompt field), and a banner
                // that covers the button the user just pressed is worse than no banner.
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                    SnackbarHost(snackbarHostState) { data ->
                        Snackbar(snackbarData = data, modifier = Modifier.padding(8.dp))
                    }
                }
            },
        ) { padding ->
            Column(Modifier.fillMaxSize().padding(padding)) {
                ModelBar(
                    state = modelState,
                    activity = activity,
                    download = download,
                    onUnload = { scope.launch { ModelHolder.close() } },
                    onGoToModels = { destination = Destination.Models },
                )

                when (destination) {
                    Destination.Models -> ModelsScreen(viewModel())
                    Destination.Chat -> ChatScreen(
                        viewModel(),
                        // "Settings" in Chat means the generation knobs, which live here.
                        onOpenSettings = {
                            configurationTab = ConfigurationTab.Generation
                            destination = Destination.Configuration
                },
            )
                    Destination.Retrieval -> RetrievalScreen(viewModel())
                    Destination.Classify -> ClassifyScreen(viewModel())
                    Destination.Train -> TrainScreen(viewModel())
                    Destination.Federated -> FederatedScreen(viewModel())
                    Destination.Configuration -> ConfigurationScreen(viewModel(), configurationTab)
                    Destination.About -> AboutScreen(onGoToModels = { destination = Destination.Models })
                }
            }
        }
    }
}

@Composable
private fun DrawerContent(
    current: Destination,
    modelState: ModelState,
    onSelect: (Destination) -> Unit,
) {
    Column(
        Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(
            "MobileTransformers",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 12.dp),
        )

        val visible = visibleDestinations(modelState)
        NavGroup.entries.forEach { group ->
            val items = visible.filter { it.group == group }
            if (items.isEmpty()) return@forEach

            Text(
                group.label,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 28.dp, top = 16.dp, bottom = 4.dp),
            )
            items.forEach { d ->
                val availability = d.availability(modelState)
                val blocked = availability as? Availability.Blocked
                NavigationDrawerItem(
                    icon = { Icon(d.icon, contentDescription = null) },
                    label = {
                        Column {
                            Text(d.label)
                            // The reason IS the instruction — "load a model first", "pull one with
                            // Training requested". Hiding it behind a tap means the user learns it
                            // from an empty state after choosing wrongly.
                            blocked?.let {
                                Text(
                                    it.reason,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    },
                    selected = d == current,
                    // Still selectable when blocked: the destination explains itself far better than
                    // a greyed-out row, and every screen already renders an honest empty state.
                    onClick = { onSelect(d) },
                    modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding),
                )
            }
        }
    }
}
