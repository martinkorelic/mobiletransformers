package com.martinkorelic.orttransformer

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBars
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.martinkorelic.orttransformer.databinding.ActivityMainBinding
import com.martinkorelic.ortmobile.repository.InferenceRepository
import com.martinkorelic.ortmobile.repository.LLMRepository
import com.martinkorelic.ortmobile.repository.RagRepository
import com.martinkorelic.ortmobile.repository.TrainingRepository
import com.martinkorelic.orttransformer.ui.theme.AppTheme
import com.martinkorelic.orttransformer.ui.theme.AppThemedContent
import com.martinkorelic.orttransformer.viewmodels.ConfigurationViewModel
import com.martinkorelic.orttransformer.viewmodels.InferenceViewModel
import com.martinkorelic.orttransformer.viewmodels.TrainingViewModel
import com.martinkorelic.orttransformer.views.ConfigurationScreen
import com.martinkorelic.orttransformer.views.InferenceScreen
import com.martinkorelic.orttransformer.views.TrainingScreen

class MainActivity : ComponentActivity() {

    private val LOG_TAG = "MainActivity"

    private lateinit var binding: ActivityMainBinding

    // Creating training and inference repository
    // Pick and play
    private lateinit var llmRepository : LLMRepository
    private lateinit var inferenceRepository : InferenceRepository
    private lateinit var trainingRepository : TrainingRepository
    private lateinit var ragRepository: RagRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // To LLMRepository pass the application files directory to access models
        llmRepository = LLMRepository(applicationContext, filesDir.absolutePath)
        inferenceRepository = InferenceRepository(llmRepository)
        trainingRepository = TrainingRepository(llmRepository)
        ragRepository = RagRepository(llmRepository)

        enableEdgeToEdge()

        setContent {
            // Change theme when needed
            val currentTheme by remember { mutableStateOf(AppTheme.FRI) }

            AppThemedContent(theme = currentTheme) {
                Surface (
                    modifier = Modifier
                        .fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainApp()
                }
            }
        }
    }

    companion object {
        // Used to load the 'ortmobile' library on application startup.
        init {
            System.loadLibrary("ortmobile")
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun MainApp() {
        var selectedTab by remember { mutableStateOf(0) }
        val tabs = listOf("Inference", "Training", "Configuration")

        // Create ViewModels

        val inferenceViewModel = remember { InferenceViewModel(llmRepository, inferenceRepository, ragRepository) }
        val trainingViewModel = remember { TrainingViewModel(llmRepository, trainingRepository) }
        val configurationViewModel = remember { ConfigurationViewModel(llmRepository) }

        Column(modifier = Modifier
            .fillMaxSize()) {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.fri_logo),
                            contentDescription = "App logo",
                            modifier = Modifier
                                .size(128.dp),
                                //.size(48.dp)
                        )
                        Text(
                            text = "ORTransformersMobile",
                            //text = "Mobile Health Assistant",
                            modifier = Modifier.fillMaxWidth(),
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.secondary
                        )
                    }

                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
            TabRow(selectedTabIndex = selectedTab, contentColor = Color.White, containerColor = Color.White,
                indicator = { tabPositions ->
                    TabRowDefaults.Indicator(
                        Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = MaterialTheme.colorScheme.primary
                    )
                }) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title, color = MaterialTheme.colorScheme.primary) },
                        selectedContentColor = MaterialTheme.colorScheme.primary,
                        unselectedContentColor = MaterialTheme.colorScheme.primary
                    )
                }
            }

            when (selectedTab) {
                0 -> InferenceScreen(viewModel = inferenceViewModel, configurationViewModel = configurationViewModel)
                1 -> TrainingScreen(viewModel = trainingViewModel)
                2 -> ConfigurationScreen(viewModel = configurationViewModel)
            }
        }
    }

}