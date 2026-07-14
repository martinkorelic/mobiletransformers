package com.martinkorelic.mobiletransformers.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.material3.*

enum class AppTheme {
    FRI,
    BETTER
}

// Medical Theme Colors
private val FriLightColors = lightColorScheme(
    primary = Color(0xFFe03229),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFE8F5F0),
    onPrimaryContainer = Color(0xFF1B4A36),

    secondary = Color(0xFF58595b),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFD1ECFF),
    onSecondaryContainer = Color(0xFF001D36),

    tertiary = Color(0xFF7B5A3C),          // Warm brown
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFFFDDBE),
    onTertiaryContainer = Color(0xFF2D1600),

    error = Color(0xFFB00020),
    onError = Color.White,
    errorContainer = Color(0xFFFDADAD),
    onErrorContainer = Color(0xFF410E0B),

    background = Color(0xFFFDFCFF),
    onBackground = Color(0xFF1A1C19),
    surface = Color(0xFFFDFCFF),
    onSurface = Color(0xFF1A1C19),
    surfaceVariant = Color(0xFFDDE5DA),
    onSurfaceVariant = Color(0xFF414941),
    outline = Color(0xFF717970),
    outlineVariant = Color(0xFFC1C9BF)
)

private val BetterLightColors = lightColorScheme(
    primary = Color(0xFF026fd0),           // Professional blue
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD1E4FF),
    onPrimaryContainer = Color(0xFF001D36),

    secondary = Color(0xFFF9F9F9),         // Blue grey
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFD7E3F7),
    onSecondaryContainer = Color(0xFF101C2B),

    tertiary = Color(0xFF6A4C93),          // Purple accent
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFEADDFF),
    onTertiaryContainer = Color(0xFF21005D),

    error = Color(0xFFD32F2F),
    onError = Color.White,
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),

    background = Color(0xFFFEFBFF),
    onBackground = Color(0xFF1B1B1F),
    surface = Color(0xFFFEFBFF),
    onSurface = Color(0xFF1B1B1F),
    surfaceVariant = Color(0xFFE2E2EC),
    onSurfaceVariant = Color(0xFF45464F),
    outline = Color(0xFF767680),
    outlineVariant = Color(0xFFC6C6D0)
)

// 5. Custom Typography per Theme
@Composable
private fun getTypography(theme: AppTheme): Typography {
    return when (theme) {
        AppTheme.FRI -> Typography(
            headlineLarge = MaterialTheme.typography.headlineLarge.copy(
                fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold
            ),
            titleMedium = MaterialTheme.typography.titleMedium.copy(
                fontWeight = androidx.compose.ui.text.font.FontWeight.Medium
            )
        )
        AppTheme.BETTER -> Typography(
            headlineLarge = MaterialTheme.typography.headlineLarge.copy(
                fontWeight = androidx.compose.ui.text.font.FontWeight.Bold
            ),
            titleMedium = MaterialTheme.typography.titleMedium.copy(
                fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold
            )
        )
    }
}

// 6. Custom Shapes per Theme
@Composable
private fun getShapes(theme: AppTheme): Shapes {
    return when (theme) {
        AppTheme.FRI -> Shapes(
            small = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
            medium = androidx.compose.foundation.shape.RoundedCornerShape(12.dp),
            large = androidx.compose.foundation.shape.RoundedCornerShape(16.dp)
        )
        AppTheme.BETTER -> Shapes(
            small = androidx.compose.foundation.shape.RoundedCornerShape(4.dp),
            medium = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
            large = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
        )
    }
}

// 4. Main Theme Composable
@Composable
fun AppThemedContent(
    theme: AppTheme,
    isDarkMode: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when (theme) {
        AppTheme.FRI -> FriLightColors
        AppTheme.BETTER -> BetterLightColors
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = getTypography(theme),
        shapes = getShapes(theme),
        content = content
    )
}