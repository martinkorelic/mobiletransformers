package com.martinkorelic.mobiletransformers.app.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat

enum class AppTheme {
    FRI,
    BETTER
}

/**
 * The FRI palette, in one family.
 *
 * It used to be three. `primary` was the project red, but `primaryContainer` (`#E8F5F0`),
 * `surfaceVariant` (`#DDE5DA`) and `background` were pale greens left over from an earlier theme, and
 * `tertiary` was a brown from a third. Since `surfaceVariant` is what the persistent model bar paints
 * itself with and `primaryContainer` is what filled chips use, the two surfaces the user sees on every
 * screen were the ones in the wrong family — a red app with a green header.
 *
 * Everything below is derived from the red: containers are tinted toward it, the neutrals are warm
 * greys rather than green-greys, and `tertiary` is a slate that reads as deliberate contrast instead
 * of as a leftover.
 */
private val FriLightColors = lightColorScheme(
    primary = Color(0xFFE03229),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFFFDAD5),
    onPrimaryContainer = Color(0xFF410100),

    secondary = Color(0xFF58595B),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE6E1E0),
    onSecondaryContainer = Color(0xFF1B1B1D),

    tertiary = Color(0xFF4A5C74),          // Slate — deliberate contrast, not a leftover brown
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFD6E2F3),
    onTertiaryContainer = Color(0xFF0C1B2A),

    error = Color(0xFFB3261E),
    onError = Color.White,
    errorContainer = Color(0xFFF9DEDC),
    onErrorContainer = Color(0xFF410E0B),

    background = Color(0xFFFDFBFB),
    onBackground = Color(0xFF1C1B1B),
    surface = Color(0xFFFDFBFB),
    onSurface = Color(0xFF1C1B1B),
    surfaceVariant = Color(0xFFF0E9E8),    // Warm neutral: the model bar's background
    onSurfaceVariant = Color(0xFF4A4644),
    outline = Color(0xFF857C7A),
    outlineVariant = Color(0xFFD8D0CE),

    // The surface-container family. Unset, these do NOT fall back to `surface` — `lightColorScheme()`
    // fills every omitted role from Material 3's **baseline palette, which is purple**. `TopAppBar`
    // and `ModalDrawerSheet` paint themselves from `surfaceContainer`/`surfaceContainerLow`, so
    // leaving them out put a lilac bar across the top of a red app. Warm greys keyed to `surface`.
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFFAF6F5),
    surfaceContainer = Color(0xFFF5F0EF),
    surfaceContainerHigh = Color(0xFFEFEAE9),
    surfaceContainerHighest = Color(0xFFE9E4E3),
    surfaceBright = Color(0xFFFDFBFB),
    surfaceDim = Color(0xFFDED9D8),
    // Tonal elevation tints surfaces with this; the default is `primary`, which would push elevated
    // surfaces pink. Neutral keeps an elevated card the same family as a flat one.
    surfaceTint = Color(0xFF857C7A),
    inverseSurface = Color(0xFF322F2E),
    inverseOnSurface = Color(0xFFF5F0EF),
    inversePrimary = Color(0xFFFFB4AA),
    scrim = Color(0xFF000000),
)

/**
 * The dark counterpart.
 *
 * `AppThemedContent(isDarkMode = …)` has always taken this parameter and never used it, so the app
 * rendered a light surface under a dark system bar. The default is still `false` — turning it on for
 * everyone would change the app's appearance for a reason nobody asked for — but the parameter now
 * means something when a caller passes it.
 */
private val FriDarkColors = darkColorScheme(
    primary = Color(0xFFFFB4AA),
    onPrimary = Color(0xFF690003),
    primaryContainer = Color(0xFF93000C),
    onPrimaryContainer = Color(0xFFFFDAD5),

    secondary = Color(0xFFC7C6C8),
    onSecondary = Color(0xFF303032),
    secondaryContainer = Color(0xFF464648),
    onSecondaryContainer = Color(0xFFE6E1E0),

    tertiary = Color(0xFFAEC7E4),
    onTertiary = Color(0xFF1A2F45),
    tertiaryContainer = Color(0xFF32455C),
    onTertiaryContainer = Color(0xFFD6E2F3),

    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color(0xFFFFDAD6),

    background = Color(0xFF141313),
    onBackground = Color(0xFFE6E1E0),
    surface = Color(0xFF141313),
    onSurface = Color(0xFFE6E1E0),
    surfaceVariant = Color(0xFF302B2A),
    onSurfaceVariant = Color(0xFFD0C7C5),
    outline = Color(0xFF9A918F),
    outlineVariant = Color(0xFF4E4746),

    // Same reason as the light scheme: omitted roles come from the baseline purple, not from surface.
    surfaceContainerLowest = Color(0xFF0E0E0E),
    surfaceContainerLow = Color(0xFF1C1B1B),
    surfaceContainer = Color(0xFF201F1F),
    surfaceContainerHigh = Color(0xFF2B2A29),
    surfaceContainerHighest = Color(0xFF363433),
    surfaceBright = Color(0xFF3A3838),
    surfaceDim = Color(0xFF141313),
    surfaceTint = Color(0xFF9A918F),
    inverseSurface = Color(0xFFE6E1E0),
    inverseOnSurface = Color(0xFF322F2E),
    inversePrimary = Color(0xFFB3261E),
    scrim = Color(0xFF000000),
)

private val BetterLightColors = lightColorScheme(
    primary = Color(0xFF026fd0),           // Professional blue
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD1E4FF),
    onPrimaryContainer = Color(0xFF001D36),

    secondary = Color(0xFF4A5C74),         // Blue grey — was #F9F9F9, invisible against onSecondary
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
    outlineVariant = Color(0xFFC6C6D0),

    // Same omission as FRI had; this theme is blue, so the baseline purple showed here too.
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFF7F8FC),
    surfaceContainer = Color(0xFFF1F3F9),
    surfaceContainerHigh = Color(0xFFEBEEF5),
    surfaceContainerHighest = Color(0xFFE5E8F0),
    surfaceTint = Color(0xFF767680),
)

private val BetterDarkColors = darkColorScheme(
    primary = Color(0xFF9FCAFF),
    onPrimary = Color(0xFF003259),
    primaryContainer = Color(0xFF00497E),
    onPrimaryContainer = Color(0xFFD1E4FF),
    secondary = Color(0xFFBBC7DB),
    onSecondary = Color(0xFF253141),
    tertiary = Color(0xFFD3BCFA),
    onTertiary = Color(0xFF3A2260),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    background = Color(0xFF131317),
    onBackground = Color(0xFFE4E2E6),
    surface = Color(0xFF131317),
    onSurface = Color(0xFFE4E2E6),
    surfaceVariant = Color(0xFF44474F),
    onSurfaceVariant = Color(0xFFC4C6D0),
    outline = Color(0xFF8E9099),
)

/**
 * Status colours, which Material 3 has no roles for.
 *
 * "The model is ready" and "the model is busy" are the two states the model bar exists to
 * distinguish, and neither is `primary` or `error`. Painting the ready dot with `primary` — which is
 * what it did — made a healthy loaded model show a red dot in a red-primary theme, i.e. the exact
 * opposite of what a status light is for.
 *
 * [busy] is deliberately the brand red rather than the error red: busy is not a fault, and the two
 * are told apart by the row beneath the dot (a failure prints its reason there, a busy model prints
 * what it is doing).
 */
@Immutable
data class StatusColors(
    /** Loaded and free — the model can take work right now. */
    val ready: Color,
    /** Loading, generating, training or merging. */
    val busy: Color,
    /** Nothing loaded. */
    val idle: Color,
    /** The last load or run failed. */
    val failed: Color,
)

private val LightStatusColors = StatusColors(
    ready = Color(0xFF2E7D32),
    busy = Color(0xFFE03229),
    idle = Color(0xFF9A918F),
    failed = Color(0xFFB3261E),
)

private val DarkStatusColors = StatusColors(
    ready = Color(0xFF7BC47F),
    busy = Color(0xFFFF8A80),
    idle = Color(0xFF9A918F),
    failed = Color(0xFFFFB4AB),
)

/** Reachable as `MaterialTheme.statusColors` from any composable inside [AppThemedContent]. */
val LocalStatusColors = staticCompositionLocalOf { LightStatusColors }

val MaterialTheme.statusColors: StatusColors
    @Composable
    get() = LocalStatusColors.current

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
    val colorScheme = when {
        theme == AppTheme.FRI && isDarkMode -> FriDarkColors
        theme == AppTheme.FRI -> FriLightColors
        isDarkMode -> BetterDarkColors
        else -> BetterLightColors
    }

    // The status bar is painted by the WINDOW, not by Compose, so a fully-themed app still sat under
    // a strip of `Theme.MaterialComponents`' `colorPrimaryVariant` — the untouched Android Studio
    // template purple (#3700B3), in both day and night. It is the same defect the surface-container
    // block above documents, one layer further out: a role nobody set, filled from a baseline palette.
    //
    // Driven from the live `colorScheme` rather than restated in `themes.xml` because there are FOUR
    // schemes here (FRI/Better x light/dark) and a hardcoded XML colour can only be right for one of
    // them. `surfaceContainer` specifically: that is what `TopAppBar` paints itself with, so the
    // status bar and the app bar read as one surface instead of a seam.
    val view = LocalView.current
    if (!view.isInEditMode) {
        val window = (view.context as Activity).window
        SideEffect {
            window.statusBarColor = colorScheme.surfaceContainer.toArgb()
            window.navigationBarColor = colorScheme.surfaceContainer.toArgb()
            // Icon contrast is a separate decision from the fill: a light bar needs dark icons or the
            // clock disappears. Keyed to the scheme, not to the system's dark-mode setting, because
            // `isDarkMode` here is the app's own choice and may disagree with the system's.
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !isDarkMode
                isAppearanceLightNavigationBars = !isDarkMode
            }
        }
    }

    CompositionLocalProvider(
        LocalStatusColors provides if (isDarkMode) DarkStatusColors else LightStatusColors,
    ) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = getTypography(theme),
            shapes = getShapes(theme),
            content = content
        )
    }
}
