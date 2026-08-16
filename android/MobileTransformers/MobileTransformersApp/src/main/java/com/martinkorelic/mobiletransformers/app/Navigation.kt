package com.martinkorelic.mobiletransformers.app

/**
 * The app's destinations, and — the part that matters — **why each one is or is not usable right now**.
 *
 * ### Why a drawer replaced the tab row
 *
 * Six destinations do not fit a `NavigationBar`, so the previous version used a `ScrollableTabRow`:
 * a horizontal strip where the later tabs are off-screen until you scroll, with no grouping and no
 * indication that the order is a dependency order. Someone opening the app saw "Models" and had no
 * way to learn that Tool calls exists, let alone that it needs a trained package first.
 *
 * A drawer shows all of them at once, grouped by what they are for, and — because [availability] is
 * computed from the loaded model — can say *why* a destination will not do anything yet instead of
 * letting the user find out by tapping it and reading an empty state.
 *
 * ### Why this file has no Compose in it
 *
 * [availability] is the app's only real navigation logic, and it is a pure function of [ModelState].
 * Keeping it out of the composable layer is what lets `ShowcaseStateTest` check every case on the JVM
 * — the rule the whole app module follows, and the reason its ViewModels hold the state mapping.
 */
enum class Destination(val label: String, val group: NavGroup) {
    /** First for a reason: on a clean install nothing else can do anything until a package exists. */
    Models("Models", NavGroup.Run),
    Chat("Chat", NavGroup.Run),
    /**
 * Search the ingested documents and show the closest passages, with nothing generated.
 *
 * Its own destination rather than a corner of Chat because it is the only part of the retrieval
 * story an **encoder** package can show at all: an embedding model has no generative head, so
 * Chat is hidden for it and grounding is unreachable. It is also the only place retrieval can be
 * judged on its own — inside a grounded answer, bad retrieval and a model ignoring good retrieval
 * are indistinguishable.
 */
    Retrieval("Retrieval", NavGroup.Run),
    /**
 * Hidden for everything except a classifier that names its labels — the exact inverse of [Chat].
 * A decoder has no classification head, so this is a capability the package genuinely does not
 * have rather than a step the user has not taken yet.
 */
    Classify("Classify", NavGroup.Run),
    Train("Training", NavGroup.Train),
    Federated("Federated", NavGroup.Train),
    Configuration("Configuration", NavGroup.Setup),
    About("About", NavGroup.Setup),
}

enum class NavGroup(val label: String) {
    Run("Run a model"),
    Train("Train on device"),
    Setup("Setup"),
}

/**
 * Whether a destination can be used, and if not, the reason in words a user can act on.
 *
 * [Hidden] is distinct from [Blocked] on purpose. A capability the loaded package genuinely does not
 * have — chat on a classification encoder — is noise in the drawer, not a locked door. A capability
 * that merely needs a step first stays visible, because the reason *is* the instruction.
 */
sealed interface Availability {
    data object Enabled : Availability

    /** Reachable, but it will not work yet. [reason] says what to do about it. */
    data class Blocked(val reason: String) : Availability

    /** Not applicable to the loaded package at all; leave it out of the drawer. */
    data object Hidden : Availability
}

/**
 * What [state] means for this destination.
 *
 * Everything here comes from `RuntimeCapabilities`, which the facade already computes from the
 * artifacts actually installed — so the drawer cannot claim a capability the package does not have,
 * and cannot withhold one it does.
 */
fun Destination.availability(state: ModelState): Availability {
    // Always reachable: it is where a model comes from, and the only useful thing to do with no
    // model loaded is to go and load one.
    if (this == Destination.Models || this == Destination.About) return Availability.Enabled

    val model = (state as? ModelState.Loaded)?.model
        ?: return Availability.Blocked(
            when (state) {
                is ModelState.Loading -> "loading ${state.repoId}…"
                is ModelState.Failed -> "${state.repoId} failed to load — see Models"
                else -> "load a model on the Models screen first"
            },
        )

    return availabilityFor(model.capabilities)
}

/**
 * The half of [availability] that depends only on what the loaded package can do.
 *
 * Split out to be reachable from a JVM test: a `ModelState.Loaded` carries a `MobileTransformerModel`,
 * which owns a native session and cannot be constructed off-device, so as one function every
 * capability branch here was untestable — including the one that decides whether Classify exists.
 */
internal fun Destination.availabilityFor(
    caps: com.martinkorelic.mobiletransformers.runtime.RuntimeCapabilities,
): Availability {
    return when (this) {
        Destination.Chat ->
        // Any encoder — a classifier OR a plain embedding model — has no generative head at all,
        // so offering a chat box for it is a promise the package cannot keep. Testing only
        // `isClassifier` covered the first and missed the second, which is exactly the package
        // the Retrieval screen exists for.
        if (caps.isEncoderOnly) Availability.Hidden else Availability.Enabled

        Destination.Retrieval ->
        // The embedding stage is the whole requirement; whether the package can also generate is
        // irrelevant here, which is what lets a pure encoder use this screen.
        if (caps.supportsRag || caps.supportsEmbedding) {
                Availability.Enabled
            } else {
            Availability.Blocked("this package has no embedding stage — pull one with RAG requested")
            }

        Destination.Classify ->
        // `supportsClassification`, not `isClassifier`: a classification graph whose labels are
        // unknown runs fine and answers `LABEL_3`, which is a number in a costume. The screen
        // would show bars with no meaning, so the honest report is that it is not applicable.
        if (caps.supportsClassification) {
                Availability.Enabled
            } else {
                Availability.Hidden
            }

        Destination.Train, Destination.Federated ->
            if (caps.supportsTraining) {
                Availability.Enabled
            } else {
                Availability.Blocked("this package has no train/ stage — pull one with Training requested")
            }

        Destination.Models, Destination.About, Destination.Configuration -> Availability.Enabled
    }
}

/** The destinations to show, in drawer order, for the current model. */
fun visibleDestinations(state: ModelState): List<Destination> =
    Destination.entries.filter { it.availability(state) !is Availability.Hidden }

/**
 * Where to send the user when the destination they are on stops being applicable.
 *
 * Loading a classifier while sitting on Chat would otherwise leave them looking at a screen that is
 * no longer in the drawer, with no way back except the drawer they cannot see behind it.
 */
fun redirectFor(current: Destination, state: ModelState): Destination =
    if (current.availability(state) is Availability.Hidden) Destination.Models else current
