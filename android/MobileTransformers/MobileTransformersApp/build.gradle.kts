import org.jetbrains.kotlin.cli.jvm.main

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

android {
    namespace = "com.martinkorelic.mobiletransformers.app"
    compileSdk = 34

    defaultConfig {

        applicationId = "com.martinkorelic.mobiletransformers.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 2
        // Derived from the root `version` property (gradle.properties), which `test_version_sites.py`
        // pins to pyproject.toml. It was the literal "1.0", which matched no other site in the repo:
        // the sample app is the one artifact a user sees a version number on, and it advertised a
        // release that does not exist.
        versionName = rootProject.findProperty("version")?.toString() ?: "0.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Hub token for pulling a PRIVATE or GATED package, taken from the build environment.
        //
        // An Android app cannot read the host's environment at runtime, so the value has to be baked
        // in at build time. Source order: `-PmtHubToken=...` wins, else `HF_TOKEN_ORG`, else
        // `HF_TOKEN`, else empty. Empty is the normal case and means "anonymous" — public packages
        // pull without any of this.
        //
        //   HF_TOKEN_ORG=hf_xxx ./gradlew :MobileTransformersApp:assembleDebug
        //   ./gradlew :MobileTransformersApp:assembleDebug -PmtHubToken=hf_xxx
        //
        // `HF_TOKEN_ORG` is tried FIRST, and the ordering is load-bearing rather than arbitrary. Four
        // of the five catalog entries are private repos under the `mobiletransformers` org, and a
        // fine-grained personal `HF_TOKEN` scoped to one repo cannot see them — so an APK built with
        // the personal token shows a full catalog whose Install button 401s on almost every row. That
        // is exactly the shape of failure this project keeps re-learning: the build succeeds, the app
        // looks right, and the capability is silently absent. Same precedence as
        // `scripts/publish_catalog.sh`, which publishes those repos.
        //
        // ⚠️ A token compiled into an APK is EXTRACTABLE by anyone holding the APK — `strings` on the
        // dex is enough. This is a development and demo affordance for reaching your own private repo,
        // not a way to ship credentials. Never build a release this way, and never commit a token to
        // `gradle.properties`. A real app should obtain a token at runtime from the user or from an
        // authenticated backend and hand it to `MobileTransformers.fromPretrained(hubConfig = ...)`,
        // which is the same public entry point this uses.
        val hubToken = (project.findProperty("mtHubToken") as String?)
            ?.takeIf { it.isNotBlank() }
            ?: System.getenv("HF_TOKEN_ORG")?.takeIf { it.isNotBlank() }
            ?: System.getenv("HF_TOKEN")?.takeIf { it.isNotBlank() }
            ?: ""
        buildConfigField("String", "HF_TOKEN", "\"$hubToken\"")

        vectorDrawables {
            useSupportLibrary = true
        }

    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    buildFeatures {
        viewBinding = true
        compose = true
        // Carries HF_TOKEN (above), so the app can reach a private package without a UI keyboard.
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    testOptions {
        unitTests {
            // The showcase app's ViewModels hold pure state-mapping logic (empty states, disabled
            // features, engine pickers) that must be provable without a device — the app module had
            // no test source set at all before the facade rewrite.
            isIncludeAndroidResources = true
            // Matches the library module: stubbed android.* methods THROW rather than returning null,
            // so a test that accidentally reaches Android fails loudly instead of asserting on a
            // silent default. Anything genuinely needing org.json/Intent uses Robolectric.
            isReturnDefaultValues = false
        }
    }
}

dependencies {


    implementation(project(":MobileTransformers"))
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)

    androidTestImplementation(libs.androidx.ui.test.junit4)
    // The BOM comes from the version catalog, like every other dependency. It used to be declared
    // here as a hardcoded 2024.10.00 while the catalog pinned 2024.04.01 for the library module, so
    // the two modules resolved different Compose versions from the same build.
    val composeBom = platform(libs.androidx.compose.bom)
    implementation(composeBom)
    androidTestImplementation(composeBom)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)

    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // Compose
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    // The model catalog is a bundled JSON asset so adding a model is editing one file; gson is
    // already the module's JSON library on the SDK side.
    implementation(libs.gson)

    implementation(libs.androidx.ui.tooling.preview)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    debugImplementation(libs.androidx.ui.test.manifest)
}