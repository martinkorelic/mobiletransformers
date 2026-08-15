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
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Hub token for pulling a PRIVATE or GATED package, taken from the build environment.
        //
        // An Android app cannot read the host's environment at runtime, so the value has to be baked
        // in at build time. Source order: `-PmtHubToken=...` wins, else the `HF_TOKEN` environment
        // variable, else empty. Empty is the normal case and means "anonymous" — public packages pull
        // without any of this.
        //
        //   HF_TOKEN=hf_xxx ./gradlew :MobileTransformersApp:assembleDebug
        //   ./gradlew :MobileTransformersApp:assembleDebug -PmtHubToken=hf_xxx
        //
        // ⚠️ A token compiled into an APK is EXTRACTABLE by anyone holding the APK — `strings` on the
        // dex is enough. This is a development and demo affordance for reaching your own private repo,
        // not a way to ship credentials. Never build a release this way, and never commit a token to
        // `gradle.properties`. A real app should obtain a token at runtime from the user or from an
        // authenticated backend and hand it to `MobileTransformers.fromPretrained(hubConfig = ...)`,
        // which is the same public entry point this uses.
        val hubToken = (project.findProperty("mtHubToken") as String?)
            ?: System.getenv("HF_TOKEN")
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
    val composeBom = platform("androidx.compose:compose-bom:2024.10.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)

    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // Compose
    implementation(libs.androidx.material3)

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