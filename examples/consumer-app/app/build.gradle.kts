plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.consumer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.consumer"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            // Only ABIs the published AAR actually carries libmobiletransformers.so for.
            abiFilters += listOf("arm64-v8a")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions { jvmTarget = "1.8" }
}

dependencies {
    // The whole point of this example: a plain Maven coordinate, no project(":...") dependency.
    implementation(
        "com.martinkorelic.mobiletransformers:mobiletransformers-android:" +
            "${project.findProperty("mobiletransformersVersion")}"
    )
}
