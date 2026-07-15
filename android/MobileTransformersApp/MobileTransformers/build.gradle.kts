plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.objectbox)
}

android {
    namespace = "com.martinkorelic.mobiletransformers"
    compileSdk = 34

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("libs")
        }
    }

    buildFeatures {
        // #22: BuildConfig carries the default-off on-device adapter-upload security flag.
        buildConfig = true
    }

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        // #22: on-device Hub adapter upload is disabled by default (privacy-gated); flip only behind a
        // security review. Product path is device -> desktop sync -> Python `push-adapter`.
        buildConfigField("boolean", "ADAPTER_UPLOAD_ENABLED", "false")
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DJSON_BuildTests=OFF"
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
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
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    // The onnxruntime-genai .so is provided BOTH by the AAR (runtime) and jniLibs (CMake link input, #10);
    // dedupe the identical native libs at packaging. Content is the same real 0.14 binary either way.
    packaging {
        jniLibs {
            pickFirsts += listOf(
                "**/libonnxruntime-genai.so",
                "**/libonnxruntime-genai-jni.so",
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
}

dependencies {

    // ONNX Runtime GenAI (#10/#11): the genai .so ships from jniLibs (patched to dlopen the genai-paired
    // stock ORT as `libort_gen.so`, keeping the training ORT as `libonnxruntime.so`). Its Java classes are
    // unused (C-API/JNI path), so the AAR is NOT a dependency — this avoids the AAR-vs-jniLibs .so conflict
    // and lets us ship the patched binary. See spikes/genai_external_swap/README.md (ORT separation).

    implementation(libs.pebble)
    implementation(libs.gson)

    // #21/#22: Hub network half — OkHttp (streaming GET/Range), WorkManager (background download), and an
    // EXPLICIT coroutines dependency (was previously only transitive).
    implementation(libs.okhttp)
    implementation(libs.androidx.work.runtime.ktx)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.okhttp.mockwebserver)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.kotlinx.coroutines.test)
}