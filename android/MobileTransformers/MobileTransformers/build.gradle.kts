plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.objectbox)
    `maven-publish`
}

android {
    namespace = "com.martinkorelic.mobiletransformers"
    compileSdk = 34

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
        // #36: federated participation is OFF unless the app shipping this turns it on. Adapter
        // factors are derived from the user's own data, so the default must be "do not send".
        buildConfigField("boolean", "FEDERATION_ENABLED", "false")
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DJSON_BuildTests=OFF"
            }
        }
        ndk {
            // v1 ships arm64-v8a only. x86_64 is NOT buildable here: jniLibs/x86_64 has the GenAI .so
            // but not `libonnxruntime.so`, `libtokenizers_c.a` or `libtokenizers_cpp.a`, which
            // CMakeLists.txt links against — so `libmobiletransformers.so` has never existed for that
            // ABI. Listing it produced an AAR with an x86_64 directory whose consumer would fail at
            // System.loadLibrary (which scripts/android_build_aar.sh already refuses to publish), and
            // broke every unqualified `assembleDebug`. Restoring x86_64 means building ORT-training and
            // tokenizers-cpp for it first; see docs/ARCHITECTURE.md "ABI support".
            abiFilters += listOf("arm64-v8a")
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
    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
    testOptions {
        unitTests {
            // Robolectric needs resources on the unit-test classpath.
            isIncludeAndroidResources = true
            // Anything Robolectric does NOT shadow should FAIL rather than silently return 0/null —
            // a returned default would let a test "pass" against a method that never actually ran.
            isReturnDefaultValues = false
        }
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
    // Robolectric provides Android SDK stubs on the JVM. Without it, every class touching
    // android.util.Log / Context / org.json failed with "Method ... not mocked", which is exactly why
    // LLMRepository, RagRepository, FileUtil and RepositoryBackedModelSession had ZERO JVM coverage —
    // and why several of the defects this pass fixed reached the audit unnoticed.
    testImplementation(libs.robolectric)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.work.testing)
}
// --- Maven publication (#30) ----------------------------------------------------------------------
// Coordinates: com.martinkorelic.mobiletransformers:mobiletransformers-android:<version>
// `group`/`version` come from gradle.properties and are overridable with -Pversion=<x>.
//
// Note `00_code_plans/04` names the group `com.martinkorelic` while `05_code_plans/03` names
// `com.martinkorelic.mobiletransformers`. The latter wins (it is the publication plan's own contract);
// the older doc is corrected rather than followed.
publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = project.group.toString()
            artifactId = "mobiletransformers-android"
            version = project.version.toString()

            afterEvaluate { from(components["release"]) }

            pom {
                name.set("MobileTransformers Android SDK")
                description.set(
                    "On-device LLM parameter-efficient fine-tuning, inference and RAG for Android."
                )
                url.set("https://github.com/martinkorelic/mobiletransformers")
                licenses {
                    license {
                        // Kept in lockstep with LICENSE.md. The Apache-2.0 relicense (#32) needs both
                        // rights holders in CITATION.cff; until it lands this MUST report the real
                        // licence, because a consumer resolving this POM relies on it.
                        name.set("Creative Commons Attribution-NonCommercial 4.0 International")
                        url.set("https://creativecommons.org/licenses/by-nc/4.0/")
                        distribution.set("repo")
                    }
                }
                developers {
                    developer {
                        id.set("martinkorelic")
                        name.set("Martin Korelič")
                    }
                    developer {
                        id.set("vpejovic")
                        name.set("Veljko Pejović")
                    }
                }
                scm {
                    url.set("https://github.com/martinkorelic/mobiletransformers")
                    connection.set("scm:git:https://github.com/martinkorelic/mobiletransformers.git")
                }
            }
        }
    }
}
