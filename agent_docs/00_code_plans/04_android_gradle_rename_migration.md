# Android Gradle / Module / App Rename Migration
**Priority (global #):** 15  |  **Prerequisites:** none  |  **Blocks:** #16 (`05_android_facade_foundation.md`)

## Purpose

Perform the `ORTTransformer` → `MobileTransformers` Gradle workspace rename as **one isolated, build-verified change** before any facade work begins. This plan renames the Gradle root, both modules, their directories, the Kotlin package namespaces, the app `applicationId`, and the inter-module project dependency — and nothing else. No new public API, no engine changes, no JNI symbol renames.

Hard rule from `00_repository_restructure_plan.md` ("What To Keep Internal For Now" and "Migration Notes" line 559): *do not rename Android internals opportunistically.* The native shared-library name and JNI C++ symbols stay on the old `ortmobile` spelling in this pass. The facade and any deeper package moves come in #16.

Success criterion: `./gradlew :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug` builds green, and the sample app launches, on a clean checkout — with zero behavioral change other than the (accepted) sample-app reinstall caused by the new `applicationId`.

## Touched / new files

Current root: `android/ORTransformer/`. Target root: `android/MobileTransformers/`.

| Current | Target | Action |
| --- | --- | --- |
| `android/ORTransformer/` | `android/MobileTransformers/` | `git mv` directory |
| `android/ORTransformer/settings.gradle.kts` | `…/MobileTransformers/settings.gradle.kts` | edit `rootProject.name`, `include(...)` |
| `android/ORTransformer/ORTransformersMobile/` | `android/MobileTransformers/MobileTransformers/` | `git mv` (SDK/AAR module) |
| `android/ORTransformer/app/` | `android/MobileTransformers/MobileTransformersApp/` | `git mv` (sample app) |
| `…/ORTransformersMobile/build.gradle.kts` | `…/MobileTransformers/build.gradle.kts` | edit `namespace` |
| `…/app/build.gradle.kts` | `…/MobileTransformersApp/build.gradle.kts` | edit `namespace`, `applicationId`, project dep |
| `…/ORTransformersMobile/src/main/java/com/martinkorelic/ortmobile/**` | `…/com/martinkorelic/mobiletransformers/**` | move package dir + edit `package`/`import` lines |
| `…/app/src/main/java/com/martinkorelic/orttransformer/**` | `…/com/martinkorelic/mobiletransformers/app/**` | move package dir + edit `package`/`import` lines |
| `…/app/src/main/res/values/themes.xml` / `styles.xml` | same path under new module | rename `Theme.ORTTransformer` style refs (used in `AndroidManifest.xml`) |
| `gradle/wrapper/`, `gradlew`, `gradlew.bat`, root `build.gradle.kts`, `gradle/libs.versions.toml` | move under new root | move as-is; no content change needed |

Existing Kotlin source inventory under `ORTransformersMobile/src/main/java/com/martinkorelic/ortmobile/` (all move to `…/mobiletransformers/`): `DataUtil.kt`, `FileUtil.kt`, `ORTChatTemplateHandler.kt`, `ORTConversationState.kt`, `ORTDataCurator.kt`, `ORTGenAINative.kt`, `ORTGenAITokenizer.kt`, `ORTGenerationConfig.kt`, `ORTGeneratorNative.kt`, `ORTProgress.kt`, `ORTRagConfig.kt`, `ORTRetriever.kt`, `ORTScheduler.kt`, `ORTTokenizerNative.kt`, `ORTTrainerNative.kt`, `ORTTrainingConfig.kt`, `ORTVectorDatabase.kt`, `PebbleUtil.kt`, `SpecialTokenModel.kt`, `entity/VectorEntity.kt`, `repository/{InferenceRepository,LLMRepository,RagRepository,TrainingRepository}.kt`, plus `src/test/.../ExampleUnitTest.kt` and `src/androidTest/.../ExampleInstrumentedTest.kt`.

App sources under `app/src/main/java/com/martinkorelic/orttransformer/`: `MainActivity.kt`, `ui/theme/{Theme,Type}.kt`, `viewmodels/{Configuration,Inference,Training}ViewModel.kt`, `views/{Configuration,Inference,Training}Screen.kt`, plus test/androidTest stubs.

New (compat-only) files added in this pass:
- `…/MobileTransformers/src/main/java/com/martinkorelic/ortmobile/Aliases.kt` — deprecated `typealias` shims under the **old** package (kept for one release).

## Data contracts / interfaces

This pass changes only identifiers, not data shapes. The contracts that must remain byte-identical:

- **`settings.gradle.kts`** end state:
  ```kotlin
  rootProject.name = "MobileTransformers"
  include(":MobileTransformersApp")
  include(":MobileTransformers")
  ```
  (The `pluginManagement` ObjectBox `resolutionStrategy` block stays unchanged.)
- **SDK module `namespace`**: `com.martinkorelic.ortmobile` → `com.martinkorelic.mobiletransformers`.
- **App module `namespace`**: `com.martinkorelic.orttransformer` → `com.martinkorelic.mobiletransformers.app`.
- **App `applicationId`**: currently `com.martinkorelic.ortmobile` → `com.martinkorelic.mobiletransformers.app`. (Note: the current app `namespace` and `applicationId` already differ; both converge to the new `.app` value.)
- **Inter-module dependency** in app `build.gradle.kts`: `implementation(project(":ORTransformersMobile"))` → `implementation(project(":MobileTransformers"))`.
- **AAR Maven coordinate** (publishing metadata, added but not necessarily published this pass): artifactId `mobiletransformers-android`, group `com.martinkorelic` (e.g. `com.martinkorelic:mobiletransformers-android`).

### Compatibility aliases (`Aliases.kt`)
Under `package com.martinkorelic.ortmobile`, provide deprecated typealiases so external callers importing the old package still compile for one release:
```kotlin
@Deprecated("Use com.martinkorelic.mobiletransformers.repository.LLMRepository",
    ReplaceWith("com.martinkorelic.mobiletransformers.repository.LLMRepository"))
typealias LLMRepository = com.martinkorelic.mobiletransformers.repository.LLMRepository
// repeat for the public-surfaced ORT* config classes and repositories that demos import
```
Scope: only types that were realistically referenced from outside the module (the four repositories and `ORT*Config`). Do **not** alias `*Native` JNI classes — see risks.

## Implementation steps

1. **Branch + baseline.** On a fresh branch, build the current tree first: `./gradlew :ORTransformersMobile:assembleDebug :app:assembleDebug`. Record success so the diff is attributable.
2. **Move the workspace root.** `git mv android/ORTransformer android/MobileTransformers`. Move module dirs: `git mv android/MobileTransformers/ORTransformersMobile android/MobileTransformers/MobileTransformers` and `git mv android/MobileTransformers/app android/MobileTransformers/MobileTransformersApp`.
3. **Edit `settings.gradle.kts`** to the end state above (`rootProject.name`, two `include` lines).
4. **Move SDK Kotlin package dir.** `git mv …/MobileTransformers/src/main/java/com/martinkorelic/ortmobile …/com/martinkorelic/mobiletransformers` (and the matching `test`/`androidTest` trees). Then rewrite `package com.martinkorelic.ortmobile` → `package com.martinkorelic.mobiletransformers` and all internal `import com.martinkorelic.ortmobile.` → `import com.martinkorelic.mobiletransformers.` across those files. Keep sub-packages (`repository`, `entity`) intact — only the prefix changes.
5. **Edit SDK `build.gradle.kts`**: `namespace = "com.martinkorelic.mobiletransformers"`. Leave `jniLibs.srcDirs("libs")`, the `externalNativeBuild { cmake { path("src/main/cpp/CMakeLists.txt") } }`, ABI filters, and the `onnxruntime-genai.aar` `files(...)` reference unchanged.
6. **Move app Kotlin package dir** to `com/martinkorelic/mobiletransformers/app`, rewrite `package`/`import` lines (`com.martinkorelic.orttransformer` → `com.martinkorelic.mobiletransformers.app`). The app also imports the SDK as `com.martinkorelic.ortmobile.repository.*` (see `MainActivity.kt`) — rewrite those to `com.martinkorelic.mobiletransformers.repository.*`.
7. **Edit app `build.gradle.kts`**: `namespace = "com.martinkorelic.mobiletransformers.app"`, `applicationId = "com.martinkorelic.mobiletransformers.app"`, and `implementation(project(":MobileTransformers"))`.
8. **Theme/resource refs.** Rename the `Theme.ORTTransformer` style and update `android:theme="@style/Theme.ORTTransformer"` occurrences in `AndroidManifest.xml` to the renamed style (e.g. `Theme.MobileTransformers`). `app_name` string is cosmetic; update for clarity.
9. **Add `Aliases.kt`** compat shims under the old package (step in Data contracts).
10. **Add AAR publishing coordinate** (optional this pass, but cheap): `maven-publish` block in SDK `build.gradle.kts` declaring artifactId `mobiletransformers-android`. Do not wire a real repository yet.
11. **Regenerate ObjectBox.** Delete stale generated sources/`objectbox-models/` if checked in; a clean build regenerates `MyObjectBox` and `VectorEntity*_` under the new package. See risks.
12. **Verify** (see Tests & smokes) and commit as a single self-contained migration commit.

## Interactions

- **JNI / `System.loadLibrary("ortmobile")` — DO NOT rename here.** The C++ exports are mangled by package path: e.g. `Java_com_martinkorelic_ortmobile_ORTGeneratorNative_createInferenceSession` and `Java_com_martinkorelic_ortmobile_ORTTrainerNative_performTraining` in `…/src/main/cpp/native-lib.cpp`, plus `Java_com_martinkorelic_ortmobile_ORTGenAINative_*` in `onnx-genai.cpp`. The Kotlin classes declaring these `external fun`s (`ORTGeneratorNative`, `ORTTrainerNative`, `ORTTokenizerNative`, `ORTGenAINative`) move to package `com.martinkorelic.mobiletransformers`, which **changes the JVM-expected symbol name** to `Java_com_martinkorelic_mobiletransformers_*`. Two options — pick (A) for this isolated pass:
  - **(A) Keep `*Native` classes in the legacy package.** Leave `ORTGeneratorNative.kt`, `ORTTrainerNative.kt`, `ORTTokenizerNative.kt`, `ORTGenAINative.kt` under `package com.martinkorelic.ortmobile` so the existing C++ symbols keep resolving. Repositories (now in `…mobiletransformers.repository`) import them from the old package. No C++ edit, no symbol churn. This preserves `loadLibrary("ortmobile")` and `project("ortmobile")` in CMake verbatim. **Recommended.**
  - (B) Move the `*Native` classes too and rename every `Java_com_martinkorelic_ortmobile_*` symbol in `native-lib.cpp`/`onnx-genai.cpp` to `…_mobiletransformers_*`. Higher risk; defer to #16 or later with JNI smoke tests in place.
- **`MainActivity` JNI mismatch (pre-existing).** `MainActivity.kt` lives in the app package and calls `System.loadLibrary("ortmobile")` with `external fun stringFromJNI()`, but the C++ symbol is `Java_com_martinkorelic_ortmobile_MainActivity_stringFromJNI` (note `ortmobile`, not `orttransformer`). This `stringFromJNI` binding is already mismatched/unused for resolution from the app package today. Do not "fix" it in this pass — just preserve current behavior. Moving `MainActivity` to `…mobiletransformers.app` does not make it match either; leave the demo call as-is or guard it. Flag for #16.
- **CMake `jniLibs` paths.** `CMakeLists.txt` uses `${PROJECT_SOURCE_DIR}/../jniLibs/${CMAKE_ANDROID_ARCH_ABI}` and `${CMAKE_SOURCE_DIR}/onnxruntime`, all **relative to the module's `src/main/cpp`**. Since the whole module dir is moved as a unit, these relative paths stay valid. `project("ortmobile")` and `add_library(${CMAKE_PROJECT_NAME} SHARED …)` produce `libortmobile.so` — keep that name (matches `loadLibrary("ortmobile")`). No CMake edits required under option (A).
- **ObjectBox generated classes.** `ORTVectorDatabase.kt` imports `com.martinkorelic.ortmobile.entity.MyObjectBox` and `VectorEntity{64..1536}_`. These are generated from the `@Entity` classes in `entity/VectorEntity.kt`. After moving the entity package to `…mobiletransformers.entity`, the generated `MyObjectBox`/`*_` classes regenerate under the new package; update the imports in `ORTVectorDatabase.kt` accordingly. The ObjectBox plugin/version (`io.objectbox` 4.3.0 via `libs.versions.toml`) and the `objectbox-models/default.json` model id mapping are unaffected by package rename, but force a clean build to avoid stale generated sources.

## Tests & smokes

1. **Build gate (primary):** `cd android/MobileTransformers && ./gradlew :MobileTransformers:assembleDebug :MobileTransformersApp:assembleDebug` must pass clean.
2. **Native packaging check:** confirm `libortmobile.so` is packaged for `arm64-v8a` and `x86_64` in the AAR/APK (`unzip -l` the outputs); confirms CMake + `loadLibrary` name survived.
3. **Existing instrumented/unit stubs:** `./gradlew :MobileTransformers:testDebugUnitTest` and the moved `ExampleInstrumentedTest` should still compile/run.
4. **Manual launch smoke:** install `MobileTransformersApp` debug APK; verify it launches and the existing train/inference/RAG screens still drive the (legacy-package) `*Native` classes without `UnsatisfiedLinkError`.
5. **Compat-alias compile check:** a throwaway source importing `com.martinkorelic.ortmobile.repository.LLMRepository` should still compile (with deprecation warning), proving the alias layer.
6. **Diff discipline:** `git diff --stat` should show only renames + identifier edits + `Aliases.kt`; no logic changes. Reviewer rejects any behavioral diff.
