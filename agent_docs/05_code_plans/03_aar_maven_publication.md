# AAR & Local-Maven Publication

**Priority #30 | Prerequisites: #16 (`00_code_plans/04_android_gradle_rename_migration.md`), #28 (`05_code_plans/01`) | Blocks: #32 (`05_code_plans/05`, release)**

## Purpose

Make the Android SDK consumable as a Maven/Gradle AAR (table stakes for "portable," matching MobileFineTuner's distribution). Add `maven-publish` to the renamed library module, publish to mavenLocal, and prove a tiny external consumer app builds against it. **Resolve the missing native artifact first** — the library currently references `./src/main/aarLibs/onnxruntime-genai.aar` and a `libs` JNI dir that do not exist in the repo, so any clean AAR build/publish fails today.

## Touched / new files

- `android/ORTransformer/ORTransformersMobile/build.gradle.kts` — add `maven-publish` + `publishing { publications { ... } }`; resolve `implementation(files("./src/main/aarLibs/onnxruntime-genai.aar"))` (currently missing) and the `srcDirs("libs")` JNI dir (currently missing).
- `android/ORTransformer/CMakeLists.txt` path (`.../cpp/CMakeLists.txt`) — links `onnxruntime` + `onnxruntime-genai` (`CMakeLists.txt:61-62`); ensure the genai lib is actually present or made optional behind Gate 0.1.
- NEW `scripts/android_build_aar.sh`, `scripts/publish_local_maven.sh` (stubs from #28).
- NEW (example) `examples/consumer-app/` — minimal app consuming the local-Maven artifact (CI smoke for #29).

## Data contracts / interfaces

### Maven coordinates (compatibility surface — SemVer-bound)

```kotlin
// in the renamed :MobileTransformers module
group   = "com.martinkorelic.mobiletransformers"      // final org namespace
artifact = "mobiletransformers-android"
version  = project.findProperty("version") ?: "0.1.0-SNAPSHOT"   // release tag drives it (#32)
```

Consumer:

```kotlin
repositories { mavenLocal() }
dependencies { implementation("com.martinkorelic.mobiletransformers:mobiletransformers-android:1.0.0") }
```

### ABI / native packaging contract

- Package native libs for `arm64-v8a` and `x86_64` (the ABIs already configured in Gradle).
- Bundle/declare `onnxruntime` and `onnxruntime-genai` correctly: third-party AARs are not transitively published via mavenLocal by default — either vendor the `.so`s into the published AAR or declare them as explicit consumer dependencies. Document which.
- The publication includes the AAR, sources jar if feasible, and generated POM (Android library publishing + Gradle Maven Publish Plugin).

## Implementation steps

1. **Restore the missing native inputs**: add the `onnxruntime-genai.aar` (or make it an optional, Gate-0.1-gated dependency) and the `libs` JNI dir, or remove the dead references if Native-only is shipped first.
2. Apply `maven-publish` to the renamed module (`:MobileTransformers` after #16); during transition, `build-aar` may call the legacy `:ORTransformersMobile` task via a compatibility target (#28).
3. Configure the publication (coordinates above), POM metadata, and ABI packaging.
4. Write `scripts/android_build_aar.sh` (assemble release AAR) and `scripts/publish_local_maven.sh` (`publishToMavenLocal`).
5. Add `examples/consumer-app/` and wire a CI smoke (#29) that builds it against the local artifact.

## Interactions

- **#16 (rename)**: publication targets the renamed module; coordinates assume the new namespace.
- **#28 (Makefile)**: `build-aar` / `publish-local` call these scripts.
- **#29 (CI)**: tag builds publish; consumer-app smoke runs against mavenLocal.
- **Gate 0.1 / #11**: whether `onnxruntime-genai` ships is gated; the AAR must build Native-only if GenAI is deferred.

## References

- `https://developer.android.com/studio/build/maven-publish-plugin` — maven-publish plugin for AAR publication.
- `https://developer.android.com/build/publish-library/upload-library` — uploading an Android library.
- `https://developer.android.com/build/publish-library/configure-pub-variants` — configuring publish variants.
- `https://vanniktech.github.io/gradle-maven-publish-plugin/` — vanniktech publish plugin (POM/sources boilerplate).
- `https://vanniktech.github.io/gradle-maven-publish-plugin/central/` — publishing to Maven Central.

## Worked example

Library publication (`maven-publish`):

```kotlin
publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.martinkorelic.mobiletransformers"
            artifactId = "mobiletransformers-android"
            version = project.findProperty("version")?.toString() ?: "0.1.0-SNAPSHOT"
            from(components["release"])
        }
    }
}
```

Consumer app resolving it from `mavenLocal()`:

```kotlin
repositories { mavenLocal() }
dependencies {
    implementation("com.martinkorelic.mobiletransformers:mobiletransformers-android:1.0.0")
}
```

Note: third-party AARs (`onnxruntime-genai`) are **not** transitively published via `mavenLocal()` — either vendor the `.so`s into the published AAR or declare them as explicit consumer dependencies.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- POM metadata validates (coordinates, license — set to Apache-2.0 per #32).
- Missing-native regression: with the genai AAR absent, the Native-only build still succeeds (no dangling `files(...)` reference); the module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `scripts/android_build_aar.sh` produces an AAR containing `arm64-v8a` + `x86_64` native libs.
- `scripts/publish_local_maven.sh` installs the artifact into mavenLocal (assert the coordinates appear under `~/.m2`).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- `examples/consumer-app/` compiles and links against the local artifact (also wired as the #29 CI smoke).

**Workflow (end-to-end)** — *checkpoint #30.*
- Build the AAR → `publishToMavenLocal` → a minimal external consumer app resolves and builds against `com.martinkorelic.mobiletransformers:mobiletransformers-android` from `mavenLocal()`, with the native libs reachable (vendored or declared).

**Definition of done** — the renamed `:MobileTransformers` module applies `maven-publish` with the canonical coordinates, the missing-native references are resolved (vendored or Gate-0.1-gated), `publishToMavenLocal` succeeds, and an external consumer app builds + resolves against the local artifact end-to-end.
