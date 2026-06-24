# AAR & Local-Maven Publication

**Priority #29 | Prerequisites: #15 (`00_code_plans/04_android_gradle_rename_migration.md`), #27 (`05_code_plans/01`) | Blocks: #31 (`05_code_plans/05`, release)**

## Purpose

Make the Android SDK consumable as a Maven/Gradle AAR (table stakes for "portable," matching MobileFineTuner's distribution). Add `maven-publish` to the renamed library module, publish to mavenLocal, and prove a tiny external consumer app builds against it. **Resolve the missing native artifact first** — the library currently references `./src/main/aarLibs/onnxruntime-genai.aar` and a `libs` JNI dir that do not exist in the repo, so any clean AAR build/publish fails today.

## Touched / new files

- `android/ORTransformer/ORTransformersMobile/build.gradle.kts` — add `maven-publish` + `publishing { publications { ... } }`; resolve `implementation(files("./src/main/aarLibs/onnxruntime-genai.aar"))` (currently missing) and the `srcDirs("libs")` JNI dir (currently missing).
- `android/ORTransformer/CMakeLists.txt` path (`.../cpp/CMakeLists.txt`) — links `onnxruntime` + `onnxruntime-genai` (`CMakeLists.txt:61-62`); ensure the genai lib is actually present or made optional behind Gate 0.1.
- NEW `scripts/android_build_aar.sh`, `scripts/publish_local_maven.sh` (stubs from #27).
- NEW (example) `examples/consumer-app/` — minimal app consuming the local-Maven artifact (CI smoke for #28).

## Data contracts / interfaces

### Maven coordinates (compatibility surface — SemVer-bound)

```kotlin
// in the renamed :MobileTransformers module
group   = "com.martinkorelic.mobiletransformers"      // final org namespace
artifact = "mobiletransformers-android"
version  = project.findProperty("version") ?: "0.1.0-SNAPSHOT"   // release tag drives it (#31)
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
2. Apply `maven-publish` to the renamed module (`:MobileTransformers` after #15); during transition, `build-aar` may call the legacy `:ORTransformersMobile` task via a compatibility target (#27).
3. Configure the publication (coordinates above), POM metadata, and ABI packaging.
4. Write `scripts/android_build_aar.sh` (assemble release AAR) and `scripts/publish_local_maven.sh` (`publishToMavenLocal`).
5. Add `examples/consumer-app/` and wire a CI smoke (#28) that builds it against the local artifact.

## Interactions

- **#15 (rename)**: publication targets the renamed module; coordinates assume the new namespace.
- **#27 (Makefile)**: `build-aar` / `publish-local` call these scripts.
- **#28 (CI)**: tag builds publish; consumer-app smoke runs against mavenLocal.
- **Gate 0.1 / #10**: whether `onnxruntime-genai` ships is gated; the AAR must build Native-only if GenAI is deferred.

## Tests & smokes

- `scripts/android_build_aar.sh` produces an AAR containing `arm64-v8a` + `x86_64` native libs.
- `scripts/publish_local_maven.sh` installs the artifact into mavenLocal.
- `examples/consumer-app/` compiles and links against the local artifact (CI smoke).
- Missing-native regression: with the genai AAR absent, the Native-only build still succeeds (no dangling `files(...)` reference).
- POM metadata validates (coordinates, license — set to Apache-2.0 per #31).
