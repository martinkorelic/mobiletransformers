# examples/

Worked examples that consume MobileTransformers the way an outside project would.

| example | what it demonstrates |
| --- | --- |
| [`consumer-app/`](consumer-app/) | A minimal Android app that depends on the published `mobiletransformers-android` AAR from `mavenLocal` and calls into it. |

`consumer-app` is deliberately tiny and deliberately *outside* the main Gradle build. It is the only
check that the artifact this project publishes is actually consumable from another project — the
library's own test suites all run inside the build that produced it, so they cannot catch a broken
POM, a missing transitive dependency or a coordinate that does not resolve.

```bash
make publish-local     # publish the AAR to ~/.m2/repository
make consumer-app      # build this against it
```

Its `gradle.properties` pins the SDK version to resolve, and `tests/unit/test_version_sites.py`
asserts that pin matches the version this repository publishes — it was silently one minor behind
until that guard was added.
