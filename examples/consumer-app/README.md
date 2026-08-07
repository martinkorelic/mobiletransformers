# Consumer app example

A minimal external project that resolves the MobileTransformers SDK **as a published Maven artifact**,
not as a source module. It exists to prove the #30 publication contract: if this builds, an outside
consumer can depend on the AAR.

## Use

```bash
# 1. publish the SDK to your local Maven repository
scripts/publish_local_maven.sh

# 2. build this project against it
cd examples/consumer-app && ./gradlew assembleDebug
```

`settings.gradle.kts` puts `mavenLocal()` first so the freshly published artifact wins over any remote.

## What it checks

- The published POM resolves and drags in the SDK's transitive dependencies.
- The public facade (`MobileTransformers.fromPretrained`, `MobileTransformerModel`) is visible to an
  outside module — i.e. nothing public is accidentally `internal`.
- The AAR carries `libmobiletransformers.so` for the consumer's ABI.

It deliberately does **not** run a model: that needs a package on a device and belongs to the
instrumented suite.
