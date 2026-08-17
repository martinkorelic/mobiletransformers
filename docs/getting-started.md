# Getting started

Three routes in, depending on what you want. They are independent — you do not need the Python side
to run the app, and you do not need Android to export a package.

| I want to… | Go to |
| --- | --- |
| **see it work on a phone**, with no Python at all | [Run the sample app](#run-the-sample-app) |
| **use the SDK in my own Android app** | [Consume the SDK](#consume-the-sdk) |
| **export my own model** | [Set up the Python side](#set-up-the-python-side) |

!!! tip "Run `make doctor` first"

    It reports every prerequisite — uv, Python 3.10 and 3.12, the current venv profile, the
    ORT-training wheel, `JAVA_HOME`, the Android SDK, `adb`, the vendored natives, the `.env` tokens
    — and the exact command that fixes each one. It downloads nothing.

## Requirements

| | |
| --- | --- |
| **Python** | 3.10–3.13 for the core and the exporter. The training-export path additionally needs **3.12**, because the ONNX Runtime Training wheel is built `cp312` only. |
| **Package manager** | [uv](https://docs.astral.sh/uv/). The lock file covers every profile; `pip` is not supported. |
| **Android** | API 24+ to run, JDK 17 and the Android SDK + NDK to build. |
| **ABI** | **arm64-v8a only.** There is no x86_64 build, so the SDK does **not** run on a standard Android emulator — you need a physical device. |
| **OS for training** | The training side is Linux x86_64, because of that same wheel. Inference, export-without-training and the whole Android side are platform-independent. |

## Run the sample app

```bash
git clone https://github.com/martinkorelic/mobiletransformers
cd mobiletransformers

make fetch-native-deps        # ~180 MB of prebuilt natives, see below
make android-build            # builds the SDK and the app
```

Then install a package from inside the app: open **Models**, pick **SmolLM2-135M-Instruct** from the
catalog (the smallest useful chat model, and the fastest to train), and press Install. Everything
else in the app unlocks from there — [take the tour](SHOWCASE.md).

!!! warning "The private catalog entries need a token at build time"

    `make android-build` does not source `.env`, and the Hub token is baked into the APK at build
    time — so an app built without one silently cannot pull the private catalog entries. Run
    `set -a && . ./.env && set +a` first if you need them. See `.env.example` for which token does
    what.

### Why a clone cannot build on its own

About 180 MB of prebuilt native binaries and vendored headers are gitignored: ONNX Runtime built for
training on Android, the GenAI engine, the tokenizer static libraries, and protobuf headers. They are
too large for git and cannot be rebuilt quickly.

`make fetch-native-deps` gets them from a public Hugging Face dataset repo. It reads
`third_party/android/manifest.json`, checks the archive's SHA-256, unpacks it, then checks every
unpacked file's SHA-256 individually — because a half-populated `jniLibs/` fails the link naming a
*symbol*, not a missing file, and that is an afternoon lost.

```bash
make fetch-native-deps                            # required to build
TRAINING=1 scripts/fetch_native_deps.sh           # + the ORT-training wheel (632 MB), export only
SYMBOLS=1  scripts/fetch_native_deps.sh           # + unstripped binaries, to symbolicate a crash
URL=file:///path/to/dir scripts/fetch_native_deps.sh   # a local mirror
```

No credentials are needed. See [Architecture ▸ native dependencies](ARCHITECTURE.md).

## Consume the SDK

The Android library publishes as `com.martinkorelic.mobiletransformers:mobiletransformers-android`.
Until it is on a public Maven repository, install it locally:

```bash
make publish-local            # -> ~/.m2/repository
```

```kotlin
repositories { mavenLocal() }
dependencies {
    implementation("com.martinkorelic.mobiletransformers:mobiletransformers-android:0.2.0")
}
```

```kotlin
val model = MobileTransformers.fromPretrained(
    context = context,
    repoId  = "mobiletransformers/SmolLM2-135M-Instruct",
    features = setOf(ModelFeature.Inference, ModelFeature.Training),
)
val result = model.generate("Summarise this in one line: …")
```

`fromPretrained` resolves the package's manifest, downloads only the feature groups you asked for,
verifies every file and installs atomically — so a killed download leaves the previous copy intact.
`examples/consumer-app/` is a minimal app that does exactly this and nothing else.

Full surface in [Using the SDK](ANDROID_SDK.md); copy-pasteable recipes per task in the
[cookbook](COOKBOOK.md); the stability contract in [Public API](PUBLIC_API.md).

## Set up the Python side

```bash
make setup                    # core + dev, Python 3.10
make check                    # lint, typecheck, enum parity, guards, unit tests
```

Export a package:

```bash
make setup-export
mobiletransformers export --model HuggingFaceTB/SmolLM2-135M-Instruct \
                          --output build/pkg --train --rag --validate
```

That single command produces the whole package: the ONNX inference graph, a PEFT-enabled training
graph with an optimiser, the tokenizer, an embedding stage if you asked for `--rag`, and the manifest
that ties them together. [Export a model](EXPORT.md) covers the flags, the supported architectures
and what each stage contains.

Push it to the Hub, or push it straight to a connected device:

```bash
mobiletransformers push --package build/pkg --repo your-org/your-model --create
make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1 RAG=1
```

### Dependency profiles do not co-install

This is the single most common way to "break" the repository, so it is worth reading once.

The `export` extra and the `ort-training-local` group **conflict on purpose**: both provide a module
called `onnxruntime`, and installing them together produces an environment where the import that wins
is undefined. `uv` is configured to refuse the combination rather than resolve it.

```bash
uv sync --frozen --group dev --python 3.10     # reset to the core profile
```

Run that before `make check`. Scripts that switch profiles (`scripts/device_package.sh`,
`scripts/publish_catalog.sh`) leave the tree on another one, and the resulting failures look like
unrelated bugs.

Always pass an explicit `--group`/`--extra` to `uv run`, and use `uv run --frozen` — a bare `uv run`
validates every source in the lock before executing, including a git-ignored 662 MB local wheel that
most machines do not have.

## Next

- [A tour of the app](SHOWCASE.md) — one section per capability, and what you should see
- [The model shelf](CATALOG.md) — six published packages, measured sizes, which to start with
- [PEFT methods](on-device-peft.md) — LoRA, LoRA-XS and MARS, and what each costs on a phone
