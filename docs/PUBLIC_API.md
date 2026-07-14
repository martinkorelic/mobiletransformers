# Public API

The SemVer-governed public surface (F5) has three peers: the **Python library API**, the **CLI**, and the
**Kotlin facade**. The Python side is exactly the surface declared in `mobiletransformers.__all__`
(owned by `00_code_plans/10`, guarded by a parity test against `src/mobiletransformers/public_api.txt`).

> The Kotlin facade (`MobileTransformers.fromPretrained`, `ModelSession`, …) is defined by #17/#19 and is
> documented here once those contracts lock. This page currently covers the **Python** + **CLI** surfaces.

## Python (`import mobiletransformers`)

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `__version__` | str | Package version. |
| `resolve` | func | Config resolution with precedence CLI > env > YAML > package default. |
| `get_settings` | func | Load secrets/settings (secrets live only in `Settings`, never in YAML). |
| `Settings` | class | Typed settings/secrets container. |
| `get_logger` | func | Library logger (`NullHandler`; no `print` in library code). |
| `configure_logging` | func | Opt-in logging configuration for applications. |
| `MobileTransformersError` | exc | Base of the exception hierarchy. |
| `ConfigValidationError` | exc | Invalid configuration. |
| `ExportError` | exc | Export/packaging failure. |
| `ManifestError` | exc | Manifest parse/validation failure. |
| `HandoffError` | exc | Weight-handoff-map failure. |
| `MergeError` | exc | Merge failure. |
| `HubError` | exc | Hub pull/push failure. |
| `UnsupportedModelError` | exc | Unsupported architecture/model. |

The exception names mirror the Kotlin hierarchy. This list is authoritative — it is regenerated/guarded
against `public_api.txt`, so any addition is a deliberate SemVer change.

## CLI (`mobiletransformers <command>`)

| Command | Purpose |
| --- | --- |
| `export` | HF model → device-ready package (`--dry-run` supported). |
| `validate` | Validate a package/manifest. |
| `package-model` | Assemble a Hub package from a build dir *(stub; body in a later plan)*. |
| `push` | Validate + publish a package to the Hub. |
| `pull` | Download a package (manifest-first, sha256-verified). |
| `install-package` | Materialize a pulled package into the SDK cache layout. |
| `support-matrix` | Generate `model_support_matrix.json` (+ `--docs`, `--md`). |
| `push-adapter` | Publish a trained adapter (PEFT Mode 1 / native Mode 2). |

Run `mobiletransformers <command> --help` for flags. `make help` lists the wrapper targets
(`export-model`, `package-model`, `android-build`, …).

## Stability

The three surfaces above are the public contract; internal modules (`export.pipeline`, `hub.*`,
`artifacts.*`, `support.*`, `adapter.*`) may change between releases. The full surface is finalized and
version-locked at #32.
