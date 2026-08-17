# Hub package format

How a MobileTransformers package is laid out **on the Hugging Face Hub**, and how a client turns a repo
id into an installed model. The manifest and weight-handoff *schemas* are specified in
[MODEL_FORMAT.md](MODEL_FORMAT.md); this page covers the repository shape, the download plan and the
verify/install flow. Owner: `src/mobiletransformers/hub/package_format.py`.

## Repository layout

A package repo is the on-disk package published verbatim — no repacking, no archives:

```
<hub-repo>/
├── mobiletransformers_manifest.json     # entry point: fetched FIRST, before any large file
├── README.md                            # model card
├── shared/
│   ├── tokenizer/                       # tokenizer.json, tokenizer_config.json, …
│   └── chat_template.jinja              # optional; present when the model declares one
└── variants/
    └── <variant-id>/                    # e.g. cpu-int4
        ├── checksums.json               # per-file sha256 for this variant
        ├── inference/
        ├── train/                       # optional (training feature)
        └── embedding/                   # optional (rag feature)
```

Everything shared across variants lives under `shared/` and is downloaded once. A variant id is
`<execution-provider>-<quantization>`, e.g. `cpu-int4`.

## Manifest-first, always

The manifest is small and names the checksums of everything else, so it is fetched before any large
file. That ordering is the reason a client can verify what it downloads:

1. `GET mobiletransformers_manifest.json` → parse + version-gate (`check_compat`).
2. **Select a variant** against device capability — ABI, available memory, requested features,
   requested engine. `manifest.defaultVariant` is a fallback, not the answer: selecting it blindly
   would happily download an ABI-incompatible variant that fails at load.
3. **Plan the file list** from the selected variant + requested features (a `train/` subtree is not
   downloaded for an inference-only install; `genai_config.json` only when GenAI was requested).
4. **Download + verify** each file against `manifest.sha256`. A mismatch aborts the install.
5. **Install** into the device cache — see [ANDROID_CACHE_FORMAT.md](ANDROID_CACHE_FORMAT.md).

Both clients implement this identically: `hub/pull.py` (Python) and `hub/HubDownloader.kt` +
`packages/VariantSelector.kt` (Kotlin), with `DownloadPlanner` resolving globs to a concrete file list.

## `sanitize_repo_id`

A Hub repo id contains `/`, which cannot be a directory name. `sanitize_repo_id()` maps it to the cache
directory name:

| Repo id | Sanitized |
| --- | --- |
| `org/Tiny-Model` | `org__Tiny-Model` |

The mapping is pinned by a shared fixture (`tests/fixtures/sanitize_repo_id_cases.json`) that the Python
and Kotlin implementations are both tested against, so the two can never disagree about where a model
lives.

## Checksums

Two layers, deliberately:

- `manifest.sha256` — every file in the package, used to verify a **download**.
- `variants/<id>/checksums.json` — the same digests scoped to one variant, installed alongside the model
  so integrity can be re-checked later without the full manifest.

Merged weights add a third, device-side layer (`<name>.bin.sha256` sidecars) with its own precedence
rule — see [MODEL_FORMAT.md](MODEL_FORMAT.md#checksum-precedence-the-sidecar-wins).

## Publishing

```bash
mobiletransformers export --model <hf-id> --output build/pkg --genai --validate
mobiletransformers push  --package build/pkg --repo <org>/<name> --token "$HF_TOKEN_ORG"
```

`push` validates the package against the manifest contract before uploading, so a broken package fails
locally rather than becoming a broken repo. See [EXPORT.md](EXPORT.md).

- **The repo must already exist.** Pass `--create` to create it. That is off by default so a mistyped
  repo id fails instead of silently making a new one — under an organisation account, a typo would
  otherwise leave a stray repo behind.
- **`--token` is explicit for a reason.** Without it `huggingface_hub` falls back to `$HF_TOKEN` and
  then to the cached CLI login, so an organisation push can succeed as the *wrong identity* and look
  exactly like success. `--dry-run` renders the model card and writes `README.md` without uploading.
- The card carries a **YAML frontmatter block** (`base_model`, `library_name`, `pipeline_tag`, `tags`,
  and `license` when the package records the base weights' licence). Without it the Hub page shows no
  licence and no link back to the model the package was exported from — the prose body says both, and
  the Hub does not read prose.

## Pulling

Pass `--token` for a private or gated repo (defaults to `$HF_TOKEN`, and honours `.env`).

```bash
mobiletransformers pull --repo-id <org>/<name> --output <cache-root>
mobiletransformers install-package --package <staged> --cache <cache-root>
```

On Android the same flow runs inside `MobileTransformers.fromPretrained`, which pulls and installs when
the package is not already in the cache. Background downloads use
`PackageDownloadWorker.enqueue(...)` (WorkManager; unmetered + storage-not-low, one unique job per repo).
