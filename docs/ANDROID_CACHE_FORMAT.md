# Android cache format

Where an installed model lives on device, and the rules the SDK follows when writing to it. The package
*schema* is in [MODEL_FORMAT.md](MODEL_FORMAT.md); the Hub side is in
[HUB_PACKAGE_FORMAT.md](HUB_PACKAGE_FORMAT.md).

## Layout

The cache root defaults to `context.filesDir`. One directory per model, named by
[`sanitize_repo_id`](HUB_PACKAGE_FORMAT.md#sanitize_repo_id):

```
<cacheDir>/
├── org__Tiny-Model/
│   ├── mobiletransformers_manifest.json
│   ├── checksums.json                    # the installed variant's digests
│   ├── tokenizer/                        # flattened from shared/tokenizer (+ chat_template.jinja)
│   ├── inference/
│   │   ├── model.onnx, model.onnx_data
│   │   ├── generation_config.json
│   │   ├── genai_config.json             # present iff the GenAI engine is supported
│   │   ├── weight_handoff_map.json
│   │   ├── <tensor>.bin, <tensor>.bin.sha256
│   │   └── merger_*.onnx
│   ├── train/                            # present iff the training feature was installed
│   │   ├── training_model.onnx, eval_model.onnx, optimizer_model.onnx
│   │   ├── training_config.json, trainable_parameters.json
│   │   ├── checkpoint/
│   │   └── training_state.json           # written by the trainer, not the installer
│   └── embedding/                        # present iff the RAG feature was installed
└── .staging/, .download/                 # transient; removed on success
```

**The variant is flattened away.** On the Hub a package holds several variants; on device exactly one is
installed, so `variants/<id>/inference/` becomes `inference/`. The installed variant id is recoverable
from the manifest that ships alongside it.

This is the layout `LLMRepository` already probed before packages existed, which is why installation is
a pure file operation and the runtime needed no changes to consume Hub models.

## Install is crash-safe

`ModelPackageInstaller` (Kotlin) and `hub/pull.py::install_package` (Python) follow the same sequence:

1. Build the complete new tree in a staging directory (`.staging/<sanitized>` / `.partial/<sanitized>`).
2. Rename any **existing** install aside to `.retired-<sanitized>-<n>`.
3. Rename the staged tree into place.
4. Delete the retired tree.

If step 3 fails, the retired tree is renamed back — a failed update is a no-op, not data loss. The
ordering matters because a model directory can hold locally trained state (`train/checkpoint`,
`training_state.json`) that exists nowhere else: deleting the live tree before the replacement is in
place would destroy it if the process died in between.

Staging directories are always fully removed on success, so `.staging/`/`.download/`/`.retired-*` are
never part of a healthy cache.

## Reading the cache

`CacheIndex.list(cacheDir)` enumerates installed models with their base model id, size and whether a
manifest is present — the backing for a package-management UI. It reads only the manifest and the
directory tree; it never loads a model.

## Writes after install

Only two things modify a model directory after installation:

| Writer | Writes | When |
| --- | --- | --- |
| `ORTTrainerNative` | `train/checkpoint`, `train/training_state.json`, `train/training_logs.json` | during/after training |
| `weight_merger.cpp` | `inference/<tensor>.bin` + `.bin.sha256` | on merge |

The merger overwrites tensors **in place** in `inference/`, atomically (write to a temp file, fsync,
rename) with a refreshed checksum sidecar. There is no separate `merged/` directory — the inference
graph references those exact filenames via `weight_handoff_map.json`, so a merge is complete the moment
the renames land. See [MODEL_FORMAT.md](MODEL_FORMAT.md#checksum-precedence-the-sidecar-wins) for which
checksum wins afterwards.

## Failure modes

All fail closed:

| Situation | Behaviour |
| --- | --- |
| No `weight_handoff_map.json` | nothing was merged — load the base weights (not an error) |
| Map present, a `.bin` missing or checksum mismatched | `MissingArtifactException` naming the tensor |
| Merged tensor's dtype/shape/size disagrees with the map | native session creation fails; no fallback to base weights |
| Partial merge | the merge reports failure; the package is not presented as trained |
| Requested variant incompatible with the device | `NoCompatibleVariantException` before downloading |
