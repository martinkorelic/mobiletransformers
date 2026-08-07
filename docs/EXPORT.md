# Export

One command turns a Hugging Face model into a device-ready MobileTransformers package. This page is
sourced from the export CLI (`02_code_plans/05`, #15) and the dependency profiles
(`00_code_plans/03`, #2).

## One-command export

```bash
mobiletransformers export --model <hf-repo-id> --output build/package
```

Common flags (see `mobiletransformers export --help`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | *(required)* | HF repo id to export. |
| `--output` | *(required)* | Output package directory. |
| `--task` | auto | Optimum task; auto-selected from the model when omitted. |
| `--peft` | `lora` | `lora` \| `lora-xs` \| `mars` \| `mars-opt0..mars-opt4`. |
| `--rank` | `8` | LoRA/MARS rank. |
| `--quant` | `int4` | `qint8` \| `int4` \| `fp16`. |
| `--variant` | `cpu-<quant>` | Variant id in the package manifest. |
| `--include-rag` | off | Also emit the embedding/RAG variant subtree. |
| `--embedding-model` | — | Embedding model id (with `--include-rag`). |
| `--genai` | off | Declare GenAI engine support for the variant (Native is always supported). |
| `--stages` | auto | Comma-separated `inference,training,embedding`; default is auto by profile. |
| `--config` | — | YAML supplying defaults for any flag not passed (CLI > YAML > default). |
| `--validate` | off | Validate the written package against the manifest contract before returning. |
| `--dry-run` | off | Resolve + print the plan and a manifest skeleton; write nothing. |

### `--config`

Any knob above except `--validate`/`--dry-run` may come from YAML instead of the command line —
including `--model` and `--output`. Explicit flags always win; unknown keys are rejected rather than
ignored.

```yaml
# export.yml — either an `export:` block or a flat mapping
export:
  model: HuggingFaceTB/SmolLM2-135M-Instruct
  output: build/pkg
  peft: mars
  rank: 16
  quant: int4
  genai: true
```

```bash
mobiletransformers export --config export.yml            # everything from YAML
mobiletransformers export --config export.yml --rank 32  # ...with rank overridden
```

### `--validate`

Re-reads the package just written and runs the #13 manifest validation over it (every declared file
resolves, the variant subtrees exist, the weight-handoff reference is present). A package that does not
validate fails the command, rather than being discovered later on a device. The same check is available
standalone:

```bash
mobiletransformers validate --package build/pkg
```

`--dry-run` needs no heavy dependencies — it resolves the export plan and prints the
`mobiletransformers_manifest.json` skeleton, so you can inspect variant selection and the download plan
before committing to a full export.

## What the package contains

The output is a single Hub-shaped package (one tree, variants declare their engines), verified against
the manifest/cache contract (`00_code_plans/06`, #13) and the Hub package format (`02_code_plans/03`,
#14):

- `mobiletransformers_manifest.json` — schema-versioned manifest (`sha256` + `fileSizes` +
  `downloadPlan`, per-variant `checksums.json`).
- A flat `inference/` layout: `model.onnx` + `frozen_base.onnx.data` (immutable quantized base) +
  per-tensor `<name>.bin` (+ `.sha256`) trainable external initializers, beside
  `generation_config.json` / `genai_config.json` / `weight_handoff_map.json`.
- `weight_handoff_map.json` — the single source of tensor identity for on-device merge
  (`00_code_plans/07`, #8).

## Profiles (dependency isolation)

The `onnxruntime`-bearing profiles collide on the `onnxruntime` import and must **never** co-install;
each `make setup*` target syncs its own environment:

| Target | Profile | Notes |
| --- | --- | --- |
| `make setup` | core + `dev` | No onnxruntime provider. Python 3.10-clean. |
| `make setup-export` | `export` extra | `optimum-onnx[onnxruntime]`; needs Python ≥ 3.11. |
| `make setup-train` | `ort-training-local` group | Source-built ORT-training wheel; **cp312 + Linux only**. |
| `make setup-genai` | `genai-smoke` group | `onnxruntime-genai`; needs Python ≥ 3.11. |

## Real vs. dry-run export

- **Dry-run** (`--dry-run`) runs anywhere the core package installs — no model download, no heavy deps.
- **Real full export** exercises the optimum inference-graph export (`export` profile) and, for training
  artifacts, the source-built ORT-training toolchain (`ort-training-local`, cp312/Linux). It is
  **environment-gated**: `mobiletransformers export` (without `--dry-run`) raises a clear message until
  run under those profiles.

## Publish

```bash
mobiletransformers push --package build/package --repo <org/name>
```

The push wraps `huggingface_hub.upload_folder`, validates the package + renders a model card before
uploading (`--dry-run` validates + renders the card without uploading). See also
`mobiletransformers pull --repo-id <org/name>` and `install-package` for the consumer side.
