# Export

One command turns a Hugging Face model into a device-ready MobileTransformers package. This page is
sourced from the export CLI and the dependency profiles declared in `pyproject.toml`.

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
| `--peft-target` | *(registry)* | Comma-separated modules PEFT adapts, e.g. `q_proj,v_proj`. Omit to use the architecture registry's row for the model — see below. |
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

### Which modules PEFT adapts (`--peft-target`)

By default this is **per model, from the architecture registry**
(`src/mobiletransformers/config/registry/architecture.py`) — one `target_modules` row per
architecture. That is the place to edit to support a new model or change a family's defaults, and it
applies everywhere without a caller having to remember a flag:

```python
"LlamaForCausalLM": ArchitectureSpec("LlamaForCausalLM", ..., ("q_proj", "v_proj")),
"DistilBertForSequenceClassification": ArchitectureSpec(..., ("q_lin", "v_lin")),
```

Override per run when you want something else:

```bash
mobiletransformers export --model <id> --output build/pkg --peft-target q_proj,k_proj,v_proj,o_proj
```

or in YAML as `peft_target: q_proj,v_proj`.

> **Changed 2026-08-09.** The default used to be a hardcoded `q_proj,k_proj` that ignored the registry
> and could not express an encoder's `query`/`value` at all. Decoder exports now adapt **Wq and Wv**
> (the LoRA convention) rather than Wq/Wk. Pass `--peft-target q_proj,k_proj` to reproduce the old
> behaviour.

### `--validate`

Re-reads the package just written and runs the full manifest validation over it (every declared file
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
the manifest/cache contract and the Hub package format:

- `mobiletransformers_manifest.json` — schema-versioned manifest (`sha256` + `fileSizes` +
  `downloadPlan`, per-variant `checksums.json`).
- A flat `inference/` layout: `model.onnx` + `frozen_base.onnx.data` (immutable quantized base) +
  per-tensor `<name>.bin` (+ `.sha256`) trainable external initializers, beside
  `generation_config.json` / `genai_config.json` / `weight_handoff_map.json`.
- `weight_handoff_map.json` — the single source of tensor identity for on-device merge
  (`src/mobiletransformers/artifacts/handoff_map.py`).

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

## Per-task flag rules

Two flags are decided by the task, not by preference, and getting either wrong fails late — or, worse,
silently produces a package that is missing half of what you wanted.

**`--task text-classification` is what makes an encoder trainable at all.** `TaskSpec.default_stages`
emits a training stage exactly when the task is `trainable`, and `FEATURE_EXTRACTION` is declared
`trainable=False`. So exporting an encoder the "natural" way yields an **inference-only** package, with
no error — it did exactly what you asked. Task auto-selection never picks `text-classification`; it has
to be named.

The consequence for a sentence encoder is that its classification head is randomly initialised, with
`LABEL_0`/`LABEL_1` labels, because no such head exists in the checkpoint. That is correct: the head is
the part fine-tuning learns, and the pretrained *backbone* is what must survive — which is what the
export-time parameter budget checks.

**`--genai` is decoder-only.** A classification or feature-extraction graph has no KV cache, and the
export refuses to write a `genai_config.json` describing a cache the graph does not have. This is a
fail-closed refusal rather than a silent omission, because a `genai_config.json` advertising
`past_key_values.N` inputs the graph lacks produces a device-side failure far from the export.

## Publish

```bash
mobiletransformers push --package build/package --repo <org/name>
```

The push wraps `huggingface_hub.upload_folder`, validates the package + renders a model card before
uploading (`--dry-run` validates + renders the card without uploading). See also
`mobiletransformers pull --repo-id <org/name>` and `install-package` for the consumer side.

### Publishing the whole shelf

```bash
make publish-catalog                              # export + gate-check + push every entry
ONLY=smollm2 PUSH=0 scripts/publish_catalog.sh    # one entry, no upload
KEEP=1 scripts/publish_catalog.sh                 # skip re-export where a package already exists
```

`scripts/publish_catalog.sh` holds the per-model task and engine flags above as data, so they cannot be
mistyped per run. It also does two things worth knowing about:

**It gate-checks that every entry ships a `train` group** and refuses otherwise. A shelf entry that
cannot be fine-tuned demonstrates half the framework, so this is asserted rather than assumed.

**It performs the two-profile dance.** The inference export needs the `export` extra; the training
stage needs the source-built `ort-training-local` wheel; the two collide on the `onnxruntime` import
and must never co-install. `uv run --group ort-training-local` alone does **not** displace the stock
onnxruntime the export profile just installed — the training wheel provides a distribution of the same
name, so the resolver considers the requirement satisfied and the training import then dies with
`ImportError: cannot import name 'PropagateCastOpsStrategy'`. An explicit `uv sync
--reinstall-package` followed by `uv run --no-sync` is what actually works.

It needs `HF_TOKEN_ORG` in `.env` — a token with `repo.write` on the target org. A fine-grained
personal token scoped to one repo returns `RepositoryNotFoundError` for every other repo, and the Hub
returns that identically for "does not exist" and "you cannot see it", so a permissions problem reads
as a typo. See [`.env.example`](https://github.com/martinkorelic/mobiletransformers/blob/main/.env.example) and [CATALOG.md](CATALOG.md).

> The script leaves the tree on the training profile. Reset with
> `uv sync --frozen --group dev --python 3.10` before running `make check`.
