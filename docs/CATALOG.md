# The model catalog

Six packages, published under [`mobiletransformers`](https://huggingface.co/mobiletransformers) on
the Hugging Face Hub. These are the entries the sample app's **Models ▸ Catalog** tab offers, and
they are what the app is meant to be tried with.

Every one ships **both an inference and a training stage**. That is a hard requirement of
`scripts/publish_catalog.sh`, asserted rather than assumed: a shelf entry that cannot be fine-tuned
demonstrates half the framework.

## What is published

| model | task | inference | total | features | PEFT |
| --- | --- | --- | --- | --- | --- |
| [SmolLM2-135M-Instruct](https://huggingface.co/mobiletransformers/SmolLM2-135M-Instruct) | text-generation | 663 MB | 935 MB | inference, train, rag | LoRA |
| [functiongemma-270m-it](https://huggingface.co/mobiletransformers/functiongemma-270m-it) | text-generation | 3557 MB | 3875 MB | inference, train | LoRA |
| [gemma-3-270m-it](https://huggingface.co/mobiletransformers/gemma-3-270m-it) | text-generation | 1814 MB | 2131 MB | inference, train | **MARS** |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/mobiletransformers/Qwen2.5-0.5B-Instruct) | text-generation | 2554 MB | 3212 MB | inference, train, rag | LoRA |
| [all-MiniLM-L6-v2](https://huggingface.co/mobiletransformers/all-MiniLM-L6-v2) | text-classification | 94 MB | 214 MB | inference, train, rag | LoRA |
| [distilbert-sst2-english](https://huggingface.co/mobiletransformers/distilbert-sst2-english) | text-classification | 270 MB | 361 MB | inference, train | LoRA |

Sizes are **measured** off each pushed package's manifest — the sum of `fileSizes` — not estimated.
"inference" is the group a plain install downloads; asking for `train` or `rag` adds to it. The `rag`
group is ~91 MB on every decoder above, because it is the same all-MiniLM-L6-v2 embedder each time.

`mobiletransformers/functiongemma-270m-it` is public. The other five are **private**; making them
public is a deliberate separate step. Set `HF_TOKEN` to reach them (see
[`.env.example`](https://github.com/martinkorelic/mobiletransformers/blob/main/.env.example)).

## Which one to start with

**SmolLM2-135M-Instruct.** Smallest useful chat model here, fastest to pull, fastest to train, and
the model every parity check in this repo is measured against.

- **Chat quality, and the memory ceiling** → Qwen2.5-0.5B-Instruct. Roughly four times the size and
  noticeably more fluent; a few tokens per second on a mid-range phone, which is the honest cost.
- **Tool calls** → functiongemma-270m-it. Turns an instruction into a structured call that the app's
  allowlist validates before anything runs.
- **Classify** → distilbert-sst2-english. The only entry with *trained* labels
  (`NEGATIVE`/`POSITIVE`), so the Classify screen says something meaningful on the first tap.
- **MARS** → gemma-3-270m-it. Multi-Adapter Rank Sharing is this project's own method: layers share
  a down-projection instead of each carrying its own, so the trainable parameter count grows with
  rank rather than with depth — 279,936 parameters against a 268M backbone. It is the only entry that
  is not LoRA, and the reason the shelf has more than one Gemma-3 on it.
- **Retrieval alone, and the smallest training run** → all-MiniLM-L6-v2. An encoder, so the drawer
  hides Chat for it; see the note below.

## Two things about the encoders that read as bugs and are not

**all-MiniLM-L6-v2 is exported as `text-classification`, not `feature-extraction`.** That is what
makes an encoder trainable at all: `TaskSpec.default_stages` emits a training stage exactly when the
task is trainable, and `FEATURE_EXTRACTION` is declared `trainable=False`, so exporting it the
"natural" way yields an inference-only package. Task auto-selection never picks `text-classification`
— it has to be named.

**So its classification head is randomly initialised, and its labels are `LABEL_0`/`LABEL_1`.** The
head does not exist in a sentence-encoder checkpoint; it is precisely the part fine-tuning learns. The
pretrained *backbone* is what must survive, and that is what the export-time parameter budget checks.
Because the labels are meaningless, `supportsClassification` is false and the app hides Classify for
this package. That is correct and self-consistent — DistilBERT is the entry to use when you want a
classifier that already works.

## Reproducing the shelf

```bash
make publish-catalog                          # export + gate-check + push all six
ONLY=smollm2 PUSH=0 scripts/publish_catalog.sh    # one entry, no upload
KEEP=1 scripts/publish_catalog.sh                 # skip re-export where a package already exists
```

Needs `HF_TOKEN_ORG` in `.env` — a token with `repo.write` on the target org. A fine-grained personal
token scoped to one repo returns `RepositoryNotFoundError` for every other, and the Hub returns that
identically for "does not exist" and "you cannot see it", so a permissions problem reads as a typo.
See [`.env.example`](https://github.com/martinkorelic/mobiletransformers/blob/main/.env.example).

The script keeps the per-model task and engine flags, which are not obvious and fail late when wrong:

- `--task text-classification` is what makes an encoder trainable (above).
- `--genai` is **decoder-only**, and off for Gemma-3 even though it is a decoder: Gemma-3 exports
  through optimum rather than the GenAI builder, so the package declares native only. A
  classification or feature-extraction graph has no KV cache at all, and the export refuses to write
  a `genai_config.json` describing a cache the graph does not have.
- `--peft` is a property of the package, not a runtime choice: the topology is baked into the
  training graph, so a device can only select what the export built.

The app's copy of this table lives in
`MobileTransformersApp/src/main/assets/model_catalog.json`. Adding a model there is editing JSON — no
code. Keep the two in step: the app claims sizes and features per entry, and a catalog that disagrees
with what was pushed is worse than no catalog.

## Measured on device

Not yet recorded for this release. Tokens/second per package needs a real device run; see
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md). Numbers are deliberately absent rather than estimated
— a throughput figure nobody measured is the kind of claim this project's gates exist to prevent.

## Related

- [SHOWCASE.md](SHOWCASE.md) — a tour of the app, and which package each capability needs
- [HUB_PACKAGE_FORMAT.md](HUB_PACKAGE_FORMAT.md) — what is actually in one of these repos
- [EXPORT.md](EXPORT.md) — producing your own
