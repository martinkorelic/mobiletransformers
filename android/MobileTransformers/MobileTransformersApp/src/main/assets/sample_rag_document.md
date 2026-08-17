# MobileTransformers — on-device retrieval notes

A short document with a few clearly distinct facts, so a retrieval result is easy to judge by eye.
Ask the Chat tab something answerable only from here (with grounding on) and the source cards should
show the matching paragraph.

## Packages

A MobileTransformers package is manifest-first. `mobiletransformers_manifest.json` is read before any
large file is fetched, and it names every file's SHA-256, so a download can be verified rather than
trusted. The manifest's `downloadPlan` splits a package into feature groups — `core`, `inference`,
`train`, `rag` and `genai` — and a client fetches only the groups it asked for.

## Weights

Trainable weights ship as ONNX external initializers, one file per tensor, alongside a single
immutable blob holding the frozen quantized base. On-device merging overwrites the per-tensor files
with an atomic rename and a checksum. There is no graph rewrite and no separate merged model.

## Engines

Two inference engines consume the same package: the native ONNX Runtime engine, which is the
guaranteed floor, and the ONNX Runtime GenAI engine, which is selectable when the package ships a
`genai_config.json` and the device probe succeeds. Asking for GenAI where it is unavailable fails
rather than silently falling back, because being given a different engine than the one named is a
wrong answer rather than a graceful degradation.

## Retrieval

Documents are chunked, embedded with the encoder shipped in the package's `rag` group, and stored in
an on-device vector database using cosine similarity. Retrieval never leaves the phone. The assembled
prompt is returned alongside the answer, because an app that cannot show what the model was actually
asked cannot debug a bad grounded answer.

## Training

Fine-tuning runs on the device against a LoRA adapter, so only a small fraction of the parameters
carry gradients. Cancelling is cooperative: the native loop stops at the next step boundary and writes
a checkpoint, which makes a cancelled run resumable rather than lost. Scheduled training runs in
charging-and-idle chunks, and each chunk re-evaluates its constraints before continuing.
