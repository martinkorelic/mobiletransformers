# Changelog

All notable changes to MobileTransformers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows Semantic Versioning from
v1.0.0 onward.

## [Unreleased]

## [0.2.0] — 2026-08-17

The showcase release: a published model shelf, a provisionable clone, and the encoder training path
finished.

### Added
- **A published model catalog.** Five packages under
  [`mobiletransformers`](https://huggingface.co/mobiletransformers), each shipping **both** an
  inference and a training stage — asserted by `scripts/publish_catalog.sh`, because a shelf entry
  that cannot be fine-tuned demonstrates half the framework. The app's
  `assets/model_catalog.json` now carries **measured** sizes taken from each pushed manifest, the
  `peft` method per entry, and real repo ids (it still named `Qwen2-0.5B`, which does not exist).
  See [docs/CATALOG.md](docs/CATALOG.md).
- **A MARS package on the shelf.** `mobiletransformers/gemma-3-270m-it`, exported with Multi-Adapter
  Rank Sharing — this project's own method, and previously demonstrated by nothing that was published.
  Verified structurally, not by its label: `shared_A` and the intermediate are shared across q_proj
  and v_proj (`adapter_index` 0/1), where LoRA has zero tensor reuse. `scripts/publish_catalog.sh`
  gained a PEFT column and now **asserts the exported method matches the one requested**, so a
  silently-ignored `--peft` cannot publish a LoRA package under a MARS label.
- **Model cards carry the banner, the framework repository and the citation.** A published package is
  not loadable by `transformers`, `optimum` or plain `onnxruntime`, and the card previously stated no
  way to run it at all. The header image is uploaded into each model repo rather than hot-linked, so
  it does not depend on the framework repository's visibility or default branch.
- **`make doctor`** — one preflight report naming every missing prerequisite and the command that
  fixes it: uv, Python 3.10/3.12, the current venv profile, the ORT-training wheel, JAVA_HOME, the
  Android SDK, adb, the vendored natives, and the `.env` tokens.
- **`make fetch-native-deps`** and `third_party/android/manifest.json` — the ~180 MB of gitignored
  Android native binaries described as data (destination, size, **sha256**, provenance) and fetched
  by a script that verifies the archive hash, unpacks, then verifies every file individually. It
  refuses rather than half-populating: a partly-filled `jniLibs/` fails the link naming a missing
  *symbol*, not a missing file.
- **`.env.example`**, committed, documenting which operations need which token.
- **`docs/SHOWCASE.md`** — a tour of the sample app, one section per capability, naming the package
  each needs and what you should see.
- **`docs/CATALOG.md`** — the published packages, and why the encoders are exported as
  `text-classification`.
- **The app has its own identity.** The launcher icon was still Android Studio's stock green robot.
  It is now an adaptive icon built from `docs/assets/mobiletransformers_logo.png` — foreground scaled
  into the 66dp safe zone, a dark `#171E22` background taken from the logo's own outline (the mark has
  a white sticker outline that vanishes on light), a line-art `monochrome` layer for Android 13+
  themed icons, and legacy pre-API-26 icons at five densities. The same mark sits in the top app bar.
  `docs/assets/README.md` records the four rules the artwork was cut with — the alpha noise floor, the
  safe zone, the background colour and the line-art monochrome — so it can be redone if the logo
  changes.
- **`docs/ARCHITECTURE.md` ▸ Native dependencies** — the section it never had: what a clone does not
  bring, why two ONNX Runtimes ship side by side, and why arm64-v8a only.

### Fixed
- **DistilBERT could not produce a training graph, and neither could four other architectures.**
  `OnnxSequenceClassificationTrainerWrapper` declares `token_type_ids`; `DistilBertOnnxConfig` does
  not — DistilBERT dropped BERT's next-sentence objective and with it the segment embedding — and
  Optimum binds the dummy inputs **positionally**, so `labels` would land in the `token_type_ids`
  slot. This is the Gemma-3 `position_ids` defect in encoder form, and the fail-closed
  signature/inputs cross-check written for that one caught it before `torch.jit`.
  Fixed with a `token_type_ids`-free wrapper, plus a **registry-wide** test that runs the production
  cross-check against every trainable row rather than the one architecture someone happened to
  export. That test immediately found four more: `GemmaForCausalLM`, `Gemma2ForCausalLM` and
  `NemotronForCausalLM` also omit `position_ids` (this is not a Gemma-3 peculiarity), and
  `RobertaForSequenceClassification` also omits `token_type_ids` — only `BertForSequenceClassification`
  keeps the task's default wrapper. All five rows corrected.
- **A strip of Android Studio's template purple sat above the app bar.** The window theme still had
  `android:statusBarColor` -> `colorPrimaryVariant` -> `purple_700` (#3700B3), in both day and night.
  Same defect the Compose scheme already documents for `surfaceContainer`, one layer further out: a
  role nobody set, filled from a baseline palette. The status bar is now driven from the live Compose
  `colorScheme.surfaceContainer` — what `TopAppBar` paints itself with, so the two read as one surface
  — because there are four schemes here and one XML value cannot be right for all of them. The five
  unused `purple_*`/`teal_*` template colours were removed with it.
- **A download waiting for Wi-Fi was displayed as an active download, forever.** Pulls default to
  Wi-Fi only; with no Wi-Fi, WorkManager parks the worker and it waits, which is deliberate.
  `ModelHolder` mapped that state to the sentence `"waiting for Wi-Fi"` and `downloadPhaseLabel` threw
  it away — it matched `Resolving`/`Verifying`/`Installing` and sent everything else to
  `"Downloading"`. Two individually-correct halves with an untested seam; found on a real phone, not
  by any suite. The state is now a boolean on `DownloadUi` rather than a magic string, the card
  explains that the pull is queued rather than failed, and it offers **Download on mobile data** —
  because the switch governing this lives inside the Advanced disclosure that someone installing from
  a catalog card has never opened. `DownloadPhaseLabelTest` pins it, verified to fail against the old
  logic.
- **MARS packages declared the wrong weight orientation.** `derive_transpose_policy` read only
  `adapter_A`; MARS names its down-projection `shared_A` because it shares one across a block, so
  every MARS layer took the "nothing to observe" branch and declared `no_transpose` for shapes that
  decide the question unambiguously ([640,1024] on disk against a [1024,640] delta). This is the
  transpose defect that once corrupted every merged weight, re-introduced through a naming difference
  rather than an unassigned field — and it was caught by
  `test_the_derivation_agrees_with_a_real_exported_package`, which reads a real artifact for exactly
  this reason, on the first MARS package ever exported. The derivation now understands both names and
  **raises** on a third rather than defaulting. The published package was re-exported.
- **The plan-identifier guard was structurally blind to config files.** Its suffix filter listed no
  `.toml`, `.yml`, `.yaml`, `.json` or `.properties`, so `pyproject.toml` and all three CI workflows
  were never scanned — the guard read as if it covered the repo and did not, which is worse than no
  guard. Found by eye on a tree the test had just passed. Filter widened, scope extended to the repo
  root, `.github/`, `config/` and `examples/`, and 31 further sites cleaned.
- **`publish_catalog.sh` sourced `.env` only when pushing.** The export needs the token too — a gated
  base model 401s on its first config read — so `PUSH=0`, the mode used to test an export, was
  precisely the case with no credentials.
- **A failed task lookup named neither the model nor the reason.** `discover_tasks` is fail-open: it
  captures the real cause in `blocker` and returns no tasks, and the caller discarded it to report
  `no supported task in preference order (...) for []`, which reads as "unsupported architecture"
  when the cause was a missing token. It now surfaces the blocker.
- **The publish script leaked its Hub token into `ps`.** It passed `--token` on the command line,
  which is world-readable in `/proc/<pid>/cmdline` for the life of the process. It now goes through
  the environment, resolved by the same `config.settings` path.
- **`JAVA_HOME` was hardcoded to a Linux Android Studio path** in the Makefile and four scripts, as
  the *only* fallback. On macOS, on a CI runner, or under any standalone JDK it resolved to a
  directory that is not there, and Gradle then reported a Java-version error naming neither
  `JAVA_HOME` nor the file that set it. All five now share `scripts/lib/java_home.sh`, which honours
  an explicit `JAVA_HOME`, then probes PATH for a JDK 17+, then falls back — and a new guard stops
  the literal spreading back out.
- The sample app's `versionName` was the literal `"1.0"`, matching no other version site in the repo.
  It now derives from the root `version` property.

### Changed
- `agent_docs/` is no longer tracked. It is implementation plans and per-cycle handoff notes —
  working material, not user documentation. The directory stays on disk.
- `README.md` rewritten around what the project actually does end to end, with the measured catalog
  table and links to the new pages. It still described a six-tab app the drawer replaced.

### Known issues
- **The native-dependency bundles are built but not hosted.** `third_party/android/manifest.json`
  carries `baseUrl: null`, so `make fetch-native-deps` works only against a local `file://` mirror
  until they are published. Owner action.
- **Nothing in this release has been exercised on a device.** The host gates are green; that has
  repeatedly proven nothing about the phone. See `agent_docs/MANUAL_DEVICE_CHECKS.md`.
- The licence remains CC-BY-NC-4.0, which contradicts the consumable-AAR goal. Unchanged and still
  the only real v1.0 blocker; it is a rights-holders decision.

### Fixed — release blockers found 2026-08-15
- **`android.permission.INTERNET` was declared nowhere.** The entire Hub-download stack —
  resolver, planner, streaming downloader with Range-resume and sha256 verify-and-retry, WorkManager
  worker, installer — was complete and JVM-tested, and **could not run on any device**: the first real
  GET throws `SecurityException`. MockWebServer runs on the JVM against localhost, where no Android
  permission model applies, so no existing test could see it. Now declared in the **library** manifest
  (so consumers inherit it) and pinned by a guard that was verified to fail first.
- **Every Hub pull leaked a full copy of the package.** `HubDownloader` never deleted its `.download/`
  staging tree, and the installer copied rather than moved into `.staging/`. A pull cost ~3× the
  package in transient disk and left ~1× behind permanently — for a 3.87 GB package, ~11.6 GB free
  needed instead of ~3.9 GB.
- **Gemma-3 could not produce a training graph**, and the cause was not the dependency pin it had been
  attributed to. `Gemma3TextOnnxConfig` declares `[input_ids, attention_mask]` where `LlamaOnnxConfig`
  declares those plus `position_ids`; Optimum passes dummy inputs **positionally**, so the decoder
  wrapper bound `labels` to `position_ids` and died inside `torch.jit`. Fixed with
  `OnnxDecoderNoPositionIdsTrainerWrapper`, a per-architecture `trainer_wrapper_class` override, and a
  fail-closed signature/inputs cross-check so this class of bug names itself.
- **The train/inference parity probe was measuring nothing meaningful off the Llama family.** Its token
  ids were Llama-family, so a Gemma-vocabulary model scored ~25 nats against a 12.48-nat uniform floor
  and a sound package passed its own integrity gate by 0.06 nats. The probe now tokenizes real English
  with the package's own tokenizer, resolved once and shared by both legs.
- `package-model` could not repair a manifest missing its inference provenance, which is exactly the
  state a two-profile train-capable export leaves behind; the model card had no Hub frontmatter, so a
  published page showed no licence and no link to its base model.

### Changed
- **`ort-training-local` now declares `transformers>=4.50,<4.58`** (was `==4.46.2`). That pin was
  believed load-bearing; three controls under 4.57.6 say otherwise — SmolLM2-135M at parity delta
  0.0111 nats (identical to its 4.46.2 figure), `all-MiniLM-L6-v2` at 73,728 trainable parameters
  (unchanged), and Gemma-3 working at all. `torch`, `peft`, `numpy` and `onnx` stay exactly pinned:
  those are the genuine ABI couplings. `transformers` appears in the wheel's `paired_stack` as a record
  of its build environment, not as a constraint — and treating the two as the same thing is what kept
  on-device FunctionGemma training blocked.
- `push` no longer creates the target repo unless `--create` is passed, so a mistyped repo id fails
  instead of silently creating one; `push` and `pull` both take `--token`.

### Added
- **Repository restructure complete.** All Python now lives in `src/mobiletransformers/`; the seven
  legacy roots (`trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`,
  `evaluation/`) are gone, along with their deprecation shims. The built wheel is self-contained,
  verified by installing it into a clean venv and importing from outside the checkout.
- Architecture registry covers every supported model (16 rows): adding one is a data row, not an
  `elif`. Branch side effects (forced execution provider/precision, `exclude_embeds`, `hidden_act`)
  are fields on the row.
- `export/quantizer_compat.py`: resolves ONNX Runtime's weight-only MatMul quantizer across the
  `MatMul4BitsQuantizer` → `MatMulNBitsQuantizer` rename, so the inference builder works on either.
- Export-time merge-contract check: every `trainingBaseLayerName` in `weight_handoff_map.json` must
  name a real checkpoint parameter, or the export fails.
- **Export-time numeric gates on the training stage.** `artifacts/parameter_budget.py` counts the
  training graph's parameters **per dtype** against the source model's own count (recorded at export
  time) and fails closed on a shortfall or on a graph with no trainable float parameters;
  `artifacts/train_inference_parity.py` runs identical tokens through the package's inference and
  training graphs and bounds the cross-entropy gap. Until now every check on the training stage was
  byte-level or structural, so a package carrying a fraction of its model would have passed them all.
- `inferenceGraphPrecision` in `inference/optimum_config.json`, **measured from the shipped graph**.
  A variant id names the training-side quantization (`--quant`), which the inference export does not
  apply — so `cpu-int4` can legitimately hold an fp32 inference graph. Precision is now declared
  rather than inferred from a directory name.
- Task registry (`config/registry/task.py`): the auto-model class, KV-cache kwargs, PEFT `task_type`,
  label shape, quantization exclusions and training-wrapper class are data per `TaskType`, removing
  the last task-shaped `if/elif` chain from the export path. Adding a training objective is a row.
- **Encoder fine-tuning, host legs complete.** `TaskType.SEQUENCE_CLASSIFICATION` plus registry
  rows for BERT/RoBERTa/DistilBERT classification: export → training artifacts → train step → metric
  works on a real encoder (loss −21.8%, accuracy 0.25 → 1.00). The Android smoke remains device-gated.
- PEFT target modules now come from the architecture registry per model, with `--peft-target` (CLI)
  and `peft_target:` (export YAML) to override. **Decoder exports now adapt `q_proj`/`v_proj`** (the
  LoRA convention) rather than `q_proj`/`k_proj`; pass `--peft-target q_proj,k_proj` for the old
  behaviour.
- Device acceptance suite (13 instrumented tests) covering load→generate, conversation reset,
  dual-engine parity (first token **and** ordered callback sequence), RAG, ObjectBox ranking, a
  four-point RSS table per engine, and train→merge→generate.
- One-command export CLI + Hub package format (manifest-first cache bridge), with `--config` (YAML
  supplying any unset flag, CLI > YAML > default) and `--validate`.
- Hub pull/install + adapter push-back (Python), background download via WorkManager (Android).
- Kotlin facade: `MobileTransformers.fromPretrained` → `MobileTransformerModel`, plus `TrainingJob`
  (status/event flows, cooperative cancel, checkpoint/resume).
- Dual inference engines (`NATIVE`, `GENAI`) behind one `ModelRuntime` over a single package.
- On-device RAG: `VectorStore` boundary, ingestion/chunking, grounded generation.
- Federated adapter record codec + FedAvg aggregation (`mobiletransformers federated simulate`).
- Per-role `tensorDtypes`/`tensorShapes` in `weight_handoff_map.json`, so the device can load packed
  quantized tensors from headerless external data.
- Test harnesses for all three languages: Robolectric (Kotlin), googletest (C++), and `make guard`
  ratchets for credential reads and registry-dispatch literals.
- `maven-publish` for the Android SDK
  (`com.martinkorelic.mobiletransformers:mobiletransformers-android`).
- Docs: `ARCHITECTURE.md`, `HUB_PACKAGE_FORMAT.md`, `ANDROID_CACHE_FORMAT.md`, `FEDERATED.md`,
  `MODEL_FORMAT.md`, `CONFIGURATION.md`, `PUBLIC_API.md`, `EXPORT.md`, `RAG.md`, and a generated
  `COMPATIBILITY_MATRIX.md`.

### Fixed
- GenAI never loaded on any package the training stage produced: `genai_config.json` carried a
  `session_options.config_entries` key that onnxruntime-genai 0.14 rejects outright, taking the whole
  config with it. The runtime then fell back to Native **silently**, so the dual-engine parity test
  compared Native with Native and passed. Unsupported keys are now stripped at export, and an
  explicitly requested engine that cannot load raises instead of substituting another.
- A generation session that failed to construct was logged and forgotten, leaving `generate` to return
  nothing with no error. The cause is now retained and re-raised.
- One shared C++ layer-name normalizer (`cpp/layer_name.h`) replaces nine open-coded prefix rewrites
  that had produced five device-only merge defects.
- `make_mlp_unpacked_lora` referenced unbound `q_proj`/`k_proj` instead of the `gate_proj`/`up_proj` it
  builds — a latent `NameError` on any unpacked-MLP LoRA export.
- PEFT's `task_type` was hardcoded to `"CAUSAL_LM"` at both LoRA call sites, which mis-configures any
  non-decoder model: PEFT uses it to decide which modules to wrap and which head stays trainable. It
  now comes from the task registry.
- The training-graph wrapper's forward signature was decoder-only (`position_ids`), so a BERT-family
  export failed inside optimum at `check_dummy_inputs_are_allowed`. Encoders get their own wrapper
  (`token_type_ids`); because the signature's parameter names become the exported ONNX input names,
  the decoder's are deliberately unchanged.
- `onnx_checktrain` pre-shifted its labels even though the exported graph already applies the HF causal
  shift, inflating every loss it printed and making it incomparable to the on-device number. The
  Android path was always correct.
- Dynamic quantization could place a **quantized activation on the gradient path**, which has no
  gradient at all (quantized *weights* are frozen and dequantize to float, so they were never the
  problem). ONNX Runtime rewrites `Gemm` → `MatMul` before quantizing and matches `nodes_to_exclude`
  against the rewritten name, so excluding such a node by its own name silently did nothing.
- `LayerNormalization` was exported with only its `Y` output, but ORT's gradient reads the optional
  saved mean / inverse-std outputs — so no gradient graph could be built through it. Decoders were
  unaffected (RMSNorm exports as `SimplifiedLayerNormalization`, which already carries them).
- Gemma2/Gemma3 were bound to the generic `GemmaOnnxConfig`; they now bind their own.
- `emit_merger_models` was missing from its module's `__all__`, so the declared public surface
  disagreed with the real one.
- Re-exporting into an existing directory silently corrupted the package: `onnx` *appends* external
  data, so a second export doubled every trainable tensor and the device rejected the sizes.
- GenAI was unreachable end to end: the config mapper never set the engine field the runtime factory
  reads, and the repository dispatched on a string that dropped every GenAI config — leaving `generate`
  suspended forever.
- Merged weights are written as raw external data but were parsed as `TensorProto`, so merged-weight
  load failed on the shipping (non-mmap) path.
- Merged-weight load failure, and a partial on-device merge, both reported success. Every gate on the
  train → merge → generate path now fails closed rather than silently serving base weights.
- Post-merge checksum contract: the `.bin.sha256` sidecar (refreshed by the merger) now takes
  precedence over the manifest-time digest, which a correct merge necessarily invalidates.
- `LinearLRScheduler.stateDict()`/`loadFromState()` were `TODO()`, crashing any run on the **default**
  schedule at its first checkpoint.
- A Mode-1 (PEFT) adapter push published `adapter_config.json` with no weights.
- RAG configuration was applied only once per session; later `topK`/`minScore`/`searchType` changes
  were silently ignored.
- Engine parity: identical sampling resolution, token count and throughput reporting across engines.
- Package installation is crash-safe (rename-aside → rename-in → delete-old); the previous
  delete-then-rename could destroy an installed model, including local training state.
- Credentials are read through `config.settings` instead of ad-hoc `os.environ[...]` reads.

### Known issues
- **Train and inference halves of one package are quantized differently, and the gap is unmeasured
  end to end.** A variant may ship a uint8 weight-quantized training graph beside an fp32 inference
  graph. The export now gates on parameter count and on a train-vs-inference loss delta, but the
  device-side numeric equivalence of the two halves is asserted only at export, not after a merge.
  *(This replaces an earlier "training starts from weights that are not the pretrained ones" entry,
  which was wrong: it counted uint8 checkpoint tensors as fp32 and concluded two thirds of the model
  was missing. The training graph carries all 135,436,915 parameters.)*
- **Memory-mapped weight loading covers only the trainable split** (~8% of weight bytes); the frozen
  base still loads through ORT's own external-data path, so whole-process peak RSS improves by ~6%.
  Within its own scope the zero-copy path realises 92.9% of the attainable saving. It is default-off
  (`debug.mtf.mmap_weights`) and does not block v1; extending it to `frozen_base.onnx.data` is tracked
  as a non-blocking follow-up.
- **arm64-v8a only.** No x86_64 build of ONNX Runtime/tokenizers exists here, so the library does not
  run on an x86_64 emulator.
- The project is CC-BY-NC-4.0, which is incompatible with distributing the AAR for commercial use.

### Non-goals
- **GPU/NPU training.** Inference may use an accelerated execution provider; training is CPU-only.
- **Multimodal training.** Text-generation and encoder tasks only.
- **Competing with server-side trainers on throughput.** The target is feasibility and privacy on a
  phone, not tokens/second parity with a datacentre.
- **On-device engine/facade device parity**, which remains gated on device acceptance runs.

## [0.1.0] — unreleased

Pre-release development line. The public API is not yet frozen; `mobiletransformers.__all__`, the CLI
surface and the Kotlin facade are version-locked at v1.0.0.
