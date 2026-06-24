# Memory-Mapping Experiments — Copied-Buffer → File-Backed Weight Handoff

**Priority #11 | Prerequisites: #8 (`01_code_plans/01_unified_merger_and_external_data_export.md`) | Blocks: nothing (non-blocking optimization of #8–#10); feeds Gate 0.2**

## Purpose

The v1 release claim is deliberately bounded: **"no on-device graph rewrite + bounded-copy weight handoff"** — *not* "true zero-copy memory mapping". This plan is the measurement harness that decides whether mmap is worth promoting from "nice optimization" to a v1 requirement. Everything here is gated on RSS/heap numbers, never asserted by faith.

Today the handoff **copies**. `InferenceSessionCache::setSessionOptions` (`session_cache.h:662-709`) builds a `WeightSessionCache`, which loads each `.tensor` file through `OrtValueSerializer::tensorproto_to_ortvalue_with_allocator` (`weight_serializer.cpp:313`). That path `allocator_.Alloc(data_size)` + `std::memcpy` per tensor (e.g. `weight_serializer.cpp:337-346` for FLOAT), then `Ort::Value::CreateTensor` wraps the **allocator-owned** buffer, and the values are handed to `options.AddExternalInitializers(initializer_names, initializer_values)` (`session_cache.h:702`). The frozen quantized base, by contrast, is loaded by ORT itself from `model.onnx` + its `.data` external file via the file-path `Ort::Session` constructor (`session_cache.h:436-438`) — that path *can* mmap, but is currently dominated by the copied per-tensor trainable handoff and the explicit config below.

One concrete current setting to experiment against: `session_cache.h:717` sets `session.use_ort_model_bytes_for_initializers` to `"0"`. The same line exists in the training session at `session_cache.h:940`. Toggling this is Experiment (c)/(d) below.

These experiments rank from lowest-risk/highest-leverage (frozen base, which is most of the bytes) to most speculative (GenAI passthrough).

## Touched / new files

Native (Android C++):
- `android/.../cpp/session_cache.h` — `InferenceSessionCache::setSessionOptions` and `WeightSessionCache` are the change sites.
- `android/.../cpp/weight_serializer.cpp` / `.h` — add a file-backed (`mmap`) tensor loader alongside `tensorproto_to_ortvalue_with_allocator`.
- NEW `android/.../cpp/mmap_tensor.h` — RAII wrapper over `open`/`fstat`/`mmap`/`munmap` whose lifetime is owned by the session cache (the mapped region MUST outlive the `Ort::Session`).
- NEW `android/.../cpp/mem_probe.h` — reads `/proc/self/statm` / `/proc/self/status` `VmRSS`; helper `log_rss(const char* tag)` for the four measurement points.

Desktop (Python, for the base-blob / GenAI-format experiments):
- NEW `spikes/mmap/measure_rss.py` — `psutil`-based RSS sampler shared with the File #9 spike harness.
- NEW `spikes/mmap/base_blob_mmap_spike.py` — desktop ORT load of the File #8 package toggling the ORT-format / external-initializer config keys.

Inputs (from File #8): the unified `inference/` package — `model.onnx` (external refs), single immutable base blob, per-tensor `<name>.bin` trainable externals, `weight_handoff_map.json`.

## Data contracts / interfaces

- **Measurement points (every experiment uses the same four):** (1) before session/model load, (2) after weight load (after `Ort::Session` ctor returns / after `WeightSessionCache::init`), (3) after first generated token, (4) after `clearWeights()` / cache release. Record **process RSS** (authoritative on Android) and, where available, allocator heap.
- **Pass/fail framing:** an experiment **passes** only if peak RSS at point (2)/(3) drops by a pre-registered margin (suggest ≥15% of the trainable+base byte size, ratified at Gate 0.2) **with byte-identical generated output** vs the current copied-buffer baseline. No correctness regression is the hard gate; memory is the win condition.
- **Baseline = current code unchanged**: copied buffers + `AddExternalInitializers` + `use_ort_model_bytes_for_initializers=0`. Capture its four-point RSS first; all experiments are reported as deltas against it.
- **mmap lifetime contract:** any `Ort::Value::CreateTensor` over a file-backed pointer requires the mapping to live until the session is destroyed. `mmap_tensor.h` instances are owned by the cache struct (parallel to today's `allocated_buffers` map in `session_cache.h:45`), freed in a `clearMappings()` that runs in the destructor, **not** eagerly after load (unlike copied buffers which `clearWeights()` frees at `session_cache.h:443`).
- **Real ORT config keys used** (all are documented session-config keys): `session.model_external_initializers_file_folder_path`, `session.use_ort_model_bytes_for_initializers`, `session.use_ort_model_bytes_directly`, and (ORT-format models only) `session.use_memory_mapped_ort_model`.

### Example — current copied-buffer path vs file-backed mmap path (illustrative sketch, not full impl)

```cpp
// CURRENT (session_cache.h:662-709 + weight_serializer.cpp:337): copy then inject
buffer_ptr = allocator_.Alloc(data_size);                  // owned heap copy
std::memcpy(buffer_ptr, raw_data.data(), raw_data.size()); // <-- the copy we want to remove
Ort::Value v = Ort::Value::CreateTensor(memory_info_, buffer_ptr, data_size,
                                        shape.data(), shape.size(), elem_type);
options.AddExternalInitializers(names, values);            // freed early in clearWeights()

// EXPERIMENT (c): map the per-tensor .bin, wrap the mapped pointer, no memcpy
struct MmapHandle {                                        // RAII; lives until session dtor
  void* addr; size_t len;
  explicit MmapHandle(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY); struct stat st; ::fstat(fd, &st);
    len  = st.st_size;
    addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0); // page-faulted in, not copied
    ::close(fd);
  }
  ~MmapHandle() { ::munmap(addr, len); }                   // NOT freed in clearWeights()
};
auto h = std::make_unique<MmapHandle>(bin_path);
Ort::Value v = Ort::Value::CreateTensor(memory_info_, h->addr, h->len,
                                        shape.data(), shape.size(), elem_type);
mapped_buffers[name] = std::move(h);                       // ownership tied to session lifetime
```

### Example — RSS measurement harness skeleton (`mem_probe.h`)

```cpp
// Read VmRSS from /proc/self/status; call at the 4 fixed points.
static long read_rss_kb() {
  std::ifstream f("/proc/self/status"); std::string k; long kb = 0;
  while (f >> k) { if (k == "VmRSS:") { f >> kb; break; } }
  return kb;
}
#define LOG_RSS(tag) LOGI("RSS[%s] = %ld kB", tag, read_rss_kb())
// usage: LOG_RSS("pre_load"); ...ctor...; LOG_RSS("post_weight_load");
//        ...first token...;   LOG_RSS("post_first_token"); clearWeights(); LOG_RSS("post_release");
```

### Example — `genai_config.json` config_entries toggling the byte-reuse keys (Experiment d)

```json
{ "model": { "decoder": { "session_options": {
  "log_id": "mt-mmap-probe",
  "config_entries": [
    ["session.use_ort_model_bytes_for_initializers", "1"],
    ["session.use_ort_model_bytes_directly", "1"],
    ["session.model_external_initializers_file_folder_path", "<inference dir>"]
  ]
} } } }
```

## Implementation steps — ranked experiments

### Experiment (a) — Frozen-base immutable-blob mmap via external-data file-path load  *(highest leverage; the base is most of the bytes)*

- **Hypothesis:** loading the frozen quantized base as a single external `.data` blob via the file-path `Ort::Session` ctor lets ORT mmap it (read-only, shareable, evictable), so RSS at point (2) tracks page-faulted-in pages, not a full copy. The base is immutable, so this is the safe, big win.
- **Change:** ensure File #8 emits the base as one external-data file referenced by relative location in `model.onnx` (already the layout). Add `options.AddConfigEntry("session.model_external_initializers_file_folder_path", inference_model_path)` so ORT resolves and (potentially) maps externals from the package dir. Do **not** add the trainable per-tensor values via `AddExternalInitializers` for this experiment in isolation — measure base-only first.
- **Measure:** RSS at the four points; compare base-blob-only load with the key set vs unset. mmap ⇒ RSS-after-load ≈ resident working set << file size; copy ⇒ RSS ≈ file size.
- **Pass/fail:** pass if RSS at (2) is materially below base-file size with identical logits.

### Experiment (b) — Per-tensor trainable external files: does ORT mmap or copy them?

- **Hypothesis:** if the trainable tensors are *also* left as external initializers resolved from the folder (via `session.model_external_initializers_file_folder_path`) instead of injected through `AddExternalInitializers`, ORT may mmap them from their per-tensor files — eliminating the `WeightSessionCache` copy entirely.
- **Change:** in `setSessionOptions`, behind a `handoff_mode == file_folder` flag, **skip** the `WeightSessionCache`/`AddExternalInitializers` block (`session_cache.h:662-709`) and instead rely on the folder-path config key, with the per-tensor `.bin`/`.data` names matching the inference initializer names from `weight_handoff_map.json`.
- **Measure:** four-point RSS vs the `AddExternalInitializers` baseline. Critically determine **whether ORT mmaps these or silently copies** — compare RSS-after-load to total trainable byte size.
- **Pass/fail:** pass if folder-path externals load with lower RSS and identical output. If ORT copies them anyway (RSS unchanged), record as "ORT copies external initializers on this build" — that result steers us back to (c).

### Experiment (c) — `AddExternalInitializers` copied-buffer vs file-backed `Ort::Value::CreateTensor`

- **Hypothesis:** the per-tensor copy in `tensorproto_to_ortvalue_with_allocator` can be replaced by mapping the tensor payload file and wrapping the mapped pointer with `Ort::Value::CreateTensor(memory_info, mapped_ptr, size, shape, ...)`, then still calling `AddExternalInitializers`. This keeps the explicit-injection path (which guarantees the trained tensor lands in the right slot) but removes the `memcpy`.
- **Change:** add `OrtValueSerializer::tensorproto_to_ortvalue_mmap(...)` returning `{Ort::Value, MmapHandle}`; it requires the `.tensor` payload to be a contiguous raw blob (the `.bin` form from File #8, not protobuf-wrapped — note today's `.tensor` files are serialized `onnx::TensorProto`, so this experiment depends on File #8's raw per-tensor `.bin` layout). Store the `MmapHandle` in a new `mapped_buffers` map parallel to `allocated_buffers` (`session_cache.h:45`); **do not** free it in `clearWeights()` — move its release into the destructor. Also toggle `session.use_ort_model_bytes_for_initializers` (`session_cache.h:717`) from `"0"` to `"1"` as a sub-variant and measure both.
- **Measure:** four-point RSS for: (i) current copy baseline, (ii) mmap + `AddExternalInitializers`, (iii) each with the `use_ort_model_bytes_for_initializers` toggle.
- **Pass/fail:** pass if the mmap variant reduces RSS at (2)/(3) with identical output and no lifetime crash (segfault on use-after-unmap = hard fail; proves lifetime wiring).

### Experiment (d) — GenAI passthrough of `use_ort_model_bytes_*` and (ORT-format) `use_memory_mapped_ort_model`

- **Hypothesis:** the GenAI engine (File #9/#10) honors the same ORT config keys via `genai_config.json` `model.decoder.session_options.config_entries`, so the base-blob mmap win from (a) transfers to GenAI for free; and for an **ORT-format** package, `session.use_memory_mapped_ort_model=1` maps the immutable model bytes.
- **Change:** have File #8's `genai_config.json` emit `config_entries` including `["session.use_ort_model_bytes_for_initializers","1"]`, `["session.use_ort_model_bytes_directly","1"]`, `["session.model_external_initializers_file_folder_path","<dir>"]`, and (only when the package is exported in ORT-format) `["session.use_memory_mapped_ort_model","1"]`. Confirm pass-through with a recognizable `log_id` (reuses the File #9 config-pass-through check).
- **Measure:** GenAI four-point RSS vs Native (a)/(c) over the **same** package; this is the cross-engine memory comparison feeding Gate 0.1/0.2.
- **Pass/fail:** pass if GenAI accepts the keys (no rejection in ORT logs) AND base-blob RSS matches Native's mmap result. If GenAI ignores the ORT-format key on the bundled build, record "key ignored on this genai/ort build" — expected risk per Tier 0.

## Interactions

- **File #8** owns the package layout these experiments read; Experiment (c) specifically needs File #8's raw per-tensor `.bin` (not protobuf `.tensor`) for a zero-copy map, and the immutable single base blob for (a). Coordinate the layout there.
- **File #9 / #10** (GenAI spike + engine abstraction): Experiment (d) shares the RSS harness (`spikes/mmap/measure_rss.py` ≈ `spikes/genai_external_swap/measure_rss.py`) and the config-pass-through check. The memory numbers from (a),(c),(d) are direct inputs to the Gate 0.1 cross-engine decision.
- **Gate 0.2** (`01_tier0_foundation_decisions.md`): the documented supported handoff path plus its memory smoke is the deliverable; these experiments produce the numbers that let the gate ratify mmap as v1-required or v1-optional.
- The `clearWeights()` early-free at `session_cache.h:443` is **incompatible** with mmap lifetime — any mmap experiment must move release to the destructor or it will fault on first run.

## Tests & smokes

- **Baseline RSS smoke:** run current code on the tiny File #8 package; record four-point RSS as the reference numbers. (No code change — just the probe wired in.)
- **Correctness invariant (all experiments):** logits of the first generated token must be byte-identical to the copied-buffer baseline for the *same* weights; any mmap variant that changes output is a hard fail (means the bytes/shape/dtype mapping is wrong).
- **Lifetime smoke:** run generation, then a second generation on the same session, with mmap-backed values; a use-after-unmap segfault fails the lifetime wiring.
- **Toggle smoke:** flip `session.use_ort_model_bytes_for_initializers` `0`↔`1` (`session_cache.h:717`) and confirm load still succeeds and output is identical; record RSS delta.
- **Config pass-through smoke (d):** set a recognizable `log_id` + the mmap keys in `genai_config.json`, enable ORT logging, confirm entries reach ORT and are not rejected.
- **Memory comparison report:** single table — copied-buffer baseline | (a) base mmap | (b) folder externals | (c) mmap injection | (d) GenAI passthrough — peak RSS at points (2)/(3), with the pass/fail verdict per the ratified margin. This table is the Gate 0.2 artifact.
