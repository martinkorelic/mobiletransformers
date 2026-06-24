# GenAI External-Data-Swap Spike

**Priority #9 | Prerequisites: #8 (`01_code_plans/01_unified_merger_and_external_data_export.md`) | Blocks: #10 (`03_inference_engine_abstraction_native_and_genai.md`); feeds Gate 0.1**

## Purpose

Validate finding **F2**: that ONNX Runtime GenAI can consume **on-device-trained, merged weights without any graph rewrite or genai fork**, simply by reading the *same* per-tensor external-data folder the Native engine reads — after the merge has overwritten the `.bin` files in place.

This is the decisive experiment for **Gate 0.1**. If it passes, GenAI becomes a selectable engine over the unified package (File #10). If it fails, we keep the manual loop and GenAI is dropped for v1.

The spike deliberately **avoids** `OgaCreateModelWithInitializers` (verified fork-only: it exists only in the vendored `ort_genai_c.h:130`, takes C++ types `std::unordered_map<std::string, OrtValue>` in a C header, and is NOT in upstream onnxruntime-genai). It also avoids `model_input`/`SetModelInput` for the primary test (that path disables prepacking/constant-folding and only feeds declared graph inputs). The primary mechanism is **stable `OgaCreateModel(<config_dir>)` + external-data file resolution from the package directory.**

Core hypothesis to prove: *overwriting the per-tensor external `.bin` files changes generated output, and `OgaCreateModel` picks up the new bytes, with memory within an accepted threshold of the Native engine.*

## Touched / new files

Desktop (Python, run first):
- NEW `spikes/genai_external_swap/desktop_spike.py` — export tiny model with per-tensor externals (reuse File #8's `export_inference_package.py`), do one train->merge (or simulate the merge by perturbing one `.bin`), run `onnxruntime_genai.Model(<dir>)`, generate one token, assert output differs from base.
- NEW `spikes/genai_external_swap/measure_rss.py` — RSS sampler (read `/proc/self/status` `VmRSS` on Linux / `psutil` cross-platform) snapshotting before load, after `OgaCreateModel`, after first token; compare mmap vs copy.
- Reuse `inference/generator_genai.py` only as a reference for the GenAI Python API shape (its `params.set_model_input` prototype at lines ~40-76 is the *fallback* path, NOT what we test here).

Android (JNI, run second):
- NEW `android/.../cpp/genai_spike.cpp` — minimal JNI: `OgaCreateModel(<inference dir>)`, tokenize, `GenerateNextToken` once, return token. This is the seed of the File #10 GenAI wrapper that replaces the commented-out `onnx-genai.cpp`.
- NEW `android/.../GenAISpikeTest.kt` — instrumented test driving the JNI spike on a packaged tiny model.
- Symbol-check script `spikes/genai_external_swap/check_symbols.sh` — `nm`/`readelf` over the linked GenAI `.so`.

Inputs (produced by File #8):
- A packaged `inference/` dir with `model.onnx` (external refs), `genai_config.json`, `weight_handoff_map.json`, `base/base_weights.onnx.data`, and per-tensor `<name>.bin`.

## Data contracts / interfaces

- Package = the exact File #8 layout. GenAI reads it via `OgaCreateModel("<...>/inference")`; the relative external-data locations recorded in `model.onnx` resolve against that dir.
- `genai_config.json` carries `model.decoder.session_options.config_entries` including `["session.model_external_initializers_file_folder_path", "<inference dir>"]` (belt-and-suspenders) and harmless probe entries (`use_env_allocators`, a custom `log_id`) to confirm pass-through.
- "Output differs" contract: capture the **logits of the first generated token** (or the greedy token id with fixed seed/temperature=0) for base vs swapped; assert they differ. Logits comparison is stronger than the token id (avoids ties).
- Memory threshold: define `ACCEPTED_RSS_DELTA` (e.g. GenAI peak RSS <= Native peak RSS + 15%, number to be ratified at Gate 0.1). Record absolute numbers regardless.

## Implementation steps

1. **Export tiny package** (SmolLM2-360M or smaller) via File #8 in `external_initializer` mode. Confirm `model.onnx` has zero inline trainable initializer data — all external — and base lives in the single base blob.

2. **Baseline generation (desktop)**: `Model(dir)` -> `GeneratorParams` -> fixed prompt, greedy (temp 0), generate **one token**; record logits vector `L_base` and token id. Snapshot RSS at: pre-load, post-`Model()`, post-first-token.

3. **One train -> merge**: run the real device-shaped merge offline (`artifact/merger.py` driver from File #8) on a checkpoint with a non-trivial delta, OR for the minimal spike, deterministically perturb exactly one trainable `.bin` (e.g. add a constant) using the handoff map's `external_file`, refresh its `.sha256`. Do NOT touch `base/`.

4. **Swapped generation (desktop)**: construct a **fresh** `Model(dir)` (GenAI caches at construction; reuse is not valid), regenerate one token; record `L_swap`. **Assert `L_swap != L_base`** beyond float tolerance. This proves the external swap is observed.

5. **Constant-folding check**: confirm the trainable tensors were NOT constant-folded into a producer node at export. Inspect `model.onnx` post-export: each `inference_initializer_name` from the handoff map must still be a live external initializer feeding a MatMul/MatMulNBits, not folded away. If GenAI/ORT folds them, the swap silently no-ops — this is a hard fail condition. (This is the "trainable externals aren't constant-folded" confirmation.)

6. **Config pass-through check**: set a recognizable `log_id` and `session.model_external_initializers_file_folder_path` in `genai_config.json`; enable ORT logging; confirm the entries reach ORT (appear in logs / no rejection). Confirms `session_options.config_entries` are honored by GenAI.

7. **Memory measurement (desktop)**: report RSS deltas mmap-vs-copy. Note: with file-path load ORT *can* mmap external initializers; record whether RSS after load is close to file size (mmap) or ~2x (copy).

8. **Symbol check**: run `nm -D --defined-only <libonnxruntime-genai.so> | grep -i CreateModelWithInitializers` and `readelf -Ws` equivalent on the **actual linked Android `.so`** (from the bundled `onnxruntime-genai.aar`, see `build.gradle.kts` `aarLibs/onnxruntime-genai.aar`, linked in `CMakeLists.txt:62`). **Assert the symbol is absent** — documents that `OgaCreateModelWithInitializers` is fork-only and we do not depend on it. Also confirm `OgaCreateModel`, `OgaGeneratorParamsSetModelInput` are present.

9. **Android port**: build `genai_spike.cpp` (`OgaCreateModel(<package dir in app storage>)` -> tokenize -> one token). Push the File #8 package into app-internal storage. Run baseline + swap on device exactly as steps 2-4. **Confirm GenAI Android resolves relative external data inside the package dir** (this is the key Android-specific unknown).

10. **Cross-engine equivalence (the Gate 0.1 core)**: take the **same** packaged folder; run Native (`ORTGeneratorNative`) and GenAI over it; assert both produce the correct token for the swapped weights, and GenAI peak RSS is within `ACCEPTED_RSS_DELTA` of Native.

11. **Write the Gate 0.1 result**: A/B decision (adopt GenAI as selectable engine | keep manual-only), with the measured numbers and the constant-folding/symbol findings.

## Example snippets (illustrative — NOT full implementations)

These show the *shape* of the spike code; real code reuses File #8's exporter and the handoff map.

**Desktop: overwrite one per-tensor `.bin`, then `og.Model(dir)` + generate** (`desktop_spike.py`)

```python
# example — illustrative only
import json, os, numpy as np, onnxruntime_genai as og

INF_DIR = "build/spike/inference"          # File #8 package (per-tensor externals)
hmap = json.load(open(f"{INF_DIR}/weight_handoff_map.json"))

def first_logits(prompt: str) -> np.ndarray:
    model = og.Model(INF_DIR)              # FRESH model — GenAI caches at construction
    tok = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(tok.encode(prompt)) + 1)
    gen = og.Generator(model, params)
    gen.append_tokens(tok.encode(prompt))
    gen.generate_next_token()
    return np.array(gen.get_output("logits"))[0, -1, :]

L_base = first_logits("Hello world")

# perturb exactly ONE trainable external (simulates a merge); never touch base/
t = hmap["tensors"][0]
path = os.path.join(INF_DIR, t["external_file"])
buf = bytearray(open(path, "rb").read())
arr = np.frombuffer(buf, dtype=np.uint8).copy(); arr[:64] ^= 0x01    # tiny deterministic delta
open(path, "wb").write(arr.tobytes())                                # atomic in real code: tmp+os.replace

L_swap = first_logits("Hello world")       # fresh Model() picks up new bytes
assert not np.allclose(L_base, L_swap), "external swap had NO effect — folded/copied?"
```

**`genai_config.json` — `session_options.config_entries` (belt-and-suspenders external folder)**

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "log_id": "mt-genai-spike",
        "use_env_allocators": true,
        "config_entries": [
          ["session.model_external_initializers_file_folder_path", "/data/.../<model>/inference"],
          ["session.qdq_matmulnbits_accuracy_level", "4"]
        ]
      }
    }
  }
}
```

**C++/JNI sketch: `OgaCreateModel(config_dir)` + RSS measurement points** (`genai_spike.cpp`)

```cpp
// example — illustrative only; error checks elided
static long rss_kb() {                         // VmRSS from /proc/self/status (Android/Linux)
    std::ifstream s("/proc/self/status"); std::string k; long v;
    while (s >> k) { if (k == "VmRSS:") { s >> v; return v; } }
    return -1;
}
extern "C" JNIEXPORT jstring JNICALL
Java_..._GenAISpike_runOneToken(JNIEnv* env, jobject, jstring jdir) {
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    long rss_pre = rss_kb();                    // (1) before load
    OgaModel* model = nullptr;
    OgaCreateModel(dir, &model);                // stable API; resolves relative externals in dir
    long rss_loaded = rss_kb();                 // (2) after OgaCreateModel  (mmap ~= file size; copy ~= 2x)
    OgaTokenizer* tok; OgaCreateTokenizer(model, &tok);
    OgaGeneratorParams* p; OgaCreateGeneratorParams(model, &p);
    // ... append prompt tokens, generate_next_token() once ...
    long rss_tok = rss_kb();                    // (3) after first token
    LOGI("RSS kB pre=%ld loaded=%ld firsttok=%ld", rss_pre, rss_loaded, rss_tok);
    // return decoded token string ...
}
```

**Symbol check: confirm `OgaCreateModelWithInitializers` is fork-only** (`check_symbols.sh`)

```sh
# example — run against the LINKED Android .so from onnxruntime-genai.aar
SO=jni/arm64-v8a/libonnxruntime-genai.so
nm -D --defined-only "$SO" | grep -i CreateModelWithInitializers   # EXPECT: no output (absent → fork-only)
readelf -Ws "$SO" | grep -i 'OgaCreateModel\b'                     # EXPECT: present (stable API)
```

## Interactions

- **File #8** produces the package; if step 5 finds folding, File #8's export must be fixed (keep trainable initializers live) before this spike can pass.
- **File #10** turns the passing spike's `genai_spike.cpp` into the production GenAI engine wrapper and consumes the Gate 0.1 decision (engine candidate in `genai_config.json` + manifest).
- **`onnx-genai.cpp`** (currently fully commented out) is the historical attempt; `genai_spike.cpp` supersedes it with the external-data-swap approach instead of the abandoned weight-cache approach.
- **`ORTGenAINative.kt`** (deprecated, all methods `throw NotImplementedError`) is NOT revived; File #10 introduces a clean wrapper.
- Memory experiments in `01_code_plans/04_memory_mapping_experiments.md` extend step 7 (mmap vs copy) but are non-blocking for this gate.

## Tests & smokes

- **Desktop swap smoke**: base vs swapped logits differ after one merge; fresh `Model()` per generation.
- **Constant-folding guard**: assert every handoff `inference_initializer_name` survives export as a live external initializer (parse `model.onnx`).
- **Config pass-through smoke**: custom `log_id` + `model_external_initializers_file_folder_path` appear in ORT logs / are not rejected.
- **Symbol smoke**: `nm`/`readelf` on the linked Android GenAI `.so` confirms `OgaCreateModelWithInitializers` ABSENT, `OgaCreateModel` PRESENT.
- **Android external-data resolution smoke**: GenAI `OgaCreateModel` on an app-storage package generates a valid token (relative externals resolve).
- **Android swap smoke**: train->merge overwrites `.bin`s; fresh `OgaCreateModel` reflects the change.
- **Gate 0.1 equivalence smoke**: SAME folder under Native and GenAI -> both correct for swapped weights; GenAI RSS within `ACCEPTED_RSS_DELTA` of Native.
- **Memory report** (informational): RSS pre-load / post-load / post-first-token for Native and GenAI, mmap vs copy annotation.

## Gate 0.1 pass/fail

**PASS** (adopt GenAI as a selectable engine) requires ALL:
1. The same File #8 package produces **correct output under BOTH Native and GenAI** reading the same folder.
2. Overwriting per-tensor external `.bin` files changes GenAI output on a fresh `OgaCreateModel` (no graph rewrite, no fork).
3. Trainable externals are NOT constant-folded (swap is observable).
4. GenAI peak RSS within the accepted threshold of Native.
5. GenAI Android resolves relative external data in the package dir.
6. `OgaCreateModelWithInitializers` confirmed fork-only and NOT required.

**FAIL** (keep manual loop only) if any: GenAI folds/copies trainable tensors so the swap no-ops; GenAI cannot resolve relative externals on Android; RSS exceeds threshold; the only working path requires the fork symbol or `model_input` rewrite.
