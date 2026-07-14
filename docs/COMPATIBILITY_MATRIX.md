# Compatibility Matrix

> **Generated** — rendered from `model_support_matrix.json` (#20). Do not hand-edit. Regenerate with `mobiletransformers support-matrix --md docs/COMPATIBILITY_MATRIX.md` under the `export` profile (live detection needs transformers + optimum).

- Generated at: `2026-07-14T00:00:00Z`
- Toolchain: optimumOnnxVersion=0.1.0, transformersVersion=4.46.2

## Axes (enumerated from the registries/enums — not hand-maintained)

- **PEFT method:** lora, lora-xs, mars, all, nolora
- **Quantization:** QInt8, QUInt8, int4
- **Merger variant:** lora, lora_q, mars_q
- **Engine:** native, genai (native is the guaranteed path; genai is opt-in)
- **Status pipeline (each implies all earlier ones):** Optimum export → Package → Train artifacts → Android inference → Android training → RAG

| Model | Type | Task | Optimum export | Package | Train artifacts | Android inference | Android training | RAG | Evidence / blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M` | llama | text-generation-with-past | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | no android probe recorded for android_inference_ready |
| `Qwen/Qwen2-0.5B` | qwen2 | text-generation-with-past | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | no android probe recorded for android_inference_ready |
| `sentence-transformers/all-MiniLM-L6-v2` | bert | feature-extraction | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | MARS/PEFT target modules not verified for this architecture |

