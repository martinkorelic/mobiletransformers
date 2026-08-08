# GenAI desktop reference (moved out of `inference/` by Migration Map S9)

`generator_genai.py` is a **desktop prototype**, not library code: it has no importers anywhere in the
repo, hardcodes a model path, and its two functions are exploratory smokes for the onnxruntime-genai
Python loop.

It is kept because the plans explicitly name it as a reference —
`agent_docs/01_code_plans/03_inference_engine_abstraction_native_and_genai.md` says it "stays as the
desktop reference for the GenAI loop", and the Tier-0 doc cites its `params.set_model_input` prototype.
It stays **out** of `src/` for the same reason the S8 benchmark scripts do: a wheel should not ship
modules that run an experiment on import.

The shipping GenAI path is the Android engine (`ORTGeneratorGenAI` + `genai_runtime.cpp`), not this.
