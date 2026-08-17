# GenAI desktop reference (moved out of `inference/` by Migration Map S9)

`generator_genai.py` is a **desktop prototype**, not library code: it has no importers anywhere in the
repo, hardcodes a model path, and its two functions are exploratory smokes for the onnxruntime-genai
Python loop.

It is kept as the **desktop reference for the GenAI loop**: the Android engine in
`ORTGeneratorGenAI.kt` is a port of it, so when the two disagree this is the version that can be
stepped through in a debugger. It stays **out** of `src/` for the same reason the benchmark scripts
do: a wheel should not ship modules that run an experiment on import.

The shipping GenAI path is the Android engine (`ORTGeneratorGenAI` + `genai_runtime.cpp`), not this.
