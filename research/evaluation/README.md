# Experiment scripts (moved out of `evaluation/` by Migration Map S8)

These are **not** library code and **not** tests, despite `scripts/` having been called `test/`.

Each file has zero classes and zero functions: it does its work in top-level statements, so importing
one *runs an experiment*. They also hardcode paths like
`experiment_results/TinyLlama_v1.1-lora_xs/...` that exist only on the machine that produced them.

They stay in the repo because they document how published numbers were produced, and they stay **out**
of `src/` because an installable wheel must not contain modules that execute a benchmark on import.
This mirrors S5's treatment of `artifact/tflite_builder.py`.

- `benchmark/` — deepeval harnesses (ARC, BoolQ, HellaSwag, LogiQA, WinoGrande).
- `scripts/` — one-off generation/visualisation checks.

Running them needs the `eval` extra (`uv sync --extra eval`) plus whatever the individual script
hardcodes.
