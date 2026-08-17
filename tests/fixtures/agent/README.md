# `mobile_actions_sample.jsonl`

Five records excerpted from **[`google/mobile-actions`](https://huggingface.co/datasets/google/mobile-actions)**,
© Google, licensed **CC-BY-4.0**. Unmodified except for key ordering.

Chosen to cover the shapes the importer has to get right, so
`tests/unit/test_mobile_actions_import.py` runs offline:

| records | why |
| --- | --- |
| 3 | `metadata: "train"`, one `tool_call` — the ordinary path |
| 1 | `metadata: "train"`, **several** `tool_calls` — the `multi_call` policy |
| 1 | `metadata: "eval"` — split filtering |

Between them they declare all 7 tools, including `send_email` and `create_contact` whose `required`
is a strict subset of `properties` — the case that `ActionSpec.required_parameters` exists for.
