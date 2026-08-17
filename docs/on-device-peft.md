# PEFT methods

Fine-tuning a whole model on a phone is not on the table: the optimiser state alone would be several
times the model's size. **Parameter-efficient fine-tuning** makes it tractable by freezing the
backbone and training a small set of added parameters instead.

This framework supports three, plus two escape hatches. Which one a package was exported with is
recorded in its manifest (`peftMethods`) and readable on device as
`RuntimeCapabilities.peftMethods` — a package can only be re-trained inside the topology it was
exported with, so this is a property of the package, not a runtime choice.

| method | `--peft` | trains | merger |
| --- | --- | --- | --- |
| [LoRA](#lora) | `lora` | two matrices per target module | yes |
| [LoRA-XS](#lora-xs) | `lora-xs` | one small matrix per target module | yes |
| [MARS](#mars-multi-adapter-rank-sharing) | `mars` | one down-projection **shared across layers** | yes |
| all linear layers | `all` | every linear weight | — |
| frozen | `nolora` | nothing (export/inference only) | — |

Everything below is a registry row in
[`config/registry/peft.py`](https://github.com/martinkorelic/mobiletransformers/blob/main/src/mobiletransformers/config/registry/peft.py).
Adding a method is a row plus an enum member — not a new branch in the exporter.

## LoRA

*Low-Rank Adaptation.* For each targeted linear layer, freeze its weight `W` and learn two small
matrices `A` (down, `d × r`) and `B` (up, `r × d`), so the layer computes `Wx + BAx`. Only `A` and
`B` receive gradients.

The trainable count scales with **rank × depth**: every targeted layer gets its own pair. It is the
default, the best-understood option, and what five of the six published packages use.

## LoRA-XS

A reparameterisation of a LoRA wrap: `A` and `B` are fixed to the SVD of the base weight, and only a
small `r × r` matrix between them is trained. Fewer trainable parameters than LoRA at the same rank,
and it shares LoRA's module layout entirely — which is why it reuses the same adapter mapping and the
same merger.

## MARS (Multi-Adapter Rank Sharing)

**This project's own method**, designed for the phone specifically.

LoRA's cost grows with depth because every layer carries a private `A`. MARS observes that the
down-projection is the expensive half and that layers do not each need their own: one
`SharedAttentionAdapter` is shared across the q/k/v projections, and one `SharedMLPAdapter` across
gate/up — with only the small per-projection parts kept separate.

The result is that **the trainable parameter count grows with rank rather than with depth**. On the
published `gemma-3-270m-it` package that is **279,936 trainable parameters against a 268M backbone**.
Fewer trained parameters means a smaller optimiser state, which is the thing that actually decides
whether a training run fits in a phone's memory — and a smaller adapter to exchange in a
[federated round](FEDERATED.md).

MARS is the reason `derive_transpose_policy` has to observe tensor orientation rather than assume it:
its shared components are named differently from LoRA's (`shared_A`, not `adapter_A`), and a
fail-open default on that name is how a merge once silently transposed every layer.

Try it: [`mobiletransformers/gemma-3-270m-it`](https://huggingface.co/mobiletransformers/gemma-3-270m-it)
is exported with MARS and is on the app's catalog shelf.

## Merging, and why it matters here

Training produces adapter factors. **Merging** folds them back into the inference weights, on the
device, so that ordinary generation afterwards comes from the fine-tuned model rather than from an
adapter applied at runtime.

This is the step that makes an on-device fine-tune real rather than a demonstration, and it is done
in C++ against the inference graph's own initialisers. Each method declares which merger variant it
needs (`MergerVariant`), because the arithmetic differs: LoRA and LoRA-XS merge a `B·A` product,
MARS merges through its shared components.

```kotlin
val result = model.train(dataset, TrainConfig(mergeAtEnd = true))
// or explicitly, later:
model.merge()
```

## Choosing a method

- **Start with LoRA.** It is the default for a reason: best understood, widest architecture support,
  and the published packages that are easiest to compare against.
- **Reach for MARS** when memory is the binding constraint, when the model is deep relative to its
  width, or when you are exchanging adapters between devices.
- **LoRA-XS** when you want fewer trainable parameters without leaving LoRA's layout.
- `all` and `nolora` are not really fine-tuning strategies: `all` is a full-fine-tune baseline for
  host-side comparison, and `nolora` exports a frozen model for inference only.

Which architectures each method can target is data, not code — see
[the compatibility matrix](COMPATIBILITY_MATRIX.md), which is generated from the registry.

## Further reading

- [Export a model](EXPORT.md) — `--peft` and the training-stage flags
- [Package format](MODEL_FORMAT.md) — how adapter tensors are named and where the mapping lives
- [Federated adapters](FEDERATED.md) — exchanging trained factors between devices
- [Measured performance](mobile_evaluation.md) — step time and peak RAM on real hardware
