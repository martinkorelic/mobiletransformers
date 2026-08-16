# Fine-tuning a model on the phone itself

## Why adapters instead of full fine-tuning

Training every weight of a language model on a phone is not a memory problem you can optimise your
way out of — the optimizer state alone is several times the size of the model. Parameter-efficient
fine-tuning trains a small number of extra weights and leaves the original ones frozen, which turns an
impossible job into one that fits in a few hundred megabytes.

## LoRA

LoRA adds two thin matrices beside a frozen projection. Instead of learning a full update to a
weight of shape `d x k`, it learns `A` of shape `d x r` and `B` of shape `r x k`, where the rank `r`
is small — 8 or 16 is typical. The product `BA` is the update. Because `r` is tiny, the number of
trained parameters drops by three or four orders of magnitude: a 135-million-parameter model trains
roughly 370 thousand weights.

## MARS

Multi-Adapter Rank Sharing goes further by sharing adapter factors across layers rather than giving
every layer its own pair. The parameter count then grows with the rank rather than with the depth of
the network, so a deeper model costs almost nothing extra to adapt. This is the technique the
MobileTransformers research contributes, and it is the one to pick when memory is the binding
constraint rather than accuracy.

## Merging

After training, the adapter can be merged back into the base weights so that inference costs exactly
what it did before — no extra matrices, no extra latency. Merging happens on the device, tensor by
tensor, with an atomic rename and a checksum per file. Nothing is rewritten in the graph.

## What a training run costs

A run is bounded by three numbers you choose: how many steps, how large a batch, and how long a
sequence. Sequence length dominates memory, because attention grows with its square. If a run is
killed by the operating system, the sequence length is the first knob to turn down, then the batch
size. An out-of-memory death on Android is a SIGKILL, so there is no exception and no stack trace —
only the process disappearing.
