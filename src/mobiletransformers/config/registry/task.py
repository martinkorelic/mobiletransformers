"""Task registry — single source of truth for what a task type implies at export time.

Replaces the task-keyed branches that survived #6's registry pass in ``export/training_export.py``:

* ``if task_type == "text-generation": AutoModelForCausalLM else AutoModel`` — the auto-model class;
* ``if spec.task == TaskType.FEATURE_EXTRACTION: cls(config, task=…) else cls(…, use_past=…)`` — the
  KV-cache kwargs;
* ``LoraConfig(…, task_type="CAUSAL_LM")``, **hardcoded at both call sites**.

The third was not a style problem. `"CAUSAL_LM"` is wrong for an encoder: PEFT uses the task type to
decide which modules to wrap and which head to keep trainable, so an encoder wrapped as `CAUSAL_LM`
is mis-configured from the first step. It was invisible while only decoders were trained, and it is
exactly the kind of latent branch #33 (encoder support) walks into — which is why the registry lands
before the encoder work rather than after it.

Adding a task is a row here plus a ``TaskType`` member, not a new ``elif``. Class bindings are lazy
dotted paths (the same convention as ``architecture.py``/``peft.py``) so the core profile imports this
without pulling transformers or torch.

## Adding a training objective

A ``TaskSpec`` is the whole description of an objective. To add one (masked-LM and contrastive
embedding are the obvious next two):

1. Add a ``TaskType`` member and mirror it in ``constants/TaskType.kt``, then regenerate with
   ``python -m mobiletransformers.codegen.enums`` — ``make parity`` fails until both sides agree.
2. Add a row here. The fields that actually differ between objectives are:
   ``auto_model_class`` (which head), ``peft_task_type`` (how PEFT wraps it), ``label_shape``
   (per-token vs per-sequence — the big one), ``model_init_kwargs`` (e.g. ``num_labels``),
   ``uses_kv_cache``, and ``trainable``.
3. Reuse a wrapper if the input set matches; add one only if the **forward signature** differs, since
   those parameter names become the exported ONNX input names.
4. Add architecture rows for the concrete `<Arch>For<Head>` classes the objective loads.

Worked example — masked-LM would be `auto_model_class="transformers.AutoModelForMaskedLM"`,
`peft_task_type=None` (PEFT has no MLM task type; it infers), `label_shape=("batch_size",
"sequence_length")` (per token, like the decoder), the existing encoder wrapper, and no
`model_init_kwargs`. No new branches anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mobiletransformers.config.constants import TaskType
from mobiletransformers.exceptions import UnsupportedModelError


@dataclass(frozen=True)
class TaskSpec:
    """What a task type implies for the export path."""

    task: TaskType

    #: Dotted path to the transformers auto-model class the export loads the source model with.
    auto_model_class: str

    #: Whether this task's ONNX config takes the KV-cache kwargs (``use_past``/``use_past_in_inputs``).
    #:
    #: Decoders cache past keys/values across steps; encoders do a single forward pass and their
    #: ``OnnxConfig`` does not accept the kwargs at all — passing them raises. This is the data behind
    #: the `spec.task == FEATURE_EXTRACTION` branch, and it is also the reason encoder support gets to
    #: **delete** the autoregressive path rather than special-case it.
    uses_kv_cache: bool

    #: PEFT's own task-type string (``peft.TaskType``), or ``None`` when PEFT should infer it.
    #:
    #: Feature-extraction models have no LM head, so ``FEATURE_EXTRACTION`` is what keeps PEFT from
    #: looking for one. Passing ``CAUSAL_LM`` here — as both LoRA call sites used to, unconditionally —
    #: mis-wraps an encoder.
    peft_task_type: str | None

    #: Dotted path to the ``torch.nn.Module`` that wraps the model for the training-graph export.
    #:
    #: A class rather than a list of input names because ``torch.onnx`` derives the exported ONNX input
    #: names from the wrapper's **forward signature** — they are part of the on-device contract, so they
    #: have to be written out literally, not assembled. Decoders take ``position_ids``, BERT-family
    #: encoders take ``token_type_ids``; a decoder-shaped wrapper fails an encoder export inside optimum
    #: at ``check_dummy_inputs_are_allowed``.
    trainer_wrapper_class: str = "mobiletransformers.export.training_export.OnnxTrainerWrapper"

    #: Whether this task can produce a **training** graph at all.
    #:
    #: ``feature-extraction`` cannot: ``AutoModel`` has no head, so its forward takes no ``labels`` and
    #: returns no loss, and the export dies deep inside torch with an unexplained
    #: ``unexpected keyword argument 'labels'``. Declaring it here turns that into one clear message
    #: naming the task and the alternative.
    trainable: bool = True

    #: Shape of the label tensor this objective supervises, as a symbolic-dim tuple.
    #:
    #: The axis on which objectives differ most. Token-level objectives (causal LM, and MLM when it
    #: lands) supervise one label per position; sequence-level ones supervise one per example. It
    #: propagates: ``OnnxConfigWithLoss``'s dummy label generator, the on-device data curator, and how
    #: a reported loss should be read all key off this.
    label_shape: tuple[str, ...] = ("batch_size", "sequence_length")

    #: Node-name substrings kept OUT of dynamic quantization, beyond the trainable modules.
    #:
    #: The backward pass has to traverse the task head to reach the adapters, and ORT registers **no
    #: gradient builder for ``DynamicQuantizeLinear``** — so quantizing anything on the gradient path
    #: makes ``generate_artifacts`` fail with "The gradient builder has not been registered". Decoders
    #: get away with ``embed_head`` alone; BERT-family classification also routes through ``pooler``
    #: and ``classifier``, and omitting them fails exactly there.
    quantization_exclude_layers: tuple[str, ...] = ("embed_head",)

    #: Extra kwargs for ``from_pretrained``, e.g. ``num_labels`` for a classification head.
    #:
    #: Values here are *defaults*; a caller-supplied value wins. Keeping them as data is what lets a
    #: new objective (masked-LM, contrastive) arrive as a row instead of another branch at the load
    #: site.
    model_init_kwargs: dict[str, object] = field(default_factory=dict)

    @property
    def is_token_level(self) -> bool:
        """True when the objective supervises one label per token rather than per sequence."""
        return "sequence_length" in self.label_shape

    def onnx_config_kwargs(self, *, training_mode: bool) -> dict[str, bool]:
        """The KV-cache kwargs this task's ``*OnnxConfig`` should be constructed with.

        Training graphs never use a cache (the backward pass needs the full sequence), so the kwargs
        are ``not training_mode`` for cached tasks and absent entirely for uncached ones.
        """
        if not self.uses_kv_cache:
            return {}
        return {"use_past": not training_mode, "use_past_in_inputs": not training_mode}


#: The closed set of export task types. Keyed by :class:`TaskType`, which is mirrored to Kotlin and
#: parity-checked, so a task cannot exist here without existing on both sides of the boundary.
TASK_REGISTRY: dict[TaskType, TaskSpec] = {
    TaskType.TEXT_GENERATION: TaskSpec(
        task=TaskType.TEXT_GENERATION,
        auto_model_class="transformers.AutoModelForCausalLM",
        uses_kv_cache=True,
        peft_task_type="CAUSAL_LM",
        label_shape=("batch_size", "sequence_length"),
    ),
    TaskType.FEATURE_EXTRACTION: TaskSpec(
        task=TaskType.FEATURE_EXTRACTION,
        auto_model_class="transformers.AutoModel",
        uses_kv_cache=False,
        peft_task_type="FEATURE_EXTRACTION",
        # BERT-family: token_type_ids in, no position_ids. Matches `BertOnnxConfig`'s dummy inputs.
        trainer_wrapper_class="mobiletransformers.export.training_export.OnnxEncoderTrainerWrapper",
        # Export/inference only — no head, therefore no loss. This is the RAG embedder's task.
        trainable=False,
        label_shape=(),
    ),
    TaskType.SEQUENCE_CLASSIFICATION: TaskSpec(
        task=TaskType.SEQUENCE_CLASSIFICATION,
        auto_model_class="transformers.AutoModelForSequenceClassification",
        uses_kv_cache=False,
        peft_task_type="SEQ_CLS",
        trainer_wrapper_class=(
            "mobiletransformers.export.training_export.OnnxSequenceClassificationTrainerWrapper"
        ),
        # One label per SEQUENCE, not per token — the axis that separates this from every decoder task.
        label_shape=("batch_size",),
        # The classification head sits on the gradient path between the loss and the adapters, and
        # DynamicQuantizeLinear has no gradient. Quantizing it breaks generate_artifacts outright.
        quantization_exclude_layers=("embed_head", "pooler", "classifier"),
        # Binary by default; a caller passing num_labels wins. The head is randomly initialised by
        # construction (it does not exist in the checkpoint), which is correct and expected — it is the
        # part fine-tuning is supposed to learn.
        model_init_kwargs={"num_labels": 2},
    ),
}


def get_task_spec(task: TaskType | str) -> TaskSpec:
    """Resolve a task type (enum or wire string) to its spec, failing closed on an unknown one.

    Accepts the wire string because the export entry points take ``task_type`` as text from the CLI,
    and a single parse at the boundary is the #6 convention — the alternative is every caller doing
    its own ``TaskType(...)`` and each getting the error message slightly wrong.
    """
    if not isinstance(task, TaskType):
        # `text-generation-with-past` and friends are KV-cache variants of the same task; the suffix
        # selects graph shape, not task identity, and TasksManager owns that selection.
        base = str(task).replace("-with-past", "")
        try:
            task = TaskType(base)
        except ValueError as exc:
            raise UnsupportedModelError(
                f"unknown task type {task!r} (expected one of {sorted(t.value for t in TASK_REGISTRY)})"
            ) from exc

    try:
        return TASK_REGISTRY[task]
    except KeyError as exc:  # pragma: no cover - unreachable while TaskType and the registry agree
        raise UnsupportedModelError(
            f"task {task.value!r} has no registry row; add one to TASK_REGISTRY rather than "
            "branching on the task at the call site"
        ) from exc
