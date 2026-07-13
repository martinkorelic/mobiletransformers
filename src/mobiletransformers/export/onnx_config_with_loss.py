"""Vendored ``OnnxConfigWithLoss`` — training-graph loss/labels wrapper for Optimum ONNX configs.

**Why this exists.** Optimum 2.1 (the ``optimum-onnx`` split) *removed* ``OnnxConfigWithLoss`` with no
replacement (verified by ``spikes/optimum_migration/check_symbols.py``). The lower-level ``export()``
function it partnered with *survives*, and so do the base classes and dummy-input generators this
wrapper needs. So rather than reconstruct the training graph by hand via ``torch.onnx`` (the plan's
Fallback A) or pin a legacy optimum (Fallback B), we vendor the wrapper: the training-graph export
stays on Optimum's durable ``export()`` path, self-owned and version-pinned here.

**Provenance.** Adapted verbatim from ``optimum/exporters/onnx/base.py::OnnxConfigWithLoss`` at
optimum v1.24.0 (Apache-2.0, © The HuggingFace Team). Only imports were repointed for optimum 2.1
(``OnnxConfigWithPast`` now lives in ``optimum.exporters.onnx.base``) and type hints tightened. To be
enumerated in ``THIRD_PARTY_NOTICES.md`` by the relicense pass (#32).

**Profile.** Requires the export/train profile (optimum + torch installed). This module imports
optimum at load, so it must never be imported on the core import path — only from the training-export
site (``trainer/builder.py``). ``mobiletransformers.export.__init__`` stays empty by design.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.base import OnnxConfigWithPast
from optimum.utils import DEFAULT_DUMMY_SHAPES, DummyLabelsGenerator


class OnnxConfigWithLoss(OnnxConfig):
    """Wrap an ``OnnxConfig`` so the exported graph carries ``labels`` inputs and a ``loss`` output.

    This is what turns an inference OnnxConfig into a *training* graph (the graph
    ``onnxruntime.training.artifacts.generate_artifacts`` consumes downstream, plan #8).
    """

    _tasks_to_extra_inputs: dict[str, dict[str, dict[int, str]]] = {
        "feature-extraction": {"labels": {0: "batch_size"}},
        "fill-mask": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "text-generation": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "text-generation-with-past": {"labels": {0: "batch_size"}},
        "text2text-generation": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "text2text-generation-with-past": {"labels": {0: "batch_size"}},
        "text-classification": {"labels": {0: "batch_size"}},
        "token-classification": {"labels": {0: "batch_size", 1: "sequence_length"}},
        "multiple-choice": {"labels": {0: "batch_size"}},
        "question-answering": {
            "start_positions": {0: "batch_size"},
            "end_positions": {0: "batch_size"},
        },
        "image-classification": {"labels": {0: "batch_size"}},
    }
    _tasks_to_extra_outputs: dict[str, OrderedDict[str, dict[int, str]]] = {
        "feature-extraction": OrderedDict({"loss": {}}),
    }

    DUMMY_EXTRA_INPUT_GENERATOR_CLASSES = (DummyLabelsGenerator,)

    def __init__(
        self,
        config: OnnxConfig,
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        legacy: bool = False,
    ) -> None:
        self._onnx_config = config
        self.task = self._onnx_config.task
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self._normalized_config = self._onnx_config._normalized_config
        self.PATCHING_SPECS = self._onnx_config.PATCHING_SPECS
        self.variant = "default"
        self.legacy = legacy

    @classmethod
    def from_onnx_config(cls, config: OnnxConfig) -> OnnxConfigWithLoss:
        return cls(config)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = self._onnx_config.inputs
        inputs.update(self._tasks_to_extra_inputs[self.task])
        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = self._onnx_config.outputs
        extra_outputs = self._tasks_to_extra_outputs["feature-extraction"]
        common_outputs.update(extra_outputs)
        for key in reversed(extra_outputs.keys()):
            common_outputs.move_to_end(key, last=False)
        return copy.deepcopy(common_outputs)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs: Any) -> dict[str, Any]:
        dummy_inputs = self._onnx_config.generate_dummy_inputs(framework=framework, **kwargs)
        input_name, _ = next(iter(self._onnx_config.inputs.items()))
        batch_size = dummy_inputs[input_name].shape[0]

        if (
            isinstance(self._onnx_config, OnnxConfigWithPast)
            and self._onnx_config.use_past_in_inputs is True
            and self.task != "text-generation"
        ):
            kwargs["sequence_length"] = 1
        else:
            for _input_name, dynamic_axes in self._tasks_to_extra_inputs[self.task].items():
                if "sequence_length" in dynamic_axes.values():
                    kwargs["sequence_length"] = DEFAULT_DUMMY_SHAPES["sequence_length"]

        kwargs["num_labels"] = self._onnx_config._config.num_labels

        dummy_inputs_generators = [
            cls_(self.task, self._normalized_config, batch_size=batch_size, **kwargs)
            for cls_ in self.DUMMY_EXTRA_INPUT_GENERATOR_CLASSES
        ]

        for input_name in self._tasks_to_extra_inputs[self.task]:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name,
                        framework=framework,
                        int_dtype=self.int_dtype,
                        float_dtype=self.float_dtype,
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy '
                    "input generator to the model ONNX config."
                )

        return dummy_inputs

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: dict[str, Any], onnx_input_names: list[str] | None = None
    ) -> dict[str, Any]:
        return self._onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)

    def flatten_decoder_past_key_values(
        self, flattened_output: dict[str, Any], name: str, idx: int, t: Any
    ) -> None:
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]

    def flatten_seq2seq_past_key_values(
        self, flattened_output: dict[str, Any], name: str, idx: int, t: Any
    ) -> None:
        if len(t) not in [2, 4]:
            raise ValueError(
                "past_key_values to flatten should be of length 2 (self-attention only) or 4 "
                "(self and cross attention)."
            )
        if len(t) == 2:
            flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
            flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        if len(t) == 4:
            flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
            flattened_output[f"{name}.{idx}.encoder.value"] = t[3]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> dict[str, Any]:
        flattened_output: dict[str, Any] = {}
        if name in ["present", "past_key_values"]:
            if "text-generation" in self.task:
                for idx, t in enumerate(field):
                    self.flatten_decoder_past_key_values(flattened_output, name, idx, t)
            elif "text2text-generation" in self.task:
                for idx, t in enumerate(field):
                    self.flatten_seq2seq_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)
        return flattened_output

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        return self._onnx_config.torch_to_onnx_input_map

    @property
    def torch_to_onnx_output_map(self) -> dict[str, str]:
        return self._onnx_config.torch_to_onnx_output_map

    @property
    def values_override(self) -> dict[str, Any] | None:
        return self._onnx_config.values_override


__all__ = ["OnnxConfigWithLoss"]
