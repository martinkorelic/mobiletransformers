from abc import ABC
import copy
import random
from typing import Any, Dict, Iterable, List, Optional, OrderedDict
from optimum.exporters.onnx.config import OnnxConfig
from optimum.exporters.onnx import OnnxConfigWithLoss
from optimum.utils import DummyInputGenerator, DEFAULT_DUMMY_SHAPES, DTYPE_MAPPER
from optimum.utils.normalized_config import NormalizedConfig

class OnnxConfigWithInference(OnnxConfig, ABC):

    def __init__(self, config: OnnxConfig, int_dtype: str = "int64", float_dtype: str = "fp32", legacy: bool = False):
        self._config = config
        self.task = self._config.task
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self._normalized_config = self._config._normalized_config
        self.PATCHING_SPECS = self._config.PATCHING_SPECS
        self.variant = "default"
        self.legacy = legacy
    
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        return self._config.generate_dummy_inputs(framework=framework, **kwargs)

    @classmethod
    def from_onnx_config(cls, config: OnnxConfig) -> "OnnxConfigWithInference":
        return cls(config)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        inputs = self._config.inputs
        
        # Update attention mask
        inputs["attention_mask"] = {0: 'batch_size', 1: 'total_sequence_length'}
    
        return inputs
    
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = self._config.outputs
        
        for key in common_outputs.keys():
            if "present" in key:
                common_outputs[key] = {0: 'batch_size', 2: 'total_sequence_length'}
            
        return copy.deepcopy(common_outputs)
    
    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return self._config.generate_dummy_inputs_for_validation(reference_model_inputs)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return self._config.torch_to_onnx_input_map

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return self._config.torch_to_onnx_output_map

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
        return self._config.values_override
    
class DummyTrainFlagGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "training_mode"
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        self.task = task

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):

        return self.constant_tensor((1,), value=1, framework=framework, dtype=int_dtype)

class OnnxConfigWithTrainer(OnnxConfigWithLoss, ABC):
    """
    Wrapper for the children classes of `optimum.exporters.onnx.OnnxConfig` to export the model through the ONNX format
    with loss in outputs and labels in the inputs. For seq-to-seq models, labels will be appended to the inputs of
    decoders.
    """

    _tasks_to_extra_inputs = {
        "feature-extraction": {"training_mode": {0: "is_training"}},
        "fill-mask": {"training_mode": {0: "is_training"}},
        "text-generation": {"training_mode": {0: "is_training"}},
        "text-generation-with-past": {"training_mode": {0: "is_training"}},
        "text2text-generation": {"training_mode": {0: "is_training"}},
        "text2text-generation-with-past": {"training_mode": {0: "is_training"}},
        "text-classification": {"training_mode": {0: "is_training"}},
        "token-classification": {"training_mode": {0: "is_training"}},
        "multiple-choice": {"training_mode": {0: "is_training"}},
        "question-answering": {
            "start_positions": {0: "is_training"},
            "end_positions": {0: "is_training"},
        },
        "image-classification": {"training_mode": {0: "is_training"}},
    }

    DUMMY_EXTRA_INPUT_GENERATOR_CLASSES = (DummyTrainFlagGenerator,)

    def __init__(self, config: OnnxConfig, int_dtype: str = "int64", float_dtype: str = "fp32", legacy: bool = False):
        self._onnx_config = config
        self.task = self._onnx_config.task
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self._normalized_config = self._onnx_config._normalized_config
        self.PATCHING_SPECS = self._onnx_config.PATCHING_SPECS
        self.variant = "default"
        self.legacy = legacy

    @classmethod
    def from_onnx_config(cls, config: OnnxConfig) -> "OnnxConfigWithTrainer":
        return cls(config)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        inputs = self._onnx_config.inputs
        inputs.update(self._tasks_to_extra_inputs[self.task])
        return inputs

    #@property
    #def outputs(self) -> Dict[str, Dict[int, str]]:
    #    common_outputs = self._onnx_config.outputs
    #    extra_outputs = self._tasks_to_extra_outputs["feature-extraction"]
    #    common_outputs.update(extra_outputs)
    #    for key in reversed(extra_outputs.keys()):
    #        common_outputs.move_to_end(key, last=False)
    #    return copy.deepcopy(common_outputs)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = self._onnx_config.generate_dummy_inputs(framework=framework, **kwargs)
        input_name, _ = next(iter(self._onnx_config.inputs.items()))
        batch_size = dummy_inputs[input_name].shape[0]

        dummy_inputs_generators = [
            cls_(self.task, self._normalized_config, batch_size=batch_size, **kwargs)
            for cls_ in self.DUMMY_EXTRA_INPUT_GENERATOR_CLASSES
        ]

        for input_name in self._tasks_to_extra_inputs[self.task]:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name, framework=framework, int_dtype=DTYPE_MAPPER.pt(self.int_dtype), float_dtype=DTYPE_MAPPER.pt(self.float_dtype)
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        return dummy_inputs

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return self._onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return self._onnx_config.torch_to_onnx_input_map

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return self._onnx_config.torch_to_onnx_output_map

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
        return self._onnx_config.values_override