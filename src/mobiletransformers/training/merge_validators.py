import argparse
import json
import os
import textwrap
from typing import Any

import numpy as np
import onnxruntime as ort

from mobiletransformers.artifacts.checkpoint_names import to_checkpoint_name
from mobiletransformers.config.constants import (
    ARTIFACT_CONFIG,
    ARTIFACT_VALIDATOR_CONFIG,
    TRAIN_CONFIG,
)
from mobiletransformers.training.validators import ORTTrainer
from mobiletransformers.utils.yaml import load_config_from_file


class PEFTMergeValidator:
    """
    A validator for merging PEFT adapters into quantized base layers in ONNX Runtime.
    """

    def __init__(self, trainer: ORTTrainer, training_artifact_dir: str):
        """
        Initialize the PEFT merge validator.

        Args:
            trainer: ORTTrainer instance with checkpoint state
            training_config_path: Path to training_config.json containing peft_mapping
        """
        self.trainer = trainer
        self.training_artifact_dir = training_artifact_dir
        self.training_config_path = os.path.join(training_artifact_dir, "training_config.json")
        self.peft_mapping = None
        self.config = {}
        self.base_layer_params = {}
        self.adapter_params = {}
        self.input_adapter_parameters = {}
        self.output_adapter_parameters = {}

        self.merger_models = {}

        self.peft_method = None

        # Load the training configuration
        self._load_training_config()

        # Extract parameters from checkpoint
        self._extract_parameters()

        self._build_merger_models()

    def _load_training_config(self):
        """Load the training configuration containing PEFT mapping."""
        try:
            with open(self.training_config_path) as f:
                self.config = json.load(f)
                self.peft_mapping = self.config.get("peft_mapping", {})

                self.peft_method = self.config["peftMethod"]

                # 1. Populate self.merger_models
                self.merger_models = {}

                # get all .onnx files ending with merger_model.onnx or qmerger_model.onnx
                all_merger_files = [
                    f for f in os.listdir(self.training_artifact_dir) if f.endswith("merger_model.onnx")
                ]

                for fname in all_merger_files:
                    # extract method name from the file name
                    # e.g. "lora_merger_model.onnx" → "lora"
                    method_name = fname.split("_")[0]
                    method_dict = self.merger_models.setdefault(method_name, {})

                    full_path = os.path.join(self.training_artifact_dir, fname)

                    if fname.endswith("qmerger_model.onnx"):
                        method_dict["quantized"] = full_path
                    else:
                        method_dict["full_precision"] = full_path

                    if not self.peft_mapping:
                        raise ValueError("No 'peft_mapping' found in training config")
        except FileNotFoundError:
            raise FileNotFoundError(f"Training config file not found: {self.training_config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in training config file: {self.training_config_path}")

    def _extract_parameters(self):
        """Extract base layer and adapter parameters from checkpoint state."""
        if self.trainer.state is None:
            raise ValueError("Trainer checkpoint state is None. Make sure training has been completed.")

        # Get parameters object from checkpoint state
        parameters = self.trainer.state.parameters

        # Get all parameter names and objects by iterating over parameters
        # Each item is a tuple: (param_name, Parameter object)
        checkpoint_params = list(parameters)

        print(f"[INFO] Found {len(checkpoint_params)} parameters in checkpoint")

        # Extract base layer parameters (quantized weights, scales, zero_points)
        self._extract_base_layer_params(checkpoint_params)

        # Extract adapter parameters
        self._extract_adapter_params(checkpoint_params)

        self.create_merged_parameters()

    def _extract_base_layer_params(self, checkpoint_params: list):
        """Extract quantized base layer parameters."""
        for base_layer_name in self.peft_mapping.keys():
            base_params = {}

            # One owner for the peft->ORT wrapper rewrite. Spelled inline it read
            # `base_model.model.model.` -> `backbone.model.`, i.e. a DECODER's first module baked
            # into a rule that is really about the two WRAPPERS — so it converted nothing for an
            # encoder (`bert.encoder.layer…`) and every layer then read as missing.
            base_layer_name = to_checkpoint_name(base_layer_name)

            # Look for quantized weight, scale, and zero_point parameters
            weight_quantized_name = f"{base_layer_name}.weight_quantized"
            weight_scale_name = f"{base_layer_name}.weight_scale"
            weight_zero_point_name = f"{base_layer_name}.weight_zero_point"
            weight_noquantized_name = f"{base_layer_name}.weight"

            # Iterate through parameter tuples: (param_name, Parameter object)
            for param_name, param_obj in checkpoint_params:
                if param_name == weight_quantized_name:
                    base_params["weight_quantized"] = param_obj.data
                    print(f"[INFO] Found quantized weight: {param_name}")
                elif param_name == weight_scale_name:
                    base_params["x_scale"] = param_obj.data
                    print(f"[INFO] Found weight scale: {param_name}")
                elif param_name == weight_zero_point_name:
                    base_params["x_zero_point"] = param_obj.data
                    print(f"[INFO] Found weight zero point: {param_name}")
                elif param_name == weight_noquantized_name:
                    base_params["weight"] = param_obj.data
                    print(f"[INFO] Found non-quantized weights: {param_name}")

            if base_params:
                self.base_layer_params[base_layer_name] = base_params
                print(f"[INFO] Extracted base layer params for: {base_layer_name}")
            else:
                print(f"[WARNING] No quantized parameters found for base layer: {base_layer_name}")

    def _extract_adapter_params(self, checkpoint_params: list):
        """Extract adapter parameters for merging."""
        # Get all unique adapter names from the mapping

        self.adapter_params = {}
        for base_layer_name, adapter_names in self.peft_mapping.items():
            base_layer_name = to_checkpoint_name(base_layer_name)

            if base_layer_name not in self.adapter_params:
                self.adapter_params[base_layer_name] = {}

            for input_name, adapter_name in adapter_names.items():
                # Do not handle non string names, just add them to adapter params
                if not isinstance(adapter_name, str):
                    self.adapter_params[base_layer_name][input_name] = adapter_name
                    continue

                adapter_name = to_checkpoint_name(adapter_name)

                adapter_name = f"{adapter_name}.weight"
                # Iterate through parameter tuples: (param_name, Parameter object)
                for param_name, param_obj in checkpoint_params:
                    param_name = to_checkpoint_name(param_name)

                    if adapter_name == param_name:
                        self.adapter_params[base_layer_name][input_name] = param_obj.data

                        print(f"[INFO] Found adapter param: {param_name}")

    def get_base_layer_params(self, base_layer_name: str) -> dict[str, np.ndarray]:
        """
        Get quantized parameters for a specific base layer.

        Args:
            base_layer_name: Name of the base layer

        Returns:
            Dictionary containing weight_quantized, weight_scale, weight_zero_point
        """
        return self.base_layer_params.get(base_layer_name, {})

    def get_adapter_params(self, adapter_name: str) -> dict[str, np.ndarray]:
        """
        Get parameters for a specific adapter.

        Args:
            adapter_name: Name of the adapter

        Returns:
            Dictionary containing adapter parameters
        """
        return self.adapter_params.get(adapter_name, {})

    def get_mapping_for_base_layer(self, base_layer_name: str) -> dict[str, Any]:
        """
        Get the PEFT mapping configuration for a specific base layer.

        Args:
            base_layer_name: Name of the base layer

        Returns:
            Dictionary containing adapter mappings for the base layer
        """
        return self.peft_mapping.get(base_layer_name, {})

    def create_merged_parameters(self):
        self.input_adapter_parameters = {}
        self.output_adapter_parameters = {}

        for base_name, base_params in self.base_layer_params.items():
            self.input_adapter_parameters[base_name] = {**base_params, **self.adapter_params[base_name]}

            self.output_adapter_parameters[base_name] = {}

            # Quantized weight output
            if "weight_quantized" in self.input_adapter_parameters[base_name]:
                self.output_adapter_parameters[base_name] = {
                    "merged_weight_quantized": None,
                    "merged_zero_point": None,
                    "merged_scale": None,
                }
            # Full precision weight output
            elif "weight" in self.input_adapter_parameters[base_name]:
                self.output_adapter_parameters[base_name] = {"merged_weight": None}

            # check for None, empty, or empty numpy arrays
            for key, value in self.input_adapter_parameters[base_name].items():
                if value is None:
                    print(f"[WARNING]: {base_name} -> {key} is None")

                elif isinstance(value, (list, tuple, dict)) and len(value) == 0:
                    print(f"[WARNING]: {base_name} -> {key} is an empty {type(value).__name__}")

                elif isinstance(value, np.ndarray) and value.size == 0:
                    print(f"[WARNING]: {base_name} -> {key} is an empty numpy array")

                # convert plain Python ints or floats to numpy arrays
                if isinstance(value, (int, np.integer)):
                    self.input_adapter_parameters[base_name][key] = np.array(value, dtype=np.int64)
                    # print(f"[INFO]: Converted {base_name} -> {key} to numpy int64 array")

                elif isinstance(value, (float, np.floating)):
                    self.input_adapter_parameters[base_name][key] = np.array(value, dtype=np.float32)
                    # print(f"[INFO]: Converted {base_name} -> {key} to numpy float32 array")

    def _build_merger_models(self):
        """
        Builds the onnxruntime sessions for each merger model
        and stores them in self.merger_models[method_name]["quantized"/"full_precision"]
        """
        for method_name, models in self.merger_models.items():
            for precision, path in models.items():
                print(f"Loading {precision} merger model for method {method_name} from {path}")
                session = ort.InferenceSession(path)
                self.merger_models[method_name][precision] = session

    def clear_merger_models(self):
        """
        Cleanly releases and deletes all loaded merger ONNX inference sessions
        from memory.
        """
        if hasattr(self, "merger_models"):
            for method_name, merger_types in self.merger_models.items():
                for precision, session in merger_types.items():
                    if isinstance(session, ort.InferenceSession):
                        print(f"[INFO] Releasing inference session for method '{method_name}' ({precision})")
                        session._sess = None
                        del session
                # remove references from dictionary
                self.merger_models[method_name].clear()
            self.merger_models.clear()

        print("[INFO] All merger inference sessions cleared from memory.")

    def compute_base_layers_from_adapters(self, save_directory: str = None):

        for base_layer, input_layers in self.input_adapter_parameters.items():
            print(f"[DEBUG] Computing merge for base_layer: {base_layer}")

            # Print input layer keys and their shapes
            for key, val in input_layers.items():
                if isinstance(val, np.ndarray):
                    print(f"[DEBUG] Input '{key}' shape: {val.shape}")
                else:
                    print(f"[WARNING] Input '{key}' is not a numpy array, type={type(val)}")

            # Run the MARS merger technique
            if "shared_A" in input_layers:
                self._run_merger_model(self.merger_models["mars"], base_layer, input_layers)
            # Run the LoRA merger technique
            elif "adapter_A" in input_layers:
                # We do not need rank input here
                input_layers.pop("rank", None)
                self._run_merger_model(self.merger_models["lora"], base_layer, input_layers)

        print("\n[DEBUG] Finished merging. Output adapter parameters:")
        for base_layer, output_dict in self.output_adapter_parameters.items():
            print(f"  - Base layer: {base_layer}")
            for name, arr in output_dict.items():
                if isinstance(arr, np.ndarray):
                    print(f"      * {name}: shape={arr.shape}")
                else:
                    print(f"      * {name}: type={type(arr)} (not a numpy array)")

        # Optionally save to disk
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            for base_layer, output_dict in self.output_adapter_parameters.items():
                save_path = os.path.join(save_directory, f"{base_layer}.npz")
                np.savez(save_path, **output_dict)
                print(f"[INFO] Saved merged parameters for {base_layer} to {save_path}")

    def _run_merger_model(self, merger_models: dict, base_layer: str, input_layers: dict):
        """
        Runs the quantized or full precision merger technique of the PEFT method.
        """

        session = None

        if "weight_quantized" in input_layers:
            session = merger_models["quantized"]

            merged_weight_quantized, merged_zero_point, merged_scale = session.run(None, input_layers)

            # Debug outputs
            print(
                f"[DEBUG] merged_weight_quantized shape: {merged_weight_quantized.shape if isinstance(merged_weight_quantized, np.ndarray) else 'not ndarray'}"
            )
            print(
                f"[DEBUG] merged_zero_point shape: {merged_zero_point.shape if isinstance(merged_zero_point, np.ndarray) else 'not ndarray'}"
            )
            print(
                f"[DEBUG] merged_scale shape: {merged_scale.shape if isinstance(merged_scale, np.ndarray) else 'not ndarray'}"
            )

            if isinstance(merged_weight_quantized, np.ndarray) and merged_weight_quantized.size == 0:
                print("[WARNING] merged_weight_quantized is empty!")
            if isinstance(merged_zero_point, np.ndarray) and merged_zero_point.size == 0:
                print("[WARNING] merged_zero_point is empty!")
            if isinstance(merged_scale, np.ndarray) and merged_scale.size == 0:
                print("[WARNING] merged_scale is empty!")

            self.output_adapter_parameters[base_layer] = {
                "weight_quantized": merged_weight_quantized,
                "weight_scale": merged_scale,
                "weight_zero_point": merged_zero_point,
            }

        elif "weight" in input_layers:
            session = merger_models["full_precision"]

            merged_weight = session.run(None, input_layers)[0]

            # Debug output
            print(f"[DEBUG] merged_weight type: {type(merged_weight)}")
            if isinstance(merged_weight, np.ndarray):
                print(f"[DEBUG] merged_weight shape: {merged_weight.shape}")
                if merged_weight.size == 0:
                    print("[WARNING] merged_weight is empty!")
            else:
                print("[WARNING] merged_weight is not a numpy array")

            self.output_adapter_parameters[base_layer] = {"weight": merged_weight}

    def print_summary(self):
        """Print a summary of extracted parameters."""
        print("\n" + "=" * 50)
        print("PEFT Merge Validator Summary")
        print("=" * 50)

        print(f"Base layers found: {len(self.base_layer_params)}")
        for base_layer_name, params in self.base_layer_params.items():
            print(f"  - {base_layer_name}: {list(params.keys())}")

        print(f"\nAdapters found: {len(self.adapter_params)}")
        for adapter_name, params in self.adapter_params.items():
            print(f"  - {adapter_name}: {list(params.keys())}")

        print(f"\nPEFT mappings: {len(self.peft_mapping)}")


def create_peft_merge_validator(trainer: ORTTrainer, training_artifact_dir: str) -> PEFTMergeValidator:
    """
    Create a PEFT merge validator instance.

    Args:
        trainer: ORTTrainer instance with checkpoint state
        training_config_path: Path to training_config.json containing peft_mapping

    Returns:
        PEFTMergeValidator instance
    """
    return PEFTMergeValidator(trainer, training_artifact_dir)


def parse_extra_options(extra_options: list[str]) -> dict[str, str]:
    """
    Parse additional options in KEY=VALUE format into a dictionary.
    """
    options_dict = {}
    for option in extra_options:
        if "=" in option:
            key, value = option.split("=", 1)
            options_dict[key] = value
        else:
            raise ValueError(f"Invalid format for extra option '{option}'. Use KEY=VALUE format.")

    print(f"Extra options: {options_dict}")
    return options_dict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validator for exported ONNX artifacts for on-device training.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--model_id", type=str, help="Identifier for the model to be converted.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments.",
    )
    parser.add_argument("--training_artifact_dir", type=str, help="Path to training artifact directory.")
    parser.add_argument(
        "--test_training_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ...
            """),
    )
    parser.add_argument(
        "--test_scheduler_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ...
            """),
    )
    args = parser.parse_args()

    user_train_generation_config = {}
    default_train_generation_config = {
        "trainFile": "",
        "taskName": "",
        "peftMethod": "",
        "numTrainEpochs": 1,
        "maxSteps": 4,
        "saveSteps": 50,
        "gradAccumSteps": 2,
        "removeLongSample": True,
        "maxSequenceLength": 512,
        "maxDatasetLength": 256,
        "datasetBatchSize": 64,
        "testRatio": 0.1,
        "split": True,
        "shuffle": True,
        "schedulerType": "cosine",
    }

    user_scheduler_generation_config = {}
    default_scheduler_generation_config = {
        "minLearningRate": 0,
        "cosineLearningRate": 0.0001,
        "warmupSteps": 10,
        "linearLearningRate": 0.0001,
        "startFactor": 1,
        "endFactor": 0.333,
    }

    config_dict = None

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        # Specific
        args.model_id = config_dict[TRAIN_CONFIG]["model_id"]
        args.training_artifact_dir = os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "train")
        args.test_scheduler_config = config_dict[ARTIFACT_VALIDATOR_CONFIG]["test_training_config"][
            "schedulerOptions"
        ]

        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_VALIDATOR_CONFIG].items():
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)

    else:
        user_train_generation_config = parse_extra_options(args.test_training_config)
        args.test_training_config = {**default_train_generation_config, **user_train_generation_config}
        user_scheduler_generation_config = parse_extra_options(args.test_scheduler_config)
        args.test_scheduler_config = {
            **default_scheduler_generation_config,
            **user_scheduler_generation_config,
        }

    return args


# Example usage
if __name__ == "__main__":
    trainer = ORTTrainer("build/train", load_from_state=True)

    trainer.train()

    validator = create_peft_merge_validator(trainer, "")

    validator.compute_base_layers_from_adapters(
        save_directory=os.path.join("training_artifact_dir", "temp_weights/")
    )
