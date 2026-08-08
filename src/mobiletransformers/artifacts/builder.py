# DECOMPOSE(#5): split graph assembly vs. quantization vs. gen_artifacts orchestration into
# src/mobiletransformers/{export,artifacts} as touched (#7/#9). ~38 KB.
"""
Script that creates the training and inference model artifacts which can be deployed to the device.
The models are utilized by the on-device application.
"""

import argparse
import contextlib
import gc
import json
import os
import textwrap
import time

import yaml
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import onnx
import onnxruntime as rt
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import (
    convert_model_to_external_data,
    set_external_data,
    write_external_data_tensors,
)
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.training import artifacts, onnxblock
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from transformers import AutoConfig, AutoTokenizer

from mobiletransformers.config.constants import (
    ARTIFACT_CONFIG,
    INFERENCE_CONFIG,
    TASK_NAME_TO_DATASET,
    TRAIN_CONFIG,
    PEFTMethod,
)
from mobiletransformers.config.registry.merger import emit_merger_models
from mobiletransformers.config.settings import get_settings
from mobiletransformers.export.tokenizer_export import export_tokenizer_config
from mobiletransformers.inference.generator import generate_tokens_onnx
from mobiletransformers.training.data import load_and_save_dataset
from mobiletransformers.utils.paths import delete_directory, move_files_excluding, move_onnx_model

#: Suffixes the quantizer appends beside a weight it packs. Same role vocabulary the handoff map uses
#: (`artifacts/handoff_map.py`): the packed payload plus its dequantization parameters.
_QUANT_COMPANION_SUFFIXES = ("_quantized", "_scale", "_zero_point")


def _is_quant_companion(name: str) -> bool:
    """True for a quantizer-produced companion of a weight (packed payload / scale / zero-point).

    ``requires_grad`` is matched by **substring**, so a trainable ``…lora_B.lora.weight`` also matches
    ``…lora_B.lora.weight_quantized`` and its ``_scale``/``_zero_point``. Those are not differentiable —
    ORT rejects the whole artifact generation with *"Cannot compute the partial derivative for
    '…weight_quantized' as it's unreachable from the output node(s)"*, so a quantized PEFT export could
    not produce training artifacts at all. This is the same quantized-name hazard `HandoffMap.validate`
    guards on the emit side.
    """
    return name.endswith(_QUANT_COMPANION_SUFFIXES)


@contextlib.contextmanager
def _onnx_external_data_overwrite():
    """Let ``onnx.save`` overwrite an existing external-data file, for the duration of the block.

    ORT-training's ``onnxblock.Block`` writes ``temp.onnx`` + ``temp.onnx.data`` into the **current
    working directory** (`blocks.py:36`) and only removes them in ``__del__``. Every Block shares that
    one filename, so with a model large enough to take the external-data path
    (``accessor.has_path``), the second Block of a ``generate_artifacts`` run saves onto the first
    Block's still-present ``temp.onnx.data`` — and onnx >= 1.16 raises
    ``FileExistsError: External data file exists in temp.onnx.data`` instead of overwriting, killing
    artifact generation.

    ORT 1.23 and onnx 1.18 are the pairing recorded in ``third_party/onnxruntime/manifest.json``, so
    this is not version drift we can pin away; Gate 0.3's smoke never hit it because its model is small
    enough that ``has_path`` is false and the whole branch is skipped. Restoring the pre-1.16 overwrite
    semantics for exactly this call is the narrowest fix available to us.
    """
    original_save = onnx.save_model

    def _save(proto, f, *args, **kwargs):
        location = kwargs.get("location")
        if location and kwargs.get("save_as_external_data") and isinstance(f, (str, os.PathLike)):
            stale = os.path.join(os.path.dirname(os.path.abspath(str(f))), str(location))
            if os.path.exists(stale):
                os.remove(stale)
        return original_save(proto, f, *args, **kwargs)

    onnx.save_model = _save
    onnx.save = _save
    try:
        yield
    finally:
        onnx.save_model = original_save
        onnx.save = original_save


def gen_artifacts(
    train_dir,
    artifact_dir="artifacts",
    model_name="quant_model.onnx",
    train_cfg_file="training_config.json",
    training_config={},
):
    """
    Generates the training artifacts from the provided model and directory.
    Needs the training configuration provided along with the model in the same directory.
    """
    onnx_model_path = os.path.join(train_dir, model_name)
    train_cfg_path = os.path.join(train_dir, train_cfg_file)

    onnx.checker.check_model(onnx_model_path, full_check=True)
    onnx_model = onnx.load(onnx_model_path)

    params = {}
    with open(train_cfg_path, encoding="utf-8") as f:
        params = json.load(f)

    requires_grad = []
    frozen_params = []

    for param in onnx_model.graph.initializer:
        trainable = any(rqp in param.name for rqp in params["requires_grad"]) and not _is_quant_companion(
            param.name
        )
        (requires_grad if trainable else frozen_params).append(param.name)

    del onnx_model
    gc.collect()

    # Generate the training artifacts
    with _onnx_external_data_overwrite():
        artifacts.generate_artifacts(
            onnx_model_path,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            # We don't need to provide a loss function, as the loss is already
            # computed from the PyTorch Transformer model
            # In the case of inference model, we don't need it
            # loss = CausalLMCE(),
            optimizer=artifacts.OptimType.AdamW,
            artifact_directory=artifact_dir,
        )

    extended_training_config = {
        "requires_grad": requires_grad,
        "peft_mapping": params["peft_mapping"],
        "rank": params["rank"],
        "alpha": params["alpha"],
        "peft_target": params["peft_target"],
        "trainable_parameter_count": params["trainable_parameter_count"],
        **training_config,
    }

    # Export training configs
    with open(f"{artifact_dir}/{train_cfg_file}", "w", encoding="utf-8") as f:
        json.dump(extended_training_config, f, ensure_ascii=False)

    return extended_training_config


def onnx_checktrain(
    model_dir,
    model_id,
    export_inference=False,
    test_inference=False,
    test_evaluate=False,
    transfer_weights=False,
    inference_model_path="inference_model.onnx",
    max_sequence_length=100,
):
    """
     Checks the model if the outputs are training correctly as well as evaluation.
     Exports model for inference if needed or transfers the weights to an already existing inference model.

    - `test_inference` - runs the model through an example prompt
    - `test_evaluate` - tests the model evaluation
    - `transfer_weights` - instead of exporting for inference, we only copy the subset of updated weights to an already created inference model after training
    - `inference_model_path` - model path of an already existing inference model or one to create
    - `max_sequence_length` - length of text generation sequence
    """

    state = CheckpointState.load_checkpoint(f"{model_dir}/checkpoint")

    sess_options = SessionOptions()
    sess_options.enable_profiling = False
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

    model = Module(
        f"{model_dir}/training_model.onnx",
        state,
        f"{model_dir}/eval_model.onnx",
        session_options=sess_options,
    )
    optimizer = Optimizer(f"{model_dir}/optimizer_model.onnx", model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_settings().require_hf_token())

    # Create dummy input
    tokenizer.pad_token_id = 0
    inputs = tokenizer(
        ["This is a test, hello from world.", "This is a test, hello to world."],
        return_tensors="pt",
        padding=True,
    )

    input_ids = inputs["input_ids"].numpy()
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    labels = inputs["input_ids"].clone().numpy()

    labels = np.copy(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Optionally, set the last token to -100 to ignore it in the loss

    inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "position_ids": position_ids,
        "labels": labels,
    }

    start_train_time = time.time()
    model.train()
    forward = model(*inputs.values())
    optimizer.step()
    model.lazy_reset_grad()
    end_train_time = time.time()

    print(f"[INFO] Training loss result: {forward[0]}")
    print(f"[INFO] Training time: {end_train_time - start_train_time} s")

    # TODO: Doesn't work with inputs?
    if test_evaluate:
        model.eval()
        forward = model(input_ids, labels)
        print("Evaluation results:")
        print(forward[0])

    exclude_nodes = ["loss"]

    if transfer_weights:
        inference_model_path = os.path.join(model_dir, inference_model_path)
        onnx_transfer_trained_weights(state, inference_model_path)
        del model
        del state
        del optimizer
        gc.collect()
    elif export_inference:
        # Model inference: we want to get only logits and hidden states for decoding
        model.export_model_for_inferencing(
            f"{model_dir}/{inference_model_path}",
            [out_name for out_name in model.output_names() if out_name not in exclude_nodes],
        )

        del model
        del state
        del optimizer
        gc.collect()
        # Load and test inference if needed
        if test_inference:
            onnx_infer(
                model_id,
                f"{model_dir}/{inference_model_path}",
                with_past=False,
                max_length=max_sequence_length,
            )


def force_dequantize_external_and_save(model, output_path, external_data_filename=None):
    """
    Force DequantizeLinear x_scale and x_zero_point tensors to be external and save the model.

    Args:
        model: Loaded ONNX model (onnx.ModelProto)
        output_path: Path where to save the modified model
        external_data_filename: Name of external data file (optional, defaults to model_name.onnx.data)

    Returns:
        model : The new inference model with external initializers
    """
    if external_data_filename is None:
        model_name = os.path.splitext(os.path.basename(output_path))[0]
        external_data_filename = f"{model_name}.onnx.data"

    forced_count = 0

    # Force DequantizeLinear tensors to external manually BEFORE converting everything else
    for initializer in model.graph.initializer:
        # Check if initializer name ends with x_scale or x_zero_point
        # TODO: Hard coded
        is_dequant_tensor = (initializer.name.startswith("model.layers")) and (
            initializer.name.endswith("MatMul.weight")
            or initializer.name.endswith("MatMul.weight_zero_point")
            or initializer.name.endswith("MatMul.weight_scale")
        )

        if is_dequant_tensor and initializer.data_location != onnx.TensorProto.EXTERNAL:
            print(f"Forcing DequantizeLinear tensor to external: {initializer.name}")

            # Convert tensor data to raw_data format first
            # This ensures the tensor has raw_data field that set_external_data expects
            if not initializer.HasField("raw_data"):
                # Convert using numpy helper to preserve exact data type and format
                tensor_array = onnx.numpy_helper.to_array(initializer)

                # Clear all existing data fields first
                initializer.ClearField("float_data")
                initializer.ClearField("int32_data")
                initializer.ClearField("int64_data")
                initializer.ClearField("string_data")
                initializer.ClearField("uint64_data")
                initializer.ClearField("double_data")
                initializer.ClearField("raw_data")

                # Set raw_data with the binary representation
                initializer.raw_data = tensor_array.tobytes()

            # Now use the proper ONNX function to set external data
            onnx.external_data_helper.set_external_data(tensor=initializer, location=external_data_filename)
            forced_count += 1

            # Now use the proper ONNX function to set external data
            set_external_data(tensor=initializer, location=external_data_filename)

            forced_count += 1

    # Convert all OTHER tensors to external data (this won't affect already external ones)
    convert_model_to_external_data(
        model, location=external_data_filename, size_threshold=0, all_tensors_to_one_file=True
    )

    # Write external data to file
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."

    return write_external_data_tensors(model, output_dir)


def get_all_metadata_from_onnx(model_path):
    """
    Extract all metadata properties from ONNX model.

    Args:
        model_path (Path or str): Path to the .onnx model file

    Returns:
        dict: Dictionary containing all metadata properties from both model and graph levels.
              Graph-level metadata will override model-level metadata if keys conflict.
    """
    import onnx

    model = onnx.load(str(model_path))
    metadata = {}

    # Read model-level metadata first
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value

    # Read graph-level metadata (will override model-level if keys conflict)
    for prop in model.graph.metadata_props:
        metadata[prop.key] = prop.value

    return metadata


def onnx_export_dummy_model(model_output="tokenizer.onnx"):
    """
    Creates a fake dummy model for the tokenization process with GenAI.
    """

    # Define the input and output tensor types
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [None, None])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None])

    node = helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])

    graph = helper.make_graph(
        nodes=[node],
        name="identity_graph",
        inputs=[input_ids],  # No inputs
        outputs=[logits],  # No outputs
    )

    # Create an empty model
    model = helper.make_model(
        graph,
        producer_name="onnx-empty-model",
        opset_imports=[helper.make_opsetid("", 14)],  # Adjust the opset version as needed
    )

    onnx.checker.check_model(model)

    onnx.save_model(model, model_output, save_as_external_data=False)


def onnx_infer(model_id, model_path="inf_model_onnx_gemma_nonq.onnx", with_past=False, max_length=100):
    """
    Test inference with the provided ONNX inference model. Model needs to have inputs:
    - input_ids
    - attention_mask
    - position_ids
    """

    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_settings().require_hf_token())
    config = AutoConfig.from_pretrained(model_id)

    prompt = "Hello, this is a message for the world. How is your day?"

    print(
        generate_tokens_onnx(prompt, tokenizer, session, config, with_past=with_past, max_length=max_length)
    )


def onnx_segment_weights(model_path, output_path):
    """
    Save the model with external data.
    """
    m = onnx.load(model_path)
    onnx.save(m, output_path, save_as_external_data=True, all_tensors_to_one_file=False)


def onnx_transfer_trained_weights(state: CheckpointState, inference_model):
    """
    Transfer the updated weights from the checkpoint traning session to the inference model and save it.
    """

    updated_weights = {}

    # Extract all updated parameters data
    for param_name, parameter in state.parameters:
        if parameter.requires_grad:
            print(param_name)
            updated_weights[param_name] = parameter.data

    # In on-device training scenario we would also save the checkpoint with gradients
    # and delete the training session from memory
    state.save_checkpoint(state)

    # Load the inference model
    onnx_inference_model = onnx.load(inference_model)

    # Overwrite the parameters that require gradient
    for i, initializer in enumerate(onnx_inference_model.graph.initializer):
        if initializer.name in updated_weights:
            W = numpy_helper.to_array(initializer)
            if not np.array_equal(W, updated_weights[initializer.name]):
                print(f"Overwriting {initializer.name}, weights changed...")
                new_tensor = numpy_helper.from_array(updated_weights[initializer.name], initializer.name)
                initializer.CopyFrom(new_tensor)
                new_numpy = numpy_helper.to_array(onnx_inference_model.graph.initializer[i])
                if np.array_equal(new_numpy, updated_weights[initializer.name]):
                    print("Copied successfully")
            else:
                print(
                    f"Weights were not changed, but the training session was performed on these weights?\nParameter: {initializer.name}"
                )

    onnx.save(onnx_inference_model)
    del onnx_inference_model
    gc.collect()


def gen_genai(
    model_id,
    model_path,
    training_config,
    new_model_name,
    new_model_path,
    weight_input=True,
    include_metadata=True,
    large_model=False,
    test_generation=False,
    test_generation_config={},
    force_external=False,
    check_model=True,
    opset_version=18,
):
    """
    Creates a GenAI compatible ONNX graph or a custom inference graph.

    - `training_config` - training configuration json
    - `new_model_path` - path to new inference model
    - `large_model` - if model is larger than 2GB
    """

    model = onnx.load(model_path)
    model_trainable_weights = {}

    new_model_namepath = f"{new_model_path}/{new_model_name}.onnx"

    if weight_input and training_config:
        requires_grad_layers = training_config["requires_grad"]

        # Extract initializers from the model
        initializers = {init.name: init for init in model.graph.initializer}

        # Create new input nodes for the specified initializers
        new_inputs = []
        for name in requires_grad_layers:
            if name in initializers:
                initializer = initializers[name]
                # Create a new input node
                new_input = helper.make_tensor_value_info(name, initializer.data_type, initializer.dims)
                new_inputs.append(new_input)

                # Store them for the testing generation input
                model_trainable_weights[name] = numpy_helper.to_array(initializer)

                # Remove initializer since it becomes the input
                model.graph.initializer.remove(initializer)

        # Create a new list of inputs (existing inputs + new inputs for specified initializers)
        new_graph_inputs = list(model.graph.input) + new_inputs

        # Remove the specified initializers from the model
        new_initializers = [init for init in model.graph.initializer if init.name not in requires_grad_layers]

        # Create a new graph with updated inputs and removed initializers
        new_graph = helper.make_graph(
            nodes=model.graph.node,
            name=model.graph.name,
            inputs=new_graph_inputs,
            outputs=model.graph.output,
            initializer=new_initializers,
        )

        # Create a new model with the modified graph
        model = helper.make_model(new_graph, opset_imports=[helper.make_operatorsetid("", opset_version)])

    if include_metadata:
        print("[INFO] Adding metadata to the inference model...")
        config = AutoConfig.from_pretrained(model_id)

        num_kv_heads = (
            config.num_key_value_heads
            if hasattr(config, "num_key_value_heads")
            else config.num_attention_heads
        )
        head_size = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        num_layers = config.num_hidden_layers

        # Add custom metadata
        model.metadata_props.append(onnx.StringStringEntryProto(key="model_id", value=str(model_id)))
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="max_context_length", value=str(config.max_position_embeddings))
        )
        model.metadata_props.append(onnx.StringStringEntryProto(key="head_dim", value=str(head_size)))
        model.metadata_props.append(onnx.StringStringEntryProto(key="num_kv_heads", value=str(num_kv_heads)))
        model.metadata_props.append(onnx.StringStringEntryProto(key="num_layers", value=str(num_layers)))

    # We set size threshold to 0 to force all tensors to be saved as externally and later to be replaced easily in inference session
    if force_external or training_config:
        print("[INFO] Forcing external initializers...")
        model = force_dequantize_external_and_save(model, new_model_namepath)
    print("[INFO] Saving the model...")
    onnx.save(
        model,
        new_model_namepath,
        save_as_external_data=True,
        location=f"{new_model_name}.onnx.data",
        size_threshold=0,
    )

    if large_model:
        # Wait so it finishes writing to disk
        time.sleep(20)
        print("[INFO] Writing large model to disk...")
        if check_model:
            onnx.checker.check_model(new_model_namepath, full_check=True)
    elif check_model:
        onnx.checker.check_model(model, full_check=True)

    print("[INFO] Saved GenAI inference model.")

    if test_generation:
        session = InferenceSession(new_model_namepath, providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_settings().require_hf_token())
        config = AutoConfig.from_pretrained(model_id)
        generate_tokens_onnx(
            tokenizer,
            session,
            config,
            model_trainable_weights,
            with_past=True,
            with_weight_input=weight_input,
            **test_generation_config,
        )


def get_layers_with_grad(model):
    """
    Function to get all layers that require gradients
    """
    layers_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layers_with_grad.append(name)
    return layers_with_grad


class CausalLMCE(onnxblock.Block):
    def __init__(self):
        super().__init__()
        # Assumes classes is the last dimension
        # e.g., predictions: (num_examples, num_classes) -> labels: (num_examples,)
        # or predictions: (batch_size, seq_len, vocab) -> labels: (batch_size, seq_len)
        self._loss1 = onnxblock.loss.CrossEntropyLoss()

    def build(self, logits, *args):
        return self._loss1(logits)


def convert_pipeline(
    model_id,
    peft_method,
    train_model_name,
    train_dir,
    inference_model_name,
    inference_dir,
    embedding_model_path,
    build_dir,
    gen_train_artifacts=False,
    gen_inference_artifacts=False,
    gen_rag_config=False,
    test_training=True,
    test_eval=True,
    test_generation=True,
    inference_export_config={},
    test_generation_config={},
    inference_config={},
    train_config={},
    rag_config={},
    export_tokenizer=True,
    export_dataset=True,
    export_inference_config=True,
    export_merger=True,
    delete_models=False,
    config_file_path="config.yml",
    **kwargs,
):
    """
    ONNX conversion for training, inference, merger and embedding artifacts.
    Creates a build folder with train and inference subfolders each with models needed for tasks.
    """

    # TODO: Infer from the given model
    large_model = True

    try:
        # Create the base directory if it doesn't exist
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        train_path = os.path.join(build_dir, "train")
        inference_path = os.path.join(build_dir, "inference")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(inference_path, exist_ok=True)

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

    extended_train_config = None
    if gen_train_artifacts:
        # Add peft method to training config
        train_config["peftMethod"] = peft_method
        train_config["modelId"] = model_id

        # Generate training artifacts
        extended_train_config = gen_artifacts(
            train_dir=train_dir,
            artifact_dir=f"{build_dir}/train",
            model_name=train_model_name,
            training_config=train_config,
        )
        print("[INFO] Generated training artifacts.")

    if test_training:
        onnx_checktrain(model_dir=f"{build_dir}/train", model_id=model_id, test_evaluate=test_eval)
        print("[INFO] Training check completed.")

    # Export native inference model
    if gen_inference_artifacts and inference_config["type"] == "native":
        gen_genai(
            model_id=model_id,
            model_path=f"{inference_dir}/{inference_model_name}",
            training_config=extended_train_config,
            new_model_name=inference_export_config["output_inference_model"],
            new_model_path=f"{build_dir}/inference",
            large_model=large_model,
            # test_generation=test_generation,
            weight_input=inference_export_config["weight_input"],
            include_metadata=inference_export_config["include_metadata"],
            opset_version=inference_export_config["opset"],
            # test_generation_config=test_generation_config,
            check_model=inference_export_config["check_model"],
            force_external=inference_export_config["force_external_initializers"],
        )
        print("[INFO] Generated the artifact inference model graph.")

        # Move the rest of the files
        move_files_excluding(inference_dir, f"{build_dir}/inference", exclude_files=[inference_model_name])
        print(f"[INFO] Moved the rest of generation configuration files to: {build_dir}/inference")

    # NOTE: If the ONNX Runtime versions do not match, you need to use inference.builder to build inference model
    elif gen_inference_artifacts and inference_config["type"] == "genai":
        raise ValueError("GenAI inference graph currently not supported.")

    # Export generation configuration
    if export_inference_config:
        with open(f"{build_dir}/inference/generation_config.json", "w", encoding="utf-8") as f:
            json.dump(inference_config, f, ensure_ascii=False)

    # Export tokenizer if needed
    if export_tokenizer:
        export_tokenizer_config(model_id, build_dir, get_settings().require_hf_token())

    if export_dataset:
        if "taskName" not in train_config or "trainFile" not in train_config:
            print("[WARNING] taskName or trainFile not defined in train_config!")
        elif train_config["taskName"] not in TASK_NAME_TO_DATASET:
            print("[WARNING] taskName unknown!")
        else:
            load_and_save_dataset(
                TASK_NAME_TO_DATASET[train_config["taskName"]],
                train_path,
                train_config["trainFile"],
                split="train",
                max_dataset_length=train_config["maxDatasetLength"],
            )

    if export_merger:
        # Registry-driven merger emit (#9): build_merger_model via resolve_merger, descriptive filenames
        # recorded in the handoff map's mergerModels — replaces the four hand-picked factory calls +
        # the peft_method == "lora"/"mars" string dispatch. resolve_merger fails closed on unknown method.
        train_dir = f"{build_dir}/train"
        quant_out = inference_export_config["quantized_merged_output"]
        method = PEFTMethod(peft_method)
        if method is PEFTMethod.MARS:
            build_cfg = kwargs["train_builder_config"].get(peft_method, {})
            keep_fp_in = build_cfg.get("optimization_level", 99) <= 1
            emit_merger_models(
                train_dir,
                PEFTMethod.MARS,
                quant_out=quant_out,
                quant_ins=((True, False) if keep_fp_in else (True,)),
            )
            # MARS packages also carry LoRA mergers for their non-MARS layers (device mixes per-layer).
            emit_merger_models(train_dir, PEFTMethod.LORA, quant_out=quant_out, quant_ins=(True, False))
        else:
            emit_merger_models(train_dir, method, quant_out=quant_out, quant_ins=(True, False))

    if gen_rag_config:
        embedding_model_metadata = get_all_metadata_from_onnx(embedding_model_path)
        # Get model id from metadata and export tokenizer in embedding/tokenizer
        export_tokenizer_config(
            embedding_model_metadata["model_id"], f"{build_dir}/embedding/", get_settings().require_hf_token()
        )

        # Move onnx embedding model
        move_onnx_model(embedding_model_path, f"{build_dir}/embedding/", delete=False)

        # Export embedding config
        with open(f"{build_dir}/embedding/rag_config.json", "w", encoding="utf-8") as f:
            # Update embedding config with correct information
            if "embedding_dim" in embedding_model_metadata:
                rag_config["embeddingDimension"] = embedding_model_metadata["embedding_dim"]

            json.dump(rag_config, f, ensure_ascii=False)

    # Clean the generated models if needed
    if delete_models:
        delete_directory(inference_dir)
        delete_directory(train_dir)
        print("[INFO] Deleted previously generated training and inference models.")


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


def load_config_from_file(config_file: str):
    """Load configurations from a YAML file into a dictionary."""
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Converting the given ONNX models into a ONNX artifacts for on-device training and inference.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--model_id", type=str, help="Identifier for the model to be converted.")
    parser.add_argument("--build_path", type=str, help="Path to convert the artifact models.")
    parser.add_argument("--inference_model", type=str, help="Name of the inference model.")
    parser.add_argument("--inference_dir", type=str, help="Path to the inference model directory.")
    parser.add_argument("--training_model", type=str, help="Name of the training model.")
    parser.add_argument("--embedding_model", type=str, help="Path to the embedding model.")
    parser.add_argument("--training_dir", type=str, help="Path to the training model directory.")
    parser.add_argument(
        "--gen_train_artifacts",
        type=bool,
        default=False,
        help="Whether to generate training artifacts. Default is False.",
    )
    parser.add_argument(
        "--gen_inference_artifacts",
        type=bool,
        default=False,
        help="Whether to generate inference artifacts. Default is False.",
    )
    parser.add_argument(
        "--gen_rag_config",
        type=bool,
        default=False,
        help="Whether to generate embedding artifacts. Default is False.",
    )
    parser.add_argument(
        "--test_training",
        type=bool,
        default=True,
        help="Whether to test training capabilities. Default is True.",
    )
    parser.add_argument(
        "--test_eval",
        type=bool,
        default=True,
        help="Whether to test evaluation capabilities. Default is True.",
    )
    parser.add_argument(
        "--delete_models", type=bool, default=True, help="Deletes the previously generated models."
    )
    parser.add_argument(
        "--export_tokenizer",
        type=bool,
        default=True,
        help="Exports tokenizer files and config in seperate dir.",
    )
    parser.add_argument(
        "--export_inference_config",
        type=bool,
        default=True,
        help="Exports inference configuration config in build dir.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments.",
    )
    parser.add_argument(
        "--export_dataset", type=bool, default=True, help="Exports dataset file in train dir."
    )
    parser.add_argument(
        "--export_merger", type=bool, default=True, help="Exports the merger model for adapters."
    )
    parser.add_argument(
        "--inference_export_config",
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
        "--inference_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ... TODO add description
            """),
    )
    parser.add_argument(
        "--train_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ... TODO add description
            """),
    )
    parser.add_argument(
        "--rag_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ... TODO add description
            """),
    )
    # parser.add_argument(
    #    "--test_generation_config",
    #    type=str,
    #    nargs="*",
    #    metavar="KEY=VALUE",
    #    default=[],
    #    help=textwrap.dedent("""\
    #     Key value pairs for various options. Currently supports:
    #        prompt = Hello... : Prompt for test generation. If using chatpot template setting, please provide the chat template format as well.
    #        decode_between = true : Whether to decode the text while it's generating.
    #        max_length = 100 : Max length of test sequence to generate.
    #        sampling = top_k : Sampling method. Should support topk and topp
    #        temperature = 0.7 : Temperature for sampling
    #        top_k = 10 : Top K for sampling
    #        top_p = 0.3 : Top P for sampling
    #        """
    #        )
    # )
    args = parser.parse_args()

    user_inference_config = {}
    default_user_inference_config = {
        "type": "genai",  # "normal", "genai"
        "weight_input": False,  # Whether to include trainable weights as model input
        "test_inference": True,  # Whether to perform inference / generation test on the inference exported model
        "include_metadata": True,  # Whether to include the model metadata
        "output_inference_name": "genai_inference",  # The new model name
        "opset_version": 20,
        "gen_config_file": "genai_config.json",
    }

    # user_test_generation_config = {}
    # default_test_generation_config = {
    #   "prompt": "Hello, this is a message for the world. How is your day?", # Prompt for test generation
    #        "decode_between": True, # Whether to decode the text while it's generating
    #        "max_length" : 100, # Max length of test sequence to generate
    #       "sampling": "topk", # Sampling method
    #        "temperature": 0.7, # Temperature for sampling
    #        "top_k": 10, # Top K for sampling
    # }

    extra_args = {}

    config_dict = None

    if args.config:
        config_dict = load_config_from_file(args.config)

        # Specific
        args.model_id = config_dict[TRAIN_CONFIG]["model_id"]
        args.peft_method = config_dict[TRAIN_CONFIG]["train_method"]

        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_CONFIG].items():
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)

        args.training_dir = config_dict[TRAIN_CONFIG]["output"]
        args.inference_dir = config_dict[INFERENCE_CONFIG]["output"]

        extra_args["train_builder_config"] = config_dict[TRAIN_CONFIG]
    else:
        user_inference_config = parse_extra_options(args.inference_config)
        args.inference_export_config = {**default_user_inference_config, **user_inference_config}
        # user_test_generation_config = parse_extra_options(args.test_generation_config)
        # args.test_generation_coinfig = {**default_test_generation_config, **user_test_generation_config}

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = parse_arguments()

    print(f"{ARTIFACT_CONFIG} arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    convert_pipeline(
        model_id=args.model_id,
        peft_method=args.peft_method,
        train_model_name=args.training_model,
        train_dir=args.training_dir,
        inference_model_name=args.inference_model,
        inference_dir=args.inference_dir,
        embedding_model_path=args.embedding_model,
        build_dir=args.build_path,
        gen_inference_artifacts=args.gen_inference_artifacts,
        gen_train_artifacts=args.gen_train_artifacts,
        gen_rag_config=args.gen_rag_config,
        test_training=args.test_training,
        test_eval=args.test_eval,
        # We avoid testing generation in this script due to package conflicts
        # test_generation=args.test_generation,
        # test_generation_config=args.test_generation_config
        inference_export_config=args.inference_export_config,
        export_tokenizer=args.export_tokenizer,
        inference_config=args.inference_config,
        train_config=args.train_config,
        rag_config=args.rag_config,
        export_dataset=args.export_dataset,
        export_inference_config=args.export_inference_config,
        export_merger=args.export_merger,
        delete_models=args.delete_models,
        config_file_path=args.config,
        **extra_args,
    )
