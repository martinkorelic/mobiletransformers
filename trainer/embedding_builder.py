# DECOMPOSE(#5): fold encoder/embedding export into src/mobiletransformers/export behind the
# task/architecture registry (#6), consumed by encoder support (#33). ~27 KB.
import json
import onnx
from onnx import helper, TensorProto
from huggingface_hub import hf_hub_download

def add_pooling_to_onnx_model(model, model_id, output_model_path):
    """
    Add pooling operations to an ONNX model based on sentence-transformer configuration.
    
    Args:
        onnx_model_path (str): Path to the original ONNX model
        model_id (str): HuggingFace model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        output_model_path (str): Path where the modified ONNX model will be saved
    
    Returns:
        str: Path to the modified ONNX model
    """
    
    # Load pooling configuration from HuggingFace Hub
    pooling_config = load_pooling_config_from_hub(model_id)
    
    # Add pooling operations to the model
    modified_model = add_pooling_operations(model, pooling_config)
    
    # Validate the modified model before saving
    try:
        onnx.checker.check_model(modified_model)
        print("✓ Model validation passed - ONNX graph is valid")
    except onnx.checker.ValidationError as e:
        print(f"✗ Model validation failed: {e}")
        raise ValueError(f"Modified ONNX model is invalid: {e}")
    
    # Additional shape inference to ensure output shapes are correct
    try:
        modified_model = onnx.shape_inference.infer_shapes(modified_model)
        print("✓ Shape inference completed successfully")
    except Exception as e:
        print(f"⚠ Warning: Shape inference failed: {e}")
        print("Model may still work, but output shapes might not be fully inferred")
    
    # Add some metadata to model
    word_embedding_dim = pooling_config.get("word_embedding_dimension", None)

    if word_embedding_dim:
        # Add metadata to model level
        model_metadata_entry = onnx.StringStringEntryProto()
        model_metadata_entry.key = "embedding_dim"
        model_metadata_entry.value = str(model_id)
        model.metadata_props.append(model_metadata_entry)
        
        # Add metadata to graph level
        graph_metadata_entry = onnx.StringStringEntryProto()
        graph_metadata_entry.key = "embedding_dim"
        graph_metadata_entry.value = str(model_id)
        model.graph.metadata_props.append(graph_metadata_entry)

    # Save the modified model
    onnx.save(modified_model, output_model_path)
    print(f"✓ Model saved successfully to: {output_model_path}")
    
    # Print summary of what was added
    print_pooling_summary(pooling_config)
    
    return output_model_path

def print_pooling_summary(pooling_config):
    """Print a summary of the pooling configuration that was applied."""
    print("\n" + "="*50)
    print("POOLING CONFIGURATION SUMMARY")
    print("="*50)
    
    word_dim = pooling_config.get("word_embedding_dimension", "unknown")
    print(f"Word embedding dimension: {word_dim}")
    
    active_modes = []
    if pooling_config.get("pooling_mode_cls_token", False):
        active_modes.append("CLS token")
    if pooling_config.get("pooling_mode_mean_tokens", False):
        active_modes.append("Mean tokens")
    if pooling_config.get("pooling_mode_max_tokens", False):
        active_modes.append("Max tokens")
    if pooling_config.get("pooling_mode_mean_sqrt_len_tokens", False):
        active_modes.append("Mean sqrt length")
    
    print(f"Active pooling modes: {', '.join(active_modes)}")
    
    if len(active_modes) > 1:
        total_dim = word_dim * len(active_modes) if isinstance(word_dim, int) else "unknown"
        print(f"Final output dimension: {total_dim} (concatenated)")
    else:
        print(f"Final output dimension: {word_dim}")
    
    print(f"Input shape: [batch_size, sequence_length, {word_dim}]")
    final_dim = word_dim * len(active_modes) if isinstance(word_dim, int) and len(active_modes) > 0 else word_dim
    print(f"Output shape: [batch_size, {final_dim}]")
    print("="*50)


def load_pooling_config_from_hub(model_id):
    """
    Load pooling configuration from HuggingFace Hub without downloading the full model.
    
    Args:
        model_id (str): HuggingFace model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        
    Returns:
        dict: Pooling configuration
    """
    pooling_config = None
    
    try:
        # First, try to download modules.json to get the exact module structure
        modules_path = hf_hub_download(
            repo_id=model_id,
            filename="modules.json",
            cache_dir=None,  # Use default cache
        )
        
        with open(modules_path, 'r') as f:
            modules = json.load(f)
        
        # Look for pooling module in modules.json
        for module in modules:
            if module.get("type") == "sentence_transformers.models.Pooling":
                pooling_dir = module.get("path", "")
                
                # Download the pooling config
                try:
                    config_path = hf_hub_download(
                        repo_id=model_id,
                        filename=f"{pooling_dir}/config.json",
                        cache_dir=None,
                    )
                    
                    with open(config_path, 'r') as f:
                        pooling_config = json.load(f)
                    break
                except Exception:
                    continue  # Try next module if this one fails
                    
    except Exception:
        # modules.json not found or failed to download
        pass
    
    # Fallback: try standard locations if modules.json approach didn't work
    if pooling_config is None:
        config_files = [
            "1_Pooling/config.json",  # Standard sentence-transformers structure
            "2_Pooling/config.json",  # Alternative numbering
            "pooling/config.json",    # Alternative naming
        ]
        
        for config_file in config_files:
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=config_file,
                    cache_dir=None,
                )
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Check if this config contains pooling information
                    if any(key.startswith('pooling_mode_') for key in config.keys()):
                        pooling_config = config
                        break
            except Exception:
                continue  # File doesn't exist, try next
    
    # If still no config, try to get from sentence_transformers_config.json
    if pooling_config is None:
        try:
            st_config_path = hf_hub_download(
                repo_id=model_id,
                filename="sentence_transformers_config.json",
                cache_dir=None,
            )
            
            with open(st_config_path, 'r') as f:
                st_config = json.load(f)
                # Sometimes pooling info is embedded here
                if "pooling" in st_config:
                    pooling_config = st_config["pooling"]
        except Exception:
            pass
    
    # Last resort: try to infer from main config.json
    if pooling_config is None:
        try:
            main_config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=None,
            )
            
            with open(main_config_path, 'r') as f:
                config = json.load(f)
                # Check if this config contains pooling information
                if any(key.startswith('pooling_mode_') for key in config.keys()):
                    pooling_config = config
        except Exception:
            pass
    
    if pooling_config is None:
        # Try to get word_embedding_dimension from transformer config as fallback
        try:
            transformer_config_path = hf_hub_download(
                repo_id=model_id,
                filename="0_Transformer/config.json",
                cache_dir=None,
            )
            
            with open(transformer_config_path, 'r') as f:
                transformer_config = json.load(f)
                hidden_size = transformer_config.get("hidden_size", 384)
                
                # Create default pooling config (mean pooling)
                pooling_config = {
                    "word_embedding_dimension": hidden_size,
                    "pooling_mode_cls_token": False,
                    "pooling_mode_mean_tokens": True,
                    "pooling_mode_max_tokens": False,
                    "pooling_mode_mean_sqrt_len_tokens": False
                }
                
                print(f"Warning: No pooling config found for {model_id}. "
                      f"Using default mean pooling with dimension {hidden_size}")
                
        except Exception:
            raise FileNotFoundError(
                f"Could not find pooling configuration for model '{model_id}'. "
                f"Please check if this is a valid sentence-transformers model or "
                f"manually provide the pooling configuration."
            )
    
    return pooling_config


def add_pooling_operations(model, pooling_config):
    """
    Add pooling operations to the ONNX model graph.
    
    Args:
        model: ONNX model
        pooling_config (dict): Pooling configuration
        
    Returns:
        Modified ONNX model
    """
    graph = model.graph
    
    # Find the last_hidden_state output
    last_hidden_state_output = None
    for output in graph.output:
        if output.name == "last_hidden_state":
            last_hidden_state_output = output
            break
    
    if last_hidden_state_output is None:
        raise ValueError("Could not find 'last_hidden_state' output in the model")
    
    # Get the word embedding dimension
    word_embedding_dim = pooling_config.get("word_embedding_dimension", 384)
    
    # Create pooling operations based on configuration
    pooling_outputs = []
    nodes_to_add = []
    
    # We need attention_mask as input for proper pooling
    # Add attention_mask as a graph input if not already present
    attention_mask_input = None
    for input_info in graph.input:
        if input_info.name == "attention_mask":
            attention_mask_input = input_info
            break
    
    if attention_mask_input is None:
        # Add attention_mask as input
        attention_mask_input = helper.make_tensor_value_info(
            "attention_mask",
            TensorProto.INT64,
            ["batch_size", "sequence_length"]
        )
        graph.input.append(attention_mask_input)
    
    # CLS token pooling
    if pooling_config.get("pooling_mode_cls_token", False):
        cls_output = add_cls_pooling(graph, "last_hidden_state", nodes_to_add)
        pooling_outputs.append(cls_output)
    
    # Mean token pooling
    if pooling_config.get("pooling_mode_mean_tokens", False):
        mean_output = add_mean_pooling(graph, "last_hidden_state", "attention_mask", nodes_to_add)
        pooling_outputs.append(mean_output)
    
    # Max token pooling
    if pooling_config.get("pooling_mode_max_tokens", False):
        max_output = add_max_pooling(graph, "last_hidden_state", "attention_mask", nodes_to_add)
        pooling_outputs.append(max_output)
    
    # Mean sqrt length pooling
    if pooling_config.get("pooling_mode_mean_sqrt_len_tokens", False):
        mean_sqrt_output = add_mean_sqrt_len_pooling(graph, "last_hidden_state", "attention_mask", nodes_to_add)
        pooling_outputs.append(mean_sqrt_output)
    
    # If multiple pooling modes are enabled, concatenate them
    if len(pooling_outputs) > 1:
        final_output = add_concatenation(graph, pooling_outputs, nodes_to_add)
        final_dim = word_embedding_dim * len(pooling_outputs)
    elif len(pooling_outputs) == 1:
        final_output = pooling_outputs[0]
        final_dim = word_embedding_dim
    else:
        raise ValueError("No pooling mode is enabled in the configuration")
    
    # Add all nodes to the graph
    graph.node.extend(nodes_to_add)
    
    # Remove the original last_hidden_state from outputs
    graph.output.remove(last_hidden_state_output)
    
    # Add the new embedding output
    embedding_output = helper.make_tensor_value_info(
        "embedding",
        TensorProto.FLOAT,
        ["batch_size", final_dim]
    )
    graph.output.append(embedding_output)
    
    # Rename the final output to "embedding"
    if final_output != "embedding":
        rename_node = helper.make_node(
            "Identity",
            inputs=[final_output],
            outputs=["embedding"],
            name="rename_to_embedding"
        )
        graph.node.append(rename_node)
    
    return model


def add_cls_pooling(graph, input_name, nodes_to_add):
    """Add CLS token pooling (first token)."""
    # Extract the first token (CLS token) from the sequence
    # Input shape: [batch_size, sequence_length, hidden_dim]
    # Output shape: [batch_size, hidden_dim]
    
    # Create constant for indices [0] to select first token
    indices_tensor = helper.make_tensor(
        name="cls_indices",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]
    )
    indices_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["cls_indices_const"],
        value=indices_tensor,
        name="cls_indices_constant"
    )
    nodes_to_add.append(indices_node)
    
    # Use Gather to extract first token
    gather_node = helper.make_node(
        "Gather",
        inputs=[input_name, "cls_indices_const"],
        outputs=["cls_pooled"],
        axis=1,  # Gather along sequence dimension
        name="cls_gather"
    )
    nodes_to_add.append(gather_node)
    
    # Create axes tensor for squeeze operation
    squeeze_axes_tensor = helper.make_tensor(
        name="squeeze_axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Squeeze dimension 1
    )
    squeeze_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes_const"],
        value=squeeze_axes_tensor,
        name="squeeze_axes_constant"
    )
    nodes_to_add.append(squeeze_axes_node)
    
    # Squeeze to remove the sequence dimension
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["cls_pooled", "squeeze_axes_const"],
        outputs=["cls_pooled_squeezed"],
        name="cls_squeeze"
    )
    nodes_to_add.append(squeeze_node)
    
    return "cls_pooled_squeezed"


def add_mean_pooling(graph, input_name, attention_mask_name, nodes_to_add):
    """Add mean token pooling with attention mask."""
    # Convert attention mask to float for multiplication
    cast_mask_node = helper.make_node(
        "Cast",
        inputs=[attention_mask_name],
        outputs=["attention_mask_float"],
        to=TensorProto.FLOAT,
        name="cast_attention_mask"
    )
    nodes_to_add.append(cast_mask_node)
    
    # Create axes tensor for unsqueeze operation
    unsqueeze_axes_tensor = helper.make_tensor(
        name="unsqueeze_axes_mean",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2]  # Unsqueeze at dimension 2
    )
    unsqueeze_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_const_mean"],
        value=unsqueeze_axes_tensor,
        name="unsqueeze_axes_constant_mean"
    )
    nodes_to_add.append(unsqueeze_axes_node)
    
    # Expand attention mask to match hidden state dimensions
    # Shape: [batch_size, sequence_length] -> [batch_size, sequence_length, 1]
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["attention_mask_float", "unsqueeze_axes_const_mean"],
        outputs=["attention_mask_expanded"],
        name="unsqueeze_attention_mask"
    )
    nodes_to_add.append(unsqueeze_node)
    
    # Multiply hidden states with attention mask
    mul_node = helper.make_node(
        "Mul",
        inputs=[input_name, "attention_mask_expanded"],
        outputs=["masked_hidden_states"],
        name="apply_attention_mask"
    )
    nodes_to_add.append(mul_node)
    
    # Create axes tensor for ReduceSum (sequence dimension = 1)
    sum_axes_tensor = helper.make_tensor(
        name="sum_axes_mean",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Sum along sequence dimension
    )
    sum_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sum_axes_const_mean"],
        value=sum_axes_tensor,
        name="sum_axes_constant_mean"
    )
    nodes_to_add.append(sum_axes_node)
    
    # Sum along sequence dimension
    sum_node = helper.make_node(
        "ReduceSum",
        inputs=["masked_hidden_states", "sum_axes_const_mean"],
        outputs=["summed_hidden_states"],
        keepdims=0,
        name="sum_hidden_states"
    )
    nodes_to_add.append(sum_node)
    
    # Create axes tensor for summing attention mask
    mask_sum_axes_tensor = helper.make_tensor(
        name="mask_sum_axes_mean",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Sum along sequence dimension
    )
    mask_sum_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mask_sum_axes_const_mean"],
        value=mask_sum_axes_tensor,
        name="mask_sum_axes_constant_mean"
    )
    nodes_to_add.append(mask_sum_axes_node)
    
    # Sum attention mask to get sequence lengths
    sum_mask_node = helper.make_node(
        "ReduceSum",
        inputs=["attention_mask_float", "mask_sum_axes_const_mean"],
        outputs=["sequence_lengths"],
        keepdims=1,
        name="sum_attention_mask"
    )
    nodes_to_add.append(sum_mask_node)
    
    # Divide by sequence lengths to get mean
    div_node = helper.make_node(
        "Div",
        inputs=["summed_hidden_states", "sequence_lengths"],
        outputs=["mean_pooled"],
        name="mean_division"
    )
    nodes_to_add.append(div_node)
    
    return "mean_pooled"


def add_max_pooling(graph, input_name, attention_mask_name, nodes_to_add):
    """Add max token pooling with attention mask."""
    # Convert attention mask to float
    cast_mask_node = helper.make_node(
        "Cast",
        inputs=[attention_mask_name],
        outputs=["attention_mask_float_max"],
        to=TensorProto.FLOAT,
        name="cast_attention_mask_max"
    )
    nodes_to_add.append(cast_mask_node)
    
    # Create a large negative value for masked positions
    neg_inf_tensor = helper.make_tensor(
        name="neg_inf_value_max",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[-1e9]
    )
    neg_inf_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_inf_const_max"],
        value=neg_inf_tensor,
        name="negative_infinity_max"
    )
    nodes_to_add.append(neg_inf_node)
    
    # Create axes tensor for unsqueeze operation
    unsqueeze_axes_tensor_max = helper.make_tensor(
        name="unsqueeze_axes_max",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2]  # Unsqueeze at dimension 2
    )
    unsqueeze_axes_node_max = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_const_max"],
        value=unsqueeze_axes_tensor_max,
        name="unsqueeze_axes_constant_max"
    )
    nodes_to_add.append(unsqueeze_axes_node_max)
    
    # Expand attention mask
    unsqueeze_max_node = helper.make_node(
        "Unsqueeze",
        inputs=["attention_mask_float_max", "unsqueeze_axes_const_max"],
        outputs=["attention_mask_expanded_max"],
        name="unsqueeze_attention_mask_max"
    )
    nodes_to_add.append(unsqueeze_max_node)
    
    # Create mask for padding positions (1 - attention_mask)
    one_tensor_max = helper.make_tensor(
        name="one_value_max",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[1.0]
    )
    one_node_max = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_const_max"],
        value=one_tensor_max,
        name="one_constant_max"
    )
    nodes_to_add.append(one_node_max)
    
    sub_node = helper.make_node(
        "Sub",
        inputs=["one_const_max", "attention_mask_expanded_max"],
        outputs=["padding_mask_max"],
        name="create_padding_mask_max"
    )
    nodes_to_add.append(sub_node)
    
    # Multiply padding mask with negative infinity
    mul_neg_inf_node = helper.make_node(
        "Mul",
        inputs=["padding_mask_max", "neg_inf_const_max"],
        outputs=["neg_inf_mask_max"],
        name="multiply_neg_inf_max"
    )
    nodes_to_add.append(mul_neg_inf_node)
    
    # Add to hidden states (this sets padding positions to -inf)
    add_mask_node = helper.make_node(
        "Add",
        inputs=[input_name, "neg_inf_mask_max"],
        outputs=["masked_hidden_states_max"],
        name="add_neg_inf_mask_max"
    )
    nodes_to_add.append(add_mask_node)
    
    # Create axes tensor for ReduceMax (sequence dimension = 1)
    max_axes_tensor = helper.make_tensor(
        name="max_axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Max along sequence dimension
    )
    max_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["max_axes_const"],
        value=max_axes_tensor,
        name="max_axes_constant"
    )
    nodes_to_add.append(max_axes_node)
    
    # Max pooling along sequence dimension
    max_node = helper.make_node(
        "ReduceMax",
        inputs=["masked_hidden_states_max", "max_axes_const"],
        outputs=["max_pooled"],
        keepdims=0,
        name="max_pooling"
    )
    nodes_to_add.append(max_node)
    
    return "max_pooled"


def add_mean_sqrt_len_pooling(graph, input_name, attention_mask_name, nodes_to_add):
    """Add mean pooling divided by sqrt of sequence length."""
    # Convert attention mask to float for sqrt calculation
    cast_mask_sqrt_node = helper.make_node(
        "Cast",
        inputs=[attention_mask_name],
        outputs=["attention_mask_float_sqrt"],
        to=TensorProto.FLOAT,
        name="cast_attention_mask_sqrt"
    )
    nodes_to_add.append(cast_mask_sqrt_node)
    
    # Create axes tensor for sum operation
    sqrt_sum_axes_tensor = helper.make_tensor(
        name="sqrt_sum_axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Sum along sequence dimension
    )
    sqrt_sum_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sqrt_sum_axes_const"],
        value=sqrt_sum_axes_tensor,
        name="sqrt_sum_axes_constant"
    )
    nodes_to_add.append(sqrt_sum_axes_node)
    
    # Sum attention mask to get sequence lengths
    sum_mask_sqrt_node = helper.make_node(
        "ReduceSum",
        inputs=["attention_mask_float_sqrt", "sqrt_sum_axes_const"],
        outputs=["sequence_lengths_sqrt"],
        keepdims=1,
        name="sum_attention_mask_sqrt"
    )
    nodes_to_add.append(sum_mask_sqrt_node)
    
    # Calculate sqrt of sequence lengths
    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=["sequence_lengths_sqrt"],
        outputs=["sqrt_sequence_lengths"],
        name="sqrt_lengths"
    )
    nodes_to_add.append(sqrt_node)
    
    # First get regular mean pooled output (reuse the function logic inline)
    # Convert attention mask to float for multiplication
    cast_mask_node = helper.make_node(
        "Cast",
        inputs=[attention_mask_name],
        outputs=["attention_mask_float_sqrt_mean"],
        to=TensorProto.FLOAT,
        name="cast_attention_mask_sqrt_mean"
    )
    nodes_to_add.append(cast_mask_node)
    
    # Create axes tensor for unsqueeze operation
    unsqueeze_axes_tensor = helper.make_tensor(
        name="unsqueeze_axes_sqrt_mean",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2]  # Unsqueeze at dimension 2
    )
    unsqueeze_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_const_sqrt_mean"],
        value=unsqueeze_axes_tensor,
        name="unsqueeze_axes_constant_sqrt_mean"
    )
    nodes_to_add.append(unsqueeze_axes_node)
    
    # Expand attention mask
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["attention_mask_float_sqrt_mean", "unsqueeze_axes_const_sqrt_mean"],
        outputs=["attention_mask_expanded_sqrt_mean"],
        name="unsqueeze_attention_mask_sqrt_mean"
    )
    nodes_to_add.append(unsqueeze_node)
    
    # Multiply hidden states with attention mask
    mul_node = helper.make_node(
        "Mul",
        inputs=[input_name, "attention_mask_expanded_sqrt_mean"],
        outputs=["masked_hidden_states_sqrt_mean"],
        name="apply_attention_mask_sqrt_mean"
    )
    nodes_to_add.append(mul_node)
    
    # Create axes tensor for ReduceSum
    sum_axes_tensor = helper.make_tensor(
        name="sum_axes_sqrt_mean",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # Sum along sequence dimension
    )
    sum_axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sum_axes_const_sqrt_mean"],
        value=sum_axes_tensor,
        name="sum_axes_constant_sqrt_mean"
    )
    nodes_to_add.append(sum_axes_node)
    
    # Sum along sequence dimension
    sum_node = helper.make_node(
        "ReduceSum",
        inputs=["masked_hidden_states_sqrt_mean", "sum_axes_const_sqrt_mean"],
        outputs=["summed_hidden_states_sqrt_mean"],
        keepdims=0,
        name="sum_hidden_states_sqrt_mean"
    )
    nodes_to_add.append(sum_node)
    
    # Divide summed by sequence lengths to get mean
    div_mean_node = helper.make_node(
        "Div",
        inputs=["summed_hidden_states_sqrt_mean", "sequence_lengths_sqrt"],
        outputs=["mean_pooled_sqrt"],
        name="mean_division_sqrt"
    )
    nodes_to_add.append(div_mean_node)
    
    # Divide mean pooled output by sqrt of sequence length
    div_sqrt_node = helper.make_node(
        "Div",
        inputs=["mean_pooled_sqrt", "sqrt_sequence_lengths"],
        outputs=["mean_sqrt_len_pooled"],
        name="divide_by_sqrt_length"
    )
    nodes_to_add.append(div_sqrt_node)
    
    return "mean_sqrt_len_pooled"


def add_concatenation(graph, pooling_outputs, nodes_to_add):
    """Concatenate multiple pooling outputs."""
    concat_node = helper.make_node(
        "Concat",
        inputs=pooling_outputs,
        outputs=["concatenated_pooling"],
        axis=1,  # Concatenate along feature dimension
        name="concatenate_pooling_outputs"
    )
    nodes_to_add.append(concat_node)
    
    return "concatenated_pooling"


# Example usage
if __name__ == "__main__":
    # Example usage with HuggingFace model ID:
    add_pooling_to_onnx_model(
        onnx_model_path="miniLM/model.onnx",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        output_model_path="model_with_pooling.onnx"
    )