"""
Deprecated script for building the TFLite models for training and inference from Huggingface.
Unfortunately, the conversion is unoptimized, models are not being exported properly and errors are present.
Feel free to modify at your own will.
"""

import os
import tensorflow as tf
import keras

import keras_nlp
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.lite.python import interpreter

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Read from the environment, never written here. A credential-shaped placeholder in a tracked
# file is an invitation to paste a real one into it and commit it by accident.
kaggle_username = os.environ.get("KAGGLE_USERNAME", "")
kaggle_key = os.environ.get("KAGGLE_KEY", "")
hf_token = os.environ.get("HF_TOKEN", "")

def convert_llm_tflite(archive_dir, tflite_path, rank=4, use_lora=True, model_type="gemma"):

    if model_type == "gemma":
        llm_model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en", preprocessor=None)
    elif model_type == "falcon":
        llm_model = keras_nlp.models.FalconCausalLM.from_preset("falcon_refinedweb_1b_en", preprocessor=None)
    elif model_type == "bloom":
        llm_model = keras_nlp.models.BloomCausalLM.from_preset("bloom_1.1b_multi", preprocessor=None)
    elif model_type == "opt":
        llm_model = keras_nlp.models.OPTCausalLM.from_preset("opt_1.3b_en", preprocessor=None)
    llm_model.summary()

    # Enable LoRA for the model and set the LoRA rank to 4.
    if use_lora:
        llm_model.backbone.enable_lora(rank=rank)
        llm_model.summary()

    # Need to define the shape of the array, to allocate bytes when loading input signature in Android
    SEQUENCE_LENGTH = 256

    # Limit the input sequence length to 256 (to control memory usage).
    #llm_model.preprocessor.sequence_length = 256
    # Use AdamW (a common optimizer for transformer models).
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    # Exclude layernorm and bias terms from decay.
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    llm_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Example usage of input data to the train function
    #x = {
    # #Token ids for "<bos> Keras is deep learning library<eos>"
    #    "token_ids": np.array([[2, 50271, 603, 5271, 6044, 9581, 1, 0]] * 2),
    #   "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    #}
    #y = np.array([[50271, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
    #sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)
    #o = llm_model.train_step([x, y, sw])
    #print(o)

    #model = llm_model.backbone
    export_archive = keras.export.ExportArchive()
    export_archive.track(llm_model)

    # Model call endpoint
    #export_archive.add_endpoint(
    #    name="call_model",
    #    fn=llm_model.backbone.call,
    #    input_signature=[{
    #        "token_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int, name="token_ids"),
    #        "padding_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="padding_mask"),
    #    }],
    #)
    # llm_model.generate
    # NotImplementedError: Cannot convert a symbolic tf.Tensor (while:2) to a numpy array. 
    # This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.

    # Inference endpoint
    export_archive.add_endpoint(
        name="call_inference",
        fn=llm_model.generate_step,
        input_signature=[{
            "token_ids": tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="token_ids"),
            "padding_mask": tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="padding_mask"),
        }],
    )

    # Training endpoint
    export_archive.add_endpoint(
        name="call_training",
        fn=llm_model.train_step,
        input_signature=[[
            {
            "token_ids": tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="x_token_ids"),
            "padding_mask": tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="x_padding_mask"),
        },
        tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="y_token_ids"),
        tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32, name="sample_weights")]],
    )
    export_archive.write_out(archive_dir)
     
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(archive_dir)
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_converter = True

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # If no post training quantization:
    # 20 GB for 2B model + 140GB RAM needed for conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def convert_gemma_hf():
    
    gemma_name = "google/gemma-2b-it"
    #gemma_model = AutoModelForCausalLM.from_pretrained(gemma_name, trust_remote_code=True, token=hf_token)
    gemma_tf = AutoModelForCausalLM.from_pretrained(gemma_name, trust_remote_code=True, token=hf_token)

    # Generate text
    prompt = "Hello, world!"
    #input_ids = gemma_tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    #output = gemma_tf.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)
    SAVED_MODEL_DIR = "gemma_tflite_normal"


    class GemmaMobileModel(Model):

        def __init__(self, gemma_lm):
            super(GemmaMobileModel, self).__init__()
            self.backbone = gemma_lm

        # The `train` function takes a batch of input images and labels.
        #@tf.function#(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
        def train(self, data):
            self.backbone.fit(data, epochs=1, batch_size=1)

        @tf.function(input_signature=[tf.TensorSpec(shape=(20,), dtype=tf.string, name="input")])
        def generate(self, encoded_text):

            print(encoded_text)
            return {
                "text": self.backbone.generate(encoded_text, max_length=256)
            }

        def extract_strings(self, tensor):
            """Extracts strings from a TensorFlow tensor without using numpy.

            Args:
                tensor: A TensorFlow tensor of dtype tf.string.

            Returns:
                A list of strings.
            """

            # Assuming a rank-1 tensor of strings
            strings = tf.strings.reduce_join(tensor)  # Combine all strings into a single string
            strings_split = tf.strings.split(strings, sep=b',')  # Split the combined string into individual strings
            print(strings_split)
            # Convert to a Python list
            return strings_split.numpy().tolist()      
    
    SAVED_MODEL_TFLITE = "gemma_tflite_mini.tflite"
    gmm = GemmaMobileModel(gemma_tf)
    tf.saved_model.save(gmm, SAVED_MODEL_DIR, signatures={"serving_default": gmm.generate.get_concrete_function(tf.TensorSpec((20,), tf.string, name="input"))})

    gmm.jit_compile = False
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.allow_custom_ops = True
    converter.target_spec.experimental_select_user_tf_ops = [
        "UnsortedSegmentJoin",
        "UpperBound"
    ]
    #converter._experimental_guarantee_all_funcs_one_use = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    # Save the model
    with open(SAVED_MODEL_TFLITE, 'wb') as f:
        f.write(tflite_model)

def convert_tflite():
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model('gemma_2_archive')
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open('gemma2-quantized-new.tflite', 'wb') as f:
        f.write(tflite_model)

def run_inference_keras_tflite(input_text, tokenizer, tfmodel_path):
    tflite_interp = tf.lite.Interpreter(tfmodel_path)
    inference = tflite_interp.get_signature_runner("call_inference")
    #print(inference.get_input_details())

    gemma_preprocess = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset("gemma_2b_en", sequence_length=256, add_end_token=True,)
    #gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en", preprocessor=gemma_preprocess)

    tokens = gemma_preprocess.generate_preprocess(input_text)
    input_ids = tf.reshape(tf.cast(tokens["token_ids"], tf.int32), [-1,256])
    attention_mask = tf.reshape(tf.cast(tokens["padding_mask"], tf.int32), [-1,256])
    output = inference(inputs=attention_mask, inputs_1=input_ids)
    decoded_text = gemma_preprocess.generate_postprocess(output)
    print(decoded_text)

def run_training_keras_tflite(input_text, tfmodel_path, do_inference=True):
    tflite_interp = tf.lite.Interpreter(tfmodel_path)
    training = tflite_interp.get_signature_runner("call_training")
    inference = tflite_interp.get_signature_runner("call_inference")

    gemma_preprocess = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset("gemma_2b_en", sequence_length=256, add_end_token=True,)
    tokens = gemma_preprocess.generate_preprocess("Keras is deep learning library")
    input_ids = tf.reshape(tf.cast(tokens["token_ids"], tf.int32), [-1,256])
    attention_mask = tf.reshape(tf.cast(tokens["padding_mask"], tf.int32), [-1,256])
    
    if do_inference:
        output = inference(padding_mask=attention_mask, token_ids=input_ids)
        decoded_text = gemma_preprocess.generate_postprocess(output)
        print("Before training 1: ", decoded_text)

    x = {
     #Token ids for "<bos> Keras is deep learning library<eos>"
        "token_ids": tf.constant([[2, 214064, 603, 5271, 6044, 9581, 1, 0]] * 2),
       "padding_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    }
    # Shifted right
    y = tf.constant([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] *2)
    sw = tf.constant([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)
    output = training(x_token_ids=x["token_ids"], x_padding_mask=x["padding_mask"], y_token_ids=y, sample_weights=sw)
    print("Batch 1:", output)

    if do_inference:
        output = inference(padding_mask=attention_mask, token_ids=input_ids)
        decoded_text = gemma_preprocess.generate_postprocess(output)
        print("After training: ", decoded_text)


def run_training_keras_lora(enable_lora=True):
    gemma_preprocess = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset("gemma_2b_en", sequence_length=256, add_end_token=True,)

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en", preprocessor=None)

    # Enable LoRA for the model and set the LoRA rank to 4.
    if enable_lora:
        gemma_lm.backbone.enable_lora(rank=4)

    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    # Exclude layernorm and bias terms from decay.
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    x = {
     #Token ids for "<bos> Keras is deep learning library<eos>"
        "token_ids": tf.constant([pad_array([2, 214064, 603, 5271, 6044, 9581, 1, 0],256)]),
       "padding_mask": tf.constant([pad_array([1, 1, 1, 1, 1, 1, 1, 0],256)]),
    }
    # Shifted right
    y = tf.constant([pad_array([214064, 603, 5271, 6044, 9581, 3, 0, 0],256)])
    sw = tf.constant([pad_array([1, 1, 1, 1, 1, 1, 0, 0],256)])

    tokens = gemma_preprocess.generate_preprocess("Keras is deep learning library")
    input_ids = tf.reshape(tf.cast(tokens["token_ids"], tf.int32), [-1,256])
    attention_mask = tf.reshape(tf.cast(tokens["padding_mask"], tf.int32), [-1,256])


    gen1 = gemma_lm.generate_step({
        "token_ids": input_ids,
        "padding_mask": attention_mask
    })
    print("Before training: ",gemma_preprocess.generate_postprocess(gen1))

    output = gemma_lm.train_step([x, y, sw])
    output1 = gemma_lm.train_step([x, y, sw])
    print(output)
    print(output1)

    gen2 = gemma_lm.generate_step({
        "token_ids": input_ids,
        "padding_mask": attention_mask
    })
    print("After training: ",gemma_preprocess.generate_postprocess(gen2))


def pad_array(array, target_length, pad_value=0):
    """Pads a Python array to a specific length.

    Args:
    array: The input array.
    target_length: The desired length of the output array.
    pad_value: The value to use for padding.

    Returns:
    The padded array.
    """

    array = np.array(array)
    padding_length = max(target_length - len(array), 0)
    return np.pad(array, (0, padding_length), mode='constant', constant_values=pad_value)

def get_signatures_tflite(tflite_path):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get the signature names
    signature_keys = interpreter.get_signature_list()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print details of input and output tensors
    print("Signatures:", signature_keys)
    print("Input details:", input_details)
    print("Output details:", output_details)

def gemma_decode(encoded_tokens):
    encoded_tokens = [2, 214064, 603, 5271, 6044, 9581, 1, 1, 1, 1, 1, 1, 1596, 3225, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901, 919, 1125, 1671, 604, 476, 1497, 1069, 235265, 108, 1718, 919, 1125, 1671, 604, 476, 1497, 1069, 901]
    padding_mask = [1] * len(encoded_tokens)
    enc = {
        "token_ids": encoded_tokens,
        "padding_mask": padding_mask
    }

    gemma_preprocess = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset("gemma_2b_en", sequence_length=256, add_end_token=True,)

    decoded_text = gemma_preprocess.generate_postprocess(enc)
    print(decoded_text)


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "torch" or "tensorflow".
    # Avoid memory fragmentation on JAX backend.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    #tf.config.run_functions_eagerly(True)

    SAVED_MODEL_TFLITE = "/path/to/opt-lora4-fix-inference+training.tflite"
    #get_signatures_tflite(SAVED_MODEL_TFLITE)
    #run_inference_keras_tflite("What is the point of number 42?", None, SAVED_MODEL_TFLITE)
    #run_inference_tflite("Test test", None, SAVED_MODEL_TFLITE)
    #run_training_keras_lora()
    #run_training_keras_tflite("Teest test", SAVED_MODEL_TFLITE, do_inference=False)
    convert_llm_tflite(archive_dir="opt_archive_inftrain", use_lora=True, rank=4, tflite_path=SAVED_MODEL_TFLITE, model_type="opt")