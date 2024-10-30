//
// Created by martinkorelic on 31/08/2024
//

#include <jni.h>
#include <string>
#include "include/onnxruntime/onnxruntime_training_cxx_api.h"
#include "session_cache.h"
#include "inference.h"
#include "utils.h"
#include "train.h"
#include "onnxruntime-genai/ort_genai.h"
#include <android/log.h>

#define LOG_TAG "ORTTransformer"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_orttransformer_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

/**
 * Creates a new user managed tensor from the parameter.
 *
 * @param parameter
 * @return
 */
Ort::Value CreateNewParameter(Ort::Value& parameter) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    auto& ortApi = Ort::GetApi();

    Ort::TensorTypeAndShapeInfo tensor_info = parameter.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> tensor_shape = tensor_info.GetShape();
    auto tensor_type = tensor_info.GetElementType();
    size_t total_elements = tensor_info.GetElementCount();
    size_t element_size = 0;

    // Determine the size of one element based on tensor type
    switch (tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            element_size = sizeof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
            break;

        default:
            throw std::runtime_error("Unsupported tensor data type");
    }

    // Allocate memory for the tensor data
    size_t data_size = total_elements * element_size;
    void* user_data = allocator.Alloc(data_size);

    // Copy data from the parameter Ort::Value to user_data
    std::memcpy(user_data, parameter.GetTensorRawData(), data_size);

    OrtValue* c_tensor;
    auto ortStatus = ortApi.CreateTensorWithDataAsOrtValue(memory_info, user_data, data_size, tensor_shape.data(), tensor_shape.size(), tensor_type, &c_tensor);

    // Wrap the C API tensor into a C++ Ort::Value
    Ort::Value user_managed_tensor(c_tensor);

    if (ortStatus != nullptr) {
        const char* error_message = ortApi.GetErrorMessage(ortStatus);
        ortApi.ReleaseStatus(ortStatus);
        allocator.Free(user_data);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", error_message);
        throw std::runtime_error("Failed to create tensor with user-managed data.");
    }

    return user_managed_tensor;
}

/**
 * Loads the weights to the inference session addInitializer function.
 *
 * @param session
 * @param checkpoint_state
 * @param layer_names
 */
void LoadWeightsToMemory(const std::unique_ptr<WeightSessionCache>& weight_cache, Ort::CheckpointState& checkpoint_state, const std::vector<std::string>& layer_names) {

    for (const auto& layer : layer_names) {
        auto parameter = checkpoint_state.GetParameter(layer);
        auto user_parameter = CreateNewParameter(parameter);

        // Ensure the weight values are not null and is a tensor
        if (user_parameter.IsTensor()) {
            weight_cache->weights.emplace(layer, std::move(user_parameter));

            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loaded %s weights into the memory.", layer.c_str());
        } else {
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "ERROR: Could not load %s weights into the memory!", layer.c_str());
        }
    }
}

/**
 * Loads the weights to the inference session addInitializer function.
 *
 * @param session
 * @param checkpoint_state
 * @param layer_names
 */
void LoadWeightsToInferenceSession(const std::unique_ptr<InferenceSessionCache>& session, Ort::CheckpointState& checkpoint_state, const std::vector<std::string>& layer_names) {

    for (const auto& layer : layer_names) {
        auto parameter = checkpoint_state.GetParameter(layer);
        auto user_parameter = CreateNewParameter(parameter);

        // Ensure the weight values are not null
        if (user_parameter) {
            session->session_options.AddInitializer(layer.c_str(), user_parameter);
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loaded %s weights into the model.", layer.c_str());
        } else {
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "ERROR: Could not load %s weights into the model!", layer.c_str());
        }
    }
}

/**
 * Transfers the weights from the given checkpoint state and layer names to the inference session options.
 *
 * @param session
 * @param checkpoint_state
 * @param layer_names
 */
void TransferWeightsFromCheckpoint(const std::unique_ptr<InferenceSessionCache>& session, Ort::CheckpointState& checkpoint_state, const std::vector<std::string>& layer_names) {
    // Extract and load the current weights from the checkpoint to the session options
    LoadWeightsToInferenceSession(session, checkpoint_state, layer_names);
}

/**
 * Transfers the weights from the given checkpoint state path and layer names to the inference session options.
 *
 * @param session
 * @param checkpoint_path
 * @param layer_names
 */
void TransferWeightsFromCheckpoint(const std::unique_ptr<InferenceSessionCache>& session, const std::string& checkpoint_path, const std::vector<std::string>& layer_names) {
    // TODO: Needs to create training session for checkpoint beforehand
    Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(checkpoint_path);
    // Extract and load the current weights from the checkpoint to the session options
    LoadWeightsToInferenceSession(session, checkpoint_state, layer_names);
}

void ReleaseTrainingSession(jlong session, jboolean saveCheckpoint) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    if (saveCheckpoint) {
        // Include optimizer state?
        Ort::CheckpointState::SaveCheckpoint(session_cache->checkpoint_state, session_cache->artifact_paths.checkpoint_path, true);
    }

    delete session_cache;
    session_cache = nullptr;
}

void ReleaseWeightSession(jlong session) {
    auto *session_cache = reinterpret_cast<WeightSessionCache *>(session);

    delete session_cache;
    session_cache = nullptr;
}

void ReleaseGenAISession(jlong session) {
    auto *session_cache = reinterpret_cast<GenAISessionCache *>(session);
    //OgaDestroyGenerator(session_cache->generator.get());
    //OgaDestroyGeneratorParams(session_cache->generatorParams.get());
    //OgaDestroyModel(session_cache->model.get());
    //OgaDestroyTokenizer(session_cache->tokenizer.get());
    //OgaDestroyTokenizerStream(session_cache->tokenizer_stream.get());
    delete session_cache;
    session_cache = nullptr;
}

Ort::SessionOptions CreateSessionOptions() {
    // Create a SessionOptions instance
    Ort::SessionOptions session_options;

    // Disable memory pattern
    session_options.DisableCpuMemArena();

    // Disable CPU memory arena
    session_options.DisableMemPattern();

    return session_options;
}

extern "C"
JNIEXPORT float JNICALL
/**
 * Performs the training step with the gradient update and optimizer step.
 * Attention mask, position ids and labels are created from the given input ids.
 *
 * @param env
 * @param session
 * @param input_ids
 * @param batch_size
 * @param sequence_length
 *
 * @return Loss value
 */
Java_com_example_orttransformer_ORTTrainerNative_performTraining(
        JNIEnv *env, jobject /* this */,
        jlong session,
        jlongArray input_ids, jint batch_size, jint sequence_length) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    // Get the input_ids array from the Java environment
    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);

    // Calculate the total size for the input data based on batch_size and sequence_length
    size_t total_size = batch_size * sequence_length;

    // Allocate memory for attention mask, position ids, and labels (assuming labels are provided here)
    std::vector<int64_t> attention_mask(total_size, 1);  // Initialized with 1s
    std::vector<int64_t> position_ids(total_size);
    std::vector<int64_t> labels(total_size, 0);

    // Populate position ids (0 to sequence_length - 1 for each batch element)
    for (int64_t i = 0; i < batch_size; ++i) {
        for (int64_t j = 0; j < sequence_length; ++j) {
            position_ids[i * sequence_length + j] = j;
        }
    }

    // Prepare attention_mask and position_ids as jlongArrays to return to Java if needed
    //jlongArray attention_mask_array = env->NewLongArray(total_size);
    //jlongArray position_ids_array = env->NewLongArray(total_size);
    //jlongArray labels_array = env->NewLongArray(total_size);

    // Copy the vectors to the Java arrays
    //env->SetLongArrayRegion(attention_mask_array, 0, total_size, attention_mask.data());
    //env->SetLongArrayRegion(position_ids_array, 0, total_size, position_ids.data());
    //env->SetLongArrayRegion(labels_array, 0, total_size, labels.data());

    utils::initialize_labels(input_ids_elements, labels.data(), batch_size, sequence_length);



    // Update the model parameters using this batch of inputs.
    return training::train_step(session_cache, input_ids_elements,
                                attention_mask.data(), position_ids.data(), labels.data(), batch_size, sequence_length);
}

extern "C"
JNIEXPORT jlong JNICALL
/**
 * Creates the training session from the given artifact paths.
 *
 * @param env
 * @param thiz
 * @param checkpoint_path
 * @param train_model_path
 * @param eval_model_path
 * @param optimizer_model_path
 * @param cache_dir_path
 * @param requires_grad
 *
 * @return Training session native model handle.
 */
Java_com_example_orttransformer_ORTTrainerNative_createTrainingSession(JNIEnv *env, jobject thiz,
                                                           jstring checkpoint_path,
                                                           jstring train_model_path,
                                                           jstring eval_model_path,
                                                           jstring optimizer_model_path,
                                                           jstring cache_dir_path,
                                                           jobjectArray requires_grad) {

    // Get the size of the input array
    jsize arrayLength = env->GetArrayLength(requires_grad);

    std::unique_ptr<TrainingSessionCache> session_cache = std::make_unique<TrainingSessionCache>(
            utils::JString2String(env, checkpoint_path),
            utils::JString2String(env, train_model_path),
            utils::JString2String(env, eval_model_path),
            utils::JString2String(env, optimizer_model_path),
            utils::JString2String(env, cache_dir_path),
            "low_mem",
            true);

    for (jsize i = 0; i < arrayLength; ++i) {
        auto jstr = (jstring) (env->GetObjectArrayElement(requires_grad, i));
        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
        session_cache->requires_grad.emplace_back(cstr);

        // Release the string
        env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }

    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT jlong JNICALL
/**
 * Caches the trainable layer weights from training session for later use. Releases the training session.
 *
 * @param env
 * @param inference_model_path
 * @return
 */
Java_com_example_orttransformer_ORTGenAINative_cacheSessionWeights(
        JNIEnv *env, jobject /* this */,
        jlong train_session,
        jobjectArray requires_grad
) {

    auto *train_session_cache = reinterpret_cast<TrainingSessionCache *>(train_session);

    // Get the size of the input array
    jsize arrayLength = env->GetArrayLength(requires_grad);

    std::vector<std::string> requires_grad_names;
    for (jsize i = 0; i < arrayLength; ++i) {
        auto jstr = (jstring) (env->GetObjectArrayElement(requires_grad, i));
        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
        requires_grad_names.emplace_back(cstr);

        // Release the string
        env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }

    std::unique_ptr<WeightSessionCache> weight_session_cache = std::make_unique<WeightSessionCache>();

    // Load the extracted weights into the inference session
    LoadWeightsToMemory(weight_session_cache, train_session_cache->checkpoint_state, requires_grad_names);

    // Release the current training session, save the checkpoints
    ReleaseTrainingSession(train_session, true);
    train_session_cache = nullptr;

    return reinterpret_cast<long>(weight_session_cache.release());
}

extern "C" JNIEXPORT jlong JNICALL
/**
 * Creates the GenAI session from the cached weights.
 *
 * @param env
 * @param inference_model_path
 * @return
 */
Java_com_example_orttransformer_ORTGenAINative_createGenAISession(
        JNIEnv *env, jobject /* this */,
        jlong weight_cache,
        jstring genai_path
) {

    auto *weight_session_cache = reinterpret_cast<WeightSessionCache *>(weight_cache);

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loading grad weight inputs...");

    std::unique_ptr<GenAISessionCache> genai_session_cache = std::make_unique<GenAISessionCache>(
            weight_session_cache,
            utils::JString2String(env, genai_path)
            );
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Preloaded grad weight inputs.");

    return reinterpret_cast<long>(genai_session_cache.release());
}

extern "C" JNIEXPORT jlong JNICALL
/**
 * Creates normal inference session from the given inference model path.
 * This inference is a custom made inference which is ready to be used for generation with KV caching.
 *
 * @param env
 * @param inference_model_path
 * @return
 */
Java_com_example_orttransformer_ORTGeneratorNative_createInferenceSession(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path,
        jstring inference_model_name
        ) {
    std::unique_ptr<InferenceSessionCache> session_cache = std::make_unique<InferenceSessionCache>(utils::JString2String(env, inference_model_path), utils::JString2String(env, inference_model_name), "high_perf", true);

    session_cache->initializeKVCache(1);

    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT jlong JNICALL
/***
 * Function for creating an inference session from the given checkpoint state.
 * This assumes that the training session was not created beforehand.
 *
 *  TODO: An inference session with Android NNAPI could be added.
 *
 * 1. Load checkpoint data
 * 2. Uses the layer names provided
 * 3. Collect the parameters of layers which are trainable
 * 4. Transfer the weights to the inference session options with the weights from the trainable layers of checkpoint state.
 * 5. Load the inference model
 * 6. The model is ready for inference with the weights that were overriden by the trainable layer weights
 */
Java_com_example_orttransformer_ORTGeneratorNative_createInferenceSessionFromCheckpoint(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path,
        jstring inference_model_name,
        jstring checkpoint_path,
        jobjectArray requires_grad
) {

    // Get the size of the input array
    jsize arrayLength = env->GetArrayLength(requires_grad);

    std::vector<std::string> requires_grad_names;
    for (jsize i = 0; i < arrayLength; ++i) {
        auto jstr = (jstring) (env->GetObjectArrayElement(requires_grad, i));
        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
        requires_grad_names.emplace_back(cstr);

        // Release the string
        env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }

    std::unique_ptr<InferenceSessionCache> inference_session_cache = std::make_unique<InferenceSessionCache>(
            utils::JString2String(env, inference_model_path),
            utils::JString2String(env, inference_model_name),
            "low_mem",
            true
    );

    // Load the extracted weights into the inference session
    TransferWeightsFromCheckpoint(inference_session_cache, utils::JString2String(env, checkpoint_path), requires_grad_names);

    // Load the model which will not deserialize the already loaded initializers
    inference_session_cache->inference_session = std::make_unique<Ort::Session>(
            inference_session_cache->ort_env, inference_session_cache->inference_model_path.c_str(),
            inference_session_cache->session_options).release();

    return reinterpret_cast<long>(inference_session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
/**
 * Deletes the current inference session.
 *
 * @param env
 * @param session
 */
Java_com_example_orttransformer_ORTGeneratorNative_releaseInferenceSession(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    delete session_cache->inference_session;
    delete session_cache;
    session_cache = nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_orttransformer_ORTTrainerNative_releaseTrainingSession(
        JNIEnv *env, jobject,
        jlong session, jboolean saveCheckpoint) {
    ReleaseTrainingSession(session, saveCheckpoint);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_orttransformer_ORTGenAINative_releaseWeightSession(
        JNIEnv *env, jobject,
        jlong weight_session) {
    ReleaseWeightSession(weight_session);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_orttransformer_ORTGenAINative_releaseGenAISession(
        JNIEnv *env, jobject,
        jlong genai_session) {
    ReleaseGenAISession(genai_session);
}

extern "C" JNIEXPORT jlong JNICALL
/***
 * Function for creating an inference session from the current training session without exporting from inference.
 * This will allow to perform inference on the other model which is compatible with ONNX GenAI API.
 *
 *  TODO: An inference session with Android NNAPI could be added.
 *
 * 1. Use the current checkpoint data
 * 2. Load the names of trainable layers from training_config.json
 * 3. Collect the parameters of layers which are trainable
 * 4. Delete the training session
 * 5. Update the inference session options with the weights from the trainable layers
 * 6. Load the inference model
 * 7. The model is ready for inference with the weights that were overriden by the trainable layer weights
 */
Java_com_example_orttransformer_ORTGeneratorNative_createInferenceSessionFromTraining(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path,
        jstring inference_model_name,
        jlong train_session
) {
    auto *train_session_cache = reinterpret_cast<TrainingSessionCache *>(train_session);

    // Create the inference session
    std::unique_ptr<InferenceSessionCache> inference_session_cache = std::make_unique<InferenceSessionCache>(
            utils::JString2String(env, inference_model_path),
            utils::JString2String(env, inference_model_path),
            "low_mem",
            true
    );

    // Load the extracted weights into the inference session
    TransferWeightsFromCheckpoint(inference_session_cache, train_session_cache->checkpoint_state, train_session_cache->requires_grad);

    // Release the current training session, save the checkpoints
    ReleaseTrainingSession(train_session, true);
    train_session_cache = nullptr;

    // TODO: Test inference with Android NNAPI
    // Source needs to be built with NNAPI enabled to add it to session
    //Ort::SessionOptions so;
    //uint32_t nnapi_flags = 0;

    // Load the model which will not deserialize the already loaded initializers
    inference_session_cache->inference_session = std::make_unique<Ort::Session>(
            inference_session_cache->ort_env, inference_session_cache->inference_model_path.c_str(),
            inference_session_cache->session_options).release();


    return reinterpret_cast<long>(inference_session_cache.release());
}


extern "C"
JNIEXPORT void JNICALL
/**
 * Utility function, example of inspecting weights in the model from the checkpoint state.
 * */
Java_com_example_orttransformer_ORTTrainerNative_inspectWeights(JNIEnv *env, jobject thiz,
                                                            jlong session, jstring layer) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    Ort::Value parameter = session_cache->checkpoint_state.GetParameter(utils::JString2String(env, layer));

    auto type_info = parameter.GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // Get tensor dimensions
    std::vector<int64_t> dimensions = tensor_info.GetShape();
    // Get the data type
    ONNXTensorElementDataType dtype = tensor_info.GetElementType();

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Type: type=%u", dtype);
    for (auto dim: dimensions) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Dimension: type=%ld", dim);
    }
}


extern "C"
JNIEXPORT jstring JNICALL
/**
 * Export model for inference from the training session.
 *
 * @param env
 * @param thiz
 * @param session - Current training session native model handle
 * @return
 */
Java_com_example_orttransformer_ORTTrainerNative_exportModelForInference(JNIEnv *env, jobject thiz,
                                                                   jlong session) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);
    session_cache->training_session.ExportModelForInferencing(session_cache->artifact_paths.inference_model_path, {"logits"});
    return env->NewStringUTF(session_cache->artifact_paths.inference_model_path.c_str());
}

extern "C"
JNIEXPORT void JNICALL
/**
 * Initializes the GenAI inference with a new prompt.
 * Uses KV caching and generation configuration provided from the file.
 *
 * @param env
 * @param thiz
 * @param genai_session - GenAI cached session
 * @param prompt - New prompt
 */
Java_com_example_orttransformer_ORTGenAINative_initializeGenAIInference(JNIEnv *env, jobject thiz,
                                                                        jlong genai_session, jstring prompt) {
    auto *session_cache = reinterpret_cast<GenAISessionCache *>(genai_session);

    auto sequences = OgaSequences::Create();
    session_cache->tokenizer->Encode(utils::JString2String(env, prompt).c_str(), *sequences);
    session_cache->generatorParams->SetInputSequences(*sequences);
    session_cache->generator = OgaGenerator::Create(*session_cache->model, *session_cache->generatorParams);
}

extern "C"
JNIEXPORT jstring JNICALL
/**
 * Performs the inference step using the exported model for GenAI inference.
 *
 * @param env
 * @param thiz
 * @param session
 * @param input_ids
 * @param attention_mask
 * @param position_ids
 * @param sequence_length
 * @param vocab_size
 *
 * @return Next token string
 */
Java_com_example_orttransformer_ORTGenAINative_performGenAIInferenceStep(JNIEnv *env, jobject thiz,
                                                                        jlong genai_session) {
    auto *session_cache = reinterpret_cast<GenAISessionCache *>(genai_session);

    if (session_cache->generator->IsDone()) {
        // TODO : Release generator after each session?
        //OgaDestroyGenerator(session_cache->generator.get());
        return env->NewStringUTF("[STOP]");
    }

    auto next_token = inference::genAiInferenceStep(session_cache);

    return env->NewStringUTF(next_token.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
/**
 * Performs the inference step using the exported model for inference.
 *
 * @param env
 * @param thiz
 * @param session
 * @param input_ids
 * @param attention_mask
 * @param position_ids
 * @param sequence_length
 * @param past_sequence_length
 * @param vocab_size
 *
 * @return Next token id
 */
Java_com_example_orttransformer_ORTGeneratorNative_performInferenceStep(JNIEnv *env, jobject thiz,
                                                                  jlong session,
                                                                  jlongArray input_ids,
                                                                  jlongArray attention_mask,
                                                                  jlongArray position_ids,
                                                                  jint batch_size,
                                                                  jint sequence_length,
                                                                  jint past_sequence_length,
                                                                  jint vocab_size
                                                                  ) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* attention_mask_elements = env->GetLongArrayElements(attention_mask, nullptr);
    jlong* position_ids_elements = env->GetLongArrayElements(position_ids, nullptr);

    if (session_cache->enable_profiling) {
        session_cache->startProfiling();
    }

    // Forward pass
    auto logits = inference::generateWithKVCache(session_cache,
                                   input_ids_elements,
                                   attention_mask_elements,
                                   position_ids_elements,
                                   batch_size,
                                   sequence_length,
                                   past_sequence_length);

    // TODO: Choose which sampling method to use, using greedy for now
    int best_index = inference::argmax(
            logits,
            past_sequence_length,
            vocab_size
            );

    if (session_cache->enable_profiling) {
        session_cache->endProfiling();
        // TODO: Either enable for further forward pass but we would only enable for prefill phase
        session_cache->enable_profiling = false;
    }

    return best_index;
}