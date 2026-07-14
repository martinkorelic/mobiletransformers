//
// Created by martinkorelic on 31/08/2024
//

#include <jni.h>
#include <string>
#include "onnxruntime/onnxruntime_training_cxx_api.h"
#include "inference.h"
#include "tokenization.h"
#include "train.h"
#include "utils.h"
#include "sampling.h"
#include <android/log.h>
#include "proto/onnx.pb.h"

#define LOG_TAG "MobileTransformers"

extern "C" JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_app_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
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

void ReleaseTokenizerSession(jlong session) {
    auto *session_cache = reinterpret_cast<TokenizerSessionCache *>(session);

    delete session_cache;
    session_cache = nullptr;
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
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_performTraining(
        JNIEnv *env, jobject /* this */,
        jlong session,
        jlongArray input_ids, jlongArray labels, jlongArray attention_mask, jint batch_size, jint sequence_length) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    // Get the input_ids array from the Java environment
    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* label_elements = env->GetLongArrayElements(labels, nullptr);
    jlong* attention_elements = env->GetLongArrayElements(attention_mask, nullptr);

    // Calculate the total size for the input data based on batch_size and sequence_length
    size_t total_size = batch_size * sequence_length;

    // Allocate memory for attention mask, position ids, and labels (assuming labels are provided here)
    //std::vector<int64_t> attention_mask(total_size, 1);  // Initialized with 1s
    std::vector<int64_t> position_ids(total_size);
    //std::vector<int64_t> labels(total_size, 0);

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

    // If need be prepare labels
    //utils::initialize_labels(input_ids_elements, labels.data(), batch_size, sequence_length);

    // Update the model parameters using this batch of inputs.
    float loss = training::train_step(session_cache, input_ids_elements,
                                attention_elements, position_ids.data(), label_elements, batch_size, sequence_length);

    env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
    env->ReleaseLongArrayElements(labels, label_elements, JNI_ABORT);
    env->ReleaseLongArrayElements(attention_mask, attention_elements, JNI_ABORT);

    return loss;
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
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_createTrainingSession(JNIEnv *env, jobject thiz,
                                                           jstring checkpoint_path,
                                                           jstring train_model_path,
                                                           jstring eval_model_path,
                                                           jstring optimizer_model_path,
                                                           jstring cache_dir_path,
                                                           jobjectArray requires_grad,
                                                           jstring memory_config_id,
                                                           jstring core_config_id,
                                                           jstring execution_provider,
                                                           jboolean enable_profiling) {

    // Get the size of the input array
    jsize arrayLength = env->GetArrayLength(requires_grad);

    std::unique_ptr<TrainingSessionCache> session_cache = std::make_unique<TrainingSessionCache>(
            utils::JString2String(env, checkpoint_path),
            utils::JString2String(env, train_model_path),
            utils::JString2String(env, eval_model_path),
            utils::JString2String(env, optimizer_model_path),
            utils::JString2String(env, cache_dir_path),
            utils::JString2String(env, memory_config_id),
            utils::JString2String(env, core_config_id),
            utils::JString2String(env, execution_provider),
            enable_profiling);

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
 * Creates normal inference session from the given inference model path.
 * This inference is a custom made inference which is ready to be used for generation with KV caching.
 *
 * If load_merged_weights is enabled:
 * 1. Transfer the weights to the inference session options with the weights from the weights that were merged and saved.
 * 2. Load the inference model
 * 3. The model is ready for inference with the merged weights.
 *
 * @param env
 * @param inference_model_path
 * @param inference_model_name
 * @param load_merged_weights - Whether to load merged weights from ".../inference/merged"
 * @return
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_createInferenceSession(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path,
        jstring inference_model_name,
        jstring cache_dir_path,
        jboolean load_merged_weights,
        jstring core_config_id,
        jstring memory_config_id,
        jstring execution_provider,
        jboolean enable_profiling
        ) {

    // If we load from merged weights, then we assume it is stored in the same directory as the inference model
    // -> inference_model_path/merged
    std::unique_ptr<InferenceSessionCache> session_cache = std::make_unique<InferenceSessionCache>(
            utils::JString2String(env, inference_model_path),
            utils::JString2String(env, inference_model_name),
            utils::JString2String(env, cache_dir_path),
            utils::JString2String(env, memory_config_id),
            utils::JString2String(env, core_config_id),
            utils::JString2String(env, execution_provider),
            load_merged_weights,
            enable_profiling);

    session_cache->initializeKVCache(1);

    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
/**
 * Deletes the current inference session.
 *
 * @param env
 * @param session
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_releaseInferenceSession(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    delete session_cache->inference_session;
    delete session_cache;
    session_cache = nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_releaseTrainingSession(
        JNIEnv *env, jobject,
        jlong session, jboolean saveCheckpoint) {
    ReleaseTrainingSession(session, saveCheckpoint);
}

extern "C"
JNIEXPORT void JNICALL
/**
 * Utility function, example of inspecting weights in the model from the checkpoint state.
 * */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_inspectWeights(JNIEnv *env, jobject thiz,
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
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_exportModelForInference(JNIEnv *env, jobject thiz,
                                                                   jlong session) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);
    session_cache->training_session.ExportModelForInferencing(session_cache->artifact_paths.inference_model_path, {"logits"});
    return env->NewStringUTF(session_cache->artifact_paths.inference_model_path.c_str());
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
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_performInferenceStep(JNIEnv *env, jobject thiz,
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

    // Forward pass
    auto logits = inference::generateWithKVCache(session_cache,
                                   input_ids_elements,
                                   attention_mask_elements,
                                   position_ids_elements,
                                   batch_size,
                                   sequence_length,
                                   past_sequence_length);

    int best_index = sampling::sampleNextToken(logits, past_sequence_length, vocab_size, session_cache->sampling_config, session_cache->random_generator);

    return best_index;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_createTokenizerSession(JNIEnv *env,
                                                                          jobject thiz,
                                                                          jstring jTokenizerFile) {
    // Convert Java string to C++ string
    const char *tokenizer_file = env->GetStringUTFChars(jTokenizerFile, nullptr);
    std::unique_ptr<TokenizerSessionCache> tokenizer = std::make_unique<TokenizerSessionCache>(tokenizer_file);

    // Release the Java string memory
    env->ReleaseStringUTFChars(jTokenizerFile, tokenizer_file);

    // Return the handle (cast the unique pointer to `jlong`)
    return reinterpret_cast<jlong>(tokenizer.release());
}


extern "C"
JNIEXPORT jintArray JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_tokenizeString(JNIEnv *env, jobject thiz,
                                                                  jlong tokenizer_model,
                                                                  jstring sequence) {
    // Convert Java string to C++ string
    const char *text = env->GetStringUTFChars(sequence, nullptr);

    std::vector<int32_t> tokens = tokenization::tokenize(tokenizer_model, text);

    // Release the Java string memory
    env->ReleaseStringUTFChars(sequence, text);
    jintArray token_array = env->NewIntArray(tokens.size());
    env->SetIntArrayRegion(token_array, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));
    return token_array;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_decodeString(JNIEnv *env, jobject thiz,
                                                                  jlong tokenizer_model,
                                                                  jintArray sequence) {

    // Convert Java int array to C++ vector
    jsize length = env->GetArrayLength(sequence);
    std::vector<int32_t> token_ids(length);
    env->GetIntArrayRegion(sequence, 0, length, reinterpret_cast<jint *>(token_ids.data()));

    // Decode the token IDs
    std::string decoded_text = tokenization::decode(tokenizer_model, token_ids);

    // Convert C++ string to Java string
    return env->NewStringUTF(decoded_text.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_releaseTokenizerSession(JNIEnv *env, jobject thiz, jlong tokenizer_model) {
    ReleaseTokenizerSession(tokenizer_model);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_decodeToken(JNIEnv *env, jobject thiz,
                                                               jlong tokenizer_model,
                                                               jint token_id) {

    // Decode the token IDs
    std::string decoded_text = tokenization::decodeToken(tokenizer_model, token_id);

    // Convert C++ string to Java string
    return env->NewStringUTF(decoded_text.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_optimizerStep(JNIEnv *env, jobject thiz,
                                                                jlong session) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    return training::optimizer_step(session_cache);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_setLearningRate(JNIEnv *env, jobject thiz,
                                                                  jlong session,
                                                                  jfloat learning_rate) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    session_cache->SetLearningRate(learning_rate);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_saveModel(JNIEnv *env, jobject thiz,
                                                            jlong session, jboolean saveOptimizer) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    Ort::CheckpointState::SaveCheckpoint(session_cache->checkpoint_state, session_cache->artifact_paths.checkpoint_path, saveOptimizer);
}

extern "C"
JNIEXPORT jboolean JNICALL
/**
 * Merges and exports the weights which are then ready for inference session.
 *
 * @param env
 * @param thiz
 * @param session
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_mergeExportWeights(JNIEnv *env, jobject thiz,
                                                                     jlong session,
                                                                     jstring peft_mapping_path,
                                                                     jstring merger_models_directory,
                                                                     jstring output_directory) {
    try {
        auto* session_cache = reinterpret_cast<TrainingSessionCache*>(session);

        // Convert Java strings to C++ strings
        const char* peft_path_cstr = env->GetStringUTFChars(peft_mapping_path, nullptr);
        const char* merger_models_dir_cstr = env->GetStringUTFChars(merger_models_directory, nullptr);
        const char* output_dir_cstr = env->GetStringUTFChars(output_directory, nullptr);

        std::string peft_path(peft_path_cstr);
        std::string merger_models_dir(merger_models_dir_cstr);
        std::string output_dir(output_dir_cstr);

        // Release Java strings
        env->ReleaseStringUTFChars(peft_mapping_path, peft_path_cstr);
        env->ReleaseStringUTFChars(merger_models_directory, merger_models_dir_cstr);
        env->ReleaseStringUTFChars(output_directory, output_dir_cstr);

        // Perform weight merging
        bool success = session_cache->weight_merger->merge_and_export_weights(
                session_cache->checkpoint_state,
                peft_path,
                merger_models_dir,
                output_dir
        );

        // Destroy the old WeightMerger instance and create a new one for next time
        session_cache->weight_merger.reset();
        session_cache->weight_merger = nullptr;
        session_cache->weight_merger = std::make_unique<WeightMerger>();

        return success ? JNI_TRUE : JNI_FALSE;

    } catch (const std::exception& e) {
        LOGE("Error in mergeExportWeights: %s", e.what());
        return JNI_FALSE;
    }
}

// Additional JNI function to configure sampling parameters
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_setSamplingConfig(JNIEnv *env, jobject thiz,
                                                                      jlong session,
                                                                      jint sampling_method,
                                                                      jfloat temperature,
                                                                      jint top_k,
                                                                      jfloat top_p,
                                                                      jint random_seed) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    auto method = static_cast<sampling::SamplingMethod>(sampling_method);

    session_cache->setSamplingConfig(method, temperature, top_k, top_p, random_seed);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_ORTRetriever_createEmbeddingSession(JNIEnv *env, jobject thiz,
                                                                     jstring embedding_model_path,
                                                                     jstring embedding_model_name,
                                                                     jstring cache_dir_path,
                                                                     jstring memory_config_id,
                                                                     jstring core_config_id,
                                                                     jstring execution_provider,
                                                                     jboolean enable_profiling) {
    try {
        // Convert Java strings to C++ strings
        const char *model_path_chars = env->GetStringUTFChars(embedding_model_path, nullptr);
        const char *model_name_chars = env->GetStringUTFChars(embedding_model_name, nullptr);
        const char *cache_path_chars = env->GetStringUTFChars(cache_dir_path, nullptr);
        const char *memory_config_chars = env->GetStringUTFChars(memory_config_id, nullptr);
        const char *core_config_chars = env->GetStringUTFChars(core_config_id, nullptr);
        const char *execution_provider_chars = env->GetStringUTFChars(execution_provider, nullptr);

        std::string model_path_str(model_path_chars);
        std::string model_name_str(model_name_chars);
        std::string cache_path_str(cache_path_chars);
        std::string memory_config_str(memory_config_chars);
        std::string core_config_str(core_config_chars);
        std::string execution_provider_str(execution_provider_chars);

        // Release Java string references
        env->ReleaseStringUTFChars(embedding_model_path, model_path_chars);
        env->ReleaseStringUTFChars(embedding_model_name, model_name_chars);
        env->ReleaseStringUTFChars(cache_dir_path, cache_path_chars);
        env->ReleaseStringUTFChars(memory_config_id, memory_config_chars);
        env->ReleaseStringUTFChars(core_config_id, core_config_chars);
        env->ReleaseStringUTFChars(execution_provider, execution_provider_chars);

        LOGI("Creating embedding session with model: %s", model_name_str.c_str());

        // Create the embedding session cache
        auto *embedding_session = new EmbeddingSessionCache(
                model_path_str,
                model_name_str,
                cache_path_str,
                memory_config_str,
                core_config_str,
                execution_provider_str,
                static_cast<bool>(enable_profiling)
        );

        LOGI("Embedding session created successfully");

        // Return the pointer as jlong
        return reinterpret_cast<jlong>(embedding_session);

    } catch (const std::exception &e) {
        LOGE("Failed to create embedding session: %s", e.what());

        // Throw Java exception
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class != nullptr) {
            env->ThrowNew(exception_class, e.what());
        }

        return 0;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTRetriever_releaseEmbeddingSession(JNIEnv *env, jobject thiz, jlong session) {
    try {

        auto *session_cache = reinterpret_cast<EmbeddingSessionCache *>(session);

        delete session_cache->embedding_session;
        delete session_cache;
        session_cache = nullptr;

    } catch (const std::exception& e) {
        LOGE("Failed to destroy embedding session: %s", e.what());
    }
}

extern "C"
JNIEXPORT jfloatArray JNICALL
/**
 * Performs the inference step using the exported model for inference.
 *
 * @param env
 * @param thiz
 * @param session
 * @param input_ids
 * @param attention_mask
 * @param token_type_ids
 * @param sequence_length
 *
 * @return Next token id
 */
Java_com_martinkorelic_mobiletransformers_ORTRetriever_performEmbeddingStep(JNIEnv *env, jobject thiz,
        jlong session,
        jlongArray input_ids,
        jlongArray attention_mask,
        jlongArray token_type_ids,
        jint batch_size,
        jint sequence_length,
        jint embedding_dim) {
    auto *session_cache = reinterpret_cast<EmbeddingSessionCache *>(session);

    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* attention_mask_elements = env->GetLongArrayElements(attention_mask, nullptr);
    jlong* token_type_ids_elements = env->GetLongArrayElements(token_type_ids, nullptr);

    // Forward pass
    auto embedding_vector = inference::generateEmbedding(session_cache,
                                        input_ids_elements,
                                     attention_mask_elements,
                                     token_type_ids_elements,
                                                 batch_size,
                                                 sequence_length);

    jint total_size = batch_size * embedding_dim;
    jfloatArray result = env->NewFloatArray(total_size);
    if (!result) {
        LOGE("Failed to create result float array");
        // Clean up the embedding vector if it was dynamically allocated
        delete[] embedding_vector; // or appropriate cleanup based on your memory management
        return nullptr;
    }

    // Copy data to Java array
    env->SetFloatArrayRegion(result, 0, total_size, embedding_vector);

    return result;
}