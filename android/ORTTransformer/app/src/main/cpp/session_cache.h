//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_SESSION_CACHE_H
#define ORT_PERSONALIZE_SESSION_CACHE_H

#include "onnxruntime_training_cxx_api.h"
#include "onnxruntime-genai/ort_genai.h"
#include "onnxruntime-genai/ort_genai_c.h"
#include <android/log.h>

struct ArtifactPaths {
    std::string checkpoint_path;
    std::string training_model_path;
    std::string eval_model_path;
    std::string optimizer_model_path;
    std::string cache_dir_path;
    std::string inference_model_path;

    ArtifactPaths(const std::string &checkpoint_path, const std::string &training_model_path,
                  const std::string &eval_model_path, const std::string &optimizer_model_path,
                  const std::string& cache_dir_path) :
            checkpoint_path(checkpoint_path), training_model_path(training_model_path),
            eval_model_path(eval_model_path), optimizer_model_path(optimizer_model_path),
            cache_dir_path(cache_dir_path), inference_model_path(cache_dir_path + "/inference.onnx") {}
};

/**
 * Caches the current trainable weights, which are then ready to be transferred into the inference session.
 * This is useful when we are loading the weights which have been changed into the custom inference model.
 * This avoids using export model for inference function. Release memory after the weights have been transferred.
 */
struct WeightSessionCache {
    // TODO: Change type
    std::unordered_map<std::string, Ort::Value> weights;
};

/**
 * Caches the current inference session variables. This should be released if a training session wants to begin or inference has ended.
 */
struct InferenceSessionCache {
    Ort::Env ort_env;
    Ort::Session* inference_session;
    Ort::SessionOptions session_options;
    std::string inference_model_path;
    std::string inference_model_name;

    // KV cache
    std::vector<std::unique_ptr<Ort::Value>> past_key_values;

    // Input and output names
    std::vector<std::string> string_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Model attributes
    int head_dim = 0;
    int num_layers = 0;
    int num_kv_heads = 0;

    InferenceSessionCache(const std::string& inference_model_path, const std::string& inference_model_name,
                          const std::string& sessionOptionsId, const bool enable_profiling) :
            ort_env(ORT_LOGGING_LEVEL_WARNING, "ORTInference"),
            inference_model_path(inference_model_path),
            inference_model_name(inference_model_name),
            inference_session(nullptr) {

        // TODO: Create and modify session options with new configurations
        // TODO: Import external initializers into the graph

        inference_session = std::make_unique<Ort::Session>(
                ort_env, inference_model_path.c_str(),
                setSessionOptions(sessionOptionsId, enable_profiling, inference_model_path)).release();

        loadModelMetadata();
        generateInputOutputNames();
    }

// Function to initialize the KV cache with provided batch size and sequence length
void initializeKVCache(int batch_size) {
    auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    past_key_values.clear();  // Clear any existing key-values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Pre-allocate the vector to avoid reallocations
    past_key_values.reserve(num_layers * 2); // *2 because we store both key and value

    // Initialize KV cache based on model configuration
    for (int i = 0; i < num_layers; ++i) {
        size_t element_count = batch_size * num_kv_heads * 1 * head_dim;

        // Create tensors for key and value, initialized to size 0
        // Each layer has 2 tensors (key, value) of shape (batch_size, num_heads, 0, head_size)
        std::vector<int64_t> kv_shape = {batch_size, num_kv_heads, 0, head_dim};

        // Create zero-initialized data for key tensor
        std::vector<float> key_data(element_count, 0.0f);

        OrtValue* new_k;
        ortApi->CreateTensorWithDataAsOrtValue(
                memory_info,
                key_data.data(), key_data.size() * sizeof(float), kv_shape.data(), kv_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &new_k);

        std::unique_ptr<Ort::Value> kv_key = std::make_unique<Ort::Value>(new_k);

        // Create empty tensor for key
        past_key_values.push_back(std::move(kv_key)); // Empty for first pass

        // Create zero-initialized data for key tensor
        std::vector<float> value_data(element_count, 0.0f);
        OrtValue* new_v;
        ortApi->CreateTensorWithDataAsOrtValue(
                memory_info,
                value_data.data(), value_data.size() * sizeof(float), kv_shape.data(), kv_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &new_v);
        std::unique_ptr<Ort::Value> kv_value = std::make_unique<Ort::Value>(new_v);

        validateTensor(kv_value.get(), "key");

        // Create empty tensor for value
        past_key_values.push_back(std::move(kv_value)); // Empty for first pass
        const auto& present_kv = past_key_values[i];
    }
    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Generated %zu KV caches.", past_key_values.size());
}

// Method to update past key-values
void updatePastKeyValues(const std::vector<Ort::Value>& present_key_values) {
    auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Get memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Present size %zu", present_key_values.size());
    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Past size %zu", past_key_values.size());

    // Copy present key-values to past key-values
    for (size_t i = 0; i < present_key_values.size(); i++) {

        auto& kv = present_key_values[i];

        if (past_key_values[i]) {
            past_key_values[i].reset();  // Release existing before overwriting
        }

        const int64_t* kv_shape = kv.GetTensorTypeAndShapeInfo().GetShape().data();
        auto size_dim = kv.GetTensorTypeAndShapeInfo().GetShape();
        size_t element_count = kv.GetTensorTypeAndShapeInfo().GetElementCount();

        size_t kv_shape_size = kv.GetTensorTypeAndShapeInfo().GetDimensionsCount();

        // Get data pointer without copying
        const float* data_ptr = kv.GetTensorData<float>();

        // Allocate new memory for the copy
        float* kv_data = new float[element_count];

        // Get the data from the existing tensor and make a deep copy
        const float* src_data = kv.GetTensorData<float>();
        std::memcpy(kv_data, src_data, element_count * sizeof(float));

        OrtValue* new_kv;
        ortApi->CreateTensorWithDataAsOrtValue(
                memory_info, kv_data, element_count * sizeof(float),
                kv_shape, kv_shape_size, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &new_kv);

        past_key_values[i] = std::make_unique<Ort::Value>(new_kv);
    }
}

private:
    static Ort::SessionOptions setSessionOptions(const std::string& config_id, const bool enable_profiling, const std::string& artifact_path) {
        Ort::SessionOptions options;

        if (config_id == "low_mem") {
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (config_id == "high_perf") {
            // Set options for high performance, if any.
            options.EnableMemPattern();
        }
        // Additional configurations based on config_id can be added here.
        if (enable_profiling) {
            auto profiling_file_path = artifact_path + "/profile.json";
            options.EnableProfiling(profiling_file_path.c_str());
        }

        return options;
    }

    void loadModelMetadata() {

        // Define the keys we want to extract
        const std::vector<std::string> metadata_keys = {"head_dim", "num_kv_heads", "num_layers"};

        auto metadata = inference_session->GetModelMetadata();

        // Extract all keys in a loop
        for (const auto& key : metadata_keys) {
            Ort::AllocatorWithDefaultOptions allocator;
            auto value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);

            if (value_ptr) {
                std::string value_str(value_ptr.get());
                if (key == "head_dim") {
                    head_dim = std::stoi(value_str);
                    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Head dimension: %d", head_dim);
                } else if (key == "num_layers") {
                    num_layers = std::stoi(value_str);
                    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Number of layers: %d", num_layers);
                } else if (key == "num_kv_heads") {
                    num_kv_heads = std::stoi(value_str);
                    __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Number of KV heads: %d", num_kv_heads);
                }
            }

        }
    }

    void validateTensor(const Ort::Value* tensor, const char* tensor_type) {
        if (!tensor) {
            throw std::runtime_error(std::string("Failed to create ") + tensor_type + " tensor");
        }
        if (!tensor->IsTensor()) {
            throw std::runtime_error(std::string(tensor_type) + " is not a tensor");
        }
    }

    void generateInputOutputNames() {
        // Clear existing data
        string_storage.clear();
        input_names.clear();
        output_names.clear();

        // Reserve space
        string_storage.reserve(4 + num_layers * 4);
        input_names.reserve(3 + num_layers * 2);
        output_names.reserve(1 + num_layers * 2);

        string_storage.push_back("input_ids");
        input_names.push_back(string_storage[0].c_str());
        string_storage.push_back("attention_mask");
        input_names.push_back(string_storage[1].c_str());
        string_storage.push_back("position_ids");
        input_names.push_back(string_storage[2].c_str());
        string_storage.push_back("logits");
        output_names.push_back(string_storage[3].c_str());

        // Generate names
        for (int i = 0; i < num_layers; i++) {
            string_storage.push_back("past_key_values." + std::to_string(i) + ".key");
            string_storage.push_back("past_key_values." + std::to_string(i) + ".value");
            string_storage.push_back("present." + std::to_string(i) + ".key");
            string_storage.push_back("present." + std::to_string(i) + ".value");

            size_t current_size = string_storage.size();
            input_names.push_back(string_storage[current_size - 4].c_str());
            input_names.push_back(string_storage[current_size - 3].c_str());
            output_names.push_back(string_storage[current_size - 2].c_str());
            output_names.push_back(string_storage[current_size - 1].c_str());
        }

    }

    // Optionally, a method to clear past key values for a new inference run
    void clearPastKeyValues() {
        past_key_values.clear();
    }

};

/**
 * Caches the current training session variables. This should be released after the training session is complete.
 */
struct TrainingSessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::CheckpointState checkpoint_state;
    Ort::SessionOptions session_options;
    Ort::TrainingSession training_session;
    std::vector<std::string> requires_grad;

    TrainingSessionCache(const std::string &checkpoint_path, const std::string &training_model_path,
                 const std::string &eval_model_path, const std::string &optimizer_model_path,
                 const std::string& cache_dir_path, const std::string sessionOptionsId, const bool enable_profiling) :
            artifact_paths(checkpoint_path, training_model_path, eval_model_path, optimizer_model_path, cache_dir_path),
            ort_env(ORT_LOGGING_LEVEL_WARNING, "ORTTraining"),
            checkpoint_state(Ort::CheckpointState::LoadCheckpoint(artifact_paths.checkpoint_path.c_str())),
            training_session(ort_env, setSessionOptions(sessionOptionsId, enable_profiling, training_model_path), checkpoint_state, artifact_paths.training_model_path.c_str(),
                             artifact_paths.eval_model_path.c_str(), artifact_paths.optimizer_model_path.c_str()) {

    }

private:
    static Ort::SessionOptions setSessionOptions(const std::string& config_id, const bool enable_profiling, const std::string& artifact_path) {
        Ort::SessionOptions options;

        if (config_id == "low_mem") {
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (config_id == "high_perf") {
            // Set options for high performance, if any.
            options.EnableMemPattern();
        }
        // Additional configurations based on config_id can be added here.
        if (enable_profiling) {
            auto profiling_file_path = artifact_path + "/profile.json";
            options.EnableProfiling(profiling_file_path.c_str());
        }

        return options;
    }

};

struct GenAISessionCache {
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaGenerator> generator;
    std::unique_ptr<OgaGeneratorParams> generatorParams;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizer_stream;

    GenAISessionCache(WeightSessionCache *weight_cache,
                      const std::string &genai_folder_path) {
        model = OgaModel::Create(genai_folder_path.c_str());
        generatorParams = OgaGeneratorParams::Create(*model);

        for (const auto& weight_pair : weight_cache->weights) {
            const std::string& layer_name = weight_pair.first;
            const Ort::Value& weight_value = weight_pair.second;
            __android_log_print(ANDROID_LOG_DEBUG, "SessionCache", "Loading %s into GenAI model...", layer_name.c_str());

            Ort::TensorTypeAndShapeInfo tensor_info = weight_value.GetTensorTypeAndShapeInfo();
            auto tensor_shape = tensor_info.GetShape();
            auto tensor_type = tensor_info.GetElementType();
            size_t total_elements = tensor_info.GetElementCount();
            auto element_type = static_cast<OgaElementType>(tensor_type);

            void* raw_data = const_cast<void *>(weight_value.GetTensorRawData());

            // Create OgaTensor from the buffer
            OgaTensor* oga_tensor = nullptr;
            OgaCreateTensorFromBuffer(raw_data, tensor_shape.data(), tensor_shape.size(), element_type, &oga_tensor);
            // Add the buffer as a model input
            OgaGeneratorParamsSetModelInput(generatorParams.get(), layer_name.c_str(), oga_tensor);
        }

        std::string text = "Hello, this is a message for the world. How is your day?";
        auto sequences = OgaSequences::Create();

        tokenizer = std::unique_ptr<OgaTokenizer>(OgaTokenizer::Create(*model));
        tokenizer_stream = std::unique_ptr<OgaTokenizerStream>(OgaTokenizerStream::Create(*tokenizer));

        tokenizer->Encode(text.c_str(), *sequences);
        generatorParams->SetInputSequences(*sequences);

        generator = OgaGenerator::Create(*model, *generatorParams);
    }
};
#endif //ORT_PERSONALIZE_SESSION_CACHE_H