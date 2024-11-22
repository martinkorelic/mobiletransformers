//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_SESSION_CACHE_H
#define ORT_PERSONALIZE_SESSION_CACHE_H

#include "onnxruntime_training_cxx_api.h"
#include "onnxruntime-genai/ort_genai.h"
#include "onnxruntime-genai/ort_genai_c.h"
#include "nnapi_provider_factory.h"
#include "tokenizers/tokenizers_cpp.h"
#include <android/log.h>
#include "utils.h"
#include "filesystem"

namespace fs = std::filesystem;

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

class path;

class path;

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
    bool has_position_ids;

    // Model attributes
    int head_dim = 0;
    int num_layers = 0;
    int num_kv_heads = 0;

    // Allocator for profiling
    Ort::AllocatorWithDefaultOptions allocator_;
    bool enable_profiling;
    std::string profiling_path;


    InferenceSessionCache(const std::string& inference_model_path, const std::string& inference_model_name,
                          const std::string& sessionOptionsId, const bool enable_profiling) :
            ort_env(ORT_LOGGING_LEVEL_VERBOSE, "ORTInference"),
            inference_model_path(inference_model_path),
            inference_model_name(inference_model_name),
            enable_profiling(enable_profiling),
            inference_session(nullptr) {

        // TODO: Create and modify session options with new configurations
        // TODO: Import external initializers into the graph
        const std::string full_path = inference_model_path + "/" + inference_model_name;

        inference_session = std::make_unique<Ort::Session>(
                ort_env, full_path.c_str(),
                setSessionOptions(sessionOptionsId, inference_model_path)).release();

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
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Copy present key-values to past key-values
        for (size_t i = 0; i < present_key_values.size(); i++) {
            const auto& kv = present_key_values[i];

            // Clear existing value if any
            if (past_key_values[i]) {
                past_key_values[i].reset();
            }

            // Move the value directly without copying data
            past_key_values[i] = std::make_unique<Ort::Value>(std::move(const_cast<Ort::Value&>(kv)));
        }
    }
//void updatePastKeyValues(const std::vector<Ort::Value>& present_key_values) {
//    auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//
//    // Get memory info
//    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//
//    // Copy present key-values to past key-values
//    for (size_t i = 0; i < present_key_values.size(); i++) {
//
//        auto& kv = present_key_values[i];
//
//        if (past_key_values[i]) {
//            past_key_values[i].reset();  // Release existing before overwriting
//        }
//
//        const int64_t* kv_shape = kv.GetTensorTypeAndShapeInfo().GetShape().data();
//        auto size_dim = kv.GetTensorTypeAndShapeInfo().GetShape();
//        size_t element_count = kv.GetTensorTypeAndShapeInfo().GetElementCount();
//
//        size_t kv_shape_size = kv.GetTensorTypeAndShapeInfo().GetDimensionsCount();
//
//        // Get data pointer without copying
//        const float* data_ptr = kv.GetTensorData<float>();
//
//        // Allocate new memory for the copy
//        std::unique_ptr<float[]> kv_data(new float[element_count]);
//
//        // Get the data from the existing tensor and make a deep copy
//        const float* src_data = kv.GetTensorData<float>();
//        std::memcpy(kv_data.get(), src_data, element_count * sizeof(float));
//
//        OrtValue* new_kv;
//        ortApi->CreateTensorWithDataAsOrtValue(
//                memory_info, kv_data.get(), element_count * sizeof(float),
//                kv_shape, kv_shape_size, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &new_kv);
//
//        past_key_values[i] = std::make_unique<Ort::Value>(new_kv);
//        kv_data.release();
//    }
//}

// End profiling and get the profile data
std::string endProfiling() {
    try {
        // Use EndProfilingAllocated to get the profile path
        //auto profile_path = inference_session->EndProfilingAllocated(allocator_);
        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        char* profile_path = nullptr;
        // End profiling and get the profile path
        auto status = ortApi->SessionEndProfiling(*inference_session, allocator_, &profile_path);

        if (!profile_path || status != nullptr) {
            throw Ort::Exception("EndProfilingAllocated returned null", ORT_RUNTIME_EXCEPTION);
        }

        // Convert to string and free the allocated memory
        std::string path_str(profile_path);

        __android_log_print(ANDROID_LOG_INFO, "InferenceProfiler",
                            "Profiling completed. Output saved to: %s", path_str.c_str());

        return path_str;
    } catch (const Ort::Exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "InferenceProfiler",
                            "Failed to end profiling: %s", e.what());
        throw;
    }
}

// Start profiling
void startProfiling() {
    __android_log_print(ANDROID_LOG_ERROR, "InferenceProfiler","Started profiling to: %s", profiling_path.c_str());
    auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);


    auto status = ortApi->EnableProfiling(session_options, profiling_path.c_str());

    // Check the status
    if (status != nullptr) {
        // An error occurred, retrieve and log the error message
        const char* error_message = ortApi->GetErrorMessage(status);
        __android_log_print(ANDROID_LOG_ERROR, "Profiler", "Failed to enable profiling: %s", error_message);

        // Release the status
        ortApi->ReleaseStatus(status);
        throw Ort::Exception(error_message, ORT_RUNTIME_EXCEPTION); // Optionally throw an exception
    } else {
        __android_log_print(ANDROID_LOG_INFO, "Profiler", "Profiling enabled successfully.");
    }


}

private:
    Ort::SessionOptions setSessionOptions(const std::string& config_id, const std::string& artifact_path) {
        Ort::SessionOptions options;

        if (config_id == "low_mem") {
            // This might cause memory issues
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (config_id == "high_perf") {
            // Set options for high performance, if any.
            //options.DisableMemPattern();
            //options.DisableCpuMemArena();
            options.AddConfigEntry("session.enable_quant_qdq_cleanup", "0");
            options.AddConfigEntry("session.disable_quant_qdq", "0");
        }
        // Set profile path file
        if (enable_profiling) {
            profiling_path = artifact_path + "inference_profile.json";
            fs::path profile_file_prefix{profiling_path};
            allocator_ = Ort::AllocatorWithDefaultOptions();
        }

        // TODO: Set based on configurations
        //NNAPI execution provider

//        uint32_t nnapi_flags = 0;
//        nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
//        nnapi_flags |= NNAPI_FLAG_USE_FP16;
//
//        auto status = OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags);

        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
//
//        const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
//        const size_t values[] = {0 /*let ort pick default max memory*/, 0, 1024, 0, 256};
//        const size_t num_keys = sizeof(keys) / sizeof(keys[0]);
//
//        OrtArenaCfg* arena_cfg = nullptr;
//        ortApi->CreateArenaCfgV2(keys, values, 5, &arena_cfg);
//
//        OrtMemoryInfo* mem_info;
//        ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
//
//        std::unordered_map<std::string, std::string> arena_options = {
//                {"max_mem", "0"},                          // Let ONNX Runtime pick the default max memory
//                {"arena_extend_strategy", "0"},            // Default arena extend strategy (next power of 2)
//                {"initial_chunk_size_bytes", "1024"},      // Initial chunk size in bytes
//                {"max_dead_bytes_per_chunk", "0"},         // Max dead bytes per chunk (can be 0 for no limit)
//                {"initial_growth_chunk_size_bytes", "256"} // Initial growth chunk size in bytes
//        };
//       ort_env.CreateAndRegisterAllocatorV2(
//                "CPUExecutionProvider",
//                mem_info,
//                arena_options,
//                arena_cfg);

        char** providers;
        int provider_length;
        ortApi->GetAvailableProviders(&providers, &provider_length);
        for (int i = 0; i < provider_length; i++) {
            __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Error when setting NNAPI support: %s",providers[i]);
        }

        session_options.SetInterOpNumThreads(1);
        session_options.SetIntraOpNumThreads(1);
        session_options.AddConfigEntry("session.dynamic_block_base", "2");
        session_options.AddConfigEntry("session.use_device_allocator_for_initializers","1");
        session_options.AddConfigEntry("session.use_env_allocators","1");
        //session_options.AddConfigEntry("session.intra_op_thread_affinities","1;2");
        session_options.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        session_options.AddConfigEntry("session.use_ort_model_bytes_for_initializers","0");
        session_options.AddConfigEntry("session.qdqisint8allowed", "1");
        session_options.AddConfigEntry("session.disable_double_qdq_remover","0");

        //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        //Set execution mode to sequential
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

//        if (status != nullptr) {
//            // Print or log the error message
//            const char* error_message = Ort::GetApi().GetErrorMessage(status);
//            __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Error when setting NNAPI support: %s", error_message);
//
//            // Release the status object
//            Ort::GetApi().ReleaseStatus(status);
//
//            throw std::runtime_error("Error setting NNAPI execution provider.");
//        } else {
//            __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "NNAPI execution provider successfully set.");
//        }

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
        has_position_ids = false;

        // Query the number of inputs and outputs
        size_t num_inputs = inference_session->GetInputCount();
        size_t num_outputs = inference_session->GetOutputCount();

        // Pre-allocate storage to prevent reallocation
        string_storage.reserve(num_inputs + num_outputs);
        input_names.reserve(num_inputs);
        output_names.reserve(num_outputs);

        __android_log_print(ANDROID_LOG_DEBUG, "InputNames", "%zu", num_inputs);
        __android_log_print(ANDROID_LOG_DEBUG, "InputNames", "%zu", num_outputs);

        Ort::AllocatorWithDefaultOptions allocator;

        // Populate input names
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = inference_session->GetInputNameAllocated(i, allocator);
            std::string name(input_name.get());

            if (name == "position_ids") {
                has_position_ids = true;
            }

            string_storage.push_back(std::move(name));
            input_names.push_back(string_storage.back().c_str());
        }

        // Populate output names
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = inference_session->GetOutputNameAllocated(i, allocator);
            string_storage.push_back(std::string(output_name.get()));
            output_names.push_back(string_storage.back().c_str());
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

struct TokenizerSessionCache {
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;

    TokenizerSessionCache(const std::string &tokenizer_file) {
        // TODO: For now only works with HF tokenizers
        auto data = utils::LoadBytesFromFile(tokenizer_file);
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(data);
    }
};

#endif //ORT_PERSONALIZE_SESSION_CACHE_H