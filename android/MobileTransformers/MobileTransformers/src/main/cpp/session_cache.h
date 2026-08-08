//
// Created by martinkorelic on 19/09/2024.
//

#ifndef SESSION_CACHE_H
#define SESSION_CACHE_H

#include "onnxruntime/onnxruntime_training_cxx_api.h"
#include "onnxruntime/nnapi_provider_factory.h"
#include "tokenizers/tokenizers_cpp.h"
#include <android/log.h>
#include "utils.h"
#include "filesystem"
#include "weight_merger.h"
#include "sampling.h"
#include "logging.h"
#include "mem_probe.h"    // #12: RSS probe (Gate 0.2)
#include "mmap_tensor.h"  // #12: RAII mmap region (default-off zero-copy load)

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

    std::unordered_map<std::string, Ort::Value > weights;
    std::unordered_map<std::string, void*> allocated_buffers; // Track allocated memory
    // #12: mmap'd regions backing zero-copy external initializers (default-off). Must outlive the
    // Ort::Values that point into them, so they are freed only in clearWeights()/destruction.
    std::vector<MmapRegion> mmap_regions_;

    Ort::MemoryInfo memory_info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Constructor
    WeightSessionCache() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {}

    // #23: initialize the cache from FLAT per-tensor <name>.bin files in the inference dir, keyed by
    // weight_handoff_map.json (the ONE reader in handoff_io.h). No <dirname>.<filestem> reconstruction:
    // initializer names come straight from inferenceInitializerNames[role]. dtype/shape are validated
    // by the map itself (per role); a missing file or size/dtype/shape mismatch fails closed (clears the
    // cache and returns false so the caller adds NO external initializers, never a partial/wrong set).
    // Checksums are already enforced by the Kotlin precondition (HandoffPrecondition) before this runs.
    bool init(const std::string& inference_dir) {
        try {
            LOG_RSS("weight-load:start");
            // #12 (Gate 0.2, default-off): load each fp per-tensor .bin zero-copy via mmap instead of the
            // buffered copy. Opt-in only — the shipping default is the #23 copy path, so this never
            // perturbs the hardened load unless it is switched on.
            const bool use_mmap = memprobe::mmap_weights_enabled();

            const std::string map_path = inference_dir + "/weight_handoff_map.json";
            std::unordered_map<std::string, HandoffEntry> handoff;
            if (!load_handoff_entries(map_path, /*readerVersion=*/"1.0", handoff)) {
                LOGE("Handoff map missing/invalid at %s; not loading merged weights", map_path.c_str());
                return false;
            }

            for (const auto& [layer, entry] : handoff) {
                for (const auto& [role, bin_name] : entry.externalDataLocation) {
                    const std::string init_name = entry.inferenceInitializerNames.count(role)
                            ? entry.inferenceInitializerNames.at(role) : bin_name;
                    const std::string path = inference_dir + "/" + bin_name;
                    if (!std::filesystem::exists(path)) {
                        LOGE("Merged weight file missing: %s (layer %s, role %s)", path.c_str(),
                             layer.c_str(), role.c_str());
                        clearWeights();
                        return false;
                    }

                    // mmap path: only well-shaped, non-quantized tensors (the frozen-base RSS win). Any
                    // other tensor falls back to the copy path so quantized/scalar cases stay correct.
                    // dtype/shape are read PER ROLE (see load_tensor_raw).
                    ONNXTensorElementDataType mmap_type =
                            use_mmap ? onnx_type_from_string(entry.dtype_for(role))
                                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                    if (use_mmap && !entry.shape_for(role).empty() && !entry.has_quantization &&
                        mmap_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                        MmapRegion region(path);
                        if (!region.valid()) {
                            LOGE("mmap failed for %s; failing closed", path.c_str());
                            clearWeights();
                            return false;
                        }
                        std::vector<int64_t> shape = entry.shape_for(role);
                        Ort::Value v = Ort::Value::CreateTensor(
                                memory_info_, const_cast<void*>(region.data()), region.size(),
                                shape.data(), shape.size(), mmap_type);
                        weights.emplace(init_name, std::move(v));
                        mmap_regions_.push_back(std::move(region));
                        LOGI("mmap'd initializer: %s <- %s (%zu bytes)", init_name.c_str(), bin_name.c_str(),
                             mmap_regions_.back().size());
                        continue;
                    }

                    // Copy path (the shipping default). load_tensor_raw already fails closed on dtype,
                    // shape and byte-size mismatch, so there is no separate post-hoc validation step:
                    // the Ort::Value is constructed FROM the map's declaration, not checked against it.
                    auto [loaded_tensor, buffer_ptr] = load_tensor_raw(path, entry, role, init_name);
                    if (!loaded_tensor) {
                        LOGE("Failed to load merged weight: %s", path.c_str());
                        clearWeights();
                        return false;
                    }
                    weights.emplace(init_name, std::move(loaded_tensor));
                    allocated_buffers[init_name] = buffer_ptr;
                    LOGI("Loaded merged initializer: %s <- %s", init_name.c_str(), bin_name.c_str());
                }
            }

            LOGI("Weight cache initialized with %zu external initializers (mmap=%d)", weights.size(),
                 use_mmap ? 1 : 0);
            LOG_RSS("weight-load:end");
            return !weights.empty();

        } catch (const std::exception& e) {
            LOGE("Error initializing weight cache: %s", e.what());
            clearWeights();
            return false;
        }
    }

    // Map the handoff dtype string onto an ORT element type (int4 packs into uint8 storage on device).
    static ONNXTensorElementDataType onnx_type_from_string(const std::string& s) {
        if (s == "float32" || s == "float") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        if (s == "float16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        if (s == "int8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        if (s == "uint8" || s == "int4") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        if (s == "int32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    // Byte width of an ORT element type (int4 is stored packed in uint8 on device).
    static size_t element_size(ONNXTensorElementDataType t) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 4;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 1;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 4;
            default:                                    return 0;
        }
    }

    // Load one flat <name>.bin as RAW external-data bytes into an allocator-owned Ort::Value.
    //
    // #23 correctness fix: these files are ONNX external-data blobs (raw tensor bytes, no header) —
    // that is what every writer emits: the exporter via onnx.write_external_data_tensors, and the
    // device merger via weight_merger.cpp::write_raw_tensor_atomic. The previous implementation parsed
    // them as serialized onnx::TensorProto, which cannot succeed on raw bytes, so the shipping (non-
    // mmap) merged-weight load failed on every device run. Shape/dtype therefore come from the handoff
    // map -- PER ROLE, since a packed weight_quantized/scale/zero_point is not shaped like its weight.
    //
    // Fail-closed: an unknown dtype, an absent/empty shape, or a file whose size is not exactly
    // numel * element_size throws rather than handing ORT a mis-shaped buffer.
    std::pair<Ort::Value, void*> load_tensor_raw(const std::string& filepath,
                                                 const HandoffEntry& entry,
                                                 const std::string& role,
                                                 const std::string& init_name) {
        const std::string& dtype_s = entry.dtype_for(role);
        const std::vector<int64_t>& shape = entry.shape_for(role);

        const ONNXTensorElementDataType elem_type = onnx_type_from_string(dtype_s);
        const size_t elem_size = element_size(elem_type);
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED || elem_size == 0) {
            throw std::runtime_error("unsupported dtype '" + dtype_s + "' for " + init_name);
        }
        if (shape.empty()) {
            throw std::runtime_error(
                    "no shape declared for role '" + role + "' of " + init_name +
                    " (raw external data carries no header; the map must declare tensorShapes)");
        }

        size_t numel = 1;
        for (int64_t d : shape) {
            if (d <= 0) throw std::runtime_error("non-positive dim in shape of " + init_name);
            numel *= static_cast<size_t>(d);
        }
        const size_t expected_bytes = numel * elem_size;

        std::error_code ec;
        const auto file_size = fs::file_size(filepath, ec);
        if (ec) throw std::runtime_error("cannot stat " + filepath + ": " + ec.message());
        if (static_cast<size_t>(file_size) != expected_bytes) {
            throw std::runtime_error(
                    "size mismatch for " + init_name + " (" + filepath + "): file has " +
                    std::to_string(file_size) + " bytes, map declares " + dtype_s + " " +
                    std::to_string(numel) + " elements = " + std::to_string(expected_bytes) + " bytes");
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filepath);

        void* buffer = allocator_.Alloc(expected_bytes);
        if (buffer == nullptr) throw std::runtime_error("allocation failed for " + init_name);
        if (!file.read(static_cast<char*>(buffer), static_cast<std::streamsize>(expected_bytes))) {
            allocator_.Free(buffer);
            throw std::runtime_error("short read for " + init_name + " from " + filepath);
        }

        Ort::Value value = Ort::Value::CreateTensor(
                memory_info_, buffer, expected_bytes,
                const_cast<int64_t*>(shape.data()), shape.size(), elem_type);
        return {std::move(value), buffer};
    }

    // Clear all cached weights with explicit cleanup
    void clearWeights() {
        if (!weights.empty()) {
            LOGI("Clearing weight cache with %zu tensors", weights.size());
        }

        // Free all allocated buffers
        for (const auto& [tensor_name, buffer_ptr] : allocated_buffers) {
            if (buffer_ptr) {
                //LOGI("Freeing buffer for tensor: %s", tensor_name.c_str());
                allocator_.Free(buffer_ptr);
            }
        }

        // Clear the maps
        weights.clear();
        allocated_buffers.clear();
        // #12: unmap any mmap'd regions (after the Ort::Values that referenced them are cleared).
        mmap_regions_.clear();

        LOGI("Weight cache cleared and memory freed");
    }

    // Check if tensor exists
    bool has_tensor(const std::string& layer_name) const {
        return weights.find(layer_name) != weights.end();
    }

    // Clear all cached weights
    void clear() {
        weights.clear();
    }
};


struct EmbeddingSessionCache {
    Ort::Env ort_env;
    Ort::Session* embedding_session;
    Ort::SessionOptions session_options;
    std::string embedding_model_path;
    std::string embedding_model_name;
    std::string cache_dir_path;

    // Input and output names
    std::vector<std::string> string_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Allocator for profiling
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string profiling_path;

    bool has_token_type_ids;

    EmbeddingSessionCache(const std::string& embedding_model_path,
                          const std::string& embedding_model_name,
                          const std::string& cache_dir_path,
                          const std::string& memoryConfigId,
                          const std::string& coreConfigId,
                          const std::string& executionProvider,
                          const bool enable_profiling = false) :
            ort_env(ORT_LOGGING_LEVEL_ERROR, "ORTEmbedding"),
            embedding_model_path(embedding_model_path),
            embedding_model_name(embedding_model_name),
            cache_dir_path(cache_dir_path),
            embedding_session(nullptr) {

        const std::string full_path = embedding_model_path + "/" + embedding_model_name;

        embedding_session = std::make_unique<Ort::Session>(
                ort_env, full_path.c_str(),
                setSessionOptions(memoryConfigId, coreConfigId, enable_profiling, embedding_model_path, executionProvider)).release();

        // Post initialization
        generateInputOutputNames();
    }

    ~EmbeddingSessionCache() {
        delete embedding_session;
    }

private:
    Ort::SessionOptions setSessionOptions(const std::string& memory_config_id, const std::string& core_config_id, const bool enable_profiling, const std::string& artifact_path, const std::string& execution_provider = "cpu") {
        Ort::SessionOptions options;

        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        char** providers;
        int provider_length;
        ortApi->GetAvailableProviders(&providers, &provider_length);
        for (int i = 0; i < provider_length; i++) {
            LOGI("Available execution providers: %s", providers[i]);
        }

        if (memory_config_id == "low_mem") {
            // This might cause memory issues
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (core_config_id == "high_perf") {
            // Set options for high performance, if any.
            options.AddConfigEntry("session.enable_quant_qdq_cleanup", "0");
            options.AddConfigEntry("session.disable_quant_qdq", "0");
        }

        if (core_config_id == "opt3") {
            options.SetInterOpNumThreads(6);
            options.SetIntraOpNumThreads(4);
        } else if (core_config_id == "opt2") {
            options.SetInterOpNumThreads(4);
            options.SetIntraOpNumThreads(2);
        } else {
            options.SetInterOpNumThreads(1);
            options.SetIntraOpNumThreads(1);
        }

        // Set execution provider
        if (execution_provider == "xnnpack") {
            try {
                // XNNPACK execution provider
                options.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "1"}});

                // XNNPACK has a separate internal threadpool which can lead to contention with the ORT intra-op threadpool.
                // To minimize this, we recommend setting the following options:
                // https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html#configuration-options
                options.SetIntraOpNumThreads(1);

                LOGI("Using XNNPACK execution provider.");
            } catch (const std::exception& e) {
                LOGI("Failed to configure XNNPACK execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "nnapi") {
            try {
                // NNAPI execution provider (does NOT work with ONNX Runtime training)
                uint32_t nnapi_flags = 0;
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
                nnapi_flags |= NNAPI_FLAG_USE_FP16;

                auto status = OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapi_flags);

                if (status != nullptr) {
                    LOGI("Failed to configure NNAPI execution provider");
                    LOGI("Falling back to CPU execution provider");
                } else {
                    LOGI("Using NNAPI execution provider.");
                }
            } catch (const std::exception& e) {
                LOGI("Failed to configure NNAPI execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "cpu") {
            LOGI("Using CPU execution provider (default)");
        }
        else {
            LOGI("Unknown execution provider '%s', falling back to CPU", execution_provider.c_str());
        }

        // Set profile path file
        if (enable_profiling) {
            // Construct the logging directory path
            fs::path logging_dir = fs::path(cache_dir_path) / "logging" / "embedding";

            // Create the directory if it doesn't exist
            try {
                fs::create_directories(logging_dir);
                LOGI("Created profiling directory: %s", logging_dir.string().c_str());
            } catch (const fs::filesystem_error& e) {
                LOGE("Failed to create profiling directory: %s", e.what());
            }

            // Set the profiling file path
            profiling_path = (logging_dir / "embedding_profile.json").string();
            LOGI("Profiling will be saved to: %s", profiling_path.c_str());

            auto status = ortApi->EnableProfiling(options, profiling_path.c_str());

            // Check the status
            if (status != nullptr) {
                // An error occurred, retrieve and log the error message
                const char* error_message = ortApi->GetErrorMessage(status);

                // Release the status
                ortApi->ReleaseStatus(status);
                LOGE("Failed to enable profiling: %s", error_message);
            } else {
                LOGI("Profiling enabled successfully.");
            }
        }

        return options;
    }

    void generateInputOutputNames() {
        if (!embedding_session) {
            LOGE("Embedding session is null, cannot generate input/output names");
            return;
        }

        Ort::AllocatorWithDefaultOptions allocator;

        has_token_type_ids = false;

        // Clear existing names
        string_storage.clear();
        input_names.clear();
        output_names.clear();

        try {
            // Get input names
            size_t num_input_nodes = embedding_session->GetInputCount();

            // Reserve space to prevent reallocation
            string_storage.reserve(num_input_nodes + embedding_session->GetOutputCount());

            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = embedding_session->GetInputNameAllocated(i, allocator);
                std::string name_str(input_name.get());
                string_storage.push_back(name_str);
                // Don't push the pointer yet - wait until all strings are added
                LOGI("Input[%zu]: %s", i, name_str.c_str());
                if (name_str == "token_type_ids") {
                    has_token_type_ids = true;
                }
            }

            // Get output names
            size_t num_output_nodes = embedding_session->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = embedding_session->GetOutputNameAllocated(i, allocator);
                std::string name_str(output_name.get());
                string_storage.push_back(name_str);
                LOGI("Output[%zu]: %s", i, name_str.c_str());
            }

            // Now populate the pointer vectors after all strings are stored
            for (size_t i = 0; i < num_input_nodes; i++) {
                input_names.push_back(string_storage[i].c_str());
            }

            for (size_t i = 0; i < num_output_nodes; i++) {
                output_names.push_back(string_storage[num_input_nodes + i].c_str());
            }

            // Debug: Verify the pointers are correct
            LOGI("Verifying input names:");
            for (size_t i = 0; i < input_names.size(); i++) {
                LOGI("  input_names[%zu]: '%s'", i, input_names[i]);
            }

            LOGI("Verifying output names:");
            for (size_t i = 0; i < output_names.size(); i++) {
                LOGI("  output_names[%zu]: '%s'", i, output_names[i]);
            }

            LOGI("Successfully generated input/output names for embedding model");
        } catch (const std::exception& e) {
            LOGE("Failed to generate input/output names: %s", e.what());
        }
    }
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
    std::string cache_dir_path;

    // KV cache
    std::vector<std::unique_ptr<Ort::Value>> past_key_values;

    // Merged weight cache
    std::unique_ptr<WeightSessionCache> weight_session;
    bool load_external_weights;

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
    std::string profiling_path;

    // Sampling
    sampling::SamplingConfig sampling_config;
    sampling::RandomGenerator random_generator;

    InferenceSessionCache(const std::string& inference_model_path, const std::string& inference_model_name, const std::string& cache_dir_path,
                          const std::string& memoryConfigId, const std::string& coreConfigId, const std::string& executionProvider, const bool load_external_weights = false, const bool enable_profiling = false) :
            ort_env(ORT_LOGGING_LEVEL_ERROR, "ORTInference"),
            inference_model_path(inference_model_path),
            inference_model_name(inference_model_name),
            cache_dir_path(cache_dir_path),
            load_external_weights(load_external_weights),
            inference_session(nullptr) {

        const std::string full_path = inference_model_path + "/" + inference_model_name;

        inference_session = std::make_unique<Ort::Session>(
                ort_env, full_path.c_str(),
                setSessionOptions(memoryConfigId, coreConfigId, enable_profiling, inference_model_path, executionProvider)).release();

        // Release the weight session cache after they have been loaded in as external initializers
        if (load_external_weights) {
            if (weight_session) {
                weight_session->clearWeights();
            }
        }

        // Post initialization
        initializeSampling();
        loadModelMetadata();
        generateInputOutputNames();
    }

    void initializeSampling() {
        // Initialize with default greedy sampling
        sampling_config.method = sampling::SamplingMethod::GREEDY;
        sampling_config.temperature = 1.0f;
        sampling_config.top_k = 50;
        sampling_config.top_p = 0.9f;
    }

    // Function to initialize the KV cache with provided batch size and sequence length
    void initializeKVCache(int batch_size) {
        past_key_values.clear();  // Clear any existing key-values

        // The geometry comes from the graph's own metadata (loadModelMetadata). If it is missing, every
        // layer count is 0, we create no past tensors, and generateWithKVCache then declares the graph's
        // full input count while binding only 3 values — an out-of-bounds read inside ORT. Fail here,
        // where the cause is nameable, instead of there.
        if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
            throw std::runtime_error(
                    "model metadata is missing the KV-cache geometry (num_layers=" +
                    std::to_string(num_layers) + ", num_kv_heads=" + std::to_string(num_kv_heads) +
                    ", head_dim=" + std::to_string(head_dim) +
                    "). The exporter must stamp these into the ONNX metadata_props.");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        // Pre-allocate the vector to avoid reallocations
        past_key_values.reserve(num_layers * 2); // *2 because we store both key and value

        // Initialize KV cache based on model configuration
        for (int i = 0; i < num_layers; ++i) {
            // Each layer has 2 tensors (key, value) of shape (batch_size, num_kv_heads, 0, head_dim) —
            // zero-length past on the first pass, which is what a `*-with-past` graph expects.
            //
            // These are allocator-owned. They used to be created with CreateTensorWithDataAsOrtValue
            // over a `std::vector<float>` local to this loop iteration: that API borrows the caller's
            // buffer without copying, so every tensor in past_key_values pointed at freed memory the
            // moment the iteration ended.
            std::vector<int64_t> kv_shape = {batch_size, num_kv_heads, 0, head_dim};

            auto kv_key = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
                    allocator, kv_shape.data(), kv_shape.size()));
            validateTensor(kv_key.get(), "key");
            past_key_values.push_back(std::move(kv_key));

            auto kv_value = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
                    allocator, kv_shape.data(), kv_shape.size()));
            validateTensor(kv_value.get(), "value");
            past_key_values.push_back(std::move(kv_value));
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

            // Log present key-value tensor shape
            auto tensor_info = kv.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            // Clear existing value if any
            if (past_key_values[i]) {
                past_key_values[i].reset();
            }

            // Move the value directly without copying data
            past_key_values[i] = std::make_unique<Ort::Value>(std::move(const_cast<Ort::Value&>(kv)));
        }
    }

    // Method to update sampling configuration
    void setSamplingConfig(sampling::SamplingMethod method,
                           float temperature = 1.0f,
                           int top_k = 50,
                           float top_p = 0.9f,
                           unsigned int seed = 0) {
        sampling_config.method = method;
        sampling_config.temperature = temperature;
        sampling_config.top_k = top_k;
        sampling_config.top_p = top_p;
        sampling_config.random_seed = seed;

        if (seed != 0) {
            random_generator.setSeed(seed);
        }
    }

private:
    Ort::SessionOptions setSessionOptions(const std::string& memory_config_id, const std::string& core_config_id, const bool enable_profiling, const std::string& artifact_path, const std::string& execution_provider = "cpu") {
        Ort::SessionOptions options;

        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        char** providers;
        int provider_length;
        ortApi->GetAvailableProviders(&providers, &provider_length);
        for (int i = 0; i < provider_length; i++) {
            LOGI("Available execution providers: %s",providers[i]);
        }

        if (memory_config_id == "low_mem") {
            // This might cause memory issues
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (core_config_id == "high_perf") {
            // Set options for high performance, if any.
            options.AddConfigEntry("session.enable_quant_qdq_cleanup", "0");
            options.AddConfigEntry("session.disable_quant_qdq", "0");
        }

        if (core_config_id == "opt3") {
            options.SetInterOpNumThreads(6);
            options.SetIntraOpNumThreads(4);
        } else if (core_config_id == "opt2") {
            options.SetInterOpNumThreads(4);
            options.SetIntraOpNumThreads(2);
        } else {
            options.SetInterOpNumThreads(1);
            options.SetIntraOpNumThreads(1);
        }

        // Set execution provider
        if (execution_provider == "xnnpack") {

            try {
                // XNNPACK execution provider
                options.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "1"}});

                // XNNPACK has a separate internal threadpool which can lead to contention with the ORT intra-op threadpool.
                // To minimize this, we recommend setting the following options:
                // https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html#configuration-options
                options.SetIntraOpNumThreads(1);

                LOGI("Using XNNPACK execution provider.");
            } catch (const std::exception& e) {
                LOGI("Failed to configure XNNPACK execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "nnapi") {

            try {
                // NNAPI execution provider (does NOT work with ONNX Runtime training)
                uint32_t nnapi_flags = 0;
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
                nnapi_flags |= NNAPI_FLAG_USE_FP16;

                auto status = OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapi_flags);

                if (status != nullptr) {
                    LOGI("Failed to configure NNAPI execution provider");
                    LOGI("Falling back to CPU execution provider");
                } else {
                    LOGI("Using NNAPI execution provider.");
                }
            } catch (const std::exception& e) {
                LOGI("Failed to configure NNAPI execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "cpu") {
            LOGI("Using CPU execution provider (default)");
        }
        else {
            LOGI("Unknown execution provider '%s', falling back to CPU", execution_provider.c_str());
        }

        // Set profile path file
        if (enable_profiling) {
            // Construct the logging directory path
            fs::path logging_dir = fs::path(cache_dir_path) / "logging" / "inference";

            // Create the directory if it doesn't exist
            try {
                fs::create_directories(logging_dir);
                LOGI("Created profiling directory: %s", logging_dir.string().c_str());
            } catch (const fs::filesystem_error& e) {
                LOGE("Failed to create profiling directory: %s", e.what());
            }

            // Set the profiling file path
            profiling_path = (logging_dir / "inference_profile.json").string();
            LOGI("Profiling will be saved to: %s", profiling_path.c_str());

            // Set the profiling file path
            profiling_path = (logging_dir / "profile.json").string();

            auto status = ortApi->EnableProfiling(options, profiling_path.c_str());

            // Check the status
            if (status != nullptr) {
                // An error occurred, retrieve and log the error message
                const char* error_message = ortApi->GetErrorMessage(status);

                // Release the status
                ortApi->ReleaseStatus(status);
                LOGE("Failed to enable profiling: %s", error_message);
            } else {
                LOGI("Profiling enabled successfully.");
            }
        }

        // #23: load merged weights as flat per-tensor external initializers from the inference dir,
        // keyed by weight_handoff_map.json (retired the inference/merged subdir). The Kotlin
        // precondition already checksum-verified these before we got here.
        if (load_external_weights) {
            const std::string weights_path = inference_model_path;
            weight_session = std::make_unique<WeightSessionCache>();

            if (!weight_session->init(weights_path)) {
                // #23 fail-closed: merged weights were explicitly REQUESTED and could not be loaded.
                // Logging and falling through would build the session with the frozen base initializers
                // — an untrained model that looks healthy, which is exactly what #23's DoD forbids.
                // The Kotlin precondition catches missing files and bad checksums, but dtype/shape and
                // byte-size validation are C++-only, so this is the only place those can be caught.
                weight_session.reset();
                throw std::runtime_error(
                        "merged weights requested but WeightSessionCache::init failed for " + weights_path +
                        " (see prior log lines for the offending tensor); refusing to fall back to base weights");
            } else {
                LOGI("Successfully initialized WeightSessionCache from path: %s", weights_path.c_str());

                // Add external initializers to the session options
                try {
                    std::vector<std::string> initializer_names;
                    std::vector<Ort::Value> initializer_values;

                    // Collect names and values from the weight cache
                    for (const auto& weight_pair : weight_session->weights) {
                        const std::string& name = weight_pair.first;
                        const Ort::Value& tensor = weight_pair.second;

                        // Get tensor info and print dimensions
                        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
                        auto shape = tensor_info.GetShape();

                        // Build dimension string
                        std::string shape_str = "[";
                        for (size_t i = 0; i < shape.size(); ++i) {
                            if (i > 0) shape_str += ", ";
                            shape_str += std::to_string(shape[i]);
                        }
                        shape_str += "]";

                        LOGI("Adding external initializer: %s, dimensions: %s", name.c_str(), shape_str.c_str());

                        initializer_names.push_back(name);
                        // We need to move or copy the OrtValue since it's stored in the map
                        initializer_values.emplace_back(std::move(const_cast<Ort::Value&>(weight_pair.second)));
                    }

                    if (!initializer_names.empty()) {
                        options.AddExternalInitializers(initializer_names, initializer_values);
                        LOGI("Successfully added %zu external initializers to session options", initializer_names.size());
                    }
                } catch (const std::exception& e) {
                    // ORT rejects a replacement whose name/dims/dtype disagree with the graph's external
                    // initializer. Swallowing that threw away the one signal that the merged tensors do
                    // not match the model — same fail-closed rule as init() above.
                    LOGE("Error adding external initializers: %s", e.what());
                    weight_session.reset();
                    throw std::runtime_error(
                            std::string("merged weights requested but AddExternalInitializers failed: ") +
                            e.what() + "; refusing to fall back to base weights");
                }
            }
        }

        options.AddConfigEntry("session.dynamic_block_base", "2");
        options.AddConfigEntry("session.use_device_allocator_for_initializers","1");
        options.AddConfigEntry("session.use_env_allocators","1");
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        //session_options.AddConfigEntry("session.intra_op_thread_affinities","1;2");
        options.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        options.AddConfigEntry("session.use_ort_model_bytes_for_initializers","0");
        options.AddConfigEntry("session.qdqisint8allowed", "1");
        options.AddConfigEntry("session.disable_double_qdq_remover","0");
        //Set execution mode to sequential
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

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
 * Caches the current training session. This should be released after the training session is complete.
 */
struct TrainingSessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::CheckpointState checkpoint_state;
    Ort::SessionOptions session_options;
    Ort::TrainingSession training_session;

    // List of layers that are trainable
    std::vector<std::string> requires_grad;

    // Weight merger for training session
    std::unique_ptr<WeightMerger> weight_merger;

    // Profiling
    std::string profiling_path;
    Ort::AllocatorWithDefaultOptions allocator_;

    TrainingSessionCache(const std::string &checkpoint_path, const std::string &training_model_path,
                 const std::string &eval_model_path, const std::string &optimizer_model_path,
                 const std::string& cache_dir_path, const std::string memoryConfigId, const std::string coreConfigId, const std::string executionProvider, const bool enable_profiling) :
            artifact_paths(checkpoint_path, training_model_path, eval_model_path, optimizer_model_path, cache_dir_path),
            ort_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ORTTraining")),
            checkpoint_state(Ort::CheckpointState::LoadCheckpoint(artifact_paths.checkpoint_path.c_str())),
            training_session(ort_env, setSessionOptions(memoryConfigId, coreConfigId, enable_profiling, training_model_path, executionProvider), checkpoint_state, artifact_paths.training_model_path.c_str(),
                             artifact_paths.eval_model_path.c_str(), artifact_paths.optimizer_model_path.c_str()),
            weight_merger(std::make_unique<WeightMerger>()) {}

    // Batch size of more than 1 can significantly increase memory usage and there is no way to disable that as of now
    // Perhaps it has something to do with arena allocation
    static Ort::Env setupSharedAllocator(Ort::Env env) {
        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        // Create memory info for CPU
        OrtMemoryInfo* mem_info;
        ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        // Configure arena for MINIMAL memory usage
        const char* keys[] = {
                "max_mem",                           // Maximum memory limit
                "arena_extend_strategy",             // How arena grows
                "initial_chunk_size_bytes",          // Initial allocation size
                "initial_growth_chunk_size_bytes",   // Growth size after shrinkage
                "max_dead_bytes_per_chunk"           // Prevent memory waste
        };

        const size_t values[] = {
                67108864,   // 64MB total limit (very conservative)
                1,          // kSameAsRequested - only allocate what's needed
                32768,      // 32KB initial chunk (small initial allocation)
                32768,      // 32KB growth after shrinkage
                4096        // 4KB max dead bytes (minimize waste)
        };

        OrtArenaCfg* arena_cfg = nullptr;
        ortApi->CreateArenaCfgV2(keys, values, 5, &arena_cfg);

        try {
            // Use the correct method signature - no status returned
            env.CreateAndRegisterAllocatorV2(
                    "CPUExecutionProvider",
                    mem_info,
                    {},  // Empty additional options
                    arena_cfg
            );

            __android_log_print(ANDROID_LOG_INFO, "SessionCache", "Shared allocator registered successfully");

        } catch (const Ort::Exception& e) {
            __android_log_print(ANDROID_LOG_ERROR, "SessionCache", "Failed to register allocator: %s", e.what());
        }

        // Clean up
        ortApi->ReleaseMemoryInfo(mem_info);
        ortApi->ReleaseArenaCfg(arena_cfg);

        return env;
    }

    void SetLearningRate(float learning_rate) {
        try {
            training_session.SetLearningRate(learning_rate);
        } catch (const Ort::Exception& e) {
            // Log error or handle exception as needed
            LOGE("Failed to set learning rate: %s", e.what());
        }
    }

private:
    Ort::SessionOptions setSessionOptions(const std::string& memory_config_id, const std::string& core_config_id, const bool enable_profiling, const std::string& artifact_path, const std::string& execution_provider = "cpu") {
        Ort::SessionOptions options;

        auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        char** providers;
        int provider_length;
        ortApi->GetAvailableProviders(&providers, &provider_length);
        for (int i = 0; i < provider_length; i++) {
            LOGI("Available execution providers: %s",providers[i]);
        }

        // OpNumThreads
        // Aggresive inter - 6, intra - 4
        // Medium inter - 4, intra - 2
        // Lowest inter - 1, intra - 1
        if (core_config_id == "opt3") {
            options.SetInterOpNumThreads(6);
            options.SetIntraOpNumThreads(4);
        } else if (core_config_id == "opt2") {
            options.SetInterOpNumThreads(4);
            options.SetIntraOpNumThreads(2);
        } else {
            options.SetInterOpNumThreads(1);
            options.SetIntraOpNumThreads(1);
        }

        //options.AddConfigEntry("session.intra_op_thread_affinities", "1;2");

        options.AddConfigEntry("session.memory.enable_memory_arena_shrinkage", "1");
        options.AddConfigEntry("session.dynamic_block_base", "2");
        options.AddConfigEntry("session.use_device_allocator_for_initializers","1");
        options.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        options.AddConfigEntry("session.use_ort_model_bytes_for_initializers","0");
        options.AddConfigEntry("session.qdqisint8allowed", "1");
        options.AddConfigEntry("session.disable_double_qdq_remover","0");

        // Training
        //options.AddConfigEntry("optimization.enable_memory_optimizer","Gelu+Cast+:1:0,Dropout+:1:1");
        //options.AddConfigEntry("optimization.enable_memory_probe_recompute_level","1");

        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Set execution mode to sequential
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        if (memory_config_id == "low_mem") {
            options.DisableMemPattern();
            options.DisableCpuMemArena();
        } else if (memory_config_id == "high_perf") {
            options.EnableMemPattern();
            options.EnableCpuMemArena();
        }

        if (execution_provider == "xnnpack") {

            try {
                // XNNPACK execution provider
                options.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "4"}});

                // XNNPACK has a separate internal threadpool which can lead to contention with the ORT intra-op threadpool.
                // To minimize this, we recommend setting the following options:
                // https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html#configuration-options
                options.SetIntraOpNumThreads(4);

                LOGI("Using XNNPACK execution provider.");
            } catch (const std::exception& e) {
                LOGI("Failed to configure XNNPACK execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "nnapi") {

            try {
                // NNAPI execution provider (does NOT work with ONNX Runtime training)
                uint32_t nnapi_flags = 0;
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
                nnapi_flags |= NNAPI_FLAG_USE_FP16;

                auto status = OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapi_flags);

                if (status != nullptr) {
                    LOGI("Failed to configure NNAPI execution provider");
                    LOGI("Falling back to CPU execution provider");
                } else {
                    LOGI("Using NNAPI execution provider.");
                }
            } catch (const std::exception& e) {
                LOGI("Failed to configure NNAPI execution provider: %s", e.what());
                LOGI("Falling back to CPU execution provider");
            }
        }
        else if (execution_provider == "cpu") {
            LOGI("Using CPU execution provider (default)");
        }
        else {
            LOGI("Unknown execution provider '%s', falling back to CPU", execution_provider.c_str());
        }

        // Set profile path
        if (enable_profiling) {
            // Construct the logging directory path
            fs::path logging_dir = fs::path(artifact_paths.cache_dir_path) / "logging" / "training";

            // Create the directory if it doesn't exist
            try {
                fs::create_directories(logging_dir);
                LOGI("Created profiling directory: %s", logging_dir.string().c_str());

            } catch (const fs::filesystem_error& e) {
                LOGE("Failed to create profiling directory: %s", e.what());
            }

            // Set the profiling file path
            profiling_path = (logging_dir / "profile.json").string();

            auto status = ortApi->EnableProfiling(options, profiling_path.c_str());

            // Check the status
            if (status != nullptr) {
                // An error occurred, retrieve and log the error message
                const char* error_message = ortApi->GetErrorMessage(status);

                // Release the status
                ortApi->ReleaseStatus(status);
                LOGE("Failed to enable profiling: %s", error_message);
            } else {
                LOGI("Profiling enabled successfully.");
            }

        }

        return options;
    }

};

/**
 * Struct for tokenizer session.\n
 * NOTE: For now only works with HF tokenizers
 */
struct TokenizerSessionCache {
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;

    TokenizerSessionCache(const std::string &tokenizer_file) {
        auto data = utils::LoadBytesFromFile(tokenizer_file);
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(data);
    }
};
#endif // SESSION_CACHE_H