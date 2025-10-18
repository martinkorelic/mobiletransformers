//
// Created by martinkorelic on 20. 07. 25.
//

#ifndef ORTTRANSFORMER_WEIGHT_MERGER_H
#define ORTTRANSFORMER_WEIGHT_MERGER_H

#include "weight_serializer.h"
#include <jni.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <fstream>
#include <filesystem>
#include <android/log.h>
#include <nlohmann/json.hpp>
#include "logging.h"

struct PeftMapping {
    std::string adapter_B;
    int rank;
    float alpha;
    std::string shared_A;
    std::string intermediate;
    int adapter_index;
    std::string adapter_A; // For LoRA
};

struct BaseLayerParams {
    std::unique_ptr<Ort::Value> weight_quantized;
    std::unique_ptr<Ort::Value> x_scale;
    std::unique_ptr<Ort::Value> x_zero_point;
    std::unique_ptr<Ort::Value> weight;

    // Buffer tracking for custom allocated memory
    void* weight_quantized_buffer = nullptr;
    void* x_scale_buffer = nullptr;
    void* x_zero_point_buffer = nullptr;
    void* weight_buffer = nullptr;

    bool has_quantized = false;
    bool has_weight = false;
};

struct AdapterParams {
    std::unique_ptr<Ort::Value> data;
    void* raw_buffer = nullptr;  // Raw pointer to user-managed memory
};

struct MergedOutput {
    std::unique_ptr<Ort::Value> merged_weight_quantized;
    std::unique_ptr<Ort::Value> merged_zero_point;
    std::unique_ptr<Ort::Value> merged_scale;
    std::unique_ptr<Ort::Value> merged_weight;

    // Buffer tracking for custom allocated memory
    void* merged_weight_quantized_buffer = nullptr;
    void* merged_zero_point_buffer = nullptr;
    void* merged_scale_buffer = nullptr;
    void* merged_weight_buffer = nullptr;

    bool has_quantized = false;
    bool has_weight = false;
};

class WeightMerger {
private:
    struct ParameterTracker {
        std::string base_layer_name;
        std::vector<std::string> used_base_params;
        std::vector<std::string> used_adapter_params;

        ParameterTracker(const std::string& layer_name);
    };

    std::unordered_map<std::string, PeftMapping> peft_mapping_;
    std::unordered_map<std::string, BaseLayerParams> base_layer_params_;
    std::unordered_map<std::string, std::unordered_map<std::string, AdapterParams>> adapter_params_;
    std::unordered_map<std::string, MergedOutput> merged_outputs_;
    std::unordered_map<std::string, std::unique_ptr<Ort::Session>> merger_sessions_;

    Ort::MemoryInfo memory_info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Helper function to create a copy of OrtValue for user-managed memory
    std::pair<std::unique_ptr<Ort::Value>, void*> CreateUserManagedCopy(const Ort::Value& original);

    // Helper function to get parameter if it matches expected type
    std::optional<Ort::Value> GetParameterIfType(
            const OrtCheckpointState* checkpoint_state,
            const char* parameter_name,
            ONNXTensorElementDataType expected_type);

    // Helper function to create scalar tensors
    template<typename T>
    std::unique_ptr<Ort::Value> CreateScalarTensor(T value);

    // Helper function to get tensor shape
    std::vector<int64_t> get_tensor_shape(const Ort::Value& tensor);

    // Replace prefix in parameter name
    std::string replace_prefix(const std::string& name, const std::string& old_prefix, const std::string& new_prefix);

    // Load and parse PEFT mapping from JSON
    bool load_peft_mapping(const std::string& json_path);

    // Extract base layer parameters from checkpoint
    void extract_base_layer_params(Ort::CheckpointState& checkpoint_state);

    // Extract adapter parameters from checkpoint
    void extract_adapter_params(Ort::CheckpointState& checkpoint_state);

    // Load ONNX merger models
    bool load_merger_models(const std::string& models_directory);

    // Determine the appropriate merger type based on available parameters
    std::string get_merger_type(const std::string& base_layer_name);

    // Run the appropriate merger model
    void run_merger_model(const std::string& merger_type, const std::string& base_layer_name);

    // Free used parameters after merging
    void free_used_parameters(const ParameterTracker& tracker);

    // Helper function to convert OrtValue to vector for saving
    template<typename T>
    std::vector<T> ortvalue_to_vector(const Ort::Value& tensor);

    // Save merged parameters using OrtValueSerializer
    void save_merged_parameters(const std::string& output_directory);

    // Helper function to create correct tensor names for inference initializers
    std::string inference_name(const std::string& layer_name);

public:
    WeightMerger();

    // Main method to perform weight merging
    bool merge_and_export_weights(Ort::CheckpointState& checkpoint_state,
                                  const std::string& peft_mapping_path,
                                  const std::string& merger_models_directory,
                                  const std::string& output_directory);
};

#endif //ORTTRANSFORMER_WEIGHT_MERGER_H