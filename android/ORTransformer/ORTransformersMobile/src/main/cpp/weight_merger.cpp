//
// Created by martinkorelic on 20. 07. 25.
//

#include "weight_merger.h"
#include <jni.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <android/log.h>
#include <nlohmann/json.hpp>
#include "logging.h"


using json = nlohmann::json;

// ParameterTracker constructor implementation
WeightMerger::ParameterTracker::ParameterTracker(const std::string& layer_name)
        : base_layer_name(layer_name) {
}

// WeightMerger constructor implementation
WeightMerger::WeightMerger()
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}



// Helper function to create a copy of OrtValue for user-managed memory
std::pair<std::unique_ptr<Ort::Value>, void*> WeightMerger::CreateUserManagedCopy(const Ort::Value& original) {
    auto tensor_info = original.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> tensor_shape = tensor_info.GetShape();
    auto tensor_type = tensor_info.GetElementType();
    size_t total_elements = tensor_info.GetElementCount();
    size_t element_size = 0;

    // Determine the size of one element based on tensor type
    switch (tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            element_size = sizeof(float);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            element_size = sizeof(int16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            element_size = sizeof(int32_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            element_size = sizeof(int64_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            element_size = sizeof(int8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            element_size = sizeof(uint8_t);
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type");
    }

    // Allocate memory for the tensor data
    size_t data_size = total_elements * element_size;
    void* user_data = allocator_.Alloc(data_size);

    // Copy data from the original tensor
    std::memcpy(user_data, original.GetTensorRawData(), data_size);

    // Create new tensor with user-managed data
    auto& ortApi = Ort::GetApi();
    OrtValue* c_tensor;
    auto ortStatus = ortApi.CreateTensorWithDataAsOrtValue(
            memory_info_, user_data, data_size,
            tensor_shape.data(), tensor_shape.size(),
            tensor_type, &c_tensor);

    if (ortStatus != nullptr) {
        const char* error_message = ortApi.GetErrorMessage(ortStatus);
        ortApi.ReleaseStatus(ortStatus);
        allocator_.Free(user_data);
        throw std::runtime_error("Failed to create tensor with user-managed data: " + std::string(error_message));
    }

    return std::make_pair(std::make_unique<Ort::Value>(c_tensor), user_data);
}

std::optional<Ort::Value> WeightMerger::GetParameterIfType(
        const OrtCheckpointState* checkpoint_state,
        const char* parameter_name,
        ONNXTensorElementDataType expected_type) {

    Ort::AllocatorWithDefaultOptions allocator;

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const OrtTrainingApi* training_api = api->GetTrainingApi(ORT_API_VERSION);

    // Check parameter type first
    OrtTensorTypeAndShapeInfo* type_info = nullptr;
    OrtStatus* status = training_api->GetParameterTypeAndShape(
            checkpoint_state, parameter_name, &type_info);

    if (status != nullptr) {
        api->ReleaseStatus(status);
        return std::nullopt; // Parameter doesn't exist
    }

    ONNXTensorElementDataType actual_type;
    status = api->GetTensorElementType(type_info, &actual_type);
    api->ReleaseTensorTypeAndShapeInfo(type_info);

    if (status != nullptr) {
        api->ReleaseStatus(status);
        return std::nullopt;
    }

    if (actual_type != expected_type) {
        LOGI("Parameter %s type mismatch: expected %d, got %d",
             parameter_name, expected_type, actual_type);
        return std::nullopt;
    }
    // Get the shape information
    size_t dim_count = 0;
    status = api->GetDimensionsCount(type_info, &dim_count);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        api->ReleaseTensorTypeAndShapeInfo(type_info);
        return std::nullopt;
    }

    std::vector<int64_t> shape(dim_count);
    status = api->GetDimensions(type_info, shape.data(), dim_count);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        api->ReleaseTensorTypeAndShapeInfo(type_info);
        return std::nullopt;
    }

    // Create an OrtValue with the correct type and shape
    OrtValue* parameter = nullptr;
    status = api->CreateTensorAsOrtValue(
            allocator,
            shape.data(),
            dim_count,
            actual_type,
            &parameter
    );

    if (status != nullptr) {
        const char* error_message = api->GetErrorMessage(status);
        LOGI("CreateTensorAsOrtValue failed: %s", error_message);
        api->ReleaseStatus(status);
        return std::nullopt;
    }

    //LOGI("Created tensor element type: %d (UINT8=%d, FLOAT=%d)", created_type,
    //     ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    if (status != nullptr) {
        api->ReleaseStatus(status);
        return std::nullopt;
    }

    // Now copy the parameter data into our pre-allocated tensor
    // NOTE: This kept failing in source code, as it always created float parameter even though we had quantized parameters
    // This fix needs to be do in source code (orttraining/orttraining/training_api/onnxruntime_training_c_api.cc::640)
    status = training_api->GetParameter(checkpoint_state, parameter_name, allocator, &parameter);

    if (status != nullptr) {
        const char* error_message = api->GetErrorMessage(status);
        LOGI("Error getting parameter type and shape for %s: %s", parameter_name, error_message);
        api->ReleaseStatus(status);
        return std::nullopt;
    }

    return Ort::Value(parameter);
}

template<typename T>
std::unique_ptr<Ort::Value> WeightMerger::CreateScalarTensor(T value) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Scalar tensor has empty shape (0 dimensions)
    std::vector<int64_t> shape = {};

    // Allocate memory for single value
    std::vector<T> data = {value};

    auto tensor = Ort::Value::CreateTensor<T>(
            memory_info,
            data.data(),
            1,  // single element
            shape.data(),
            shape.size()
    );

    return std::make_unique<Ort::Value>(std::move(tensor));
}

// Helper function to get tensor shape
std::vector<int64_t> WeightMerger::get_tensor_shape(const Ort::Value& tensor) {
    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

// Replace prefix in parameter name
std::string WeightMerger::replace_prefix(const std::string& name, const std::string& old_prefix, const std::string& new_prefix) {
    if (name.substr(0, old_prefix.length()) == old_prefix) {
        return new_prefix + name.substr(old_prefix.length());
    }
    return name;
}

// Load and parse PEFT mapping from JSON
bool WeightMerger::load_peft_mapping(const std::string& json_path) {
    try {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            LOGE("Failed to open PEFT mapping file: %s", json_path.c_str());
            return false;
        }

        json j;
        file >> j;

        if (!j.contains("peft_mapping")) {
            LOGE("JSON file does not contain 'peft_mapping' key");
            return false;
        }

        for (const auto& [base_layer_name, mapping_data] : j["peft_mapping"].items()) {
            PeftMapping mapping;

            if (mapping_data.contains("adapter_B")) {
                mapping.adapter_B = mapping_data["adapter_B"];
            }
            if (mapping_data.contains("rank")) {
                mapping.rank = mapping_data["rank"];
            }
            if (mapping_data.contains("alpha")) {
                mapping.alpha = mapping_data["alpha"];
            }
            if (mapping_data.contains("shared_A")) {
                mapping.shared_A = mapping_data["shared_A"];
            }
            if (mapping_data.contains("intermediate")) {
                mapping.intermediate = mapping_data["intermediate"];
            }
            if (mapping_data.contains("adapter_index")) {
                mapping.adapter_index = mapping_data["adapter_index"];
            }
            if (mapping_data.contains("adapter_A")) {
                mapping.adapter_A = mapping_data["adapter_A"];
            }

            peft_mapping_[base_layer_name] = mapping;
            LOGI("Loaded PEFT mapping for: %s", base_layer_name.c_str());
        }

        LOGI("Successfully loaded %zu PEFT mappings", peft_mapping_.size());
        return true;
    } catch (const std::exception& e) {
        LOGE("Error loading PEFT mapping: %s", e.what());
        return false;
    }
}

// Extract base layer parameters from checkpoint
void WeightMerger::extract_base_layer_params(Ort::CheckpointState& checkpoint_state) {
    LOGI("Extracting base layer parameters...");

    for (const auto& [base_layer_name, _] : peft_mapping_) {
        std::string adjusted_name = replace_prefix(base_layer_name, "base_model.model.model.", "backbone.model.");

        BaseLayerParams base_params;

        // Look for different weight parameter types
        std::string weight_quantized_name = adjusted_name + ".weight_quantized";
        std::string weight_scale_name = adjusted_name + ".weight_scale";
        std::string weight_zero_point_name = adjusted_name + ".weight_zero_point";
        std::string weight_name = adjusted_name + ".weight";

        // Try to get quantized weight
        auto quantized_tensor = GetParameterIfType(
                checkpoint_state,
                weight_quantized_name.c_str(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        );

        if (quantized_tensor.has_value()) {
            auto [tensor, buffer] = CreateUserManagedCopy(quantized_tensor.value());
            base_params.weight_quantized = std::move(tensor);
            base_params.weight_quantized_buffer = buffer;
            base_params.has_quantized = true;
            LOGI("Found quantized weight: %s", weight_quantized_name.c_str());
        } else {
            // Parameter doesn't exist or has wrong type
            LOGI("Quantized weight %s not found or has wrong type", weight_quantized_name.c_str());
        }

        // Try to get weight scale
        auto scale_tensor = GetParameterIfType(
                checkpoint_state,
                weight_scale_name.c_str(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT  // Assuming scales are float
        );

        if (scale_tensor.has_value()) {
            auto [tensor, buffer] = CreateUserManagedCopy(scale_tensor.value());
            base_params.x_scale = std::move(tensor);
            base_params.x_scale_buffer = buffer;
            LOGI("Found weight scale: %s", weight_scale_name.c_str());
        } else {
            LOGI("Weight scale %s not found or has wrong type", weight_scale_name.c_str());
        }

        // Try to get weight zero point
        auto zero_point_tensor = GetParameterIfType(
                checkpoint_state,
                weight_zero_point_name.c_str(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        );

        if (zero_point_tensor.has_value()) {
            auto [tensor, buffer] = CreateUserManagedCopy(zero_point_tensor.value());
            base_params.x_zero_point = std::move(tensor);
            base_params.x_zero_point_buffer = buffer;
            LOGI("Found weight zero point: %s", weight_zero_point_name.c_str());
        } else {
            LOGI("Weight zero point %s not found or has wrong type", weight_zero_point_name.c_str());
        }

        // Try to get regular weight
        auto weight_tensor = GetParameterIfType(
                checkpoint_state,
                weight_name.c_str(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT  // Regular weights are typically float
        );

        if (weight_tensor.has_value()) {
            auto [tensor, buffer] = CreateUserManagedCopy(weight_tensor.value());
            base_params.weight = std::move(tensor);
            base_params.weight_buffer = buffer;
            base_params.has_weight = true;
            LOGI("Found non-quantized weight: %s", weight_name.c_str());
        } else {
            LOGI("Non-quantized weight %s not found or has wrong type", weight_name.c_str());
        }

        if (base_params.has_quantized || base_params.has_weight) {
            base_layer_params_[adjusted_name] = std::move(base_params);
            LOGI("Extracted base layer params for: %s", adjusted_name.c_str());
        } else {
            LOGW("No parameters found for base layer: %s", adjusted_name.c_str());
        }
    }
}

// Extract adapter parameters from checkpoint
void WeightMerger::extract_adapter_params(Ort::CheckpointState& checkpoint_state) {
    LOGI("Extracting adapter parameters...");

    for (const auto& [base_layer_name, mapping] : peft_mapping_) {
        std::string adjusted_base_name = replace_prefix(base_layer_name, "base_model.model.model.", "backbone.model.");

        adapter_params_[adjusted_base_name] = std::unordered_map<std::string, AdapterParams>();

        // Extract adapter_B
        if (!mapping.adapter_B.empty()) {
            std::string adapter_name = replace_prefix(mapping.adapter_B, "base_model.model.model.", "backbone.model.");
            adapter_name += ".weight";

            try {
                Ort::Value tensor = checkpoint_state.GetParameter(adapter_name);
                AdapterParams params;
                auto [adapter_tensor, buffer] = CreateUserManagedCopy(tensor);
                params.data = std::move(adapter_tensor);
                params.raw_buffer = buffer;
                adapter_params_[adjusted_base_name]["adapter_B"] = std::move(params);
                LOGI("Found adapter_B param: %s", adapter_name.c_str());
            } catch (const std::exception& e) {
                LOGW("Parameter not found or error extracting adapter_B for %s: %s", adapter_name.c_str(), e.what());
            }
        }

        // Extract shared_A
        if (!mapping.shared_A.empty()) {
            std::string adapter_name = replace_prefix(mapping.shared_A, "base_model.model.model.", "backbone.model.");
            adapter_name += ".weight";

            try {
                Ort::Value tensor = checkpoint_state.GetParameter(adapter_name);
                AdapterParams params;
                auto [adapter_tensor, buffer] = CreateUserManagedCopy(tensor);
                params.data = std::move(adapter_tensor);
                params.raw_buffer = buffer;
                adapter_params_[adjusted_base_name]["shared_A"] = std::move(params);
                LOGI("Found shared_A param: %s", adapter_name.c_str());
            } catch (const std::exception& e) {
                LOGW("Parameter not found or error extracting shared_A for %s: %s", adapter_name.c_str(), e.what());
            }
        }

        // Extract intermediate
        if (!mapping.intermediate.empty()) {
            std::string adapter_name = replace_prefix(mapping.intermediate, "base_model.model.model.", "backbone.model.");
            adapter_name += ".weight";

            try {
                Ort::Value tensor = checkpoint_state.GetParameter(adapter_name);
                AdapterParams params;
                auto [adapter_tensor, buffer] = CreateUserManagedCopy(tensor);
                params.data = std::move(adapter_tensor);
                params.raw_buffer = buffer;
                adapter_params_[adjusted_base_name]["intermediate"] = std::move(params);
                LOGI("Found intermediate param: %s", adapter_name.c_str());
            } catch (const std::exception& e) {
                LOGW("Parameter not found or error extracting intermediate for %s: %s", adapter_name.c_str(), e.what());
            }
        }

        // Extract adapter_A (for LoRA)
        if (!mapping.adapter_A.empty()) {
            std::string adapter_name = replace_prefix(mapping.adapter_A, "base_model.model.model.", "backbone.model.");
            adapter_name += ".weight";

            try {
                Ort::Value tensor = checkpoint_state.GetParameter(adapter_name);
                AdapterParams params;
                auto [adapter_tensor, buffer] = CreateUserManagedCopy(tensor);
                params.data = std::move(adapter_tensor);
                params.raw_buffer = buffer;
                adapter_params_[adjusted_base_name]["adapter_A"] = std::move(params);
                LOGI("Found adapter_A param: %s", adapter_name.c_str());
            } catch (const std::exception& e) {
                LOGW("Parameter not found or error extracting adapter_A for %s: %s", adapter_name.c_str(), e.what());
            }
        }
    }
}

// Load ONNX merger models
bool WeightMerger::load_merger_models(const std::string& models_directory) {
    LOGI("Loading merger models from: %s", models_directory.c_str());

    try {
        // Load LoRA merger model (full precision)
        std::string lora_model_path = models_directory + "/lora_merger_model.onnx";
        merger_sessions_["lora"] = std::make_unique<Ort::Session>(
                Ort::Env(), lora_model_path.c_str(), Ort::SessionOptions{}
        );
        LOGI("Loaded LoRA merger model");

        // Load LoRA quantized merger model
        std::string lora_q_model_path = models_directory + "/lora_qmerger_model.onnx";
        merger_sessions_["lora_q"] = std::make_unique<Ort::Session>(
                Ort::Env(), lora_q_model_path.c_str(), Ort::SessionOptions{}
        );
        LOGI("Loaded LoRA quantized merger model");

        // Load MARS quantized merger model
        std::string mars_q_model_path = models_directory + "/mars_qmerger_model.onnx";
        merger_sessions_["mars_q"] = std::make_unique<Ort::Session>(
                Ort::Env(), mars_q_model_path.c_str(), Ort::SessionOptions{}
        );
        LOGI("Loaded MARS quantized merger model");

        return true;
    } catch (const std::exception& e) {
        LOGE("Error loading merger models: %s", e.what());
        return false;
    }
}

// Determine the appropriate merger type based on available parameters
std::string WeightMerger::get_merger_type(const std::string& base_layer_name) {
    auto adapter_it = adapter_params_.find(base_layer_name);
    if (adapter_it == adapter_params_.end()) {
        return "";
    }

    auto& adapters = adapter_it->second;
    bool has_shared_A = adapters.find("shared_A") != adapters.end();
    bool has_adapter_A = adapters.find("adapter_A") != adapters.end();
    bool has_quantized = base_layer_params_[base_layer_name].has_quantized;

    if (has_shared_A && has_quantized) {
        return "mars_q";  // MARS with quantized weights
    } else if (has_adapter_A && has_quantized) {
        return "lora_q";  // LoRA with quantized weights
    } else if (has_adapter_A && !has_quantized) {
        return "lora";    // LoRA with full precision weights
    }
        // TODO: Custom merger model?
    else {
        LOGW("Unable to determine merger type for: %s", base_layer_name.c_str());
        return "";
    }
}

void WeightMerger::run_merger_model(const std::string& merger_type, const std::string& base_layer_name) {
    LOGI("Running %s merger for: %s", merger_type.c_str(), base_layer_name.c_str());

    if (merger_sessions_.find(merger_type) == merger_sessions_.end()) {
        LOGE("Merger model not found: %s", merger_type.c_str());
        return;
    }

    try {
        auto& session = merger_sessions_[merger_type];
        auto& base_params = base_layer_params_[base_layer_name];
        auto& adapter_params = adapter_params_[base_layer_name];

        // Create parameter tracker
        ParameterTracker tracker(base_layer_name);

        // Prepare input tensors based on merger type
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        // Storage for scalar values (must persist during inference)
        float alpha_value = peft_mapping_[base_layer_name].alpha;
        int64_t adapter_index_value = peft_mapping_[base_layer_name].adapter_index;
        int64_t rank_value = peft_mapping_[base_layer_name].rank;

        if (merger_type == "lora") {
            // LoRA merger inputs: base_weight, adapter_A, adapter_B, alpha
            if (!base_params.weight) {
                LOGE("Missing base weight for LoRA merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.weight));
            input_names.push_back("weight");
            tracker.used_base_params.push_back("weight");

            if (adapter_params.find("adapter_A") == adapter_params.end() ||
                !adapter_params["adapter_A"].data) {
                LOGE("Missing adapter_A for LoRA merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_A"].data));
            input_names.push_back("adapter_A");
            tracker.used_adapter_params.push_back("adapter_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for LoRA merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_B"].data));
            input_names.push_back("adapter_B");
            tracker.used_adapter_params.push_back("adapter_B");

            // Create alpha tensor with persistent memory
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> scalar_shape = {};
            auto alpha_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, &alpha_value, 1, scalar_shape.data(), scalar_shape.size());
            input_tensors.push_back(std::move(alpha_tensor));
            input_names.push_back("alpha");

        } else if (merger_type == "lora_q") {
            // LoRA quantized merger inputs
            if (!base_params.weight_quantized) {
                LOGE("Missing quantized weight for LoRA quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.weight_quantized));
            input_names.push_back("weight_quantized");
            tracker.used_base_params.push_back("weight_quantized");

            if (!base_params.x_scale) {
                LOGE("Missing x_scale for LoRA quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.x_scale));
            input_names.push_back("x_scale");
            tracker.used_base_params.push_back("x_scale");

            if (!base_params.x_zero_point) {
                LOGE("Missing x_zero_point for LoRA quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.x_zero_point));
            input_names.push_back("x_zero_point");
            tracker.used_base_params.push_back("x_zero_point");

            if (adapter_params.find("adapter_A") == adapter_params.end() ||
                !adapter_params["adapter_A"].data) {
                LOGE("Missing adapter_A for LoRA quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_A"].data));
            input_names.push_back("adapter_A");
            tracker.used_adapter_params.push_back("adapter_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for LoRA quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_B"].data));
            input_names.push_back("adapter_B");
            tracker.used_adapter_params.push_back("adapter_B");

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> scalar_shape = {};
            auto alpha_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, &alpha_value, 1, scalar_shape.data(), scalar_shape.size());
            input_tensors.push_back(std::move(alpha_tensor));
            input_names.push_back("alpha");

        } else if (merger_type == "mars_q") {
            // MARS quantized merger inputs
            if (!base_params.weight_quantized) {
                LOGE("Missing quantized weight for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.weight_quantized));
            input_names.push_back("weight_quantized");
            tracker.used_base_params.push_back("weight_quantized");

            if (!base_params.x_scale) {
                LOGE("Missing x_scale for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.x_scale));
            input_names.push_back("x_scale");
            tracker.used_base_params.push_back("x_scale");

            if (!base_params.x_zero_point) {
                LOGE("Missing x_zero_point for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*base_params.x_zero_point));
            input_names.push_back("x_zero_point");
            tracker.used_base_params.push_back("x_zero_point");

            if (adapter_params.find("shared_A") == adapter_params.end() ||
                !adapter_params["shared_A"].data) {
                LOGE("Missing shared_A for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["shared_A"].data));
            input_names.push_back("shared_A");
            tracker.used_adapter_params.push_back("shared_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_B"].data));
            input_names.push_back("adapter_B");
            tracker.used_adapter_params.push_back("adapter_B");

            if (adapter_params.find("intermediate") == adapter_params.end() ||
                !adapter_params["intermediate"].data) {
                LOGE("Missing intermediate for MARS quantized merger");
                return;
            }
            input_tensors.push_back(std::move(*adapter_params["intermediate"].data));
            input_names.push_back("intermediate");
            tracker.used_adapter_params.push_back("intermediate");

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> scalar_shape = {};

            auto alpha_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, &alpha_value, 1, scalar_shape.data(), scalar_shape.size());
            input_tensors.push_back(std::move(alpha_tensor));
            input_names.push_back("alpha");

            auto adapter_index_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info, &adapter_index_value, 1, scalar_shape.data(), scalar_shape.size());
            input_tensors.push_back(std::move(adapter_index_tensor));
            input_names.push_back("adapter_index");

            auto rank_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info, &rank_value, 1, scalar_shape.data(), scalar_shape.size());
            input_tensors.push_back(std::move(rank_tensor));
            input_names.push_back("rank");
        }

        // Get output names
        std::vector<const char*> output_names;
        if (merger_type == "lora") {
            output_names.push_back("merged_weight");
        } else { // lora_q or mars_q
            output_names.push_back("merged_weight_quantized");
            output_names.push_back("merged_zero_point");
            output_names.push_back("merged_scale");
        }

        // Run inference
        std::vector<Ort::Value> output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                input_tensors.data(),
                input_tensors.size(),
                output_names.data(),
                output_names.size()
        );

        // Store outputs BEFORE freeing input memory
        MergedOutput output;
        if (merger_type == "lora") {
            output.has_weight = true;
            auto [output_tensor, buffer] = CreateUserManagedCopy(output_tensors[0]);
            output.merged_weight_buffer = buffer;
            output.merged_weight = std::move(output_tensor);
        } else { // lora_q or mars_q
            output.has_quantized = true;
            auto [output_tensor, buffer] = CreateUserManagedCopy(output_tensors[0]);
            output.merged_weight_quantized_buffer = buffer;
            output.merged_weight_quantized = std::move(output_tensor);

            auto [output_tensor1, buffer1] = CreateUserManagedCopy(output_tensors[1]);
            output.merged_zero_point_buffer = buffer1;
            output.merged_zero_point = std::move(output_tensor1);

            auto [output_tensor2, buffer2] = CreateUserManagedCopy(output_tensors[2]);
            output.merged_scale_buffer = buffer2;
            output.merged_scale = std::move(output_tensor2);
        }

        // Store the merged output
        merged_outputs_[base_layer_name] = std::move(output);

        // Now free the used parameters
        free_used_parameters(tracker);

        // Clear input and output tensors
        input_tensors.clear();
        output_tensors.clear();

        LOGI("Completed %s merger for: %s", merger_type.c_str(), base_layer_name.c_str());

    } catch (const std::exception& e) {
        LOGE("Error running merger model %s for %s: %s", merger_type.c_str(), base_layer_name.c_str(), e.what());
    }
}

// Add this method to your WeightMerger class
void WeightMerger::free_used_parameters(const ParameterTracker& tracker) {
    //LOGI("Freeing used parameters for layer: %s", tracker.base_layer_name.c_str());

    // Free base layer parameters that were used
    auto base_it = base_layer_params_.find(tracker.base_layer_name);
    if (base_it != base_layer_params_.end()) {
        auto& base_params = base_it->second;

        for (const auto& param_name : tracker.used_base_params) {
            if (param_name == "weight_quantized" && base_params.weight_quantized_buffer) {
                //LOGI("Freeing base weight_quantized buffer");
                allocator_.Free(base_params.weight_quantized_buffer);
                base_params.weight_quantized_buffer = nullptr;
                base_params.weight_quantized.reset();
            }
            else if (param_name == "x_scale" && base_params.x_scale_buffer) {
                //LOGI("Freeing base x_scale buffer");
                allocator_.Free(base_params.x_scale_buffer);
                base_params.x_scale_buffer = nullptr;
                base_params.x_scale.reset();
            }
            else if (param_name == "x_zero_point" && base_params.x_zero_point_buffer) {
                //LOGI("Freeing base x_zero_point buffer");
                allocator_.Free(base_params.x_zero_point_buffer);
                base_params.x_zero_point_buffer = nullptr;
                base_params.x_zero_point.reset();
            }
            else if (param_name == "weight" && base_params.weight_buffer) {
                //LOGI("Freeing base weight buffer");
                allocator_.Free(base_params.weight_buffer);
                base_params.weight_buffer = nullptr;
                base_params.weight.reset();
            }
        }
    }

    // Free adapter parameters that were used
    auto adapter_it = adapter_params_.find(tracker.base_layer_name);
    if (adapter_it != adapter_params_.end()) {
        auto& adapter_map = adapter_it->second;

        for (const auto& param_name : tracker.used_adapter_params) {
            auto param_it = adapter_map.find(param_name);
            if (param_it != adapter_map.end() && param_it->second.raw_buffer) {
                //LOGI("Freeing adapter %s buffer", param_name.c_str());
                allocator_.Free(param_it->second.raw_buffer);
                param_it->second.raw_buffer = nullptr;
                param_it->second.data.reset();
                // Remove the empty adapter parameter entry
                adapter_map.erase(param_it);
            }
        }

        // If no more adapter parameters for this layer, remove the entire entry
        if (adapter_map.empty()) {
            adapter_params_.erase(adapter_it);
        }
    }
}


// Helper function to convert OrtValue to vector for saving
template<typename T>
std::vector<T> WeightMerger::ortvalue_to_vector(const Ort::Value& tensor) {
    const T* data = tensor.GetTensorData<T>();
    size_t size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<T>(data, data + size);
}

// Save merged parameters using OrtValueSerializer
void WeightMerger::save_merged_parameters(const std::string& output_directory) {
    LOGI("Saving merged parameters to: %s", output_directory.c_str());

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_directory);

    int z = 0;

    for (auto& [base_layer_name, output] : merged_outputs_) {
        try {
            // Create safe filename by replacing invalid characters
            std::string safe_name = inference_name(base_layer_name);

            if (output.has_quantized) {
                // Save quantized weights in a subdirectory
                std::string quant_dir = output_directory + "/" + safe_name;
                std::filesystem::create_directories(quant_dir);

                // Save quantized weight
                if (output.merged_weight_quantized) {
                    std::string quant_file = quant_dir + "/weight_quantized.tensor";

                    if (!OrtValueSerializer::save_tensor(quant_file, *output.merged_weight_quantized,
                                                         safe_name + ".weight_quantized")) {
                        LOGE("Failed to save quantized weight for %s", base_layer_name.c_str());
                    }
                }

                // Save zero point
                if (output.merged_zero_point) {
                    std::string zero_point_file = quant_dir + "/weight_zero_point.tensor";
                    if (!OrtValueSerializer::save_tensor(zero_point_file, *output.merged_zero_point,
                                                         safe_name + ".weight_zero_point")) {
                        LOGE("Failed to save zero point for %s", base_layer_name.c_str());
                    }
                }

                // Save scale
                if (output.merged_scale) {
                    std::string scale_file = quant_dir + "/weight_scale.tensor";
                    if (!OrtValueSerializer::save_tensor(scale_file, *output.merged_scale,
                                                         base_layer_name + ".weight_scale")) {
                        LOGE("Failed to save scale for %s", base_layer_name.c_str());
                    }
                }

                // Free quantized buffers
                if (output.merged_weight_quantized_buffer) {
                    allocator_.Free(output.merged_weight_quantized_buffer);
                    output.merged_weight_quantized_buffer = nullptr;
                }
                if (output.merged_zero_point_buffer) {
                    allocator_.Free(output.merged_zero_point_buffer);
                    output.merged_zero_point_buffer = nullptr;
                }
                if (output.merged_scale_buffer) {
                    allocator_.Free(output.merged_scale_buffer);
                    output.merged_scale_buffer = nullptr;
                }
                output.merged_weight_quantized.reset();
                output.merged_zero_point.reset();
                output.merged_scale.reset();

            } else if (output.has_weight) {
                // Save regular weight
                if (output.merged_weight) {
                    std::string weight_file = output_directory + "/" + safe_name + ".tensor";
                    if (OrtValueSerializer::save_tensor(weight_file, *output.merged_weight, base_layer_name)) {
                        LOGI("Saved weight for %s to %s", base_layer_name.c_str(), weight_file.c_str());
                    } else {
                        LOGE("Failed to save weight for %s", base_layer_name.c_str());
                    }
                }

                // Free weight buffer
                if (output.merged_weight_buffer) {
                    allocator_.Free(output.merged_weight_buffer);
                    output.merged_weight_buffer = nullptr;
                }
                output.merged_weight.reset();
            }

        } catch (const std::exception& e) {
            LOGE("Error saving parameters for layer %s: %s", base_layer_name.c_str(), e.what());
        }
    }
}

// Helper function to create correct tensor names for inference initializers
std::string WeightMerger::inference_name(const std::string& layer_name) {
    std::string result = layer_name;

    // Remove "backbone." prefix if present
    if (result.find("backbone.") == 0) {
        result = result.substr(9); // Remove "backbone."
    }

    // Replace "self_attn" with "attn"
    size_t pos = result.find("self_attn");
    if (pos != std::string::npos) {
        result.replace(pos, 9, "attn"); // "self_attn" is 9 characters
    }

    // Replace "base_layer" with "MatMul.weight"
    pos = result.find("base_layer");
    if (pos != std::string::npos) {
        result.replace(pos, 10, "MatMul"); // "base_layer" is 10 characters
    }

    return result;
}



// Main method to perform weight merging
bool WeightMerger::merge_and_export_weights(Ort::CheckpointState& checkpoint_state,
                              const std::string& peft_mapping_path,
                              const std::string& merger_models_directory,
                              const std::string& output_directory) {
    LOGI("Starting weight merging process...");

    // Load PEFT mapping
    if (!load_peft_mapping(peft_mapping_path)) {
        LOGE("Failed to load PEFT mapping");
        return false;
    }

    // Load merger models
    if (!load_merger_models(merger_models_directory)) {
        LOGE("Failed to load merger models");
        return false;
    }

    // Extract parameters from checkpoint
    extract_base_layer_params(checkpoint_state);
    extract_adapter_params(checkpoint_state);

    // Process each base layer
    for (const auto& [base_layer_name, mapping] : peft_mapping_) {
        std::string adjusted_name = replace_prefix(base_layer_name, "base_model.model.model.", "backbone.model.");

        // Determine appropriate merger type
        std::string merger_type = get_merger_type(adjusted_name);
        if (merger_type.empty()) {
            LOGW("Skipping layer due to unknown merger type: %s", adjusted_name.c_str());
            continue;
        }

        // Run the appropriate merger
        run_merger_model(merger_type, adjusted_name);
    }

    // Save merged parameters
    save_merged_parameters(output_directory);

    LOGI("Weight merging process completed successfully");
    return true;
}