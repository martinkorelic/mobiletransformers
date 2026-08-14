//
// Created by martinkorelic on 20. 07. 25.
//

#include "weight_merger.h"

#include "layer_name.h"
#include <jni.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <system_error>
#include <android/log.h>
#include <nlohmann/json.hpp>
#include "logging.h"


using json = nlohmann::json;

namespace {

// ---- Compact SHA-256 (public-domain style) so device checksums match Python hashlib.sha256 hex. ----
struct Sha256 {
    uint32_t s[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t len = 0;
    uint8_t buf[64];
    size_t buf_len = 0;

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (p[i*4] << 24) | (p[i*4+1] << 16) | (p[i*4+2] << 8) | p[i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
    }

    void update(const uint8_t* data, size_t n) {
        len += n;
        while (n) {
            size_t take = std::min<size_t>(64 - buf_len, n);
            std::memcpy(buf + buf_len, data, take);
            buf_len += take; data += take; n -= take;
            if (buf_len == 64) { block(buf); buf_len = 0; }
        }
    }

    std::string hex() {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buf_len != 56) update(&zero, 1);
        uint8_t lenbe[8];
        for (int i = 0; i < 8; i++) lenbe[i] = (bits >> (56 - 8*i)) & 0xff;
        // update() bumps len; recompute block directly to append length without altering `len`.
        std::memcpy(buf + buf_len, lenbe, 8);
        block(buf);
        static const char* hexd = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; i++)
            for (int j = 3; j >= 0; j--) {
                uint8_t byte = (s[i] >> (8*j)) & 0xff;
                out.push_back(hexd[byte >> 4]);
                out.push_back(hexd[byte & 0xf]);
            }
        return out;
    }
};

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    Sha256 h;
    char chunk[1 << 16];
    while (f) {
        f.read(chunk, sizeof(chunk));
        std::streamsize got = f.gcount();
        if (got > 0) h.update(reinterpret_cast<const uint8_t*>(chunk), static_cast<size_t>(got));
    }
    return h.hex();
}

// parse_version / check_compat now live inline in handoff_io.h (shared with the load side, #23).

size_t dtype_byte_size(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
        default: return 1; // int8/uint8/bool
    }
}

// Write the tensor's raw bytes (external-data layout) atomically: temp -> fsync -> rename, then a
// sibling ".sha256". This matches the per-tensor .bin the inference graph references and the offline
// exporter's checksum, so offline and device merges are byte-identical.
bool write_raw_tensor_atomic(const std::string& final_path, const Ort::Value& tensor,
                             const std::vector<int64_t>& declared_shape = {}) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    size_t count = info.GetElementCount();
    size_t bytes = count * dtype_byte_size(info.GetElementType());
    const void* data = tensor.GetTensorData<uint8_t>();

    // #37 ROOT CAUSE: the merged weight must be written in the INFERENCE graph's layout, which is the
    // TRANSPOSE of the one the merger computes in.
    //
    // The merger works in checkpoint convention — `base_layer.weight` is a PyTorch `nn.Linear` weight,
    // `[out_features, in_features]` — and the merger graph declares its `weight` input and
    // `merged_weight` output the same way. The inference initializer that consumes the result is an
    // ONNX `MatMul` right-hand side, `[in_features, out_features]`. Writing the merger's output raw
    // therefore stored every merged weight TRANSPOSED.
    //
    // Proven on device 2026-08-14: after a merge whose delta was exactly zero (`adapter_B l2=0`,
    // scale 1.0, correct shapes), `max|written - original.T| = 1.9e-09` — the written bytes were the
    // transpose of the correct ones, to float round-trip precision. The model went from 4.65 nats to
    // 15.45 on the same text, i.e. worse than uniform.
    //
    // Why nothing caught it:
    //   * `q_proj` is square, so the shape never disagreed and no load-time check fired;
    //   * `v_proj` is `[576,192]` vs `[192,576]` — the SAME element count, so the raw external-data
    //     read succeeded too;
    //   * L2 norm and absmax are transpose-INVARIANT, so every numeric probe in the project matched;
    //   * `TrainMergeGenerateTest` asserts only that generation is non-empty, and
    //     `PostMergeNumericsTest` compared two near-uniform cross-entropies over arbitrary token ids.
    //
    // NOTE for the owner of `artifacts/handoff_map.py`: this package declares
    // `transposePolicy = "no_transpose"`, which describes what the code did and is contradicted by the
    // measurement above. The policy field is therefore NOT used as the authority here; the transpose is
    // applied and then VERIFIED against the map's declared on-disk shape, which fails closed. Whether
    // the exporter should be emitting `already_transposed_for_inference` is a #8 decision, not one to
    // make silently inside the merge.
    std::vector<float> transposed;
    auto shape = info.GetShape();
    if (shape.size() == 2 && info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const int64_t rows = shape[0], cols = shape[1];
        const float* src = tensor.GetTensorData<float>();
        transposed.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                transposed[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
            }
        }
        // Fail closed when the transposed shape contradicts what the graph will read. For a square
        // weight both orders satisfy this, which is exactly why the defect above survived — so the
        // check is a backstop, not the mechanism.
        if (declared_shape.size() == 2 &&
            (declared_shape[0] != cols || declared_shape[1] != rows)) {
            LOGE("merged tensor %s is [%lldx%lld]; transposed that is [%lldx%lld] but the handoff map "
                 "declares [%lldx%lld] on disk. Refusing to write a weight the graph cannot read.",
                 final_path.c_str(), (long long) rows, (long long) cols, (long long) cols,
                 (long long) rows, (long long) declared_shape[0], (long long) declared_shape[1]);
            return false;
        }
        data = transposed.data();
    }
    {
        // Diagnostic: a merged tensor that does not match the handoff map's declared shape is rejected
        // at load time by WeightSessionCache, far from here. Name the shape at the point of writing.
        auto shape = info.GetShape();
        std::string dims;
        for (size_t i = 0; i < shape.size(); ++i) {
            dims += (i ? "x" : "") + std::to_string(shape[i]);
        }
        LOGI("writing merged tensor %s: shape=[%s] elemtype=%d count=%zu bytes=%zu",
             final_path.c_str(), dims.c_str(), static_cast<int>(info.GetElementType()), count, bytes);
    }
    std::string tmp = final_path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { LOGE("cannot open temp for %s", final_path.c_str()); return false; }
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        out.flush();
        if (!out) { LOGE("write failed for %s", final_path.c_str()); return false; }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, final_path, ec);
    if (ec) { LOGE("atomic rename failed for %s: %s", final_path.c_str(), ec.message().c_str()); return false; }
    std::string digest = sha256_file(final_path);
    if (!digest.empty()) {
        std::ofstream sc(final_path + ".sha256", std::ios::trunc);
        sc << digest << "\n";
    }
    return true;
}

}  // namespace

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

        // #37: the merger graph's `alpha` input is a MULTIPLIER — the EFFECTIVE adapter scale, not the
        // raw hyper-parameter. MARS supplies it per layer already divided (`MarsLayer.alpha =
        // alpha / rank`), which is why MARS merges correctly. `create_lora_mapping`, by contrast,
        // emits ONLY `adapter_A`/`adapter_B` — so for LoRA these two fields were never assigned, and
        // `PeftMapping mapping;` (default-, not value-initialization) left the scalars INDETERMINATE.
        // The merger therefore computed `base + <uninitialized float> * (B @ A)`.
        //
        // Measured on an S21 FE, 2026-08-14: pristine graph 4.65 nats on English, post-merge 15.55
        // nats on the same text — above the 10.80 uniform-prediction floor, i.e. the merge left the
        // model worse than random. It hid for months because every prior gate merged a 1- or 3-step
        // adapter, where `B @ A` is ~0 and any multiplier leaves the graph ~unchanged; only a long run
        // makes the delta large enough for the wrong scale to matter. An earlier incarnation of this
        // same defect read the value through `std::map::operator[]`, which VALUE-initializes to 0.0 —
        // so the merge was a silent no-op instead of silent corruption (see the note in
        // `merge_and_export_weights`). The read was moved to a stack struct; the fact that LoRA never
        // populates alpha at all was never addressed.
        //
        // The file's top-level `alpha`/`rank` are the authority for LoRA (`training_export.py` writes
        // them beside `peft_mapping`). Fail closed rather than guessing: a wrong multiplier is silent
        // corruption, which is precisely the failure mode this path keeps producing.
        const float file_alpha =
                j.contains("alpha") ? j["alpha"].get<float>() : std::numeric_limits<float>::quiet_NaN();
        const int file_rank = j.contains("rank") ? j["rank"].get<int>() : 0;

        for (const auto& [base_layer_name, mapping_data] : j["peft_mapping"].items()) {
            // Value-initialized: `rank`/`alpha`/`adapter_index` are scalars with no default member
            // initializer, so plain `PeftMapping mapping;` leaves them indeterminate.
            PeftMapping mapping{};

            if (mapping_data.contains("adapter_B")) {
                mapping.adapter_B = mapping_data["adapter_B"];
            }
            if (mapping_data.contains("rank")) {
                mapping.rank = mapping_data["rank"];
            } else if (file_rank > 0) {
                mapping.rank = file_rank;
            }
            if (mapping_data.contains("alpha")) {
                // MARS: already the effective scale. Left exactly as it was.
                mapping.alpha = mapping_data["alpha"];
            } else {
                if (!std::isfinite(file_alpha) || file_rank <= 0) {
                    LOGE("peft_mapping entry '%s' declares no 'alpha', and the file carries no usable "
                         "top-level alpha/rank (alpha=%f rank=%d). Refusing to merge at a guessed "
                         "scale — a wrong multiplier corrupts the weights silently.",
                         base_layer_name.c_str(), file_alpha, file_rank);
                    return false;
                }
                mapping.alpha = file_alpha / static_cast<float>(file_rank);
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
            // The scale is logged because it is the value that decides whether the merge is correct,
            // and it was previously unobservable — the old line said only that a mapping loaded.
            LOGI("Loaded PEFT mapping for: %s (merge scale=%f, rank=%d)",
                 base_layer_name.c_str(), mapping.alpha, mapping.rank);
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
        std::string adjusted_name = layer_name::to_checkpoint(base_layer_name);

        BaseLayerParams base_params;

        // peft wraps the original Linear as `base_layer`, so the frozen base weight is
        // `<layer>.base_layer.weight` — the adapters sit beside it as `<layer>.lora_A.lora.weight`.
        // Looking up `<layer>.weight` (no `.base_layer`) matched nothing in the checkpoint for ANY
        // layer, so every merge aborted with "Missing base weight for LoRA merger".
        //
        // This mirrors what the Python codec already does when it seeds its lookup
        // (`inference_package.py`: `base if base.endswith(".base_layer") else base + ".base_layer"`),
        // and it is the same name the handoff map records as `trainingBaseLayerName`.
        const std::string base_module = layer_name::with_base_layer(adjusted_name);

        // Look for different weight parameter types
        std::string weight_quantized_name = base_module + ".weight_quantized";
        std::string weight_scale_name = base_module + ".weight_scale";
        std::string weight_zero_point_name = base_module + ".weight_zero_point";
        std::string weight_name = base_module + ".weight";

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
        std::string adjusted_base_name = layer_name::to_checkpoint(base_layer_name);

        adapter_params_[adjusted_base_name] = std::unordered_map<std::string, AdapterParams>();

        // Extract adapter_B
        if (!mapping.adapter_B.empty()) {
            std::string adapter_name = layer_name::to_checkpoint(mapping.adapter_B);
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
            std::string adapter_name = layer_name::to_checkpoint(mapping.shared_A);
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
            std::string adapter_name = layer_name::to_checkpoint(mapping.intermediate);
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
            std::string adapter_name = layer_name::to_checkpoint(mapping.adapter_A);
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

// Load + version-gate weight_handoff_map.json — the single source of tensor identity (#8/#9/#23).
// Delegates to the ONE shared reader in handoff_io.h (also used by the load side, session_cache.h).
bool WeightMerger::load_handoff_map(const std::string& json_path) {
    bool ok = load_handoff_entries(json_path, /*readerVersion=*/"1.0", handoff_map_, &merger_models_);
    LOGI("Loaded handoff map: %zu entries, %zu merger model(s)", handoff_map_.size(), merger_models_.size());
    return ok;
}

const HandoffEntry* WeightMerger::find_handoff_entry(const std::string& base_layer_name) const {
    // The merge loop works in "adjusted" space (`backbone.model.<layer>`, no `.base_layer`), while the
    // handoff map keys entries by the raw training name (`base_model.model.model.<layer>.base_layer`).
    // Only the `.base_layer` half of that difference was handled, so every merged layer failed to find
    // its entry and `save_merged_parameters` wrote nothing while reporting per-layer errors — the merge
    // ran 60/60 and still left all 60 `.bin` files untouched.
    //
    // Try both prefix forms and both suffix forms rather than assuming one direction.
    for (const auto& key : layer_name::candidate_handoff_keys(base_layer_name)) {
        auto it = handoff_map_.find(key);
        if (it != handoff_map_.end()) return &it->second;
    }
    return nullptr;
}

bool WeightMerger::load_merger_models(const std::string& models_directory) {
    LOGI("Loading merger models from: %s", models_directory.c_str());
    if (merger_models_.empty()) {
        LOGE("No merger models in the handoff map; load_handoff_map must run first");
        return false;
    }
    try {
        // Load one session per resolved MergerVariant, filename from the map (no hard-coded names).
        // #6: the map's tag is parsed into the typed enum here, so an unrecognized variant fails
        // closed at load time instead of silently never matching at dispatch time.
        for (const auto& [tag, filename] : merger_models_) {
            std::optional<MergerVariant> variant = merger_variant_from_wire(tag);
            if (!variant) {
                LOGE("handoff map declares unknown merger variant '%s'", tag.c_str());
                return false;
            }
            std::string path = models_directory + "/" + filename;
            merger_sessions_[*variant] = std::make_unique<Ort::Session>(
                    Ort::Env(), path.c_str(), Ort::SessionOptions{});
            LOGI("Loaded merger session '%s' <- %s", tag.c_str(), filename.c_str());
        }
        return true;
    } catch (const std::exception& e) {
        LOGE("Error loading merger models: %s", e.what());
        return false;
    }
}

// Resolve this layer's merger variant from its adapter shape + quantization (#6: a typed
// MergerVariant, not a manufactured string that had to coincidentally match the handoff map's
// mergerModels keys). nullopt = no merger applies to this layer, which the caller treats as fatal.
std::optional<MergerVariant> WeightMerger::resolve_merger_variant(const std::string& base_layer_name) {
    auto adapter_it = adapter_params_.find(base_layer_name);
    if (adapter_it == adapter_params_.end()) {
        return std::nullopt;
    }

    auto& adapters = adapter_it->second;
    const bool has_shared_A = adapters.find("shared_A") != adapters.end();   // MARS shares A
    const bool has_adapter_A = adapters.find("adapter_A") != adapters.end(); // LoRA has its own A
    const bool has_quantized = base_layer_params_[base_layer_name].has_quantized;

    if (has_shared_A && has_quantized) return MergerVariant::MARS_Q;
    if (has_adapter_A && has_quantized) return MergerVariant::LORA_Q;
    if (has_adapter_A && !has_quantized) return MergerVariant::LORA;

    LOGW("Unable to determine merger variant for: %s", base_layer_name.c_str());
    return std::nullopt;
}

bool WeightMerger::run_merger_model(MergerVariant variant, const std::string& base_layer_name,
                                   const PeftMapping& mapping) {
    LOGI("Running %s merger for: %s", to_wire(variant), base_layer_name.c_str());

    if (merger_sessions_.find(variant) == merger_sessions_.end()) {
        LOGE("Merger model not found for variant: %s", to_wire(variant));
        return false;
    }

    try {
        auto& session = merger_sessions_[variant];
        auto& base_params = base_layer_params_[base_layer_name];
        auto& adapter_params = adapter_params_[base_layer_name];

        // Create parameter tracker
        ParameterTracker tracker(base_layer_name);

        // Prepare input tensors based on merger type
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        // Storage for scalar values (must persist during inference)
        // Take these from the caller's mapping. They used to be read as
        // `peft_mapping_[base_layer_name]`, but `base_layer_name` here is the ADJUSTED name
        // (`backbone.model.…`) while `peft_mapping_` is keyed by the RAW name
        // (`base_model.model.model.…`). `operator[]` therefore inserted a fresh default entry on every
        // single layer, which caused BOTH observed failures:
        //   1. it mutated `peft_mapping_` while `merge_and_export_weights` was range-for iterating it,
        //      rehashing mid-traversal — the loop ran 12 times for 60 layers and revisited one;
        //   2. alpha/rank/adapter_index came back default-constructed 0, so the merger computed
        //      `weight + 0 * (B @ A)` == weight and every "successful" merge wrote byte-identical data
        //      (the `merge wrote no new weights: all 60 unchanged` assertion).
        float alpha_value = mapping.alpha;
        int64_t adapter_index_value = mapping.adapter_index;
        int64_t rank_value = mapping.rank;

        if (variant == MergerVariant::LORA) {
            // LoRA merger inputs: base_weight, adapter_A, adapter_B, alpha
            if (!base_params.weight) {
                LOGE("Missing base weight for LoRA merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.weight));
            input_names.push_back("weight");
            tracker.used_base_params.push_back("weight");

            if (adapter_params.find("adapter_A") == adapter_params.end() ||
                !adapter_params["adapter_A"].data) {
                LOGE("Missing adapter_A for LoRA merger");
                return false;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_A"].data));
            input_names.push_back("adapter_A");
            tracker.used_adapter_params.push_back("adapter_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for LoRA merger");
                return false;
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

            // #37 instrumentation, first merged layer only: the shapes and magnitudes actually fed to
            // `base + scale * (B @ A)`. A merge that is the exact identity at B == 0 but ruinous at
            // B != 0 is decided entirely by these tensors, and nothing logged them.
            static bool logged_lora_inputs = false;
            if (!logged_lora_inputs) {
                logged_lora_inputs = true;
                for (size_t i = 0; i < input_tensors.size(); ++i) {
                    if (!input_tensors[i].IsTensor()) continue;
                    auto info = input_tensors[i].GetTensorTypeAndShapeInfo();
                    auto shape = info.GetShape();
                    std::string dims;
                    for (size_t d = 0; d < shape.size(); ++d) {
                        dims += std::to_string(shape[d]);
                        if (d + 1 < shape.size()) dims += "x";
                    }
                    double norm = 0.0, amax = 0.0;
                    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                        const float* p = input_tensors[i].GetTensorData<float>();
                        const size_t n = info.GetElementCount();
                        for (size_t k = 0; k < n; ++k) {
                            const double v = static_cast<double>(p[k]);
                            norm += v * v;
                            amax = std::max(amax, std::abs(v));
                        }
                        norm = std::sqrt(norm);
                    }
                    LOGI("MERGE-INPUT %s dims=[%s] count=%zu l2=%f absmax=%f",
                         input_names[i], dims.c_str(), (size_t) info.GetElementCount(), norm, amax);
                }
            }

        } else if (variant == MergerVariant::LORA_Q) {
            // LoRA quantized merger inputs
            if (!base_params.weight_quantized) {
                LOGE("Missing quantized weight for LoRA quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.weight_quantized));
            input_names.push_back("weight_quantized");
            tracker.used_base_params.push_back("weight_quantized");

            if (!base_params.x_scale) {
                LOGE("Missing x_scale for LoRA quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.x_scale));
            input_names.push_back("x_scale");
            tracker.used_base_params.push_back("x_scale");

            if (!base_params.x_zero_point) {
                LOGE("Missing x_zero_point for LoRA quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.x_zero_point));
            input_names.push_back("x_zero_point");
            tracker.used_base_params.push_back("x_zero_point");

            if (adapter_params.find("adapter_A") == adapter_params.end() ||
                !adapter_params["adapter_A"].data) {
                LOGE("Missing adapter_A for LoRA quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_A"].data));
            input_names.push_back("adapter_A");
            tracker.used_adapter_params.push_back("adapter_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for LoRA quantized merger");
                return false;
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

        } else if (variant == MergerVariant::MARS_Q) {
            // MARS quantized merger inputs
            if (!base_params.weight_quantized) {
                LOGE("Missing quantized weight for MARS quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.weight_quantized));
            input_names.push_back("weight_quantized");
            tracker.used_base_params.push_back("weight_quantized");

            if (!base_params.x_scale) {
                LOGE("Missing x_scale for MARS quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.x_scale));
            input_names.push_back("x_scale");
            tracker.used_base_params.push_back("x_scale");

            if (!base_params.x_zero_point) {
                LOGE("Missing x_zero_point for MARS quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*base_params.x_zero_point));
            input_names.push_back("x_zero_point");
            tracker.used_base_params.push_back("x_zero_point");

            if (adapter_params.find("shared_A") == adapter_params.end() ||
                !adapter_params["shared_A"].data) {
                LOGE("Missing shared_A for MARS quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*adapter_params["shared_A"].data));
            input_names.push_back("shared_A");
            tracker.used_adapter_params.push_back("shared_A");

            if (adapter_params.find("adapter_B") == adapter_params.end() ||
                !adapter_params["adapter_B"].data) {
                LOGE("Missing adapter_B for MARS quantized merger");
                return false;
            }
            input_tensors.push_back(std::move(*adapter_params["adapter_B"].data));
            input_names.push_back("adapter_B");
            tracker.used_adapter_params.push_back("adapter_B");

            if (adapter_params.find("intermediate") == adapter_params.end() ||
                !adapter_params["intermediate"].data) {
                LOGE("Missing intermediate for MARS quantized merger");
                return false;
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
        if (variant == MergerVariant::LORA) {
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
        if (variant == MergerVariant::LORA) {
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

        // Deliberately NOT freeing here: `free_used_parameters` erases from `adapter_params_` and
        // releases allocator buffers, and doing that mid-merge mutates state the surrounding loop
        // still depends on. (This was not what caused the revisited-layer bug — that was the
        // `peft_mapping_[...]` insert below — but mutating containers mid-traversal is the same
        // hazard and is not worth keeping.) Deferred to release_merge_inputs(); costs ~50 MB of
        // retained base weights on a 135M model, and free-as-you-go can return as a measured
        // optimization once it is demonstrably safe.
        merge_trackers_.push_back(tracker);

        // Clear input and output tensors
        input_tensors.clear();
        output_tensors.clear();

        LOGI("Completed %s merger for: %s", to_wire(variant), base_layer_name.c_str());

    } catch (const std::exception& e) {
        LOGE("Error running merger model %s for %s: %s", to_wire(variant), base_layer_name.c_str(), e.what());
        return false;
    }
    return true;
}

// Add this method to your WeightMerger class
// Release every buffer the merge borrowed, after the loop has finished. Split out from the merge so
// the maps are never mutated while `merge_and_export_weights` is walking them.
void WeightMerger::release_merge_inputs() {
    for (const auto& tracker : merge_trackers_) {
        free_used_parameters(tracker);
    }
    merge_trackers_.clear();
}

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
            LOGI("free: erasing adapter entry for %s", tracker.base_layer_name.c_str());
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

// Write every merged tensor to the exact per-tensor .bin the inference graph references.
// #9 fail-closed: returns false if ANY role of ANY layer could not be written. A partially-merged
// inference/ directory is worse than no merge at all — some tensors trained, some frozen — so the
// caller must surface this rather than report success.
bool WeightMerger::save_merged_parameters(const std::string& output_directory) {
    LOGI("Saving merged parameters to: %s", output_directory.c_str());
    std::filesystem::create_directories(output_directory);

    bool all_ok = true;
    for (auto& [base_layer_name, output] : merged_outputs_) {
        try {
            // Tensor identity comes from the handoff map — NO string-rewrite on device (the old
            // inference_name path is gone). Fail closed if a merged layer has no map entry.
            const HandoffEntry* entry = find_handoff_entry(base_layer_name);
            if (!entry) {
                LOGE("no handoff entry for merged layer %s; refusing to guess a filename",
                     base_layer_name.c_str());
                all_ok = false;
                continue;
            }

            // Local copy so the lambda captures a plain variable (capturing a structured binding is a
            // C++20 extension; this file is built as C++17).
            const std::string layer_name = base_layer_name;

            // Write the merged tensor's raw bytes to the exact per-tensor .bin the inference graph
            // references (map's externalDataLocation[role]), atomically + with a .sha256 sidecar.
            auto save_role = [&](const std::string& role, const std::unique_ptr<Ort::Value>& val) {
                if (!val) return;
                auto loc = entry->externalDataLocation.find(role);
                if (loc == entry->externalDataLocation.end()) {
                    LOGE("handoff entry %s has no externalDataLocation[%s]",
                         layer_name.c_str(), role.c_str());
                    all_ok = false;
                    return;
                }
                std::string path = output_directory + "/" + loc->second;
                // Pass the map's declared on-disk shape so the write can verify the layout it produces
                // is the one the inference graph will read (#37).
                if (write_raw_tensor_atomic(path, *val, entry->shape_for(role))) {
                    LOGI("merged %s [%s] -> %s", layer_name.c_str(), role.c_str(), loc->second.c_str());
                } else {
                    LOGE("failed writing merged %s [%s]", layer_name.c_str(), role.c_str());
                    all_ok = false;
                }
            };

            if (output.has_quantized) {
                save_role("weight_quantized", output.merged_weight_quantized);
                save_role("zero_point", output.merged_zero_point);
                save_role("scale", output.merged_scale);

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
                save_role("weight", output.merged_weight);
                if (output.merged_weight_buffer) {
                    allocator_.Free(output.merged_weight_buffer);
                    output.merged_weight_buffer = nullptr;
                }
                output.merged_weight.reset();
            }

        } catch (const std::exception& e) {
            LOGE("Error saving parameters for layer %s: %s", base_layer_name.c_str(), e.what());
            all_ok = false;
        }
    }
    return all_ok;
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

    // Load the handoff map FIRST — it provides the resolved merger filenames + tensor identity that
    // both the session loading and the save side now key off (single source of truth, #8/#9). The map
    // lives in the merger models directory (the inference package dir).
    handoff_dir_ = merger_models_directory;
    if (!load_handoff_map(merger_models_directory + "/weight_handoff_map.json")) {
        LOGE("Failed to load weight_handoff_map.json (required by #9)");
        return false;
    }

    // Load merger models (filenames resolved from the handoff map's mergerModels).
    if (!load_merger_models(merger_models_directory)) {
        LOGE("Failed to load merger models");
        return false;
    }

    // Extract parameters from checkpoint
    extract_base_layer_params(checkpoint_state);
    extract_adapter_params(checkpoint_state);

    // Process each base layer
    for (const auto& [base_layer_name, mapping] : peft_mapping_) {
        std::string adjusted_name = layer_name::to_checkpoint(base_layer_name);

        // Diagnostic: adapter_params_ should shrink by exactly one entry per merged layer. Anything
        // else means entries are disappearing that this loop did not consume.
        LOGI("merge loop: layer=%s adapters_remaining=%zu base_remaining=%zu",
             adjusted_name.c_str(), adapter_params_.size(), base_layer_params_.size());

        // Determine appropriate merger type
        std::optional<MergerVariant> variant = resolve_merger_variant(adjusted_name);
        if (!variant) {
            // #9 fail-closed: skipping leaves this layer at its frozen base weights while its peers
            // are merged, i.e. a silently half-trained model. Abort instead.
            LOGE("unresolved merger variant for layer %s; aborting the merge", adjusted_name.c_str());
            return false;
        }

        // Run the appropriate merger. A miss here (no merger graph shipped for this variant) used to
        // LOGE and continue, so `merge()` reported success having merged NOTHING — the package shipped a
        // `lora_q` graph while the device resolved `lora`, and all 60 tensors stayed at base weights.
        // Same fail-closed rule as the unresolved-variant branch above.
        if (!run_merger_model(*variant, adjusted_name, mapping)) {
            LOGE("merger failed for layer %s; aborting the merge", adjusted_name.c_str());
            return false;
        }
    }

    // Save merged parameters. A partial write must NOT report success (#9).
    const bool saved = save_merged_parameters(output_directory);
    release_merge_inputs();  // deferred cleanup: safe now that nothing is iterating
    if (!saved) {
        LOGE("Weight merging failed: one or more merged tensors could not be written");
        return false;
    }

    LOGI("Weight merging process completed successfully");
    return true;
}