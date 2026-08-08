//
// Created by martinkorelic on 01/09/2024.
//

#include "inference.h"
#include <cassert>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include <android/log.h>

namespace inference {

    float* forward(InferenceSessionCache* session_cache,
                                          int64_t* input_ids,
                                          int64_t* attention_mask,
                                          int64_t* position_ids,
                                          int64_t batch_size,
                                          int64_t sequence_length,
                                          size_t vocab_size) {
        std::vector<const char *> input_names = {"input_ids", "attention_mask", "position_ids"};
        size_t input_count = 3;

        std::vector<const char *> output_names = {"logits"};
        size_t output_count = 1;

        const std::vector<int64_t> input_ids_shape({batch_size, sequence_length});
        const std::vector<int64_t> attention_mask_shape({batch_size, sequence_length});
        const std::vector<int64_t> position_ids_shape({batch_size, sequence_length});

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_values;

        // Input ids
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, input_ids,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           input_ids_shape.data(), input_ids_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Attention mask
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, attention_mask,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           attention_mask_shape.data(), attention_mask_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Position ids
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, position_ids,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           position_ids_shape.data(), position_ids_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        std::vector<Ort::Value> output_values;
        output_values.emplace_back(nullptr);

        // Get the logits
        session_cache->inference_session->Run(Ort::RunOptions(), input_names.data(), input_values.data(),
                                              input_count, output_names.data(), output_values.data(), output_count);

        float *output = output_values.front().GetTensorMutableData<float>();

        return output;
    }

    void logTensorMemoryUsage(const std::vector<Ort::Value>& tensors) {
        constexpr size_t BYTES_IN_MB = 1024 * 1024;
        size_t total_bytes = 0;

        for (const auto& tensor : tensors) {
            if (!tensor.IsTensor()) {

                LOGI("Non-tensor value in vector, skipping.");
                continue;
            }

            try {
                auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
                size_t element_count = tensor_info.GetElementCount();

                size_t element_size = [&tensor_info]() {
                    switch (tensor_info.GetElementType()) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return sizeof(float);
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return sizeof(int32_t);
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return sizeof(int64_t);
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return sizeof(double);
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     return sizeof(bool);
                        default:
                            LOGI("Unsupported tensor data type.");
                            return static_cast<size_t>(0);
                    }
                }();

                if (element_size > 0) {
                    total_bytes += element_count * element_size;
                }
            } catch (const std::exception& ex) {
                LOGI("Error processing tensor: %s", ex.what());
            }
        }

        double total_mb = static_cast<double>(total_bytes) / BYTES_IN_MB;
        LOGI("Total tensor memory usage: %.2f MB", total_mb);
    }

    // Forward pass with KV caching
    float* generateWithKVCache(InferenceSessionCache* session_cache,
                               int64_t* input_ids,
                               int64_t* attention_mask,
                               int64_t* position_ids,
                               int64_t batch_size,
                               int64_t sequence_length,
                               int64_t past_sequence_length) {

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<const char *> input_names = session_cache->input_names;
        size_t input_count = input_names.size();

        std::vector<const char *> output_names = session_cache->output_names;
        size_t output_count = output_names.size();

        const std::vector<int64_t> input_ids_shape({batch_size, past_sequence_length});
        const std::vector<int64_t> attention_mask_shape({batch_size, sequence_length});

        std::vector<int64_t> position_ids_shape;
        if (session_cache->has_position_ids) {
            position_ids_shape = std::vector<int64_t>{batch_size, past_sequence_length};
        }

        std::vector<Ort::Value> input_values;
        input_values.reserve(input_names.size());
        // Input ids
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, input_ids,
                                                           batch_size * past_sequence_length * sizeof(int64_t),
                                                           input_ids_shape.data(), input_ids_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Attention mask
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, attention_mask,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           attention_mask_shape.data(), attention_mask_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Position ids
        if (session_cache->has_position_ids) {
            input_values.emplace_back(Ort::Value::CreateTensor(memory_info, position_ids,
                                                               batch_size * past_sequence_length * sizeof(int64_t),
                                                               position_ids_shape.data(), position_ids_shape.size(),
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        }
        
        std::vector<Ort::Value> output_values;
        output_values.reserve(output_names.size());

        // Load and move past KV caches to the input
        for (auto& kv : session_cache->past_key_values) {
            input_values.emplace_back(std::move(*kv));
            output_values.emplace_back(Ort::Value(nullptr));
        }

        logTensorMemoryUsage(input_values);
        output_values.emplace_back(nullptr);

        // Fail closed on an input-count mismatch. `Run` is handed `input_count` (the graph's input
        // count) and reads that many values out of `input_values`; if the KV cache was never
        // initialized — which happens when the graph carries no `num_layers` metadata — this reads
        // past the end of the vector and segfaults inside ORT with an unrelated-looking stack. An
        // exception naming the two counts points straight at the exporter.
        if (input_values.size() != input_count) {
            throw std::runtime_error(
                    "inference input count mismatch: graph expects " + std::to_string(input_count) +
                    " inputs but " + std::to_string(input_values.size()) + " were bound (" +
                    std::to_string(session_cache->past_key_values.size()) +
                    " KV tensors). The graph is missing the num_layers/num_kv_heads/head_dim metadata "
                    "the KV cache is sized from — re-export the package.");
        }
        if (output_values.size() != output_count) {
            throw std::runtime_error(
                    "inference output count mismatch: graph declares " + std::to_string(output_count) +
                    " outputs but " + std::to_string(output_values.size()) + " slots were prepared.");
        }

        auto session_run_opts = Ort::RunOptions();

        // Get the logits
        session_cache->inference_session->Run(session_run_opts, input_names.data(), input_values.data(),
                                              input_count, output_names.data(), output_values.data(), output_count);

        std::unique_ptr<Ort::Value> output = std::make_unique<Ort::Value>(std::move(output_values.front()));

        if (!output_values.empty()) {
            output_values.erase(output_values.begin());
        }

        std::vector<Ort::Value> kv_outputs(
                std::make_move_iterator(output_values.begin()),
                std::make_move_iterator(output_values.end())
        );

        // Update KV caches
        session_cache->updatePastKeyValues(kv_outputs);

        // Explicitly release input values to free memory
        input_values.clear();
        output_values.clear();

        return output->GetTensorMutableData<float>();
    }

    float* generateEmbedding(EmbeddingSessionCache* session_cache,
                               int64_t* input_ids,
                               int64_t* attention_mask,
                               int64_t* token_type_ids,
                               int64_t batch_size,
                               int64_t sequence_length) {

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<const char *> input_names = session_cache->input_names;
        size_t input_count = input_names.size();

        std::vector<const char *> output_names = session_cache->output_names;
        size_t output_count = output_names.size();

        const std::vector<int64_t> input_ids_shape({batch_size, sequence_length});
        const std::vector<int64_t> attention_mask_shape({batch_size, sequence_length});

        std::vector<int64_t> token_type_ids_shape;
        if (session_cache->has_token_type_ids) {
            token_type_ids_shape = std::vector<int64_t>{batch_size, sequence_length};
        }

        std::vector<Ort::Value> input_values;
        input_values.reserve(input_names.size());
        // Input ids
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, input_ids,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           input_ids_shape.data(), input_ids_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Attention mask
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, attention_mask,
                                                           batch_size * sequence_length * sizeof(int64_t),
                                                           attention_mask_shape.data(), attention_mask_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Token type ids
        if (session_cache->has_token_type_ids) {
            input_values.emplace_back(Ort::Value::CreateTensor(memory_info, token_type_ids,
                                                               batch_size * sequence_length * sizeof(int64_t),
                                                               token_type_ids_shape.data(), token_type_ids_shape.size(),
                                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        }

        std::vector<Ort::Value> output_values;
        output_values.reserve(output_names.size());
        output_values.emplace_back(nullptr);

        auto session_run_opts = Ort::RunOptions();

        // Embedding vector
        session_cache->embedding_session->Run(session_run_opts, input_names.data(), input_values.data(),
                                              input_count, output_names.data(), output_values.data(), output_count);

        std::unique_ptr<Ort::Value> output = std::make_unique<Ort::Value>(std::move(output_values.front()));

        // Explicitly release input values to free memory
        input_values.clear();
        output_values.clear();

        return output->GetTensorMutableData<float>();
    }

} // namespace inference
