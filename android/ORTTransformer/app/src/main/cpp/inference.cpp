//
// Created by martinkorelic on 01/09/2024.
//

#include "inference.h"
#include <cassert>

#include "onnxruntime_cxx_api.h"

#include <android/log.h>

#define LOG_TAG "ORTTransformer"

namespace inference {

    std::string genAiInferenceStep(GenAISessionCache* session_cache) {

        session_cache->generator->ComputeLogits();
        session_cache->generator->GenerateNextToken();

        const auto num_tokens = session_cache->generator->GetSequenceCount(0);
        const auto new_token = session_cache->generator->GetSequenceData(0)[num_tokens - 1];
        return session_cache->tokenizer_stream->Decode(new_token);
    }

    std::vector<std::vector<float>> getLastTokenLogits(const float* logits, int batch_size, int sequence_length, int vocab_size, float temperature) {

        std::vector<std::vector<float>> output;

        // Resize output vector to match batch size
        output.resize(batch_size);

        // Iterate over each batch
        for (int i = 0; i < batch_size; ++i) {
            // Calculate the offset to the last token logits in the flattened array
            int offset = i * (sequence_length * vocab_size) + (sequence_length - 1) * vocab_size;

            output[i].resize(vocab_size);

            // Access and store the last token logits, applying temperature scaling
            for (int j = 0; j < vocab_size; ++j) {
                output[i][j] = logits[offset + j] / temperature;  // Apply temperature scaling
            }
        }

        return output;
    }


    int argmax(float* logits, int sequence_length, int vocab_size) {

        assert(logits != nullptr && "Logits pointer should not be null");
        assert(sequence_length > 0 && "Sequence length should be positive");
        assert(vocab_size > 0 && "Vocabulary size should be positive");

        int last_token_start_index = (sequence_length - 1) * vocab_size;
        float* last_token_logits = &logits[last_token_start_index];

        // Find the max logit for the most probable token ID in the last token's logits
        float* max_logit_ptr = std::max_element(last_token_logits, last_token_logits + vocab_size);

        // Calculate and return the index of the maximum logit in the last token's logits
        return std::distance(last_token_logits, max_logit_ptr);
    }

    std::vector<float> Softmax(std::vector<float> logits, size_t num_logits) {
        std::vector<float> probabilities(num_logits, 0);
        float sum = 0;
        for (size_t i = 0; i < num_logits; ++i) {
            probabilities[i] = exp(logits[i]);
            sum += probabilities[i];
        }

        if (sum != 0.0f) {
            for (size_t i = 0; i < num_logits; ++i) {
                probabilities[i] /= sum;
            }
        }

        return probabilities;
    }

    size_t greedySampling(float* logits,
                          int64_t batch_size,
                          int64_t sequence_length,
                          size_t vocab_size) {

        auto lastTokenLogits = getLastTokenLogits(logits, batch_size, sequence_length, vocab_size, 1);

        // Run softmax and get the probabilities of each class
        // TODO: Only works for one batch for now
        std::vector<float> probabilities = Softmax(lastTokenLogits[0], vocab_size);
        size_t best_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
        return best_index;
    }

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

                __android_log_print(ANDROID_LOG_WARN, "TensorMemory", "Non-tensor value in vector, skipping.");
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
                            __android_log_print(ANDROID_LOG_WARN, "TensorMemory", "Unsupported tensor data type.");
                            return static_cast<size_t>(0);
                    }
                }();

                if (element_size > 0) {
                    total_bytes += element_count * element_size;
                }
            } catch (const std::exception& ex) {
                __android_log_print(ANDROID_LOG_ERROR, "TensorMemory", "Error processing tensor: %s", ex.what());
            }
        }

        double total_mb = static_cast<double>(total_bytes) / BYTES_IN_MB;
        __android_log_print(ANDROID_LOG_INFO, "TensorMemory", "Total tensor memory usage: %.2f MB", total_mb);
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

        auto session_run_opts = Ort::RunOptions();
        //session_run_opts.AddConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0");
        //session_run_opts.SetRunLogSeverityLevel(0);
        //session_run_opts.SetRunLogVerbosityLevel(0);
        
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



} // namespace inference
