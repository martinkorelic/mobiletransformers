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
        const std::vector<int64_t> position_ids_shape({batch_size, past_sequence_length});

        std::vector<Ort::Value> input_values;

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
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, position_ids,
                                                           batch_size * past_sequence_length * sizeof(int64_t),
                                                           position_ids_shape.data(), position_ids_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        std::vector<Ort::Value> output_values;

        // Load and move past KV caches to the input
        //int i = 0;
        for (auto& kv : session_cache->past_key_values) {

//            float* data = kv->GetTensorMutableData<float>(); // Assuming they are float tensors
//            size_t count = kv->GetTensorTypeAndShapeInfo().GetElementCount();
//            __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "Past key value i: %d", i);
//
//            for (size_t j = 0; j < 5; ++j) {
//                __android_log_print(ANDROID_LOG_DEBUG, "InferenceSessionCache", "%f", data[j]);
//            }
//            i++;
            input_values.emplace_back(std::move(*kv));
            output_values.emplace_back(nullptr);
        }

        output_values.emplace_back(nullptr);

        // Get the logits
        session_cache->inference_session->Run(Ort::RunOptions(), input_names.data(), input_values.data(),
                                              input_count, output_names.data(), output_values.data(), output_count);

        float *output = output_values.front().GetTensorMutableData<float>();

        if (!output_values.empty()) {
            output_values.erase(output_values.begin());
        }

        // Update KV caches
        session_cache->updatePastKeyValues(output_values);

        return output;
    }

} // namespace inference
