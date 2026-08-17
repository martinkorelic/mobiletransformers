//
// Created by martinkorelic on 31/08/2024
//

#include "train.h"
#include "training_inputs.h"
#include <android/log.h>
#include <string>
#include <vector>

#define LOG_TAG "MobileTransformers"

namespace training {

    namespace {

        size_t element_count_of(const std::vector<int64_t>& shape) {
            size_t count = 1;
            for (const int64_t dim : shape) {
                count *= static_cast<size_t>(dim);
            }
            return count;
        }

        Ort::Value make_int64_tensor(const Ort::MemoryInfo& memory_info, int64_t* data,
                                     const std::vector<int64_t>& shape) {
            return Ort::Value::CreateTensor(memory_info, data, element_count_of(shape) * sizeof(int64_t),
                                            shape.data(), shape.size(),
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        }

    } // namespace

    float train_step(TrainingSessionCache* session_cache,
                     int64_t* input_ids,
                     int64_t* attention_mask,
                     int64_t* labels,
                     int64_t batch_size,
                     int64_t sequence_length,
                     int64_t labels_count) {

        // The GRAPH decides which inputs exist and in what order TrainStep wants them. This is what
        // lets one binder serve a decoder (position_ids, per-token labels) and an encoder classifier
        // (token_type_ids, per-sequence labels) with no task switch in C++. The decision itself is
        // ORT-free and host-tested — see training_inputs.h.
        const std::vector<std::string> input_names =
                session_cache->training_session.InputNames(/*training=*/true);
        const std::vector<BoundInput> plan =
                plan_training_inputs(input_names, batch_size, sequence_length, labels_count);

        // Backing storage for synthesized inputs, declared out here so the buffers outlive the
        // Ort::Values — which do not own their data.
        std::vector<int64_t> position_ids;
        std::vector<int64_t> token_type_ids;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> user_inputs;
        user_inputs.reserve(plan.size());

        for (const BoundInput& bound : plan) {
            switch (bound.source) {
                case InputSource::CallerInputIds:
                    user_inputs.emplace_back(make_int64_tensor(memory_info, input_ids, bound.shape));
                    break;
                case InputSource::CallerAttentionMask:
                    user_inputs.emplace_back(make_int64_tensor(memory_info, attention_mask, bound.shape));
                    break;
                case InputSource::CallerLabels:
                    user_inputs.emplace_back(make_int64_tensor(memory_info, labels, bound.shape));
                    break;
                case InputSource::SyntheticPositions: {
                    position_ids.resize(element_count_of(bound.shape));
                    for (int64_t i = 0; i < batch_size; ++i) {
                        for (int64_t j = 0; j < sequence_length; ++j) {
                            position_ids[i * sequence_length + j] = j;
                        }
                    }
                    user_inputs.emplace_back(make_int64_tensor(memory_info, position_ids.data(), bound.shape));
                    break;
                }
                case InputSource::SyntheticTokenTypes:
                    token_type_ids.assign(element_count_of(bound.shape), 0);
                    user_inputs.emplace_back(make_int64_tensor(memory_info, token_type_ids.data(), bound.shape));
                    break;
            }
        }

        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "train_step: bound %zu inputs by name [%s]",
                            user_inputs.size(), join_names(input_names).c_str());

        // Run the train step and execute the forward + loss + backward.
        float loss = *(session_cache->training_session.TrainStep(user_inputs)
                               .front()
                               .GetTensorMutableData<float>());

        user_inputs.clear();

        return loss;
    }

    void optimizer_step(TrainingSessionCache* session_cache) {
        // Update the model parameters by taking a step in the direction of the gradients
        session_cache->training_session.OptimizerStep();

        // Reset the gradients now that the parameters have been updated.
        // New set of gradients can then be computed for the next round of inputs.
        session_cache->training_session.LazyResetGrad();
    }

} // namespace training
