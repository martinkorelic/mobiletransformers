//
// Originally created by bmeswani on 2/20/2023.
// Edited by martinkorelic on 31/08/2024
//

#include "train.h"
#include <android/log.h>

#define LOG_TAG "ORTTransformer"

namespace training {

    float train_step(TrainingSessionCache* session_cache,
                     int64_t* input_ids,
                     int64_t* attention_mask,
                     int64_t* position_ids,
                     int64_t* labels,
                     int64_t batch_size,
                     int64_t sequence_length) {
        const std::vector<int64_t> input_ids_shape({batch_size, sequence_length});
        const std::vector<int64_t> attention_mask_shape({batch_size, sequence_length});
        const std::vector<int64_t> position_ids_shape({batch_size, sequence_length});
        const std::vector<int64_t> labels_shape({batch_size, sequence_length});

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> user_inputs; // {input_ids, attention_mask, position_ids, labels}
        // Input_ids batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, input_ids,
                                                          batch_size * sequence_length * sizeof(int64_t),
                                                          input_ids_shape.data(), input_ids_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Attention mask batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, attention_mask,
                                                          batch_size * sequence_length * sizeof(int64_t),
                                                          attention_mask_shape.data(), attention_mask_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Position ids batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, position_ids,
                                                          batch_size * sequence_length * sizeof(int64_t),
                                                          position_ids_shape.data(), position_ids_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Labels batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels,
                                                          batch_size * sequence_length * sizeof(int64_t),
                                                          labels_shape.data(), labels_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Run the train step and execute the forward + loss + backward.
        float loss = *(session_cache->training_session.TrainStep(user_inputs).front().GetTensorMutableData<float>());

        // Update the model parameters by taking a step in the direction of the gradients computed above.
        session_cache->training_session.OptimizerStep();

        // Reset the gradients now that the parameters have been updated.
        // New set of gradients can then be computed for the next round of inputs.
        session_cache->training_session.LazyResetGrad();

        return loss;
    }

} // namespace training