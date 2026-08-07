//
// Created by martinkorelic on 19/09/2024.
//

#include "onnxruntime/onnxruntime_training_cxx_api.h"
#include "session_cache.h"

namespace training {

    // returns the output of the training graph (loss) and updates the parameters
    // based on the gradients computed.
    float train_step(TrainingSessionCache* session_cache,
                     int64_t* input_ids,
                     int64_t* attention_mask,
                     int64_t* position_ids,
                     int64_t* labels,
                     int64_t batch_size,
                     int64_t sequence_length);

    void optimizer_step(TrainingSessionCache* session_cache);

} // namespace training
