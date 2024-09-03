//
// Created by bmeswani on 2/17/2023.
//

#ifndef ORT_PERSONALIZE_INFERENCE_H
#define ORT_PERSONALIZE_INFERENCE_H

#include <cstdint>
#include <string>

#include "session_cache.h"

namespace inference {

    // Runs the inference graph and returns:
    //   - predicted next token based on sampling method
    float *forward(InferenceSessionCache *session_cache, int64_t *input_ids, int64_t *attention_mask,
                   int64_t *position_ids, int64_t batch_size, int64_t sequence_length, size_t vocab_size);

    size_t greedySampling(InferenceSessionCache* session_cache,
                          int64_t* input_ids,
                          int64_t* attention_mask,
                          int64_t* position_ids,
                          int64_t batch_size,
                          int64_t sequence_length,
                          size_t vocab_size);

} // namespace inference

#endif //ORT_PERSONALIZE_INFERENCE_H