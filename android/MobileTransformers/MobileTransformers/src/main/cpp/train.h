//
// Created by martinkorelic on 19/09/2024.
//

#include "onnxruntime/onnxruntime_training_cxx_api.h"
#include "session_cache.h"

namespace training {

    // returns the output of the training graph (loss) and updates the parameters
    // based on the gradients computed.
    //
    // Inputs are bound BY NAME, in the order the training graph declares them
    // (`Ort::TrainingSession::InputNames(true)`), not by a fixed positional list. A decoder graph
    // declares {input_ids, attention_mask, position_ids, labels[batch,seq]}; an encoder
    // classification graph declares {input_ids, attention_mask, token_type_ids, labels[batch]} —
    // different names, a different count, and a different label rank. Binding positionally either
    // threw inside ORT or, worse, bound the wrong tensor to the wrong input.
    //
    // `position_ids` and `token_type_ids` are SYNTHESIZED here, and only when the graph asks for
    // them. `labels_count` is the number of label elements the caller actually supplied; the label
    // rank is derived from it (== batch*seq -> per-token [batch, seq]; == batch -> per-sequence
    // [batch]) and any other value fails closed naming the counts. An input name the binder does
    // not know also fails closed naming it, rather than being silently skipped.
    float train_step(TrainingSessionCache* session_cache,
                     int64_t* input_ids,
                     int64_t* attention_mask,
                     int64_t* labels,
                     int64_t batch_size,
                     int64_t sequence_length,
                     int64_t labels_count);

    void optimizer_step(TrainingSessionCache* session_cache);

} // namespace training
