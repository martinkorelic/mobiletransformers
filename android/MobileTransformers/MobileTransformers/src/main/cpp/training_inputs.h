#ifndef MOBILETRANSFORMERS_TRAINING_INPUTS_H
#define MOBILETRANSFORMERS_TRAINING_INPUTS_H

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file training_inputs.h
 * How a training graph's user inputs are bound — decided here, executed in `train.cpp`.
 *
 * ## Why this exists
 *
 * `train_step` used to build its input vector positionally and with fixed shapes:
 *
 * ```
 * std::vector<Ort::Value> user_inputs; // {input_ids, attention_mask, position_ids, labels}
 * const std::vector<int64_t> labels_shape({batch_size, sequence_length});
 * ```
 *
 * That is one architecture's answer, hardcoded. A decoder training graph declares
 * `{input_ids, attention_mask, position_ids, labels[batch, seq]}`; an encoder **classification**
 * graph declares `{input_ids, attention_mask, token_type_ids, labels[batch]}` — a different set, a
 * different order, and a different label rank (`TaskSpec.label_shape` on the Python side calls the
 * latter `["batch_size"]`). Feeding an encoder graph through the positional binder either throws
 * inside ORT or binds the wrong tensor to the wrong input, which is far worse.
 *
 * So the **graph** decides. `Ort::TrainingSession::InputNames(true)` gives the names in the order
 * `TrainStep` wants them; this header turns that list plus the batch geometry into an ordered plan,
 * and `train.cpp` just executes it. Keeping the decision ORT-free is what makes it testable on a
 * host — the same reason `layer_name.h`, `handoff_io.h` and `constants/merger_variant.h` are shaped
 * this way, and the reason the encoder shape can be proven without a phone.
 *
 * ## Fail closed
 *
 * An input name the binder cannot supply is an error naming the input and listing the graph's
 * inputs — never a silent skip, which would surface much later as a wrong number. A label count that
 * matches neither `batch*sequence` nor `batch` is an error naming both counts.
 */
namespace training {

    //: Canonical user-input names of the training graphs this library produces.
    inline constexpr const char* kInputIds = "input_ids";
    inline constexpr const char* kAttentionMask = "attention_mask";
    inline constexpr const char* kPositionIds = "position_ids";
    inline constexpr const char* kTokenTypeIds = "token_type_ids";
    inline constexpr const char* kLabels = "labels";

    /** Where a bound tensor's data comes from. */
    enum class InputSource {
        CallerInputIds,      //!< supplied by the caller
        CallerAttentionMask, //!< supplied by the caller
        CallerLabels,        //!< supplied by the caller
        SyntheticPositions,  //!< 0..sequence_length-1 per batch row, generated here
        SyntheticTokenTypes, //!< all zeros: the classification objective supervises single sequences
    };

    /** One input to bind, in the order the graph declared it. */
    struct BoundInput {
        std::string name;
        InputSource source;
        std::vector<int64_t> shape;
    };

    inline std::string join_names(const std::vector<std::string>& values) {
        std::ostringstream out;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) out << ", ";
            out << values[i];
        }
        return out.str();
    }

    /**
     * Per-token `[batch, seq]` or per-sequence `[batch]`, decided by what the caller actually
     * supplied rather than by a declared constant that can drift away from the data.
     *
     * When `sequence_length == 1` the two are indistinguishable by count, and `[batch, seq]` wins so
     * the decoder path keeps its exact shipped behaviour.
     */
    inline std::vector<int64_t> labels_shape(int64_t batch_size, int64_t sequence_length,
                                             int64_t labels_count) {
        if (batch_size <= 0 || sequence_length <= 0) {
            std::ostringstream msg;
            msg << "batch_size (" << batch_size << ") and sequence_length (" << sequence_length
                << ") must both be positive";
            throw std::runtime_error(msg.str());
        }
        if (labels_count == batch_size * sequence_length) {
            return {batch_size, sequence_length};
        }
        if (labels_count == batch_size) {
            return {batch_size};
        }
        std::ostringstream msg;
        msg << "labels has " << labels_count << " elements, which is neither batch*sequence ("
            << batch_size << "*" << sequence_length << " = " << batch_size * sequence_length
            << ") for a per-token objective nor batch (" << batch_size
            << ") for a per-sequence objective";
        throw std::runtime_error(msg.str());
    }

    /**
     * Turn the graph's declared input names into an ordered binding plan.
     *
     * @param input_names     `Ort::TrainingSession::InputNames(true)`, in TrainStep order.
     * @param batch_size      rows in the batch.
     * @param sequence_length tokens per row.
     * @param labels_count    label elements the caller supplied (decides the label rank).
     * @throws std::runtime_error naming any input this build cannot supply.
     */
    inline std::vector<BoundInput> plan_training_inputs(const std::vector<std::string>& input_names,
                                                        int64_t batch_size, int64_t sequence_length,
                                                        int64_t labels_count) {
        const std::vector<int64_t> token_shape({batch_size, sequence_length});
        std::vector<BoundInput> plan;
        plan.reserve(input_names.size());

        for (const std::string& name : input_names) {
            if (name == kInputIds) {
                plan.push_back({name, InputSource::CallerInputIds, token_shape});
            } else if (name == kAttentionMask) {
                plan.push_back({name, InputSource::CallerAttentionMask, token_shape});
            } else if (name == kPositionIds) {
                plan.push_back({name, InputSource::SyntheticPositions, token_shape});
            } else if (name == kTokenTypeIds) {
                plan.push_back({name, InputSource::SyntheticTokenTypes, token_shape});
            } else if (name == kLabels) {
                plan.push_back({name, InputSource::CallerLabels,
                                labels_shape(batch_size, sequence_length, labels_count)});
            } else {
                std::ostringstream msg;
                msg << "training graph declares an input this build cannot supply: '" << name
                    << "' (graph inputs: " << join_names(input_names) << ")";
                throw std::runtime_error(msg.str());
            }
        }
        return plan;
    }

} // namespace training

#endif // MOBILETRANSFORMERS_TRAINING_INPUTS_H
