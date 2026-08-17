#ifndef MOBILETRANSFORMERS_LOGITS_METRICS_H
#define MOBILETRANSFORMERS_LOGITS_METRICS_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

/**
 * @file logits_metrics.h
 * Numbers read off an inference forward pass, so the device can assert what the host already asserts.
 *
 * ## Why this exists
 *
 * The export pipeline gates a package on `artifacts/train_inference_parity.py` — the same tokens through
 * the train and inference graphs, one cross-entropy each, one bounded delta. **Nothing did that after an
 * on-device merge.** `TrainMergeGenerateTest` hashes the trainable `.bin` files and says so in its own
 * comment: proving the bytes changed is not proving the numbers are right. A merge that wrote plausible
 * bytes to the correct filenames and corrupted the values would pass every gate this project had.
 *
 * That is the recurring failure shape: two halves each verified alone (the bytes moved; the loss falls),
 * with the seam between them — do the merged weights actually compute the right thing? — unverified.
 *
 * ## What is here
 *
 * `causal_cross_entropy` is a **deliberate mirror** of the host's `causal_cross_entropy`, down to the
 * shift convention: `logits[:, :-1]` against `input_ids[:, 1:]`. The host docstring records why that
 * matters — pre-shifting once already inflated every printed loss, and a device number computed under a
 * different convention is not comparable to a host number, which defeats the point of measuring it.
 *
 * Accumulation is `double` for the same reason the host casts to float64: a broken graph emits logits in
 * the 1e8 range, and the stable-log-softmax subtraction loses precision at those magnitudes in float32.
 *
 * ## Why it is ORT-free
 *
 * Same reason as `layer_name.h`, `handoff_io.h`, `training_inputs.h` and `constants/merger_variant.h`:
 * the decision is host-testable, so it is pinned by googletest rather than only by a phone. Every one of
 * those headers exists because a defect in exactly this kind of pure logic survived to a device run.
 */
namespace logits_metrics {

    /**
     * A stable reduction of one logits row, used to answer "did the numbers move?".
     *
     * Four independent statistics rather than one: a merge that shifts every logit by a constant leaves
     * `argmax` untouched, and a merge that permutes values leaves `sum` untouched. Together they do not
     * have a plausible common blind spot.
     */
    struct Fingerprint {
        //: Index of the largest logit — the token greedy decoding would emit.
        int64_t argmax = -1;
        //: The largest logit value itself.
        double max_logit = 0.0;
        //: Sum over the vocabulary. Order-dependent in principle, but the order is fixed.
        double sum = 0.0;
        //: Sum of squares — moves when values are redistributed without changing the sum.
        double sum_of_squares = 0.0;
    };

    /**
     * Reduces the logits of the LAST position, which is the row generation actually samples from.
     *
     * @param logits           `[batch, seq, vocab]`, contiguous, as `generateWithKVCache` returns it.
     * @param sequence_length  number of positions in this pass.
     * @param vocab_size       vocabulary width.
     * @throws std::invalid_argument on a non-positive dimension or a null pointer — a caller that has
     *   lost track of the shape must fail here rather than read past the buffer, which is the failure
     *   mode `generateWithKVCache`'s own input-count guard was added for.
     */
    inline Fingerprint fingerprint_last_position(const float *logits,
                                                 int64_t sequence_length,
                                                 int64_t vocab_size) {
        if (logits == nullptr) {
            throw std::invalid_argument("logits fingerprint: null logits pointer");
        }
        if (sequence_length <= 0 || vocab_size <= 0) {
            throw std::invalid_argument(
                    "logits fingerprint: non-positive dimension (sequence_length=" +
                    std::to_string(sequence_length) + ", vocab_size=" + std::to_string(vocab_size) + ")");
        }

        const float *row = logits + (sequence_length - 1) * vocab_size;

        Fingerprint fp;
        fp.max_logit = -std::numeric_limits<double>::infinity();
        for (int64_t v = 0; v < vocab_size; ++v) {
            const double value = static_cast<double>(row[v]);
            fp.sum += value;
            fp.sum_of_squares += value * value;
            if (value > fp.max_logit) {
                fp.max_logit = value;
                fp.argmax = v;
            }
        }
        return fp;
    }

    /**
     * Mean next-token cross-entropy in nats, under the SAME causal shift as the host gate.
     *
     * Position `t` of the logits predicts token `t+1` of the input, so the last position has no target
     * and is dropped — exactly `logits[:, :-1]` vs `input_ids[:, 1:]`.
     *
     * @param logits           `[1, sequence_length, vocab_size]`, contiguous. Batch is 1 on device.
     * @param input_ids        the `sequence_length` token ids that produced those logits.
     * @param sequence_length  must be >= 2, or there is no (prediction, target) pair to score.
     * @param vocab_size       vocabulary width.
     * @throws std::invalid_argument on a null pointer, `sequence_length < 2`, or a target id outside
     *   the vocabulary. The last one is worth failing on rather than clamping: an out-of-range target
     *   means the tokenizer and the graph disagree about the vocabulary, which is a real package defect
     *   and silently scoring it would report a plausible-looking loss for a broken pairing.
     */
    inline double causal_cross_entropy(const float *logits,
                                       const int64_t *input_ids,
                                       int64_t sequence_length,
                                       int64_t vocab_size) {
        if (logits == nullptr || input_ids == nullptr) {
            throw std::invalid_argument("causal cross-entropy: null pointer");
        }
        if (vocab_size <= 0) {
            throw std::invalid_argument("causal cross-entropy: non-positive vocab_size");
        }
        if (sequence_length < 2) {
            throw std::invalid_argument(
                    "causal cross-entropy needs at least 2 positions to form one (prediction, target) "
                    "pair, got sequence_length=" + std::to_string(sequence_length));
        }

        double total = 0.0;
        for (int64_t t = 0; t + 1 < sequence_length; ++t) {
            const float *row = logits + t * vocab_size;
            const int64_t target = input_ids[t + 1];
            if (target < 0 || target >= vocab_size) {
                throw std::invalid_argument(
                        "causal cross-entropy: target token id " + std::to_string(target) +
                        " at position " + std::to_string(t + 1) + " is outside the vocabulary [0, " +
                        std::to_string(vocab_size) + ") — the tokenizer and the graph disagree.");
            }

            // Stable log-softmax: subtract the row max before exponentiating, in double.
            double row_max = -std::numeric_limits<double>::infinity();
            for (int64_t v = 0; v < vocab_size; ++v) {
                const double value = static_cast<double>(row[v]);
                if (value > row_max) {
                    row_max = value;
                }
            }
            double sum_exp = 0.0;
            for (int64_t v = 0; v < vocab_size; ++v) {
                sum_exp += std::exp(static_cast<double>(row[v]) - row_max);
            }
            const double target_log_prob =
                    (static_cast<double>(row[target]) - row_max) - std::log(sum_exp);
            total += -target_log_prob;
        }
        return total / static_cast<double>(sequence_length - 1);
    }

} // namespace logits_metrics

#endif //MOBILETRANSFORMERS_LOGITS_METRICS_H
