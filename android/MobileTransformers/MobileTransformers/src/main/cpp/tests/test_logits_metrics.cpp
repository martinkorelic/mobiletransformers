// Host tests for logits_metrics.h — the numbers the device reads off a forward pass.
//
// These exist so the on-device post-merge assertion rests on arithmetic that was checked somewhere
// cheap. The cross-entropy here must match `artifacts/train_inference_parity.py::causal_cross_entropy`
// including its shift convention, or the device number is not comparable to the host gate it mirrors —
// which would make the whole measurement decorative.

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "logits_metrics.h"

namespace {

    // Two positions, three-token vocabulary. Small enough to compute the expected value by hand.
    std::vector<float> tiny_logits() {
        return {
                // t=0
                1.0f, 2.0f, 3.0f,
                // t=1
                0.5f, 0.5f, 0.5f,
        };
    }

    TEST(LogitsFingerprint, ReducesTheLastPositionNotTheFirst) {
        const auto logits = tiny_logits();

        const auto fp = logits_metrics::fingerprint_last_position(logits.data(), 2, 3);

        // The last row is uniform 0.5 — argmax falls on index 0, and sum is 1.5. If it had reduced the
        // FIRST row instead, argmax would be 2 and sum 6.0.
        EXPECT_EQ(fp.argmax, 0);
        EXPECT_DOUBLE_EQ(fp.sum, 1.5);
        EXPECT_DOUBLE_EQ(fp.max_logit, 0.5);
        EXPECT_DOUBLE_EQ(fp.sum_of_squares, 0.75);
    }

    TEST(LogitsFingerprint, DistinguishesARedistributionThatKeepsTheSum) {
        // sum alone is blind to this pair; sum_of_squares is not. That is why the fingerprint carries
        // four statistics rather than one — a merge that corrupts values without changing their total
        // must not read as "unchanged".
        const std::vector<float> a = {1.0f, 1.0f, 1.0f};
        const std::vector<float> b = {0.0f, 0.0f, 3.0f};

        const auto fa = logits_metrics::fingerprint_last_position(a.data(), 1, 3);
        const auto fb = logits_metrics::fingerprint_last_position(b.data(), 1, 3);

        EXPECT_DOUBLE_EQ(fa.sum, fb.sum);
        EXPECT_NE(fa.sum_of_squares, fb.sum_of_squares);
    }

    TEST(LogitsFingerprint, FailsClosedOnBadShapeOrNullPointer) {
        const auto logits = tiny_logits();

        EXPECT_THROW(logits_metrics::fingerprint_last_position(nullptr, 2, 3), std::invalid_argument);
        EXPECT_THROW(logits_metrics::fingerprint_last_position(logits.data(), 0, 3), std::invalid_argument);
        EXPECT_THROW(logits_metrics::fingerprint_last_position(logits.data(), 2, 0), std::invalid_argument);
    }

    TEST(CausalCrossEntropy, AppliesTheSameShiftAsTheHostGate) {
        // Position 0 predicts token 1. Only that one pair is scored: position 1 has no target, exactly
        // as `logits[:, :-1]` vs `input_ids[:, 1:]` on the host.
        const auto logits = tiny_logits();
        const std::vector<int64_t> input_ids = {0, 2};

        const double loss = logits_metrics::causal_cross_entropy(logits.data(), input_ids.data(), 2, 3);

        // -log softmax([1,2,3])[2] = log(e^-2 + e^-1 + 1) = 0.40760596...
        const double expected = std::log(std::exp(-2.0) + std::exp(-1.0) + 1.0);
        EXPECT_NEAR(loss, expected, 1e-12);
    }

    TEST(CausalCrossEntropy, AUniformRowScoresLogVocabSize) {
        // The self-calibrating reference the host gate leans on: a model predicting nothing sits at
        // ln(vocab_size). A device number at or above this floor means the weights are gone.
        const std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const std::vector<int64_t> input_ids = {0, 1};

        const double loss = logits_metrics::causal_cross_entropy(logits.data(), input_ids.data(), 2, 3);

        EXPECT_NEAR(loss, std::log(3.0), 1e-12);
    }

    TEST(CausalCrossEntropy, IsStableAtMagnitudesThatOverflowFloat32Exp) {
        // Broken graphs have been observed emitting logits in the 1e8 range. Without the max
        // subtraction this is inf/nan; the host casts to float64 for the same reason.
        const std::vector<float> logits = {1e8f, 2e8f, 3e8f, 0.0f, 0.0f, 0.0f};
        const std::vector<int64_t> input_ids = {0, 2};

        const double loss = logits_metrics::causal_cross_entropy(logits.data(), input_ids.data(), 2, 3);

        EXPECT_TRUE(std::isfinite(loss));
        EXPECT_NEAR(loss, 0.0, 1e-9);  // token 2 is the argmax by 1e8 — probability ~1, loss ~0
    }

    TEST(CausalCrossEntropy, FailsClosedOnATargetOutsideTheVocabulary) {
        // Tokenizer and graph disagreeing about the vocabulary is a real package defect. Scoring it
        // anyway would report a plausible-looking number for a broken pairing.
        const auto logits = tiny_logits();
        const std::vector<int64_t> bad = {0, 7};

        EXPECT_THROW(logits_metrics::causal_cross_entropy(logits.data(), bad.data(), 2, 3),
                     std::invalid_argument);
    }

    TEST(CausalCrossEntropy, FailsClosedWhenThereIsNoPairToScore) {
        const auto logits = tiny_logits();
        const std::vector<int64_t> input_ids = {0};

        EXPECT_THROW(logits_metrics::causal_cross_entropy(logits.data(), input_ids.data(), 1, 3),
                     std::invalid_argument);
    }

} // namespace
