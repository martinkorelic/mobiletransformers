// Host tests for training_inputs.h — the ORT-free half of the training-input binding (#33 B2).
//
// `train_step` used to build its input vector positionally, with `labels` always shaped
// [batch, sequence]. That is one architecture's answer hardcoded: an encoder classification graph
// declares a different input SET (token_type_ids instead of position_ids), a different ORDER, and a
// per-sequence label rank. The decision is extracted here so both shapes can be proven on a host,
// with the decoder pinned as a regression — a phone is not required to know the plan is right.

#include <gtest/gtest.h>

#include "training_inputs.h"

using training::BoundInput;
using training::InputSource;
using training::labels_shape;
using training::plan_training_inputs;

namespace {

// Exactly what a SmolLM2-style causal-LM training graph declares.
const std::vector<std::string> kDecoderInputs{"input_ids", "attention_mask", "position_ids", "labels"};

// Exactly what the BERT sequence-classification training graph declares (verified against the real
// export in tests/integration/test_encoder_training_gate.py: labels is [batch_size], and the input
// set carries token_type_ids, not position_ids).
const std::vector<std::string> kEncoderInputs{"input_ids", "attention_mask", "token_type_ids", "labels"};

} // namespace

// --- the decoder regression: the shipped shape must not move ---------------------------------

TEST(TrainingInputs, DecoderPlanIsUnchangedFromThePositionalBinder) {
    const auto plan = plan_training_inputs(kDecoderInputs, /*batch=*/2, /*seq=*/8, /*labels=*/16);

    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[0].name, "input_ids");
    EXPECT_EQ(plan[0].source, InputSource::CallerInputIds);
    EXPECT_EQ(plan[1].name, "attention_mask");
    EXPECT_EQ(plan[1].source, InputSource::CallerAttentionMask);
    EXPECT_EQ(plan[2].name, "position_ids");
    EXPECT_EQ(plan[2].source, InputSource::SyntheticPositions);
    EXPECT_EQ(plan[3].name, "labels");
    EXPECT_EQ(plan[3].source, InputSource::CallerLabels);

    // Every tensor is [batch, seq] — including labels, the per-token objective.
    for (const BoundInput& bound : plan) {
        EXPECT_EQ(bound.shape, (std::vector<int64_t>{2, 8})) << bound.name;
    }
}

// --- the encoder: a different set, a different order, a different label rank -------------------

TEST(TrainingInputs, EncoderClassificationBindsTokenTypesAndPerSequenceLabels) {
    const auto plan = plan_training_inputs(kEncoderInputs, /*batch=*/8, /*seq=*/12, /*labels=*/8);

    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[2].name, "token_type_ids");
    EXPECT_EQ(plan[2].source, InputSource::SyntheticTokenTypes);
    EXPECT_EQ(plan[2].shape, (std::vector<int64_t>{8, 12}));

    // The contract that defines this objective: ONE label per sequence, rank 1.
    EXPECT_EQ(plan[3].name, "labels");
    EXPECT_EQ(plan[3].shape, (std::vector<int64_t>{8}));

    // No position_ids anywhere — the graph never asked for one, so none is synthesized.
    for (const BoundInput& bound : plan) {
        EXPECT_NE(bound.source, InputSource::SyntheticPositions);
    }
}

TEST(TrainingInputs, PlanFollowsTheGraphsDeclaredOrderNotAFixedOne) {
    // ORT hands back whatever order the graph declares; TrainStep is positional against THAT order,
    // so the plan must follow it rather than a canonical one.
    const std::vector<std::string> shuffled{"labels", "token_type_ids", "input_ids", "attention_mask"};
    const auto plan = plan_training_inputs(shuffled, /*batch=*/4, /*seq=*/6, /*labels=*/4);

    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[0].name, "labels");
    EXPECT_EQ(plan[0].shape, (std::vector<int64_t>{4}));
    EXPECT_EQ(plan[1].name, "token_type_ids");
    EXPECT_EQ(plan[2].name, "input_ids");
    EXPECT_EQ(plan[3].name, "attention_mask");
}

// --- label rank comes from the data, not from a declared constant -----------------------------

TEST(TrainingInputs, LabelRankIsDerivedFromWhatTheCallerSupplied) {
    EXPECT_EQ(labels_shape(2, 8, 16), (std::vector<int64_t>{2, 8}));  // per-token
    EXPECT_EQ(labels_shape(8, 12, 8), (std::vector<int64_t>{8}));     // per-sequence
}

TEST(TrainingInputs, SingleTokenSequencesKeepTheDecoderRank) {
    // batch*seq == batch when seq == 1, so the two are indistinguishable by count. [batch, seq] wins
    // so the decoder path keeps its exact shipped behaviour rather than silently changing rank.
    EXPECT_EQ(labels_shape(4, 1, 4), (std::vector<int64_t>{4, 1}));
}

// --- fail closed, naming the entity -----------------------------------------------------------

TEST(TrainingInputs, MismatchedLabelCountFailsClosedNamingBothCounts) {
    try {
        labels_shape(2, 8, 5);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("5"), std::string::npos) << what;   // what was supplied
        EXPECT_NE(what.find("16"), std::string::npos) << what;  // what per-token would need
    }
}

TEST(TrainingInputs, UnknownInputFailsClosedNamingItAndTheGraph) {
    const std::vector<std::string> inputs{"input_ids", "pixel_values", "labels"};
    try {
        plan_training_inputs(inputs, 2, 8, 16);
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        // Naming the offending input is the difference between a five-minute fix and an
        // export->push->run cycle spent guessing.
        EXPECT_NE(what.find("pixel_values"), std::string::npos) << what;
        EXPECT_NE(what.find("input_ids"), std::string::npos) << what;
    }
}

TEST(TrainingInputs, NonPositiveGeometryFailsClosed) {
    EXPECT_THROW(labels_shape(0, 8, 0), std::runtime_error);
    EXPECT_THROW(labels_shape(2, 0, 0), std::runtime_error);
}
