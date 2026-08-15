// The sampler's vocabulary bound.
//
// `sampling::effectiveVocabSize` exists because a package's declared vocab_size is a JSON field and
// the graph's logits width is a fact. When they disagreed on device, generation died two steps later
// inside ORT:
//
//   Gather node ... indices element out of data bounds, idx=262145 ... range [-262144,262143]
//
// FunctionGemma's tokenizer declares <image_soft_token> (262144) and <end_of_image> (262145) above a
// 262144-row embedding table, and the exporter sized the vocabulary from the tokenizer. These tests
// pin both halves: the arithmetic of the clamp, and the property it exists to guarantee — that no id
// the sampler returns can be outside the embedding table.

#include <gtest/gtest.h>

#include <vector>

#include "sampling.h"

namespace {

// The real numbers, so the test names the case it came from.
constexpr int kGraphRows = 262144;
constexpr int kDeclaredWithImageTokens = 262146;

TEST(EffectiveVocabSize, ClampsADeclarationWiderThanTheGraph) {
    EXPECT_EQ(sampling::effectiveVocabSize(kDeclaredWithImageTokens, kGraphRows), kGraphRows);
}

TEST(EffectiveVocabSize, LeavesAnAgreeingDeclarationAlone) {
    EXPECT_EQ(sampling::effectiveVocabSize(kGraphRows, kGraphRows), kGraphRows);
}

TEST(EffectiveVocabSize, DoesNotWidenANarrowerDeclaration) {
    // A caller restricting the sampler to a prefix of the vocabulary is deliberate. Widening it
    // would be the same mistake as trusting an over-declaration, in the other direction.
    EXPECT_EQ(sampling::effectiveVocabSize(1000, kGraphRows), 1000);
}

TEST(EffectiveVocabSize, FallsBackToTheDeclarationWhenTheShapeIsUnreadable) {
    // lastLogitsWidth() returns 0 before the first forward pass and when the shape query throws.
    // That must not turn into "sample over zero tokens".
    EXPECT_EQ(sampling::effectiveVocabSize(kGraphRows, 0), kGraphRows);
    EXPECT_EQ(sampling::effectiveVocabSize(kGraphRows, -1), kGraphRows);
}

TEST(EffectiveVocabSize, UsesTheGraphWhenNothingWasDeclared) {
    // vocabSize is 0 on a package whose tokenizer config could not be read at all.
    EXPECT_EQ(sampling::effectiveVocabSize(0, kGraphRows), kGraphRows);
}

// The two failure modes, on a small board. `rows` is what the graph produces; `declared` is what the
// package claims. The arena is oversized so the over-declared reads stay inside test memory — in
// production they run off the end of the tensor, which is the same bug with less predictable values.
constexpr int kRows = 8;
constexpr int kDeclared = 10;

// One decode step: the exact shape of the FunctionGemma failure. The sampler scans two entries past
// the end of the row and returns an id (declared - 1) that the embedding table has no row for — on
// device, 262145 against a 262144-row table.
TEST(GreedySampling, AnOverDeclaredVocabularySelectsAnIdWithNoEmbeddingRow) {
    std::vector<float> arena(kDeclared + 4, 0.0f);
    for (int v = 0; v < kRows; ++v) {
        arena[v] = static_cast<float>(v) * 0.01f;  // a real row, argmax at 7
    }
    for (size_t i = kRows; i < arena.size(); ++i) {
        arena[i] = 999.0f;  // whatever follows the tensor
    }

    const int unclamped = sampling::greedySampling(arena.data(), 1, kDeclared);
    EXPECT_GE(unclamped, kRows) << "the unclamped sampler is expected to reach outside the table; if "
                                   "it no longer does, this test has stopped exercising the hazard";

    const int clamped =
            sampling::greedySampling(arena.data(), 1, sampling::effectiveVocabSize(kDeclared, kRows));
    EXPECT_EQ(clamped, 7) << "a clamped sampler must return the real row's argmax — an id outside "
                             "the table becomes the next input and fails in Gather";
}

// Prefill: `(sequence_length - 1) * vocab_size` is the row offset, so an over-declared vocabulary
// reads the wrong row entirely. This one produces a plausible-looking token rather than a crash,
// which is worse.
TEST(GreedySampling, AnOverDeclaredVocabularyStridesToTheWrongRowOnPrefill) {
    constexpr int sequence_length = 3;
    std::vector<float> arena(sequence_length * kDeclared + 4, 0.0f);
    for (int t = 0; t < sequence_length; ++t) {
        for (int v = 0; v < kRows; ++v) {
            // Each row's argmax is a different id, so reading the wrong row is visible in the answer.
            arena[t * kRows + v] = (v == (t + 2)) ? 5.0f : 0.0f;
        }
    }

    const int correct = sampling::greedySampling(
            arena.data(), sequence_length, sampling::effectiveVocabSize(kDeclared, kRows));
    EXPECT_EQ(correct, 4) << "the last row's argmax";

    const int unclamped = sampling::greedySampling(arena.data(), sequence_length, kDeclared);
    EXPECT_NE(unclamped, correct) << "the over-declared stride is expected to land on other memory; "
                                     "if it agrees, this test has stopped exercising the hazard";
}

}  // namespace
