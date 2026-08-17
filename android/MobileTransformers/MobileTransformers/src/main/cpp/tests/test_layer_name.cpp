// Host tests for layer_name.h — the single definition of how an adapted layer is spelled in C++.
//
// Every case here is a bug that actually shipped and could only be caught on a device. The conversions
// were previously open-coded at nine call sites with the prefixes as string literals; when two of those
// literals disagreed the merge still reported success and wrote nothing.

#include <gtest/gtest.h>

#include "layer_name.h"

namespace {

// The five spellings of ONE layer, taken verbatim from a real SmolLM2-135M package.
constexpr const char* kRaw = "base_model.model.model.layers.9.self_attn.q_proj";
constexpr const char* kCheckpoint = "backbone.model.layers.9.self_attn.q_proj";
constexpr const char* kHandoffKey = "base_model.model.model.layers.9.self_attn.q_proj.base_layer";

TEST(LayerName, RawToCheckpointAndBack) {
    EXPECT_EQ(layer_name::to_checkpoint(kRaw), kCheckpoint);
    EXPECT_EQ(layer_name::to_raw(kCheckpoint), kRaw);
}

TEST(LayerName, ConversionsRoundTrip) {
    EXPECT_EQ(layer_name::to_raw(layer_name::to_checkpoint(kRaw)), kRaw);
    EXPECT_EQ(layer_name::to_checkpoint(layer_name::to_raw(kCheckpoint)), kCheckpoint);
}

TEST(LayerName, ConversionsLeaveForeignNamesAlone) {
    // Not a no-op guard for its own sake: a silently-rewritten unrelated name would produce a lookup
    // miss that looks exactly like a genuinely absent parameter.
    const std::string other = "model.layers.9.self_attn.q_proj.MatMul.weight";
    EXPECT_EQ(layer_name::to_checkpoint(other), other);
    EXPECT_EQ(layer_name::to_raw(other), other);
}

// The "Missing base weight for LoRA merger" defect: peft wraps the original Linear as `base_layer`,
// so `<layer>.weight` matches nothing in the checkpoint — for any layer, ever.
TEST(LayerName, CheckpointWeightParamIncludesBaseLayer) {
    EXPECT_EQ(layer_name::checkpoint_weight_param(kCheckpoint),
              "backbone.model.layers.9.self_attn.q_proj.base_layer.weight");
}

TEST(LayerName, CheckpointWeightParamCarriesQuantRoles) {
    EXPECT_EQ(layer_name::checkpoint_weight_param(kCheckpoint, "weight_scale"),
              "backbone.model.layers.9.self_attn.q_proj.base_layer.weight_scale");
}

// Idempotence matters because the name reaching these helpers may already carry the suffix (the handoff
// map records `trainingBaseLayerName` WITH it). Doubling it up would miss just as surely as omitting it.
TEST(LayerName, BaseLayerSuffixIsIdempotent) {
    const std::string once = layer_name::with_base_layer(kCheckpoint);
    EXPECT_EQ(layer_name::with_base_layer(once), once);
    EXPECT_EQ(layer_name::without_base_layer(once), kCheckpoint);
    EXPECT_EQ(layer_name::without_base_layer(kCheckpoint), kCheckpoint);
}

// The defect that made all 60 merges write nothing: find_handoff_entry varied the SUFFIX but not the
// PREFIX, so a merge loop working in checkpoint space never matched a raw-keyed map.
TEST(LayerName, CandidateKeysCoverBothPrefixAndSuffix) {
    const auto keys = layer_name::candidate_handoff_keys(kCheckpoint);
    EXPECT_NE(std::find(keys.begin(), keys.end(), kHandoffKey), keys.end())
        << "the real handoff-map key must be reachable from the merge loop's spelling";
    EXPECT_NE(std::find(keys.begin(), keys.end(), kRaw), keys.end());
    EXPECT_NE(std::find(keys.begin(), keys.end(), kCheckpoint), keys.end());
}

TEST(LayerName, CandidateKeysWorkFromEitherDirection) {
    // A caller may hold the raw name instead; it must still reach the checkpoint spellings.
    const auto keys = layer_name::candidate_handoff_keys(kRaw);
    EXPECT_NE(std::find(keys.begin(), keys.end(), kHandoffKey), keys.end());
}

TEST(LayerName, CandidateKeysAreDeduplicated) {
    // `layer` may already be raw and already suffixed, collapsing all four forms into one. Duplicates
    // are harmless for correctness but mask which form actually matched when debugging a miss.
    const auto keys = layer_name::candidate_handoff_keys(kHandoffKey);
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            EXPECT_NE(keys[i], keys[j]) << "duplicate candidate at " << i << " and " << j;
        }
    }
}

// Guards the cross-language contract: these literals must equal Python's in
// artifacts/checkpoint_names.py (and packages/WeightHandoffMap.kt). They describe one wire format
// (weight_handoff_map.json).
TEST(LayerName, WrapperVocabularyMatchesThePythonTwin) {
    EXPECT_STREQ(layer_name::kRawPrefix, "base_model.model.");
    EXPECT_STREQ(layer_name::kCheckpointPrefix, "backbone.");
    EXPECT_STREQ(layer_name::kBaseLayerSuffix, ".base_layer");
}

// #33: the pair is the two WRAPPERS, not a decoder's own first module. Spelled
// `base_model.model.model.` -> `backbone.model.` it is identical for every decoder (asserted above via
// kRaw/kCheckpoint) and converts NOTHING for an encoder, whose path is `bert.encoder.layer…`.
TEST(LayerName, ConvertsAnEncoderPathAsWellAsADecoderPath) {
    const std::string encoder_raw = "base_model.model.bert.encoder.layer.0.attention.self.query";
    const std::string encoder_ckpt = "backbone.bert.encoder.layer.0.attention.self.query";

    EXPECT_EQ(layer_name::to_checkpoint(encoder_raw), encoder_ckpt);
    EXPECT_EQ(layer_name::to_raw(encoder_ckpt), encoder_raw);
    EXPECT_EQ(layer_name::to_checkpoint(kRaw), kCheckpoint);  // decoder mapping unchanged
    // checkpoint_weight_param takes a name already in checkpoint space — it only adds the suffix.
    EXPECT_EQ(layer_name::checkpoint_weight_param(layer_name::to_checkpoint(encoder_raw)),
              "backbone.bert.encoder.layer.0.attention.self.query.base_layer.weight");
}

}  // namespace
