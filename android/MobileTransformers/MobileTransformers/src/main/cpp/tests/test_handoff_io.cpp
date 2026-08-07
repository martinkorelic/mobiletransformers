// #8/#23: the weight_handoff_map.json reader + the C++ check_compat mirror.
//
// The C++ check_compat mirror has existed since #8 with NO test — the shared cross-language fixture
// (tests/fixtures/check_compat_cases.json) was consumed by Python and Kotlin only. This closes that.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include "handoff_io.h"

namespace {

std::string write_temp(const std::string& contents) {
    std::string path = std::string(std::tmpnam(nullptr)) + ".json";
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

}  // namespace

// --- check_compat: byte-identical semantics with Python/Kotlin -------------------------------------

TEST(CheckCompat, MatchesTheSharedCrossLanguageFixture) {
    const std::string fixture =
            std::string(MTF_REPO_ROOT) + "/tests/fixtures/check_compat_cases.json";
    std::ifstream in(fixture);
    ASSERT_TRUE(in.is_open()) << "shared fixture not found: " << fixture;

    nlohmann::json j;
    in >> j;
    ASSERT_FALSE(j["cases"].empty());

    for (const auto& c : j["cases"]) {
        const std::string doc = c["doc"].get<std::string>();
        const std::string min_reader = c["minReader"].get<std::string>();
        const std::string reader = c["reader"].get<std::string>();
        const bool expect_accept = c["expect"].get<std::string>() == "accept";
        EXPECT_EQ(check_compat(doc, min_reader, reader), expect_accept)
                << "case: " << c["why"].get<std::string>();
    }
}

// --- load_handoff_entries -------------------------------------------------------------------------

TEST(LoadHandoffEntries, ReadsPerRoleDtypeAndShape) {
    // #23: each <name>.bin is RAW external data with no header, so per-role dtype/shape in the map is
    // the loader's only description of a packed weight_quantized/scale/zero_point.
    const std::string path = write_temp(R"({
      "schemaVersion": "1.0", "minReaderVersion": "1.0",
      "entries": [{
        "trainingBaseLayerName": "layer0.base_layer",
        "dtype": "float16", "shape": [8, 4],
        "tensorDtypes": {"weight_quantized": "uint8", "scale": "float16"},
        "tensorShapes": {"weight_quantized": [8, 2], "scale": [8, 1]},
        "inferenceInitializerNames": {"weight_quantized": "l0.qweight", "scale": "l0.scales"},
        "externalDataLocation": {"weight_quantized": "l0.qweight.bin", "scale": "l0.scales.bin"},
        "quantization": {"weightQuantizedName": "l0.qweight"}
      }]
    })");

    std::unordered_map<std::string, HandoffEntry> out;
    ASSERT_TRUE(load_handoff_entries(path, "1.0", out));
    ASSERT_EQ(out.size(), 1u);

    const HandoffEntry& e = out.at("layer0.base_layer");
    EXPECT_TRUE(e.has_quantization);
    EXPECT_EQ(e.dtype_for("weight_quantized"), "uint8");
    EXPECT_EQ(e.shape_for("weight_quantized"), (std::vector<int64_t>{8, 2}));
    EXPECT_EQ(e.dtype_for("scale"), "float16");
    EXPECT_EQ(e.shape_for("scale"), (std::vector<int64_t>{8, 1}));
    std::remove(path.c_str());
}

TEST(LoadHandoffEntries, FallsBackToEntryLevelDtypeAndShape) {
    // Maps written before tensorDtypes/tensorShapes existed must still resolve their single role.
    const std::string path = write_temp(R"({
      "schemaVersion": "1.0", "minReaderVersion": "1.0",
      "entries": [{
        "trainingBaseLayerName": "layer0.base_layer",
        "dtype": "float16", "shape": [8, 4],
        "inferenceInitializerNames": {"weight": "l0.weight"},
        "externalDataLocation": {"weight": "l0.weight.bin"}
      }]
    })");

    std::unordered_map<std::string, HandoffEntry> out;
    ASSERT_TRUE(load_handoff_entries(path, "1.0", out));
    const HandoffEntry& e = out.at("layer0.base_layer");
    EXPECT_EQ(e.dtype_for("weight"), "float16");
    EXPECT_EQ(e.shape_for("weight"), (std::vector<int64_t>{8, 4}));
    std::remove(path.c_str());
}

TEST(LoadHandoffEntries, FailsClosedOnIncompatibleSchema) {
    const std::string path = write_temp(R"({
      "schemaVersion": "2.0", "minReaderVersion": "2.0",
      "entries": [{"trainingBaseLayerName": "l", "dtype": "float16", "shape": [1]}]
    })");
    std::unordered_map<std::string, HandoffEntry> out;
    EXPECT_FALSE(load_handoff_entries(path, "1.0", out));
    EXPECT_TRUE(out.empty()) << "a rejected map must leave NO entries behind";
    std::remove(path.c_str());
}

TEST(LoadHandoffEntries, FailsClosedOnMissingFileAndOnGarbage) {
    std::unordered_map<std::string, HandoffEntry> out;
    EXPECT_FALSE(load_handoff_entries("/nonexistent/weight_handoff_map.json", "1.0", out));

    const std::string path = write_temp("{ this is not json ");
    EXPECT_FALSE(load_handoff_entries(path, "1.0", out));
    EXPECT_TRUE(out.empty());
    std::remove(path.c_str());
}

TEST(LoadHandoffEntries, CollectsMergerModelsByVariantTag) {
    const std::string path = write_temp(R"({
      "schemaVersion": "1.0", "minReaderVersion": "1.0",
      "mergerModels": {"lora": "merger_lora_fpin_fpout.onnx", "mars_q": "merger_mars_q_qin_qout.onnx"},
      "entries": [{
        "trainingBaseLayerName": "l", "dtype": "float16", "shape": [2],
        "inferenceInitializerNames": {"weight": "w"},
        "externalDataLocation": {"weight": "w.bin"}
      }]
    })");
    std::unordered_map<std::string, HandoffEntry> out;
    std::unordered_map<std::string, std::string> mergers;
    ASSERT_TRUE(load_handoff_entries(path, "1.0", out, &mergers));
    EXPECT_EQ(mergers.at("lora"), "merger_lora_fpin_fpout.onnx");
    EXPECT_EQ(mergers.at("mars_q"), "merger_mars_q_qin_qout.onnx");
    std::remove(path.c_str());
}
