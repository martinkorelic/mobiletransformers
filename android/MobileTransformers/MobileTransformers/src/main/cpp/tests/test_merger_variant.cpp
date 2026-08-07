// #6: the typed MergerVariant that replaced `merger_type == "lora"` string dispatch.

#include <gtest/gtest.h>

#include <string>

#include "constants/merger_variant.h"

TEST(MergerVariant, WireValuesMatchThePythonEnum) {
    // Mirrors ENUM_REGISTRY["MergerVariant"] in config/constants.py (checked by `make parity`).
    EXPECT_STREQ(to_wire(MergerVariant::LORA), "lora");
    EXPECT_STREQ(to_wire(MergerVariant::LORA_Q), "lora_q");
    EXPECT_STREQ(to_wire(MergerVariant::MARS_Q), "mars_q");
}

TEST(MergerVariant, RoundTripsThroughTheWireValue) {
    for (const auto& [value, wire] : kMergerVariantWire) {
        const auto parsed = merger_variant_from_wire(wire);
        ASSERT_TRUE(parsed.has_value()) << wire;
        EXPECT_EQ(*parsed, value);
    }
}

TEST(MergerVariant, FailsClosedOnUnknownTag) {
    // The handoff map's mergerModels keys are parsed at LOAD time, so a bad tag surfaces immediately
    // rather than silently never matching at dispatch time.
    EXPECT_FALSE(merger_variant_from_wire("").has_value());
    EXPECT_FALSE(merger_variant_from_wire("LORA").has_value());  // case-sensitive
    EXPECT_FALSE(merger_variant_from_wire("lora_xs").has_value());
    EXPECT_FALSE(merger_variant_from_wire("mars").has_value());  // fp MARS has no merger variant
}
