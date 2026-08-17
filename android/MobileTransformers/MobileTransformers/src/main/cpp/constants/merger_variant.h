//
// #6: C++ mirror of the MergerVariant enum owned by
// src/mobiletransformers/config/constants.py (and mirrored in Kotlin at
// constants/MergerVariant.kt). Wire values are checked against Python by
// `make parity` — do not edit them by hand without regenerating.
//
// This replaces the raw `merger_type == "lora"` string dispatch that used to live in
// weight_merger.cpp. The variant is RESOLVED data (adapter shape + quantization), and
// weight_handoff_map.json's `mergerModels` is keyed by exactly these wire values, so the merger
// session is now selected by a typed tag rather than by a manufactured string that had to
// coincidentally match the map's keys.
//

#ifndef MOBILETRANSFORMERS_MERGER_VARIANT_H
#define MOBILETRANSFORMERS_MERGER_VARIANT_H

#include <optional>
#include <string>
#include <utility>

enum class MergerVariant {
    LORA,
    LORA_Q,
    MARS_Q,
};

// The single wire-value table (parity-checked against ENUM_REGISTRY["MergerVariant"]).
inline constexpr std::pair<MergerVariant, const char*> kMergerVariantWire[] = {
        {MergerVariant::LORA, "lora"},
        {MergerVariant::LORA_Q, "lora_q"},
        {MergerVariant::MARS_Q, "mars_q"},
};

inline const char* to_wire(MergerVariant v) {
    for (const auto& [value, wire] : kMergerVariantWire) {
        if (value == v) return wire;
    }
    return "";  // unreachable: MergerVariant is a closed enum
}

// Fail-closed parse: an unknown tag yields nullopt rather than a silent default.
inline std::optional<MergerVariant> merger_variant_from_wire(const std::string& wire) {
    for (const auto& [value, w] : kMergerVariantWire) {
        if (wire == w) return value;
    }
    return std::nullopt;
}

#endif  // MOBILETRANSFORMERS_MERGER_VARIANT_H
