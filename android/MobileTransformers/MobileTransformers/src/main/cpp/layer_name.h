#ifndef MOBILETRANSFORMERS_LAYER_NAME_H
#define MOBILETRANSFORMERS_LAYER_NAME_H

#include <string>
#include <vector>

/**
 * @file layer_name.h
 * ONE definition of how an adapted layer is spelled, for every C++ consumer.
 *
 * ## Why this exists
 *
 * The same layer carries five different names across this system, and every consumer used to re-derive
 * the conversion inline:
 *
 * | space                    | spelling                                                        |
 * |--------------------------|-----------------------------------------------------------------|
 * | inference graph          | `model.layers.0.self_attn.q_proj.MatMul.weight`                  |
 * | ORT checkpoint           | `backbone.model.layers.0.self_attn.q_proj.base_layer.weight`     |
 * | `peft_mapping` key (raw) | `base_model.model.model.layers.0.self_attn.q_proj`               |
 * | handoff-map key          | `base_model.model.model.layers.0.self_attn.q_proj.base_layer`     |
 * | merger runtime           | `backbone.model.layers.0.self_attn.q_proj`                        |
 *
 * Five separate defects were one of these forms being compared against another, each time in a code
 * path that could only be exercised on a device:
 *
 *  - the checkpoint lookup omitted `.base_layer`, so *no* layer's base weight was ever found;
 *  - `find_handoff_entry` handled the suffix difference but not the prefix one, so all 60 merges wrote
 *    nothing while reporting success;
 *  - `peft_mapping_[adjusted]` queried a raw-keyed map with an adjusted name, so `operator[]`
 *    default-inserted mid-iteration — corrupting the traversal *and* zeroing `alpha`, which made the
 *    merger compute `weight + 0 * (B @ A)` and emit byte-identical output.
 *
 * Each was found by running it, not by reading it, because the conversions were scattered literals with
 * no single place to be wrong in. Routing every lookup through here is what makes the next one a
 * compile-time or one-site concern.
 *
 * ## Contract
 *
 * This is the C++ twin of Python's `artifacts/handoff_map.py::_strip_wrapper_prefixes` +
 * `candidate_inference_names`, and it deliberately uses the same vocabulary. **If you change the
 * wrapper set here, change it there too** — they describe one wire format, and `weight_handoff_map.json`
 * is the artifact they must agree about.
 *
 * Pure string transforms: no ORT types, no I/O, so it is unit-testable on the host (see
 * `cpp_tests/test_layer_name.cpp`).
 */
namespace layer_name {

/**
 * peft's module wrapper, as it appears in `training_config.json`'s `peft_mapping` keys.
 *
 * The WRAPPER only — not the model's own first module. This pair used to read
 * `base_model.model.model.` / `backbone.model.`, i.e. the same rule with a decoder's `model.layers…`
 * baked in; it is identical for every decoder and matches nothing for an encoder, whose path is
 * `bert.encoder.layer…`.
 */
inline constexpr const char* kRawPrefix = "base_model.model.";

/** ORT's training-graph wrapper, as parameters are named inside the CheckpointState. */
inline constexpr const char* kCheckpointPrefix = "backbone.";

/** peft wraps the original `Linear` as `base_layer`; the adapters sit beside it. */
inline constexpr const char* kBaseLayerSuffix = ".base_layer";

/** Replace @p old_prefix with @p new_prefix when present; otherwise return @p name unchanged. */
inline std::string replace_prefix(const std::string& name,
                                  const std::string& old_prefix,
                                  const std::string& new_prefix) {
    if (name.compare(0, old_prefix.size(), old_prefix) == 0) {
        return new_prefix + name.substr(old_prefix.size());
    }
    return name;
}

inline bool has_suffix(const std::string& name, const std::string& suffix) {
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * raw (`peft_mapping` key) -> checkpoint/merger space.
 * `base_model.model.model.layers.9.self_attn.q_proj` -> `backbone.model.layers.9.self_attn.q_proj`
 * `base_model.model.bert.encoder.layer.9.attention.self.query` ->
 *     `backbone.bert.encoder.layer.9.attention.self.query`
 */
inline std::string to_checkpoint(const std::string& raw_name) {
    return replace_prefix(raw_name, kRawPrefix, kCheckpointPrefix);
}

/**
 * checkpoint/merger space -> raw (`peft_mapping` / handoff-map key) space. Inverse of to_checkpoint.
 *
 * Needed because the merge loop works in checkpoint space while `peft_mapping_` and the handoff map are
 * both keyed raw. Querying one with the other is the single most repeated bug in this file's history.
 */
inline std::string to_raw(const std::string& checkpoint_name) {
    return replace_prefix(checkpoint_name, kCheckpointPrefix, kRawPrefix);
}

/** Append `.base_layer` unless already present. Idempotent, so it cannot double up. */
inline std::string with_base_layer(const std::string& name) {
    return has_suffix(name, kBaseLayerSuffix) ? name : name + kBaseLayerSuffix;
}

/** Strip a trailing `.base_layer` if present. */
inline std::string without_base_layer(const std::string& name) {
    return has_suffix(name, kBaseLayerSuffix)
               ? name.substr(0, name.size() - std::string(kBaseLayerSuffix).size())
               : name;
}

/**
 * The checkpoint parameter holding a layer's frozen base weight.
 *
 * peft keeps the original `Linear` under `.base_layer`, so the weight is
 * `<layer>.base_layer.weight` — NOT `<layer>.weight`, which matches nothing for any layer and was the
 * "Missing base weight for LoRA merger" failure.
 */
inline std::string checkpoint_weight_param(const std::string& layer, const std::string& role = "weight") {
    return with_base_layer(layer) + "." + role;
}

/**
 * Every key the handoff map might legitimately use for @p layer, in probe order.
 *
 * Two independent axes vary — prefix (raw vs checkpoint) and suffix (with vs without `.base_layer`) —
 * so a lookup that fixes only one of them misses. Returning all four from one place means a caller
 * cannot handle three of them and forget the fourth, which is exactly what happened.
 */
inline std::vector<std::string> candidate_handoff_keys(const std::string& layer) {
    const std::string raw = to_raw(layer);
    std::vector<std::string> keys{
        layer,
        with_base_layer(layer),
        raw,
        with_base_layer(raw),
    };
    // De-duplicate while preserving probe order (`layer` may already carry the suffix, or already be raw).
    std::vector<std::string> unique;
    for (const auto& k : keys) {
        bool seen = false;
        for (const auto& u : unique) {
            if (u == k) { seen = true; break; }
        }
        if (!seen) unique.push_back(k);
    }
    return unique;
}

}  // namespace layer_name

#endif  // MOBILETRANSFORMERS_LAYER_NAME_H
