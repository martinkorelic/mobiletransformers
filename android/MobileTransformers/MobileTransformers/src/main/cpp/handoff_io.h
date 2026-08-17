//
// #23: the ONE reader of weight_handoff_map.json (#8 schema), shared by the merger WRITE side
// (weight_merger.cpp) and the inference LOAD side (session_cache.h) so tensor identity is derived from
// a single place on device — no second reader, no <dirname>.<filestem> name reconstruction.
//
// Deliberately ORT-free (JSON + POD only): turning the declared per-role dtype/shape into an Ort::Value
// belongs to the load side (session_cache.h), which has the ORT types. Checksums are the primary responsibility
// of the Kotlin precondition (HandoffPrecondition.loadMergedWeightsReady) which runs BEFORE the session
// is created; this reader + the load side add the map-driven naming + dtype/shape fail-closed check.
//

#ifndef MOBILETRANSFORMERS_HANDOFF_IO_H
#define MOBILETRANSFORMERS_HANDOFF_IO_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <fstream>
#include <nlohmann/json.hpp>
#include "logging.h"

// One entry of weight_handoff_map.json (#8 schema / #9 write + #23 load consumer): the SINGLE source of
// tensor identity for the on-device merge AND load. externalDataLocation[role] is the per-tensor .bin;
// inferenceInitializerNames[role] is the canonical graph initializer name. No string-rewrite on device.
struct HandoffEntry {
    std::string trainingBaseLayerName;
    std::unordered_map<std::string, std::string> externalDataLocation;   // role -> "<name>.bin"
    std::unordered_map<std::string, std::string> inferenceInitializerNames; // role -> canonical name
    std::string dtype;                    // entry-level (weight-like role) dtype
    std::vector<int64_t> shape;           // entry-level (weight-like role) shape
    // Per-role on-disk dtype/shape. REQUIRED by the load side: each <name>.bin holds RAW external-data
    // bytes with no header, so this map is the only description of a packed weight_quantized/scale/
    // zero_point tensor, whose layout differs from the entry-level weight's. Empty for maps written
    // before the field existed -> the entry-level pair is the fallback (sound for a single fp role).
    std::unordered_map<std::string, std::string> tensorDtypes;            // role -> dtype
    std::unordered_map<std::string, std::vector<int64_t>> tensorShapes;   // role -> shape
    std::string transposePolicy;
    bool has_quantization = false;

    // On-disk dtype/shape of [role], falling back to the entry-level pair.
    const std::string& dtype_for(const std::string& role) const {
        auto it = tensorDtypes.find(role);
        return it != tensorDtypes.end() ? it->second : dtype;
    }
    const std::vector<int64_t>& shape_for(const std::string& role) const {
        auto it = tensorShapes.find(role);
        return it != tensorShapes.end() ? it->second : shape;
    }
};

// ---- Semver gate mirroring mobiletransformers/artifacts/versioning.py::check_compat (F1). ----
inline bool parse_version(const std::string& v, int& major, int& minor) {
    auto dot = v.find('.');
    if (dot == std::string::npos) return false;
    try {
        major = std::stoi(v.substr(0, dot));
        minor = std::stoi(v.substr(dot + 1));
    } catch (...) { return false; }
    return major >= 0 && minor >= 0;
}

inline bool check_compat(const std::string& docSchema, const std::string& docMinReader,
                         const std::string& readerSchema) {
    int dMaj, dMin, rqMaj, rqMin, rMaj, rMin;
    if (!parse_version(docSchema, dMaj, dMin) || !parse_version(docMinReader, rqMaj, rqMin) ||
        !parse_version(readerSchema, rMaj, rMin)) {
        return false;
    }
    if (dMaj > rMaj) return false;                                   // doc needs a newer major SDK
    if (rMaj < rqMaj || (rMaj == rqMaj && rMin < rqMin)) return false; // reader below doc minReaderVersion
    return true;
}

// Load + version-gate weight_handoff_map.json into [out] keyed by trainingBaseLayerName. Optionally
// collects mergerModels (MergerVariant tag -> ONNX filename). Returns false (and leaves [out] empty) on
// open/parse/schema failure — the caller fails closed.
inline bool load_handoff_entries(const std::string& json_path, const std::string& readerVersion,
                                 std::unordered_map<std::string, HandoffEntry>& out,
                                 std::unordered_map<std::string, std::string>* merger_models = nullptr) {
    using nlohmann::json;
    out.clear();
    if (merger_models) merger_models->clear();
    try {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            LOGE("Failed to open handoff map: %s", json_path.c_str());
            return false;
        }
        json j;
        file >> j;

        const std::string docSchema = j.value("schemaVersion", "");
        const std::string docMinReader = j.value("minReaderVersion", "");
        if (!check_compat(docSchema, docMinReader, readerVersion)) {
            LOGE("handoff map schema %s (minReader %s) incompatible with reader %s",
                 docSchema.c_str(), docMinReader.c_str(), readerVersion.c_str());
            return false;
        }

        if (merger_models && j.contains("mergerModels")) {
            for (const auto& [variant, filename] : j["mergerModels"].items())
                (*merger_models)[variant] = filename.get<std::string>();
        }

        for (const auto& entry_json : j.value("entries", json::array())) {
            HandoffEntry entry;
            entry.trainingBaseLayerName = entry_json.value("trainingBaseLayerName", "");
            entry.dtype = entry_json.value("dtype", "");
            entry.transposePolicy = entry_json.value("transposePolicy", "no_transpose");
            entry.has_quantization =
                entry_json.contains("quantization") && !entry_json["quantization"].is_null();
            if (entry_json.contains("shape")) {
                for (const auto& d : entry_json["shape"]) entry.shape.push_back(d.get<int64_t>());
            }
            if (entry_json.contains("tensorDtypes")) {
                for (const auto& [role, dt] : entry_json["tensorDtypes"].items())
                    entry.tensorDtypes[role] = dt.get<std::string>();
            }
            if (entry_json.contains("tensorShapes")) {
                for (const auto& [role, dims] : entry_json["tensorShapes"].items()) {
                    std::vector<int64_t> shape;
                    for (const auto& d : dims) shape.push_back(d.get<int64_t>());
                    entry.tensorShapes[role] = std::move(shape);
                }
            }
            if (entry_json.contains("externalDataLocation")) {
                for (const auto& [role, loc] : entry_json["externalDataLocation"].items())
                    entry.externalDataLocation[role] = loc.get<std::string>();
            }
            if (entry_json.contains("inferenceInitializerNames")) {
                for (const auto& [role, name] : entry_json["inferenceInitializerNames"].items())
                    entry.inferenceInitializerNames[role] = name.get<std::string>();
            }
            out[entry.trainingBaseLayerName] = entry;
        }
        LOGI("Loaded handoff map: %zu entries", out.size());
        return !out.empty();
    } catch (const std::exception& e) {
        LOGE("Error loading handoff map: %s", e.what());
        out.clear();
        return false;
    }
}

#endif //MOBILETRANSFORMERS_HANDOFF_IO_H
