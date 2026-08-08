//
// #12: process RSS probe for the memory-mapping experiments (Gate 0.2). Mirrors the Python sampler
// spikes/genai_external_swap/measure_rss.py (VmRSS from /proc/self/status). Header-only, no deps.
//

#ifndef MOBILETRANSFORMERS_MEM_PROBE_H
#define MOBILETRANSFORMERS_MEM_PROBE_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <fstream>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#include "logging.h"

namespace memprobe {

// Resident set size in KiB from /proc/self/status (VmRSS). Returns -1 if unavailable.
inline int64_t read_rss_kb() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) return -1;
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            int64_t value = -1;
            status >> value;  // value is in kB
            return value;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return -1;
}

// Parse VmRSS (kB) out of a /proc/self/status-shaped string. Unit-testable without /proc.
inline int64_t parse_vmrss_kb(const std::string& status_text) {
    const std::string needle = "VmRSS:";
    auto pos = status_text.find(needle);
    if (pos == std::string::npos) return -1;
    pos += needle.size();
    // skip spaces
    while (pos < status_text.size() && (status_text[pos] == ' ' || status_text[pos] == '\t')) ++pos;
    int64_t value = 0;
    bool any = false;
    while (pos < status_text.size() && status_text[pos] >= '0' && status_text[pos] <= '9') {
        value = value * 10 + (status_text[pos] - '0');
        any = true;
        ++pos;
    }
    return any ? value : -1;
}

// #12 (Gate 0.2) toggle for the zero-copy weight load. Default OFF — the shipping path is #23's copy.
//
// An instrumented test cannot set an environment variable in the app process it is measuring, so the
// four-point RSS table (base/merged x copy/mmap) was unreachable while this was env-only. The switch is
// therefore also a system property:
//
//     adb shell setprop debug.mtf.mmap_weights 1
//
// The env var stays as the desktop/spike override and still wins, so existing spike scripts are
// unaffected. Reads the property on every call: the test flips it between session constructions.
inline bool mmap_weights_enabled() {
    if (std::getenv("MTF_MMAP_WEIGHTS") != nullptr) return true;
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.mtf.mmap_weights", value) > 0) {
        return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y';
    }
#endif
    return false;
}

}  // namespace memprobe

#define LOG_RSS(tag) LOGI("[rss] %s: %lld kB", (tag), (long long) memprobe::read_rss_kb())

#endif  // MOBILETRANSFORMERS_MEM_PROBE_H
