//
// #12: process RSS probe for the memory-mapping experiments (Gate 0.2). Mirrors the Python sampler
// spikes/genai_external_swap/measure_rss.py (VmRSS from /proc/self/status). Header-only, no deps.
//

#ifndef MOBILETRANSFORMERS_MEM_PROBE_H
#define MOBILETRANSFORMERS_MEM_PROBE_H

#include <cstdio>
#include <cstdint>
#include <string>
#include <fstream>
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

}  // namespace memprobe

#define LOG_RSS(tag) LOGI("[rss] %s: %lld kB", (tag), (long long) memprobe::read_rss_kb())

#endif  // MOBILETRANSFORMERS_MEM_PROBE_H
