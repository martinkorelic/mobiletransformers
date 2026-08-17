//
// Host shim for <android/log.h> so the ORT-free headers (handoff_io.h, mem_probe.h,
// constants/merger_variant.h) can be compiled and unit-tested on a desktop.
// Only the logging macros' backing function is needed; output goes to stderr.
//
#ifndef MOBILETRANSFORMERS_HOST_ANDROID_LOG_H
#define MOBILETRANSFORMERS_HOST_ANDROID_LOG_H

#include <cstdarg>
#include <cstdio>

enum android_LogPriority {
    ANDROID_LOG_VERBOSE = 2,
    ANDROID_LOG_DEBUG,
    ANDROID_LOG_INFO,
    ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR,
};

inline int __android_log_print(int prio, const char* tag, const char* fmt, ...) {
    (void)prio;
    (void)tag;
    va_list args;
    va_start(args, fmt);
    const int n = std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    return n;
}

#endif  // MOBILETRANSFORMERS_HOST_ANDROID_LOG_H
