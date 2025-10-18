//
// Created by martinkorelic on 20. 07. 25.
//

#ifndef ORTTRANSFORMER_LOGGING_H
#define ORTTRANSFORMER_LOGGING_H

#include <android/log.h>

#define LOG_TAG "ORTransformersMobile"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)

#endif //ORTTRANSFORMER_LOGGING_H
