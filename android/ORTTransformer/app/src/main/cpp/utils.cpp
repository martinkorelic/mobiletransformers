//
// Created by bmeswani on 2/16/2023.
//

#include "utils.h"
#include <android/log.h>

#define LOG_TAG "ORTTransformer"

namespace utils {

    std::string JString2String(JNIEnv *env, jstring jStr) {
        if (!jStr)
            return std::string();

        const jclass stringClass = env->GetObjectClass(jStr);
        const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes",
                                                    "(Ljava/lang/String;)[B");
        const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                           env->NewStringUTF(
                                                                                   "UTF-8"));

        size_t length = (size_t) env->GetArrayLength(stringJbytes);
        jbyte *pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

        std::string ret = std::string((char *) pBytes, length);
        env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

        env->DeleteLocalRef(stringJbytes);
        env->DeleteLocalRef(stringClass);
        return ret;
    }

    /*
     * Function to initialize labels.
     * Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100. Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
     * */
    void initialize_labels(int64_t* input_ids, int64_t* labels, int64_t batch_size, int64_t sequence_length) {
        for (int64_t i = 0; i < batch_size; ++i) {
            for (int64_t j = 0; j < sequence_length; ++j) {

                if (input_ids[i * sequence_length + j] == 0) {  // Assuming '0' is padding token ID
                    labels[i * sequence_length + j] = -100;  // Ignore padding tokens in loss computation
                } else {
                    // For language modeling, shift input ids for labels
                    if (j < sequence_length - 1) {
                        labels[i * sequence_length + j] = input_ids[i * sequence_length + j + 1];
                    } else {
                        labels[i * sequence_length + j] = -100;  // Last token does not have a next token
                    }
                }
            }
        }
    }

} // utils
