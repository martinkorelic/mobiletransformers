//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_UTILS_H
#define ORT_PERSONALIZE_UTILS_H

#include <string>
#include <jni.h>


namespace utils {

    std::string LoadBytesFromFile(const std::string& path);

    // Convert jstring to std::string
    std::string JString2String(JNIEnv *env, jstring jStr);

    void initialize_labels(int64_t* input_ids, int64_t* labels, int64_t batch_size, int64_t sequence_length);

} // utils



#endif //ORT_PERSONALIZE_UTILS_H
