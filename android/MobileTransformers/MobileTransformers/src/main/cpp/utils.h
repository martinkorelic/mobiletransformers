//
// Created by martinkorelic on 19/9/2024.
//

#include <string>
#include <jni.h>
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

/**
 * Useful utility functions
 */
namespace utils {

    /**
     * Loads bytes from a given file path.
     *
     * @param path
     * @return
     */
    std::string LoadBytesFromFile(const std::string& path);

    /**
     * Convert jstring to std::string
     *
     * @param env
     * @param jStr
     * @return
     */
    std::string JString2String(JNIEnv *env, jstring jStr);

    /**
     * Initialize labels from batch size and sequence length. \n
     *
     * NOTE: Not being actively used.
     *
     * @param input_ids
     * @param labels
     * @param batch_size
     * @param sequence_length
     */
    void initialize_labels(int64_t* input_ids, int64_t* labels, int64_t batch_size, int64_t sequence_length);

    /**
     * Logs out tensor information
     * @param tensor
     */
    //void print_tensor_info(const Ort::Value& tensor);

} // utils