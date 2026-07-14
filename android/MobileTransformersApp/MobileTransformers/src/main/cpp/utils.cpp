//
// Created by martinkorelic on 19/9/2024.
//

#include "utils.h"
#include <android/log.h>
#include <fstream>
#include <iostream>
#include "logging.h"

namespace utils {

    std::string LoadBytesFromFile(const std::string& path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            __android_log_print(ANDROID_LOG_ERROR, "utils.LoadBytesFromFile", "Cannot open file: %s", path.c_str());
            return ""; // Return an empty string to indicate failure
        }

        // Seek to the end to determine file size
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);

        // If file is empty, log a warning and return an empty string
        if (size == 0) {
            __android_log_print(ANDROID_LOG_ERROR, "utils.LoadBytesFromFile", "File is empty: %s", path.c_str());
            return "";
        }

        // Resize the string to fit the file contents
        std::string data(size, '\0');
        fs.read(data.data(), size);

        // Check if the read operation failed
        if (!fs) {
            __android_log_print(ANDROID_LOG_ERROR, "utils.LoadBytesFromFile", "Error reading file: %s", path.c_str());
            return "";
        }

        return data;
    }

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

//    void print_tensor_info(const Ort::Value& tensor) {
//        try {
//            auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
//            auto element_type = tensor_info.GetElementType();
//            auto shape = tensor_info.GetShape();
//
//            // Print data type
//            std::string type_name;
//            switch (element_type) {
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
//                    type_name = "FLOAT";
//                    break;
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
//                    type_name = "DOUBLE";
//                    break;
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
//                    type_name = "INT8";
//                    break;
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
//                    type_name = "UINT8";
//                    break;
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
//                    type_name = "INT32";
//                    break;
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
//                    type_name = "INT64";
//                    break;
//                default:
//                    type_name = "UNKNOWN(" + std::to_string(element_type) + ")";
//            }
//
//            LOGI("Tensor Data Type: %s", type_name.c_str());
//
//            // Print dimensions
//            std::string shape_str = "[";
//            size_t total_elements = 1;
//            for (size_t i = 0; i < shape.size(); ++i) {
//                if (i > 0) shape_str += ", ";
//                shape_str += std::to_string(shape[i]);
//                total_elements *= shape[i];
//            }
//            shape_str += "]";
//            LOGI("Tensor Dimensions: %s", shape_str.c_str());
//
//            // Print first 5 elements
//            if (total_elements == 0) {
//                LOGI("Tensor is empty");
//                return;
//            }
//
//            size_t elements_to_print = std::min(total_elements, size_t(5));
//
//            switch (element_type) {
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
//                    const float* data = tensor.GetTensorData<float>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(data[i]);
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
//                    const double* data = tensor.GetTensorData<double>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(data[i]);
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
//                    const int8_t* data = tensor.GetTensorData<int8_t>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(static_cast<int>(data[i]));
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
//                    const uint8_t* data = tensor.GetTensorData<uint8_t>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(static_cast<unsigned int>(data[i]));
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
//                    const int32_t* data = tensor.GetTensorData<int32_t>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(data[i]);
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
//                    const int64_t* data = tensor.GetTensorData<int64_t>();
//                    if (data) {
//                        std::string values = "First " + std::to_string(elements_to_print) + " values: [";
//                        for (size_t i = 0; i < elements_to_print; ++i) {
//                            if (i > 0) values += ", ";
//                            values += std::to_string(data[i]);
//                        }
//                        values += "]";
//                        LOGI("%s", values.c_str());
//                    }
//                    break;
//                }
//                default:
//                    LOGI("Cannot print values for unsupported data type");
//                    break;
//            }
//
//        } catch (const std::exception& e) {
//            LOGE("Error getting tensor info: %s", e.what());
//        }
//    }

} // utils
