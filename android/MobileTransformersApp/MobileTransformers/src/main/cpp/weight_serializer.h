//
// Created by martin on 17. 07. 25.
//

#ifndef MOBILETRANSFORMERS_WEIGHT_SERIALIZER_H
#define MOBILETRANSFORMERS_WEIGHT_SERIALIZER_H

#include <onnxruntime_training_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include "proto/onnx.pb.h"

class OrtValueSerializer {
public:
    static bool save_tensor(const std::string& filepath, const Ort::Value& tensor, const std::string& tensor_name = "");
    static std::unique_ptr<Ort::Value> load_tensor(const std::string& filepath);
    static bool is_valid_tensor_file(const std::string& filepath);

    static std::pair<Ort::Value, void*>  tensorproto_to_ortvalue_with_allocator(
            const onnx::TensorProto& tensor,
            Ort::MemoryInfo& memory_info_,
            Ort::AllocatorWithDefaultOptions& allocator_);

private:
    static onnx::TensorProto ortvalue_to_tensorproto(const Ort::Value& value, const std::string& name);
    static std::unique_ptr<Ort::Value> tensorproto_to_ortvalue(const onnx::TensorProto& tensor);

    template<typename T>
    static std::vector<T> ortvalue_to_vector(const Ort::Value& value) {
        auto tensor_info = value.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        size_t element_count = 1;
        for (auto dim : shape) {
            element_count *= dim;
        }

        const T* data = value.GetTensorData<T>();
        return std::vector<T>(data, data + element_count);
    }
};

#endif //MOBILETRANSFORMERS_WEIGHT_SERIALIZER_H
