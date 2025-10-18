//
// Created by martin on 17. 07. 25.
//


#include <fstream>
#include <iostream>
#include <cstring>
#include "weight_serializer.h"

// Implementation

bool OrtValueSerializer::save_tensor(const std::string& filepath, const Ort::Value& tensor, const std::string& tensor_name) {
    try {
        // Convert OrtValue to TensorProto
        onnx::TensorProto tensor_proto = ortvalue_to_tensorproto(tensor, tensor_name);

        // Write to file
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        bool success = tensor_proto.SerializeToOstream(&file);
        file.close();
        return success;

    } catch (const std::exception& e) {
        return false;
    }
}

std::unique_ptr<Ort::Value> OrtValueSerializer::load_tensor(const std::string& filepath) {
    try {
        // Read from file
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        onnx::TensorProto tensor_proto;
        if (!tensor_proto.ParseFromIstream(&file)) {
            file.close();
            throw std::runtime_error("Failed to parse TensorProto from file: " + filepath);
        }
        file.close();

        // Convert TensorProto to OrtValue
        return tensorproto_to_ortvalue(tensor_proto);

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading tensor: " + std::string(e.what()));
    }
}

bool OrtValueSerializer::is_valid_tensor_file(const std::string& filepath) {
    try {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        onnx::TensorProto tensor_proto;
        bool valid = tensor_proto.ParseFromIstream(&file);
        file.close();
        return valid;

    } catch (const std::exception& e) {
        return false;
    }
}

onnx::TensorProto OrtValueSerializer::ortvalue_to_tensorproto(const Ort::Value& value, const std::string& name) {
    onnx::TensorProto tensor_proto;

    // Set name if provided
    if (!name.empty()) {
        tensor_proto.set_name(name);
    }

    // Get shape and set dimensions
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    for (auto dim : shape) {
        tensor_proto.add_dims(dim);
    }

    // Get element type
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();

    // Set data type and copy data
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            tensor_proto.set_data_type(onnx::TensorProto::FLOAT);
            auto data_vec = ortvalue_to_vector<float>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(float));
            break;
        }

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            tensor_proto.set_data_type(onnx::TensorProto::INT8);
            auto data_vec = ortvalue_to_vector<int8_t>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(int8_t));
            break;
        }

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            tensor_proto.set_data_type(onnx::TensorProto::UINT8);
            auto data_vec = ortvalue_to_vector<uint8_t>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(uint8_t));
            break;
        }

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            tensor_proto.set_data_type(onnx::TensorProto::INT32);
            auto data_vec = ortvalue_to_vector<int32_t>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(int32_t));
            break;
        }

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            tensor_proto.set_data_type(onnx::TensorProto::INT64);
            auto data_vec = ortvalue_to_vector<int64_t>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(int64_t));
            break;
        }

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            tensor_proto.set_data_type(onnx::TensorProto::DOUBLE);
            auto data_vec = ortvalue_to_vector<double>(value);
            tensor_proto.set_raw_data(data_vec.data(), data_vec.size() * sizeof(double));
            break;
        }

        default:
            throw std::runtime_error("Unsupported tensor element type: " + std::to_string(element_type));
    }

    return tensor_proto;
}


std::unique_ptr<Ort::Value> OrtValueSerializer::tensorproto_to_ortvalue(const onnx::TensorProto& tensor) {
    // Get shape
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(tensor.dims(i));
    }

    // Calculate total size
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    // Create memory info for CPU
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Handle different data types
    switch (tensor.data_type()) {
        case onnx::TensorProto::FLOAT: {
            // Create vector to hold the data (RAII managed)
            std::vector<float> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(float)) {
                    throw std::runtime_error("Raw data size mismatch for FLOAT tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.float_data_size() != total_size) {
                    throw std::runtime_error("Float data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = tensor.float_data(i);
                }
            }

            // Create tensor with data copy - ORT will manage memory internally
            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<float>(memory_info, data_vec.data(), total_size,
                                                    shape.data(), shape.size())
            );
        }

        case onnx::TensorProto::INT8: {
            std::vector<int8_t> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(int8_t)) {
                    throw std::runtime_error("Raw data size mismatch for INT8 tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    throw std::runtime_error("Int8 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = static_cast<int8_t>(tensor.int32_data(i));
                }
            }

            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<int8_t>(memory_info, data_vec.data(), total_size,
                                                     shape.data(), shape.size())
            );
        }

        case onnx::TensorProto::UINT8: {
            std::vector<uint8_t> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(uint8_t)) {
                    throw std::runtime_error("Raw data size mismatch for UINT8 tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    throw std::runtime_error("Uint8 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = static_cast<uint8_t>(tensor.int32_data(i));
                }
            }

            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<uint8_t>(memory_info, data_vec.data(), total_size,
                                                      shape.data(), shape.size())
            );
        }

        case onnx::TensorProto::INT32: {
            std::vector<int32_t> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(int32_t)) {
                    throw std::runtime_error("Raw data size mismatch for INT32 tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    throw std::runtime_error("Int32 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = tensor.int32_data(i);
                }
            }

            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<int32_t>(memory_info, data_vec.data(), total_size,
                                                      shape.data(), shape.size())
            );
        }

        case onnx::TensorProto::INT64: {
            std::vector<int64_t> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(int64_t)) {
                    throw std::runtime_error("Raw data size mismatch for INT64 tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.int64_data_size() != total_size) {
                    throw std::runtime_error("Int64 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = tensor.int64_data(i);
                }
            }

            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<int64_t>(memory_info, data_vec.data(), total_size,
                                                      shape.data(), shape.size())
            );
        }

        case onnx::TensorProto::DOUBLE: {
            std::vector<double> data_vec(total_size);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != total_size * sizeof(double)) {
                    throw std::runtime_error("Raw data size mismatch for DOUBLE tensor");
                }
                std::memcpy(data_vec.data(), raw_data.data(), raw_data.size());
            } else {
                if (tensor.double_data_size() != total_size) {
                    throw std::runtime_error("Double data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data_vec[i] = tensor.double_data(i);
                }
            }

            return std::make_unique<Ort::Value>(
                    Ort::Value::CreateTensor<double>(memory_info, data_vec.data(), total_size,
                                                     shape.data(), shape.size())
            );
        }

        default:
            throw std::runtime_error("Unsupported tensor data type: " + std::to_string(tensor.data_type()));
    }
}


std::pair<Ort::Value, void*> OrtValueSerializer::tensorproto_to_ortvalue_with_allocator(
        const onnx::TensorProto& tensor,
        Ort::MemoryInfo& memory_info_,
        Ort::AllocatorWithDefaultOptions& allocator_) {

    // Get shape
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(tensor.dims(i));
    }

    // Calculate total size
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    void* buffer_ptr = nullptr;
    Ort::Value ort_value{nullptr};

    // Handle different data types
    switch (tensor.data_type()) {
        case onnx::TensorProto::FLOAT: {
            size_t data_size = total_size * sizeof(float);
            buffer_ptr = allocator_.Alloc(data_size);
            float* data = static_cast<float*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for FLOAT tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.float_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Float data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = tensor.float_data(i);
                }
            }

            // Create tensor with user-managed data
            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create FLOAT tensor: " + std::string(e.what()));
            }
            break;
        }

        case onnx::TensorProto::INT8: {
            size_t data_size = total_size * sizeof(int8_t);
            buffer_ptr = allocator_.Alloc(data_size);
            int8_t* data = static_cast<int8_t*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for INT8 tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Int8 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = static_cast<int8_t>(tensor.int32_data(i));
                }
            }

            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create INT8 tensor: " + std::string(e.what()));
            }
            break;
        }

        case onnx::TensorProto::UINT8: {
            size_t data_size = total_size * sizeof(uint8_t);
            buffer_ptr = allocator_.Alloc(data_size);
            uint8_t* data = static_cast<uint8_t*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for UINT8 tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Uint8 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = static_cast<uint8_t>(tensor.int32_data(i));
                }
            }

            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create UINT8 tensor: " + std::string(e.what()));
            }
            break;
        }

        case onnx::TensorProto::INT32: {
            size_t data_size = total_size * sizeof(int32_t);
            buffer_ptr = allocator_.Alloc(data_size);
            int32_t* data = static_cast<int32_t*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for INT32 tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.int32_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Int32 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = tensor.int32_data(i);
                }
            }

            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create INT32 tensor: " + std::string(e.what()));
            }
            break;
        }

        case onnx::TensorProto::INT64: {
            size_t data_size = total_size * sizeof(int64_t);
            buffer_ptr = allocator_.Alloc(data_size);
            int64_t* data = static_cast<int64_t*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for INT64 tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.int64_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Int64 data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = tensor.int64_data(i);
                }
            }

            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create INT64 tensor: " + std::string(e.what()));
            }
            break;
        }

        case onnx::TensorProto::DOUBLE: {
            size_t data_size = total_size * sizeof(double);
            buffer_ptr = allocator_.Alloc(data_size);
            double* data = static_cast<double*>(buffer_ptr);

            if (tensor.has_raw_data()) {
                const std::string& raw_data = tensor.raw_data();
                if (raw_data.size() != data_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Raw data size mismatch for DOUBLE tensor");
                }
                std::memcpy(data, raw_data.data(), raw_data.size());
            } else {
                if (tensor.double_data_size() != total_size) {
                    allocator_.Free(buffer_ptr);
                    throw std::runtime_error("Double data size mismatch");
                }
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = tensor.double_data(i);
                }
            }

            try {
                ort_value = Ort::Value::CreateTensor(
                        memory_info_,
                        buffer_ptr,
                        data_size,
                        shape.data(),
                        shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
            } catch (const std::exception& e) {
                allocator_.Free(buffer_ptr);
                throw std::runtime_error("Failed to create DOUBLE tensor: " + std::string(e.what()));
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported tensor data type: " + std::to_string(tensor.data_type()));
    }

    return std::make_pair(std::move(ort_value), buffer_ptr);
}