//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_SESSION_CACHE_H
#define ORT_PERSONALIZE_SESSION_CACHE_H

#include "onnxruntime_training_cxx_api.h"
#include "onnxruntime-genai/ort_genai.h"
#include "onnxruntime-genai/ort_genai_c.h"
#include <android/log.h>

struct ArtifactPaths {
    std::string checkpoint_path;
    std::string training_model_path;
    std::string eval_model_path;
    std::string optimizer_model_path;
    std::string cache_dir_path;
    std::string inference_model_path;

    ArtifactPaths(const std::string &checkpoint_path, const std::string &training_model_path,
                  const std::string &eval_model_path, const std::string &optimizer_model_path,
                  const std::string& cache_dir_path) :
            checkpoint_path(checkpoint_path), training_model_path(training_model_path),
            eval_model_path(eval_model_path), optimizer_model_path(optimizer_model_path),
            cache_dir_path(cache_dir_path), inference_model_path(cache_dir_path + "/inference.onnx") {}
};

/**
 * Caches the current trainable weights, which are then ready to be transferred into the inference session.
 * This is useful when we are loading the weights which have been changed into the custom inference model.
 * This avoids using export model for inference function. Release memory after the weights have been transferred.
 */
struct WeightSessionCache {
    // TODO: Change type
    std::unordered_map<std::string, Ort::Value> weights;
};

/**
 * Caches the current inference session variables. This should be released if a training session wants to begin.
 */
struct InferenceSessionCache {
    Ort::Env ort_env;
    Ort::Session* inference_session;
    Ort::SessionOptions session_options;
    std::string inference_model_path;

    InferenceSessionCache(const std::string& inference_model_path) :
            ort_env(ORT_LOGGING_LEVEL_WARNING, "ORTInference"),
            session_options(),
            inference_model_path(inference_model_path),
            inference_session(nullptr) {}
    ;
};

/**
 * Caches the current training session variables. This should be released after the training session is complete.
 */
struct TrainingSessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::CheckpointState checkpoint_state;
    Ort::SessionOptions session_options;
    Ort::TrainingSession training_session;
    std::vector<std::string> requires_grad;

    TrainingSessionCache(const std::string &checkpoint_path, const std::string &training_model_path,
                 const std::string &eval_model_path, const std::string &optimizer_model_path,
                 const std::string& cache_dir_path) :
            artifact_paths(checkpoint_path, training_model_path, eval_model_path, optimizer_model_path, cache_dir_path),
            ort_env(ORT_LOGGING_LEVEL_WARNING, "ORTTraining"), session_options(),
            checkpoint_state(Ort::CheckpointState::LoadCheckpoint(artifact_paths.checkpoint_path.c_str())),
            training_session(ort_env, session_options, checkpoint_state, artifact_paths.training_model_path.c_str(),
                             artifact_paths.eval_model_path.c_str(), artifact_paths.optimizer_model_path.c_str()) {}

};

struct GenAISessionCache {
    std::unique_ptr<OgaModel> model;
    std::unique_ptr<OgaGenerator> generator;
    std::unique_ptr<OgaGeneratorParams> generatorParams;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizer_stream;

    GenAISessionCache(WeightSessionCache *weight_cache,
                      const std::string &genai_folder_path) {
        model = OgaModel::Create(genai_folder_path.c_str());
        generatorParams = OgaGeneratorParams::Create(*model);

        for (const auto& weight_pair : weight_cache->weights) {
            const std::string& layer_name = weight_pair.first;
            const Ort::Value& weight_value = weight_pair.second;
            __android_log_print(ANDROID_LOG_DEBUG, "SessionCache", "Loading %s into GenAI model...", layer_name.c_str());

            Ort::TensorTypeAndShapeInfo tensor_info = weight_value.GetTensorTypeAndShapeInfo();
            auto tensor_shape = tensor_info.GetShape();
            auto tensor_type = tensor_info.GetElementType();
            size_t total_elements = tensor_info.GetElementCount();
            auto element_type = static_cast<OgaElementType>(tensor_type);

            void* raw_data = const_cast<void *>(weight_value.GetTensorRawData());
//            __android_log_print(ANDROID_LOG_INFO, "LOG_TAG", "Tensor shape: ");
//            std::string shape_str = "";
//            for (auto dim : tensor_shape) {
//                shape_str += std::to_string(dim) + " ";
//            }
//            __android_log_print(ANDROID_LOG_INFO, "LOG_TAG", "%s", shape_str.c_str());
            // Print some elements
//            __android_log_print(ANDROID_LOG_INFO, "LOG_TAG", "Tensor elements (first 10 or total elements if fewer): ");
//            size_t num_elements_to_print = std::min<size_t>(10, total_elements); // Print first 10 elements or total if fewer
//            float* data = static_cast<float*>(raw_data);
//            std::string elements_str = "";
//            for (size_t i = 0; i < num_elements_to_print; ++i) {
//                elements_str += std::to_string(data[i]) + " ";
//            }
//            __android_log_print(ANDROID_LOG_INFO, "LOG_TAG", "%s", elements_str.c_str());

            // Create OgaTensor from the buffer
            OgaTensor* oga_tensor = nullptr;
            OgaCreateTensorFromBuffer(raw_data, tensor_shape.data(), tensor_shape.size(), element_type, &oga_tensor);
            // Add the buffer as a model input
            OgaGeneratorParamsSetModelInput(generatorParams.get(), layer_name.c_str(), oga_tensor);
        }

        std::string text = "Hello, this is a message for the world. How is your day?";
        auto sequences = OgaSequences::Create();

        tokenizer = std::unique_ptr<OgaTokenizer>(OgaTokenizer::Create(*model));
        tokenizer_stream = std::unique_ptr<OgaTokenizerStream>(OgaTokenizerStream::Create(*tokenizer));

        tokenizer->Encode(text.c_str(), *sequences);
        generatorParams->SetInputSequences(*sequences);

        generator = OgaGenerator::Create(*model, *generatorParams);

//        while (!generator->IsDone()) {
//            generator->ComputeLogits();
//            generator->GenerateNextToken();
//
//            const auto num_tokens = generator->GetSequenceCount(0);
//            const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
//            __android_log_print(ANDROID_LOG_INFO, "LOG_TAG", "%s", tokenizer_stream->Decode(new_token));
//        }
    }
};



#endif //ORT_PERSONALIZE_SESSION_CACHE_H