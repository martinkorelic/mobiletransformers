//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_SESSION_CACHE_H
#define ORT_PERSONALIZE_SESSION_CACHE_H

#include "onnxruntime_training_cxx_api.h"

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
 * This avoids using export model for inference function. Release after the weights have been transferred.
 */
struct WeightsSessionCache {
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

#endif //ORT_PERSONALIZE_SESSION_CACHE_H