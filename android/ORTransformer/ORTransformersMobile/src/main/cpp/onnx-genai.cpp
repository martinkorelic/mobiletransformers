//
// Deprecated ONNX GenAI functions, due to incompatibility with ORTransformersMobile functionalities
//
// Created by martinkorelic on 20. 07. 25.
//

//#include "onnxruntime-genai/ort_genai.h"
//#include "onnxruntime-genai/ort_genai_c.h"
//

//    std::string genAiInferenceStep(GenAISessionCache* session_cache) {
//
//        session_cache->generator->ComputeLogits();
//        session_cache->generator->GenerateNextToken();
//
//        const auto num_tokens = session_cache->generator->GetSequenceCount(0);
//        const auto new_token = session_cache->generator->GetSequenceData(0)[num_tokens - 1];
//        return session_cache->tokenizer_stream->Decode(new_token);
//    }

//extern "C" JNIEXPORT void JNICALL
//Java_com_martinkorelic_ortmobile_ORTGenAINative_releaseWeightSession(
//        JNIEnv *env, jobject,
//jlong weight_session) {
//ReleaseWeightSession(weight_session);
//}

//
//extern "C" JNIEXPORT jlong JNICALL
///**
// * Caches the trainable layer weights from training session for later use. Releases the training session.
// *
// * @param env
// * @param inference_model_path
// * @return
// */
//Java_com_martinkorelic_ortmobile_ORTGenAINative_cacheSessionWeights(
//        JNIEnv *env, jobject /* this */,
//        jlong train_session,
//        jobjectArray requires_grad
//) {
//
//    auto *train_session_cache = reinterpret_cast<TrainingSessionCache *>(train_session);
//
//    // Get the size of the input array
//    jsize arrayLength = env->GetArrayLength(requires_grad);
//
//    std::vector<std::string> requires_grad_names;
//    for (jsize i = 0; i < arrayLength; ++i) {
//        auto jstr = (jstring) (env->GetObjectArrayElement(requires_grad, i));
//        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
//        requires_grad_names.emplace_back(cstr);
//
//        // Release the string
//        env->ReleaseStringUTFChars(jstr, cstr);
//        env->DeleteLocalRef(jstr);
//    }
//
//    std::unique_ptr<WeightSessionCache> weight_session_cache = std::make_unique<WeightSessionCache>();
//
//    // Load the extracted weights into the inference session
//    LoadWeightsToMemory(weight_session_cache, train_session_cache->checkpoint_state, requires_grad_names);
//
//    // Release the current training session, save the checkpoints
//    ReleaseTrainingSession(train_session, true);
//    train_session_cache = nullptr;
//
//    return reinterpret_cast<long>(weight_session_cache.release());
//}

//extern "C" JNIEXPORT void JNICALL
//Java_com_martinkorelic_ortmobile_ORTGenAINative_releaseGenAISession(
//        JNIEnv *env, jobject,
//jlong genai_session) {
//ReleaseGenAISession(genai_session);
//}

//void ReleaseGenAISession(jlong session) {
//    auto *session_cache = reinterpret_cast<GenAISessionCache *>(session);
//    //OgaDestroyGenerator(session_cache->generator.get());
//    //OgaDestroyGeneratorParams(session_cache->generatorParams.get());
//    //OgaDestroyModel(session_cache->model.get());
//    //OgaDestroyTokenizer(session_cache->tokenizer.get());
//    //OgaDestroyTokenizerStream(session_cache->tokenizer_stream.get());
//    delete session_cache;
//    session_cache = nullptr;
//}

//extern "C" JNIEXPORT jlong JNICALL
///**
// * Creates the GenAI session from the cached weights.
// *
// * @param env
// * @param inference_model_path
// * @return
// */
//Java_com_martinkorelic_ortmobile_ORTGenAINative_createGenAISession(
//        JNIEnv *env, jobject /* this */,
//        jlong weight_cache,
//        jstring genai_path
//) {
//
//    auto *weight_session_cache = reinterpret_cast<WeightSessionCache *>(weight_cache);
//
//    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loading grad weight inputs...");
//
//    std::unique_ptr<GenAISessionCache> genai_session_cache = std::make_unique<GenAISessionCache>(
//            weight_session_cache,
//            utils::JString2String(env, genai_path)
//    );
//    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Preloaded grad weight inputs.");
//
//    return reinterpret_cast<long>(genai_session_cache.release());
//}

//extern "C"
//JNIEXPORT void JNICALL
///**
// * Initializes the GenAI inference with a new prompt.
// * Uses KV caching and generation configuration provided from the file.
// *
// * @param env
// * @param thiz
// * @param genai_session - GenAI cached session
// * @param prompt - New prompt
// */
//Java_com_martinkorelic_ortmobile_ORTGenAINative_initializeGenAIInference(JNIEnv *env, jobject thiz,
//        jlong genai_session, jstring prompt) {
//auto *session_cache = reinterpret_cast<GenAISessionCache *>(genai_session);
//
//auto sequences = OgaSequences::Create();
//session_cache->tokenizer->Encode(utils::JString2String(env, prompt).c_str(), *sequences);
//session_cache->generatorParams->SetInputSequences(*sequences);
//session_cache->generator = OgaGenerator::Create(*session_cache->model, *session_cache->generatorParams);
//}
//
//
//extern "C"
//JNIEXPORT jstring JNICALL
///**
// * Performs the inference step using the exported model for GenAI inference.
// *
// * @param env
// * @param thiz
// * @param session
// * @param input_ids
// * @param attention_mask
// * @param position_ids
// * @param sequence_length
// * @param vocab_size
// *
// * @return Next token string
// */
//Java_com_martinkorelic_ortmobile_ORTGenAINative_performGenAIInferenceStep(JNIEnv *env, jobject thiz,
//jlong genai_session) {
//auto *session_cache = reinterpret_cast<GenAISessionCache *>(genai_session);
//
//if (session_cache->generator->IsDone()) {
//// TODO : Release generator after each session?
////OgaDestroyGenerator(session_cache->generator.get());
//return env->NewStringUTF("[STOP]");
//}
//
//auto next_token = inference::genAiInferenceStep(session_cache);
//
//return env->NewStringUTF(next_token.c_str());
//}


//struct GenAISessionCache {
//    std::unique_ptr<OgaModel> model;
//    std::unique_ptr<OgaGenerator> generator;
//    std::unique_ptr<OgaGeneratorParams> generatorParams;
//    std::unique_ptr<OgaTokenizer> tokenizer;
//    std::unique_ptr<OgaTokenizerStream> tokenizer_stream;
//
//    GenAISessionCache(WeightSessionCache *weight_cache,
//                      const std::string &genai_folder_path) {
//        model = OgaModel::Create(genai_folder_path.c_str());
//        generatorParams = OgaGeneratorParams::Create(*model);
//
//        std::string text = "Hello, this is a message for the world. How is your day?";
//        auto sequences = OgaSequences::Create();
//
//        tokenizer = std::unique_ptr<OgaTokenizer>(OgaTokenizer::Create(*model));
//        tokenizer_stream = std::unique_ptr<OgaTokenizerStream>(OgaTokenizerStream::Create(*tokenizer));
//
//        tokenizer->Encode(text.c_str(), *sequences);
//        generatorParams->SetInputSequences(*sequences);
//
//        generator = OgaGenerator::Create(*model, *generatorParams);
//    }
//};
