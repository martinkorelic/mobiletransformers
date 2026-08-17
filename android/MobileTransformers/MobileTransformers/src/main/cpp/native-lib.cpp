//
// Created by martinkorelic on 31/08/2024
//

#include <jni.h>
#include <string>
#include "onnxruntime/onnxruntime_training_cxx_api.h"
#include "inference.h"
#include "tokenization.h"
#include "train.h"
#include "utils.h"
#include "sampling.h"
#include "mem_probe.h"
#include "logits_metrics.h"
#include <android/log.h>

#define LOG_TAG "MobileTransformers"

extern "C" JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_app_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

// #12 (Gate 0.2): expose the native RSS sampler so an instrumented test can build the four-point
// base/merged x copy/mmap table. Debug.getPss() would measure the JVM's view; the weights are mapped
// by native code, so VmRSS from /proc/self/status is the number the gate is specified against.
extern "C" JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_runtime_MemoryProbe_nativeCurrentRssKb(
        JNIEnv* /* env */,
        jobject /* this */) {
    return static_cast<jlong>(memprobe::read_rss_kb());
}

// Whether the zero-copy weight load is currently switched on (env var or `debug.mtf.mmap_weights`).
// The RSS harness asserts it actually flipped rather than trusting setprop to have taken effect.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_martinkorelic_mobiletransformers_runtime_MemoryProbe_nativeMmapWeightsEnabled(
        JNIEnv* /* env */,
        jobject /* this */) {
    return memprobe::mmap_weights_enabled() ? JNI_TRUE : JNI_FALSE;
}

void ReleaseTrainingSession(jlong session, jboolean saveCheckpoint) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    if (saveCheckpoint) {
        // Include optimizer state?
        Ort::CheckpointState::SaveCheckpoint(session_cache->checkpoint_state, session_cache->artifact_paths.checkpoint_path, true);
    }

    delete session_cache;
    session_cache = nullptr;
}

void ReleaseWeightSession(jlong session) {
    auto *session_cache = reinterpret_cast<WeightSessionCache *>(session);

    delete session_cache;
    session_cache = nullptr;
}

void ReleaseTokenizerSession(jlong session) {
    auto *session_cache = reinterpret_cast<TokenizerSessionCache *>(session);

    delete session_cache;
    session_cache = nullptr;
}

extern "C"
JNIEXPORT float JNICALL
/**
 * Performs the training step with the gradient update and optimizer step.
 *
 * Inputs are bound BY NAME inside `training::train_step`, from the names the training graph itself
 * declares — so the same entry point serves a decoder (which asks for `position_ids`) and an encoder
 * classifier (which asks for `token_type_ids` and per-sequence `labels`). `position_ids` and
 * `token_type_ids` are synthesized there, and only if the graph asks for them.
 *
 * The label RANK is derived from how many label elements the caller actually supplied, which is why
 * the array length is read here and passed down.
 *
 * @param env
 * @param session
 * @param input_ids
 * @param batch_size
 * @param sequence_length
 *
 * @return Loss value
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_performTraining(
        JNIEnv *env, jobject /* this */,
        jlong session,
        jlongArray input_ids, jlongArray labels, jlongArray attention_mask, jint batch_size, jint sequence_length) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    // Get the input_ids array from the Java environment
    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* label_elements = env->GetLongArrayElements(labels, nullptr);
    jlong* attention_elements = env->GetLongArrayElements(attention_mask, nullptr);

    // What the caller actually supplied — the ground truth for the label rank, rather than a
    // declared constant that can drift away from the data.
    const jsize labels_count = env->GetArrayLength(labels);

    float loss = 0.0f;
    std::string error;
    try {
        // Update the model parameters using this batch of inputs.
        loss = training::train_step(session_cache, input_ids_elements, attention_elements,
                                    label_elements, batch_size, sequence_length, labels_count);
    } catch (const std::exception& e) {
        error = e.what();
    }

    env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
    env->ReleaseLongArrayElements(labels, label_elements, JNI_ABORT);
    env->ReleaseLongArrayElements(attention_mask, attention_elements, JNI_ABORT);

    if (!error.empty()) {
        // A C++ exception crossing JNI calls std::terminate and kills the WHOLE instrumentation run,
        // so later tests never report. Convert it, after releasing the arrays.
        jclass runtime_exception = env->FindClass("java/lang/RuntimeException");
        if (runtime_exception != nullptr) {
            env->ThrowNew(runtime_exception, (std::string("training step failed: ") + error).c_str());
        }
        return 0.0f;
    }

    return loss;
}

extern "C"
JNIEXPORT jlong JNICALL
/**
 * Creates the training session from the given artifact paths.
 *
 * @param env
 * @param thiz
 * @param checkpoint_path
 * @param train_model_path
 * @param eval_model_path
 * @param optimizer_model_path
 * @param cache_dir_path
 * @param requires_grad
 *
 * @return Training session native model handle.
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_createTrainingSession(JNIEnv *env, jobject thiz,
                                                           jstring checkpoint_path,
                                                           jstring train_model_path,
                                                           jstring eval_model_path,
                                                           jstring optimizer_model_path,
                                                           jstring cache_dir_path,
                                                           jobjectArray requires_grad,
                                                           jstring memory_config_id,
                                                           jstring core_config_id,
                                                           jstring execution_provider,
                                                           jboolean enable_profiling) {

    // Get the size of the input array
    jsize arrayLength = env->GetArrayLength(requires_grad);

    std::unique_ptr<TrainingSessionCache> session_cache = std::make_unique<TrainingSessionCache>(
            utils::JString2String(env, checkpoint_path),
            utils::JString2String(env, train_model_path),
            utils::JString2String(env, eval_model_path),
            utils::JString2String(env, optimizer_model_path),
            utils::JString2String(env, cache_dir_path),
            utils::JString2String(env, memory_config_id),
            utils::JString2String(env, core_config_id),
            utils::JString2String(env, execution_provider),
            enable_profiling);

    for (jsize i = 0; i < arrayLength; ++i) {
        auto jstr = (jstring) (env->GetObjectArrayElement(requires_grad, i));
        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
        session_cache->requires_grad.emplace_back(cstr);

        // Release the string
        env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }

    return reinterpret_cast<long>(session_cache.release());
}



extern "C" JNIEXPORT jlong JNICALL
/**
 * Creates normal inference session from the given inference model path.
 * This inference is a custom made inference which is ready to be used for generation with KV caching.
 *
 * If load_merged_weights is enabled:
 * 1. Transfer the weights to the inference session options with the weights from the weights that were merged and saved.
 * 2. Load the inference model
 * 3. The model is ready for inference with the merged weights.
 *
 * @param env
 * @param inference_model_path
 * @param inference_model_name
 * @param load_merged_weights - Whether to load merged weights as flat per-tensor external initializers
 *        from ".../inference/" keyed by weight_handoff_map.json (#23; the /merged subdir is retired)
 * @return
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_createInferenceSession(
        JNIEnv *env, jobject /* this */,
        jstring inference_model_path,
        jstring inference_model_name,
        jstring cache_dir_path,
        jboolean load_merged_weights,
        jstring core_config_id,
        jstring memory_config_id,
        jstring execution_provider,
        jboolean enable_profiling
        ) {

    // #23: session construction FAILS CLOSED when merged weights were requested but could not be
    // loaded (missing/mis-shaped tensor, or ORT rejecting the external initializers). Returning 0
    // rather than a session built from the frozen base weights is the whole point — a silent downgrade
    // yields an untrained model that looks healthy. Kotlin turns the 0 into MissingArtifactException.
    // A C++ exception must not cross the JNI boundary, so it is converted here.
    try {
        std::unique_ptr<InferenceSessionCache> session_cache = std::make_unique<InferenceSessionCache>(
                utils::JString2String(env, inference_model_path),
                utils::JString2String(env, inference_model_name),
                utils::JString2String(env, cache_dir_path),
                utils::JString2String(env, memory_config_id),
                utils::JString2String(env, core_config_id),
                utils::JString2String(env, execution_provider),
                load_merged_weights,
                enable_profiling);

        session_cache->initializeKVCache(1);

        return reinterpret_cast<long>(session_cache.release());
    } catch (const std::exception& e) {
        LOGE("createInferenceSession failed: %s", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
/**
 * Deletes the current inference session.
 *
 * @param env
 * @param session
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_releaseInferenceSession(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    // A 0 handle means the session was never created — `createInferenceSession` returns 0 on failure,
    // and `destroySession()` is still reached through the normal `finally`/`release()` path. Without
    // this guard that path dereferenced a null pointer (`SIGSEGV`, fault addr 0x8 — the offset of
    // `inference_session`), which kills the ENTIRE instrumentation run rather than failing one test.
    // That is the same class of hazard as the C++ exception that used to cross JNI and call
    // std::terminate: a recoverable error taking the process with it.
    if (session == 0) {
        return;
    }
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    delete session_cache->inference_session;
    delete session_cache;
    session_cache = nullptr;
}

extern "C"
JNIEXPORT jdoubleArray JNICALL
/**
 * One forward pass, reduced to numbers a test can assert on. A PROBE: it leaves no state behind.
 *
 * Exists because nothing checked post-merge numerical correctness on device. The export pipeline gates
 * a package on `train_inference_parity.py` (same tokens, both graphs, one bounded delta), but the
 * device only ever hashed the merged `.bin` files — `TrainMergeGenerateTest` says so itself, and a
 * merge that wrote plausible bytes with corrupted values passed every gate the project had.
 *
 * `performInferenceStep` cannot serve this: it samples internally and returns only a token id, so the
 * logits never reach Kotlin. Rather than marshal a vocab-sized float array across JNI on every call,
 * this returns the reduction.
 *
 * @return `[argmax, maxLogit, sum, sumOfSquares, causalCrossEntropyNats]` for a single prefill pass.
 *   The cross-entropy uses the SAME causal shift as the host gate, so the two numbers are comparable;
 *   computing it under a different convention would make the measurement decorative.
 *
 * The KV cache is reset afterwards because `generateWithKVCache` updates it — a measurement that
 * silently advanced the conversation would corrupt whatever ran next, which is exactly the
 * package-mutation hazard the device suite already has to work around.
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_nativeInferenceMetrics(
        JNIEnv *env, jobject /* this */,
        jlong session,
        jlongArray input_ids,
        jlongArray attention_mask,
        jlongArray position_ids,
        jint batch_size,
        jint sequence_length,
        jint new_token_count,
        jint vocab_size) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    jlong *input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong *attention_mask_elements = env->GetLongArrayElements(attention_mask, nullptr);
    jlong *position_ids_elements = env->GetLongArrayElements(position_ids, nullptr);

    auto release = [&]() {
        env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(attention_mask, attention_mask_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(position_ids, position_ids_elements, JNI_ABORT);
    };

    try {
        float *logits = inference::generateWithKVCache(session_cache,
                                                       input_ids_elements,
                                                       attention_mask_elements,
                                                       position_ids_elements,
                                                       batch_size,
                                                       sequence_length,
                                                       new_token_count);

        // Same clamp as performInferenceStep, and for the same reason: these two also stride by
        // vocab_size to find the row. A measurement taken over the wrong stride is not a measurement.
        const int metric_vocab_size =
                sampling::effectiveVocabSize(vocab_size, session_cache->lastLogitsWidth());

        const auto fp = logits_metrics::fingerprint_last_position(logits, new_token_count, metric_vocab_size);
        const double loss = logits_metrics::causal_cross_entropy(
                logits, input_ids_elements, new_token_count, metric_vocab_size);

        // A probe must not advance the conversation.
        session_cache->initializeKVCache(batch_size);

        release();

        jdouble values[5] = {
                static_cast<jdouble>(fp.argmax),
                fp.max_logit,
                fp.sum,
                fp.sum_of_squares,
                loss,
        };
        jdoubleArray out = env->NewDoubleArray(5);
        if (out != nullptr) {
            env->SetDoubleArrayRegion(out, 0, 5, values);
        }
        return out;
    } catch (const std::exception &e) {
        release();
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "nativeInferenceMetrics failed: %s", e.what());
        jclass runtime_exception = env->FindClass("java/lang/RuntimeException");
        if (runtime_exception != nullptr) {
            env->ThrowNew(runtime_exception,
                          (std::string("inference metrics failed: ") + e.what()).c_str());
        }
        return nullptr;
    }
}

extern "C"
JNIEXPORT jint JNICALL
/**
 * How many tokens the session's KV cache actually holds.
 *
 * The session is the single authority on this. Kotlin previously kept its own counter
 * (`pastAttentionMaskLength = attentionMask.size - 1`) and built the next turn's attention mask from
 * it; the two could drift, and a mask shorter than `cache + new` fails inside ORT on a transformers
 * >= 4.57 graph with a message naming neither. Reading it back removes the second source of truth.
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_nativePastSequenceLength(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    if (session == 0) {
        return 0;
    }
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);
    return static_cast<jint>(session_cache->pastSequenceLength());
}

extern "C"
JNIEXPORT void JNICALL
/**
 * Drops the KV cache back to an empty (zero-length past) state for a new conversation.
 *
 * Re-initialises rather than merely clearing: `generateWithKVCache` binds one value per graph input,
 * so an EMPTY `past_key_values` vector would under-bind and read past the end. `initializeKVCache`
 * recreates the zero-length tensors a `*-with-past` graph expects on a first pass.
 *
 * Before this existed, `resetConversation()` reset only Kotlin's counter and history — the native
 * cache survived, so "reset" left the two halves disagreeing about how many tokens were cached.
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_nativeResetKvCache(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    if (session == 0) {
        return;
    }
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);
    try {
        session_cache->initializeKVCache(1);
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "nativeResetKvCache failed: %s", e.what());
        jclass runtime_exception = env->FindClass("java/lang/RuntimeException");
        if (runtime_exception != nullptr) {
            env->ThrowNew(runtime_exception, (std::string("kv cache reset failed: ") + e.what()).c_str());
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_releaseTrainingSession(
        JNIEnv *env, jobject,
        jlong session, jboolean saveCheckpoint) {
    ReleaseTrainingSession(session, saveCheckpoint);
}

extern "C"
JNIEXPORT void JNICALL
/**
 * Utility function, example of inspecting weights in the model from the checkpoint state.
 * */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_inspectWeights(JNIEnv *env, jobject thiz,
                                                            jlong session, jstring layer) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    Ort::Value parameter = session_cache->checkpoint_state.GetParameter(utils::JString2String(env, layer));

    auto type_info = parameter.GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // Get tensor dimensions
    std::vector<int64_t> dimensions = tensor_info.GetShape();
    // Get the data type
    ONNXTensorElementDataType dtype = tensor_info.GetElementType();

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Type: type=%u", dtype);
    for (auto dim: dimensions) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Dimension: type=%ld", dim);
    }
}


extern "C"
JNIEXPORT jbyteArray JNICALL
/**
 * Raw little-endian bytes of ONE checkpoint parameter, or null when the checkpoint has no such name.
 *
 * ## Why this, and not `exportTrainableTensors(session, handoffMapPath) -> ByteArray`
 *
 * The federated plan names that wider signature — the whole record built in C++. This deliberately
 * does less, for one reason: the record's byte layout is **already owned** by
 * `federated/AdapterTensorCodec.kt`, which is pinned byte-for-byte against
 * `tests/federated/fixtures/federated_record.golden.bin`. Building the record here would be a SECOND
 * implementation of the exact format that golden exists to keep from drifting, and it would need
 * `handoff_io.h` extended to model the adapter fields plus a JSON writer reproducing Python's
 * `sort_keys` separators. Two implementations of one wire format is the failure this project keeps
 * paying for.
 *
 * So C++ moves tensor bytes and Kotlin owns the format: `AdapterTensorCodec.build(payloadFor = ...)`
 * composes them, and the order/naming/dtype all still come from `weight_handoff_map.json`.
 *
 * Returns **null** rather than throwing on a missing name: the caller (the codec) already fails closed
 * with a message naming the tensor and explaining that the package and checkpoint disagree, and that
 * message is better than one from here.
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_nativeExportCheckpointTensor(
        JNIEnv *env, jobject /* this */,
        jlong session, jstring name) {
    if (session == 0) {
        return nullptr;
    }
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);
    const std::string parameter_name = utils::JString2String(env, name);

    try {
        Ort::Value parameter = session_cache->checkpoint_state.GetParameter(parameter_name);
        auto tensor_info = parameter.GetTypeInfo().GetTensorTypeAndShapeInfo();

        const size_t element_count = tensor_info.GetElementCount();
        const ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        size_t element_size;
        switch (dtype) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   element_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: element_size = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  element_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   element_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   element_size = 1; break;
            default:
                // An unsupported dtype must not be silently re-interpreted as bytes — the receiver
                // would decode a plausible-looking tensor of the wrong type.
                __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                    "nativeExportCheckpointTensor: '%s' has unsupported dtype %u",
                                    parameter_name.c_str(), dtype);
                return nullptr;
        }

        const size_t byte_length = element_count * element_size;
        jbyteArray out = env->NewByteArray(static_cast<jsize>(byte_length));
        if (out == nullptr) {
            return nullptr;
        }
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(byte_length),
                                reinterpret_cast<const jbyte *>(parameter.GetTensorRawData()));
        return out;
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                            "nativeExportCheckpointTensor('%s'): %s", parameter_name.c_str(), e.what());
        return nullptr;
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
/**
 * Writes raw little-endian bytes back into ONE checkpoint parameter.
 *
 * The import half of the federated round: the aggregated factors arrive as a record, the Kotlin codec
 * decodes it, and each tensor is written back **by name**. Matching by name rather than by iteration
 * order is not a preference — the Python simulation had exactly that defect, pairing tensors by
 * checkpoint iteration order, which would write one layer's `lora_A` over another's and was caught
 * only "mostly", by differing shapes.
 *
 * Fails (returns false) rather than truncating or padding when the incoming byte count does not match
 * the parameter's own element count and dtype: a size mismatch means the sender and this checkpoint
 * disagree about the adapter geometry, and writing anyway would corrupt training silently.
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_nativeImportCheckpointTensor(
        JNIEnv *env, jobject /* this */,
        jlong session, jstring name, jbyteArray data) {
    if (session == 0 || data == nullptr) {
        return JNI_FALSE;
    }
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);
    const std::string parameter_name = utils::JString2String(env, name);

    try {
        // Read the EXISTING parameter to learn the shape and dtype the checkpoint expects. The record
        // carries a declared shape too, but the checkpoint is the authority on its own storage, and
        // trusting the sender's description would let a malformed record reshape local state.
        Ort::Value existing = session_cache->checkpoint_state.GetParameter(parameter_name);
        auto tensor_info = existing.GetTypeInfo().GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = tensor_info.GetShape();
        const size_t element_count = tensor_info.GetElementCount();
        const ONNXTensorElementDataType dtype = tensor_info.GetElementType();

        if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            // Adapter factors are float32 by construction: the trainable-tensor gate guarantees a
            // declared-trainable tensor is never quantized, so anything else here is a real mismatch.
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "nativeImportCheckpointTensor: '%s' is dtype %u, expected float32",
                                parameter_name.c_str(), dtype);
            return JNI_FALSE;
        }

        const jsize incoming = env->GetArrayLength(data);
        const size_t expected = element_count * sizeof(float);
        if (static_cast<size_t>(incoming) != expected) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "nativeImportCheckpointTensor: '%s' got %d bytes, checkpoint needs %zu",
                                parameter_name.c_str(), incoming, expected);
            return JNI_FALSE;
        }

        std::vector<float> values(element_count);
        env->GetByteArrayRegion(data, 0, incoming, reinterpret_cast<jbyte *>(values.data()));

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value updated = Ort::Value::CreateTensor<float>(
                memory_info, values.data(), element_count, shape.data(), shape.size());
        session_cache->checkpoint_state.UpdateParameter(parameter_name, updated);
        return JNI_TRUE;
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "nativeImportCheckpointTensor('%s'): %s", parameter_name.c_str(), e.what());
        return JNI_FALSE;
    }
}

extern "C"
JNIEXPORT jstring JNICALL
/**
 * Export model for inference from the training session.
 *
 * @param env
 * @param thiz
 * @param session - Current training session native model handle
 * @return
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_exportModelForInference(JNIEnv *env, jobject thiz,
                                                                   jlong session) {
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(session);
    session_cache->training_session.ExportModelForInferencing(session_cache->artifact_paths.inference_model_path, {"logits"});
    return env->NewStringUTF(session_cache->artifact_paths.inference_model_path.c_str());
}


extern "C"
JNIEXPORT jint JNICALL
/**
 * Performs the inference step using the exported model for inference.
 *
 * @param env
 * @param thiz
 * @param session
 * @param input_ids
 * @param attention_mask
 * @param position_ids
 * @param sequence_length
 * @param past_sequence_length
 * @param vocab_size
 *
 * @return Next token id
 */
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_performInferenceStep(JNIEnv *env, jobject thiz,
                                                                  jlong session,
                                                                  jlongArray input_ids,
                                                                  jlongArray attention_mask,
                                                                  jlongArray position_ids,
                                                                  jint batch_size,
                                                                  jint sequence_length,
                                                                  jint past_sequence_length,
                                                                  jint vocab_size
                                                                  ) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* attention_mask_elements = env->GetLongArrayElements(attention_mask, nullptr);
    jlong* position_ids_elements = env->GetLongArrayElements(position_ids, nullptr);

    // A C++ exception must never cross the JNI boundary: an Ort::Exception escaping here calls
    // std::terminate and aborts the whole process, so a recoverable shape/IO error took the app (and,
    // in CI, the entire instrumentation run) down instead of surfacing as a catchable failure. Kotlin's
    // `catch (e: Throwable)` around generate() cannot see a C++ throw — it has to be converted here.
    try {
        // Forward pass
        auto logits = inference::generateWithKVCache(session_cache,
                                       input_ids_elements,
                                       attention_mask_elements,
                                       position_ids_elements,
                                       batch_size,
                                       sequence_length,
                                       past_sequence_length);

        // The graph decides how wide the vocabulary is, not the package's JSON declaration. An
        // over-declared vocab_size both mis-strides the row offset and lets the argmax return an id
        // with no embedding row, which the NEXT step reports as an out-of-bounds Gather far from the
        // cause. See sampling::effectiveVocabSize.
        const long long graph_width = session_cache->lastLogitsWidth();
        const int sampled_vocab_size = sampling::effectiveVocabSize(vocab_size, graph_width);
        if (sampled_vocab_size != vocab_size) {
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                                "declared vocab_size %d exceeds the graph's logits width %lld; "
                                "sampling over %d. Re-export the package: its "
                                "mobiletransformers_tokenizer_config.json is wrong.",
                                vocab_size, graph_width, sampled_vocab_size);
        }

        int best_index = sampling::sampleNextToken(logits, past_sequence_length, sampled_vocab_size, session_cache->sampling_config, session_cache->random_generator);

        env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(attention_mask, attention_mask_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(position_ids, position_ids_elements, JNI_ABORT);
        return best_index;
    } catch (const std::exception& e) {
        env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(attention_mask, attention_mask_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(position_ids, position_ids_elements, JNI_ABORT);
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "performInferenceStep failed: %s", e.what());
        jclass runtime_exception = env->FindClass("java/lang/RuntimeException");
        if (runtime_exception != nullptr) {
            env->ThrowNew(runtime_exception, (std::string("inference step failed: ") + e.what()).c_str());
        }
        return -1;
    }
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_createTokenizerSession(JNIEnv *env,
                                                                          jobject thiz,
                                                                          jstring jTokenizerFile) {
    // Convert Java string to C++ string
    const char *tokenizer_file = env->GetStringUTFChars(jTokenizerFile, nullptr);
    std::unique_ptr<TokenizerSessionCache> tokenizer = std::make_unique<TokenizerSessionCache>(tokenizer_file);

    // Release the Java string memory
    env->ReleaseStringUTFChars(jTokenizerFile, tokenizer_file);

    // Return the handle (cast the unique pointer to `jlong`)
    return reinterpret_cast<jlong>(tokenizer.release());
}


extern "C"
JNIEXPORT jintArray JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_tokenizeString(JNIEnv *env, jobject thiz,
                                                                  jlong tokenizer_model,
                                                                  jstring sequence) {
    // Convert Java string to C++ string
    const char *text = env->GetStringUTFChars(sequence, nullptr);

    std::vector<int32_t> tokens = tokenization::tokenize(tokenizer_model, text);

    // Release the Java string memory
    env->ReleaseStringUTFChars(sequence, text);
    jintArray token_array = env->NewIntArray(tokens.size());
    env->SetIntArrayRegion(token_array, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));
    return token_array;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_decodeString(JNIEnv *env, jobject thiz,
                                                                  jlong tokenizer_model,
                                                                  jintArray sequence) {

    // Convert Java int array to C++ vector
    jsize length = env->GetArrayLength(sequence);
    std::vector<int32_t> token_ids(length);
    env->GetIntArrayRegion(sequence, 0, length, reinterpret_cast<jint *>(token_ids.data()));

    // Decode the token IDs
    std::string decoded_text = tokenization::decode(tokenizer_model, token_ids);

    // Convert C++ string to Java string
    return env->NewStringUTF(decoded_text.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_releaseTokenizerSession(JNIEnv *env, jobject thiz, jlong tokenizer_model) {
    ReleaseTokenizerSession(tokenizer_model);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTokenizerNative_decodeToken(JNIEnv *env, jobject thiz,
                                                               jlong tokenizer_model,
                                                               jint token_id) {

    // Decode the token IDs
    std::string decoded_text = tokenization::decodeToken(tokenizer_model, token_id);

    // Convert C++ string to Java string
    return env->NewStringUTF(decoded_text.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_optimizerStep(JNIEnv *env, jobject thiz,
                                                                jlong session) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    return training::optimizer_step(session_cache);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_setLearningRate(JNIEnv *env, jobject thiz,
                                                                  jlong session,
                                                                  jfloat learning_rate) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    session_cache->SetLearningRate(learning_rate);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_saveModel(JNIEnv *env, jobject thiz,
                                                            jlong session, jboolean saveOptimizer) {
    auto* session_cache = reinterpret_cast<TrainingSessionCache *>(session);

    Ort::CheckpointState::SaveCheckpoint(session_cache->checkpoint_state, session_cache->artifact_paths.checkpoint_path, saveOptimizer);
}

extern "C"
JNIEXPORT jboolean JNICALL
/**
 * Merges and exports the weights which are then ready for inference session.
 *
 * @param env
 * @param thiz
 * @param session
 */
Java_com_martinkorelic_mobiletransformers_ORTTrainerNative_mergeExportWeights(JNIEnv *env, jobject thiz,
                                                                     jlong session,
                                                                     jstring peft_mapping_path,
                                                                     jstring merger_models_directory,
                                                                     jstring output_directory) {
    try {
        auto* session_cache = reinterpret_cast<TrainingSessionCache*>(session);

        // Convert Java strings to C++ strings
        const char* peft_path_cstr = env->GetStringUTFChars(peft_mapping_path, nullptr);
        const char* merger_models_dir_cstr = env->GetStringUTFChars(merger_models_directory, nullptr);
        const char* output_dir_cstr = env->GetStringUTFChars(output_directory, nullptr);

        std::string peft_path(peft_path_cstr);
        std::string merger_models_dir(merger_models_dir_cstr);
        std::string output_dir(output_dir_cstr);

        // Release Java strings
        env->ReleaseStringUTFChars(peft_mapping_path, peft_path_cstr);
        env->ReleaseStringUTFChars(merger_models_directory, merger_models_dir_cstr);
        env->ReleaseStringUTFChars(output_directory, output_dir_cstr);

        // Perform weight merging
        bool success = session_cache->weight_merger->merge_and_export_weights(
                session_cache->checkpoint_state,
                peft_path,
                merger_models_dir,
                output_dir
        );

        // Destroy the old WeightMerger instance and create a new one for next time
        session_cache->weight_merger.reset();
        session_cache->weight_merger = nullptr;
        session_cache->weight_merger = std::make_unique<WeightMerger>();

        return success ? JNI_TRUE : JNI_FALSE;

    } catch (const std::exception& e) {
        LOGE("Error in mergeExportWeights: %s", e.what());
        return JNI_FALSE;
    }
}

// Additional JNI function to configure sampling parameters
extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorNative_setSamplingConfig(JNIEnv *env, jobject thiz,
                                                                      jlong session,
                                                                      jint sampling_method,
                                                                      jfloat temperature,
                                                                      jint top_k,
                                                                      jfloat top_p,
                                                                      jint random_seed) {
    auto *session_cache = reinterpret_cast<InferenceSessionCache *>(session);

    auto method = static_cast<sampling::SamplingMethod>(sampling_method);

    session_cache->setSamplingConfig(method, temperature, top_k, top_p, random_seed);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_ORTRetriever_createEmbeddingSession(JNIEnv *env, jobject thiz,
                                                                     jstring embedding_model_path,
                                                                     jstring embedding_model_name,
                                                                     jstring cache_dir_path,
                                                                     jstring memory_config_id,
                                                                     jstring core_config_id,
                                                                     jstring execution_provider,
                                                                     jboolean enable_profiling) {
    try {
        // Convert Java strings to C++ strings
        const char *model_path_chars = env->GetStringUTFChars(embedding_model_path, nullptr);
        const char *model_name_chars = env->GetStringUTFChars(embedding_model_name, nullptr);
        const char *cache_path_chars = env->GetStringUTFChars(cache_dir_path, nullptr);
        const char *memory_config_chars = env->GetStringUTFChars(memory_config_id, nullptr);
        const char *core_config_chars = env->GetStringUTFChars(core_config_id, nullptr);
        const char *execution_provider_chars = env->GetStringUTFChars(execution_provider, nullptr);

        std::string model_path_str(model_path_chars);
        std::string model_name_str(model_name_chars);
        std::string cache_path_str(cache_path_chars);
        std::string memory_config_str(memory_config_chars);
        std::string core_config_str(core_config_chars);
        std::string execution_provider_str(execution_provider_chars);

        // Release Java string references
        env->ReleaseStringUTFChars(embedding_model_path, model_path_chars);
        env->ReleaseStringUTFChars(embedding_model_name, model_name_chars);
        env->ReleaseStringUTFChars(cache_dir_path, cache_path_chars);
        env->ReleaseStringUTFChars(memory_config_id, memory_config_chars);
        env->ReleaseStringUTFChars(core_config_id, core_config_chars);
        env->ReleaseStringUTFChars(execution_provider, execution_provider_chars);

        LOGI("Creating embedding session with model: %s", model_name_str.c_str());

        // Create the embedding session cache
        auto *embedding_session = new EmbeddingSessionCache(
                model_path_str,
                model_name_str,
                cache_path_str,
                memory_config_str,
                core_config_str,
                execution_provider_str,
                static_cast<bool>(enable_profiling)
        );

        LOGI("Embedding session created successfully");

        // Return the pointer as jlong
        return reinterpret_cast<jlong>(embedding_session);

    } catch (const std::exception &e) {
        LOGE("Failed to create embedding session: %s", e.what());

        // Throw Java exception
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class != nullptr) {
            env->ThrowNew(exception_class, e.what());
        }

        return 0;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTRetriever_releaseEmbeddingSession(JNIEnv *env, jobject thiz, jlong session) {
    try {

        auto *session_cache = reinterpret_cast<EmbeddingSessionCache *>(session);

        delete session_cache->embedding_session;
        delete session_cache;
        session_cache = nullptr;

    } catch (const std::exception& e) {
        LOGE("Failed to destroy embedding session: %s", e.what());
    }
}

extern "C"
JNIEXPORT jfloatArray JNICALL
/**
 * Performs the inference step using the exported model for inference.
 *
 * @param env
 * @param thiz
 * @param session
 * @param input_ids
 * @param attention_mask
 * @param token_type_ids
 * @param sequence_length
 *
 * @return Next token id
 */
Java_com_martinkorelic_mobiletransformers_ORTRetriever_performEmbeddingStep(JNIEnv *env, jobject thiz,
        jlong session,
        jlongArray input_ids,
        jlongArray attention_mask,
        jlongArray token_type_ids,
        jint batch_size,
        jint sequence_length,
        jint embedding_dim) {
    auto *session_cache = reinterpret_cast<EmbeddingSessionCache *>(session);

    jlong* input_ids_elements = env->GetLongArrayElements(input_ids, nullptr);
    jlong* attention_mask_elements = env->GetLongArrayElements(attention_mask, nullptr);
    jlong* token_type_ids_elements = env->GetLongArrayElements(token_type_ids, nullptr);

    // Forward pass
    auto embedding_vector = inference::generateEmbedding(session_cache,
                                        input_ids_elements,
                                     attention_mask_elements,
                                     token_type_ids_elements,
                                                 batch_size,
                                                 sequence_length);

    jint total_size = batch_size * embedding_dim;
    jfloatArray result = env->NewFloatArray(total_size);

    // JNI_ABORT: nothing was written back into these, so there is no copy worth committing. Without
    // the release the three arrays pinned above leak on EVERY embedding call — once per chunk during
    // a RAG ingest, which is the workload that makes it matter.
    auto release_inputs = [&] {
    env->ReleaseLongArrayElements(input_ids, input_ids_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(attention_mask, attention_mask_elements, JNI_ABORT);
        env->ReleaseLongArrayElements(token_type_ids, token_type_ids_elements, JNI_ABORT);
    };

    if (!result) {
        LOGE("Failed to create result float array");
        // NB: `embedding_vector` is NOT ours to free. It points into
        // `EmbeddingSessionCache::last_output`, owned by the cache and valid until the next forward
        // pass. This path used to `delete` it — a pointer that was never `new`-allocated, and
        // since the fix above, one that belongs to a live object.
        release_inputs;
        return nullptr;
    }

    // Copy data to Java array
    env->SetFloatArrayRegion(result, 0, total_size, embedding_vector);

    release_inputs;
    return result;
}