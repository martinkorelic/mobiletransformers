// GenAI ModelRuntime engine (#11) — promoted from genai_spike.cpp. A session-handle wrapper over the stable
// ONNX Runtime GenAI C API with token-by-token streaming, so ORTGeneratorGenAI.kt can drive the SAME
// GenerationCallback/InferenceProgress sequence as the Native engine (loop in Kotlin, one JNI call per
// token). GenAI runs on the genai-paired stock ORT shipped as libort_gen.so (see
// spikes/genai_external_swap/README.md — ORT separation); no OgaCreateModelWithInitializers (fork-only).

#include <jni.h>
#include <android/log.h>
#include <string>

#include "onnxruntime-genai/ort_genai_c.h"

#define LOG_TAG "GenAIRuntime"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct GenAISession {
    OgaModel* model = nullptr;
    OgaTokenizer* tok = nullptr;
    OgaTokenizerStream* stream = nullptr;
    OgaGeneratorParams* params = nullptr;
    OgaGenerator* gen = nullptr;
    int last_token = -1;
    // sampling (applied at start): 0 greedy, 1 top_k, 2 top_p
    int method = 0;
    float temperature = 1.0f;
    int top_k = 10;
    float top_p = 0.9f;
    int seed = 42;
};

bool oga_failed(OgaResult* r, const char* where) {
    if (r == nullptr) return false;
    const char* msg = OgaResultGetError(r);
    LOGE("%s: %s", where, msg ? msg : "(null)");
    OgaDestroyResult(r);
    return true;
}

void reset_generation(GenAISession* s) {
    if (s->gen) { OgaDestroyGenerator(s->gen); s->gen = nullptr; }
    if (s->params) { OgaDestroyGeneratorParams(s->params); s->params = nullptr; }
}

GenAISession* handle(jlong h) { return reinterpret_cast<GenAISession*>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeCreate(
    JNIEnv* env, jobject, jstring jdir) {
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    auto* s = new GenAISession();
    bool ok = false;
    try {
        if (!oga_failed(OgaCreateModel(dir, &s->model), "OgaCreateModel") &&
            !oga_failed(OgaCreateTokenizer(s->model, &s->tok), "OgaCreateTokenizer") &&
            !oga_failed(OgaCreateTokenizerStream(s->tok, &s->stream), "OgaCreateTokenizerStream")) {
            ok = true;
        }
    } catch (...) {
        LOGE("nativeCreate: exception (ORT/GenAI ABI?)");
    }
    env->ReleaseStringUTFChars(jdir, dir);
    if (!ok) {
        if (s->stream) OgaDestroyTokenizerStream(s->stream);
        if (s->tok) OgaDestroyTokenizer(s->tok);
        if (s->model) OgaDestroyModel(s->model);
        delete s;
        return 0;
    }
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeSetSampling(
    JNIEnv*, jobject, jlong h, jint method, jfloat temperature, jint topK, jfloat topP, jint seed) {
    auto* s = handle(h);
    if (!s) return;
    s->method = method;
    s->temperature = temperature;
    s->top_k = topK;
    s->top_p = topP;
    s->seed = seed;
}

JNIEXPORT jboolean JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeStart(
    JNIEnv* env, jobject, jlong h, jstring jprompt, jint maxNewTokens) {
    auto* s = handle(h);
    if (!s) return JNI_FALSE;
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    OgaSequences* seqs = nullptr;
    bool ok = false;
    try {
        reset_generation(s);
        if (oga_failed(OgaCreateSequences(&seqs), "OgaCreateSequences")) goto done;
        if (oga_failed(OgaTokenizerEncode(s->tok, prompt, seqs), "OgaTokenizerEncode")) goto done;
        {
            size_t prompt_len = OgaSequencesGetSequenceCount(seqs, 0);
            if (oga_failed(OgaCreateGeneratorParams(s->model, &s->params), "OgaCreateGeneratorParams")) goto done;
            OgaGeneratorParamsSetSearchNumber(s->params, "max_length", (double)(prompt_len + maxNewTokens));
            OgaGeneratorParamsSetSearchBool(s->params, "do_sample", s->method != 0);
            if (s->method == 1) OgaGeneratorParamsSetSearchNumber(s->params, "top_k", s->top_k);
            if (s->method == 2) OgaGeneratorParamsSetSearchNumber(s->params, "top_p", s->top_p);
            if (s->method != 0) OgaGeneratorParamsSetSearchNumber(s->params, "temperature", s->temperature);
            if (oga_failed(OgaCreateGenerator(s->model, s->params, &s->gen), "OgaCreateGenerator")) goto done;
            if (oga_failed(OgaGenerator_AppendTokenSequences(s->gen, seqs), "AppendTokenSequences")) goto done;
            ok = true;
        }
    } catch (...) {
        LOGE("nativeStart: exception");
    }
done:
    if (seqs) OgaDestroySequences(seqs);
    env->ReleaseStringUTFChars(jprompt, prompt);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeIsDone(JNIEnv*, jobject, jlong h) {
    auto* s = handle(h);
    if (!s || !s->gen) return JNI_TRUE;
    return OgaGenerator_IsDone(s->gen) ? JNI_TRUE : JNI_FALSE;
}

// Generate one token; return its decoded (streamed) piece. Empty string on error/no-token.
JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeStep(JNIEnv* env, jobject, jlong h) {
    auto* s = handle(h);
    if (!s || !s->gen) return env->NewStringUTF("");
    std::string piece;
    try {
        if (oga_failed(OgaGenerator_GenerateNextToken(s->gen), "GenerateNextToken")) return env->NewStringUTF("");
        const int32_t* next = nullptr;
        size_t count = 0;
        if (!oga_failed(OgaGenerator_GetNextTokens(s->gen, &next, &count), "GetNextTokens") && next && count > 0) {
            s->last_token = next[count - 1];
            const char* out = nullptr;
            if (!oga_failed(OgaTokenizerStreamDecode(s->stream, s->last_token, &out), "StreamDecode") && out) {
                piece = out;  // 'out' is owned by the stream; copy before returning
            }
        }
    } catch (...) {
        LOGE("nativeStep: exception");
    }
    return env->NewStringUTF(piece.c_str());
}

JNIEXPORT jint JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeLastToken(JNIEnv*, jobject, jlong h) {
    auto* s = handle(h);
    return s ? s->last_token : -1;
}

JNIEXPORT void JNICALL
Java_com_martinkorelic_mobiletransformers_ORTGeneratorGenAI_nativeRelease(JNIEnv*, jobject, jlong h) {
    auto* s = handle(h);
    if (!s) return;
    reset_generation(s);
    if (s->stream) OgaDestroyTokenizerStream(s->stream);
    if (s->tok) OgaDestroyTokenizer(s->tok);
    if (s->model) OgaDestroyModel(s->model);
    delete s;
}

// genaiAvailable() native probe (#11 / Gate 0.1): the genai stack is linked and OgaCreateModel resolves,
// otherwise this library would not have loaded. Gate 0.1 passed (see spikes/genai_external_swap).
JNIEXPORT jboolean JNICALL
Java_com_martinkorelic_mobiletransformers_runtime_GenAiSupport_nativeGenAiAvailable(JNIEnv*, jobject) {
    return JNI_TRUE;
}

}  // extern "C"
