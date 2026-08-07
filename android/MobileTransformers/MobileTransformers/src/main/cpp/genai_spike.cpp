// GenAI external-data-swap spike (#10, Gate 0.1) — minimal JNI over the stable ONNX Runtime GenAI C API.
//
// Proves finding F2 on-device: OgaCreateModel(<inference dir>) resolves the package's relative external
// data, generates one token, and a FRESH model reflects overwritten external .bin bytes (no graph rewrite,
// no fork — OgaCreateModelWithInitializers is confirmed fork-only/absent, see check_symbols.sh). This is the
// seed of the File #11 GenAI engine wrapper that replaces the abandoned onnx-genai.cpp.
//
// runOneToken(dir, prompt) returns "token=<id>;fp=<logits fingerprint>;rssPre=<kB>;rssLoaded=<kB>;rssTok=<kB>".
// The instrumented test (GenAISpikeTest.kt) calls it before and after perturbing one external weight and
// asserts the fingerprint changes (swap observed).

#include <jni.h>
#include <android/log.h>
#include <fstream>
#include <string>
#include <vector>

#include "onnxruntime-genai/ort_genai_c.h"

#define LOG_TAG "GenAISpike"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

long rss_kb() {  // VmRSS from /proc/self/status (Android/Linux)
    std::ifstream s("/proc/self/status");
    std::string k;
    while (s >> k) {
        if (k == "VmRSS:") {
            long v = -1;
            s >> v;
            return v;
        }
    }
    return -1;
}

std::string g_last_error;  // last GenAI error text, surfaced back to the test

// Throw-free error surfacing: capture + log the GenAI error and return whether the call failed.
bool failed(OgaResult* r, const char* where) {
    if (r == nullptr) return false;
    const char* msg = OgaResultGetError(r);
    g_last_error = std::string(where) + ": " + (msg ? msg : "(null)");
    LOGE("%s", g_last_error.c_str());
    OgaDestroyResult(r);
    return true;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_martinkorelic_mobiletransformers_GenAISpike_runOneToken(
    JNIEnv* env, jobject /*thiz*/, jstring jdir, jstring jprompt) {
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    std::string out = "error";
    g_last_error.clear();

  try {  // catch C++ exceptions crossing the C-API boundary so the test fails cleanly (not SIGABRT)
    long rss_pre = rss_kb();  // (1) before load

    OgaModel* model = nullptr;
    OgaTokenizer* tok = nullptr;
    OgaSequences* seqs = nullptr;
    OgaGeneratorParams* params = nullptr;
    OgaGenerator* gen = nullptr;
    OgaTensor* logits = nullptr;

    do {
        if (failed(OgaCreateModel(dir, &model), "OgaCreateModel")) break;
        long rss_loaded = rss_kb();  // (2) after load — mmap ~= file size, copy ~= 2x

        if (failed(OgaCreateTokenizer(model, &tok), "OgaCreateTokenizer")) break;
        if (failed(OgaCreateSequences(&seqs), "OgaCreateSequences")) break;
        if (failed(OgaTokenizerEncode(tok, prompt, seqs), "OgaTokenizerEncode")) break;

        size_t prompt_len = OgaSequencesGetSequenceCount(seqs, 0);
        if (failed(OgaCreateGeneratorParams(model, &params), "OgaCreateGeneratorParams")) break;
        OgaGeneratorParamsSetSearchNumber(params, "max_length", (double)(prompt_len + 1));
        OgaGeneratorParamsSetSearchBool(params, "do_sample", false);
        if (failed(OgaCreateGenerator(model, params, &gen), "OgaCreateGenerator")) break;
        if (failed(OgaGenerator_AppendTokenSequences(gen, seqs), "AppendTokenSequences")) break;
        if (failed(OgaGenerator_GenerateNextToken(gen), "GenerateNextToken")) break;
        long rss_tok = rss_kb();  // (3) after first token

        // Logits fingerprint (order-sensitive) so the test can detect a swap without shipping the vector.
        double fp = 0.0;
        if (!failed(OgaGenerator_GetLogits(gen, &logits), "GetLogits") && logits) {
            void* data = nullptr;
            size_t rank = 0;
            OgaTensorGetShapeRank(logits, &rank);
            std::vector<int64_t> shape(rank);
            OgaTensorGetShape(logits, shape.data(), rank);
            size_t count = 1;
            for (size_t i = 0; i < rank; ++i) count *= (size_t)shape[i];
            if (!failed(OgaTensorGetData(logits, &data), "GetData") && data) {
                const float* f = static_cast<const float*>(data);
                for (size_t i = 0; i < count; ++i) fp += (double)f[i] * (double)((i % 1024) + 1);
            }
        }

        size_t seq_len = OgaGenerator_GetSequenceCount(gen, 0);
        const int32_t* seq = OgaGenerator_GetSequenceData(gen, 0);
        int token = (seq && seq_len > 0) ? seq[seq_len - 1] : -1;

        LOGI("dir=%s token=%d fp=%.6f rss pre=%ld loaded=%ld tok=%ld", dir, token, fp, rss_pre, rss_loaded, rss_tok);
        out = "token=" + std::to_string(token) + ";fp=" + std::to_string(fp) +
              ";rssPre=" + std::to_string(rss_pre) + ";rssLoaded=" + std::to_string(rss_loaded) +
              ";rssTok=" + std::to_string(rss_tok);
    } while (false);

    if (logits) OgaDestroyTensor(logits);
    if (gen) OgaDestroyGenerator(gen);
    if (params) OgaDestroyGeneratorParams(params);
    if (seqs) OgaDestroySequences(seqs);
    if (tok) OgaDestroyTokenizer(tok);
    if (model) OgaDestroyModel(model);
  } catch (const std::exception& e) {
    g_last_error = std::string("exception: ") + e.what();
    LOGE("%s", g_last_error.c_str());
  } catch (...) {
    g_last_error = "exception: non-std (ORT/GenAI ABI mismatch?)";
    LOGE("%s", g_last_error.c_str());
  }

    if (out == "error") out = "error=" + g_last_error;
    env->ReleaseStringUTFChars(jdir, dir);
    env->ReleaseStringUTFChars(jprompt, prompt);
    return env->NewStringUTF(out.c_str());
}
