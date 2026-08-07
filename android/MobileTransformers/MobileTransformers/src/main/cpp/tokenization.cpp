//
// Created by martinkorelic on 18/11/2024.
//

#include "tokenizers/tokenizers_cpp.h"
#include "session_cache.h"

namespace tokenization {
    std::vector<int32_t> tokenize(jlong tokenizerCache, const std::string &text) {
        auto *tokenizer_session = reinterpret_cast<TokenizerSessionCache *>(tokenizerCache);
        auto encoding = tokenizer_session->tokenizer->Encode(text);
        return encoding;
    }

    std::string decode(jlong tokenizerCache, const std::vector<int> &ids) {
        auto *tokenizer_session = reinterpret_cast<TokenizerSessionCache *>(tokenizerCache);
        auto decoding = tokenizer_session->tokenizer->Decode(ids);
        return decoding;
    }

    std::string decodeToken(jlong tokenizerCache, const int &id) {
        auto *tokenizer_session = reinterpret_cast<TokenizerSessionCache *>(tokenizerCache);
        auto decoding = tokenizer_session->tokenizer->IdToToken(id);
        return decoding;
    }

}

