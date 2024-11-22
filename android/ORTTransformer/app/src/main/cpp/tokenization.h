//
// Created by martin on 18. 11. 24.
//

#ifndef ORTTRANSFORMER_TOKENIZATION_H
#define ORTTRANSFORMER_TOKENIZATION_H

#include "session_cache.h"

namespace tokenization {
    std::vector<int32_t> tokenize(jlong tokenizerCache, const std::string &text);
    std::string decode(jlong tokenizerCache, const std::vector<int> &ids);
    std::string decodeToken(jlong tokenizerCache, const int &id);
}

#endif //ORTTRANSFORMER_TOKENIZATION_H
