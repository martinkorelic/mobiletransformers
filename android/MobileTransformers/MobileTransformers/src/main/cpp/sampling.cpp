//
// Created by martinkorelic on 20. 07. 25.
//

#include "sampling.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>


namespace sampling {

        // RandomGenerator implementation
        RandomGenerator::RandomGenerator(unsigned int seed) {
            if (seed == 0) {
                generator_.seed(rd_());
            } else {
                generator_.seed(seed);
            }
        }

        int RandomGenerator::sampleFromDistribution(const std::vector<float>& probabilities) {
            std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
            return distribution(generator_);
        }

        void RandomGenerator::setSeed(unsigned int seed) {
            if (seed == 0) {
                generator_.seed(rd_());
            } else {
                generator_.seed(seed);
            }
        }

        // Utility functions
        bool compareTokenProbDesc(const TokenProb& a, const TokenProb& b) {
            return a.value > b.value;
        }

        void applyTemperature(std::vector<float>& logits, float temperature) {
            assert(temperature > 0.0f && "Temperature should be positive");

            if (temperature != 1.0f) {
                for (float& logit : logits) {
                    logit /= temperature;
                }
            }
        }

        std::vector<float> softmax(const std::vector<float>& logits, size_t num_logits) {
            assert(num_logits > 0 && "Number of logits should be positive");
            assert(logits.size() >= num_logits && "Logits vector size should be at least num_logits");

            std::vector<float> probabilities(num_logits, 0.0f);

            // Find max for numerical stability
            float max_logit = *std::max_element(logits.begin(), logits.begin() + num_logits);

            // Compute exp and sum
            float sum = 0.0f;
            for (size_t i = 0; i < num_logits; ++i) {
                probabilities[i] = std::exp(logits[i] - max_logit);
                sum += probabilities[i];
            }

            // Normalize
            if (sum > 0.0f) {
                for (size_t i = 0; i < num_logits; ++i) {
                    probabilities[i] /= sum;
                }
            }

            return probabilities;
        }

        int greedySampling(float* logits, int sequence_length, int vocab_size) {
            assert(logits != nullptr && "Logits pointer should not be null");
            assert(sequence_length > 0 && "Sequence length should be positive");
            assert(vocab_size > 0 && "Vocabulary size should be positive");

            int last_token_start_index = (sequence_length - 1) * vocab_size;
            float* last_token_logits = &logits[last_token_start_index];

            // Find the max logit for the most probable token ID in the last token's logits
            float* max_logit_ptr = std::max_element(last_token_logits, last_token_logits + vocab_size);

            // Calculate and return the index of the maximum logit in the last token's logits
            return std::distance(last_token_logits, max_logit_ptr);
        }

        int topKSampling(float* logits, int sequence_length, int vocab_size,
                         const SamplingConfig& config, RandomGenerator& rng) {
            assert(logits != nullptr && "Logits pointer should not be null");
            assert(sequence_length > 0 && "Sequence length should be positive");
            assert(vocab_size > 0 && "Vocabulary size should be positive");
            assert(config.top_k > 0 && config.top_k <= vocab_size && "top_k should be between 1 and vocab_size");
            assert(config.temperature > 0.0f && "Temperature should be positive");

            int last_token_start_index = (sequence_length - 1) * vocab_size;
            float* last_token_logits = &logits[last_token_start_index];

            // Copy and apply temperature scaling
            std::vector<float> scaled_logits(last_token_logits, last_token_logits + vocab_size);
            applyTemperature(scaled_logits, config.temperature);

            // Create vector of token-probability pairs
            std::vector<TokenProb> token_probs;
            token_probs.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i) {
                token_probs.emplace_back(i, scaled_logits[i]);
            }

            // Sort by logit value (descending)
            std::sort(token_probs.begin(), token_probs.end(), compareTokenProbDesc);

            // Keep only top k tokens
            int effective_k = std::min(config.top_k, vocab_size);

            // Convert logits to probabilities using softmax
            std::vector<float> top_k_logits(effective_k);
            for (int i = 0; i < effective_k; ++i) {
                top_k_logits[i] = token_probs[i].value;
            }

            // Keep only the top-k tokens
            token_probs.erase(token_probs.begin() + effective_k, token_probs.end());

            std::vector<float> probabilities = softmax(top_k_logits, effective_k);

            // Sample from the top-k distribution
            int sampled_idx = rng.sampleFromDistribution(probabilities);

            return token_probs[sampled_idx].index;
        }

        int topPSampling(float* logits, int sequence_length, int vocab_size,
                         const SamplingConfig& config, RandomGenerator& rng) {
            assert(logits != nullptr && "Logits pointer should not be null");
            assert(sequence_length > 0 && "Sequence length should be positive");
            assert(vocab_size > 0 && "Vocabulary size should be positive");
            assert(config.top_p > 0.0f && config.top_p <= 1.0f && "top_p should be between 0 and 1");
            assert(config.temperature > 0.0f && "Temperature should be positive");

            int last_token_start_index = (sequence_length - 1) * vocab_size;
            float* last_token_logits = &logits[last_token_start_index];

            // Copy and apply temperature scaling
            std::vector<float> scaled_logits(last_token_logits, last_token_logits + vocab_size);
            applyTemperature(scaled_logits, config.temperature);

            // Convert all logits to probabilities first
            std::vector<float> all_probabilities = softmax(scaled_logits, vocab_size);

            // Create vector of token-probability pairs
            std::vector<TokenProb> token_probs;
            token_probs.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i) {
                token_probs.emplace_back(i, all_probabilities[i]);
            }

            // Sort by probability (descending)
            std::sort(token_probs.begin(), token_probs.end(), compareTokenProbDesc);

            // Find nucleus (top-p subset)
            float cumulative_prob = 0.0f;
            int nucleus_size = 0;

            for (int i = 0; i < vocab_size; ++i) {
                cumulative_prob += token_probs[i].value;
                nucleus_size++;

                if (cumulative_prob >= config.top_p) {
                    break;
                }
            }

            // Ensure at least one token is included
            nucleus_size = std::max(nucleus_size, 1);

            // Renormalize probabilities for the nucleus
            std::vector<float> nucleus_probabilities(nucleus_size);
            float nucleus_sum = 0.0f;

            for (int i = 0; i < nucleus_size; ++i) {
                nucleus_probabilities[i] = token_probs[i].value;
                nucleus_sum += token_probs[i].value;
            }

            // Normalize
            if (nucleus_sum > 0.0f) {
                for (int i = 0; i < nucleus_size; ++i) {
                    nucleus_probabilities[i] /= nucleus_sum;
                }
            }

            // Sample from the nucleus
            int sampled_idx = rng.sampleFromDistribution(nucleus_probabilities);

            return token_probs[sampled_idx].index;
        }

        int sampleNextToken(float* logits, int sequence_length, int vocab_size,
                            const SamplingConfig& config, RandomGenerator& rng) {

            switch (config.method) {
                case SamplingMethod::GREEDY:
                    return greedySampling(logits, sequence_length, vocab_size);
                case SamplingMethod::TOP_K:
                    return topKSampling(logits, sequence_length, vocab_size, config, rng);
                case SamplingMethod::TOP_P:
                    return topPSampling(logits, sequence_length, vocab_size, config, rng);
                default:
                    return greedySampling(logits, sequence_length, vocab_size);
            }
        }

} // namespace inference::sampling