//
// Created by martinkorelic on 20. 07. 25.
//

#ifndef MOBILETRANSFORMERS_SAMPLING_H
#define MOBILETRANSFORMERS_SAMPLING_H

#include <vector>
#include <random>

namespace sampling {

        /**
         * Enumeration for different sampling methods
         */
        enum class SamplingMethod {
            GREEDY = 0,
            TOP_K = 1,
            TOP_P = 2
        };

        /**
         * Configuration structure for sampling parameters
         */
        struct SamplingConfig {
            SamplingMethod method = SamplingMethod::GREEDY;
            float temperature = 1.0f;
            int top_k = 50;
            float top_p = 0.9f;
            unsigned int random_seed = 0; // 0 means use random device

            SamplingConfig() = default;

            SamplingConfig(SamplingMethod m, float temp = 1.0f, int k = 50, float p = 0.9f, unsigned int seed = 0)
                    : method(m), temperature(temp), top_k(k), top_p(p), random_seed(seed) {}
        };

        /**
         * Thread-safe random number generator wrapper
         */
        class RandomGenerator {
        public:
            explicit RandomGenerator(unsigned int seed = 0);

            // Sample from a discrete probability distribution
            int sampleFromDistribution(const std::vector<float>& probabilities);

            // Reset with new seed
            void setSeed(unsigned int seed);

        private:
            std::mt19937 generator_;
            std::random_device rd_;
        };

        /**
         * Structure to hold token index and its probability/logit value
         */
        struct TokenProb {
            int index;
            float value;

            TokenProb(int idx, float val) : index(idx), value(val) {}
        };

        /**
         * Core sampling functions
         */

        /**
         * Apply softmax to convert logits to probabilities
         * @param logits Input logits vector
         * @param num_logits Number of logits to process
         * @return Probability distribution
         */
        std::vector<float> softmax(const std::vector<float>& logits, size_t num_logits);

        /**
         * Greedy sampling - selects the token with highest probability
         * @param logits Pointer to logits array
         * @param sequence_length Current sequence length
         * @param vocab_size Size of vocabulary
         * @return Index of selected token
         */
        int greedySampling(float* logits, int sequence_length, int vocab_size);

        /**
         * Top-k sampling
         * @param logits Pointer to logits array
         * @param sequence_length Current sequence length
         * @param vocab_size Size of vocabulary
         * @param config Sampling configuration
         * @param rng Random number generator
         * @return Index of selected token
         */
        int topKSampling(float* logits, int sequence_length, int vocab_size,
                         const SamplingConfig& config, RandomGenerator& rng);

        /**
         * Top-p (nucleus) sampling
         * @param logits Pointer to logits array
         * @param sequence_length Current sequence length
         * @param vocab_size Size of vocabulary
         * @param config Sampling configuration
         * @param rng Random number generator
         * @return Index of selected token
         */
        int topPSampling(float* logits, int sequence_length, int vocab_size,
                         const SamplingConfig& config, RandomGenerator& rng);

        /**
         * Main sampling function that dispatches to appropriate method
         * @param logits Pointer to logits array
         * @param sequence_length Current sequence length
         * @param vocab_size Size of vocabulary
         * @param config Sampling configuration
         * @param rng Random number generator
         * @return Index of selected token
         */
        int sampleNextToken(float* logits, int sequence_length, int vocab_size,
                            const SamplingConfig& config, RandomGenerator& rng);

    /**
     * Utility functions
     */

        // Comparison function for sorting TokenProb by value (descending)
        bool compareTokenProbDesc(const TokenProb& a, const TokenProb& b);

        // Apply temperature scaling to logits
        void applyTemperature(std::vector<float>& logits, float temperature);

} // namespace inference::sampling

#endif //MOBILETRANSFORMERS_SAMPLING_H