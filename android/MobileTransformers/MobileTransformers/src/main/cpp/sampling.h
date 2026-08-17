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
         * How many logits the sampler may actually look at.
         *
         * ### Why the declared vocabulary cannot be trusted
         *
         * `vocab_size` reaches this layer from `mobiletransformers_tokenizer_config.json` — a file the
         * exporter writes and the phone reads. The graph's logits width is the number of embedding
         * rows that exist. When the two disagree, the graph is right by construction: an id the
         * embedding table has no row for is not a token, whatever a JSON file says.
         *
         * Over-declaring is not a cosmetic error. Every sampler here computes its row offset as
         * `(sequence_length - 1) * vocab_size`, so a vocab two entries too wide reads
         * `2 * (sequence_length - 1)` floats past the intended row on prefill, and then scans two
         * more past the end of the buffer. If that garbage wins the argmax the id is fed back as the
         * next `input_ids` and ORT fails the embedding lookup:
         *
         *     Gather node ... indices element out of data bounds, idx=262145 ... [-262144,262143]
         *
         * which is precisely what FunctionGemma did on device — its tokenizer declares two image
         * tokens above a 262144-row table, and the exporter sized the vocabulary from the tokenizer.
         * The exporter is fixed (`export/tokenizer_export.py`), but **every package already installed
         * on a device still carries the wrong number**, so the runtime must not depend on it being
         * right.
         *
         * Under-declaring is left alone: a caller narrowing the sampler to a prefix of the vocabulary
         * is a deliberate restriction, not a defect, and silently widening it would be the same class
         * of mistake in the other direction.
         *
         * @param declared_vocab_size the vocabulary the package declares. Non-positive means "unknown".
         * @param graph_logits_width  the last dimension of the logits tensor the graph produced.
         *   Non-positive means the shape could not be read, in which case the declaration is all
         *   there is.
         */
        inline int effectiveVocabSize(int declared_vocab_size, long long graph_logits_width) {
            if (graph_logits_width <= 0) {
                return declared_vocab_size;
            }
            if (declared_vocab_size <= 0 || declared_vocab_size > graph_logits_width) {
                return static_cast<int>(graph_logits_width);
            }
            return declared_vocab_size;
        }

    /**
     * Utility functions
     */

        // Comparison function for sorting TokenProb by value (descending)
        bool compareTokenProbDesc(const TokenProb& a, const TokenProb& b);

        // Apply temperature scaling to logits
        void applyTemperature(std::vector<float>& logits, float temperature);

} // namespace inference::sampling

#endif //MOBILETRANSFORMERS_SAMPLING_H