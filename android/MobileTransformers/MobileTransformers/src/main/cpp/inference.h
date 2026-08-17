//
// Created by martinkorelic on 19/09/2024.
//

#include <cstdint>
#include <string>
#include "session_cache.h"

namespace inference {

    /**
     * Executes a single forward pass of the LLM using the provided session cache and input tensors.
     * This method computes logits for the current input sequence without using any past key/value cache.
     *
     * @param session_cache        Pointer to the inference session cache (e.g., holds weights, buffers).
     * @param input_ids            Pointer to input token IDs of shape [batch_size, sequence_length].
     * @param attention_mask       Pointer to the attention mask of shape [batch_size, sequence_length].
     * @param position_ids         Pointer to position IDs of shape [batch_size, sequence_length].
     * @param batch_size           Number of input sequences in the batch.
     * @param sequence_length      Length of each input sequence.
     * @param vocab_size           Size of the vocabulary (used for determining output dimensions).
     *
     * @return Pointer to the output logits of shape [batch_size, sequence_length, vocab_size].
     *         The caller is responsible for managing the memory if applicable.
     */
    float* forward(InferenceSessionCache *session_cache, int64_t *input_ids, int64_t *attention_mask,
                   int64_t *position_ids, int64_t batch_size, int64_t sequence_length, size_t vocab_size);

    /**
     * Executes a forward pass optimized for autoregressive generation using cached key/value pairs
     * (KV cache) from previous tokens. Only processes the latest tokens and reuses past computation.
     *
     * @param session_cache         Pointer to the inference session cache holding KV cache and state.
     * @param input_ids             Pointer to new input token IDs of shape [batch_size, sequence_length].
     * @param attention_mask        Pointer to the attention mask of shape [batch_size, total_sequence_length].
     * @param position_ids          Pointer to position IDs of shape [batch_size, sequence_length].
     * @param batch_size            Number of sequences in the batch.
     * @param sequence_length       Number of new tokens being processed in this generation step.
     * @param past_sequence_length  Number of tokens already processed (length of cached sequence).
     *
     * @return Pointer to the output logits of shape [batch_size, sequence_length, vocab_size].
     *         Typically, only the logits of the last token are used during generation, but this returns all logits.
     */
    float* generateWithKVCache(InferenceSessionCache* session_cache,
                               int64_t* input_ids,
                               int64_t* attention_mask,
                               int64_t* position_ids,
                               int64_t batch_size,
                               int64_t sequence_length,
                               int64_t past_sequence_length);

    /**
    * Generates embeddings for input text sequences using a transformer-based embedding model.
    *
    * @param session_cache   Pointer to the embedding session cache containing the loaded model
    *                        and any necessary state information.
    * @param input_ids       Pointer to input token IDs of shape [batch_size, sequence_length].
    *                        Contains the tokenized representation of input text sequences.
    * @param attention_mask  Pointer to the attention mask of shape [batch_size, sequence_length].
    *                        Binary mask where 1 indicates real tokens and 0 indicates padding.
    *                        Can be null if the model doesn't require attention masking.
    * @param position_ids    Pointer to position IDs of shape [batch_size, sequence_length].
    *                        Specifies the position of each token in the sequence. Can be null
    *                        if the model uses default positional encoding.
    * @param batch_size      Number of text sequences being processed simultaneously.
    * @param sequence_length Maximum length of input sequences (including padding).
    *
    * @return Pointer to the output embeddings of shape [batch_size, embedding_dimension].
    *         Returns dense vector representations where each row corresponds to the embedding
    *         of one input sequence. The embedding dimension depends on the model architecture
    *         (e.g., 384 for MiniLM-L6, 768 for BERT-base). Returns nullptr on error.
    *
    */
    float* generateEmbedding(EmbeddingSessionCache* session_cache,
                             int64_t* input_ids,
                             int64_t* attention_mask,
                             int64_t* token_type_ids,
                             int64_t batch_size,
                             int64_t sequence_length);

} // namespace inference