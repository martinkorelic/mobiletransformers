package com.martinkorelic.mobiletransformers.internal.runtime

/**
 * The three tensors an inference step binds, planned as pure data.
 *
 * @property inputIds the new tokens to run through the model — only the delta, never the cached prefix.
 * @property attentionMask one slot per token the model may attend to: `pastLength + inputIds.size`.
 * @property positionIds the absolute position of each entry in [inputIds], continuing from the cache.
 */
data class GenerationInputPlan(
    val inputIds: MutableList<Long>,
    val attentionMask: MutableList<Long>,
    val positionIds: MutableList<Long>,
)

/**
 * Plans the per-step inference inputs from the new tokens plus how much KV cache already exists.
 *
 * Deliberately pure and free of ORT/JNI, for the same reason `cpp/training_inputs.h`,
 * `cpp/layer_name.h` and `cpp/handoff_io.h` are: the decision is host-testable, so a JVM test can pin
 * the turn boundary that previously only a phone could reach.
 *
 * **The invariant, stated once:** the attention mask covers `past + new`, and the position ids are the
 * absolute indices of the new tokens *within that same span* — `past..past+new-1`. Both inputs must
 * agree about where the sequence is.
 *
 * That invariant was violated. `ORTGeneratorNative.createModelInputs` built the mask as
 * `pastAttentionMaskLength + k` but the positions as `0..k-1`, so the second turn of a conversation
 * told the graph "you have N cached tokens" and "these tokens start at 0" simultaneously. transformers
 * 4.46.2 tolerated the contradiction; 4.57.6 does not, and surfaced it as a `Gather` index out of
 * bounds on the second prompt only. The upgrade exposed the defect rather than causing it.
 */
object GenerationInputs {

    /**
     * @param inputIds the new tokens for this step.
     * @param pastLength the number of tokens already in the KV cache (0 on the first turn).
     * @throws IllegalArgumentException if [pastLength] is negative — a caller that has lost track of
     *   the cache length must fail here, not bind a nonsensical mask and get an opaque ORT error.
     */
    fun plan(inputIds: IntArray, pastLength: Int): GenerationInputPlan {
        require(pastLength >= 0) { "pastLength must be >= 0, got $pastLength" }
        return GenerationInputPlan(
            inputIds = inputIds.map { it.toLong() }.toMutableList(),
            attentionMask = MutableList(pastLength + inputIds.size) { 1L },
            positionIds = MutableList(inputIds.size) { (pastLength + it).toLong() },
        )
    }
}
