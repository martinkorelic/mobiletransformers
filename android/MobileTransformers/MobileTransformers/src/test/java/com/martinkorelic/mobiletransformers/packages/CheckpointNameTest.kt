package com.martinkorelic.mobiletransformers.packages

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The peft→ORT wrapper rewrite, which four implementations must agree on:
 * `artifacts/checkpoint_names.py`, `cpp/layer_name.h`, `artifacts/handoff_map.py` and this one.
 *
 * The pair is the two **wrappers** — peft's `base_model.model.<path>` and the ORT training wrapper's
 * `backbone.<path>` — not a model's own first module. All four used to spell it
 * `base_model.model.model.` → `backbone.model.`, which is the same rule with a *decoder's*
 * `model.layers…` baked in: identical for every decoder, and a silent no-op for an encoder. The #33
 * encoder export failed closed on exactly that, with 12/12 handoff entries naming
 * `base_model.model.bert.…base_layer.weight` against a checkpoint holding `backbone.bert.…`.
 */
class CheckpointNameTest {

    @Test
    fun decoderNamesConvertExactlyAsBefore() {
        assertEquals(
            "backbone.model.layers.9.self_attn.q_proj",
            WeightHandoffMap.toCheckpointName("base_model.model.model.layers.9.self_attn.q_proj"),
        )
    }

    @Test
    fun encoderNamesConvertToo() {
        // Real names from an all-MiniLM-L6-v2 LoRA export and its ORT checkpoint (2026-08-10).
        assertEquals(
            "backbone.bert.encoder.layer.0.attention.self.query",
            WeightHandoffMap.toCheckpointName(
                "base_model.model.bert.encoder.layer.0.attention.self.query"
            ),
        )
    }

    @Test
    fun aNameAlreadyInCheckpointSpaceIsLeftAlone() {
        // Only the peft wrapper is rewritten; re-prefixing would produce `backbone.backbone.…`.
        val already = "backbone.bert.encoder.layer.0.attention.self.query"
        assertEquals(already, WeightHandoffMap.toCheckpointName(already))
    }
}
