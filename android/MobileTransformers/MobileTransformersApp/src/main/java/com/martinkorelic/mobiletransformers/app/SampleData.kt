package com.martinkorelic.mobiletransformers.app

import android.content.Context
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import java.io.File

/**
 * Bundled sample data, and the one-tap installers that put it where the SDK looks for it.
 *
 * ### Why this exists
 *
 * Two screens were structurally unreachable on a freshly pulled package, which is exactly the state a
 * new user is in:
 *
 * - **Train** reads its dataset from `<cacheDir>/<repoId>/train/<trainFile>.jsonl`. Model packages
 *   deliberately ship no training data — the task belongs with the data, and the data is the caller's —
 *   so on a clean install that file does not exist and Start fails. Every instrumented test writes its
 *   own copy; a person using the app had no equivalent.
 * - **RAG** retrieves from a vector store that only `MobileTransformerModel.ingest` populates, and
 *   nothing in the app called it. The toggle therefore always returned zero sources.
 *
 * Neither is an SDK defect, and neither should be fixed by making the SDK ship data. What was missing
 * is the *worked example* of supplying it — which is this app's whole job.
 *
 * ### Why the assets are what they are
 *
 * [TRAIN_ASSET] is generated from the **same allowlist** the Tool calls screen declares, so training on
 * it teaches this app's real action boundary and every completion is one `FunctionCallValidator`
 * accepts. Regenerate it with the command below rather than editing it by hand — a row whose completion
 * the validator would reject teaches the model to produce something the app then refuses.
 *
 * ```
 * mobiletransformers agent-dataset --source generated \
 *   --allowlist  app/src/main/assets/sample_action_schema.json \
 *   --templates  app/src/main/assets/sample_action_templates.json \
 *   --name sample_mobile_actions --per-action 16 --seed 7 --output <dir>
 * ```
 *
 * Both inputs ship beside the output so that command is runnable from a clean checkout.
 * `sample_action_schema.json` is a copy of the `ToolCallViewModel` allowlist, and the templates are
 * needed because the generator's built-in `set_alarm` phrasings reference a `label` parameter this app
 * does not declare — it refuses to emit a row whose slots the action never declared, which is the
 * check that keeps the training set and the validator from drifting apart.
 */
object SampleData {

    /** Tool-call training rows (`{"prompt", "completion"}`), matching the `mobile_actions` task. */
    const val TRAIN_ASSET = "sample_mobile_actions.jsonl"

    /** The preprocessor that parses [TRAIN_ASSET]; goes into `DatasetConfig.task`. */
    const val TRAIN_TASK = "mobile_actions"

    /** `DatasetConfig.trainFile` this installs as — the name, without the `.jsonl` the SDK appends. */
    const val TRAIN_FILE = "sample_mobile_actions"

    /** A short document for the RAG ingest example. */
    const val RAG_ASSET = "sample_rag_document.md"

    /**
 * The bundled retrieval corpus — four short documents on deliberately distinct subjects.
 *
 * One document is enough to show that retrieval *runs*, and useless for showing that it
 * **works**: every query returns chunks of the only thing in the store, so a perfect ranking and a
 * random one look identical. With four separable subjects — the package format, fine-tuning,
 * privacy, and device behaviour — a query like "what leaves my phone during federated learning"
 * has a right answer a user can check at a glance, which is the whole point of the Retrieval
 * screen.
 */
    val RAG_ASSETS = listOf(
        RAG_ASSET,
        "sample_rag_training.md",
        "sample_rag_privacy.md",
        "sample_rag_device.md",
)

    /** Example queries whose best match is a *different* document each time. */
    val RAG_EXAMPLE_QUERIES = listOf(
        "what leaves my phone during federated learning?",
        "how does MARS differ from LoRA?",
        "why does a scheduled run start late?",
        "how is a package verified before it is downloaded?",
)

    /**
     * Copy [TRAIN_ASSET] into the installed package's `train/` stage as `<TRAIN_FILE>.jsonl`.
     *
     * The stage directory is resolved through [PackagePaths], never by appending `"train"` to a path:
     * a package declares where its stages live, and the cache layout is not simply the hub layout with
     * the `variants/<id>/` prefix removed.
     *
     * @param cacheDir the cache root the model was loaded from.
     * @param sanitizedRepoId the package's directory name (`PackageFormat.sanitizeRepoId`).
     * @return the installed file, or `null` when the package has no `train/` stage to put it in —
     *   the honest outcome for an inference-only package, rather than creating a directory the trainer
     *   will never read.
     */
    fun installTrainingSet(context: Context, cacheDir: File, sanitizedRepoId: String): File? {
        val trainDir = PackagePaths.forCache(cacheDir, sanitizedRepoId).train
        if (!trainDir.isDirectory) return null
        val target = File(trainDir, "$TRAIN_FILE.jsonl")
        copyAsset(context, TRAIN_ASSET, target)
        return target
    }

    /**
     * Copy [RAG_ASSET] into the app's own files dir and return it, ready to hand to `ingest`.
     *
     * Not written into the package: an ingested document is user data, not part of the model, and
     * putting it inside the package tree would make it collateral damage of the next reinstall.
     */
    fun installRagDocument(context: Context): File {
        val target = File(context.filesDir, RAG_ASSET)
        copyAsset(context, RAG_ASSET, target)
        return target
    }

    /**
 * Copy the whole [RAG_ASSETS] corpus into the app's files dir, ready to ingest.
 *
 * Returned in declaration order so the caller can report them predictably. Like
 * [installRagDocument] these are written to app storage rather than into the package: an ingested
 * document is user data, and putting it inside the package tree would make it collateral damage
 * of the next reinstall.
 */
    fun installRagCorpus(context: Context): List<File> =
    RAG_ASSETS.map { asset ->
        File(context.filesDir, asset).also { copyAsset(context, asset, it) }
    }

    private fun copyAsset(context: Context, asset: String, target: File) {
        target.parentFile?.mkdirs()
        context.assets.open(asset).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
    }
}
