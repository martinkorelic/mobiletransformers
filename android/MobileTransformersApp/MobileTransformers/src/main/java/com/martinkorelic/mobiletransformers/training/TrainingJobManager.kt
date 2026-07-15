package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.ORTTrainingConfig
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.repository.LLMRepository

/**
 * A reconstructable description of a training job (#18) — the seam a future WorkManager `Worker` (#34) uses
 * to rebuild a [TrainingJob] after process death. Defined here; **no WorkManager dependency is added**.
 */
data class TrainingJobSpec(
    val repoId: String,
    val config: ORTTrainingConfig,
)

/**
 * Owns one [TrainingJob] per sanitized repo id (#18). Foundation hook for scheduled training (#34); this
 * class only manages job identity/lifetime, not scheduling.
 */
class TrainingJobManager(private val repo: LLMRepository) {
    private val jobs = mutableMapOf<String, TrainingJob>()

    fun getOrCreate(repoId: String): TrainingJob {
        val key = PackageFormat.sanitizeRepoId(repoId)
        return jobs.getOrPut(key) { TrainingJob(repo, key) }
    }

    fun get(repoId: String): TrainingJob? = jobs[PackageFormat.sanitizeRepoId(repoId)]

    fun clear() = jobs.clear()
}
