package com.martinkorelic.mobiletransformers.hub

import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest
import com.martinkorelic.mobiletransformers.packages.ModelPackageInstaller
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient

/**
 * #21: manifest-first Hub pull → verify → atomic install, mirroring the Python `hub/pull.py`
 * (`pull_package` + `install_package`). Downloads the manifest first, plans the file list
 * ([DownloadPlanner]), streams + sha256-verifies each file ([PackageDownloader]), then materializes via
 * the existing [ModelPackageInstaller] (atomic rename). Reuses the `packages/` verify/select/install half;
 * this is only the network front-end. `client` is injectable for tests.
 */
object HubDownloader {

    suspend fun downloadAndInstall(
        cacheDir: File,
        repoId: String,
        revision: String = "main",
        variant: String? = null,
        features: Set<String> = setOf("inference"),
        genai: Boolean = false,
        endpoint: String = HubResolver.DEFAULT_ENDPOINT,
        token: String? = null,
        client: OkHttpClient = OkHttpClient(),
        onProgress: (done: Int, total: Int, path: String) -> Unit = { _, _, _ -> },
    ): ModelPackageInstaller.Installed =
        withContext(Dispatchers.IO) {
            val sanitized = PackageFormat.sanitizeRepoId(repoId)
            val staging = File(cacheDir, ".download/$sanitized").apply {
                deleteRecursively()
                mkdirs()
            }
            val headers = HubResolver.authHeaders(token)
            val urlFor = { path: String -> HubResolver.fileUrl(endpoint, repoId, revision, path) }

            // Manifest first (no large GET precedes it) — no checksum yet (it names the others' checksums).
            PackageDownloader.download(
                client = client,
                files = listOf(PackageFormat.MANIFEST_FILENAME),
                urlFor = urlFor,
                headers = headers,
                expectedSha = emptyMap(),
                destRoot = staging,
            )
            val manifest = MobileTransformersManifest.load(File(staging, PackageFormat.MANIFEST_FILENAME))
            val variantId = variant ?: manifest.defaultVariant

            val files = DownloadPlanner.planFiles(manifest, variantId, features, genai)
            PackageDownloader.download(
                client = client,
                files = files,
                urlFor = urlFor,
                headers = headers,
                expectedSha = manifest.sha256,
                destRoot = staging,
                onProgress = onProgress,
            )

            ModelPackageInstaller.install(staging, cacheDir, repoId, variantId)
        }
}
