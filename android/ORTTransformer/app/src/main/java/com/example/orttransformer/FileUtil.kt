package com.example.orttransformer

import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream

fun copyAssetFile(assetManager: AssetManager, assetPath: String, dstFile: File) {
    // This function copies the asset file named by `assetPath` to the file specified by `dstFile`.
    check(!dstFile.exists() || dstFile.isFile)

    dstFile.parentFile?.mkdirs()

    val assetContents = assetManager.open(assetPath).use { assetStream ->
        val size: Int = assetStream.available()
        val buffer = ByteArray(size)
        assetStream.read(buffer)
        buffer
    }

    java.io.FileOutputStream(dstFile).use { dstStream ->
        dstStream.write(assetContents)
    }
}

fun copyAssetFileOrDir(assetManager: AssetManager, assetPath: String, dstFileOrDir: File) {
    // This function copies the asset file or directory named by `assetPath` to the file or
    // directory specified by `dstFileOrDir`.
    val assets: Array<String>? = assetManager.list(assetPath)
    if (assets!!.isEmpty()) {
        // asset is a file
        copyAssetFile(assetManager, assetPath, dstFileOrDir)
    } else {
        // asset is a dir. loop over dir and copy all files or sub dirs to cache dir
        for (i in assets.indices) {
            val assetChild = (if (assetPath.isEmpty()) "" else "$assetPath/") + assets[i]
            val dstChild = dstFileOrDir.resolve(assets[i])
            copyAssetFileOrDir(assetManager, assetChild, dstChild)
        }
    }
}

fun loadTrainableLayerNamesJSON(fileName: String): Array<String>? {
    val jsonString: String
    try {
        // Get the file path from internal storage
        val file = File(fileName)
        // Open the file input stream to read the file
        val fis = FileInputStream(file)
        val size = fis.available()
        // Create a byte array to hold the file contents
        val buffer = ByteArray(size)
        // Read the file into the buffer
        fis.read(buffer)
        fis.close()
        // Convert the buffer into a string
        jsonString = String(buffer)
    } catch (ex: Exception) {
        ex.printStackTrace()
        return null
    }

    val jsonObject = JSONObject(jsonString)
    val jsonArray = jsonObject.getJSONArray("requires_grad")
    val requiresGradList = mutableListOf<String>()

    for (i in 0 until jsonArray.length()) {
        requiresGradList.add(jsonArray.getString(i))
    }

    return requiresGradList.toTypedArray()
}