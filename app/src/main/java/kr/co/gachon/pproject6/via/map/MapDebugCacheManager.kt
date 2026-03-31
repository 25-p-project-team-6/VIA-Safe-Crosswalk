package kr.co.gachon.pproject6.via.map

import android.content.Context
import java.io.File

object MapDebugCacheManager {
    private const val TILE_CACHE_DIR = "via-debug-tile-cache"
    private const val OSM_CACHE_DIR = "via-overpass-cache"

    fun tileCacheDir(context: Context): File =
        File(context.cacheDir, TILE_CACHE_DIR)

    fun osmCacheDir(context: Context): File =
        File(context.cacheDir, OSM_CACHE_DIR)

    fun clearAll(context: Context): Int {
        val tileDeleted = deleteRecursively(tileCacheDir(context))
        val osmDeleted = deleteRecursively(osmCacheDir(context))
        return tileDeleted + osmDeleted
    }

    private fun deleteRecursively(file: File): Int {
        if (!file.exists()) {
            return 0
        }
        var deletedCount = 0
        if (file.isDirectory) {
            file.listFiles()?.forEach { child ->
                deletedCount += deleteRecursively(child)
            }
        }
        if (file.delete()) {
            deletedCount += 1
        }
        return deletedCount
    }
}
