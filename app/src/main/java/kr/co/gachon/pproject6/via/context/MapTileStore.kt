package kr.co.gachon.pproject6.via.context

import android.content.Context
import android.util.Log
import kr.co.gachon.pproject6.via.BuildConfig
import kr.co.gachon.pproject6.via.onboarding.AppPreferences
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

internal class MapTileStore(
    context: Context,
    private val preferences: AppPreferences = AppPreferences(context),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private val appContext = context.applicationContext
    private val refreshInFlight = AtomicBoolean(false)
    private val bundledTileCache = mutableMapOf<String, List<MapFeatureRecord>>()
    private val remoteTileCache = mutableMapOf<String, List<MapFeatureRecord>>()

    @Volatile
    private var bundledManifestCache: MapDatasetManifest? = null

    @Volatile
    private var remoteManifestCache: MapDatasetManifest? = null

    fun loadAround(point: GeoPoint): MapTileLoadResult {
        val requestedTileIds = MapTileGrid.neighboringTileIds(point)
        return loadTileIds(requestedTileIds)
    }

    fun loadWithinRadius(
        point: GeoPoint,
        radiusMeters: Int
    ): MapTileLoadResult {
        val requestedTileIds = MapTileGrid.tileIdsForRadius(point, radiusMeters)
        return loadTileIds(requestedTileIds)
    }

    private fun loadTileIds(
        requestedTileIds: Set<String>
    ): MapTileLoadResult {
        val bundledManifest = loadBundledManifest()
        val remoteManifest = loadRemoteManifest()
        val features = mutableListOf<MapFeatureRecord>()
        var usedRemoteData = false

        requestedTileIds.forEach { tileId ->
            val remoteDescriptor = remoteManifest?.tiles?.get(tileId)
            val remoteTileFile = remoteTileFile(tileId)
            if (remoteDescriptor != null && remoteTileFile.exists()) {
                features += loadRemoteTile(tileId, remoteDescriptor, remoteManifest.version)
                usedRemoteData = true
            } else {
                val bundledDescriptor = bundledManifest.tiles[tileId] ?: return@forEach
                features += loadBundledTile(tileId, bundledDescriptor, bundledManifest.version)
            }
        }

        val datasetVersion = when {
            usedRemoteData && remoteManifest != null -> remoteManifest.version
            else -> bundledManifest.version
        }

        return MapTileLoadResult(
            features = features,
            datasetVersion = datasetVersion,
            usedRemoteData = usedRemoteData
        )
    }

    fun scheduleRefreshAround(point: GeoPoint) {
        val manifestUrl = BuildConfig.MAP_DATA_MANIFEST_URL
        if (manifestUrl.isBlank()) {
            return
        }

        val requestedTileIds = MapTileGrid.neighboringTileIds(point)
        val now = timeProvider()
        val remoteManifest = loadRemoteManifest()
        val missingTileData =
            remoteManifest?.let { manifest ->
                requestedTileIds.any { tileId ->
                    manifest.tiles[tileId]?.downloadUrl != null && !remoteTileFile(tileId).exists()
                }
            } ?: true
        val refreshDue =
            now - preferences.mapLastDatasetCheckAtMillis >= BuildConfig.MAP_DATA_REFRESH_INTERVAL_MS

        if (!refreshDue && !missingTileData) {
            return
        }
        if (!refreshInFlight.compareAndSet(false, true)) {
            return
        }

        Thread(
            {
                try {
                    refreshRemoteManifestAndTiles(manifestUrl, requestedTileIds, now)
                } catch (e: Exception) {
                    Log.w("MapTileStore", "Map dataset refresh failed", e)
                } finally {
                    refreshInFlight.set(false)
                }
            },
            "map-data-refresh"
        ).apply {
            isDaemon = true
            start()
        }
    }

    fun currentDatasetVersion(): String? {
        return loadRemoteManifest()?.version ?: loadBundledManifest().version
    }

    fun hasRemoteDataset(): Boolean {
        return loadRemoteManifest() != null
    }

    private fun refreshRemoteManifestAndTiles(
        manifestUrl: String,
        requestedTileIds: Set<String>,
        checkTimeMillis: Long
    ) {
        val remoteManifestJson = downloadString(manifestUrl)
        val remoteManifest = parseMapDatasetManifest(remoteManifestJson)
        val previousVersion = loadRemoteManifest()?.version
        if (previousVersion != null && previousVersion != remoteManifest.version) {
            remoteTileDirectory().deleteRecursively()
            synchronized(remoteTileCache) {
                remoteTileCache.clear()
            }
        }

        remoteTileDirectory().mkdirs()
        requestedTileIds.forEach { tileId ->
            val descriptor = remoteManifest.tiles[tileId] ?: return@forEach
            val downloadUrl = descriptor.downloadUrl ?: return@forEach
            val bytes = downloadBytes(downloadUrl)
            verifyChecksum(bytes, descriptor.checksum)
            val tileFile = remoteTileFile(tileId)
            val tempFile = File(tileFile.parentFile, "${tileFile.name}.tmp")
            tempFile.parentFile?.mkdirs()
            tempFile.writeBytes(bytes)
            if (!tempFile.renameTo(tileFile)) {
                tempFile.copyTo(tileFile, overwrite = true)
                tempFile.delete()
            }
        }

        val manifestFile = remoteManifestFile()
        val tempManifest = File(manifestFile.parentFile, "${manifestFile.name}.tmp")
        tempManifest.parentFile?.mkdirs()
        tempManifest.writeText(remoteManifestJson, StandardCharsets.UTF_8)
        if (!tempManifest.renameTo(manifestFile)) {
            tempManifest.copyTo(manifestFile, overwrite = true)
            tempManifest.delete()
        }

        preferences.mapDatasetVersion = remoteManifest.version
        preferences.mapLastDatasetCheckAtMillis = checkTimeMillis
        remoteManifestCache = remoteManifest
    }

    private fun loadBundledManifest(): MapDatasetManifest {
        bundledManifestCache?.let { return it }
        val manifestJson =
            appContext.assets.open(BUNDLED_MANIFEST_PATH).bufferedReader(StandardCharsets.UTF_8).use {
                it.readText()
            }
        return parseMapDatasetManifest(manifestJson).also { bundledManifestCache = it }
    }

    private fun loadRemoteManifest(): MapDatasetManifest? {
        remoteManifestCache?.let { return it }
        val manifestFile = remoteManifestFile()
        if (!manifestFile.exists()) {
            return null
        }
        val manifest =
            parseMapDatasetManifest(manifestFile.readText(StandardCharsets.UTF_8))
        remoteManifestCache = manifest
        return manifest
    }

    private fun loadBundledTile(
        tileId: String,
        descriptor: MapTileDescriptor,
        datasetVersion: String
    ): List<MapFeatureRecord> {
        synchronized(bundledTileCache) {
            bundledTileCache[tileId]?.let { return it }
        }
        val tileJson =
            appContext.assets.open(descriptor.filePath).bufferedReader(StandardCharsets.UTF_8).use {
                it.readText()
            }
        val parsed = parseMapTileFeatures(tileId, datasetVersion, tileJson)
        synchronized(bundledTileCache) {
            bundledTileCache[tileId] = parsed
        }
        return parsed
    }

    private fun loadRemoteTile(
        tileId: String,
        descriptor: MapTileDescriptor,
        datasetVersion: String
    ): List<MapFeatureRecord> {
        synchronized(remoteTileCache) {
            remoteTileCache[tileId]?.let { return it }
        }
        val tileFile = remoteTileFile(tileId)
        if (!tileFile.exists()) {
            return emptyList()
        }
        val parsed = parseMapTileFeatures(tileId, datasetVersion, tileFile.readText(StandardCharsets.UTF_8))
        synchronized(remoteTileCache) {
            remoteTileCache[tileId] = parsed
        }
        return parsed
    }

    private fun downloadString(url: String): String {
        return downloadBytes(url).toString(StandardCharsets.UTF_8)
    }

    private fun downloadBytes(url: String): ByteArray {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = NETWORK_TIMEOUT_MS
            readTimeout = NETWORK_TIMEOUT_MS
            requestMethod = "GET"
        }

        try {
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                throw IOException("HTTP $responseCode for $url")
            }
            return connection.inputStream.use { it.readBytes() }
        } finally {
            connection.disconnect()
        }
    }

    private fun verifyChecksum(bytes: ByteArray, expectedChecksum: String?) {
        if (expectedChecksum.isNullOrBlank()) {
            return
        }
        val actualChecksum =
            MessageDigest.getInstance("SHA-256")
                .digest(bytes)
                .joinToString("") { byte -> "%02x".format(byte) }
        if (!actualChecksum.equals(expectedChecksum, ignoreCase = true)) {
            throw IOException("Checksum mismatch for map tile payload")
        }
    }

    private fun remoteManifestFile(): File {
        return File(remoteRootDirectory(), REMOTE_MANIFEST_FILE_NAME)
    }

    private fun remoteTileFile(tileId: String): File {
        return File(remoteTileDirectory(), "$tileId.json")
    }

    private fun remoteRootDirectory(): File {
        return File(appContext.filesDir, REMOTE_ROOT_DIRECTORY)
    }

    private fun remoteTileDirectory(): File {
        return File(remoteRootDirectory(), REMOTE_TILE_DIRECTORY)
    }

    companion object {
        private const val BUNDLED_MANIFEST_PATH = "mapdata_manifest.json"
        private const val REMOTE_ROOT_DIRECTORY = "mapdata"
        private const val REMOTE_TILE_DIRECTORY = "tiles"
        private const val REMOTE_MANIFEST_FILE_NAME = "manifest.json"
        private const val NETWORK_TIMEOUT_MS = 5_000
    }
}

internal fun parseMapDatasetManifest(json: String): MapDatasetManifest {
    val root = SimpleJsonParser.parseObject(json)
    val version = root["version"].jsonStringOrDefault("bundled-placeholder-v1")
    val tiles = mutableMapOf<String, MapTileDescriptor>()
    val tileArray = root["tiles"].jsonArrayOrEmpty()

    for (itemValue in tileArray) {
        val item = itemValue.jsonObjectOrNull() ?: continue
        val tileId = item["tileId"].jsonStringOrNull().orEmpty()
        if (tileId.isBlank()) {
            continue
        }
        val filePath = item["file"].jsonStringOrNull().orEmpty()
        tiles[tileId] =
            MapTileDescriptor(
                tileId = tileId,
                filePath = if (filePath.isBlank()) "$tileId.json" else filePath,
                checksum = item["checksum"].jsonStringOrNull()?.takeIf { it.isNotBlank() },
                downloadUrl = item["downloadUrl"].jsonStringOrNull()?.takeIf { it.isNotBlank() }
            )
    }

    return MapDatasetManifest(version = version, tiles = tiles)
}

internal fun parseMapTileFeatures(
    tileId: String,
    datasetVersion: String,
    json: String
): List<MapFeatureRecord> {
    val root = SimpleJsonParser.parseObject(json)
    val featuresArray = root["features"].jsonArrayOrEmpty()
    return buildList {
        for (itemValue in featuresArray) {
            val item = itemValue.jsonObjectOrNull() ?: continue
            val featureId = item["id"].jsonStringOrNull().orEmpty()
            if (featureId.isBlank()) {
                continue
            }
            val kind = MapFeatureKind.fromWireName(item["kind"].jsonStringOrNull().orEmpty()) ?: continue
            val latitude = item["lat"].jsonDoubleOrNull() ?: Double.NaN
            val longitude = item["lon"].jsonDoubleOrNull() ?: Double.NaN
            if (!latitude.isFinite() || !longitude.isFinite()) {
                continue
            }
            val triggerRadiusMeters =
                item["triggerRadiusMeters"].jsonDoubleOrNull()?.toFloat() ?: 35f
            val exitRadiusMeters =
                item["exitRadiusMeters"].jsonDoubleOrNull()?.toFloat()
                    ?: defaultExitRadiusMeters(triggerRadiusMeters)
            val approachBearings = mutableListOf<Float>()
            val bearingArray = item["approachBearings"].jsonArrayOrEmpty()
            for (bearingValue in bearingArray) {
                val bearing = bearingValue.jsonDoubleOrNull() ?: Double.NaN
                if (bearing.isFinite()) {
                    approachBearings += normalizeBearingDegrees(bearing.toFloat())
                }
            }
            add(
                MapFeatureRecord(
                    id = featureId,
                    kind = kind,
                    point = GeoPoint(latitude = latitude, longitude = longitude),
                    triggerRadiusMeters = triggerRadiusMeters,
                    exitRadiusMeters = exitRadiusMeters,
                    approachBearings = approachBearings,
                    regionTileId = item["regionTileId"].jsonStringOrDefault(tileId),
                    datasetVersion = datasetVersion
                )
            )
        }
    }
}
