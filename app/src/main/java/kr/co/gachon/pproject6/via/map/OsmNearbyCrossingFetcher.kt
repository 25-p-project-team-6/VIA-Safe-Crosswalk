package kr.co.gachon.pproject6.via.map

import android.content.Context
import android.util.Log
import kr.co.gachon.pproject6.via.context.GeoPoint
import kr.co.gachon.pproject6.via.context.SimpleJsonParser
import kr.co.gachon.pproject6.via.context.haversineDistanceMeters
import kr.co.gachon.pproject6.via.context.jsonArrayOrEmpty
import kr.co.gachon.pproject6.via.context.jsonDoubleOrNull
import kr.co.gachon.pproject6.via.context.jsonObjectOrNull
import kr.co.gachon.pproject6.via.context.jsonStringOrNull
import java.io.BufferedWriter
import java.io.File
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Locale

data class OsmNearbyCrossing(
    val id: String,
    val lat: Double,
    val lon: Double,
    val kind: String,
    val distanceMeters: Float,
    val signalControlled: Boolean,
    val geometry: List<GeoPoint> = emptyList()
)

class OsmNearbyCrossingFetcher(
    context: Context? = null,
    private val endpointUrl: String = "https://overpass-api.de/api/interpreter"
) {
    private val appContext = context?.applicationContext

    fun fetchNearby(
        point: GeoPoint,
        radiusMeters: Int = 400,
        limit: Int = 12
    ): List<OsmNearbyCrossing> {
        val cacheKey = buildCacheKey(point, radiusMeters, limit)
        readCached(cacheKey, point, maxAgeMs = FRESH_CACHE_MS)?.let { return it }
        val query =
            """
            [out:json][timeout:12];
            (
              node(around:$radiusMeters,${point.latitude},${point.longitude})["highway"="crossing"];
              node(around:$radiusMeters,${point.latitude},${point.longitude})["crossing"];
              node(around:$radiusMeters,${point.latitude},${point.longitude})["crossing:island"];
              node(around:$radiusMeters,${point.latitude},${point.longitude})["highway"="crossing"]["crossing:signals"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["highway"="footway"]["footway"="crossing"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["highway"="path"]["footway"="crossing"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["footway"="crossing"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["crossing"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["crossing:island"];
              relation(around:$radiusMeters,${point.latitude},${point.longitude})["type"="multipolygon"]["crossing"];
              node(around:$radiusMeters,${point.latitude},${point.longitude})["crossing"="traffic_signals"];
              way(around:$radiusMeters,${point.latitude},${point.longitude})["crossing"="traffic_signals"];
              relation(around:$radiusMeters,${point.latitude},${point.longitude})["type"="multipolygon"]["crossing"="traffic_signals"];
            );
            out geom tags;
            """.trimIndent()

        val connection = (URL(endpointUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 7_500
            readTimeout = 12_000
            doOutput = true
            setRequestProperty(
                "User-Agent",
                "VIA-DebugMap/1.0 (https://github.com/kinetic27)"
            )
            setRequestProperty("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
            setRequestProperty("Accept", "application/json")
        }

        return try {
            BufferedWriter(OutputStreamWriter(connection.outputStream, StandardCharsets.UTF_8)).use {
                it.write("data=" + urlEncode(query))
            }

            val statusCode = connection.responseCode
            val body =
                (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                    ?.bufferedReader(StandardCharsets.UTF_8)
                    ?.use { it.readText() }
                    .orEmpty()
            if (statusCode !in 200..299) {
                Log.w("OsmNearby", "Overpass HTTP $statusCode: $body")
                return emptyList()
            }

            parseOverpassCrossings(body, point)
                .sortedBy { it.distanceMeters }
                .take(limit)
                .also { writeCache(cacheKey, it) }
        } catch (error: Exception) {
            Log.w("OsmNearby", "Overpass fetch failed", error)
            readCached(cacheKey, point, maxAgeMs = STALE_CACHE_MS).orEmpty()
        } finally {
            connection.disconnect()
        }
    }

    private fun buildCacheKey(
        point: GeoPoint,
        radiusMeters: Int,
        limit: Int
    ): String {
        val latBucket = kotlin.math.round(point.latitude * 2000.0) / 2000.0
        val lonBucket = kotlin.math.round(point.longitude * 2000.0) / 2000.0
        return String.format(Locale.US, "r%04d_l%03d_%.4f_%.4f", radiusMeters, limit, latBucket, lonBucket)
    }

    private fun readCached(
        cacheKey: String,
        currentPoint: GeoPoint,
        maxAgeMs: Long
    ): List<OsmNearbyCrossing>? {
        val cacheFile = cacheFile(cacheKey) ?: return null
        if (!cacheFile.exists()) {
            return null
        }
        return runCatching {
            val lines = cacheFile.readLines(StandardCharsets.UTF_8)
            val timestamp = lines.firstOrNull()?.toLongOrNull() ?: return null
            if (System.currentTimeMillis() - timestamp > maxAgeMs) {
                return null
            }
            val json = lines.drop(1).joinToString("\n")
            val root = SimpleJsonParser.parseObject(json)
            root["items"].jsonArrayOrEmpty().mapNotNull { value ->
                val item = value.jsonObjectOrNull() ?: return@mapNotNull null
                val id = item["id"].jsonStringOrNull() ?: return@mapNotNull null
                val lat = item["lat"].jsonDoubleOrNull() ?: return@mapNotNull null
                val lon = item["lon"].jsonDoubleOrNull() ?: return@mapNotNull null
                val kind = item["kind"].jsonStringOrNull() ?: "osm_crossing"
                val distance = haversineDistanceMeters(currentPoint, GeoPoint(lat, lon))
                val signalControlled = (item["signalControlled"] as? Boolean) ?: false
                val geometry =
                    item["geometry"].jsonArrayOrEmpty().mapNotNull { geoValue ->
                        val geo = geoValue.jsonObjectOrNull() ?: return@mapNotNull null
                        val geoLat = geo["lat"].jsonDoubleOrNull() ?: return@mapNotNull null
                        val geoLon = geo["lon"].jsonDoubleOrNull() ?: return@mapNotNull null
                        GeoPoint(geoLat, geoLon)
                    }
                OsmNearbyCrossing(
                    id = id,
                    lat = lat,
                    lon = lon,
                    kind = kind,
                    distanceMeters = distance,
                    signalControlled = signalControlled,
                    geometry = geometry
                )
            }
        }.getOrNull()
    }

    private fun writeCache(
        cacheKey: String,
        items: List<OsmNearbyCrossing>
    ) {
        val cacheFile = cacheFile(cacheKey) ?: return
        runCatching {
            cacheFile.parentFile?.mkdirs()
            val jsonItems =
                items.joinToString(prefix = "[", postfix = "]") { item ->
                    val geometry =
                        item.geometry.joinToString(prefix = "[", postfix = "]") { point ->
                            """{"lat":${point.latitude},"lon":${point.longitude}}"""
                        }
                    """{"id":${jsonString(item.id)},"lat":${item.lat},"lon":${item.lon},"kind":${jsonString(item.kind)},"distanceMeters":${item.distanceMeters},"signalControlled":${item.signalControlled},"geometry":$geometry}"""
                }
            cacheFile.writeText(
                buildString {
                    appendLine(System.currentTimeMillis().toString())
                    append("""{"items":$jsonItems}""")
                },
                StandardCharsets.UTF_8
            )
        }
    }

    private fun cacheFile(cacheKey: String): File? {
        val context = appContext ?: return null
        return MapDebugCacheManager.osmCacheDir(context).resolve("$cacheKey.json")
    }

    companion object {
        private const val FRESH_CACHE_MS = 2 * 60 * 1000L
        private const val STALE_CACHE_MS = 20 * 60 * 1000L
    }
}

internal fun parseOverpassCrossings(
    json: String,
    currentPoint: GeoPoint
): List<OsmNearbyCrossing> {
    val root = SimpleJsonParser.parseObject(json)
    val deduped = linkedMapOf<String, OsmNearbyCrossing>()

    for (elementValue in root["elements"].jsonArrayOrEmpty()) {
        val element = elementValue.jsonObjectOrNull() ?: continue
        val type = element["type"].jsonStringOrNull().orEmpty()
        val id = element["id"].jsonDoubleOrNull()?.toLong()?.toString() ?: continue
        val tags = element["tags"].jsonObjectOrNull().orEmpty()
        val lat =
            element["lat"].jsonDoubleOrNull()
                ?: element["center"].jsonObjectOrNull()?.get("lat").jsonDoubleOrNull()
                ?: continue
        val lon =
            element["lon"].jsonDoubleOrNull()
                ?: element["center"].jsonObjectOrNull()?.get("lon").jsonDoubleOrNull()
                ?: continue

        val crossingTag = tags["crossing"].jsonStringOrNull()
        val crossingSignalsTag = tags["crossing:signals"].jsonStringOrNull()
        val crossingIslandTag = tags["crossing:island"].jsonStringOrNull()
        val highwayTag = tags["highway"].jsonStringOrNull()
        val footwayTag = tags["footway"].jsonStringOrNull()
        val signalControlled =
            crossingTag == "traffic_signals" ||
                crossingSignalsTag == "yes" ||
                crossingSignalsTag == "traffic_signals"
        val kind =
            when {
                signalControlled -> "osm_signal_crossing"
                highwayTag == "crossing" -> "osm_crossing"
                footwayTag == "crossing" -> "osm_crossing_way"
                crossingIslandTag != null -> "osm_crossing_island"
                else -> "osm_crossing"
            }

        val crossing = OsmNearbyCrossing(
            id = "$type:$id",
            lat = lat,
            lon = lon,
            kind = kind,
            distanceMeters = haversineDistanceMeters(currentPoint, GeoPoint(lat, lon)),
            signalControlled = signalControlled,
            geometry = parseGeometryPoints(element)
        )
        val bucketKey =
            String.format(
                Locale.US,
                "%.5f,%.5f",
                lat,
                lon
            )
        val previous = deduped[bucketKey]
        deduped[bucketKey] =
            when {
                previous == null -> crossing
                crossing.signalControlled && !previous.signalControlled -> crossing
                crossing.geometry.size > previous.geometry.size -> crossing
                crossing.distanceMeters < previous.distanceMeters -> crossing
                else -> previous
            }
    }

    return deduped.values.toList()
}

private fun parseGeometryPoints(
    element: Map<String, Any?>
): List<GeoPoint> {
    return element["geometry"].jsonArrayOrEmpty()
        .mapNotNull { pointValue ->
            val point = pointValue.jsonObjectOrNull() ?: return@mapNotNull null
            val lat = point["lat"].jsonDoubleOrNull() ?: return@mapNotNull null
            val lon = point["lon"].jsonDoubleOrNull() ?: return@mapNotNull null
            GeoPoint(lat, lon)
        }
}

private fun urlEncode(value: String): String {
    return java.net.URLEncoder.encode(value, StandardCharsets.UTF_8.name())
}

private fun jsonString(value: String): String {
    return buildString {
        append('"')
        value.forEach { ch ->
            when (ch) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(ch)
            }
        }
        append('"')
    }
}
