package kr.co.gachon.pproject6.via.map

import android.util.Log
import kr.co.gachon.pproject6.via.context.GeoPoint
import kr.co.gachon.pproject6.via.context.SimpleJsonParser
import kr.co.gachon.pproject6.via.context.haversineDistanceMeters
import kr.co.gachon.pproject6.via.context.jsonArrayOrEmpty
import kr.co.gachon.pproject6.via.context.jsonDoubleOrNull
import kr.co.gachon.pproject6.via.context.jsonObjectOrNull
import kr.co.gachon.pproject6.via.context.jsonStringOrNull
import java.io.BufferedWriter
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
    private val endpointUrl: String = "https://overpass-api.de/api/interpreter"
) {
    fun fetchNearby(
        point: GeoPoint,
        radiusMeters: Int = 400,
        limit: Int = 12
    ): List<OsmNearbyCrossing> {
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
        } catch (error: Exception) {
            Log.w("OsmNearby", "Overpass fetch failed", error)
            emptyList()
        } finally {
            connection.disconnect()
        }
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
