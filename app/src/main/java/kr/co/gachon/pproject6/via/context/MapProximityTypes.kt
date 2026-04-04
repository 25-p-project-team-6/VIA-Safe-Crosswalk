package kr.co.gachon.pproject6.via.context

import java.util.Locale
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

data class GeoPoint(
    val latitude: Double,
    val longitude: Double
)

enum class MapFeatureKind(val wireName: String) {
    CROSSWALK("crosswalk"),
    PED_SIGNAL("ped_signal");

    companion object {
        fun fromWireName(value: String): MapFeatureKind? {
            return entries.firstOrNull { it.wireName.equals(value, ignoreCase = true) }
        }
    }
}

enum class MapFeatureSource(val wireName: String) {
    BUNDLED("bundled"),
    OSM("osm"),
    HYBRID("hybrid");
}

enum class MapClusterTransitionKind(val wireName: String) {
    NONE("none"),
    SAME_CROSSING("same_crossing"),
    NEW_CROSSING("new_crossing");
}

data class MapFeatureRecord(
    val id: String,
    val kind: MapFeatureKind,
    val point: GeoPoint,
    val triggerRadiusMeters: Float = 35f,
    val exitRadiusMeters: Float = 55f,
    val approachBearings: List<Float> = emptyList(),
    val regionTileId: String,
    val datasetVersion: String,
    val source: MapFeatureSource = MapFeatureSource.BUNDLED
)

data class CrosswalkClusterMember(
    val featureId: String,
    val kind: MapFeatureKind,
    val source: MapFeatureSource,
    val point: GeoPoint,
    val approachBearings: List<Float>
)

data class CrosswalkCluster(
    val clusterId: String,
    val centerPoint: GeoPoint,
    val preferredAnchorId: String,
    val preferredAnchorPoint: GeoPoint,
    val kind: MapFeatureKind,
    val source: MapFeatureSource,
    val memberIds: List<String>,
    val memberCount: Int,
    val spanMeters: Float,
    val hasPedSignal: Boolean,
    val triggerRadiusMeters: Float,
    val exitRadiusMeters: Float,
    val approachBearings: List<Float>,
    val datasetVersion: String
)

data class MapProximitySnapshot(
    val isNearKnownFeature: Boolean = false,
    val matchedFeatureId: String? = null,
    val matchedKind: MapFeatureKind? = null,
    val matchedSource: MapFeatureSource? = null,
    val matchedLatitude: Double? = null,
    val matchedLongitude: Double? = null,
    val distanceMeters: Float? = null,
    val datasetVersion: String? = null,
    val usedRemoteData: Boolean = false,
    val matchedClusterId: String? = null,
    val matchedAnchorId: String? = null,
    val matchedMemberCount: Int = 0,
    val matchedClusterSpanMeters: Float? = null,
    val matchedHasPedSignal: Boolean = false,
    val clusterTransitionKind: MapClusterTransitionKind = MapClusterTransitionKind.NONE,
    val clusterTransitionDistanceMeters: Float? = null
) {
    fun toDebugSummary(): String {
        val distanceSummary =
            distanceMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "n/a"
        val spanSummary =
            matchedClusterSpanMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "n/a"
        return buildString {
            append("mapNear=")
            append(isNearKnownFeature)
            append(", mapKind=")
            append(matchedKind?.wireName ?: "none")
            append(", mapSource=")
            append(matchedSource?.wireName ?: "none")
            append(", mapDist=")
            append(distanceSummary)
            append(", mapId=")
            append(matchedFeatureId ?: "none")
            append(", mapCluster=")
            append(matchedClusterId ?: "none")
            append(", mapMembers=")
            append(matchedMemberCount)
            append(", mapSpan=")
            append(spanSummary)
            append(", mapTransition=")
            append(clusterTransitionKind.wireName)
            append(", mapVer=")
            append(datasetVersion ?: "none")
            if (usedRemoteData) {
                append(", mapRemote=true")
            }
        }
    }
}

internal data class MapTileLoadResult(
    val features: List<MapFeatureRecord>,
    val datasetVersion: String?,
    val usedRemoteData: Boolean
)

internal data class MapDatasetManifest(
    val version: String,
    val tiles: Map<String, MapTileDescriptor>
)

internal data class MapTileDescriptor(
    val tileId: String,
    val filePath: String,
    val checksum: String? = null,
    val downloadUrl: String? = null
)

internal object MapTileGrid {
    fun tileIdFor(point: GeoPoint): String {
        return tileIdFor(point.latitude, point.longitude)
    }

    fun tileIdFor(latitude: Double, longitude: Double): String {
        val latIndex = floor(latitude * 100.0).toInt()
        val lonIndex = floor(longitude * 100.0).toInt()
        return "${latIndex}_${lonIndex}"
    }

    fun neighboringTileIds(point: GeoPoint): Set<String> {
        val latIndex = floor(point.latitude * 100.0).toInt()
        val lonIndex = floor(point.longitude * 100.0).toInt()
        return buildSet {
            for (latOffset in -1..1) {
                for (lonOffset in -1..1) {
                    add("${latIndex + latOffset}_${lonIndex + lonOffset}")
                }
            }
        }
    }

    fun tileIdsForRadius(
        point: GeoPoint,
        radiusMeters: Int
    ): Set<String> {
        val latitudeDelta = radiusMeters / 111_320.0
        val longitudeDelta = radiusMeters / (111_320.0 * cos(Math.toRadians(point.latitude)).coerceAtLeast(1e-6))
        val minLatIndex = floor((point.latitude - latitudeDelta) * 100.0).toInt()
        val maxLatIndex = floor((point.latitude + latitudeDelta) * 100.0).toInt()
        val minLonIndex = floor((point.longitude - longitudeDelta) * 100.0).toInt()
        val maxLonIndex = floor((point.longitude + longitudeDelta) * 100.0).toInt()
        return buildSet {
            for (latIndex in minLatIndex..maxLatIndex) {
                for (lonIndex in minLonIndex..maxLonIndex) {
                    add("${latIndex}_${lonIndex}")
                }
            }
        }
    }
}

internal fun haversineDistanceMeters(a: GeoPoint, b: GeoPoint): Float {
    val earthRadiusMeters = 6_371_000.0
    val latitudeDelta = Math.toRadians(b.latitude - a.latitude)
    val longitudeDelta = Math.toRadians(b.longitude - a.longitude)
    val startLatitude = Math.toRadians(a.latitude)
    val endLatitude = Math.toRadians(b.latitude)

    val haversine =
        sin(latitudeDelta / 2.0).pow(2.0) +
            cos(startLatitude) * cos(endLatitude) * sin(longitudeDelta / 2.0).pow(2.0)
    val arc = 2.0 * asin(min(1.0, sqrt(haversine)))
    return (earthRadiusMeters * arc).toFloat()
}

internal fun angularDifferenceDegrees(a: Float, b: Float): Float {
    val normalizedA = normalizeBearingDegrees(a)
    val normalizedB = normalizeBearingDegrees(b)
    val difference = kotlin.math.abs(normalizedA - normalizedB)
    return min(difference, 360f - difference)
}

internal fun normalizeBearingDegrees(value: Float): Float {
    val normalized = value % 360f
    return if (normalized < 0f) normalized + 360f else normalized
}

internal fun defaultExitRadiusMeters(triggerRadiusMeters: Float): Float {
    return max(triggerRadiusMeters + 20f, 55f)
}

internal fun featurePriority(feature: MapFeatureRecord): Int {
    return when (feature.source) {
        MapFeatureSource.HYBRID -> 6
        MapFeatureSource.BUNDLED ->
            if (feature.kind == MapFeatureKind.PED_SIGNAL) 5 else 4
        MapFeatureSource.OSM ->
            if (feature.kind == MapFeatureKind.PED_SIGNAL) 3 else 2
    }
}

internal fun buildCrosswalkClusters(
    features: List<MapFeatureRecord>,
    mergeDistanceMeters: Float = 12f,
    maxClusterSpanMeters: Float = 22f
): List<CrosswalkCluster> {
    val accumulators = mutableListOf<MutableCrosswalkCluster>()
    val ordered = features.sortedWith(compareByDescending<MapFeatureRecord> { featurePriority(it) }.thenBy { it.id })
    ordered.forEach { feature ->
        val bestIndex =
            accumulators.withIndex()
                .filter { (_, cluster) -> cluster.canAccept(feature, mergeDistanceMeters, maxClusterSpanMeters) }
                .minByOrNull { (_, cluster) -> haversineDistanceMeters(cluster.centerPoint(), feature.point) }
                ?.index
        if (bestIndex == null) {
            accumulators += MutableCrosswalkCluster(feature)
        } else {
            accumulators[bestIndex].add(feature)
        }
    }
    return accumulators.map { it.build() }
}

private class MutableCrosswalkCluster(
    seed: MapFeatureRecord
) {
    private val members = mutableListOf(seed)

    fun add(feature: MapFeatureRecord) {
        members += feature
    }

    fun centerPoint(): GeoPoint {
        val lat = members.sumOf { it.point.latitude } / members.size
        val lon = members.sumOf { it.point.longitude } / members.size
        return GeoPoint(lat, lon)
    }

    fun canAccept(
        feature: MapFeatureRecord,
        mergeDistanceMeters: Float,
        maxClusterSpanMeters: Float
    ): Boolean {
        val closeEnough =
            members.any { existing ->
                haversineDistanceMeters(existing.point, feature.point) <= mergeDistanceMeters
            } || haversineDistanceMeters(centerPoint(), feature.point) <= mergeDistanceMeters
        if (!closeEnough) {
            return false
        }
        return clusterSpanMeters(members + feature) <= maxClusterSpanMeters
    }

    fun build(): CrosswalkCluster {
        val sortedMembers =
            members.sortedWith(
                compareByDescending<MapFeatureRecord> { if (it.kind == MapFeatureKind.PED_SIGNAL) 1 else 0 }
                    .thenByDescending { featurePriority(it) }
                    .thenBy { it.id }
            )
        val canonical = sortedMembers.first()
        val preferred = sortedMembers.first()
        val memberSources = members.map { it.source }.distinct()
        val source =
            if (memberSources.size == 1) {
                memberSources.first()
            } else {
                MapFeatureSource.HYBRID
            }
        val hasPedSignal = members.any { it.kind == MapFeatureKind.PED_SIGNAL }
        return CrosswalkCluster(
            clusterId = canonical.id,
            centerPoint = centerPoint(),
            preferredAnchorId = preferred.id,
            preferredAnchorPoint = preferred.point,
            kind = if (hasPedSignal) MapFeatureKind.PED_SIGNAL else canonical.kind,
            source = source,
            memberIds = members.map { it.id }.sorted(),
            memberCount = members.size,
            spanMeters = clusterSpanMeters(members),
            hasPedSignal = hasPedSignal,
            triggerRadiusMeters = members.maxOf { it.triggerRadiusMeters },
            exitRadiusMeters = members.maxOf { it.exitRadiusMeters },
            approachBearings = members.flatMap { it.approachBearings }.distinct(),
            datasetVersion = canonical.datasetVersion
        )
    }
}

private fun clusterSpanMeters(features: List<MapFeatureRecord>): Float {
    var maxDistance = 0f
    for (firstIndex in 0 until features.size) {
        for (secondIndex in firstIndex + 1 until features.size) {
            maxDistance = max(maxDistance, haversineDistanceMeters(features[firstIndex].point, features[secondIndex].point))
        }
    }
    return maxDistance
}
