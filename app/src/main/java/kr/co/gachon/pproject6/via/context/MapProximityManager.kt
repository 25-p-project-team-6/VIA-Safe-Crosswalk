package kr.co.gachon.pproject6.via.context

import android.content.Context
import android.location.Location
import kr.co.gachon.pproject6.via.map.OsmNearbyCrossing
import kr.co.gachon.pproject6.via.map.OsmNearbyCrossingFetcher
import java.util.concurrent.atomic.AtomicBoolean

class MapProximityManager(
    context: Context,
    timeProvider: () -> Long = System::currentTimeMillis
) {
    private companion object {
        private const val OSM_RUNTIME_RADIUS_METERS = 120
        private const val OSM_RUNTIME_LIMIT = 32
        private const val OSM_REFRESH_MIN_DISTANCE_METERS = 20f
        private const val OSM_REFRESH_MAX_AGE_MS = 60_000L
    }

    private val tileStore = MapTileStore(context, timeProvider = timeProvider)
    private val engine = MapProximityEngine()
    private val osmFetcher = OsmNearbyCrossingFetcher(context)
    private val now = timeProvider
    private val osmRefreshInFlight = AtomicBoolean(false)
    @Volatile private var latestClusters: List<CrosswalkCluster> = emptyList()
    @Volatile private var latestBundledFeatures: List<MapFeatureRecord> = emptyList()
    @Volatile private var latestOsmFeatures: List<MapFeatureRecord> = emptyList()
    @Volatile private var lastPoint: GeoPoint? = null
    @Volatile private var lastAccuracyMeters: Float? = null
    @Volatile private var lastHeadingDegrees: Float? = null
    @Volatile private var lastDatasetVersion: String? = null
    @Volatile private var lastUsedRemoteData: Boolean = false
    @Volatile private var lastOsmFetchPoint: GeoPoint? = null
    @Volatile private var lastOsmFetchAtMs: Long = Long.MIN_VALUE
    private var currentSnapshot =
        MapProximitySnapshot(
            datasetVersion = tileStore.currentDatasetVersion(),
            usedRemoteData = tileStore.hasRemoteDataset()
        )

    fun onLocation(
        location: Location,
        headingDegrees: Float?
    ) {
        val point = GeoPoint(latitude = location.latitude, longitude = location.longitude)
        lastPoint = point
        tileStore.scheduleRefreshAround(point)
        val tileLoadResult = tileStore.loadAround(point)
        val accuracyMeters = if (location.hasAccuracy()) location.accuracy else null
        lastAccuracyMeters = accuracyMeters
        lastHeadingDegrees = headingDegrees
        lastDatasetVersion = tileLoadResult.datasetVersion
        lastUsedRemoteData = tileLoadResult.usedRemoteData
        latestBundledFeatures = tileLoadResult.features
        currentSnapshot = updateSnapshot()
        scheduleOsmRefreshIfNeeded(point)
    }

    fun snapshot(): MapProximitySnapshot = currentSnapshot

    fun reset() {
        engine.reset()
        latestBundledFeatures = emptyList()
        latestOsmFeatures = emptyList()
        lastPoint = null
        lastAccuracyMeters = null
        lastHeadingDegrees = null
        lastDatasetVersion = null
        lastUsedRemoteData = false
        lastOsmFetchPoint = null
        lastOsmFetchAtMs = Long.MIN_VALUE
        currentSnapshot =
            MapProximitySnapshot(
                datasetVersion = tileStore.currentDatasetVersion(),
                usedRemoteData = tileStore.hasRemoteDataset()
            )
    }

    @Synchronized
    private fun updateSnapshot(): MapProximitySnapshot {
        val point = lastPoint ?: return currentSnapshot
        latestClusters = buildCrosswalkClusters(latestBundledFeatures + latestOsmFeatures)
        currentSnapshot =
            engine.update(
                point = point,
                accuracyMeters = lastAccuracyMeters,
                headingDegrees = lastHeadingDegrees,
                clusters = latestClusters,
                datasetVersion = datasetLabel(),
                usedRemoteData = lastUsedRemoteData
            )
        return currentSnapshot
    }

    private fun scheduleOsmRefreshIfNeeded(point: GeoPoint) {
        val lastFetchPoint = lastOsmFetchPoint
        val refreshDue =
            lastFetchPoint == null ||
                haversineDistanceMeters(lastFetchPoint, point) >= OSM_REFRESH_MIN_DISTANCE_METERS ||
                now() - lastOsmFetchAtMs >= OSM_REFRESH_MAX_AGE_MS
        if (!refreshDue || !osmRefreshInFlight.compareAndSet(false, true)) {
            return
        }
        Thread(
            {
                try {
                    val fetched =
                        osmFetcher.fetchNearby(
                            point = point,
                            radiusMeters = OSM_RUNTIME_RADIUS_METERS,
                            limit = OSM_RUNTIME_LIMIT
                        )
                    val shouldKeepPrevious =
                        fetched.isEmpty() &&
                            latestOsmFeatures.isNotEmpty() &&
                            lastOsmFetchPoint != null &&
                            haversineDistanceMeters(lastOsmFetchPoint!!, point) <= OSM_RUNTIME_RADIUS_METERS
                    if (!shouldKeepPrevious) {
                        latestOsmFeatures = fetched.map(::toOsmFeatureRecord)
                    }
                    lastOsmFetchPoint = point
                    lastOsmFetchAtMs = now()
                    synchronized(this) {
                        updateSnapshot()
                    }
                } finally {
                    osmRefreshInFlight.set(false)
                }
            },
            "map-proximity-osm-refresh"
        ).apply {
            isDaemon = true
            start()
        }
    }

    private fun datasetLabel(): String? {
        return when {
            latestOsmFeatures.isNotEmpty() && !lastDatasetVersion.isNullOrBlank() -> "${lastDatasetVersion}+osm"
            latestOsmFeatures.isNotEmpty() -> "osm-live"
            else -> lastDatasetVersion
        }
    }

    private fun toOsmFeatureRecord(
        crossing: OsmNearbyCrossing
    ): MapFeatureRecord {
        return MapFeatureRecord(
            id = crossing.id,
            kind = if (crossing.signalControlled) MapFeatureKind.PED_SIGNAL else MapFeatureKind.CROSSWALK,
            point = GeoPoint(crossing.lat, crossing.lon),
            triggerRadiusMeters = 35f,
            exitRadiusMeters = 55f,
            approachBearings = emptyList(),
            regionTileId = MapTileGrid.tileIdFor(crossing.lat, crossing.lon),
            datasetVersion = "osm-live",
            source = MapFeatureSource.OSM
        )
    }
}
