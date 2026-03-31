package kr.co.gachon.pproject6.via.context

import android.content.Context
import android.location.Location

class MapProximityManager(
    context: Context,
    timeProvider: () -> Long = System::currentTimeMillis
) {
    private val tileStore = MapTileStore(context, timeProvider = timeProvider)
    private val engine = MapProximityEngine()
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
        tileStore.scheduleRefreshAround(point)
        val tileLoadResult = tileStore.loadAround(point)
        val accuracyMeters = if (location.hasAccuracy()) location.accuracy else null
        currentSnapshot =
            engine.update(
                point = point,
                accuracyMeters = accuracyMeters,
                headingDegrees = headingDegrees,
                features = tileLoadResult.features,
                datasetVersion = tileLoadResult.datasetVersion,
                usedRemoteData = tileLoadResult.usedRemoteData
            )
    }

    fun snapshot(): MapProximitySnapshot = currentSnapshot

    fun reset() {
        engine.reset()
        currentSnapshot =
            MapProximitySnapshot(
                datasetVersion = tileStore.currentDatasetVersion(),
                usedRemoteData = tileStore.hasRemoteDataset()
            )
    }
}
