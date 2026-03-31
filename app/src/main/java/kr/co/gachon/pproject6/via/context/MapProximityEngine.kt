package kr.co.gachon.pproject6.via.context

class MapProximityEngine(
    private val maxAcceptedAccuracyMeters: Float = 25f,
    private val consecutiveFixesRequired: Int = 2,
    private val headingTieDistanceMeters: Float = 10f,
    private val switchDistanceAdvantageMeters: Float = 8f
) {
    private var activeMatchId: String? = null
    private var pendingCandidateId: String? = null
    private var pendingCandidateCount: Int = 0
    private var lastSnapshot: MapProximitySnapshot = MapProximitySnapshot()

    fun update(
        point: GeoPoint,
        accuracyMeters: Float?,
        headingDegrees: Float?,
        features: List<MapFeatureRecord>,
        datasetVersion: String?,
        usedRemoteData: Boolean
    ): MapProximitySnapshot {
        if (accuracyMeters != null && accuracyMeters > maxAcceptedAccuracyMeters) {
            clearPending()
            lastSnapshot = lastSnapshot.withDatasetMetadata(datasetVersion, usedRemoteData)
            return lastSnapshot
        }

        val candidates =
            features.map { feature ->
                MapCandidate(
                    feature = feature,
                    distanceMeters = haversineDistanceMeters(point, feature.point),
                    headingDeltaDegrees = calculateHeadingDelta(feature, headingDegrees)
                )
            }

        val activeCandidate =
            activeMatchId?.let { activeId -> candidates.firstOrNull { it.feature.id == activeId } }
        val entryCandidates = candidates.filter { it.distanceMeters <= it.feature.triggerRadiusMeters }
        val selectedCandidate =
            if (entryCandidates.isEmpty()) {
                null
            } else {
                selectCandidate(entryCandidates, headingDegrees)
            }
        val retainedActiveCandidate =
            activeCandidate?.takeIf { it.distanceMeters <= it.feature.exitRadiusMeters }

        if (retainedActiveCandidate != null) {
            if (selectedCandidate != null && shouldSwitchActiveMatch(retainedActiveCandidate, selectedCandidate)) {
                lastSnapshot =
                    promoteCandidate(
                        candidate = selectedCandidate,
                        datasetVersion = datasetVersion,
                        usedRemoteData = usedRemoteData,
                        fallbackSnapshot = retainedActiveCandidate.toSnapshot(datasetVersion, usedRemoteData)
                    )
                return lastSnapshot
            }

            clearPending()
            lastSnapshot = retainedActiveCandidate.toSnapshot(datasetVersion, usedRemoteData)
            return lastSnapshot
        }
        clearActive()

        if (entryCandidates.isEmpty()) {
            clearPending()
            lastSnapshot =
                MapProximitySnapshot(
                    datasetVersion = datasetVersion ?: lastSnapshot.datasetVersion,
                    usedRemoteData = usedRemoteData
                )
            return lastSnapshot
        }

        lastSnapshot =
            promoteCandidate(
                candidate = selectedCandidate!!,
                datasetVersion = datasetVersion,
                usedRemoteData = usedRemoteData,
                fallbackSnapshot =
                    MapProximitySnapshot(
                        datasetVersion = datasetVersion ?: lastSnapshot.datasetVersion,
                        usedRemoteData = usedRemoteData
                    )
            )
        return lastSnapshot
    }

    fun reset() {
        clearActive()
        clearPending()
        lastSnapshot = MapProximitySnapshot(datasetVersion = lastSnapshot.datasetVersion)
    }

    private fun selectCandidate(
        candidates: List<MapCandidate>,
        headingDegrees: Float?
    ): MapCandidate {
        val nearest = candidates.minByOrNull { it.distanceMeters } ?: error("candidates must not be empty")
        if (headingDegrees == null) {
            return nearest
        }

        val nearbyCandidates =
            candidates.filter { it.distanceMeters <= nearest.distanceMeters + headingTieDistanceMeters }
        val candidatesWithHeading = nearbyCandidates.filter { it.headingDeltaDegrees != null }
        return candidatesWithHeading.minWithOrNull(
            compareBy<MapCandidate> { it.headingDeltaDegrees ?: Float.MAX_VALUE }
                .thenBy { it.distanceMeters }
        ) ?: nearest
    }

    private fun calculateHeadingDelta(
        feature: MapFeatureRecord,
        headingDegrees: Float?
    ): Float? {
        if (headingDegrees == null || feature.approachBearings.isEmpty()) {
            return null
        }
        return feature.approachBearings.minOfOrNull {
            angularDifferenceDegrees(it, headingDegrees)
        }
    }

    private fun shouldSwitchActiveMatch(
        activeCandidate: MapCandidate,
        selectedCandidate: MapCandidate
    ): Boolean {
        if (selectedCandidate.feature.id == activeCandidate.feature.id) {
            return false
        }
        if (selectedCandidate.distanceMeters + switchDistanceAdvantageMeters <= activeCandidate.distanceMeters) {
            return true
        }
        return activeCandidate.distanceMeters > activeCandidate.feature.triggerRadiusMeters &&
            selectedCandidate.distanceMeters <= selectedCandidate.feature.triggerRadiusMeters
    }

    private fun promoteCandidate(
        candidate: MapCandidate,
        datasetVersion: String?,
        usedRemoteData: Boolean,
        fallbackSnapshot: MapProximitySnapshot
    ): MapProximitySnapshot {
        if (candidate.feature.id == pendingCandidateId) {
            pendingCandidateCount += 1
        } else {
            pendingCandidateId = candidate.feature.id
            pendingCandidateCount = 1
        }

        if (pendingCandidateCount < consecutiveFixesRequired) {
            return fallbackSnapshot.withDatasetMetadata(datasetVersion, usedRemoteData)
        }

        activeMatchId = candidate.feature.id
        clearPending()
        return candidate.toSnapshot(datasetVersion, usedRemoteData)
    }

    private fun clearPending() {
        pendingCandidateId = null
        pendingCandidateCount = 0
    }

    private fun clearActive() {
        activeMatchId = null
    }

    private data class MapCandidate(
        val feature: MapFeatureRecord,
        val distanceMeters: Float,
        val headingDeltaDegrees: Float?
    ) {
        fun toSnapshot(
            datasetVersion: String?,
            usedRemoteData: Boolean
        ): MapProximitySnapshot {
            return MapProximitySnapshot(
                isNearKnownFeature = true,
                matchedFeatureId = feature.id,
                matchedKind = feature.kind,
                matchedLatitude = feature.point.latitude,
                matchedLongitude = feature.point.longitude,
                distanceMeters = distanceMeters,
                datasetVersion = datasetVersion ?: feature.datasetVersion,
                usedRemoteData = usedRemoteData
            )
        }
    }
}

private fun MapProximitySnapshot.withDatasetMetadata(
    datasetVersion: String?,
    usedRemoteData: Boolean
): MapProximitySnapshot {
    return copy(
        datasetVersion = datasetVersion ?: this.datasetVersion,
        usedRemoteData = this.usedRemoteData || usedRemoteData
    )
}
