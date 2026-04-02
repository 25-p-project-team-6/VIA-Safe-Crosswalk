package kr.co.gachon.pproject6.via.context

class MapProximityEngine(
    private val maxAcceptedAccuracyMeters: Float = 25f,
    private val consecutiveFixesRequired: Int = 2,
    private val headingTieDistanceMeters: Float = 10f,
    private val sameCrossingTransitionDistanceMeters: Float = 14f
) {
    private var activeClusterId: String? = null
    private var pendingClusterId: String? = null
    private var pendingCandidateCount: Int = 0
    private var lastSnapshot: MapProximitySnapshot = MapProximitySnapshot()

    fun update(
        point: GeoPoint,
        accuracyMeters: Float?,
        headingDegrees: Float?,
        clusters: List<CrosswalkCluster>,
        datasetVersion: String?,
        usedRemoteData: Boolean
    ): MapProximitySnapshot {
        if (accuracyMeters != null && accuracyMeters > maxAcceptedAccuracyMeters) {
            clearPending()
            lastSnapshot = lastSnapshot.withDatasetMetadata(datasetVersion, usedRemoteData)
            return lastSnapshot
        }

        val candidates =
            clusters.map { cluster ->
                ClusterCandidate(
                    cluster = cluster,
                    distanceMeters = haversineDistanceMeters(point, cluster.centerPoint),
                    headingDeltaDegrees = calculateHeadingDelta(cluster, headingDegrees)
                )
            }

        val activeCandidate =
            activeClusterId?.let { activeId -> candidates.firstOrNull { it.cluster.clusterId == activeId } }
        val entryCandidates = candidates.filter { it.distanceMeters <= it.cluster.triggerRadiusMeters }
        val selectedCandidate =
            if (entryCandidates.isEmpty()) {
                null
            } else {
                selectCandidate(entryCandidates, headingDegrees)
            }
        val retainedActiveCandidate =
            activeCandidate?.takeIf { it.distanceMeters <= it.cluster.exitRadiusMeters }

        if (retainedActiveCandidate != null) {
            if (selectedCandidate != null && shouldSwitchActiveMatch(retainedActiveCandidate, selectedCandidate)) {
                val transitionKind = transitionKind(retainedActiveCandidate.cluster, selectedCandidate.cluster)
                lastSnapshot =
                    promoteCandidate(
                        candidate = selectedCandidate,
                        datasetVersion = datasetVersion,
                        usedRemoteData = usedRemoteData,
                        fallbackSnapshot =
                            retainedActiveCandidate.toSnapshot(
                                datasetVersion = datasetVersion,
                                usedRemoteData = usedRemoteData,
                                transitionKind = MapClusterTransitionKind.NONE,
                                transitionDistanceMeters = null
                            ),
                        transitionKind = transitionKind,
                        transitionDistanceMeters =
                            haversineDistanceMeters(
                                retainedActiveCandidate.cluster.centerPoint,
                                selectedCandidate.cluster.centerPoint
                            )
                    )
                return lastSnapshot
            }

            clearPending()
            lastSnapshot =
                retainedActiveCandidate.toSnapshot(
                    datasetVersion = datasetVersion,
                    usedRemoteData = usedRemoteData,
                    transitionKind = MapClusterTransitionKind.NONE,
                    transitionDistanceMeters = null
                )
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
                    ),
                transitionKind = MapClusterTransitionKind.NONE,
                transitionDistanceMeters = null
            )
        return lastSnapshot
    }

    fun reset() {
        clearActive()
        clearPending()
        lastSnapshot = MapProximitySnapshot(datasetVersion = lastSnapshot.datasetVersion)
    }

    private fun selectCandidate(
        candidates: List<ClusterCandidate>,
        headingDegrees: Float?
    ): ClusterCandidate {
        val nearest = candidates.minByOrNull { it.distanceMeters } ?: error("candidates must not be empty")
        if (headingDegrees == null) {
            return nearest
        }

        val nearbyCandidates =
            candidates.filter { it.distanceMeters <= nearest.distanceMeters + headingTieDistanceMeters }
        val candidatesWithHeading = nearbyCandidates.filter { it.headingDeltaDegrees != null }
        return candidatesWithHeading.minWithOrNull(
            compareBy<ClusterCandidate> { it.headingDeltaDegrees ?: Float.MAX_VALUE }
                .thenByDescending { if (it.cluster.hasPedSignal) 1 else 0 }
                .thenByDescending { clusterSourcePriority(it.cluster.source) }
                .thenBy { it.distanceMeters }
        ) ?: nearest
    }

    private fun calculateHeadingDelta(
        cluster: CrosswalkCluster,
        headingDegrees: Float?
    ): Float? {
        if (headingDegrees == null || cluster.approachBearings.isEmpty()) {
            return null
        }
        return cluster.approachBearings.minOfOrNull {
            angularDifferenceDegrees(it, headingDegrees)
        }
    }

    private fun shouldSwitchActiveMatch(
        activeCandidate: ClusterCandidate,
        selectedCandidate: ClusterCandidate
    ): Boolean {
        if (selectedCandidate.cluster.clusterId == activeCandidate.cluster.clusterId) {
            return false
        }
        return selectedCandidate.distanceMeters < activeCandidate.distanceMeters
    }

    private fun promoteCandidate(
        candidate: ClusterCandidate,
        datasetVersion: String?,
        usedRemoteData: Boolean,
        fallbackSnapshot: MapProximitySnapshot,
        transitionKind: MapClusterTransitionKind,
        transitionDistanceMeters: Float?
    ): MapProximitySnapshot {
        if (candidate.cluster.clusterId == pendingClusterId) {
            pendingCandidateCount += 1
        } else {
            pendingClusterId = candidate.cluster.clusterId
            pendingCandidateCount = 1
        }

        if (pendingCandidateCount < consecutiveFixesRequired) {
            return fallbackSnapshot.withDatasetMetadata(datasetVersion, usedRemoteData)
        }

        activeClusterId = candidate.cluster.clusterId
        clearPending()
        return candidate.toSnapshot(
            datasetVersion = datasetVersion,
            usedRemoteData = usedRemoteData,
            transitionKind = transitionKind,
            transitionDistanceMeters = transitionDistanceMeters
        )
    }

    private fun transitionKind(
        previous: CrosswalkCluster,
        current: CrosswalkCluster
    ): MapClusterTransitionKind {
        val distanceMeters = haversineDistanceMeters(previous.centerPoint, current.centerPoint)
        return if (distanceMeters <= sameCrossingTransitionDistanceMeters) {
            MapClusterTransitionKind.SAME_CROSSING
        } else {
            MapClusterTransitionKind.NEW_CROSSING
        }
    }

    private fun clearPending() {
        pendingClusterId = null
        pendingCandidateCount = 0
    }

    private fun clearActive() {
        activeClusterId = null
    }

    private data class ClusterCandidate(
        val cluster: CrosswalkCluster,
        val distanceMeters: Float,
        val headingDeltaDegrees: Float?
    ) {
        fun toSnapshot(
            datasetVersion: String?,
            usedRemoteData: Boolean,
            transitionKind: MapClusterTransitionKind,
            transitionDistanceMeters: Float?
        ): MapProximitySnapshot {
            return MapProximitySnapshot(
                isNearKnownFeature = true,
                matchedFeatureId = cluster.clusterId,
                matchedKind = cluster.kind,
                matchedSource = cluster.source,
                matchedLatitude = cluster.preferredAnchorPoint.latitude,
                matchedLongitude = cluster.preferredAnchorPoint.longitude,
                distanceMeters = distanceMeters,
                datasetVersion = datasetVersion ?: cluster.datasetVersion,
                usedRemoteData = usedRemoteData,
                matchedClusterId = cluster.clusterId,
                matchedAnchorId = cluster.preferredAnchorId,
                matchedMemberCount = cluster.memberCount,
                matchedClusterSpanMeters = cluster.spanMeters,
                matchedHasPedSignal = cluster.hasPedSignal,
                clusterTransitionKind = transitionKind,
                clusterTransitionDistanceMeters = transitionDistanceMeters
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

private fun clusterSourcePriority(
    source: MapFeatureSource
): Int {
    return when (source) {
        MapFeatureSource.HYBRID -> 3
        MapFeatureSource.BUNDLED -> 2
        MapFeatureSource.OSM -> 1
    }
}
