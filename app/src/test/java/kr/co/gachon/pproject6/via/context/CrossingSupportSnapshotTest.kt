package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrossingSupportSnapshotTest {
    @Test
    fun tiltFromUprightDetectsDownwardOrientation() {
        assertTrue(calculateTiltFromUprightDegrees(0f, -9.8f, 0f) < 1f)
        assertTrue(calculateTiltFromUprightDegrees(0f, 0f, -9.8f) > 80f)
        assertTrue(calculateTiltFromUprightDegrees(9.8f, 0f, 0f) < 1f)
    }

    @Test
    fun signedTiltDistinguishesDownwardAndUpwardDirections() {
        assertTrue(calculateSignedTiltFromUprightDegrees(0f, 0f, -9.8f) > 80f)
        assertTrue(calculateSignedTiltFromUprightDegrees(0f, 0f, 9.8f) < -80f)
    }

    @Test
    fun walkContinuationBecomesTrueWhenMotionExists() {
        val snapshot = CrossingSupportSnapshot(hasRecentGyroMotion = true)

        assertTrue(snapshot.supportsWalkContinuation)
    }

    @Test
    fun walkContinuationAlsoBecomesTrueWhenLookingDown() {
        val snapshot = CrossingSupportSnapshot(isLookingDown = true)

        assertTrue(snapshot.supportsWalkContinuation)
    }

    @Test
    fun rawCrossingWindowMetricsRemainSeparateFromWalkContinuationEvidence() {
        val snapshot = CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentLocationMovement = false,
            crossingWindowDistanceMeters = 8.5f,
            crossingWindowElapsedMs = 6_100L,
            nextCrosswalkDistanceThresholdMeters = 8.0f,
            nextCrosswalkMinActiveMs = 6_000L
        )

        assertFalse(snapshot.supportsWalkContinuation)
        assertTrue(snapshot.isCrossingWindowActive)
        assertTrue(snapshot.crossingWindowDistanceMeters >= snapshot.nextCrosswalkDistanceThresholdMeters)
        assertTrue(snapshot.crossingWindowElapsedMs >= snapshot.nextCrosswalkMinActiveMs)
    }

    @Test
    fun debugSummaryIncludesRawWindowFields() {
        val snapshot = CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentLocationMovement = true,
            crossingWindowDistanceMeters = 3.0f,
            crossingWindowElapsedMs = 10_000L
        )

        val summary = snapshot.toDebugSummary()
        assertTrue(summary.contains("window=true"))
        assertTrue(summary.contains("dist=3.0"))
        assertTrue(summary.contains("elapsed=10000ms"))
    }

    @Test
    fun debugSummaryAlsoIncludesMapProximityFields() {
        val snapshot = CrossingSupportSnapshot(
            mapProximitySnapshot = MapProximitySnapshot(
                isNearKnownFeature = true,
                matchedFeatureId = "crosswalk-a",
                matchedKind = MapFeatureKind.CROSSWALK,
                matchedSource = MapFeatureSource.BUNDLED,
                distanceMeters = 12.5f,
                datasetVersion = "bundled-v1"
            )
        )

        val summary = snapshot.toDebugSummary()
        assertTrue(summary.contains("mapNear=true"))
        assertTrue(summary.contains("mapKind=crosswalk"))
        assertTrue(summary.contains("mapSource=bundled"))
        assertTrue(summary.contains("mapId=crosswalk-a"))
        assertTrue(summary.contains("mapVer=bundled-v1"))
    }

    @Test
    fun debugSummaryAlsoIncludesSignedTiltAndGpsFixFields() {
        val snapshot = CrossingSupportSnapshot(
            isLookingDown = true,
            isLookingUp = false,
            currentTiltDegrees = 32f,
            currentSignedTiltDegrees = 32f,
            currentLocationLatitude = 37.450123,
            currentLocationLongitude = 127.128456,
            currentLocationAccuracyMeters = 5.5f,
            currentHeadingDegrees = 182f
        )

        val summary = snapshot.toDebugSummary()
        assertTrue(summary.contains("down=true"))
        assertTrue(summary.contains("up=false"))
        assertTrue(summary.contains("signedTilt=32"))
        assertTrue(summary.contains("lat=37.450123"))
        assertTrue(summary.contains("lon=127.128456"))
        assertTrue(summary.contains("acc=5.5m"))
        assertTrue(summary.contains("heading=182"))
    }
}
