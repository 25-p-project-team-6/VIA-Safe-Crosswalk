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
    fun walkContinuationBecomesTrueWhenMotionExists() {
        val snapshot = CrossingSupportSnapshot(hasRecentGyroMotion = true)

        assertTrue(snapshot.supportsWalkContinuation)
        assertFalse(snapshot.supportsNextCrosswalkTransition)
    }

    @Test
    fun walkContinuationAlsoBecomesTrueWhenLookingDown() {
        val snapshot = CrossingSupportSnapshot(isLookingDown = true)

        assertTrue(snapshot.supportsWalkContinuation)
    }

    @Test
    fun nextCrosswalkTransitionRequiresDistanceAndDurationWhileCrossing() {
        val snapshot = CrossingSupportSnapshot(
            isCrossingActive = true,
            hasRecentLocationMovement = true,
            distanceSinceCrossingStartMeters = 8.5f,
            crossingActiveDurationMs = 6_100L,
            nextCrosswalkDistanceThresholdMeters = 8.0f,
            nextCrosswalkMinActiveMs = 6_000L
        )

        assertTrue(snapshot.supportsNextCrosswalkTransition)
    }

    @Test
    fun nextCrosswalkTransitionStaysFalseWhenMovementIsTooShort() {
        val snapshot = CrossingSupportSnapshot(
            isCrossingActive = true,
            hasRecentLocationMovement = true,
            distanceSinceCrossingStartMeters = 3.0f,
            crossingActiveDurationMs = 10_000L,
            nextCrosswalkDistanceThresholdMeters = 8.0f,
            nextCrosswalkMinActiveMs = 6_000L
        )

        assertFalse(snapshot.supportsNextCrosswalkTransition)
    }
}
