package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrossingSupportSnapshotTest {
    @Test
    fun walkContinuationBecomesTrueWhenMotionExists() {
        val snapshot = CrossingSupportSnapshot(hasRecentGyroMotion = true)

        assertTrue(snapshot.supportsWalkContinuation)
        assertFalse(snapshot.supportsNextCrosswalkTransition)
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
