package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertTrue
import org.junit.Test

class ObjectTrackerScoringTest {
    @Test
    fun largerOffCenterSignalCanBeatSmallerCenteredSignal() {
        val smallerCentered =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.010f,
                centerDistance = 0.03f
            )
        val largerOffCenter =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.020f,
                centerDistance = 0.28f
            )

        assertTrue(largerOffCenter > smallerCentered)
    }

    @Test
    fun centerStillWinsWhenSizesAreNearlyEqual() {
        val centered =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.012f,
                centerDistance = 0.04f
            )
        val slightlyLargerOffCenter =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.013f,
                centerDistance = 0.30f
            )

        assertTrue(centered > slightlyLargerOffCenter)
    }
}
