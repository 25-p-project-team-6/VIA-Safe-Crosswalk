package kr.co.gachon.pproject6.via.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CameraFlickerMitigationPolicyTest {
    @Test
    fun antibandingPrefersExplicitSixtyHertzWhenAvailable() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseAntibandingMode(
                availableModes = intArrayOf(0, 1, 2, 3),
                preferredMode = 2,
                autoMode = 3,
                offMode = 0
            )

        assertEquals(2, chosen)
    }

    @Test
    fun antibandingFallsBackToAutoBeforeOtherNonOffModes() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseAntibandingMode(
                availableModes = intArrayOf(0, 1, 3),
                preferredMode = 2,
                autoMode = 3,
                offMode = 0
            )

        assertEquals(3, chosen)
    }

    @Test
    fun antibandingIgnoresOffOnlyCapabilities() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseAntibandingMode(
                availableModes = intArrayOf(0),
                preferredMode = 2,
                autoMode = 3,
                offMode = 0
            )

        assertNull(chosen)
    }

    @Test
    fun targetFpsRangePrefersStableThirtyFpsCap() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(15, 30),
                    CameraFpsRange(30, 30),
                    CameraFpsRange(24, 30)
                )
            )

        assertEquals(CameraFpsRange(30, 30), chosen)
    }

    @Test
    fun targetFpsRangeFallsBackToHighestRangeCappedAtThirty() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(15, 30),
                    CameraFpsRange(24, 30),
                    CameraFpsRange(15, 15)
                )
            )

        assertEquals(CameraFpsRange(24, 30), chosen)
    }

    @Test
    fun targetFpsRangeDoesNotChooseSixtyFpsRanges() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(30, 60),
                    CameraFpsRange(60, 60)
                )
            )

        assertNull(chosen)
    }
}
