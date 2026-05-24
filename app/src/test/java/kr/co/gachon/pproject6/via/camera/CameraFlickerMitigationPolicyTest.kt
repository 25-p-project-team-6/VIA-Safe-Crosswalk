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
    fun targetFpsRangePrefersExactTwentyFpsWhenAvailable() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(15, 30),
                    CameraFpsRange(30, 30),
                    CameraFpsRange(20, 20),
                    CameraFpsRange(15, 20)
                )
            )

        assertEquals(CameraFpsRange(20, 20), chosen)
    }

    @Test
    fun targetFpsRangePrefersNearbyFixedRangeOverVariableFifteenToTwenty() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(15, 30),
                    CameraFpsRange(30, 30),
                    CameraFpsRange(24, 24),
                    CameraFpsRange(15, 20),
                    CameraFpsRange(15, 15)
                )
            )

        assertEquals(CameraFpsRange(24, 24), chosen)
    }

    @Test
    fun targetFpsRangeFallsBackToBestVariableRangeWhenNoNearbyFixedRangeExists() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(15, 60),
                    CameraFpsRange(15, 30),
                    CameraFpsRange(15, 20),
                    CameraFpsRange(15, 15)
                )
            )

        assertEquals(CameraFpsRange(15, 30), chosen)
    }

    @Test
    fun targetFpsRangeDoesNotForceFixedThirtyWhenNoUsableFallbackExists() {
        val chosen =
            CameraFlickerMitigationPolicy.chooseTargetFpsRange(
                listOf(
                    CameraFpsRange(30, 60),
                    CameraFpsRange(60, 60)
                )
            )

        assertNull(chosen)
    }
}
