package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceStateStabilizerTest {
    @Test
    fun waitNeedsTimeDebounceBeforeReplacingGo() {
        var now = 0L
        val stabilizer =
            GuidanceStateStabilizer(
                GuidanceStateStabilizerConfig(
                    goConfirmDurationMs = 250L,
                    stopConfirmDurationMs = 150L,
                    waitConfirmDurationMs = 350L,
                    cautionConfirmDurationMs = 400L,
                    goMinimumHoldMs = 500L
                ),
                timeProvider = { now }
            )

        stabilizer.stabilize(goSnapshot())
        now += 100L
        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        now += 500L
        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        now += 350L
        assertEquals(waitSnapshot(GuidanceBlockReason.NO_SIGNAL), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
    }

    @Test
    fun stopCanOverrideGoSoonerThanWait() {
        var now = 0L
        val stabilizer =
            GuidanceStateStabilizer(
                GuidanceStateStabilizerConfig(
                    goConfirmDurationMs = 250L,
                    stopConfirmDurationMs = 150L,
                    waitConfirmDurationMs = 350L,
                    cautionConfirmDurationMs = 400L,
                    goMinimumHoldMs = 500L
                ),
                timeProvider = { now }
            )

        stabilizer.stabilize(goSnapshot())
        now += 200L
        assertEquals(goSnapshot(), stabilizer.stabilize(stopSnapshot()))
        now += 150L
        assertEquals(stopSnapshot(), stabilizer.stabilize(stopSnapshot()))
    }

    @Test
    fun cautionNeedsLongerDebounceThanGo() {
        var now = 0L
        val stabilizer =
            GuidanceStateStabilizer(
                GuidanceStateStabilizerConfig(
                    goConfirmDurationMs = 250L,
                    stopConfirmDurationMs = 150L,
                    waitConfirmDurationMs = 350L,
                    cautionConfirmDurationMs = 400L,
                    goMinimumHoldMs = 500L
                ),
                timeProvider = { now }
            )

        stabilizer.stabilize(goSnapshot())
        now += 300L
        assertEquals(goSnapshot(), stabilizer.stabilize(goCautionSnapshot()))
        now += 400L
        assertEquals(goCautionSnapshot(), stabilizer.stabilize(goCautionSnapshot()))
    }

    private fun stopSnapshot(): GuidanceSnapshot {
        return GuidanceSnapshot(
            trafficState = TrafficLightState.RED,
            userGuidanceState = UserGuidanceState.STOP,
            guidancePhase = GuidancePhase.READY_FOR_GREEN_TRANSITION,
            guidanceBlockReason = GuidanceBlockReason.NONE
        )
    }

    private fun goSnapshot(): GuidanceSnapshot {
        return GuidanceSnapshot(
            trafficState = TrafficLightState.GREEN,
            userGuidanceState = UserGuidanceState.GO,
            guidancePhase = GuidancePhase.WALK_ALLOWED,
            guidanceBlockReason = GuidanceBlockReason.NONE
        )
    }

    private fun goCautionSnapshot(): GuidanceSnapshot {
        return GuidanceSnapshot(
            trafficState = TrafficLightState.GREEN,
            userGuidanceState = UserGuidanceState.GO,
            guidancePhase = GuidancePhase.WALK_ALLOWED,
            guidanceBlockReason = GuidanceBlockReason.NONE,
            occupancyCaution = true
        )
    }

    private fun waitSnapshot(reason: GuidanceBlockReason): GuidanceSnapshot {
        return GuidanceSnapshot(
            trafficState = TrafficLightState.UNKNOWN,
            userGuidanceState = UserGuidanceState.WAIT,
            guidancePhase = GuidancePhase.WALK_ALLOWED,
            guidanceBlockReason = reason
        )
    }
}
