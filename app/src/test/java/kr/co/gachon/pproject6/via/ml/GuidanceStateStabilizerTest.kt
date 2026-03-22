package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceStateStabilizerTest {
    private val stabilizer = GuidanceStateStabilizer(
        GuidanceStateStabilizerConfig(actionConfirmFrames = 2, waitConfirmFrames = 3)
    )

    @Test
    fun firstSnapshotBecomesActiveImmediately() {
        val initial = waitSnapshot(GuidanceBlockReason.NEED_RED_BASELINE)

        assertEquals(initial, stabilizer.stabilize(initial))
    }

    @Test
    fun goTransitionRequiresTwoConsecutiveFrames() {
        stabilizer.stabilize(stopSnapshot())

        assertEquals(stopSnapshot(), stabilizer.stabilize(goSnapshot()))
        assertEquals(goSnapshot(), stabilizer.stabilize(goSnapshot()))
    }

    @Test
    fun waitTransitionRequiresThreeConsecutiveFrames() {
        stabilizer.stabilize(goSnapshot())

        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        assertEquals(waitSnapshot(GuidanceBlockReason.NO_SIGNAL), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
    }

    @Test
    fun transientWaitCandidateClearsWhenGoReturns() {
        stabilizer.stabilize(goSnapshot())

        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        assertEquals(goSnapshot(), stabilizer.stabilize(goSnapshot()))
        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        assertEquals(goSnapshot(), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
        assertEquals(waitSnapshot(GuidanceBlockReason.NO_SIGNAL), stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NO_SIGNAL)))
    }

    @Test
    fun sameGuidanceStateRefreshesSnapshotDetailsImmediately() {
        stabilizer.stabilize(waitSnapshot(GuidanceBlockReason.NEED_RED_BASELINE))

        val updated = waitSnapshot(GuidanceBlockReason.NO_SIGNAL)
        assertEquals(updated, stabilizer.stabilize(updated))
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

    private fun waitSnapshot(reason: GuidanceBlockReason): GuidanceSnapshot {
        return GuidanceSnapshot(
            trafficState = if (reason == GuidanceBlockReason.NEED_RED_BASELINE) {
                TrafficLightState.GREEN
            } else {
                TrafficLightState.UNKNOWN
            },
            userGuidanceState = UserGuidanceState.WAIT,
            guidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE,
            guidanceBlockReason = reason
        )
    }
}
