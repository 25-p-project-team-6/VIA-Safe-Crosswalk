package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.context.MapClusterTransitionKind
import kr.co.gachon.pproject6.via.context.MapFeatureKind
import kr.co.gachon.pproject6.via.context.MapFeatureSource
import kr.co.gachon.pproject6.via.context.MapProximitySnapshot
import org.junit.Assert.assertEquals
import org.junit.Test

class ConservativeWalkSignalPolicyTest {
    @Test
    fun startupGreenDoesNotAllowWalking() {
        val policy = ConservativeWalkSignalPolicy()

        val decision = policy.update(TrafficLightState.GREEN)

        assertEquals(UserGuidanceState.WAIT, decision.state)
        assertEquals(GuidanceBlockReason.NEED_RED_BASELINE, decision.blockReason)
    }

    @Test
    fun redBaselineThenGreenAllowsWalking() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN).state)
    }

    @Test
    fun unknownDuringWalkUsesMatchedMovingDownTierGrace() {
        var now = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { now })
        val snapshot =
            supportSnapshot(
                hasRecentLocationMovement = true,
                isLookingDown = true,
                matchedClusterId = "cluster-a"
            )

        policy.update(TrafficLightState.RED)
        policy.update(TrafficLightState.GREEN)

        now += 4_700L
        val stillGo = policy.update(TrafficLightState.UNKNOWN, snapshot)
        now += 4_900L
        val wait = policy.update(TrafficLightState.UNKNOWN, snapshot)

        assertEquals(UserGuidanceState.GO, stillGo.state)
        assertEquals(GuidanceContinuityTier.MATCHED_MOVING_DOWN, stillGo.continuityTier)
        assertEquals(UserGuidanceState.WAIT, wait.state)
    }

    @Test
    fun unknownAfterGraceWithoutNewCrossingWaitsButKeepsWalkPhase() {
        var now = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { now })
        val snapshot = supportSnapshot(matchedClusterId = "cluster-a")

        policy.update(TrafficLightState.RED)
        policy.update(TrafficLightState.GREEN)

        policy.update(TrafficLightState.UNKNOWN, snapshot)
        now += 3_600L
        val wait = policy.update(TrafficLightState.UNKNOWN, snapshot)
        val reacquired = policy.update(TrafficLightState.GREEN, snapshot)

        assertEquals(UserGuidanceState.WAIT, wait.state)
        assertEquals(GuidancePhase.WALK_ALLOWED, wait.phase)
        assertEquals(UserGuidanceState.GO, reacquired.state)
    }

    @Test
    fun newCrossingTransitionResetsBackToBaselineAfterUnknownGrace() {
        var now = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { now })
        val snapshot =
            supportSnapshot(
                hasRecentLocationMovement = true,
                crossingWindowDistanceMeters = 24f,
                crossingWindowElapsedMs = 13_000L,
                matchedClusterId = "cluster-b",
                transitionKind = MapClusterTransitionKind.NEW_CROSSING
            )

        policy.update(TrafficLightState.RED)
        policy.update(TrafficLightState.GREEN)

        policy.update(TrafficLightState.UNKNOWN, snapshot)
        now += 3_600L
        val wait = policy.update(TrafficLightState.UNKNOWN, snapshot)
        val nextGreen = policy.update(TrafficLightState.GREEN, snapshot)

        assertEquals(UserGuidanceState.WAIT, wait.state)
        assertEquals(CrosswalkHandoffDecision.NEW_CROSSING, wait.handoffDecision)
        assertEquals(GuidancePhase.WAITING_FOR_RED_BASELINE, wait.phase)
        assertEquals(UserGuidanceState.WAIT, nextGreen.state)
        assertEquals(GuidanceBlockReason.NEED_RED_BASELINE, nextGreen.blockReason)
    }

    @Test
    fun sameCrossingHandoffKeepsWalkAllowedAfterGraceEnds() {
        var now = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { now })
        val snapshot =
            supportSnapshot(
                hasRecentLocationMovement = true,
                crossingWindowDistanceMeters = 10f,
                crossingWindowElapsedMs = 8_000L,
                matchedClusterId = "cluster-b",
                transitionKind = MapClusterTransitionKind.SAME_CROSSING
            )

        policy.update(TrafficLightState.RED)
        policy.update(TrafficLightState.GREEN)

        policy.update(TrafficLightState.UNKNOWN, snapshot)
        now += 3_600L
        val wait = policy.update(TrafficLightState.UNKNOWN, snapshot)

        assertEquals(UserGuidanceState.WAIT, wait.state)
        assertEquals(GuidancePhase.WALK_ALLOWED, wait.phase)
        assertEquals(CrosswalkHandoffDecision.SAME_CROSSING, wait.handoffDecision)
    }

    private fun supportSnapshot(
        hasRecentLocationMovement: Boolean = false,
        isLookingDown: Boolean = false,
        crossingWindowDistanceMeters: Float = 0f,
        crossingWindowElapsedMs: Long = 0L,
        matchedClusterId: String? = null,
        transitionKind: MapClusterTransitionKind = MapClusterTransitionKind.NONE
    ): CrossingSupportSnapshot {
        return CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentLocationMovement = hasRecentLocationMovement,
            isLookingDown = isLookingDown,
            crossingWindowDistanceMeters = crossingWindowDistanceMeters,
            crossingWindowElapsedMs = crossingWindowElapsedMs,
            mapProximitySnapshot =
                MapProximitySnapshot(
                    isNearKnownFeature = matchedClusterId != null,
                    matchedFeatureId = matchedClusterId,
                    matchedKind = if (matchedClusterId != null) MapFeatureKind.CROSSWALK else null,
                    matchedSource = if (matchedClusterId != null) MapFeatureSource.HYBRID else null,
                    matchedClusterId = matchedClusterId,
                    clusterTransitionKind = transitionKind
                )
        )
    }
}
