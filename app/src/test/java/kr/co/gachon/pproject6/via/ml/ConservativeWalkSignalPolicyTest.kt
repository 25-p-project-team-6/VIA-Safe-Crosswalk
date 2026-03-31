package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConservativeWalkSignalPolicyTest {
    @Test
    fun startupGreenDoesNotAllowWalking() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(GuidanceBlockReason.NEED_RED_BASELINE, policy.update(TrafficLightState.GREEN, false).blockReason)
    }

    @Test
    fun redBaselineThenGreenAllowsWalking() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun greenWithoutRedBaselineKeepsWaitingAcrossUnknown() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(GuidanceBlockReason.NO_SIGNAL, policy.update(TrafficLightState.UNKNOWN, false).blockReason)
        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun greenWithoutRedBaselineStillWaitsEvenWithStrongContextEvidence() {
        val policy = ConservativeWalkSignalPolicy()
        val snapshot = CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentGyroMotion = true,
            isLookingDown = true,
            hasRecentLocationMovement = true,
            crossingWindowDistanceMeters = 20f,
            crossingWindowElapsedMs = 12_000L
        )

        val decision = policy.update(TrafficLightState.GREEN, false, snapshot)
        assertEquals(UserGuidanceState.WAIT, decision.state)
        assertEquals(GuidanceBlockReason.NEED_RED_BASELINE, decision.blockReason)
    }

    @Test
    fun unknownAfterGoKeepsWalkPhaseUntilNextTransitionEvidenceAppears() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.UNKNOWN, false).state)
        currentTime += 1_501L
        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.UNKNOWN, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun redAfterGoStopsAndPreparesForNextCycle() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun blockingRiskPreventsGoUntilRiskClears() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(GuidanceBlockReason.BLOCKING_RISK, policy.update(TrafficLightState.GREEN, true).blockReason)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun blockingRiskDuringWalkDowngradesToWaitButDoesNotLoseCycle() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(GuidanceBlockReason.BLOCKING_RISK, policy.update(TrafficLightState.GREEN, true).blockReason)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun configCanAllowImmediateGoWithoutRedBaseline() {
        val policy = ConservativeWalkSignalPolicy(
            ConservativeWalkSignalConfig(requireRedBaselineBeforeGo = false)
        )

        val decision = policy.update(TrafficLightState.GREEN, false)
        assertEquals(UserGuidanceState.GO, decision.state)
        assertEquals(GuidancePhase.WALK_ALLOWED, decision.phase)
    }

    @Test
    fun briefUnknownDuringWalkKeepsGoState() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)

        currentTime += 1_000L
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.UNKNOWN, false).state)

        currentTime += 200L
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun crossingSupportExtendsUnknownGraceDuringWalk() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })
        val support = CrossingSupportSnapshot(hasRecentGyroMotion = true)

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)

        currentTime += 2_000L
        assertEquals(
            UserGuidanceState.GO,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = support
            ).state
        )

        currentTime += 2_000L
        assertEquals(
            UserGuidanceState.WAIT,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = CrossingSupportSnapshot()
            ).state
        )
    }

    @Test
    fun lookingDownExtendsUnknownGraceEvenLongerDuringWalk() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })
        val support = CrossingSupportSnapshot(isLookingDown = true)

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)

        currentTime += 4_000L
        assertEquals(
            UserGuidanceState.GO,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = support
            ).state
        )

        currentTime += 700L
        assertEquals(
            UserGuidanceState.GO,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = CrossingSupportSnapshot()
            ).state
        )

        currentTime += 900L
        assertEquals(
            UserGuidanceState.WAIT,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = CrossingSupportSnapshot()
            ).state
        )
    }

    @Test
    fun nextTransitionEvidenceCanStillResetBaselineAfterUnknown() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })
        val transitionContext = CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentLocationMovement = true,
            crossingWindowDistanceMeters = 8.5f,
            crossingWindowElapsedMs = 6_100L,
            nextCrosswalkDistanceThresholdMeters = 8.0f,
            nextCrosswalkMinActiveMs = 6_000L
        )

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)

        currentTime += 4_000L
        assertEquals(
            UserGuidanceState.GO,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = transitionContext
            ).state
        )

        currentTime += 3_600L
        assertEquals(
            UserGuidanceState.WAIT,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = transitionContext
            ).state
        )
        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.GREEN, false).state)
    }

    @Test
    fun longUnknownWithoutTransitionEvidenceWaitsButDoesNotResetBaseline() {
        var currentTime = 1_000L
        val policy = ConservativeWalkSignalPolicy(timeProvider = { currentTime })
        val continuationOnly = CrossingSupportSnapshot(
            isCrossingWindowActive = true,
            hasRecentLocationMovement = false,
            hasRecentGyroMotion = true,
            crossingWindowDistanceMeters = 8.5f,
            crossingWindowElapsedMs = 6_100L,
            nextCrosswalkDistanceThresholdMeters = 8.0f,
            nextCrosswalkMinActiveMs = 6_000L
        )

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)

        currentTime += 3_600L
        assertEquals(
            UserGuidanceState.GO,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = continuationOnly
            ).state
        )

        currentTime += 3_600L
        assertEquals(
            UserGuidanceState.WAIT,
            policy.update(
                TrafficLightState.UNKNOWN,
                false,
                crossingSupportSnapshot = continuationOnly
            ).state
        )
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
    }
}
