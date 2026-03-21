package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
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
    fun unknownAfterGoResetsToRequireNewRedBaseline() {
        val policy = ConservativeWalkSignalPolicy()

        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
        assertEquals(UserGuidanceState.GO, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.UNKNOWN, false).state)
        assertEquals(UserGuidanceState.WAIT, policy.update(TrafficLightState.GREEN, false).state)
        assertEquals(UserGuidanceState.STOP, policy.update(TrafficLightState.RED, false).state)
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
}
