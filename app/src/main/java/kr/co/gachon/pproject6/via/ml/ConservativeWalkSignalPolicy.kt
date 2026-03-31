package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot

class ConservativeWalkSignalPolicy(
    private val config: ConservativeWalkSignalConfig = ConservativeWalkSignalConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var phase: GuidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE
    private var walkUnknownStartedAt: Long = Long.MIN_VALUE

    fun update(
        state: TrafficLightState,
        hasBlockingRisk: Boolean,
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): GuidanceDecision {
        val currentTime = timeProvider()
        return when (phase) {
            GuidancePhase.WAITING_FOR_RED_BASELINE ->
                handleWaitingForRedBaseline(state, hasBlockingRisk)

            GuidancePhase.READY_FOR_GREEN_TRANSITION ->
                handleReadyForGreenTransition(state, hasBlockingRisk)

            GuidancePhase.WALK_ALLOWED ->
                handleWalkAllowed(
                    state = state,
                    hasBlockingRisk = hasBlockingRisk,
                    crossingSupportSnapshot = crossingSupportSnapshot,
                    currentTime = currentTime
                )
        }
    }

    fun reset() {
        phase = GuidancePhase.WAITING_FOR_RED_BASELINE
        walkUnknownStartedAt = Long.MIN_VALUE
    }

    private fun handleWaitingForRedBaseline(
        state: TrafficLightState,
        hasBlockingRisk: Boolean
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.RED -> stopAndMoveTo(GuidancePhase.READY_FOR_GREEN_TRANSITION)
            TrafficLightState.GREEN ->
                when {
                    config.requireRedBaselineBeforeGo ->
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NEED_RED_BASELINE)

                    isBlockedByRisk(hasBlockingRisk) ->
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)

                    else -> allowWalk()
                }

            TrafficLightState.UNKNOWN -> waitForNoSignal()
        }
    }

    private fun handleReadyForGreenTransition(
        state: TrafficLightState,
        hasBlockingRisk: Boolean
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.RED -> stopAndKeepPhase()
            TrafficLightState.GREEN ->
                if (isBlockedByRisk(hasBlockingRisk)) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                } else {
                    allowWalk()
                }

            TrafficLightState.UNKNOWN -> waitForNoSignal()
        }
    }

    private fun handleWalkAllowed(
        state: TrafficLightState,
        hasBlockingRisk: Boolean,
        crossingSupportSnapshot: CrossingSupportSnapshot,
        currentTime: Long
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.GREEN -> {
                clearUnknownGrace()
                if (isBlockedByRisk(hasBlockingRisk)) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                } else {
                    GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                }
            }

            TrafficLightState.RED -> stopAndMoveTo(GuidancePhase.READY_FOR_GREEN_TRANSITION)

            TrafficLightState.UNKNOWN -> handleUnknownDuringWalk(crossingSupportSnapshot, currentTime)
        }
    }

    private fun handleUnknownDuringWalk(
        crossingSupportSnapshot: CrossingSupportSnapshot,
        currentTime: Long
    ): GuidanceDecision {
        if (walkUnknownStartedAt == Long.MIN_VALUE) {
            walkUnknownStartedAt = currentTime
        }

        val allowedGraceMs = allowedUnknownGraceMs(crossingSupportSnapshot)
        return if (currentTime - walkUnknownStartedAt <= allowedGraceMs) {
            GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NO_SIGNAL)
        } else if (hasNextCrosswalkTransitionEvidence(crossingSupportSnapshot)) {
            phase =
                if (config.resetToBaselineOnUnknownDuringWalk) {
                    GuidancePhase.WAITING_FOR_RED_BASELINE
                } else {
                    GuidancePhase.READY_FOR_GREEN_TRANSITION
                }
            clearUnknownGrace()
            GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
        } else {
            GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
        }
    }

    private fun allowedUnknownGraceMs(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ): Long {
        return when {
            crossingSupportSnapshot.isLookingDown -> config.walkAllowedUnknownGraceLookingDownMs
            crossingSupportSnapshot.supportsWalkContinuation -> config.walkAllowedUnknownGraceWithContextMs
            else -> config.walkAllowedUnknownGraceMs
        }
    }

    private fun isBlockedByRisk(
        hasBlockingRisk: Boolean
    ): Boolean = config.blockGoWhenRiskDetected && hasBlockingRisk

    private fun stopAndMoveTo(
        nextPhase: GuidancePhase
    ): GuidanceDecision {
        phase = nextPhase
        clearUnknownGrace()
        return GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
    }

    private fun stopAndKeepPhase(): GuidanceDecision {
        clearUnknownGrace()
        return GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
    }

    private fun allowWalk(): GuidanceDecision {
        phase = GuidancePhase.WALK_ALLOWED
        clearUnknownGrace()
        return GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
    }

    private fun waitForNoSignal(): GuidanceDecision {
        return GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
    }

    private fun clearUnknownGrace() {
        walkUnknownStartedAt = Long.MIN_VALUE
    }

    private fun hasNextCrosswalkTransitionEvidence(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ): Boolean {
        return phase == GuidancePhase.WALK_ALLOWED &&
            crossingSupportSnapshot.isCrossingWindowActive &&
            crossingSupportSnapshot.hasRecentLocationMovement &&
            crossingSupportSnapshot.crossingWindowDistanceMeters >= crossingSupportSnapshot.nextCrosswalkDistanceThresholdMeters &&
            crossingSupportSnapshot.crossingWindowElapsedMs >= crossingSupportSnapshot.nextCrosswalkMinActiveMs
    }
}

data class ConservativeWalkSignalConfig(
    val requireRedBaselineBeforeGo: Boolean = true,
    val resetToBaselineOnUnknownDuringWalk: Boolean = true,
    val blockGoWhenRiskDetected: Boolean = true,
    val walkAllowedUnknownGraceMs: Long = 1_500L,
    val walkAllowedUnknownGraceWithContextMs: Long = 3_500L,
    val walkAllowedUnknownGraceLookingDownMs: Long = 4_500L
)
