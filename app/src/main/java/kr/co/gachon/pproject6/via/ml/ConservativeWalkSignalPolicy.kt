package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.context.MapClusterTransitionKind

class ConservativeWalkSignalPolicy(
    private val config: ConservativeWalkSignalConfig = ConservativeWalkSignalConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var phase: GuidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE
    private var walkUnknownStartedAt: Long = Long.MIN_VALUE

    fun update(
        state: TrafficLightState,
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): GuidanceDecision {
        val currentTime = timeProvider()
        return when (phase) {
            GuidancePhase.WAITING_FOR_RED_BASELINE -> handleWaitingForRedBaseline(state)
            GuidancePhase.READY_FOR_GREEN_TRANSITION -> handleReadyForGreenTransition(state)
            GuidancePhase.WALK_ALLOWED ->
                handleWalkAllowed(
                    state = state,
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
        state: TrafficLightState
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.RED -> stopAndMoveTo(GuidancePhase.READY_FOR_GREEN_TRANSITION)
            TrafficLightState.GREEN ->
                if (config.requireRedBaselineBeforeGo) {
                    waitDecision(GuidanceBlockReason.NEED_RED_BASELINE)
                } else {
                    allowWalk()
                }

            TrafficLightState.UNKNOWN -> waitDecision(GuidanceBlockReason.NO_SIGNAL)
        }
    }

    private fun handleReadyForGreenTransition(
        state: TrafficLightState
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.RED -> stopAndKeepPhase()
            TrafficLightState.GREEN -> allowWalk()
            TrafficLightState.UNKNOWN -> waitDecision(GuidanceBlockReason.NO_SIGNAL)
        }
    }

    private fun handleWalkAllowed(
        state: TrafficLightState,
        crossingSupportSnapshot: CrossingSupportSnapshot,
        currentTime: Long
    ): GuidanceDecision {
        return when (state) {
            TrafficLightState.GREEN -> {
                clearUnknownGrace()
                goDecision(
                    continuityTier = continuityTier(crossingSupportSnapshot),
                    handoffDecision = handoffDecision(crossingSupportSnapshot)
                )
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
        val continuityTier = continuityTier(crossingSupportSnapshot)
        val handoffDecision = handoffDecision(crossingSupportSnapshot)
        val allowedGraceMs = allowedUnknownGraceMs(continuityTier)
        return if (currentTime - walkUnknownStartedAt <= allowedGraceMs) {
            goDecision(
                blockReason = GuidanceBlockReason.NO_SIGNAL,
                continuityTier = continuityTier,
                handoffDecision = handoffDecision
            )
        } else if (handoffDecision == CrosswalkHandoffDecision.NEW_CROSSING) {
            phase = GuidancePhase.WAITING_FOR_RED_BASELINE
            clearUnknownGrace()
            waitDecision(
                blockReason = GuidanceBlockReason.NO_SIGNAL,
                continuityTier = continuityTier,
                handoffDecision = handoffDecision
            )
        } else {
            waitDecision(
                blockReason = GuidanceBlockReason.NO_SIGNAL,
                continuityTier = continuityTier,
                handoffDecision = handoffDecision
            )
        }
    }

    private fun continuityTier(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ): GuidanceContinuityTier {
        val hasMatched = crossingSupportSnapshot.mapProximitySnapshot.matchedClusterId != null
        return when {
            hasMatched && crossingSupportSnapshot.hasRecentLocationMovement && crossingSupportSnapshot.isLookingDown ->
                GuidanceContinuityTier.MATCHED_MOVING_DOWN

            hasMatched && crossingSupportSnapshot.hasRecentLocationMovement ->
                GuidanceContinuityTier.MATCHED_MOVING

            hasMatched -> GuidanceContinuityTier.MATCHED
            else -> GuidanceContinuityTier.NONE
        }
    }

    private fun handoffDecision(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ): CrosswalkHandoffDecision {
        val mapSnapshot = crossingSupportSnapshot.mapProximitySnapshot
        return when (mapSnapshot.clusterTransitionKind) {
            MapClusterTransitionKind.NONE -> CrosswalkHandoffDecision.NONE
            MapClusterTransitionKind.SAME_CROSSING ->
                if (isSameCrossingContinuation(crossingSupportSnapshot)) {
                    CrosswalkHandoffDecision.SAME_CROSSING
                } else {
                    CrosswalkHandoffDecision.NEW_CROSSING
                }

            MapClusterTransitionKind.NEW_CROSSING -> CrosswalkHandoffDecision.NEW_CROSSING
        }
    }

    private fun isSameCrossingContinuation(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ): Boolean {
        return crossingSupportSnapshot.hasRecentLocationMovement &&
            crossingSupportSnapshot.crossingWindowDistanceMeters <= config.sameCrossingMaxDistanceMeters &&
            crossingSupportSnapshot.crossingWindowElapsedMs <= config.sameCrossingMaxElapsedMs
    }

    private fun allowedUnknownGraceMs(
        continuityTier: GuidanceContinuityTier
    ): Long {
        return when (continuityTier) {
            GuidanceContinuityTier.NONE -> config.walkAllowedUnknownGraceMs
            GuidanceContinuityTier.MATCHED -> config.walkAllowedUnknownGraceMatchedMs
            GuidanceContinuityTier.MATCHED_MOVING -> config.walkAllowedUnknownGraceMatchedMovingMs
            GuidanceContinuityTier.MATCHED_MOVING_DOWN -> config.walkAllowedUnknownGraceMatchedMovingDownMs
        }
    }

    private fun stopAndMoveTo(
        nextPhase: GuidancePhase
    ): GuidanceDecision {
        phase = nextPhase
        clearUnknownGrace()
        return GuidanceDecision(
            state = UserGuidanceState.STOP,
            phase = phase,
            blockReason = GuidanceBlockReason.NONE
        )
    }

    private fun stopAndKeepPhase(): GuidanceDecision {
        clearUnknownGrace()
        return GuidanceDecision(
            state = UserGuidanceState.STOP,
            phase = phase,
            blockReason = GuidanceBlockReason.NONE
        )
    }

    private fun allowWalk(): GuidanceDecision {
        phase = GuidancePhase.WALK_ALLOWED
        clearUnknownGrace()
        return goDecision()
    }

    private fun goDecision(
        blockReason: GuidanceBlockReason = GuidanceBlockReason.NONE,
        continuityTier: GuidanceContinuityTier = GuidanceContinuityTier.NONE,
        handoffDecision: CrosswalkHandoffDecision = CrosswalkHandoffDecision.NONE
    ): GuidanceDecision {
        return GuidanceDecision(
            state = UserGuidanceState.GO,
            phase = phase,
            blockReason = blockReason,
            continuityTier = continuityTier,
            handoffDecision = handoffDecision
        )
    }

    private fun waitDecision(
        blockReason: GuidanceBlockReason,
        continuityTier: GuidanceContinuityTier = GuidanceContinuityTier.NONE,
        handoffDecision: CrosswalkHandoffDecision = CrosswalkHandoffDecision.NONE
    ): GuidanceDecision {
        return GuidanceDecision(
            state = UserGuidanceState.WAIT,
            phase = phase,
            blockReason = blockReason,
            continuityTier = continuityTier,
            handoffDecision = handoffDecision
        )
    }

    private fun clearUnknownGrace() {
        walkUnknownStartedAt = Long.MIN_VALUE
    }
}

data class ConservativeWalkSignalConfig(
    val requireRedBaselineBeforeGo: Boolean = true,
    val walkAllowedUnknownGraceMs: Long = 1_200L,
    val walkAllowedUnknownGraceMatchedMs: Long = 2_200L,
    val walkAllowedUnknownGraceMatchedMovingMs: Long = 3_500L,
    val walkAllowedUnknownGraceMatchedMovingDownMs: Long = 4_800L,
    val sameCrossingMaxDistanceMeters: Float = 20f,
    val sameCrossingMaxElapsedMs: Long = 12_000L
)
