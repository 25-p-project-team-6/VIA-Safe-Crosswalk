package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot

class ConservativeWalkSignalPolicy(
    private val config: ConservativeWalkSignalConfig = ConservativeWalkSignalConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var phase: GuidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE
    private var lastRedBaselineAt: Long = Long.MIN_VALUE
    private var walkUnknownStartedAt: Long = Long.MIN_VALUE

    fun update(
        state: TrafficLightState,
        hasBlockingRisk: Boolean,
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): GuidanceDecision {
        val currentTime = timeProvider()
        return when (phase) {
            GuidancePhase.WAITING_FOR_RED_BASELINE -> when (state) {
                TrafficLightState.RED -> {
                    phase = GuidancePhase.READY_FOR_GREEN_TRANSITION
                    lastRedBaselineAt = currentTime
                    walkUnknownStartedAt = Long.MIN_VALUE
                    GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.GREEN -> if (config.requireRedBaselineBeforeGo) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NEED_RED_BASELINE)
                } else if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                } else {
                    phase = GuidancePhase.WALK_ALLOWED
                    walkUnknownStartedAt = Long.MIN_VALUE
                    GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.UNKNOWN -> GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
            }

            GuidancePhase.READY_FOR_GREEN_TRANSITION -> when (state) {
                TrafficLightState.RED -> {
                    lastRedBaselineAt = currentTime
                    walkUnknownStartedAt = Long.MIN_VALUE
                    GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                }
                TrafficLightState.GREEN -> {
                    if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                    } else {
                        phase = GuidancePhase.WALK_ALLOWED
                        walkUnknownStartedAt = Long.MIN_VALUE
                        GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                    }
                }

                TrafficLightState.UNKNOWN -> GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
            }

            GuidancePhase.WALK_ALLOWED -> when (state) {
                TrafficLightState.GREEN -> {
                    walkUnknownStartedAt = Long.MIN_VALUE
                    if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                    } else {
                        GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                    }
                }

                TrafficLightState.RED -> {
                    phase = GuidancePhase.READY_FOR_GREEN_TRANSITION
                    lastRedBaselineAt = currentTime
                    walkUnknownStartedAt = Long.MIN_VALUE
                    GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.UNKNOWN -> {
                    if (walkUnknownStartedAt == Long.MIN_VALUE) {
                        walkUnknownStartedAt = currentTime
                    }

                    val allowedGraceMs =
                        if (crossingSupportSnapshot.supportsWalkContinuation) {
                            config.walkAllowedUnknownGraceWithContextMs
                        } else {
                            config.walkAllowedUnknownGraceMs
                        }

                    if (currentTime - walkUnknownStartedAt <= allowedGraceMs) {
                        GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NO_SIGNAL)
                    } else {
                        phase =
                            if (config.resetToBaselineOnUnknownDuringWalk) {
                                GuidancePhase.WAITING_FOR_RED_BASELINE
                            } else {
                                GuidancePhase.READY_FOR_GREEN_TRANSITION
                            }
                        walkUnknownStartedAt = Long.MIN_VALUE
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
                    }
                }
            }
        }
    }

    fun reset() {
        phase = GuidancePhase.WAITING_FOR_RED_BASELINE
        lastRedBaselineAt = Long.MIN_VALUE
        walkUnknownStartedAt = Long.MIN_VALUE
    }

    fun shouldResetOnTargetSessionChange(
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): Boolean {
        if (phase == GuidancePhase.WALK_ALLOWED && crossingSupportSnapshot.supportsWalkContinuation) {
            return false
        }

        if (phase != GuidancePhase.READY_FOR_GREEN_TRANSITION) {
            return true
        }

        return lastRedBaselineAt == Long.MIN_VALUE ||
            timeProvider() - lastRedBaselineAt > config.preserveReadyBaselineMs
    }
}

data class ConservativeWalkSignalConfig(
    val requireRedBaselineBeforeGo: Boolean = true,
    val resetToBaselineOnUnknownDuringWalk: Boolean = true,
    val blockGoWhenRiskDetected: Boolean = true,
    val preserveReadyBaselineMs: Long = 2_500L,
    val walkAllowedUnknownGraceMs: Long = 1_500L,
    val walkAllowedUnknownGraceWithContextMs: Long = 3_500L
)
