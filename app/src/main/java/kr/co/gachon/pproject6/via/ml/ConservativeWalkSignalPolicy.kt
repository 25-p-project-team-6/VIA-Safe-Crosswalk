package kr.co.gachon.pproject6.via.ml

class ConservativeWalkSignalPolicy(
    private val config: ConservativeWalkSignalConfig = ConservativeWalkSignalConfig()
) {
    private var phase: GuidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE

    fun update(
        state: TrafficLightState,
        hasBlockingRisk: Boolean
    ): GuidanceDecision {
        return when (phase) {
            GuidancePhase.WAITING_FOR_RED_BASELINE -> when (state) {
                TrafficLightState.RED -> {
                    phase = GuidancePhase.READY_FOR_GREEN_TRANSITION
                    GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.GREEN -> if (config.requireRedBaselineBeforeGo) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NEED_RED_BASELINE)
                } else if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                } else {
                    phase = GuidancePhase.WALK_ALLOWED
                    GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.UNKNOWN -> GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
            }

            GuidancePhase.READY_FOR_GREEN_TRANSITION -> when (state) {
                TrafficLightState.RED -> GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                TrafficLightState.GREEN -> {
                    if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                    } else {
                        phase = GuidancePhase.WALK_ALLOWED
                        GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                    }
                }

                TrafficLightState.UNKNOWN -> GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
            }

            GuidancePhase.WALK_ALLOWED -> when (state) {
                TrafficLightState.GREEN -> {
                    if (config.blockGoWhenRiskDetected && hasBlockingRisk) {
                        GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.BLOCKING_RISK)
                    } else {
                        GuidanceDecision(UserGuidanceState.GO, phase, GuidanceBlockReason.NONE)
                    }
                }

                TrafficLightState.RED -> {
                    phase = GuidancePhase.READY_FOR_GREEN_TRANSITION
                    GuidanceDecision(UserGuidanceState.STOP, phase, GuidanceBlockReason.NONE)
                }

                TrafficLightState.UNKNOWN -> {
                    phase =
                        if (config.resetToBaselineOnUnknownDuringWalk) {
                            GuidancePhase.WAITING_FOR_RED_BASELINE
                        } else {
                            GuidancePhase.READY_FOR_GREEN_TRANSITION
                        }
                    GuidanceDecision(UserGuidanceState.WAIT, phase, GuidanceBlockReason.NO_SIGNAL)
                }
            }
        }
    }

    fun reset() {
        phase = GuidancePhase.WAITING_FOR_RED_BASELINE
    }
}

data class ConservativeWalkSignalConfig(
    val requireRedBaselineBeforeGo: Boolean = true,
    val resetToBaselineOnUnknownDuringWalk: Boolean = true,
    val blockGoWhenRiskDetected: Boolean = true
)
