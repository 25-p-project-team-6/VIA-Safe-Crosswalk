package kr.co.gachon.pproject6.via.ml

data class GuidanceDecision(
    val state: UserGuidanceState,
    val phase: GuidancePhase,
    val blockReason: GuidanceBlockReason,
    val continuityTier: GuidanceContinuityTier = GuidanceContinuityTier.NONE,
    val handoffDecision: CrosswalkHandoffDecision = CrosswalkHandoffDecision.NONE
)

enum class GuidancePhase {
    WAITING_FOR_RED_BASELINE,
    READY_FOR_GREEN_TRANSITION,
    WALK_ALLOWED
}

enum class GuidanceBlockReason {
    NONE,
    NEED_RED_BASELINE,
    BLOCKING_RISK,
    NO_SIGNAL
}

enum class GuidanceContinuityTier {
    NONE,
    MATCHED,
    MATCHED_MOVING,
    MATCHED_MOVING_DOWN
}

enum class CrosswalkHandoffDecision {
    NONE,
    SAME_CROSSING,
    NEW_CROSSING
}
