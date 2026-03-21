package kr.co.gachon.pproject6.via.ml

data class GuidanceDecision(
    val state: UserGuidanceState,
    val phase: GuidancePhase,
    val blockReason: GuidanceBlockReason
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
