package kr.co.gachon.pproject6.via.ml

class GuidanceStateStabilizer(
    private val config: GuidanceStateStabilizerConfig = GuidanceStateStabilizerConfig()
) {
    private var activeSnapshot: GuidanceSnapshot? = null
    private var candidateSnapshot: GuidanceSnapshot? = null
    private var candidateCount = 0

    fun stabilize(rawSnapshot: GuidanceSnapshot): GuidanceSnapshot {
        val currentActive = activeSnapshot
        if (currentActive == null) {
            activeSnapshot = rawSnapshot
            clearCandidate()
            return rawSnapshot
        }

        if (rawSnapshot.userGuidanceState == currentActive.userGuidanceState) {
            activeSnapshot = rawSnapshot
            clearCandidate()
            return rawSnapshot
        }

        if (candidateSnapshot?.userGuidanceState == rawSnapshot.userGuidanceState) {
            candidateSnapshot = rawSnapshot
            candidateCount++
        } else {
            candidateSnapshot = rawSnapshot
            candidateCount = 1
        }

        return if (candidateCount >= requiredConfirmFrames(rawSnapshot.userGuidanceState)) {
            val committedSnapshot = candidateSnapshot ?: rawSnapshot
            activeSnapshot = committedSnapshot
            clearCandidate()
            committedSnapshot
        } else {
            currentActive
        }
    }

    fun reset() {
        activeSnapshot = null
        clearCandidate()
    }

    private fun requiredConfirmFrames(state: UserGuidanceState): Int {
        return when (state) {
            UserGuidanceState.WAIT -> config.waitConfirmFrames
            UserGuidanceState.STOP, UserGuidanceState.GO -> config.actionConfirmFrames
        }
    }

    private fun clearCandidate() {
        candidateSnapshot = null
        candidateCount = 0
    }
}

data class GuidanceStateStabilizerConfig(
    val actionConfirmFrames: Int = 2,
    val waitConfirmFrames: Int = 3
)

data class GuidanceSnapshot(
    val trafficState: TrafficLightState,
    val userGuidanceState: UserGuidanceState,
    val guidancePhase: GuidancePhase,
    val guidanceBlockReason: GuidanceBlockReason
)
