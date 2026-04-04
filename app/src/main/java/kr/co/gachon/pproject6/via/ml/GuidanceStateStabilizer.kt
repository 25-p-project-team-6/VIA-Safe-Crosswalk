package kr.co.gachon.pproject6.via.ml

class GuidanceStateStabilizer(
    private val config: GuidanceStateStabilizerConfig = GuidanceStateStabilizerConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var activeSnapshot: GuidanceSnapshot? = null
    private var candidateSnapshot: GuidanceSnapshot? = null
    private var candidateSince: Long = Long.MIN_VALUE
    private var goFamilyHoldUntil: Long = Long.MIN_VALUE

    fun stabilize(rawSnapshot: GuidanceSnapshot): GuidanceSnapshot {
        val now = timeProvider()
        val currentActive = activeSnapshot
        if (currentActive == null) {
            activeSnapshot = rawSnapshot
            refreshHold(now, rawSnapshot)
            clearCandidate()
            return rawSnapshot
        }

        val activeMode = currentActive.presentationMode()
        val rawMode = rawSnapshot.presentationMode()

        if (activeMode == rawMode) {
            activeSnapshot = rawSnapshot
            refreshHold(now, rawSnapshot)
            clearCandidate()
            return rawSnapshot
        }

        if (activeMode.isGoFamily() && rawMode == GuidancePresentationMode.WAIT && now < goFamilyHoldUntil) {
            return currentActive
        }

        if (candidateSnapshot?.presentationMode() != rawMode) {
            candidateSnapshot = rawSnapshot
            candidateSince = now
        } else {
            candidateSnapshot = rawSnapshot
        }

        val requiredDurationMs = requiredConfirmDurationMs(rawMode)
        if (candidateSince != Long.MIN_VALUE && now - candidateSince >= requiredDurationMs) {
            val committed = candidateSnapshot ?: rawSnapshot
            activeSnapshot = committed
            refreshHold(now, committed)
            clearCandidate()
            return committed
        }

        return currentActive
    }

    fun reset() {
        activeSnapshot = null
        goFamilyHoldUntil = Long.MIN_VALUE
        clearCandidate()
    }

    private fun requiredConfirmDurationMs(mode: GuidancePresentationMode): Long {
        return when (mode) {
            GuidancePresentationMode.STOP -> config.stopConfirmDurationMs
            GuidancePresentationMode.GO -> config.goConfirmDurationMs
            GuidancePresentationMode.GO_CAUTION -> config.cautionConfirmDurationMs
            GuidancePresentationMode.WAIT -> config.waitConfirmDurationMs
        }
    }

    private fun refreshHold(
        now: Long,
        snapshot: GuidanceSnapshot
    ) {
        goFamilyHoldUntil =
            if (snapshot.presentationMode().isGoFamily()) {
                now + config.goMinimumHoldMs
            } else {
                Long.MIN_VALUE
            }
    }

    private fun clearCandidate() {
        candidateSnapshot = null
        candidateSince = Long.MIN_VALUE
    }
}

data class GuidanceStateStabilizerConfig(
    val goConfirmDurationMs: Long = 250L,
    val stopConfirmDurationMs: Long = 150L,
    val waitConfirmDurationMs: Long = 350L,
    val cautionConfirmDurationMs: Long = 400L,
    val goMinimumHoldMs: Long = 500L
)

data class GuidanceSnapshot(
    val trafficState: TrafficLightState,
    val userGuidanceState: UserGuidanceState,
    val guidancePhase: GuidancePhase,
    val guidanceBlockReason: GuidanceBlockReason,
    val guidanceContinuityTier: GuidanceContinuityTier = GuidanceContinuityTier.NONE,
    val handoffDecision: CrosswalkHandoffDecision = CrosswalkHandoffDecision.NONE,
    val occupancyCaution: Boolean = false
) {
    fun presentationMode(): GuidancePresentationMode {
        return when {
            userGuidanceState == UserGuidanceState.STOP -> GuidancePresentationMode.STOP
            userGuidanceState == UserGuidanceState.GO && occupancyCaution -> GuidancePresentationMode.GO_CAUTION
            userGuidanceState == UserGuidanceState.GO -> GuidancePresentationMode.GO
            else -> GuidancePresentationMode.WAIT
        }
    }
}

enum class GuidancePresentationMode {
    STOP,
    GO,
    GO_CAUTION,
    WAIT
}

private fun GuidancePresentationMode.isGoFamily(): Boolean {
    return this == GuidancePresentationMode.GO || this == GuidancePresentationMode.GO_CAUTION
}
