package kr.co.gachon.pproject6.via.ml

class AdvisoryAssessmentStabilizer(
    private val config: AdvisoryAssessmentStabilizerConfig = AdvisoryAssessmentStabilizerConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var activeVolatilePresentation: VolatileUncertainPresentation? = null
    private var candidateVolatilePresentation: VolatileUncertainPresentation? = null
    private var candidateSinceMs: Long = Long.MIN_VALUE

    fun stabilize(rawAssessment: AdvisoryAssessment): AdvisoryAssessment {
        val currentPresentation = rawAssessment.toVolatileUncertainPresentation()
        if (currentPresentation == null) {
            reset()
            return rawAssessment
        }

        val activePresentation = activeVolatilePresentation
        if (activePresentation == null) {
            commit(currentPresentation)
            return rawAssessment
        }

        if (activePresentation.key == currentPresentation.key) {
            activeVolatilePresentation = currentPresentation
            clearCandidate()
            return rawAssessment
        }

        val now = timeProvider()
        if (candidateVolatilePresentation?.key != currentPresentation.key) {
            candidateVolatilePresentation = currentPresentation
            candidateSinceMs = now
        } else {
            candidateVolatilePresentation = currentPresentation
        }

        if (candidateSinceMs != Long.MIN_VALUE &&
            now - candidateSinceMs >= config.volatileUncertainSwitchHoldMs
        ) {
            val committed = candidateVolatilePresentation ?: currentPresentation
            commit(committed)
            return rawAssessment
        }

        return rawAssessment.withPresentation(activePresentation)
    }

    fun reset() {
        activeVolatilePresentation = null
        clearCandidate()
    }

    private fun commit(presentation: VolatileUncertainPresentation) {
        activeVolatilePresentation = presentation
        clearCandidate()
    }

    private fun clearCandidate() {
        candidateVolatilePresentation = null
        candidateSinceMs = Long.MIN_VALUE
    }
}

data class AdvisoryAssessmentStabilizerConfig(
    val volatileUncertainSwitchHoldMs: Long = 1_200L
)

private enum class VolatileUncertainKey {
    VEHICLE_SIGNAL_ONLY,
    SIGNAL_MISSING
}

private data class VolatileUncertainPresentation(
    val key: VolatileUncertainKey,
    val titleText: String,
    val detailText: String,
    val speechText: String
)

private fun AdvisoryAssessment.toVolatileUncertainPresentation(): VolatileUncertainPresentation? {
    if (state != AdvisoryState.UNCERTAIN_VIEW) {
        return null
    }

    val key =
        when (detailText) {
            "차량 신호." -> VolatileUncertainKey.VEHICLE_SIGNAL_ONLY
            "신호 미탐지." -> VolatileUncertainKey.SIGNAL_MISSING
            else -> return null
        }

    return VolatileUncertainPresentation(
        key = key,
        titleText = titleText,
        detailText = detailText,
        speechText = speechText
    )
}

private fun AdvisoryAssessment.withPresentation(
    presentation: VolatileUncertainPresentation
): AdvisoryAssessment {
    return copy(
        titleText = presentation.titleText,
        detailText = presentation.detailText,
        speechText = presentation.speechText
    )
}
