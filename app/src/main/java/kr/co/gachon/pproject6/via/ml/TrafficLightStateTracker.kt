package kr.co.gachon.pproject6.via.ml

class TrafficLightStateTracker(
    private val config: TrafficLightStateTrackingConfig = TrafficLightStateTrackingConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var acceptedState: TrafficLightState = TrafficLightState.UNKNOWN
    private var lastEvidenceTime: Long = Long.MIN_VALUE
    private var candidateState: TrafficLightState = TrafficLightState.UNKNOWN
    private var candidateStateSince: Long = Long.MIN_VALUE

    fun update(
        currentState: TrafficLightState,
        isHighConfidence: Boolean
    ): TrafficLightState {
        val currentTime = timeProvider()

        if (currentState == TrafficLightState.UNKNOWN) {
            clearCandidate()
            return if (isAcceptedFresh(currentTime)) {
                acceptedState
            } else {
                clearAcceptedState()
                TrafficLightState.UNKNOWN
            }
        }

        if (currentState == acceptedState) {
            lastEvidenceTime = currentTime
            clearCandidate()
            return acceptedState
        }

        if (currentState != candidateState) {
            candidateState = currentState
            candidateStateSince = currentTime
        } else {
            candidateStateSince = minOf(candidateStateSince, currentTime)
        }

        val isReadyByFastTrack = config.allowHighConfidenceImmediateCommit && isHighConfidence
        val requiredDurationMs = requiredDurationFor(currentState)
        val candidateObservedMs =
            if (candidateStateSince == Long.MIN_VALUE) {
                0L
            } else {
                currentTime - candidateStateSince
            }
        if (isReadyByFastTrack || candidateObservedMs >= requiredDurationMs) {
            acceptedState = currentState
            lastEvidenceTime = currentTime
            clearCandidate()
            return acceptedState
        }

        return if (isAcceptedFresh(currentTime)) acceptedState else TrafficLightState.UNKNOWN
    }

    fun reset() {
        clearAcceptedState()
        clearCandidate()
    }

    private fun isAcceptedFresh(currentTime: Long): Boolean {
        return acceptedState != TrafficLightState.UNKNOWN &&
            lastEvidenceTime != Long.MIN_VALUE &&
            currentTime - lastEvidenceTime < persistenceDurationFor(acceptedState)
    }

    private fun persistenceDurationFor(state: TrafficLightState): Long {
        return when (state) {
            TrafficLightState.RED -> config.redPersistenceDurationMs
            TrafficLightState.GREEN -> config.greenPersistenceDurationMs
            TrafficLightState.UNKNOWN -> 0L
        }
    }

    private fun requiredDurationFor(currentState: TrafficLightState): Long {
        return when {
            acceptedState == TrafficLightState.UNKNOWN -> config.confirmDurationMs
            acceptedState == TrafficLightState.RED && currentState == TrafficLightState.GREEN ->
                config.redToGreenSwitchConfirmDurationMs
            acceptedState == TrafficLightState.GREEN && currentState == TrafficLightState.RED ->
                config.greenToRedSwitchConfirmDurationMs
            else -> config.switchConfirmDurationMs
        }
    }

    private fun clearAcceptedState() {
        acceptedState = TrafficLightState.UNKNOWN
        lastEvidenceTime = Long.MIN_VALUE
    }

    private fun clearCandidate() {
        candidateState = TrafficLightState.UNKNOWN
        candidateStateSince = Long.MIN_VALUE
    }
}

data class TrafficLightStateTrackingConfig(
    val confirmDurationMs: Long = 250L,
    val switchConfirmDurationMs: Long = 400L,
    val redToGreenSwitchConfirmDurationMs: Long = 200L,
    val greenToRedSwitchConfirmDurationMs: Long = 400L,
    // Red can stay sticky longer for stop stability, but green should expire fast so
    // the next intersection requires a fresh red baseline after the previous signal is lost.
    val redPersistenceDurationMs: Long = 5_000L,
    val greenPersistenceDurationMs: Long = 2_500L,
    val allowHighConfidenceImmediateCommit: Boolean = false
)
