package kr.co.gachon.pproject6.via.ml

class TrafficLightStateTracker(
    private val persistenceDurationMs: Long = 5000L,
    private val triggerThreshold: Int = 3,
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var acceptedState: TrafficLightState = TrafficLightState.UNKNOWN
    private var lastEvidenceTime: Long = Long.MIN_VALUE
    private var candidateState: TrafficLightState = TrafficLightState.UNKNOWN
    private var consecutiveCount = 0

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
            candidateState = currentState
            consecutiveCount = triggerThreshold
            return acceptedState
        }

        if (currentState == candidateState) {
            consecutiveCount++
        } else {
            candidateState = currentState
            consecutiveCount = 1
        }

        if (isHighConfidence || consecutiveCount >= triggerThreshold) {
            acceptedState = currentState
            lastEvidenceTime = currentTime
            consecutiveCount = maxOf(consecutiveCount, triggerThreshold)
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
            currentTime - lastEvidenceTime < persistenceDurationMs
    }

    private fun clearAcceptedState() {
        acceptedState = TrafficLightState.UNKNOWN
        lastEvidenceTime = Long.MIN_VALUE
    }

    private fun clearCandidate() {
        candidateState = TrafficLightState.UNKNOWN
        consecutiveCount = 0
    }
}
