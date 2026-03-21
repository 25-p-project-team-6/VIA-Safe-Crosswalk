package kr.co.gachon.pproject6.via.feedback

import kr.co.gachon.pproject6.via.ml.UserGuidanceState

class SignalFeedbackPolicy(
    private val timingConfig: SignalFeedbackTimingConfig = SignalFeedbackTimingConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var activeState: UserGuidanceState? = null
    private var lastEmissionAt: Long = Long.MIN_VALUE

    fun shouldEmit(state: UserGuidanceState): Boolean {
        val currentTime = timeProvider()

        if (state != activeState) {
            activeState = state
            lastEmissionAt = currentTime
            return true
        }

        val repeatInterval =
            if (state == UserGuidanceState.WAIT) {
                timingConfig.waitRepeatIntervalMs
            } else {
                timingConfig.actionRepeatIntervalMs
            }

        if (lastEmissionAt == Long.MIN_VALUE || currentTime - lastEmissionAt >= repeatInterval) {
            lastEmissionAt = currentTime
            return true
        }

        return false
    }

    fun clear() {
        activeState = null
        lastEmissionAt = Long.MIN_VALUE
    }
}

data class SignalFeedbackTimingConfig(
    val actionRepeatIntervalMs: Long = 4_000L,
    val waitRepeatIntervalMs: Long = 8_000L
)
