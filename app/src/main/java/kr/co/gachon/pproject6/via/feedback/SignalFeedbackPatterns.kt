package kr.co.gachon.pproject6.via.feedback

import kr.co.gachon.pproject6.via.ml.AdvisoryState

enum class SignalFeedbackPattern {
    RED_CONFIRMED,
    GREEN_CONFIRMED,
    GREEN_WITH_CAUTION,
    WAIT_OR_UNCERTAIN
}

object SignalFeedbackPatterns {
    fun forAdvisoryState(state: AdvisoryState): LongArray =
        when (state) {
            AdvisoryState.RED_CONFIRMED -> copyOf(SignalFeedbackPattern.RED_CONFIRMED)
            AdvisoryState.GREEN_CONFIRMED -> copyOf(SignalFeedbackPattern.GREEN_CONFIRMED)
            AdvisoryState.GREEN_WITH_CAUTION -> copyOf(SignalFeedbackPattern.GREEN_WITH_CAUTION)
            AdvisoryState.TRANSITION_WAIT,
            AdvisoryState.UNCERTAIN_VIEW -> copyOf(SignalFeedbackPattern.WAIT_OR_UNCERTAIN)
        }

    fun copyOf(pattern: SignalFeedbackPattern): LongArray =
        when (pattern) {
            SignalFeedbackPattern.RED_CONFIRMED -> longArrayOf(0, 400, 200, 400)
            SignalFeedbackPattern.GREEN_CONFIRMED -> longArrayOf(0, 180, 120, 180, 120, 180)
            SignalFeedbackPattern.GREEN_WITH_CAUTION -> longArrayOf(0, 150, 100, 150, 250, 150)
            SignalFeedbackPattern.WAIT_OR_UNCERTAIN -> longArrayOf(0, 140)
        }
}
