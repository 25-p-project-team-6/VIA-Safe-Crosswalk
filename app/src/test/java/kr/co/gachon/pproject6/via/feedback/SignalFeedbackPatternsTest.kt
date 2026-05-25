package kr.co.gachon.pproject6.via.feedback

import kr.co.gachon.pproject6.via.ml.AdvisoryState
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNotSame
import org.junit.Test

class SignalFeedbackPatternsTest {
    @Test
    fun runtimeAdvisoryStatesMapToSharedPracticePatterns() {
        assertArrayEquals(
            SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.RED_CONFIRMED),
            SignalFeedbackPatterns.forAdvisoryState(AdvisoryState.RED_CONFIRMED)
        )
        assertArrayEquals(
            SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.GREEN_CONFIRMED),
            SignalFeedbackPatterns.forAdvisoryState(AdvisoryState.GREEN_CONFIRMED)
        )
        assertArrayEquals(
            SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.GREEN_WITH_CAUTION),
            SignalFeedbackPatterns.forAdvisoryState(AdvisoryState.GREEN_WITH_CAUTION)
        )
        assertArrayEquals(
            SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.WAIT_OR_UNCERTAIN),
            SignalFeedbackPatterns.forAdvisoryState(AdvisoryState.UNCERTAIN_VIEW)
        )
        assertArrayEquals(
            SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.WAIT_OR_UNCERTAIN),
            SignalFeedbackPatterns.forAdvisoryState(AdvisoryState.TRANSITION_WAIT)
        )
    }

    @Test
    fun patternsReturnDefensiveCopies() {
        val first = SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.RED_CONFIRMED)
        val second = SignalFeedbackPatterns.copyOf(SignalFeedbackPattern.RED_CONFIRMED)

        assertNotSame(first, second)
        first[1] = 1
        assertArrayEquals(longArrayOf(0, 400, 200, 400), second)
    }
}
