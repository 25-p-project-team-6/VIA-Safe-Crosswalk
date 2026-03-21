package kr.co.gachon.pproject6.via.feedback

import kr.co.gachon.pproject6.via.ml.UserGuidanceState
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalFeedbackPolicyTest {
    @Test
    fun emitsImmediatelyAndRepeatsAfterInterval() {
        var currentTime = 0L
        val policy = SignalFeedbackPolicy(
            timingConfig = SignalFeedbackTimingConfig(
                actionRepeatIntervalMs = 3_000L,
                waitRepeatIntervalMs = 6_000L
            ),
            timeProvider = { currentTime }
        )

        assertTrue(policy.shouldEmit(UserGuidanceState.STOP))
        assertFalse(policy.shouldEmit(UserGuidanceState.STOP))

        currentTime += 2_999L
        assertFalse(policy.shouldEmit(UserGuidanceState.STOP))

        currentTime += 1L
        assertTrue(policy.shouldEmit(UserGuidanceState.STOP))
    }

    @Test
    fun stateChangeEmitsImmediately() {
        var currentTime = 0L
        val policy = SignalFeedbackPolicy(
            timingConfig = SignalFeedbackTimingConfig(
                actionRepeatIntervalMs = 3_000L,
                waitRepeatIntervalMs = 6_000L
            ),
            timeProvider = { currentTime }
        )

        assertTrue(policy.shouldEmit(UserGuidanceState.STOP))

        currentTime += 500L
        assertTrue(policy.shouldEmit(UserGuidanceState.GO))
    }

    @Test
    fun waitRepeatsLessOftenThanActionStates() {
        var currentTime = 0L
        val policy = SignalFeedbackPolicy(
            timingConfig = SignalFeedbackTimingConfig(
                actionRepeatIntervalMs = 3_000L,
                waitRepeatIntervalMs = 6_000L
            ),
            timeProvider = { currentTime }
        )

        assertTrue(policy.shouldEmit(UserGuidanceState.WAIT))

        currentTime += 5_999L
        assertFalse(policy.shouldEmit(UserGuidanceState.WAIT))

        currentTime += 1L
        assertTrue(policy.shouldEmit(UserGuidanceState.WAIT))

        currentTime += 100L
        assertTrue(policy.shouldEmit(UserGuidanceState.GO))
    }
}
