package kr.co.gachon.pproject6.via.feedback

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

        assertTrue(policy.shouldEmit("빨간불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))
        assertFalse(policy.shouldEmit("빨간불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))

        currentTime += 2_999L
        assertFalse(policy.shouldEmit("빨간불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))

        currentTime += 1L
        assertTrue(policy.shouldEmit("빨간불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))
    }

    @Test
    fun messageChangeEmitsImmediately() {
        var currentTime = 0L
        val policy = SignalFeedbackPolicy(
            timingConfig = SignalFeedbackTimingConfig(
                actionRepeatIntervalMs = 3_000L,
                waitRepeatIntervalMs = 6_000L
            ),
            timeProvider = { currentTime }
        )

        assertTrue(policy.shouldEmit("빨간불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))

        currentTime += 500L
        assertTrue(policy.shouldEmit("다음 신호 전환을 기다립니다", FeedbackRepeatFamily.WAIT_LIKE))
    }

    @Test
    fun waitLikeMessagesRepeatLessOftenThanActionLikeMessages() {
        var currentTime = 0L
        val policy = SignalFeedbackPolicy(
            timingConfig = SignalFeedbackTimingConfig(
                actionRepeatIntervalMs = 3_000L,
                waitRepeatIntervalMs = 6_000L
            ),
            timeProvider = { currentTime }
        )

        assertTrue(policy.shouldEmit("신호 확인이 불안정합니다", FeedbackRepeatFamily.WAIT_LIKE))

        currentTime += 5_999L
        assertFalse(policy.shouldEmit("신호 확인이 불안정합니다", FeedbackRepeatFamily.WAIT_LIKE))

        currentTime += 1L
        assertTrue(policy.shouldEmit("신호 확인이 불안정합니다", FeedbackRepeatFamily.WAIT_LIKE))

        currentTime += 100L
        assertTrue(policy.shouldEmit("초록불이 확인됩니다", FeedbackRepeatFamily.ACTION_LIKE))
    }
}
