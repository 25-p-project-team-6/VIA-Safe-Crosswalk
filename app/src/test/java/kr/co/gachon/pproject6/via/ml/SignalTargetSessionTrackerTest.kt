package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalTargetSessionTrackerTest {
    private val firstTarget = NormalizedTargetBox(0.10f, 0.10f, 0.20f, 0.22f)
    private val sameTargetAfterGap = NormalizedTargetBox(0.11f, 0.11f, 0.21f, 0.23f)
    private val nextIntersectionTarget = NormalizedTargetBox(0.68f, 0.12f, 0.78f, 0.24f)

    @Test
    fun reacquiringDifferentTargetAfterGapRequestsReset() {
        val tracker = SignalTargetSessionTracker()

        assertFalse(tracker.onFrame(firstTarget))
        repeat(5) {
            assertFalse(tracker.onFrame(null))
        }
        assertTrue(tracker.onFrame(nextIntersectionTarget))
    }

    @Test
    fun reacquiringSameTargetAfterGapDoesNotReset() {
        val tracker = SignalTargetSessionTracker()

        assertFalse(tracker.onFrame(firstTarget))
        repeat(5) {
            assertFalse(tracker.onFrame(null))
        }
        assertFalse(tracker.onFrame(sameTargetAfterGap))
    }

    @Test
    fun switchingTargetsWithoutGapDoesNotReset() {
        val tracker = SignalTargetSessionTracker()

        assertFalse(tracker.onFrame(firstTarget))
        assertFalse(tracker.onFrame(nextIntersectionTarget))
    }

    @Test
    fun briefSingleFrameLossDoesNotResetEvenIfTargetMoves() {
        val tracker = SignalTargetSessionTracker()

        assertFalse(tracker.onFrame(firstTarget))
        assertFalse(tracker.onFrame(null))
        assertFalse(tracker.onFrame(nextIntersectionTarget))
    }

    @Test
    fun resetClearsPreviousTargetHistory() {
        val tracker = SignalTargetSessionTracker()

        assertFalse(tracker.onFrame(firstTarget))
        repeat(5) {
            assertFalse(tracker.onFrame(null))
        }
        tracker.reset()

        assertFalse(tracker.onFrame(nextIntersectionTarget))
    }
}
