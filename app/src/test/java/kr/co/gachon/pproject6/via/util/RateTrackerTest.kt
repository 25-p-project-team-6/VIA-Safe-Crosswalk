package kr.co.gachon.pproject6.via.util

import org.junit.Assert.assertEquals
import org.junit.Test

class RateTrackerTest {
    @Test
    fun markCalculatesRateFromInjectedClock() {
        var currentTime = 0L
        val tracker = RateTracker(label = "Input FPS", timeProvider = { currentTime })

        tracker.mark()
        currentTime = 250L
        tracker.mark()
        currentTime = 500L
        tracker.mark()
        currentTime = 1_000L
        tracker.mark()

        assertEquals("Input FPS: 4.00", tracker.rateStr)
    }

    @Test
    fun displayStringCanOverrideLabelWithoutChangingTrackedRate() {
        var currentTime = 0L
        val tracker = RateTracker(label = "Camera FPS", timeProvider = { currentTime })

        tracker.mark()
        currentTime = 500L
        tracker.mark()
        currentTime = 1_000L
        tracker.mark()

        assertEquals("Camera FPS: 3.00", tracker.rateStr)
        assertEquals("Replay FPS: 3.00", tracker.displayString("Replay FPS"))
        assertEquals("Camera FPS: 3.00", tracker.rateStr)
    }

    @Test
    fun clearResetsRateLabel() {
        var currentTime = 0L
        val tracker = RateTracker(label = "Input FPS", timeProvider = { currentTime })

        tracker.mark()
        currentTime = 1_000L
        tracker.mark()
        tracker.clear()

        assertEquals("Input FPS: 0.00", tracker.rateStr)
        assertEquals("Replay FPS: 0.00", tracker.displayString("Replay FPS"))
    }
}
