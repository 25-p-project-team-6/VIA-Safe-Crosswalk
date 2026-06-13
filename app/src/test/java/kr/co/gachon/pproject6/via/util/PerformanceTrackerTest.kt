package kr.co.gachon.pproject6.via.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PerformanceTrackerTest {
    @Test
    fun updateCalculatesAverageLatencyAndFpsFromInjectedClock() {
        var currentTime = 0L
        val tracker = PerformanceTracker(timeProvider = { currentTime })

        tracker.update(100)
        currentTime = 1_000
        tracker.update(200)

        assertEquals("FPS: 2.00", tracker.currentFpsStr)
        assertEquals("Avg FPS: 2.00", tracker.avgFpsStr)
        assertEquals("Avg Latency: 150ms", tracker.avgLatencyStr)
    }

    @Test
    fun clearResetsDisplayedStats() {
        var currentTime = 0L
        val tracker = PerformanceTracker(timeProvider = { currentTime })

        tracker.update(120)
        currentTime = 1_000
        tracker.update(120)
        tracker.clear()

        assertEquals("FPS: 0", tracker.currentFpsStr)
        assertEquals("Avg FPS: 0", tracker.avgFpsStr)
        assertEquals("Avg Latency: 0ms", tracker.avgLatencyStr)
        assertEquals("Stages: n/a", tracker.stageBreakdownStr)
    }

    @Test
    fun updateBuildsAverageStageBreakdownFromSlidingWindow() {
        var currentTime = 0L
        val tracker = PerformanceTracker(timeProvider = { currentTime })

        tracker.update(
            inferenceTime = 10,
            stageDurationsMs = linkedMapOf(
                "copy" to 4L,
                "rotate" to 2L,
                "detect" to 10L
            )
        )
        currentTime = 1_000L
        tracker.update(
            inferenceTime = 14,
            stageDurationsMs = linkedMapOf(
                "copy" to 6L,
                "rotate" to 4L,
                "detect" to 14L,
                "analyze" to 2L
            )
        )

        assertTrue(tracker.stageBreakdownStr.contains("copy 5ms"))
        assertTrue(tracker.stageBreakdownStr.contains("rotate 3ms"))
        assertTrue(tracker.stageBreakdownStr.contains("detect 12ms"))
        assertTrue(tracker.stageBreakdownStr.contains("analyze 1ms"))
    }
}
