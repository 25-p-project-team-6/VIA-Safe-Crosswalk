package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class TrafficLightStateTrackerTest {
    @Test
    fun lowConfidenceSignalRequiresThreeConsecutiveFrames() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))
    }

    @Test
    fun highConfidenceSignalFastTracksAndPersistsUntilTimeout() {
        var currentTime = 10_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, true))

        currentTime += 1_000
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 5_100
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }

    @Test
    fun oppositeStateRequiresRepeatedEvidenceBeforeSwitching() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, true))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 100
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))
    }

    @Test
    fun acceptedStateClearsAfterTimeoutWithoutEvidence() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, true))

        currentTime += 4_999L
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 2L
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }
}
