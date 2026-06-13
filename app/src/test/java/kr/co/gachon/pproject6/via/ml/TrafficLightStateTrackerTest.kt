package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class TrafficLightStateTrackerTest {
    @Test
    fun lowConfidenceSignalRequiresMinimumObservationTime() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))
    }

    @Test
    fun lowConfidenceSignalDoesNotCommitBeforeRequiredTime() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))
    }

    @Test
    fun highConfidenceSignalStillRequiresConfiguredTimeByDefault() {
        var currentTime = 10_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, true))

        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, true))

        currentTime += 125
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, true))
    }

    @Test
    fun highConfidenceFastTrackCanStillBeEnabledExplicitly() {
        var currentTime = 10_000L
        val tracker = TrafficLightStateTracker(
            config = TrafficLightStateTrackingConfig(allowHighConfidenceImmediateCommit = true),
            timeProvider = { currentTime }
        )

        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, true))

        currentTime += 2_499
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 2
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }

    @Test
    fun redCandidateSurvivesBriefCameraArtifactUnknownGapsBeforeConfirmation() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 80
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 80
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 80
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 20
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))
    }

    @Test
    fun redCandidateResetsAfterLongCameraArtifactUnknownGapBeforeConfirmation() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 60
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 1
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 249
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 1
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))
    }

    @Test
    fun greenCandidateDoesNotBridgeUnknownGapsBeforeConfirmation() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 80
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 180
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 249
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 1
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))
    }

    @Test
    fun oppositeStateRequiresSustainedObservationTimeBeforeSwitching() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 100
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 100
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))
    }

    @Test
    fun briefOppositeColorNoiseDoesNotFlipAcceptedState() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))
        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))
        currentTime += 125
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 100
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.RED, false))

        currentTime += 100
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))
    }

    @Test
    fun acceptedRedStateClearsAfterTimeoutWithoutEvidence() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))

        currentTime += 125
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))

        currentTime += 4_999L
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 2L
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }

    @Test
    fun acceptedGreenStateSurvivesBriefUnknownGap() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))
        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.GREEN, false))
        currentTime += 125
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.GREEN, false))

        currentTime += 2_000L
        assertEquals(TrafficLightState.GREEN, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 501L
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }

    @Test
    fun acceptedRedReacquisitionRefreshesPersistenceAfterCameraArtifactGap() {
        var currentTime = 1_000L
        val tracker = TrafficLightStateTracker(timeProvider = { currentTime })

        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))
        currentTime += 125
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.RED, false))
        currentTime += 125
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))

        currentTime += 4_000L
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 900L
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.RED, false))

        currentTime += 4_999L
        assertEquals(TrafficLightState.RED, tracker.update(TrafficLightState.UNKNOWN, false))

        currentTime += 2L
        assertEquals(TrafficLightState.UNKNOWN, tracker.update(TrafficLightState.UNKNOWN, false))
    }
}
