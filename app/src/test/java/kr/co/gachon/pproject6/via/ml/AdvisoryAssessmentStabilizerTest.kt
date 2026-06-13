package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class AdvisoryAssessmentStabilizerTest {
    @Test
    fun vehicleSignalBlipDoesNotReplaceMissingSignalCopyBeforeHold() {
        var currentTime = 0L
        val stabilizer =
            AdvisoryAssessmentStabilizer(
                config = AdvisoryAssessmentStabilizerConfig(volatileUncertainSwitchHoldMs = 1_000L),
                timeProvider = { currentTime }
            )

        assertEquals("신호 미탐지.", stabilizer.stabilize(missingSignal()).detailText)

        currentTime = 100L
        val firstVehicleFrame = stabilizer.stabilize(vehicleSignalOnly())

        assertEquals("확인 중", firstVehicleFrame.titleText)
        assertEquals("신호 미탐지.", firstVehicleFrame.detailText)
        assertEquals("확인 필요.", firstVehicleFrame.speechText)

        currentTime = 900L
        assertEquals("신호 미탐지.", stabilizer.stabilize(vehicleSignalOnly()).detailText)
    }

    @Test
    fun persistentVehicleSignalReplacesMissingSignalCopyAfterHold() {
        var currentTime = 0L
        val stabilizer =
            AdvisoryAssessmentStabilizer(
                config = AdvisoryAssessmentStabilizerConfig(volatileUncertainSwitchHoldMs = 1_000L),
                timeProvider = { currentTime }
            )

        stabilizer.stabilize(missingSignal())
        currentTime = 100L
        stabilizer.stabilize(vehicleSignalOnly())

        currentTime = 1_100L
        val committedVehicle = stabilizer.stabilize(vehicleSignalOnly())

        assertEquals("차량 신호", committedVehicle.titleText)
        assertEquals("차량 신호.", committedVehicle.detailText)
        assertEquals("차량 신호.", committedVehicle.speechText)
    }

    @Test
    fun missingSignalBlipDoesNotReplaceVehicleSignalCopyBeforeHold() {
        var currentTime = 0L
        val stabilizer =
            AdvisoryAssessmentStabilizer(
                config = AdvisoryAssessmentStabilizerConfig(volatileUncertainSwitchHoldMs = 1_000L),
                timeProvider = { currentTime }
            )

        assertEquals("차량 신호.", stabilizer.stabilize(vehicleSignalOnly()).detailText)

        currentTime = 100L
        val firstMissingFrame = stabilizer.stabilize(missingSignal())

        assertEquals("차량 신호", firstMissingFrame.titleText)
        assertEquals("차량 신호.", firstMissingFrame.detailText)
        assertEquals("차량 신호.", firstMissingFrame.speechText)
    }

    @Test
    fun confirmedSignalResetsVolatileUncertainHold() {
        var currentTime = 0L
        val stabilizer =
            AdvisoryAssessmentStabilizer(
                config = AdvisoryAssessmentStabilizerConfig(volatileUncertainSwitchHoldMs = 1_000L),
                timeProvider = { currentTime }
            )

        stabilizer.stabilize(missingSignal())
        currentTime = 100L
        stabilizer.stabilize(vehicleSignalOnly())
        currentTime = 200L
        assertEquals("빨간불 확인.", stabilizer.stabilize(redConfirmed()).detailText)

        currentTime = 300L
        assertEquals("차량 신호.", stabilizer.stabilize(vehicleSignalOnly()).detailText)
    }

    @Test
    fun nonVolatileUncertainReasonsBypassHold() {
        var currentTime = 0L
        val stabilizer =
            AdvisoryAssessmentStabilizer(
                config = AdvisoryAssessmentStabilizerConfig(volatileUncertainSwitchHoldMs = 1_000L),
                timeProvider = { currentTime }
            )

        stabilizer.stabilize(missingSignal())
        currentTime = 100L
        val multipleSignals = stabilizer.stabilize(multipleSignals())

        assertEquals("여러 신호", multipleSignals.titleText)
        assertEquals("여러 신호.", multipleSignals.detailText)
    }

    private fun missingSignal(): AdvisoryAssessment {
        return uncertain(
            titleText = "확인 중",
            detailText = "신호 미탐지.",
            speechText = "확인 필요."
        )
    }

    private fun vehicleSignalOnly(): AdvisoryAssessment {
        return uncertain(
            titleText = "차량 신호",
            detailText = "차량 신호.",
            speechText = "차량 신호.",
            reasons = listOf(AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE)
        )
    }

    private fun multipleSignals(): AdvisoryAssessment {
        return uncertain(
            titleText = "여러 신호",
            detailText = "여러 신호.",
            speechText = "여러 신호.",
            reasons = listOf(AdvisoryConfidenceReason.MULTIPLE_SIGNALS)
        )
    }

    private fun redConfirmed(): AdvisoryAssessment {
        return AdvisoryAssessment(
            state = AdvisoryState.RED_CONFIRMED,
            confidenceLevel = AdvisoryConfidenceLevel.HIGH,
            confidenceScore = 80,
            confidenceReasons = listOf(AdvisoryConfidenceReason.STABLE_SIGNAL),
            titleText = "빨간불",
            detailText = "빨간불 확인.",
            speechText = "빨간불."
        )
    }

    private fun uncertain(
        titleText: String,
        detailText: String,
        speechText: String,
        reasons: List<AdvisoryConfidenceReason> = emptyList()
    ): AdvisoryAssessment {
        return AdvisoryAssessment(
            state = AdvisoryState.UNCERTAIN_VIEW,
            confidenceLevel = AdvisoryConfidenceLevel.LOW,
            confidenceScore = 40,
            confidenceReasons = reasons,
            titleText = titleText,
            detailText = detailText,
            speechText = speechText
        )
    }
}
