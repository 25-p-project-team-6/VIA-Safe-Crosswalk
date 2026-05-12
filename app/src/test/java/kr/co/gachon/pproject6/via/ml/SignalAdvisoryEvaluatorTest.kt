package kr.co.gachon.pproject6.via.ml

import android.graphics.RectF
import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.context.MapFeatureKind
import kr.co.gachon.pproject6.via.context.MapFeatureSource
import kr.co.gachon.pproject6.via.context.MapProximitySnapshot
import kr.co.gachon.pproject6.via.ui.OverlayView
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalAdvisoryEvaluatorTest {
    private val evaluator = SignalAdvisoryEvaluator(GuidanceTuningDefaults.advisoryConfig)

    @Test
    fun needRedBaselineMapsToTransitionWait() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.WAIT,
                guidanceBlockReason = GuidanceBlockReason.NEED_RED_BASELINE
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.TRANSITION_WAIT, advisory.state)
        assertEquals("전환 대기.", advisory.detailText)
        assertEquals("대기.", advisory.speechText)
    }

    @Test
    fun stableGreenWithSingleMatchedClusterBecomesGreenConfirmed() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                targetScore = 0.88f,
                trafficLightCount = 1,
                crossingSupportSnapshot = matchedPedSignalSnapshot()
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertEquals(AdvisoryConfidenceLevel.HIGH, advisory.confidenceLevel)
        assertEquals("초록불", advisory.titleText)
        assertEquals("초록 확인.", advisory.detailText)
        assertEquals("초록불.", advisory.speechText)
    }

    @Test
    fun walkAllowedGreenDoesNotFallBackToZoomSuggestionOnlyBecauseTargetIsSmall() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                targetScore = 0.82f,
                trafficLightCount = 1,
                needsZoomSuggestion = true
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.TARGET_SMALL))
        assertEquals("초록 확인.", advisory.detailText)
        assertTrue(!advisory.detailText.contains("신호가 작"))
    }

    @Test
    fun multiplePedestrianSignalsDoNotSuppressWalkAllowedGreen() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                trafficLightCount = 2,
                multipleSignalDetected = true,
                needsZoomSuggestion = true
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.MULTIPLE_SIGNALS))
        assertEquals("초록 확인.", advisory.detailText)
    }

    @Test
    fun reacquiredTargetDoesNotDestabilizeWalkAllowedGreenCopy() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                trafficLightCount = 1,
                targetRecentlyReacquired = true
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.TARGET_RECENTLY_REACQUIRED))
        assertEquals("초록 확인.", advisory.detailText)
    }

    @Test
    fun vehicleSignalOnlyWithoutWalkAllowedStaysUncertain() {
        val result =
            baseResult(
                trafficState = TrafficLightState.UNKNOWN,
                userGuidanceState = UserGuidanceState.WAIT,
                trafficLightCount = 0,
                vehicleTrafficLightCount = 1,
                multipleSignalDetected = false,
                needsZoomSuggestion = false
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.UNCERTAIN_VIEW, advisory.state)
        assertTrue(advisory.speechText.contains("여러 개") || advisory.speechText.contains("차량 신호"))
    }

    @Test
    fun walkAllowedGreenSurvivesBriefVehicleOnlyFrame() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                trafficLightCount = 0,
                vehicleTrafficLightCount = 1,
                targetScore = 0f
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE))
        assertEquals("초록 확인.", advisory.detailText)
    }

    @Test
    fun occupancyBecomesGreenWithCaution() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                occupancyCaution = true,
                occupancyCautionLabels = listOf(DetectionLabels.VEHICLE)
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_WITH_CAUTION, advisory.state)
        assertEquals("초록불 주의", advisory.titleText)
        assertEquals("차량 주의.", advisory.detailText)
        assertEquals("차량 주의.", advisory.speechText)
    }

    @Test
    fun lowVisionTitlesUseConsistentStateLanguage() {
        val red = evaluator.evaluate(baseResult(trafficState = TrafficLightState.RED))
        val wait =
            evaluator.evaluate(
                baseResult(
                    trafficState = TrafficLightState.GREEN,
                    guidanceBlockReason = GuidanceBlockReason.NEED_RED_BASELINE
                )
            )
        val uncertain = evaluator.evaluate(baseResult(trafficState = TrafficLightState.UNKNOWN))

        assertEquals("빨간불", red.titleText)
        assertEquals("대기 중", wait.titleText)
        assertEquals("확인 중", uncertain.titleText)
    }

    @Test
    fun vehicleSignalOnlyStaysUncertainAndRequestsPedestrianSignal() {
        val result =
            baseResult(
                trafficState = TrafficLightState.UNKNOWN,
                userGuidanceState = UserGuidanceState.WAIT,
                targetScore = 0f,
                trafficLightCount = 0,
                vehicleTrafficLightCount = 1
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.UNCERTAIN_VIEW, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE))
        assertEquals("차량 신호.", advisory.detailText)
        assertEquals("차량 신호.", advisory.speechText)
    }

    @Test
    fun uncertainCopyUsesConciseCrosswalkMissingTextWhenSignalExistsButMapDoesNot() {
        val result =
            baseResult(
                trafficState = TrafficLightState.UNKNOWN,
                userGuidanceState = UserGuidanceState.WAIT,
                targetBox = sampleTargetBox(),
                crossingSupportSnapshot = CrossingSupportSnapshot()
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.UNCERTAIN_VIEW, advisory.state)
        assertEquals("횡단보도 미탐지.", advisory.detailText)
        assertTrue(!advisory.detailText.contains("기준이 아직 약합니다"))
    }

    @Test
    fun pedestrianGreenTargetSurvivesVehicleSignalVisible() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                targetScore = 0.9f,
                trafficLightCount = 1,
                vehicleTrafficLightCount = 1,
                crossingSupportSnapshot = matchedPedSignalSnapshot()
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE))
        assertEquals("초록 확인.", advisory.detailText)
    }

    @Test
    fun redCanRemainConfirmedWhileVehicleSignalReasonLowersConfidence() {
        val result =
            baseResult(
                trafficState = TrafficLightState.RED,
                userGuidanceState = UserGuidanceState.STOP,
                targetScore = 0.85f,
                trafficLightCount = 1,
                vehicleTrafficLightCount = 1
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.RED_CONFIRMED, advisory.state)
        assertTrue(advisory.confidenceReasons.contains(AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE))
    }

    @Test
    fun advisoryCopyDoesNotUseCommandStyleCrossingInstructions() {
        val assessments =
            listOf(
                evaluator.evaluate(baseResult(trafficState = TrafficLightState.RED)),
                evaluator.evaluate(
                    baseResult(
                        trafficState = TrafficLightState.GREEN,
                        userGuidanceState = UserGuidanceState.GO,
                        crossingSupportSnapshot = matchedPedSignalSnapshot()
                    )
                ),
                evaluator.evaluate(
                    baseResult(
                        trafficState = TrafficLightState.GREEN,
                        userGuidanceState = UserGuidanceState.GO,
                        occupancyCaution = true,
                        occupancyCautionLabels = listOf(DetectionLabels.VEHICLE)
                    )
                ),
                evaluator.evaluate(
                    baseResult(
                        trafficState = TrafficLightState.GREEN,
                        guidanceBlockReason = GuidanceBlockReason.NEED_RED_BASELINE
                    )
                ),
                evaluator.evaluate(baseResult(trafficState = TrafficLightState.UNKNOWN))
            )
        val forbiddenPhrases = listOf("건너세요", "멈추세요", "건너도", "건널 수")

        assessments.forEach { assessment ->
            val text = "${assessment.titleText}\n${assessment.detailText}\n${assessment.speechText}"
            forbiddenPhrases.forEach { phrase ->
                assertTrue("advisory copy should not contain '$phrase'", !text.contains(phrase))
            }
        }
    }

    @Test
    fun confidenceThresholdsStayAlignedWithEvaluationPlan() {
        val config = GuidanceTuningDefaults.advisoryConfig

        assertEquals(75, config.highConfidenceMinScore)
        assertEquals(55, config.mediumConfidenceMinScore)
        assertTrue(config.highConfidenceMinScore > config.mediumConfidenceMinScore)
    }

    @Test
    fun ambiguityReasonsCoverEvaluationDimensions() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                trafficLightCount = 2,
                vehicleTrafficLightCount = 1,
                multipleSignalDetected = true,
                needsZoomSuggestion = true,
                targetRecentlyReacquired = true,
                recentMatchedClusterChangeCount = 1
            )

        val advisory = evaluator.evaluate(result)

        listOf(
            AdvisoryConfidenceReason.MULTIPLE_SIGNALS,
            AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE,
            AdvisoryConfidenceReason.TARGET_SMALL,
            AdvisoryConfidenceReason.TARGET_RECENTLY_REACQUIRED,
            AdvisoryConfidenceReason.MATCHED_CLUSTER_CHANGED
        ).forEach { reason ->
            assertTrue("missing $reason", advisory.confidenceReasons.contains(reason))
        }
        assertEquals(AdvisoryState.UNCERTAIN_VIEW, advisory.state)
    }

    private fun matchedPedSignalSnapshot(): CrossingSupportSnapshot {
        return CrossingSupportSnapshot(
            mapProximitySnapshot =
                MapProximitySnapshot(
                    isNearKnownFeature = true,
                    matchedClusterId = "cluster-a",
                    matchedKind = MapFeatureKind.PED_SIGNAL,
                    matchedSource = MapFeatureSource.HYBRID
                )
        )
    }

    private fun baseResult(
        trafficState: TrafficLightState = TrafficLightState.UNKNOWN,
        userGuidanceState: UserGuidanceState = UserGuidanceState.WAIT,
        guidanceBlockReason: GuidanceBlockReason = GuidanceBlockReason.NO_SIGNAL,
        targetScore: Float = 0.75f,
        trafficLightCount: Int = 1,
        vehicleTrafficLightCount: Int = 0,
        multipleSignalDetected: Boolean = false,
        needsZoomSuggestion: Boolean = false,
        targetRecentlyReacquired: Boolean = false,
        recentMatchedClusterChangeCount: Int = 0,
        occupancyCaution: Boolean = false,
        occupancyCautionLabels: List<String> = emptyList(),
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot(),
        targetBox: OverlayView.BoundingBox? = null
    ): SignalAnalysisResult {
        return SignalAnalysisResult(
            boxesToShow = emptyList(),
            targetBox = targetBox,
            targetScore = targetScore,
            targetClassName = DetectionLabels.HUMAN_GREEN,
            trafficLightCount = trafficLightCount,
            vehicleTrafficLightCount = vehicleTrafficLightCount,
            multipleSignalDetected = multipleSignalDetected,
            needsZoomSuggestion = needsZoomSuggestion,
            targetRecentlyReacquired = targetRecentlyReacquired,
            recentMatchedClusterChangeCount = recentMatchedClusterChangeCount,
            trafficState = trafficState,
            userGuidanceState = userGuidanceState,
            guidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE,
            guidanceBlockReason = guidanceBlockReason,
            guidanceContinuityTier = GuidanceContinuityTier.NONE,
            handoffDecision = CrosswalkHandoffDecision.NONE,
            crossingSupportSnapshot = crossingSupportSnapshot,
            occupancyCaution = occupancyCaution,
            occupancyCautionLabels = occupancyCautionLabels
        )
    }

    private fun sampleTargetBox(): OverlayView.BoundingBox {
        return OverlayView.BoundingBox(
            box = RectF(0.4f, 0.2f, 0.6f, 0.4f),
            clsName = DetectionLabels.HUMAN_GREEN,
            score = 0.75f
        )
    }
}
