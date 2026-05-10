package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.context.MapFeatureKind
import kr.co.gachon.pproject6.via.context.MapFeatureSource
import kr.co.gachon.pproject6.via.context.MapProximitySnapshot
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
        assertTrue(advisory.detailText.contains("다음 신호 전환"))
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
        assertEquals("보행자 신호 초록으로 보임", advisory.titleText)
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
        assertTrue(advisory.detailText.contains("신호가 작지만"))
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
        assertTrue(advisory.detailText.contains("여러 보행자 신호"))
    }

    @Test
    fun vehicleSignalOnlyConflictStillSuppressesWalkAllowedGreen() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
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
        assertEquals("초록으로 보이나 주의 필요", advisory.titleText)
        assertTrue(advisory.detailText.contains("점유"))
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

        assertEquals("보행자 신호 빨간색으로 보임", red.titleText)
        assertEquals("다음 신호 대기 권장", wait.titleText)
        assertEquals("신호 확인 불확실", uncertain.titleText)
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
        assertTrue(advisory.speechText.contains("차량 신호"))
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
        assertTrue(advisory.detailText.contains("차량 신호도 보이나"))
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
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): SignalAnalysisResult {
        return SignalAnalysisResult(
            boxesToShow = emptyList(),
            targetBox = null,
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
}
