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
                crossingSupportSnapshot =
                    CrossingSupportSnapshot(
                        mapProximitySnapshot =
                            MapProximitySnapshot(
                                isNearKnownFeature = true,
                                matchedClusterId = "cluster-a",
                                matchedKind = MapFeatureKind.PED_SIGNAL,
                                matchedSource = MapFeatureSource.HYBRID
                            )
                    )
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.GREEN_CONFIRMED, advisory.state)
        assertEquals(AdvisoryConfidenceLevel.HIGH, advisory.confidenceLevel)
    }

    @Test
    fun multipleSignalsTriggerUncertainView() {
        val result =
            baseResult(
                trafficState = TrafficLightState.GREEN,
                userGuidanceState = UserGuidanceState.GO,
                trafficLightCount = 2,
                multipleSignalDetected = true,
                needsZoomSuggestion = true
            )

        val advisory = evaluator.evaluate(result)

        assertEquals(AdvisoryState.UNCERTAIN_VIEW, advisory.state)
        assertTrue(advisory.speechText.contains("여러 개") || advisory.speechText.contains("가까이"))
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
        assertTrue(advisory.detailText.contains("점유"))
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

    private fun baseResult(
        trafficState: TrafficLightState = TrafficLightState.UNKNOWN,
        userGuidanceState: UserGuidanceState = UserGuidanceState.WAIT,
        guidanceBlockReason: GuidanceBlockReason = GuidanceBlockReason.NO_SIGNAL,
        targetScore: Float = 0.75f,
        trafficLightCount: Int = 1,
        vehicleTrafficLightCount: Int = 0,
        multipleSignalDetected: Boolean = false,
        needsZoomSuggestion: Boolean = false,
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
            targetRecentlyReacquired = false,
            recentMatchedClusterChangeCount = 0,
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
