package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.ui.OverlayView

data class SignalAnalysisResult(
    val boxesToShow: List<OverlayView.BoundingBox>,
    val targetBox: OverlayView.BoundingBox?,
    val targetScore: Float,
    val targetClassName: String,
    val trafficLightCount: Int,
    val vehicleTrafficLightCount: Int = 0,
    val multipleSignalDetected: Boolean,
    val needsZoomSuggestion: Boolean,
    val targetRecentlyReacquired: Boolean,
    val recentMatchedClusterChangeCount: Int,
    val trafficState: TrafficLightState,
    val userGuidanceState: UserGuidanceState,
    val guidancePhase: GuidancePhase,
    val guidanceBlockReason: GuidanceBlockReason,
    val guidanceContinuityTier: GuidanceContinuityTier,
    val handoffDecision: CrosswalkHandoffDecision,
    val crossingSupportSnapshot: CrossingSupportSnapshot,
    val occupancyCaution: Boolean,
    val occupancyCautionLabels: List<String>,
    val advisoryState: AdvisoryState = AdvisoryState.UNCERTAIN_VIEW,
    val advisoryConfidenceLevel: AdvisoryConfidenceLevel = AdvisoryConfidenceLevel.LOW,
    val advisoryConfidenceScore: Int = 0,
    val advisoryConfidenceReasons: List<AdvisoryConfidenceReason> = emptyList(),
    val advisoryTitleText: String = "",
    val advisoryDetailText: String = "",
    val advisorySpeechText: String = ""
)

fun SignalAnalysisResult.toGuidanceSnapshot(): GuidanceSnapshot {
    return GuidanceSnapshot(
        trafficState = trafficState,
        userGuidanceState = userGuidanceState,
        guidancePhase = guidancePhase,
        guidanceBlockReason = guidanceBlockReason,
        guidanceContinuityTier = guidanceContinuityTier,
        handoffDecision = handoffDecision,
        occupancyCaution = occupancyCaution
    )
}

fun SignalAnalysisResult.withGuidanceSnapshot(snapshot: GuidanceSnapshot): SignalAnalysisResult {
    return copy(
        trafficState = snapshot.trafficState,
        userGuidanceState = snapshot.userGuidanceState,
        guidancePhase = snapshot.guidancePhase,
        guidanceBlockReason = snapshot.guidanceBlockReason,
        guidanceContinuityTier = snapshot.guidanceContinuityTier,
        handoffDecision = snapshot.handoffDecision,
        occupancyCaution = snapshot.occupancyCaution
    )
}

fun SignalAnalysisResult.withAdvisoryAssessment(assessment: AdvisoryAssessment): SignalAnalysisResult {
    return copy(
        advisoryState = assessment.state,
        advisoryConfidenceLevel = assessment.confidenceLevel,
        advisoryConfidenceScore = assessment.confidenceScore,
        advisoryConfidenceReasons = assessment.confidenceReasons,
        advisoryTitleText = assessment.titleText,
        advisoryDetailText = assessment.detailText,
        advisorySpeechText = assessment.speechText
    )
}
