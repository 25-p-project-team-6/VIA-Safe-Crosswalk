package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.ui.OverlayView

data class SignalAnalysisResult(
    val boxesToShow: List<OverlayView.BoundingBox>,
    val targetBox: OverlayView.BoundingBox?,
    val targetScore: Float,
    val targetClassName: String,
    val trafficState: TrafficLightState,
    val userGuidanceState: UserGuidanceState,
    val guidancePhase: GuidancePhase,
    val guidanceBlockReason: GuidanceBlockReason,
    val guidanceContinuityTier: GuidanceContinuityTier,
    val handoffDecision: CrosswalkHandoffDecision,
    val crossingSupportSnapshot: CrossingSupportSnapshot,
    val occupancyCaution: Boolean,
    val occupancyCautionLabels: List<String>
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
