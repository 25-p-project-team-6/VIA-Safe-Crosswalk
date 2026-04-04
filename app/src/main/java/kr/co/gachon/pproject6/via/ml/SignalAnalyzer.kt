package kr.co.gachon.pproject6.via.ml

import android.graphics.Bitmap
import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.ui.OverlayView

class SignalAnalyzer(
    private val objectTracker: ObjectTracker = ObjectTracker(),
    private val walkSignalPolicy: ConservativeWalkSignalPolicy =
        ConservativeWalkSignalPolicy(GuidanceTuningDefaults.walkSignalConfig),
    private val occupancyEvaluator: CrosswalkOccupancyEvaluator =
        CrosswalkOccupancyEvaluator(GuidanceTuningDefaults.occupancyConfig)
) {
    fun analyze(
        bitmap: Bitmap,
        rawBoxes: List<OverlayView.BoundingBox>,
        enableTrafficLogic: Boolean,
        enableHighlight: Boolean,
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): SignalAnalysisResult {
        if (!enableTrafficLogic) {
            reset()
            return SignalAnalysisResult(
                boxesToShow = rawBoxes,
                targetBox = null,
                targetScore = 0f,
                targetClassName = "None",
                trafficState = TrafficLightState.UNKNOWN,
                userGuidanceState = UserGuidanceState.WAIT,
                guidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE,
                guidanceBlockReason = GuidanceBlockReason.NO_SIGNAL,
                guidanceContinuityTier = GuidanceContinuityTier.NONE,
                handoffDecision = CrosswalkHandoffDecision.NONE,
                crossingSupportSnapshot = crossingSupportSnapshot,
                occupancyCaution = false,
                occupancyCautionLabels = emptyList()
            )
        }

        val correctedBoxes = PostProcessor.applyColorCorrection(bitmap, rawBoxes)
        val targetData = objectTracker.selectTarget(correctedBoxes)

        val targetBox = targetData?.first
        if (targetBox != null && enableHighlight) {
            targetBox.isTarget = true
        }

        val trafficState = PostProcessor.updateTrafficLightState(targetBox)
        val guidanceDecision = walkSignalPolicy.update(
            state = trafficState,
            crossingSupportSnapshot = crossingSupportSnapshot
        )
        val activeOccupancy =
            occupancyEvaluator.findActiveOccupancy(
                boxes = correctedBoxes,
                eligible = guidanceDecision.state == UserGuidanceState.GO
            )

        return SignalAnalysisResult(
            boxesToShow = correctedBoxes,
            targetBox = targetBox,
            targetScore = targetData?.second ?: 0f,
            targetClassName = targetBox?.clsName ?: "None",
            trafficState = trafficState,
            userGuidanceState = guidanceDecision.state,
            guidancePhase = guidanceDecision.phase,
            guidanceBlockReason = guidanceDecision.blockReason,
            guidanceContinuityTier = guidanceDecision.continuityTier,
            handoffDecision = guidanceDecision.handoffDecision,
            crossingSupportSnapshot = crossingSupportSnapshot,
            occupancyCaution = activeOccupancy.isNotEmpty(),
            occupancyCautionLabels = activeOccupancy.map { it.clsName }.distinct().sorted()
        )
    }

    fun reset() {
        objectTracker.reset()
        PostProcessor.resetState()
        walkSignalPolicy.reset()
        occupancyEvaluator.reset()
    }
}
