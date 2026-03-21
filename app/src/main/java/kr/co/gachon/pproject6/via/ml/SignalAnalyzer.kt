package kr.co.gachon.pproject6.via.ml

import android.graphics.Bitmap
import kr.co.gachon.pproject6.via.ui.OverlayView

class SignalAnalyzer(
    private val objectTracker: ObjectTracker = ObjectTracker(),
    private val walkSignalPolicy: ConservativeWalkSignalPolicy =
        ConservativeWalkSignalPolicy(GuidanceTuningDefaults.walkSignalConfig),
    private val riskObjectEvaluator: RiskObjectEvaluator =
        RiskObjectEvaluator(GuidanceTuningDefaults.riskObjectConfig)
) {
    fun analyze(
        bitmap: Bitmap,
        rawBoxes: List<OverlayView.BoundingBox>,
        enableTrafficLogic: Boolean,
        enableHighlight: Boolean
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
                hasBlockingRisk = false,
                blockingRiskLabels = emptyList()
            )
        }

        val correctedBoxes = PostProcessor.applyColorCorrection(bitmap, rawBoxes)
        val targetData = objectTracker.selectTarget(correctedBoxes)

        val targetBox = targetData?.first
        if (targetBox != null && enableHighlight) {
            targetBox.isTarget = true
        }

        val trafficState = PostProcessor.updateTrafficLightState(targetBox)
        val blockingRisks = riskObjectEvaluator.findBlockingRisks(correctedBoxes)
        val guidanceDecision = walkSignalPolicy.update(
            state = trafficState,
            hasBlockingRisk = blockingRisks.isNotEmpty()
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
            hasBlockingRisk = blockingRisks.isNotEmpty(),
            blockingRiskLabels = blockingRisks.map { it.clsName }.distinct().sorted()
        )
    }

    fun reset() {
        objectTracker.reset()
        PostProcessor.resetState()
        walkSignalPolicy.reset()
    }
}
