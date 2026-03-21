package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.feedback.SignalFeedbackTimingConfig

object GuidanceTuningDefaults {
    val walkSignalConfig = ConservativeWalkSignalConfig(
        requireRedBaselineBeforeGo = true,
        resetToBaselineOnUnknownDuringWalk = true,
        blockGoWhenRiskDetected = true
    )

    val riskObjectConfig = RiskObjectConfig(
        blockingLabels = setOf("bicycle", "car", "motorcycle", "bus", "train", "truck"),
        minScore = 0.35f,
        minBottom = 0.45f,
        minArea = 0.015f,
        blockingCenterBand = 0.2f..0.8f
    )

    val feedbackTimingConfig = SignalFeedbackTimingConfig(
        actionRepeatIntervalMs = 4_000L,
        waitRepeatIntervalMs = 8_000L
    )

    fun toDebugSummary(): String {
        return buildString {
            append("risk score≥")
            append("%.2f".format(riskObjectConfig.minScore))
            append(", bottom≥")
            append("%.2f".format(riskObjectConfig.minBottom))
            append(", area≥")
            append("%.3f".format(riskObjectConfig.minArea))
            append(", band=")
            append("%.2f".format(riskObjectConfig.blockingCenterBand.start))
            append("..")
            append("%.2f".format(riskObjectConfig.blockingCenterBand.endInclusive))
            append(", wait=")
            append(feedbackTimingConfig.waitRepeatIntervalMs)
            append("ms, action=")
            append(feedbackTimingConfig.actionRepeatIntervalMs)
            append("ms")
        }
    }
}
