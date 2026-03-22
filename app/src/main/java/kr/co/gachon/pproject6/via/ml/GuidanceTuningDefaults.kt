package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.feedback.SignalFeedbackTimingConfig

object GuidanceTuningDefaults {
    val signalStateTrackingConfig = TrafficLightStateTrackingConfig(
        confirmDurationMs = 250L,
        switchConfirmDurationMs = 400L,
        redPersistenceDurationMs = 5_000L,
        greenPersistenceDurationMs = 2_500L,
        allowHighConfidenceImmediateCommit = false
    )

    val walkSignalConfig = ConservativeWalkSignalConfig(
        requireRedBaselineBeforeGo = true,
        resetToBaselineOnUnknownDuringWalk = true,
        blockGoWhenRiskDetected = true,
        preserveReadyBaselineMs = 2_500L
    )

    val guidanceStabilizerConfig = GuidanceStateStabilizerConfig(
        actionConfirmFrames = 2,
        waitConfirmFrames = 3
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
            append("signal hold=")
            append(signalStateTrackingConfig.confirmDurationMs)
            append("ms")
            append(", switch hold=")
            append(signalStateTrackingConfig.switchConfirmDurationMs)
            append("ms")
            append(", green keep=")
            append(signalStateTrackingConfig.greenPersistenceDurationMs)
            append("ms")
            append(", go/stop frames=")
            append(guidanceStabilizerConfig.actionConfirmFrames)
            append(", wait frames=")
            append(guidanceStabilizerConfig.waitConfirmFrames)
            append(", ready hold=")
            append(walkSignalConfig.preserveReadyBaselineMs)
            append("ms")
            append(", ")
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
