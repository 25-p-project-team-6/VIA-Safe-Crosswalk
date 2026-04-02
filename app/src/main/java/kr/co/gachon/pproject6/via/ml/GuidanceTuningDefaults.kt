package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.context.CrossingSupportConfig
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
        walkAllowedUnknownGraceMs = 1_200L,
        walkAllowedUnknownGraceMatchedMs = 2_200L,
        walkAllowedUnknownGraceMatchedMovingMs = 3_500L,
        walkAllowedUnknownGraceMatchedMovingDownMs = 4_800L,
        sameCrossingMaxDistanceMeters = 20f,
        sameCrossingMaxElapsedMs = 12_000L
    )

    val crossingSupportConfig = CrossingSupportConfig(
        gyroMotionThresholdRadPerSec = 0.8f,
        motionHoldMs = 2_500L,
        lookingDownRawTiltRangeStartDegrees = -160f,
        lookingDownRawTiltRangeEndDegrees = -90f,
        lookingDownHoldMs = 900L,
        lookingUpRawTiltRangeStartDegrees = 90f,
        lookingUpRawTiltRangeEndDegrees = 120f,
        lookingUpHoldMs = 900L,
        locationSpeedThresholdMps = 0.7f,
        locationDistanceThresholdMeters = 2.0f,
        locationHoldMs = 4_000L,
        locationMinUpdateIntervalMs = 1_000L,
        locationMinDistanceMeters = 0.5f
    )

    val guidanceStabilizerConfig = GuidanceStateStabilizerConfig(
        goConfirmDurationMs = 250L,
        stopConfirmDurationMs = 150L,
        waitConfirmDurationMs = 350L,
        cautionConfirmDurationMs = 400L,
        goMinimumHoldMs = 500L
    )

    val occupancyConfig = CrosswalkOccupancyConfig(
        labels = setOf("bicycle", "car", "motorcycle", "bus", "train", "truck"),
        minScore = 0.35f,
        minBottom = 0.45f,
        minArea = 0.015f,
        centerBand = 0.2f..0.8f,
        confirmDurationMs = 400L
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
            append(", go confirm=")
            append(guidanceStabilizerConfig.goConfirmDurationMs)
            append("ms")
            append(", stop confirm=")
            append(guidanceStabilizerConfig.stopConfirmDurationMs)
            append("ms")
            append(", wait confirm=")
            append(guidanceStabilizerConfig.waitConfirmDurationMs)
            append("ms")
            append(", caution confirm=")
            append(guidanceStabilizerConfig.cautionConfirmDurationMs)
            append("ms")
            append(", go hold=")
            append(guidanceStabilizerConfig.goMinimumHoldMs)
            append("ms")
            append(", walk unknown=")
            append(walkSignalConfig.walkAllowedUnknownGraceMs)
            append("ms")
            append(", walk unknown matched=")
            append(walkSignalConfig.walkAllowedUnknownGraceMatchedMs)
            append("ms")
            append(", walk unknown moving=")
            append(walkSignalConfig.walkAllowedUnknownGraceMatchedMovingMs)
            append("ms")
            append(", walk unknown down=")
            append(walkSignalConfig.walkAllowedUnknownGraceMatchedMovingDownMs)
            append("ms")
            append(", ctx motion=")
            append(crossingSupportConfig.motionHoldMs)
            append("ms")
            append(", ctx down=")
            append(crossingSupportConfig.lookingDownRawTiltRangeStartDegrees)
            append("..")
            append(crossingSupportConfig.lookingDownRawTiltRangeEndDegrees)
            append("raw/")
            append(crossingSupportConfig.lookingDownHoldMs)
            append("ms")
            append(", ctx up=")
            append(crossingSupportConfig.lookingUpRawTiltRangeStartDegrees)
            append("..")
            append(crossingSupportConfig.lookingUpRawTiltRangeEndDegrees)
            append("raw/")
            append(crossingSupportConfig.lookingUpHoldMs)
            append("ms")
            append(", ctx gps=")
            append(crossingSupportConfig.locationHoldMs)
            append("ms")
            append(", ctx next=")
            append(walkSignalConfig.sameCrossingMaxDistanceMeters)
            append("m/")
            append(walkSignalConfig.sameCrossingMaxElapsedMs)
            append("ms")
            append(", ")
            append("occupancy score≥")
            append("%.2f".format(occupancyConfig.minScore))
            append(", bottom≥")
            append("%.2f".format(occupancyConfig.minBottom))
            append(", area≥")
            append("%.3f".format(occupancyConfig.minArea))
            append(", band=")
            append("%.2f".format(occupancyConfig.centerBand.start))
            append("..")
            append("%.2f".format(occupancyConfig.centerBand.endInclusive))
            append(", caution=")
            append(occupancyConfig.confirmDurationMs)
            append("ms")
            append(", wait=")
            append(feedbackTimingConfig.waitRepeatIntervalMs)
            append("ms, action=")
            append(feedbackTimingConfig.actionRepeatIntervalMs)
            append("ms")
        }
    }
}
