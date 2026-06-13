package kr.co.gachon.pproject6.via.ml

class SignalAdvisoryEvaluator(
    private val config: AdvisoryHeuristicsConfig = AdvisoryHeuristicsConfig()
) {
    fun evaluate(result: SignalAnalysisResult): AdvisoryAssessment {
        val reasons = linkedSetOf<AdvisoryConfidenceReason>()
        var score = 50

        when (result.trafficState) {
            TrafficLightState.RED -> {
                score += config.stableRedBonus
                reasons += AdvisoryConfidenceReason.STABLE_SIGNAL
            }

            TrafficLightState.GREEN -> {
                score += config.stableGreenBonus
                reasons += AdvisoryConfidenceReason.STABLE_SIGNAL
            }

            TrafficLightState.UNKNOWN -> Unit
        }

        score += when {
            result.targetScore >= 0.8f -> config.targetScoreHighBonus
            result.targetScore >= 0.6f -> config.targetScoreMediumBonus
            else -> 0
        }

        if (result.guidanceBlockReason == GuidanceBlockReason.NEED_RED_BASELINE) {
            reasons += AdvisoryConfidenceReason.NEED_RED_BASELINE
            score -= 12
        }
        if (result.multipleSignalDetected) {
            reasons += AdvisoryConfidenceReason.MULTIPLE_SIGNALS
            score -= config.multipleSignalPenalty
        }
        if (result.vehicleTrafficLightCount > 0) {
            reasons += AdvisoryConfidenceReason.VEHICLE_SIGNAL_VISIBLE
            score -= config.vehicleSignalPenalty
        }
        if (result.needsZoomSuggestion) {
            reasons += AdvisoryConfidenceReason.TARGET_SMALL
            score -= config.targetSmallPenalty
        }
        if (result.targetRecentlyReacquired) {
            reasons += AdvisoryConfidenceReason.TARGET_RECENTLY_REACQUIRED
            score -= config.recentReacquirePenalty
        }
        if (result.recentMatchedClusterChangeCount >= config.recentClusterChangeAlertThreshold) {
            reasons += AdvisoryConfidenceReason.MATCHED_CLUSTER_CHANGED
            score -= config.clusterChangedPenalty
        } else if (result.crossingSupportSnapshot.mapProximitySnapshot.matchedClusterId != null) {
            reasons += AdvisoryConfidenceReason.MATCHED_CLUSTER_STABLE
            score += config.matchedStableBonus
        } else {
            reasons += AdvisoryConfidenceReason.MATCHED_CLUSTER_MISSING
            score -= config.noMatchPenalty
        }
        if (result.trafficState == TrafficLightState.UNKNOWN && result.userGuidanceState == UserGuidanceState.GO) {
            reasons += AdvisoryConfidenceReason.SIGNAL_LOST_GRACE
            score -= config.lostSignalPenalty
        }
        if (result.crossingSupportSnapshot.isLookingDown && result.trafficState == TrafficLightState.UNKNOWN) {
            reasons += AdvisoryConfidenceReason.LOOKING_DOWN
        }
        if (result.occupancyCaution) {
            reasons += AdvisoryConfidenceReason.OCCUPANCY_CAUTION
            score -= config.cautionPenalty
        }

        val confidenceScore = score.coerceIn(0, 100)
        val confidenceLevel =
            when {
                confidenceScore >= config.highConfidenceMinScore -> AdvisoryConfidenceLevel.HIGH
                confidenceScore >= config.mediumConfidenceMinScore -> AdvisoryConfidenceLevel.MEDIUM
                else -> AdvisoryConfidenceLevel.LOW
            }

        val isWalkAllowedGreen =
            result.userGuidanceState == UserGuidanceState.GO &&
                result.trafficState == TrafficLightState.GREEN
        val hasVehicleOnlyConflict =
            result.vehicleTrafficLightCount > 0 &&
                result.trafficLightCount == 0
        val hasHardVisualConflict =
            (hasVehicleOnlyConflict && !isWalkAllowedGreen) ||
                result.recentMatchedClusterChangeCount >= config.recentClusterChangeAlertThreshold

        val state =
            when {
                result.guidanceBlockReason == GuidanceBlockReason.NEED_RED_BASELINE -> AdvisoryState.TRANSITION_WAIT
                result.trafficState == TrafficLightState.RED -> AdvisoryState.RED_CONFIRMED
                isWalkAllowedGreen && result.occupancyCaution -> AdvisoryState.GREEN_WITH_CAUTION
                isWalkAllowedGreen && !hasHardVisualConflict -> AdvisoryState.GREEN_CONFIRMED
                else -> AdvisoryState.UNCERTAIN_VIEW
            }

        val titleText =
            when (state) {
                AdvisoryState.RED_CONFIRMED -> "빨간불"
                AdvisoryState.GREEN_CONFIRMED -> "초록불"
                AdvisoryState.GREEN_WITH_CAUTION -> "초록불 주의"
                AdvisoryState.TRANSITION_WAIT -> "대기 중"
                AdvisoryState.UNCERTAIN_VIEW ->
                    when {
                        result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                            "차량 신호"
                        result.multipleSignalDetected -> "여러 신호"
                        result.needsZoomSuggestion -> "작게 보임"
                        else -> "확인 중"
                    }
            }

        val detailText =
            when (state) {
                AdvisoryState.RED_CONFIRMED ->
                    "빨간불 확인."

                AdvisoryState.GREEN_CONFIRMED ->
                    "초록불 확인."

                AdvisoryState.GREEN_WITH_CAUTION ->
                    "차량 주의."

                AdvisoryState.TRANSITION_WAIT ->
                    when {
                        result.recentMatchedClusterChangeCount > 0 -> "대상 변경."
                        else -> "전환 대기."
                    }

                AdvisoryState.UNCERTAIN_VIEW ->
                    when {
                        result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                            "차량 신호."
                        result.multipleSignalDetected -> "여러 신호."
                        result.needsZoomSuggestion -> "신호 작음."
                        result.recentMatchedClusterChangeCount > 0 -> "대상 변경."
                        result.trafficState == TrafficLightState.UNKNOWN && result.userGuidanceState == UserGuidanceState.GO ->
                            "초록 재확인."
                        result.targetBox == null -> "신호 미탐지."
                        else -> clusterSummary(result)
                    }
            }

        val speechText =
            when (state) {
                AdvisoryState.RED_CONFIRMED -> "빨간불."
                AdvisoryState.GREEN_CONFIRMED -> "초록불."
                AdvisoryState.GREEN_WITH_CAUTION -> "차량 주의."
                AdvisoryState.TRANSITION_WAIT -> "대기."
                AdvisoryState.UNCERTAIN_VIEW ->
                    when {
                        result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                            "차량 신호."
                        result.multipleSignalDetected -> "여러 신호."
                        result.needsZoomSuggestion -> "신호 작음."
                        else -> "확인 필요."
                    }
            }

        return AdvisoryAssessment(
            state = state,
            confidenceLevel = confidenceLevel,
            confidenceScore = confidenceScore,
            confidenceReasons = reasons.toList(),
            titleText = titleText,
            detailText = detailText,
            speechText = speechText
        )
    }

    private fun clusterSummary(result: SignalAnalysisResult): String {
        val map = result.crossingSupportSnapshot.mapProximitySnapshot
        return when {
            map.matchedClusterId == null -> "횡단보도 미탐지."
            result.recentMatchedClusterChangeCount > 0 -> "대상 변경."
            else -> "기준 유지."
        }
    }
}
