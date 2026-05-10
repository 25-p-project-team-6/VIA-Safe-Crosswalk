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
                    "${confidenceLabel(confidenceLevel)} · ${clusterSummary(result)}"

                AdvisoryState.GREEN_CONFIRMED ->
                    greenConfirmedDetail(result, confidenceLevel)

                AdvisoryState.GREEN_WITH_CAUTION ->
                    "차량이 횡단보도를 점유하고 있을 수 있습니다"

                AdvisoryState.TRANSITION_WAIT ->
                    when {
                        result.recentMatchedClusterChangeCount > 0 -> "횡단보도 후보가 바뀌어 새 신호 전환을 기다립니다"
                        else -> "현재 초록은 이 횡단보도 신호로 확정되지 않아 다음 신호 전환을 기다립니다"
                    }

                AdvisoryState.UNCERTAIN_VIEW ->
                    when {
                        result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                            "차량용 신호가 보입니다. 보행자 신호등을 화면 중앙에 맞춰주세요"
                        result.multipleSignalDetected -> "신호등이 여러 개 보입니다. 화면 중앙 하나만 비춰주세요"
                        result.needsZoomSuggestion -> "신호가 작게 보입니다. 더 가까이 비추거나 확대해 주세요"
                        result.recentMatchedClusterChangeCount > 0 -> "횡단보도 후보가 ${result.recentMatchedClusterChangeCount}번 바뀌었습니다"
                        result.trafficState == TrafficLightState.UNKNOWN && result.userGuidanceState == UserGuidanceState.GO ->
                            "초록 신호를 다시 확인하는 중입니다"
                        result.targetBox == null -> "신호등을 화면 중앙에 맞춰주세요"
                        else -> "${confidenceLabel(confidenceLevel)} · 추가 확인이 필요합니다"
                    }
            }

        val speechText =
            when (state) {
                AdvisoryState.RED_CONFIRMED -> "빨간불이 확인됩니다"
                AdvisoryState.GREEN_CONFIRMED -> "초록불이 확인됩니다"
                AdvisoryState.GREEN_WITH_CAUTION -> "초록불이 확인되지만 차량 점유 가능성이 있습니다"
                AdvisoryState.TRANSITION_WAIT -> "다음 신호 전환을 기다립니다"
                AdvisoryState.UNCERTAIN_VIEW ->
                    when {
                        result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                            "차량 신호가 보여 보행자 신호 확인이 필요합니다"
                        result.multipleSignalDetected -> "신호등이 여러 개 보여 추가 확인이 필요합니다"
                        result.needsZoomSuggestion -> "신호가 작게 보입니다. 더 가까이 비추거나 확대해 주세요"
                        else -> "신호 확인이 불안정합니다"
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

    private fun confidenceLabel(level: AdvisoryConfidenceLevel): String {
        return when (level) {
            AdvisoryConfidenceLevel.HIGH -> "신뢰 높음"
            AdvisoryConfidenceLevel.MEDIUM -> "신뢰 보통"
            AdvisoryConfidenceLevel.LOW -> "신뢰 낮음"
        }
    }

    private fun greenConfirmedDetail(
        result: SignalAnalysisResult,
        level: AdvisoryConfidenceLevel
    ): String {
        val confidence = confidenceLabel(level)
        return when {
            result.vehicleTrafficLightCount > 0 && result.trafficLightCount == 0 ->
                "$confidence · 보행자 초록 신호가 최근 확인되어 유지됩니다"
            result.vehicleTrafficLightCount > 0 -> "$confidence · 차량 신호도 보이나 보행자 초록 신호를 추적 중입니다"
            result.multipleSignalDetected -> "$confidence · 여러 보행자 신호 중 추적 대상은 초록으로 확인됩니다"
            result.needsZoomSuggestion -> "$confidence · 신호가 작지만 초록으로 확인됩니다"
            result.targetRecentlyReacquired -> "$confidence · 초록 신호를 다시 확인했습니다"
            else -> "$confidence · ${clusterSummary(result)}"
        }
    }

    private fun clusterSummary(result: SignalAnalysisResult): String {
        val map = result.crossingSupportSnapshot.mapProximitySnapshot
        return when {
            map.matchedClusterId == null -> "횡단보도 기준이 아직 약합니다"
            result.recentMatchedClusterChangeCount > 0 -> "횡단보도 후보가 최근 바뀌었습니다"
            else -> "같은 횡단보도 기준이 유지됩니다"
        }
    }
}
