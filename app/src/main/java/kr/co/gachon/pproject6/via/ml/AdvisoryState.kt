package kr.co.gachon.pproject6.via.ml

enum class AdvisoryState {
    RED_CONFIRMED,
    GREEN_CONFIRMED,
    GREEN_WITH_CAUTION,
    TRANSITION_WAIT,
    UNCERTAIN_VIEW
}

enum class AdvisoryConfidenceLevel {
    HIGH,
    MEDIUM,
    LOW
}

enum class AdvisoryConfidenceReason {
    STABLE_SIGNAL,
    NEED_RED_BASELINE,
    MULTIPLE_SIGNALS,
    TARGET_SMALL,
    TARGET_RECENTLY_REACQUIRED,
    MATCHED_CLUSTER_STABLE,
    MATCHED_CLUSTER_CHANGED,
    MATCHED_CLUSTER_MISSING,
    SIGNAL_LOST_GRACE,
    LOOKING_DOWN,
    OCCUPANCY_CAUTION,
    VEHICLE_SIGNAL_VISIBLE
}

data class AdvisoryAssessment(
    val state: AdvisoryState,
    val confidenceLevel: AdvisoryConfidenceLevel,
    val confidenceScore: Int,
    val confidenceReasons: List<AdvisoryConfidenceReason>,
    val titleText: String,
    val detailText: String,
    val speechText: String
)

data class AdvisoryHeuristicsConfig(
    val highConfidenceMinScore: Int = 75,
    val mediumConfidenceMinScore: Int = 55,
    val smallTargetAreaThreshold: Float = 0.015f,
    val recentClusterChangeAlertThreshold: Int = 1,
    val recentReacquireAlertThreshold: Int = 1,
    val multipleSignalPenalty: Int = 22,
    val targetSmallPenalty: Int = 12,
    val recentReacquirePenalty: Int = 12,
    val clusterChangedPenalty: Int = 18,
    val noMatchPenalty: Int = 12,
    val lostSignalPenalty: Int = 14,
    val cautionPenalty: Int = 8,
    val vehicleSignalPenalty: Int = 10,
    val matchedStableBonus: Int = 10,
    val stableRedBonus: Int = 18,
    val stableGreenBonus: Int = 14,
    val targetScoreHighBonus: Int = 10,
    val targetScoreMediumBonus: Int = 5
)
